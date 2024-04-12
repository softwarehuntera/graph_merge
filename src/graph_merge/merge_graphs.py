"""Tools for merging multiple graphs into a single graph."""

import copy
import hashlib
import multiprocessing
from collections import Counter
from itertools import chain
from pathlib import Path
from typing import Any, Iterable, Literal

import networkx as nx
import pandas as pd

from graph_merge.dfs import calc_depth_first_search_mapping
from graph_merge.similarity import compare_multiple

MERGE_GRAPH_COLOR_PALETTE = {
    "G1": "#FFA600",
    "G2": "#BC5090",
    "Both": "#003F5C",
    "Possible Match": "#69CC3E",
}


SimilarityAttributes = list[str] | list["SimilarityAttributes"]


def compile_set_of_attributes(attributes: SimilarityAttributes) -> set[str]:
    """Return the set of all unique attributes that are included within a SimilarityAttributes structure.

    Args:
        attributes (SimilarityAttributes): SimilarityAttribute structure to compile.

    Returns:
        set[str]: Set of all unique attributes discovered.
    """
    if attributes:
        if isinstance(attributes[0], list):
            return set().union(*[compile_set_of_attributes(attribute) for attribute in attributes])
        return set(attributes)
    return set()


def _check_similarity_of_values(
    left_values: dict[str, Any],
    right_values: dict[str, Any],
    attributes: SimilarityAttributes,
    match_function_args: dict[str, dict[str, Any]],
) -> bool:
    if attributes:
        if isinstance(attributes[0], list):
            # assumed to be list of lists
            # `any` is the equivalent of logical ORs
            return any(
                _check_similarity_of_values(left_values, right_values, x, match_function_args) for x in attributes
            )
        # assumed to be list of keys
        # `all` is the equivalent of logical ANDs
        return all(
            compare_multiple(
                str1=left_values[attribute], str2=right_values[attribute], match_function_args=match_function_args
            )
            for attribute in attributes
        )
    raise ValueError(
        "attributes is None, but must be a nonempty list."
        if attributes is None
        else "attributes is empty, but must contain atleast one element"
    )


def _find_matches_for_k(
    k: Any,  # noqa: ANN401
    v: dict,
    attributes: SimilarityAttributes,
    match_function_args: dict[str, dict[str, Any]],
    search_items: Iterable[tuple[Any, dict[str, str]]],
) -> tuple[Any, set]:
    """Find all possible matches for k within search_items.

    This function was designed to be used exclusively within the GraphMerger._node_attribute_mapping method.
    It has to be defined at the global level or else multiprocessing is unable to pickle it.

    Args:
        k (Any): Key to be finding matches for.
        v (dict): Values for k to use in comparison.
        attributes (list[str]): Names of values in v and search_items to actually use for comparisons.
        match_function_args (dict): Matching function names and arguments to use for comparisons.
        search_items (iterable[tuple[Any, dict]]): Iterable of key-value pairs to use for comparison
            with the values for k.

    Returns:
        tuple[Any, set]: Returns k and the set of keys in search_items that were found to match, respectively.
    """
    result = set()
    for search_key, search_values in search_items:
        if _check_similarity_of_values(v, search_values, attributes, match_function_args):
            result.add(search_key)
    return k, result


class NodeMap:
    """Object for managing mappings of nodes in one graph to nodes in another."""

    def __init__(
        self,
        map_type: Literal["similarity", "depth_first_search"],
        node_map: dict,
        attributes: SimilarityAttributes,
        match_function_args: dict,
        **kwargs: dict,
    ) -> None:
        """Initialize NodeMap object.

        Args:
            map_type (Literal["similarity", "depth_first_search"]): Type of node mapping.
            node_map (dict): Mapping of nodes in G1 to nodes in G2
            attributes (SimilarityAttributes): Name(s) of attribute(s) to use.
            match_function_args (dict): Dictionary of similarity function(s) with respective arguments.
            kwargs (dict): Additional keyword arguments to keep.
        """
        self.map_type = map_type
        self.attributes = attributes
        self.match_function_args = match_function_args
        self.node_map = node_map
        self.kwargs = kwargs

    @staticmethod
    def order_dict(dictionary: dict) -> dict:
        """Order dictionary keys and values.

        https://stackoverflow.com/questions/22721579/sorting-a-nested-ordereddict-by-key-recursively

        Args:
            dictionary (dict): Input dictionary.

        Returns:
            dict: Ordered dictionary.
        """
        result = {}
        for k, v in sorted(dictionary.items()):
            if isinstance(v, dict):
                result[k] = NodeMap.order_dict(v)
            else:
                result[k] = v
        return result

    @staticmethod
    def _string_sort_attributes(attributes: SimilarityAttributes) -> str:
        """Convert attribute list to sorted string..

        Args:
            attributes (list[str]): Name(s) of attribute(s) to use.

        Returns:
            str: hashable string.
        """
        if attributes and isinstance(attributes[0], list):
            return str("".join(sorted(NodeMap._string_sort_attributes(attribute) for attribute in attributes)))
        return str("".join(sorted(attributes)))

    @staticmethod
    def _string_sort_kwargs(o: set | tuple | list | dict | str | float) -> str:
        """Transform kwargs into hashable string.

        Args:
            o (set | tuple | list | dict | str | float): Object to convert.

        Returns:
            str: hashable string.
        """
        if isinstance(o, (set, tuple, list)):
            return tuple([NodeMap._string_sort_kwargs(e) for e in o])

        if not isinstance(o, dict):
            return o

        new_o = copy.deepcopy(o)
        for k, v in new_o.items():
            new_o[k] = NodeMap._string_sort_kwargs(v)

        return str(tuple(new_o.items()))

    @staticmethod
    def map_hash(
        map_type: Literal["similarity", "depth_first_search"],
        attributes: SimilarityAttributes,
        match_function_args: dict,
        **kwargs: dict,
    ) -> str:
        """Create a sha256 string of attributes and match_function_args.

        Args:
            map_type (Literal["similarity", "depth_first_search"]): Type of node mapping.
            attributes (SimilarityAttributes): Name(s) of attribute(s) to use.
            match_function_args (dict): Dictionary of similarity function(s) with respective arguments.
            kwargs (dict): Additional keyword arguments to use in hash generation.

        Returns:
            str: Hash of mapping parameters.
        """
        sorted_attributes = str(sorted("".join(attributes)))

        sorted_match_function_args = NodeMap.order_dict(dictionary=match_function_args)
        sorted_match_function_args = NodeMap._string_sort_kwargs(sorted_match_function_args)

        sorted_kwargs = NodeMap.order_dict(dictionary=kwargs)
        sorted_kwargs = NodeMap._string_sort_kwargs(sorted_kwargs)

        string_to_hash = map_type + sorted_attributes + sorted_match_function_args + sorted_kwargs
        return hashlib.sha256(string_to_hash.encode("utf-8")).hexdigest()


class GraphMerger:
    """Merging graphs."""

    def __init__(self, G1: nx.Graph, G2: nx.Graph, node_deconfliction_string: str = "_2") -> None:
        """Merge two graphs together for comparison.

        Args:
            G1 (nx.Graph): Graph 1 (baseline).
            G2 (nx.Graph): Graph 2 (comparison).
            node_deconfliction_string (str, optional): Suffix to use to deconflict the ID's of nodes in G1 and G2.
        """
        self.G1 = copy.deepcopy(G1)
        self.G2 = copy.deepcopy(G2)
        self.node_deconfliction_string = node_deconfliction_string
        self._deconflict_nodes()
        self._node_uuid_as_attribute()

        self.g1_nodes = dict(self.G1.nodes(data=True))
        self.g2_nodes = dict(self.G2.nodes(data=True))

        self.similarity_maps = {}

        self.current_graph = None

    @staticmethod
    def _check_node_attributes(G: nx.Graph, attributes: SimilarityAttributes) -> None:
        """Check the node attributes to make sure they all exist on the nodes of the graph.

        Args:
            G (nx.Graph): Networkx graph
            attributes (SimilarityAttributes): list of node attributes

        Raises:
            ValueError: The node attribute list was empty.
            ValueError: A given node attribute was not an attribute for a node
        """
        if not attributes:
            raise ValueError("The node attribute list was empty.")

        for node in G.nodes(data=True):
            for attr in compile_set_of_attributes(attributes):
                if attr not in node[1]:
                    raise ValueError(f"({attr}) was not found as a node attribute for node {node[0]}.")

    @staticmethod
    def _inverse_node_map(node_map: dict) -> dict:
        """Inverts the G1 nodes and G2 nodes in a node map.

        Example:
            _inverse_node_map(node_map={"key1": {"val1", "val2"}})
            >>>  {"val1": {"key1"}, "val2": {"key2"}}

        Args:
            node_map (dict): Mapping of nodes in G1 to nodes in G2.

        Returns:
            dict: Mapping of nodes in G2 nodes to nodes in G1.
        """
        inverse_mapping = {}
        for g1_node, g2_nodes in node_map.items():
            for g2_node in g2_nodes:
                if g2_node not in inverse_mapping:
                    inverse_mapping[g2_node] = {g1_node}
                else:
                    inverse_mapping[g2_node].update([g1_node])
        return inverse_mapping

    def _deconflict_nodes(self) -> None:
        """Ensure that the graph node names don't conflict.

        Nodes in G2 will have self.node_deconfliction_string appended if there are node collisions.
        """
        if set(self.G1.nodes()).intersection(set(self.G2.nodes())):
            self.G2 = nx.relabel_nodes(
                self.G2,
                {node: str(node) + self.node_deconfliction_string for node in self.G2.nodes()},
            )

    def _node_uuid_as_attribute(self) -> None:
        """Assign node uuids as metadata."""
        nx.set_node_attributes(self.G1, {node: node for node in self.G1.nodes()}, "uuid")
        nx.set_node_attributes(self.G2, {node: node for node in self.G2.nodes()}, "uuid")

    @staticmethod
    def get_starting_node(node_map: dict) -> tuple[str, str]:
        """Find a viable starting node from a node map defined as having exactly 1 mapping.

        Args:
            node_map (dict): Mapping of nodes in G1 to nodes in G2.

        Raises:
            ValueError: No potential g1_starting_nodes found.

        Returns:
            tuple[str, str]: Node ID for a viable starting node in G1 and G2, respectively.
        """
        g1_starting_node = None
        for k, v in node_map.items():
            if len(v) == 1:
                g1_starting_node = k
                g2_starting_node = next(iter(v))
                break
        if not g1_starting_node:
            raise ValueError("No potential g1_starting_nodes found.")
        return g1_starting_node, g2_starting_node

    def get_similarity_map(self, attributes: SimilarityAttributes, match_function_args: dict) -> NodeMap:
        """Get node mapping where nodes match on given attributes using provided similarity functions.

        The `attributes` parameter is a recursive data structure that allows for interesting similarity definitions.
        There are two governing rules for `attributes`:
          1. A list of strings represent a list of attributes that must all be similar (logically ANDed together)
          2. A list of lists represents a list of similarity checks where atleast one of them
             must be true (logically ORed together)

        Examples:
            - Check if the `name` attribute is similar: `attributes = ['name']`
            - Check if the `name` and `color` attributes are similar: `attributes = ['name', 'color']`
            - Check if the `name` or `color` attributes are similar: `attributes = [['name'], ['color']]`
            - Check if the `name` attribute is similar or if the `color` and `shape`
              attributes are: `attributes = [['name'], ['color', 'shape']]`

        Note that you cannot mix types of lists.
        For example, this is not allowed: `attributes = ['name', ['color', 'shape']]`

        Additionally, `match_function_args` is applied to every attribute the same.

        Args:
            attributes (SimilarityAttributes): Name(s) of attribute(s) to use.
            match_function_args (dict): Dictionary of similarity function(s) with respective arguments.

        Returns:
            NodeMap: Mapping of nodes in G1 to nodes in G2.
        """
        self._check_node_attributes(self.G1, attributes)
        self._check_node_attributes(self.G2, attributes)

        map_hash = NodeMap.map_hash(
            map_type="similarity",
            attributes=attributes,
            match_function_args=match_function_args,
        )

        if map_hash not in self.similarity_maps:
            self.similarity_maps[map_hash] = NodeMap(
                map_type="similarity",
                node_map=GraphMerger._node_attribute_mapping(
                    g1_node_attributes=self.g1_nodes,
                    g2_node_attributes=self.g2_nodes,
                    attributes=attributes,
                    match_function_args=match_function_args,
                ),
                attributes=attributes,
                match_function_args=match_function_args,
            )

        return self.similarity_maps[map_hash]

    def get_depth_first_search_map(
        self,
        attributes: SimilarityAttributes,
        match_function_args: dict,
        g1_starting_node: str,
        g2_starting_node: str | None = None,
    ) -> NodeMap:
        """Get the node mapping using the depth first search algorithm.

        The `attributes` parameter is a recursive data structure that allows for interesting similarity definitions.
        There are two governing rules for `attributes`:
          1. A list of strings represent a list of attributes that must all be similar (logically ANDed together)
          2. A list of lists represents a list of similarity checks where atleast one of them
             must be true (logically ORed together)

        Examples:
            - Check if the `name` attribute is similar: `attributes = ['name']`
            - Check if the `name` and `color` attributes are similar: `attributes = ['name', 'color']`
            - Check if the `name` or `color` attributes are similar: `attributes = [['name'], ['color']]`
            - Check if the `name` attribute is similar or if the `color` and `shape`
              attributes are: `attributes = [['name'], ['color', 'shape']]`

        Note that you cannot mix types of lists.
        For example, this is not allowed: `attributes = ['name', ['color', 'shape']]`

        Additionally, `match_function_args` is applied to every attribute the same.

        Args:
            attributes (SimilarityAttributes): Name(s) of attribute(s) to use.
            match_function_args (dict): Dictionary of similarity function(s) with respective arguments.
            g1_starting_node (str): Node in G1 to match in G2 to start the
                DFS (must be a node with only 1 possible mapping).
            g2_starting_node (str|None): Node in G2 that is the unique possible mapping from the g1_starting_node.
                If None then G2 will be searched for a candidate using the supplied similarity function(s) and
                argument(s). If only one is found then it will be used. Otherwise a ValueError will be raised.
                Defaults to None.

        Returns:
            NodeMap: Mapping of nodes in G1 to nodes in G2.

        Raises:
            ValueError: If g2_starting_node is not supplied and an unambiguous candidate can't be located.
            ValueError: Mapping attribute(s) do not exist in G1 or G2.
        """
        self._check_node_attributes(self.G1, attributes)
        self._check_node_attributes(self.G2, attributes)

        if g2_starting_node is None:
            possible_map = self.get_similarity_map(
                attributes=attributes,
                match_function_args=match_function_args,
            )
            candidates = possible_map.node_map.get(g1_starting_node, [])
            if len(candidates) != 1:
                raise ValueError(
                    "G1 starting node does not have a unique similarity mapping. "
                    "Choose a new starting node or specify the corresponding G2 node. "
                    f"Number of unique mappings: {len(candidates)}"
                )
            g2_starting_node = next(iter(candidates))
        elif g2_starting_node not in self.G2:
            deconflicted_form = g2_starting_node + self.node_deconfliction_string
            if deconflicted_form in self.G2:
                g2_starting_node = deconflicted_form
            else:
                raise ValueError(
                    f"[{g2_starting_node}] and [{deconflicted_form}] do not exist in G2. "
                    "Either select an existing node in G2 to start with or attempt automatic selection of a G2 node."
                )

        map_hash = NodeMap.map_hash(
            map_type="depth_first_search",
            attributes=attributes,
            match_function_args=match_function_args,
            g1_starting_node=g1_starting_node,
            g2_starting_node=g2_starting_node,
        )

        if map_hash not in self.similarity_maps:
            similarity_map = self.get_similarity_map(
                attributes=attributes,
                match_function_args=match_function_args,
            ).node_map

            node_map = calc_depth_first_search_mapping(
                G1=self.G1,
                G2=self.G2,
                similarity_map=similarity_map,
                g1_starting_node=g1_starting_node,
                g2_starting_node=g2_starting_node,
            )

            self.similarity_maps[map_hash] = NodeMap(
                map_type="depth_first_search",
                node_map=node_map,
                attributes=attributes,
                match_function_args=match_function_args,
                g1_starting_node=g1_starting_node,
                g2_starting_node=g2_starting_node,
            )

        return self.similarity_maps[map_hash]

    @staticmethod
    def _node_attribute_mapping(
        g1_node_attributes: dict,
        g2_node_attributes: dict,
        attributes: SimilarityAttributes,
        match_function_args: dict,
    ) -> dict:
        """Create mapping of nodes in G1 to nodes in G2 where attribute(s) match according to similarity function(s).

        Currently maps if ANY of the similarity functions are satisfied.

        Args:
            g1_node_attributes (dict): Node attribute dictionary of G1.
            g2_node_attributes (dict): Node attribute dictionary of G2.
            attributes (list[str]): Name(s) of attribute(s) to use.
            match_function_args (dict): Dictionary of similarity function(s) with respective arguments.

        Returns:
            dict: Node mapping dictionary.
        """
        with multiprocessing.Pool() as pool:
            g2_items = list(g2_node_attributes.items())
            maps = [(k1, v1, attributes, match_function_args, g2_items) for k1, v1 in g1_node_attributes.items()]
            results = pool.starmap(_find_matches_for_k, maps)
        node_mapping = {k1: result for k1, result in results if result}
        return node_mapping

    def create_merge_graph_from_mapping(
        self,
        node_map: NodeMap,
        g1_name: str = "G1",
        g2_name: str = "G2",
        collapse: bool = True,
    ) -> nx.Graph:
        """Combine the two graphs using mapping to determine which nodes are identical.

        Args:
            node_map (NodeMap): {node in G1: node in G2}
            g1_name (str): name of G1 to set as "graph" node attribute.
                        Defaults to "G1".
            g2_name (str): name of G2 to set as "graph" node attribute.
                        Defaults to "G2".
            collapse (bool): Whether to collapse the leaf nodes. Defaults to True.

        Raises:
            ValueError: Unexpected node found.
            ValueError: Unexpected edge found.

        Returns:
            nx.Graph: combined G1/G2 graph.
        """
        if g1_name is None:
            g1_name = "G1"

        if g2_name is None:
            g2_name = "G2"

        node_mapping = node_map.node_map
        g1_matched_nodes = set(node_mapping.keys())
        g2_matched_nodes = set(chain.from_iterable(node_mapping.values()))
        inverse_node_mapping = GraphMerger._inverse_node_map(node_map=node_mapping)

        g1_only_nodes = set(self.g1_nodes.keys()).difference(g1_matched_nodes)
        g2_only_nodes = set(self.g2_nodes.keys()).difference(g2_matched_nodes)
        g2_added = set()
        G = copy.deepcopy(self.G1)

        # Add G2 only nodes and edges
        for node in g2_only_nodes:
            # Add node
            G.add_nodes_from([(node, self.g2_nodes[node])])
            g2_added.add(node)

            # Find neighbors
            node2_neigh = list(self.G2.neighbors(node))
            for neigh2 in node2_neigh:
                if neigh2 in inverse_node_mapping:
                    # Add edge
                    neighbor = list(inverse_node_mapping[neigh2])
                    if len(neighbor) == 1:
                        G.add_edge(node, neighbor[0], graph=g2_name)
                elif neigh2 in g2_added:
                    G.add_edge(node, neigh2, graph=g2_name)

        self._set_merge_graph_node_attributes(
            G=G,
            g1_only_nodes=g1_only_nodes,
            g2_only_nodes=g2_only_nodes,
            g1_matched_nodes=g1_matched_nodes,
            g1_name=g1_name,
            g2_name=g2_name,
        )

        self._set_merge_graph_edge_attributes(
            G=G,
            g1_matched_nodes=g1_matched_nodes,
            g1_only_nodes=g1_only_nodes,
            g2_only_nodes=g2_only_nodes,
            g1_name=g1_name,
            g2_name=g2_name,
        )

        if collapse:
            G = GraphMerger.collapse_nodes(G)
        self.current_graph = G

    def overlay_node_map_on_graph(self, attributes: SimilarityAttributes, match_function_args: dict) -> dict:
        """Overlay node mapping and add new edges.

        Args:
            attributes (SimilarityAttributes): Name(s) of attribute(s) to use.
            match_function_args (dict): Dictionary of similarity function(s) with respective arguments.

        Raises:
            ValueError: Must create initial graph using create_merge_graph_from_mapping().

        Returns:
            dict: Suggested mappings of G1 only nodes to G2 only nodes.
        """
        if self.current_graph is None:
            raise ValueError("Must create initial graph using create_merge_graph_from_mapping().")
        g1_only_nodes = {node[0]: node[1] for node in self.current_graph.nodes(data=True) if node[1]["graph"] == "G1"}
        g2_only_nodes = {node[0]: node[1] for node in self.current_graph.nodes(data=True) if node[1]["graph"] == "G2"}

        suggested_node_mapping = GraphMerger._node_attribute_mapping(
            g1_node_attributes=g1_only_nodes,
            g2_node_attributes=g2_only_nodes,
            attributes=attributes,
            match_function_args=match_function_args,
        )
        # Drop mappings if they don't share a one-hop neighbor.
        for k, v in suggested_node_mapping.items():
            updated_set = set()
            for i in v:
                if self.shared_neighbor(node1_name=k, node2_name=i):
                    updated_set.add(i)
            suggested_node_mapping.update({k: updated_set})

        for node_a in suggested_node_mapping:
            for node_b in suggested_node_mapping[node_a]:
                self.current_graph.add_edge(
                    u_of_edge=node_a,
                    v_of_edge=node_b,
                    graph="Possible Match",
                    color=MERGE_GRAPH_COLOR_PALETTE["Possible Match"],
                    alpha=0.4,
                )
        return suggested_node_mapping

    @staticmethod
    def collapse_nodes(G: nx.Graph) -> nx.Graph:
        """Collapse leaf nodes in 'Both' graphs to supernode.

        Create a new graph from a combined graph in which leaf nodes that are
        in both graphs are reduced down to one node. Add a node attribute for
        that shows number of original nodes in each node of new graph.

        Args:
            G (nx.Graph): networkx graph with node attribute "graph".

        Returns:
            nx.Graph: networkx graph with node attribute "num_nodes".
        """
        degrees = G.degree()
        nodes = G.nodes(data=True)
        subgraph_nodes = []
        supernodes = {}
        node_attrib_names = list(next(iter(G.nodes(data=True)))[1].keys()).copy()

        # Get nodes to be collapsed and nodes not collapsed
        for node in nodes:
            # Collapsed nodes contain leaves in both graphs with same neighbor
            if node[1]["graph"] == "Both" and degrees[node[0]] == 1:
                neighbor = next(G.neighbors(node[0]))
                if neighbor in supernodes:
                    supernodes[neighbor] += 1
                else:
                    supernodes[neighbor] = 1
            # Subgraph nodes are nodes that are not collapsed
            else:
                subgraph_nodes.append(node[0])

        # Make subgraph of nodes that are in both graphs
        collapsed_G = G.subgraph(subgraph_nodes).copy()
        nx.set_node_attributes(collapsed_G, 1, "num_nodes")

        # Add collapsed nodes
        for node, value in supernodes.items():
            new_node = str(node) + "_supernode"
            collapsed_G.add_edge(
                u_of_edge=new_node,
                v_of_edge=node,
                graph="Both",
                color=MERGE_GRAPH_COLOR_PALETTE["Both"],
                alpha=1.0,
            )
            node_attrib = {key: "None" for key in node_attrib_names}
            node_attrib["graph"] = "Both"
            node_attrib["num_nodes"] = value
            node_attrib["color"] = MERGE_GRAPH_COLOR_PALETTE["Both"]
            node_attrib["alpha"] = 1.0
            node_attrib["size"] = 100.0
            nx.set_node_attributes(collapsed_G, {new_node: node_attrib})

        return collapsed_G

    def write_merge_graph_nodes(self, save_path: Path) -> None:
        """Save a dataframe of the current graph nodes to a csv.

        Args:
            save_path (Path): path to save csv
        """
        df = pd.DataFrame.from_dict(dict(self.current_graph.nodes(data=True)), orient="index")
        df.reset_index(inplace=True)
        df.to_csv(save_path / "merge_graph_nodes.csv", index=False)

    def _set_merge_graph_node_attributes(
        self,
        G: nx.Graph,
        g1_only_nodes: set[str],
        g2_only_nodes: set[str],
        g1_matched_nodes: set[str],
        g1_name: str,
        g2_name: str,
    ) -> None:
        for node in G.nodes():
            if node in g1_only_nodes:
                nx.set_node_attributes(G, {node: g1_name}, "graph")
                nx.set_node_attributes(G, {node: MERGE_GRAPH_COLOR_PALETTE["G1"]}, "color")
            elif node in g2_only_nodes:
                nx.set_node_attributes(G, {node: g2_name}, "graph")
                nx.set_node_attributes(G, {node: MERGE_GRAPH_COLOR_PALETTE["G2"]}, "color")
            elif node in g1_matched_nodes:
                nx.set_node_attributes(G, {node: "Both"}, "graph")
                nx.set_node_attributes(G, {node: MERGE_GRAPH_COLOR_PALETTE["Both"]}, "color")
            else:
                raise ValueError(f"Unexpected node found. {node}")
            nx.set_node_attributes(G, {node: 1.0}, "alpha")
            nx.set_node_attributes(G, {node: 75.0}, "size")

    def _set_merge_graph_edge_attributes(
        self,
        G: nx.Graph,
        g1_matched_nodes: set[str],
        g1_only_nodes: set[str],
        g2_only_nodes: set[str],
        g1_name: str,
        g2_name: str,
    ) -> None:
        for edge in G.edges():
            if (edge[0] in g1_matched_nodes) and (edge[1] in g1_matched_nodes):
                nx.set_edge_attributes(G, {edge: "Both"}, "graph")
                nx.set_edge_attributes(G, {edge: MERGE_GRAPH_COLOR_PALETTE["Both"]}, "color")
            elif (edge[0] in g1_only_nodes) or (edge[1] in g1_only_nodes):
                nx.set_edge_attributes(G, {edge: g1_name}, "graph")
                nx.set_edge_attributes(G, {edge: MERGE_GRAPH_COLOR_PALETTE["G1"]}, "color")
            elif (edge[0] in g2_only_nodes) or (edge[1] in g2_only_nodes):
                nx.set_edge_attributes(G, {edge: g2_name}, "graph")
                nx.set_edge_attributes(G, {edge: MERGE_GRAPH_COLOR_PALETTE["G2"]}, "color")
            else:
                raise ValueError("Unexpected edge found.")
            nx.set_edge_attributes(G, {edge: 1.0}, "alpha")

    def shared_neighbor(self, node1_name: str, node2_name: str) -> bool:
        """Checks if two nodes share a common neighbor.

        Args:
            node1_name (str): Name of node 1.
            node2_name (str): Name of node 2.

        Returns:
            bool: True if nodes share a neighbor.
        """
        node1_neighbors = self.current_graph.neighbors(node1_name)
        node2_neighbors = [
            self._strip_deduplication_formatting(node_name=n) for n in self.current_graph.neighbors(node2_name)
        ]
        shared_neighbors = list(set(node1_neighbors).intersection(set(node2_neighbors)))
        return len(shared_neighbors) > 0

    def _strip_deduplication_formatting(self, node_name: str) -> str:
        """Strip the deduplication that was added to deconflict graphs if needed.

        Args:
            node_name (str): Name of node.

        Returns:
            str: Node name without deduplication formatting.
        """
        string = node_name
        if node_name.endswith(self.node_deconfliction_string):
            node_deduplication_string_length = len(self.node_deconfliction_string)
            string = string[:-node_deduplication_string_length]
        return string

    def save_gml(self, save_path: Path) -> None:
        """Save graph as .gml.

        Path should include the file name ending in .gml.

        Args:
            save_path (Path): Path to save the graph data file.

        Raises:
            ValueError: Invalid node attribute value of None.

        """
        # Check for None in node attributes:
        for n in self.current_graph.nodes(data=True):
            if None in n[1].values():
                raise ValueError("Invalid node attribute value of None. Run fix_node_attributes to fix.")

        nx.write_gml(self.current_graph, save_path)

    def fix_node_attributes(self) -> None:
        """Change None node attributes to "None" for saving to gml."""
        node_attributes = {}
        # Check for None in node attributes:
        for n in self.current_graph.nodes(data=True):
            for k, v in n[1].items():
                if v is None:
                    node_attributes[n[0]] = {k: "None"}

        nx.set_node_attributes(self.current_graph, node_attributes)

    def count_nodes(self) -> Counter:
        """Count the number of nodes by 'graph' attribute.

        Returns:
            (dict): 'graph' attribute value: count of number of nodes with that value

        """
        return Counter(nx.get_node_attributes(self.current_graph, "graph").values())
