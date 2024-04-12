"""Depth first search mapping logic."""

import itertools
from typing import Iterator

import networkx as nx


class DepthFirstSearchGraph:
    """Wrapper around a graph for a DepthFirstSearch to provide utility attributes."""

    def __init__(self, G: nx.Graph) -> None:
        """Initialize DepthFirstSearchGraph object.

        Args:
            G: Graph to wrap for a depth first search.
        """
        self.visited = set()
        self.not_mapped = set()
        self.added = set()

        # Initialize graph stats for quick lookup
        self.G = G
        self.neighbors = {n: set(G.neighbors(n)) for n in G.nodes()}
        self.degree = nx.degree(G)

    def __contains__(self, node: str) -> bool:
        return node in self.G


class DepthFirstSearchState:
    """State of a depth first search."""

    def __init__(self, G1: nx.Graph, G2: nx.Graph, similarity_map: dict[str, set[str]]) -> None:
        """Initialize the Depth First Search State.

        Args:
            G1: Graph to attempt to map onto G2.
            G2: Graph to be mapped on to by G1.
            similarity_map: Precomputed similarity mappings of nodes.
        """
        self.stack = []
        self.similarity_map = similarity_map
        self.mapping = {}
        self.path = []
        self.stack = []
        self.duplicates = {}  # for pompoms that are very similar

        self.G1 = DepthFirstSearchGraph(G1)
        self.G2 = DepthFirstSearchGraph(G2)

    def map_nodes(self, node1: str, node2: str) -> None:
        """Map a node in G1 to a node in G2.

        Args:
            node1: Node in G1 to map.
            node2: Node in G2 to map.
        """
        if node1 in self.mapping:
            raise ValueError(
                f"Attempting to map [{node2}] in G2 to [{node1}] in G1, but [{node1}] has already been mapped to [{self.mapping[node1]}]"  # noqa: E501
            )

        if node2 in self.G2.added:
            other_node1 = next(k for k, v in self.mapping.items() if v == node2)
            raise ValueError(
                f"Attempting to map [{node2}] in G2 to [{node1}] in G1, but [{node2}] has already been mapped to [{other_node1}]"  # noqa: E501
            )

        if node1 not in self.G1:
            raise ValueError(
                f"Attempting to map [{node2}] in G2 to [{node1}] in G1, but [{node1}] does not exist in G1"
            )

        if node2 not in self.G2:
            raise ValueError(
                f"Attempting to map [{node2}] in G2 to [{node1}] in G1, but [{node2}] does not exist in G2"
            )

        self.mapping[node1] = node2
        self.G2.added.add(node2)


def _filter_path(path: list[str], state: DepthFirstSearchState) -> Iterator[str]:
    for path_node1 in reversed(path):
        # If node is already mapped, then skip
        if path_node1 in state.mapping:
            continue

        # If path_node1 doesn't have any possible G2 mappings, don't map
        if path_node1 not in state.similarity_map:
            state.G1.not_mapped.add(path_node1)
            continue

        yield path_node1


def traverse_path_backwards(state: DepthFirstSearchState) -> None:
    """Traverse the current path backwards and attempt to map nodes during a depth first search.

    Args:
        state: Current state of the depth first search.
    """
    for path_node1 in _filter_path(path=state.path, state=state):
        # If there exists a similarity mapping with common edge, map

        # Get possible G2 node similarities for path_node1
        path_node1_possible_matches = state.similarity_map[path_node1].difference(state.G2.added)

        # Get neighbors of path_node1 similarity matches
        path_node1_possible_matches_neighs = {n: set(state.G2.neighbors[n]) for n in path_node1_possible_matches}

        # Get neighbors of path_node1 with degree > 1
        path_node1_neighs = list(state.G1.neighbors[path_node1])

        # Get neighbors of path_node1 that have already been mapped
        path_n1_neighbor_matches = {state.mapping[n] for n in path_node1_neighs if n in state.mapping}

        # Get similarity of path_node1's unmapped neighbors
        path_node1_similar_neighs = [
            state.similarity_map[n] for n in path_node1_neighs if n not in state.mapping and n in state.similarity_map
        ]

        # Get any nodes in G2 whose neighbors are valid similarities of path_node1's neighbors
        remaining_n2 = [
            n
            for n, value in path_node1_possible_matches_neighs.items()
            if (
                value.intersection(path_n1_neighbor_matches)
                or value.intersection(set().union(*path_node1_similar_neighs))
            )
        ]

        # there are no common neighbors so skip
        if not remaining_n2:
            state.G1.not_mapped.add(path_node1)
            continue

        # Map if there's only one possible mapping
        if len(remaining_n2) == 1:
            state.map_nodes(path_node1, remaining_n2[0])
            continue

        # Count number of neighbors with potential mappings
        possible_mapping_counts = {
            p: len(path_node1_possible_matches_neighs[p].intersection(path_n1_neighbor_matches))
            for p in path_node1_possible_matches
        }

        for path_node1_neigh_similarity, p in itertools.product(path_node1_similar_neighs, possible_mapping_counts):
            if path_node1_neigh_similarity.intersection(path_node1_possible_matches_neighs[p]):
                possible_mapping_counts[p] += 1

        # Determine the most likely mapping based on the neighbors
        # and map if unambiguous
        best_count = max(possible_mapping_counts.values())
        best_option = [k for k, v in possible_mapping_counts.items() if v == best_count]

        if len(best_option) == 1:
            state.map_nodes(path_node1, best_option[0])
            continue

        # Map if there is an unambiguous mapping after looking for the same degree
        same_degrees = [n for n in best_option if state.G1.degree[path_node1] == state.G2.degree[n]]

        if len(same_degrees) == 1:
            state.map_nodes(path_node1, same_degrees[0])
            continue

        if same_degrees:
            # Map if there is an unambiguous mapping after
            # removing nodes where the degree in G1 matches all
            # of the possible neighbor matches
            # note: sort this list to make it deterministic
            identicals = sorted(n for n in same_degrees if possible_mapping_counts[n] == state.G1.degree[path_node1])
            if identicals:
                state.map_nodes(path_node1, identicals[0])
            else:
                state.G1.not_mapped.add(path_node1)
                state.duplicates[path_node1] = same_degrees
        else:
            state.G1.not_mapped.add(path_node1)

    state.path.clear()


def try_to_map_node(
    node1: str,
    state: DepthFirstSearchState,
) -> None:
    """Attempt to map a node to a node in another graph during a depth first search.

    Args:
        node1: Node to attempt to map.
        state: Current depth first search state.
    """
    # Get available G2 nodes that can be mapped to from node1
    node2_poss = state.similarity_map.get(node1, set()).difference(state.G2.added)

    # Add unambiquous mappings (there only exists one node in G2 that node1 can map to)
    if len(node2_poss) == 1:
        # Get only possible G2 node
        node2 = node2_poss.pop()

        # Get neighbors of node2
        node2_neighs = state.G2.neighbors[node2]

        # Get possible matches of node1_neighs
        node1_visited_neighs = [n for n in set(state.G1.neighbors[node1]) if n in state.G1.visited]
        node1_neighs_similarity = set().union(
            *[state.similarity_map[n] for n in node1_visited_neighs if n in state.similarity_map]
        )

        # Add node1-node2 mapping if no contradiction with neighbors
        if set(node1_neighs_similarity).intersection(node2_neighs):
            state.map_nodes(node1, node2)


def try_to_map_internal_nodes(state: DepthFirstSearchState) -> None:
    """Attempt to map internal (i.e. non-leaf) nodes during a depth first search.

    Args:
        state: Current state of the depth first search.
    """
    # Run through all non-leaf nodes in G1 that aren't mapped and check for any matches
    for node1 in state.G1.not_mapped:
        if node1 in state.similarity_map and state.G1.degree[node1] > 1:
            # Add the mapping if there is an unambiguous option
            if len(state.similarity_map[node1].difference(state.G2.added)) == 1:
                state.map_nodes(node1, next(iter(state.similarity_map[node1].difference(state.G2.added))))
            # Add a mapping if there are still multiple best options
            # This covers the case where there are multiple almost identical pompoms
            elif node1 in state.duplicates:
                remaining = sorted(set(state.duplicates[node1]).difference(state.G2.added))
                if remaining:
                    state.map_nodes(node1, remaining[0])


def try_to_map_leaf_nodes(state: DepthFirstSearchState) -> None:
    """Attempt to map leaf nodes as a stage of a depth first search.

    Args:
        state: Current state of the depth first search.
    """
    # Add leaves (node of degree 1) for each node in G1 with degree > 1
    G1_leaves_not_mapped = set()
    for node1 in sorted(state.G1.visited):
        # Get all degree 1 neighbors
        node1_neighs = sorted(
            n for n in set(state.G1.neighbors[node1]).difference(state.G1.visited) if state.G1.degree[n] == 1
        )

        # Try to map node1 leaf neighbors
        for leaf1 in node1_neighs:
            # Check if mapping exists
            if leaf1 in state.similarity_map:
                # Get G2 nodes that are similar to neigh1
                neigh1_similarity = sorted(state.similarity_map[leaf1].difference(state.G2.added))
                # Check if node1 is mapped or similar to neighbor of nodes in neigh1_similarity
                for neigh2 in neigh1_similarity:
                    # Only consider nodes in G2 that have degree 1
                    if (
                        state.G2.degree[neigh2] == 1
                        and node1 in state.mapping
                        and state.mapping[node1] in state.G2.neighbors[neigh2]
                    ):
                        # Map if neigh2's neighbor is a mapping of node1
                        state.map_nodes(leaf1, neigh2)
                        break
            if leaf1 not in state.mapping:
                G1_leaves_not_mapped.add(leaf1)

    # Run through all leaves in G1 that aren't mapped and check for any matches
    for node1 in sorted(G1_leaves_not_mapped):
        # If the node has a similarity mapping, map it even if they don't share a neighbor
        available_mappings = sorted(state.similarity_map.get(node1, set()).difference(state.G2.added))
        if available_mappings:
            state.map_nodes(node1, available_mappings[0])


def calc_depth_first_search_mapping(
    G1: nx.Graph,
    G2: nx.Graph,
    similarity_map: dict,
    g1_starting_node: str,
    g2_starting_node: str,
) -> dict:
    """Use a depth first search (DFS) approach to map nodes in G1 to nodes in G2 based on node attributes and neighbors.

    Returns:
        dict: Mapping of nodes in G1 to nodes in G2.
    """
    print(f"Starting node is: {g1_starting_node} mapping to {g2_starting_node}")

    state = DepthFirstSearchState(G1=G1, G2=G2, similarity_map=similarity_map)

    # Initialize mapping
    if g2_starting_node:
        state.map_nodes(g1_starting_node, g2_starting_node)
    else:
        state.map_nodes(g1_starting_node, next(iter(similarity_map[g1_starting_node])))

    # Initialize variables
    state.G1.visited.add(g1_starting_node)
    state.stack = sorted(state.G1.neighbors[g1_starting_node])

    # Loop through all nodes of degree > 1 via depth first search traversal
    while state.stack:
        # Get next node in the graph
        node1 = state.stack.pop()
        state.G1.visited.add(node1)
        state.path.append(node1)

        try_to_map_node(
            node1=node1,
            state=state,
        )

        # Get unvisited neighbors of node1 of degree > 1
        node1_neighs = sorted(
            n for n in set(state.G1.neighbors[node1]).difference(state.G1.visited) if state.G1.degree[n] > 1
        )
        if node1_neighs:
            for neigh1 in node1_neighs:
                if neigh1 not in state.stack:
                    state.stack.append(neigh1)

        # This must be the end of the path so we will traverse backwards
        else:
            traverse_path_backwards(state=state)

    try_to_map_internal_nodes(state=state)

    try_to_map_leaf_nodes(state=state)

    # Convert values to sets
    mapping = {k: {v} for k, v in state.mapping.items()}
    return mapping
