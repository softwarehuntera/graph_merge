# Test merge_graphs.py
import pytest

import networkx as nx

from graph_merge.merge_graphs import (
    _check_similarity_of_values,
    compile_set_of_attributes,
    GraphMerger,
    NodeMap,
)


def test_node_map():
    """Initialization of NodeMap."""
    map_type = "depth_first_search"
    node_map = {"a1": {"a2"}, "b1": {"b2"}, "c1": {"c2", "c3"}}
    attributes = ["name"]
    match_function_args = {"exact_match": {}}
    starting_node = "a1"

    node_mapping = NodeMap(
        map_type=map_type,
        node_map=node_map,
        attributes=attributes,
        match_function_args=match_function_args,
        starting_node=starting_node,
    )

    assert isinstance(node_mapping, NodeMap)
    assert node_mapping.map_type == map_type
    assert node_mapping.attributes == attributes
    assert node_mapping.match_function_args == match_function_args
    assert node_mapping.node_map == node_map
    assert node_mapping.kwargs == {"starting_node": "a1"}


def test_order_dict():
    """Dictionary id ordered to ensure hash is reproducible."""
    test_dict = {
        "third": {"third": "third"},
        "first": {"end": 2, "beginning": 1},
        "second": {1: 3, 3: 2, 2: 1},
    }
    assert NodeMap.order_dict(dictionary={}) == {}
    assert NodeMap.order_dict(dictionary={"single_key": "single_value"}) == {"single_key": "single_value"}
    assert NodeMap.order_dict(dictionary=test_dict) == {
        "first": {"beginning": 1, "end": 2},
        "second": {1: 3, 2: 1, 3: 2},
        "third": {"third": "third"},
    }


def test__string_sort_attributes():
    """Attribute names are correctly sorted to ensure hash is reproducible."""
    assert NodeMap._string_sort_attributes(attributes=["name"]) == "name"
    assert NodeMap._string_sort_attributes(attributes=["name", "md5"]) == "md5name"


def test__string_sort_kwargs():
    """kwargs are correctly sorted in order to ensure hash is reproducible."""
    assert NodeMap._string_sort_kwargs(o={"exact_match": {}}) == "(('exact_match', '()'),)"
    assert (
        NodeMap._string_sort_kwargs(
            o={
                "jaro_match": {"threshold": 0.5},
                "string_similarity_cosine": {"threshold": 0.5, "ngram_size": 3},
                "exact_match": {},
            }
        )
        == "(('jaro_match', \"(('threshold', 0.5),)\"), ('string_similarity_cosine', \"(('threshold', 0.5), ('ngram_size', 3))\"), ('exact_match', '()'))"
    )


def test_map_hash():
    """Node map hashes are correctly calculated."""
    assert (
        NodeMap.map_hash(
            map_type="depth_first_search",
            attributes=["name"],
            match_function_args={"exact_match": {}},
            starting_node="some_node_id",
        )
        == "c04c74b32152960a5b005e0378abf0d3ab84363ed0591f98cec6c9ff06aa3a9d"
    )

    assert (
        NodeMap.map_hash(
            map_type="depth_first_search",
            attributes=["name", "md5"],
            match_function_args={
                "jaro_match": {"threshold": 0.5},
                "string_similarity_cosine": {"threshold": 0.5, "ngram_size": 3},
                "exact_match": {},
            },
            starting_node="some_node_id",
        )
        == "726f75d5cf90f160ab626ece97fac80c9a092fcc548ba4ca58c161d9336dde5b"
    )

    assert (
        NodeMap.map_hash(
            map_type="depth_first_search",
            attributes=["md5", "name"],
            match_function_args={
                "exact_match": {},
                "jaro_match": {"threshold": 0.5},
                "string_similarity_cosine": {"ngram_size": 3, "threshold": 0.5},
            },
            starting_node="some_node_id",
        )
        == "726f75d5cf90f160ab626ece97fac80c9a092fcc548ba4ca58c161d9336dde5b"
    )


@pytest.fixture
def graph1() -> nx.Graph:
    """Baseline graph for merge.

    Returns:
        nx.Graph: Graph.
    """
    G = nx.Graph()
    G.add_nodes_from(
        [
            ("node1", {"name": "alpha", "type": "aardvark"}),
            ("node2", {"name": "beta", "type": "beaver"}),
            ("node3", {"name": "gamma", "type": "capybara"}),
            ("node4", {"name": "delta", "type": "deer"}),
            ("node5", {"name": "epsilon", "type": "elephant"}),
        ]
    )
    G.add_edges_from(
        [
            ("node1", "node2", {"relationship": "connected"}),
            ("node1", "node3", {"relationship": "connected"}),
            ("node3", "node5", {"relationship": "connected"}),
            ("node3", "node4", {"relationship": "connected"}),
        ]
    )
    return G


@pytest.fixture
def graph2() -> nx.Graph:
    """Graph to merge onto the baseline graph.

    Returns:
        nx.Graph: Graph.
    """
    G = nx.Graph()
    G.add_nodes_from(
        [
            ("node1", {"name": "alpha", "type": "aardvarks"}),
            ("node2", {"name": "betas", "type": "beavers"}),
            ("node3", {"name": "garbage", "type": "caymen"}),
            ("node4", {"name": "deltas", "type": "dog"}),
            ("node5", {"name": "epsilons", "type": "elephant"}),
        ]
    )
    G.add_edges_from(
        [
            ("node1", "node2", {"relationship": "connected"}),
            ("node1", "node3", {"relationship": "connected"}),
            ("node3", "node5", {"relationship": "connected"}),
            ("node3", "node4", {"relationship": "connected"}),
        ]
    )
    return G


@pytest.fixture
def graph_merge(graph1: nx.Graph, graph2: nx.Graph) -> GraphMerger:
    """GraphMerger object for testing.

    Args:
        graph1 (nx.Graph): Baseline graph for merge.
        graph2 (nx.Graph): Graph to merge onto the baseline graph.

    Returns:
        GraphMerger: Merged graph object.
    """
    graph_merge = GraphMerger(G1=graph1, G2=graph2)
    graph_merge.get_similarity_map(
        attributes=["name"],
        match_function_args={"jaro_match": {"threshold": 0.7}},
    )
    return graph_merge


def test_graphmerger(graph_merge: GraphMerger):
    """Able to instantiate GraphMerger object.

    Args:
        graph_merge (GraphMerger): _description_
    """
    assert isinstance(graph_merge, GraphMerger)
    assert isinstance(graph_merge.G1, nx.Graph)
    assert isinstance(graph_merge.G2, nx.Graph)
    assert isinstance(graph_merge.similarity_maps, dict)
    assert isinstance(
        graph_merge.similarity_maps["09e04e98c2527d25c1c5135e08168e13e9ce7faed79b383fa14dd4c0f03fb12a"],
        NodeMap,
    )
    assert len(graph_merge.similarity_maps) == 1
    assert graph_merge.current_graph is None


def test__inverse_node_map(graph_merge: GraphMerger):
    """Invert the node mapping dictionary.

    Args:
        graph_merge (GraphMerger): _description_
    """
    test_map = {
        "a_node1": {"b_node1", "b_node2"},
        "a_node2": {"b_node2", "b_node3"},
        "a_node3": {"b_node4"},
    }
    assert GraphMerger._inverse_node_map(node_map=test_map) == {
        "b_node1": {"a_node1"},
        "b_node2": {"a_node1", "a_node2"},
        "b_node3": {"a_node2"},
        "b_node4": {"a_node3"},
    }


def test__deconflict_nodes(graph_merge: GraphMerger):
    """Deconflict identically named nodes between the graphs.

    Args:
        graph_merge (GraphMerger): _description_
    """
    assert set(graph_merge.G1.nodes()).intersection(set(graph_merge.G2.nodes())) == set()
    assert set(graph_merge.G2.nodes()) == {
        "node1_2",
        "node2_2",
        "node3_2",
        "node4_2",
        "node5_2",
    }


def test__node_attribute_mapping(graph_merge: GraphMerger):
    """Node mapping is accurate with different methods.

    Args:
        graph_merge (GraphMerger): _description_
    """
    assert GraphMerger._node_attribute_mapping(
        g1_node_attributes=graph_merge.g1_nodes,
        g2_node_attributes=graph_merge.g2_nodes,
        attributes=["name"],
        match_function_args={"jaro_match": {"threshold": 0.7}},
    ) == {
        "node1": {"node1_2"},
        "node2": {"node2_2", "node4_2"},
        "node4": {"node2_2", "node4_2"},
        "node5": {"node5_2"},
    }

    assert GraphMerger._node_attribute_mapping(
        g1_node_attributes=graph_merge.g1_nodes,
        g2_node_attributes=graph_merge.g2_nodes,
        attributes=[["name"], ["type"]],
        match_function_args={"exact_match": {}},
    ) == {"node1": {"node1_2"}, "node5": {"node5_2"}}


def test_get_maps(graph_merge: GraphMerger):
    """Successfully create and manage a simlilarity map.

    Args:
        graph_merge (GraphMerger): _description_
    """
    # Create a map, check the hash
    assert (
        graph_merge.get_similarity_map(
            attributes=["name"],
            match_function_args={"jaro_match": {"threshold": 0.7}},
        )
        == graph_merge.similarity_maps["09e04e98c2527d25c1c5135e08168e13e9ce7faed79b383fa14dd4c0f03fb12a"]
    )
    assert len(graph_merge.similarity_maps.keys()) == 1
    assert graph_merge.similarity_maps["09e04e98c2527d25c1c5135e08168e13e9ce7faed79b383fa14dd4c0f03fb12a"].kwargs == {}

    # Modify the map
    graph_merge.similarity_maps["09e04e98c2527d25c1c5135e08168e13e9ce7faed79b383fa14dd4c0f03fb12a"].kwargs = {
        "modified": "map"
    }

    # Call same map. Verify it isn't recalculating by recovering our modification.
    assert (
        graph_merge.get_similarity_map(
            attributes=["name"],
            match_function_args={"jaro_match": {"threshold": 0.7}},
        )
        == graph_merge.similarity_maps["09e04e98c2527d25c1c5135e08168e13e9ce7faed79b383fa14dd4c0f03fb12a"]
    )
    assert graph_merge.similarity_maps["09e04e98c2527d25c1c5135e08168e13e9ce7faed79b383fa14dd4c0f03fb12a"].kwargs == {
        "modified": "map"
    }
    assert len(graph_merge.similarity_maps.keys()) == 1

    # New hash and new map
    g1_starting_node = "node1"
    g2_starting_node = "node1_2"

    assert (
        graph_merge.get_depth_first_search_map(
            attributes=["name"],
            match_function_args={"exact_match": {}},
            g1_starting_node=g1_starting_node,
            g2_starting_node=g2_starting_node,
        )
        == graph_merge.similarity_maps["64323e24ac20aea3b31418436bd97e19444917d600f893a44c7199d62cd3de1a"]
    )

    actual_result = graph_merge.get_depth_first_search_map(
        attributes=["name"],
        match_function_args={"exact_match": {}},
        g1_starting_node=g1_starting_node,
        g2_starting_node=g2_starting_node,
    ).node_map

    expected_result = {"node1": {"node1_2"}}

    assert actual_result == expected_result

    assert len(graph_merge.similarity_maps.keys()) == 3


def test_get_dfs_map_without_g2_starting_node(graph_merge):
    # New hash and new map
    g1_starting_node = "node1"

    actual_result = graph_merge.get_depth_first_search_map(
        attributes=["name"],
        match_function_args={"exact_match": {}},
        g1_starting_node=g1_starting_node,
    ).node_map

    expected_result = {"node1": {"node1_2"}}

    assert actual_result == expected_result

    assert len(graph_merge.similarity_maps.keys()) == 3


def test_get_dfs_map_without_deconflicted_g2_node(graph_merge):
    # New hash and new map
    g1_starting_node = "node1"
    g2_starting_node = "node1"

    actual_result = graph_merge.get_depth_first_search_map(
        attributes=["name"],
        match_function_args={"exact_match": {}},
        g1_starting_node=g1_starting_node,
        g2_starting_node=g2_starting_node,
    ).node_map

    expected_result = {"node1": {"node1_2"}}

    assert actual_result == expected_result

    assert len(graph_merge.similarity_maps.keys()) == 3


def test_get_dfs_map_with_nonexistent_node(graph_merge):
    # New hash and new map
    g1_starting_node = "node1"
    g2_starting_node = "not a node"

    with pytest.raises(ValueError, match=r"\[not a node\] and \[not a node_2\] do not exist in G2") as e:
        graph_merge.get_depth_first_search_map(
            attributes=["name"],
            match_function_args={"exact_match": {}},
            g1_starting_node=g1_starting_node,
            g2_starting_node=g2_starting_node,
        ).node_map


def test_get_dfs_map_with_bad_g1_start(graph_merge):
    g1_starting_node = "node3"

    with pytest.raises(ValueError):
        graph_merge.get_depth_first_search_map(
            attributes=["name"],
            match_function_args={"exact_match": {}},
            g1_starting_node=g1_starting_node,
        ).node_map


def test_get_starting_node():
    """Find the correct starting node or handle when no match found."""
    test_map1 = {
        "a_node1": {"b_node1", "b_node2"},
        "a_node2": {"b_node2", "b_node3"},
        "a_node3": {"b_node4"},
    }
    assert GraphMerger.get_starting_node(node_map=test_map1) == ("a_node3", "b_node4")

    test_map2 = {
        "a_node1": {"b_node1", "b_node2"},
        "a_node2": {"b_node2", "b_node3"},
        "a_node3": {"b_node4", "b_node3"},
    }
    with pytest.raises(ValueError):
        GraphMerger.get_starting_node(node_map=test_map2)


def test_create_merge_graph_from_mapping_collapse_true(graph_merge: GraphMerger):
    """Created graph collapses leaf nodes.

    Args:
        graph_merge (GraphMerger): _description_
    """
    graph_merge.create_merge_graph_from_mapping(
        node_map=graph_merge.similarity_maps["09e04e98c2527d25c1c5135e08168e13e9ce7faed79b383fa14dd4c0f03fb12a"],
        collapse=True,
    )
    assert {node for node in graph_merge.current_graph.nodes()} == {
        "node1",
        "node3",
        "node5",
        "node3_2",
        "node1_supernode",
        "node3_supernode",
    }
    assert graph_merge.current_graph.nodes()["node1_supernode"]["num_nodes"] == 1


def test_create_merge_graph_from_mapping_collapse_false(graph_merge: GraphMerger):
    """Created graph does not collapse leaf nodes.

    Args:
        graph_merge (GraphMerger): _description_
    """
    graph_merge.create_merge_graph_from_mapping(
        node_map=graph_merge.similarity_maps["09e04e98c2527d25c1c5135e08168e13e9ce7faed79b383fa14dd4c0f03fb12a"],
        collapse=False,
    )
    assert {node for node in graph_merge.current_graph.nodes()} == {
        "node1",
        "node2",
        "node3",
        "node3_2",
        "node4",
        "node5",
    }


def test_create_merge_graph_from_mapping_graph_name(graph_merge: GraphMerger):
    """Graphs are correctly renamed in the merge graph.

    Args:
        graph_merge (GraphMerger): _description_
    """
    graph_merge.create_merge_graph_from_mapping(
        node_map=graph_merge.similarity_maps["09e04e98c2527d25c1c5135e08168e13e9ce7faed79b383fa14dd4c0f03fb12a"],
        g1_name="graph_label_1",
        g2_name="graph_label_2",
    )
    assert graph_merge.current_graph.nodes["node3"]["graph"] == "graph_label_1"


def test_overlay_node_map_on_graph(graph_merge: GraphMerger):
    """Overlay_node_map function suggests a possible match.

    Args:
        graph_merge (GraphMerger): _description_
    """
    graph_merge.create_merge_graph_from_mapping(
        node_map=graph_merge.similarity_maps["09e04e98c2527d25c1c5135e08168e13e9ce7faed79b383fa14dd4c0f03fb12a"],
        collapse=False,
    )
    assert len(graph_merge.current_graph.nodes()) == 6
    assert len(graph_merge.current_graph.edges()) == 6
    graph_merge.overlay_node_map_on_graph(attributes=["name"], match_function_args={"jaro_match": {"threshold": 0.6}})
    assert len(graph_merge.current_graph.nodes()) == 6
    assert len(graph_merge.current_graph.edges()) == 7
    assert graph_merge.current_graph.get_edge_data("node3", "node3_2") == {
        "graph": "Possible Match",
        "color": "#69CC3E",
        "alpha": 0.4,
    }


def test_compile_set_of_attributes():
    assert compile_set_of_attributes(["A", "B"]) == {"A", "B"}
    assert compile_set_of_attributes([]) == set()
    assert compile_set_of_attributes([["A", "B"], ["C", "D"]]) == {"A", "B", "C", "D"}


def test__check_similarity_of_values():
    match_function_args = {"exact_match": {}}

    left_values = {"name": "A"}
    right_values = {"name": "A"}
    attributes = ["name"]
    assert _check_similarity_of_values(left_values, right_values, attributes, match_function_args)


def test__check_similarity_of_values_and():
    match_function_args = {"exact_match": {}}

    left_values = {"name": "A", "color": "blue"}
    right_values = {"name": "A", "color": "blue"}
    attributes = ["name", "color"]
    assert _check_similarity_of_values(left_values, right_values, attributes, match_function_args)

    left_values = {"name": "A", "color": "blue"}
    right_values = {"name": "A", "color": "green"}
    attributes = ["name", "color"]
    assert not _check_similarity_of_values(left_values, right_values, attributes, match_function_args)


def test__check_similarity_of_values_or():
    match_function_args = {"exact_match": {}}

    left_values = {"name": "A", "color": "blue"}
    right_values = {"name": "B", "color": "blue"}
    attributes = [["name"], ["color"]]
    assert _check_similarity_of_values(left_values, right_values, attributes, match_function_args)

    left_values = {"name": "A", "color": "blue"}
    right_values = {"name": "A", "color": "green"}
    attributes = [["name"], ["color"]]
    assert _check_similarity_of_values(left_values, right_values, attributes, match_function_args)

    left_values = {"name": "A", "color": "blue"}
    right_values = {"name": "B", "color": "green"}
    attributes = [["name"], ["color"]]
    assert not _check_similarity_of_values(left_values, right_values, attributes, match_function_args)


def test__check_similarity_of_values_and_or():
    match_function_args = {"exact_match": {}}

    left_values = {"name": "A", "color": "blue", "shape": "square"}
    right_values = {"name": "B", "color": "blue", "shape": "square"}
    attributes = [["name"], ["color", "shape"]]
    assert _check_similarity_of_values(left_values, right_values, attributes, match_function_args)

    left_values = {"name": "A", "color": "blue", "shape": "square"}
    right_values = {"name": "A", "color": "green", "shape": "square"}
    attributes = [["name"], ["color", "shape"]]
    assert _check_similarity_of_values(left_values, right_values, attributes, match_function_args)

    left_values = {"name": "A", "color": "blue", "shape": "square"}
    right_values = {"name": "B", "color": "green", "shape": "square"}
    attributes = [["name"], ["color", "shape"]]
    assert not _check_similarity_of_values(left_values, right_values, attributes, match_function_args)


def test__check_similarity_of_values_bad_attributes():
    match_function_args = {"exact_match": {}}

    left_values = {"name": "A"}
    right_values = {"name": "B"}

    with pytest.raises(ValueError) as e:
        _check_similarity_of_values(left_values, right_values, [], match_function_args)

    assert "is empty" in str(e.value)

    with pytest.raises(ValueError) as e:
        _check_similarity_of_values(left_values, right_values, None, match_function_args)

    assert "is None" in str(e.value)
