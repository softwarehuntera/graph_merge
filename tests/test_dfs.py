import networkx as nx
import pytest

from graph_merge.dfs import (
    DepthFirstSearchState,
    traverse_path_backwards,
    try_to_map_node,
    try_to_map_internal_nodes,
    try_to_map_leaf_nodes,
    calc_depth_first_search_mapping,
)


def test_dfsstate_existing_mapping_g1():
    state = DepthFirstSearchState(
        G1=nx.Graph({"A": {}}),
        G2=nx.Graph({"a": {}, "b": {}}),
        similarity_map={},
    )
    state.map_nodes("A", "b")

    with pytest.raises(ValueError) as e:
        state.map_nodes("A", "a")

    assert "already been mapped" in str(e.value)


def test_dfsstate_existing_mapping_g2():
    state = DepthFirstSearchState(
        G1=nx.Graph({"A": {}, "B": {}}),
        G2=nx.Graph({"a": {}}),
        similarity_map={},
    )

    state.map_nodes("A", "a")

    with pytest.raises(ValueError) as e:
        state.map_nodes("B", "a")

    assert "already been mapped" in str(e.value)


def test_dfsstate_nonexistent_g2_node():
    state = DepthFirstSearchState(
        G1=nx.Graph({"A": {}}),
        G2=nx.Graph(),
        similarity_map={},
    )
    with pytest.raises(ValueError) as e:
        state.map_nodes("A", "b")

    assert "does not exist in G2" in str(e.value)


def test_dfsstate_nonexistent_g1_node():
    state = DepthFirstSearchState(
        G1=nx.Graph(),
        G2=nx.Graph({"b": {}}),
        similarity_map={},
    )
    with pytest.raises(ValueError) as e:
        state.map_nodes("A", "b")

    assert "does not exist in G1" in str(e.value)


def test_traverse_path_backwards():
    state = DepthFirstSearchState(
        G1=nx.Graph([("A", "B"), ("B", "C"), ("B", "D"), ("C", "E")]),
        G2=nx.Graph([("a", "b"), ("b", "c"), ("c", "e")]),
        similarity_map={
            "A": {"a"},
            "B": {"b"},
            "C": {"c"},
            "E": {"e"},
        },
    )

    state.path = ["A", "B", "C", "E"]
    traverse_path_backwards(state=state)

    assert len(state.path) == 0
    assert len(state.G2.added) == 4
    assert len(state.duplicates) == 0
    assert len(state.stack) == 0
    assert len(state.G1.not_mapped) == 0
    assert state.mapping == {"A": "a", "B": "b", "C": "c", "E": "e"}


def test_traverse_path_backwards_existing_mapping():
    state = DepthFirstSearchState(
        G1=nx.Graph([("A", "B"), ("B", "C"), ("B", "D"), ("C", "E")]),
        G2=nx.Graph([("a", "b"), ("b", "c"), ("c", "e")]),
        similarity_map={
            "A": {"a"},
            "B": {"b"},
            "C": {"c"},
            "E": {"e"},
        },
    )

    state.path = ["A", "B", "C", "E"]
    state.map_nodes("C", "a")
    traverse_path_backwards(state=state)

    assert len(state.path) == 0
    assert len(state.G2.added) == 2
    assert len(state.duplicates) == 0
    assert len(state.stack) == 0
    assert len(state.G1.not_mapped) == 2
    assert state.mapping == {"B": "b", "C": "a"}


def test_traverse_path_backwards_no_similarity_mapping():
    state = DepthFirstSearchState(
        G1=nx.Graph([("A", "B"), ("B", "C"), ("B", "D"), ("C", "E")]),
        G2=nx.Graph([("a", "b"), ("b", "c"), ("c", "e")]),
        similarity_map={
            "A": {"a"},
            "B": {"b"},
            "E": {"e"},
        },
    )

    state.path = ["A", "B", "C", "E"]
    traverse_path_backwards(state=state)

    assert len(state.path) == 0
    assert len(state.G2.added) == 2
    assert len(state.duplicates) == 0
    assert len(state.stack) == 0
    assert len(state.G1.not_mapped) == 2
    assert state.mapping == {"A": "a", "B": "b"}


def test_traverse_path_backwards_larger_graph():
    state = DepthFirstSearchState(
        G1=nx.Graph([("A", "B"), ("B", "C"), ("B", "D"), ("B", "F"), ("C", "E")]),
        G2=nx.Graph(
            [
                ("a", "b1"),
                ("b1", "c"),
                ("b1", "d"),
                ("b1", "f"),
                ("b2", "c"),
                ("b2", "d"),
                ("b2", "f"),
                ("c", "e"),
            ]
        ),
        similarity_map={
            "A": {"a"},
            "B": {"b1", "b2"},
            "C": {"c"},
            "D": {"d"},
            "E": {"e"},
            "F": {"f"},
        },
    )

    state.path = ["A", "B", "C", "E"]
    traverse_path_backwards(state=state)

    assert len(state.path) == 0
    assert len(state.G2.added) == 4
    assert len(state.duplicates) == 0
    assert len(state.stack) == 0
    assert len(state.G1.not_mapped) == 0
    assert state.mapping == {"A": "a", "B": "b1", "C": "c", "E": "e"}


def test_traverse_path_backwards_multiple_best_options():
    state = DepthFirstSearchState(
        G1=nx.Graph([("A", "B"), ("B", "C"), ("B", "D"), ("B", "F"), ("C", "E")]),
        G2=nx.Graph(
            [
                ("a", "b1"),
                ("a", "b2"),
                ("b1", "c"),
                ("b1", "d"),
                ("b1", "f"),
                ("b2", "c"),
                ("b2", "d"),
                ("b2", "f"),
                ("c", "e"),
            ]
        ),
        similarity_map={
            "A": {"a"},
            "B": {"b1", "b2"},
            "C": {"c"},
            "D": {"d"},
            "E": {"e"},
            "F": {"f"},
        },
    )

    state.path = ["A", "B", "C", "E"]
    traverse_path_backwards(state=state)

    assert len(state.path) == 0
    assert len(state.G2.added) == 4
    assert len(state.duplicates) == 0
    assert len(state.stack) == 0
    assert len(state.G1.not_mapped) == 0
    assert state.mapping == {"A": "a", "B": "b1", "C": "c", "E": "e"}


def test_try_to_map_node():
    state = DepthFirstSearchState(
        G1=nx.Graph([("A", "B"), ("B", "C"), ("B", "D"), ("B", "F"), ("C", "E")]),
        G2=nx.Graph([("a", "b"), ("b", "c"), ("c", "e")]),
        similarity_map={
            "A": {"a"},
            "B": {"b"},
            "C": {"c"},
            "E": {"e"},
        },
    )

    state.path.append("A")
    state.G1.visited.add("A")
    state.map_nodes("A", "a")

    try_to_map_node(node1="B", state=state)

    assert state.mapping.get("B") == "b"
    assert state.path == ["A"]


def test_try_to_map_node_ambiguous_choice():
    state = DepthFirstSearchState(
        G1=nx.Graph([("A", "B"), ("B", "C"), ("B", "D"), ("B", "F"), ("C", "E")]),
        G2=nx.Graph([("a", "b1"), ("b1", "c"), ("a", "b2"), ("b2", "c"), ("c", "e")]),
        similarity_map={
            "A": {"a"},
            "B": {"b1", "b2"},
            "C": {"c"},
            "E": {"e"},
        },
    )

    state.path.append("A")
    state.G1.visited.add("A")
    state.map_nodes("A", "a")

    try_to_map_node(node1="B", state=state)

    assert state.mapping.get("B") is None
    assert state.path == ["A"]


def test_try_to_map_internal_nodes():
    state = DepthFirstSearchState(
        G1=nx.Graph([("A", "B"), ("B", "C"), ("B", "D"), ("B", "F"), ("C", "E")]),
        G2=nx.Graph([("a", "b"), ("b", "c"), ("c", "e")]),
        similarity_map={
            "A": {"a"},
            "B": {"b"},
            "C": {"c"},
            "E": {"e"},
        },
    )

    state.G1.not_mapped.add("C")

    try_to_map_internal_nodes(state=state)

    assert state.mapping.get("C") == "c"


def test_try_to_map_internal_nodes_leaf_unmapped():
    state = DepthFirstSearchState(
        G1=nx.Graph([("A", "B"), ("B", "C"), ("B", "D"), ("B", "F"), ("C", "E")]),
        G2=nx.Graph([("a", "b"), ("b", "c"), ("c", "e")]),
        similarity_map={
            "A": {"a"},
            "B": {"b"},
            "C": {"c"},
            "E": {"e"},
        },
    )

    try_to_map_internal_nodes(state=state)

    assert "E" not in state.mapping


def test_try_to_map_internal_nodes_ambiguous_choice():
    state = DepthFirstSearchState(
        G1=nx.Graph([("A", "B"), ("B", "C"), ("B", "D"), ("B", "F"), ("C", "E")]),
        G2=nx.Graph([("a", "b1"), ("b1", "c"), ("a", "b2"), ("b2", "c"), ("c", "e")]),
        similarity_map={
            "A": {"a"},
            "B": {"b1", "b2"},
            "C": {"c"},
            "E": {"e"},
        },
    )

    state.G1.not_mapped.add("B")

    try_to_map_internal_nodes(state=state)

    assert "B" not in state.mapping


def test_try_to_map_internal_nodes_duplicates():
    state = DepthFirstSearchState(
        G1=nx.Graph([("A", "B"), ("B", "C"), ("B", "D"), ("B", "F"), ("C", "E")]),
        G2=nx.Graph([("a", "b1"), ("b1", "c"), ("a", "b2"), ("b2", "c"), ("c", "e")]),
        similarity_map={
            "A": {"a"},
            "B": {"b1", "b2"},
            "C": {"c"},
            "E": {"e"},
        },
    )

    state.G1.not_mapped.add("B")

    try_to_map_internal_nodes(state=state)

    assert "B" not in state.mapping


def test_try_to_map_leaf_nodes():
    state = DepthFirstSearchState(
        G1=nx.Graph([("A", "B"), ("B", "C"), ("B", "D"), ("B", "F"), ("C", "E")]),
        G2=nx.Graph([("a", "b"), ("b", "c"), ("c", "e")]),
        similarity_map={
            "A": {"a"},
            "B": {"b"},
            "C": {"c"},
            "E": {"e"},
        },
    )

    state.G1.visited.add("C")
    state.map_nodes("C", "c")
    try_to_map_leaf_nodes(state=state)

    assert state.mapping.get("E") == "e"


def test_try_to_map_leaf_nodes_unmapped_neighbor():
    state = DepthFirstSearchState(
        G1=nx.Graph([("A", "B"), ("B", "C"), ("B", "D"), ("B", "F"), ("C", "E")]),
        G2=nx.Graph([("a", "b"), ("b", "c"), ("c", "e")]),
        similarity_map={
            "A": {"a"},
            "B": {"b"},
            "C": {"c"},
            "E": {"e"},
        },
    )

    state.G1.visited.add("C")
    try_to_map_leaf_nodes(state=state)

    assert state.mapping.get("E") == "e"


def test_calc_depth_first_search_mapping():
    G1 = nx.Graph([("A", "B"), ("B", "C"), ("B", "D"), ("B", "F"), ("C", "E")])
    G2 = nx.Graph([("a", "b"), ("b", "c"), ("c", "e")])
    similarity_map = {
        "A": {"a"},
        "B": {"b"},
        "C": {"c"},
        "E": {"e"},
    }

    actual_result = calc_depth_first_search_mapping(
        G1=G1, G2=G2, similarity_map=similarity_map, g1_starting_node="A", g2_starting_node="a"
    )
    expected_result = {"A": {"a"}, "B": {"b"}, "C": {"c"}, "E": {"e"}}
    assert actual_result == expected_result


def test_calc_depth_first_search_mapping_arbitrary_path_is_deterministic():
    G1 = nx.Graph([("A", "B"), ("B", "C"), ("B", "D"), ("B", "F"), ("C", "E")])
    G2 = nx.Graph([("a", "b1"), ("b1", "c"), ("a", "b2"), ("b2", "b"), ("c", "e")])
    similarity_map = {
        "A": {"a"},
        "B": {"b1", "b2"},
        "C": {"c"},
        "E": {"e"},
    }

    actual_result = calc_depth_first_search_mapping(
        G1=G1, G2=G2, similarity_map=similarity_map, g1_starting_node="A", g2_starting_node="a"
    )
    expected_result = {"A": {"a"}, "B": {"b1"}, "C": {"c"}, "E": {"e"}}
    assert actual_result == expected_result


def test_calc_depth_first_search_mapping_larger_graph():
    G1 = nx.Graph(
        [
            ("A", "B"),
            ("B", "C"),
            ("C", "D"),
            ("B", "E"),
            ("E", "F"),
            ("B", "G"),
        ]
    )
    G2 = nx.Graph([("a", "b"), ("b", "c"), ("c", "d"), ("b", "e"), ("e", "f"), ("b", "g")])
    similarity_map = {
        "A": {"a"},
        "B": {"b"},
        "C": {"c"},
        "D": {"d"},
        "E": {"e"},
    }

    actual_result = calc_depth_first_search_mapping(
        G1=G1, G2=G2, similarity_map=similarity_map, g1_starting_node="A", g2_starting_node="a"
    )
    expected_result = {"A": {"a"}, "B": {"b"}, "C": {"c"}, "D": {"d"}, "E": {"e"}}
    assert actual_result == expected_result


def test_calc_depth_first_search_mapping_cycle():
    G1 = nx.Graph([("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "A")])
    G2 = nx.Graph([("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")])
    similarity_map = {
        "A": {"a"},
        "C": {"c"},
        "D": {"d"},
        "E": {"e"},
    }

    actual_result = calc_depth_first_search_mapping(
        G1=G1, G2=G2, similarity_map=similarity_map, g1_starting_node="A", g2_starting_node="a"
    )
    expected_result = {"A": {"a"}, "C": {"c"}, "D": {"d"}, "E": {"e"}}
    assert actual_result == expected_result


def test_calc_depth_first_search_mapping_missing_y():
    G1 = nx.Graph([("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "A"), ("E", "F")])
    G2 = nx.Graph([("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f")])
    similarity_map = {
        "A": {"a"},
        "B": {"b"},
        "C": {"c"},
        "D": {"d"},
        "E": {"e"},
        "F": {"f"},
    }

    actual_result = calc_depth_first_search_mapping(
        G1=G1, G2=G2, similarity_map=similarity_map, g1_starting_node="A", g2_starting_node="a"
    )
    expected_result = {"A": {"a"}, "B": {"b"}, "C": {"c"}, "D": {"d"}, "E": {"e"}, "F": {"f"}}
    assert actual_result == expected_result
