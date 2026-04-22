"""Unit tests for the small helpers in
:mod:`alphaDeesp.core.graphs.graph_utils`: colour filtering, node/edge
path conversion, incident edges, name-indexed BFS, and simple-path
enumeration over multiple sources/targets."""

import pytest
import networkx as nx

from alphaDeesp.core.graphsAndPaths import (
    delete_color_edges,
    nodepath_to_edgepath,
    incident_edges,
    from_edges_get_nodes,
    all_simple_edge_paths_multi,
    find_multidigraph_edges_by_name,
)
from alphaDeesp.tests.graphs_test_helpers import (
    make_colored_multidigraph,
    make_linear_multidigraph,
)


# ──────────────────────────────────────────────────────────────────────
# delete_color_edges
# ──────────────────────────────────────────────────────────────────────

class TestDeleteColorEdges:

    def test_removes_gray_edges(self):
        g = make_colored_multidigraph()
        result = delete_color_edges(g, "gray")
        assert "gray" not in set(nx.get_edge_attributes(result, "color").values())

    def test_removes_blue_edges(self):
        g = make_colored_multidigraph()
        result = delete_color_edges(g, "blue")
        colors = set(nx.get_edge_attributes(result, "color").values())
        assert "blue" not in colors
        assert "gray" in colors and "coral" in colors

    def test_removes_coral_edges(self):
        g = make_colored_multidigraph()
        result = delete_color_edges(g, "coral")
        assert "coral" not in set(nx.get_edge_attributes(result, "color").values())

    def test_removes_isolated_nodes(self):
        """After removing gray edges, C becomes isolated (only connected by gray)."""
        g = nx.MultiDiGraph()
        g.add_edge("A", "B", color="blue", capacity=-5., name="line_AB")
        g.add_edge("B", "C", color="gray", capacity=0., name="line_BC")
        result = delete_color_edges(g, "gray")
        assert "C" not in result.nodes()
        assert "A" in result.nodes() and "B" in result.nodes()

    def test_nonexistent_color_returns_same(self):
        g = make_colored_multidigraph()
        assert delete_color_edges(g, "purple").number_of_edges() == g.number_of_edges()

    def test_does_not_modify_original(self):
        g = make_colored_multidigraph()
        original = g.number_of_edges()
        delete_color_edges(g, "gray")
        assert g.number_of_edges() == original

    def test_empty_graph(self):
        result = delete_color_edges(nx.MultiDiGraph(), "gray")
        assert result.number_of_edges() == 0
        assert result.number_of_nodes() == 0

    def test_all_same_color_removes_everything(self):
        g = nx.MultiDiGraph()
        g.add_edge("A", "B", color="gray", capacity=0., name="l1")
        g.add_edge("B", "C", color="gray", capacity=0., name="l2")
        result = delete_color_edges(g, "gray")
        assert result.number_of_edges() == 0
        assert result.number_of_nodes() == 0


# ──────────────────────────────────────────────────────────────────────
# nodepath_to_edgepath
# ──────────────────────────────────────────────────────────────────────

class TestNodepathToEdgepath:

    def test_simple_digraph(self):
        g = nx.DiGraph()
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        assert nodepath_to_edgepath(g, ["A", "B", "C"]) == [("A", "B"), ("B", "C")]

    def test_multidigraph_with_keys(self):
        g = nx.MultiDiGraph()
        g.add_edge("A", "B", name="e1")
        g.add_edge("B", "C", name="e2")
        result = nodepath_to_edgepath(g, ["A", "B", "C"], with_keys=True)
        assert len(result) == 2
        assert result[0][:2] == ("A", "B")
        assert len(result[0]) == 3

    def test_multidigraph_parallel_edges_returns_all_keys(self):
        g = nx.MultiDiGraph()
        g.add_edge("A", "B", name="e1")
        g.add_edge("A", "B", name="e2")
        g.add_edge("B", "C", name="e3")
        result = nodepath_to_edgepath(g, ["A", "B", "C"], with_keys=True)
        ab_edges = [e for e in result if e[0] == "A" and e[1] == "B"]
        assert len(ab_edges) == 2

    def test_single_edge_path(self):
        g = nx.DiGraph()
        g.add_edge("A", "B")
        assert nodepath_to_edgepath(g, ["A", "B"]) == [("A", "B")]

    def test_single_node_returns_empty(self):
        g = nx.DiGraph()
        g.add_node("A")
        assert nodepath_to_edgepath(g, ["A"]) == []

    def test_empty_path_returns_empty(self):
        assert nodepath_to_edgepath(nx.DiGraph(), []) == []


# ──────────────────────────────────────────────────────────────────────
# incident_edges
# ──────────────────────────────────────────────────────────────────────

class TestIncidentEdges:

    def test_basic_directed(self):
        g = nx.DiGraph()
        g.add_edge("A", "B", weight=1)
        g.add_edge("C", "A", weight=2)
        assert len(incident_edges(g, "A", data=True)) == 2

    def test_multidigraph_with_keys(self):
        g = nx.MultiDiGraph()
        g.add_edge("A", "B", name="e1")
        g.add_edge("C", "A", name="e2")
        result = incident_edges(g, "A", data=True, keys=True)
        assert len(result) == 2
        assert len(result[0]) == 4  # (u, v, key, data)

    def test_isolated_node(self):
        g = nx.DiGraph()
        g.add_node("A")
        assert incident_edges(g, "A", data=True) == []

    def test_self_loop(self):
        g = nx.DiGraph()
        g.add_edge("A", "A", weight=1)
        # self-loop appears in both in and out iterators
        assert len(incident_edges(g, "A", data=False)) == 2


# ──────────────────────────────────────────────────────────────────────
# from_edges_get_nodes
# ──────────────────────────────────────────────────────────────────────

class TestFromEdgesGetNodes:

    def test_basic_edges(self):
        result = from_edges_get_nodes(
            [("A", "B", 0), ("B", "C", 0)], "amont", ("C", "D", 0))
        assert result == ["A", "B", "C"]

    def test_preserves_order_and_deduplicates(self):
        result = from_edges_get_nodes(
            [("A", "B", 0), ("B", "C", 0), ("A", "C", 0)], "amont", ("C", "D", 0))
        assert result == ["A", "B", "C"]

    def test_empty_edges_amont(self):
        assert from_edges_get_nodes([], "amont", ("X", "Y", 0)) == ["X"]

    def test_empty_edges_aval(self):
        assert from_edges_get_nodes([], "aval", ("X", "Y", 0)) == ["Y"]

    def test_empty_edges_invalid_direction_raises(self):
        with pytest.raises(ValueError):
            from_edges_get_nodes([], "invalid", ("X", "Y", 0))

    def test_single_edge(self):
        assert from_edges_get_nodes([("A", "B", 0)], "amont", ("B", "C", 0)) == ["A", "B"]


# ──────────────────────────────────────────────────────────────────────
# find_multidigraph_edges_by_name
# ──────────────────────────────────────────────────────────────────────

class TestFindMultidigraphEdgesByName:

    def test_finds_direct_neighbor(self):
        g = make_linear_multidigraph()
        assert "e1" in find_multidigraph_edges_by_name(g, 1, {"e1"}, depth=1)

    def test_finds_within_depth(self):
        g = make_linear_multidigraph()
        assert "e2" in find_multidigraph_edges_by_name(g, 1, {"e2"}, depth=2)

    def test_does_not_find_beyond_depth(self):
        g = make_linear_multidigraph()
        assert "e3" not in find_multidigraph_edges_by_name(g, 1, {"e3"}, depth=1)

    def test_finds_nothing_with_nonexistent_name(self):
        g = make_linear_multidigraph()
        assert find_multidigraph_edges_by_name(g, 1, {"no_such"}, depth=5) == []

    def test_multiple_targets(self):
        g = make_linear_multidigraph()
        result = find_multidigraph_edges_by_name(g, 1, {"e1", "e2"}, depth=2)
        assert "e1" in result and "e2" in result

    def test_depth_zero_finds_nothing(self):
        g = make_linear_multidigraph()
        assert find_multidigraph_edges_by_name(g, 1, {"e1"}, depth=0) == []

    def test_parallel_edges(self):
        g = nx.MultiDiGraph()
        g.add_edge("A", "B", name="line1")
        g.add_edge("A", "B", name="line2")
        result = find_multidigraph_edges_by_name(g, "A", {"line1", "line2"}, depth=1)
        assert "line1" in result and "line2" in result

    def test_isolated_source(self):
        g = nx.MultiDiGraph()
        g.add_node("X")
        g.add_edge("A", "B", name="line1")
        assert find_multidigraph_edges_by_name(g, "X", {"line1"}, depth=5) == []


# ──────────────────────────────────────────────────────────────────────
# all_simple_edge_paths_multi
# ──────────────────────────────────────────────────────────────────────

class TestAllSimpleEdgePathsMulti:

    def test_basic_path(self):
        g = nx.DiGraph()
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        paths = list(all_simple_edge_paths_multi(g, ["A"], ["C"]))
        assert len(paths) == 1
        assert paths[0] == [("A", "B"), ("B", "C")]

    def test_multiple_sources_and_targets(self):
        g = nx.DiGraph()
        g.add_edge("A", "C")
        g.add_edge("B", "C")
        g.add_edge("A", "D")
        paths = list(all_simple_edge_paths_multi(g, ["A", "B"], ["C", "D"]))
        assert len(paths) >= 2

    def test_same_source_and_target_skipped(self):
        g = nx.DiGraph()
        g.add_edge("A", "B")
        assert list(all_simple_edge_paths_multi(g, ["A"], ["A"])) == []

    def test_no_path_exists(self):
        g = nx.DiGraph()
        g.add_edge("A", "B")
        g.add_edge("C", "D")
        assert list(all_simple_edge_paths_multi(g, ["A"], ["D"])) == []

    def test_with_cutoff(self):
        g = nx.DiGraph()
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "D")
        assert list(all_simple_edge_paths_multi(g, ["A"], ["D"], cutoff=1)) == []

    def test_source_not_in_graph(self):
        g = nx.DiGraph()
        g.add_edge("A", "B")
        assert list(all_simple_edge_paths_multi(g, ["X"], ["B"])) == []
