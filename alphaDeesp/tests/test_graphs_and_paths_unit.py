"""
Unit tests for helper functions and edge cases in graphsAndPaths.py.
Uses hand-crafted graphs to test each function in isolation.
"""
import pytest
import networkx as nx
from alphaDeesp.core.graphsAndPaths import (
    ConstrainedPath,
    delete_color_edges,
    nodepath_to_edgepath,
    incident_edges,
    from_edges_get_nodes,
    all_simple_edge_paths_multi,
    add_double_edges_null_redispatch,
    remove_unused_added_double_edge,
    find_multidigraph_edges_by_name,
    shortest_path_with_promoted_edges,
    shortest_path_min_weight_then_hops,
    OverFlowGraph,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures: small hand-crafted graphs
# ──────────────────────────────────────────────────────────────────────

def _make_colored_multidigraph():
    """
    Build a small MultiDiGraph with mixed edge colors:

      A --blue--> B --gray--> C --coral--> D
                  B --black-> C
                  A --gray--> D
    """
    g = nx.MultiDiGraph()
    g.add_edge("A", "B", color="blue", capacity=-5., name="line_AB")
    g.add_edge("B", "C", color="gray", capacity=0., name="line_BC")
    g.add_edge("C", "D", color="coral", capacity=3., name="line_CD")
    g.add_edge("B", "C", color="black", capacity=-10., name="line_BC_black")
    g.add_edge("A", "D", color="gray", capacity=0., name="line_AD")
    return g


def _make_linear_multidigraph():
    """
    Build a linear MultiDiGraph for path testing:

      1 --e1--> 2 --e2--> 3 --e3--> 4 --e4--> 5

    All edges gray with capacity 0, suitable for null-flow tests.
    """
    g = nx.MultiDiGraph()
    g.add_edge(1, 2, color="gray", capacity=0., name="e1")
    g.add_edge(2, 3, color="gray", capacity=0., name="e2")
    g.add_edge(3, 4, color="gray", capacity=0., name="e3")
    g.add_edge(4, 5, color="gray", capacity=0., name="e4")
    return g


def _make_branching_multidigraph():
    """
    Build a branching MultiDiGraph:

      S --s1--> A --a1--> B --b1--> T
                A --a2--> C --c1--> T

    All edges gray with capacity 0, with disconnect line on a2.
    """
    g = nx.MultiDiGraph()
    g.add_edge("S", "A", color="gray", capacity=0., name="s1")
    g.add_edge("A", "B", color="gray", capacity=0., name="a1")
    g.add_edge("B", "T", color="gray", capacity=0., name="b1")
    g.add_edge("A", "C", color="gray", capacity=0., name="a2_disconnected")
    g.add_edge("C", "T", color="gray", capacity=0., name="c1")
    return g


def _make_detect_edges_graph():
    """
    Build a graph suitable for detect_edges_to_keep testing.

    Structure:
      SRC1 --gray(0)--> MID --gray(0)--> TGT1
              (line_disconnect is on MID->TGT1 edge)

    Returns (OverFlowGraph-like object with .g attribute, graph, edges_of_interest)
    """
    g = nx.MultiDiGraph()
    g.add_edge("SRC1", "MID", color="gray", capacity=0., name="line_SM")
    g.add_edge("MID", "TGT1", color="gray", capacity=0., name="line_disconnect")
    return g


# ──────────────────────────────────────────────────────────────────────
# Tests for delete_color_edges
# ──────────────────────────────────────────────────────────────────────

class TestDeleteColorEdges:

    def test_removes_gray_edges(self):
        g = _make_colored_multidigraph()
        result = delete_color_edges(g, "gray")
        colors = set(nx.get_edge_attributes(result, "color").values())
        assert "gray" not in colors

    def test_removes_blue_edges(self):
        g = _make_colored_multidigraph()
        result = delete_color_edges(g, "blue")
        colors = set(nx.get_edge_attributes(result, "color").values())
        assert "blue" not in colors
        assert "gray" in colors
        assert "coral" in colors

    def test_removes_coral_edges(self):
        g = _make_colored_multidigraph()
        result = delete_color_edges(g, "coral")
        colors = set(nx.get_edge_attributes(result, "color").values())
        assert "coral" not in colors

    def test_removes_isolated_nodes(self):
        """After removing gray edges, node D should become isolated if only connected by gray."""
        g = nx.MultiDiGraph()
        g.add_edge("A", "B", color="blue", capacity=-5., name="line_AB")
        g.add_edge("B", "C", color="gray", capacity=0., name="line_BC")
        result = delete_color_edges(g, "gray")
        assert "C" not in result.nodes()
        assert "A" in result.nodes()
        assert "B" in result.nodes()

    def test_nonexistent_color_returns_same(self):
        g = _make_colored_multidigraph()
        original_edge_count = g.number_of_edges()
        result = delete_color_edges(g, "purple")
        assert result.number_of_edges() == original_edge_count

    def test_does_not_modify_original(self):
        g = _make_colored_multidigraph()
        original_edges = g.number_of_edges()
        _ = delete_color_edges(g, "gray")
        assert g.number_of_edges() == original_edges

    def test_empty_graph(self):
        g = nx.MultiDiGraph()
        result = delete_color_edges(g, "gray")
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
# Tests for nodepath_to_edgepath
# ──────────────────────────────────────────────────────────────────────

class TestNodepathToEdgepath:

    def test_simple_digraph(self):
        g = nx.DiGraph()
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        result = nodepath_to_edgepath(g, ["A", "B", "C"])
        assert result == [("A", "B"), ("B", "C")]

    def test_multidigraph_with_keys(self):
        g = nx.MultiDiGraph()
        g.add_edge("A", "B", name="e1")
        g.add_edge("B", "C", name="e2")
        result = nodepath_to_edgepath(g, ["A", "B", "C"], with_keys=True)
        assert len(result) == 2
        assert result[0][0] == "A"
        assert result[0][1] == "B"
        assert len(result[0]) == 3  # (u, v, key)

    def test_multidigraph_parallel_edges_returns_all_keys(self):
        g = nx.MultiDiGraph()
        g.add_edge("A", "B", name="e1")
        g.add_edge("A", "B", name="e2")
        g.add_edge("B", "C", name="e3")
        result = nodepath_to_edgepath(g, ["A", "B", "C"], with_keys=True)
        # Should return both parallel edges A->B
        ab_edges = [e for e in result if e[0] == "A" and e[1] == "B"]
        assert len(ab_edges) == 2

    def test_single_edge_path(self):
        g = nx.DiGraph()
        g.add_edge("A", "B")
        result = nodepath_to_edgepath(g, ["A", "B"])
        assert result == [("A", "B")]

    def test_single_node_returns_empty(self):
        g = nx.DiGraph()
        g.add_node("A")
        result = nodepath_to_edgepath(g, ["A"])
        assert result == []

    def test_empty_path_returns_empty(self):
        g = nx.DiGraph()
        result = nodepath_to_edgepath(g, [])
        assert result == []


# ──────────────────────────────────────────────────────────────────────
# Tests for incident_edges
# ──────────────────────────────────────────────────────────────────────

class TestIncidentEdges:

    def test_basic_directed(self):
        g = nx.DiGraph()
        g.add_edge("A", "B", weight=1)
        g.add_edge("C", "A", weight=2)
        result = incident_edges(g, "A", data=True)
        assert len(result) == 2  # one out, one in

    def test_multidigraph_with_keys(self):
        g = nx.MultiDiGraph()
        g.add_edge("A", "B", name="e1")
        g.add_edge("C", "A", name="e2")
        result = incident_edges(g, "A", data=True, keys=True)
        assert len(result) == 2
        # Each edge should be (u, v, key, data_dict)
        assert len(result[0]) == 4

    def test_isolated_node(self):
        g = nx.DiGraph()
        g.add_node("A")
        result = incident_edges(g, "A", data=True)
        assert result == []

    def test_self_loop(self):
        g = nx.DiGraph()
        g.add_edge("A", "A", weight=1)
        result = incident_edges(g, "A", data=False)
        # Self-loop appears in both out and in edges
        assert len(result) == 2


# ──────────────────────────────────────────────────────────────────────
# Tests for from_edges_get_nodes
# ──────────────────────────────────────────────────────────────────────

class TestFromEdgesGetNodes:

    def test_basic_edges(self):
        edges = [("A", "B", 0), ("B", "C", 0)]
        constrained_edge = ("C", "D", 0)
        result = from_edges_get_nodes(edges, "amont", constrained_edge)
        assert result == ["A", "B", "C"]

    def test_preserves_order_and_deduplicates(self):
        edges = [("A", "B", 0), ("B", "C", 0), ("A", "C", 0)]
        constrained_edge = ("C", "D", 0)
        result = from_edges_get_nodes(edges, "amont", constrained_edge)
        assert result == ["A", "B", "C"]

    def test_empty_edges_amont(self):
        constrained_edge = ("X", "Y", 0)
        result = from_edges_get_nodes([], "amont", constrained_edge)
        assert result == ["X"]

    def test_empty_edges_aval(self):
        constrained_edge = ("X", "Y", 0)
        result = from_edges_get_nodes([], "aval", constrained_edge)
        assert result == ["Y"]

    def test_empty_edges_invalid_direction_raises(self):
        constrained_edge = ("X", "Y", 0)
        with pytest.raises(ValueError):
            from_edges_get_nodes([], "invalid", constrained_edge)

    def test_single_edge(self):
        edges = [("A", "B", 0)]
        constrained_edge = ("B", "C", 0)
        result = from_edges_get_nodes(edges, "amont", constrained_edge)
        assert result == ["A", "B"]


# ──────────────────────────────────────────────────────────────────────
# Tests for find_multidigraph_edges_by_name
# ──────────────────────────────────────────────────────────────────────

class TestFindMultidigraphEdgesByName:

    def test_finds_direct_neighbor(self):
        g = _make_linear_multidigraph()
        result = find_multidigraph_edges_by_name(g, 1, {"e1"}, depth=1)
        assert "e1" in result

    def test_finds_within_depth(self):
        g = _make_linear_multidigraph()
        result = find_multidigraph_edges_by_name(g, 1, {"e2"}, depth=2)
        assert "e2" in result

    def test_does_not_find_beyond_depth(self):
        g = _make_linear_multidigraph()
        result = find_multidigraph_edges_by_name(g, 1, {"e3"}, depth=1)
        assert "e3" not in result

    def test_finds_nothing_with_nonexistent_name(self):
        g = _make_linear_multidigraph()
        result = find_multidigraph_edges_by_name(g, 1, {"no_such_edge"}, depth=5)
        assert result == []

    def test_multiple_targets(self):
        g = _make_linear_multidigraph()
        result = find_multidigraph_edges_by_name(g, 1, {"e1", "e2"}, depth=2)
        assert "e1" in result
        assert "e2" in result

    def test_depth_zero_finds_nothing(self):
        g = _make_linear_multidigraph()
        result = find_multidigraph_edges_by_name(g, 1, {"e1"}, depth=0)
        assert result == []

    def test_parallel_edges(self):
        g = nx.MultiDiGraph()
        g.add_edge("A", "B", name="line1")
        g.add_edge("A", "B", name="line2")
        result = find_multidigraph_edges_by_name(g, "A", {"line1", "line2"}, depth=1)
        assert "line1" in result
        assert "line2" in result

    def test_isolated_source(self):
        g = nx.MultiDiGraph()
        g.add_node("X")
        g.add_edge("A", "B", name="line1")
        result = find_multidigraph_edges_by_name(g, "X", {"line1"}, depth=5)
        assert result == []


# ──────────────────────────────────────────────────────────────────────
# Tests for all_simple_edge_paths_multi
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
        assert len(paths) >= 2  # A->C, B->C, A->D

    def test_same_source_and_target_skipped(self):
        g = nx.DiGraph()
        g.add_edge("A", "B")
        paths = list(all_simple_edge_paths_multi(g, ["A"], ["A"]))
        assert len(paths) == 0

    def test_no_path_exists(self):
        g = nx.DiGraph()
        g.add_edge("A", "B")
        g.add_edge("C", "D")
        paths = list(all_simple_edge_paths_multi(g, ["A"], ["D"]))
        assert len(paths) == 0

    def test_with_cutoff(self):
        g = nx.DiGraph()
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "D")
        # cutoff=1 means max 1 edge in path
        paths = list(all_simple_edge_paths_multi(g, ["A"], ["D"], cutoff=1))
        assert len(paths) == 0  # 3 edges needed, cutoff is 1

    def test_source_not_in_graph(self):
        g = nx.DiGraph()
        g.add_edge("A", "B")
        paths = list(all_simple_edge_paths_multi(g, ["X"], ["B"]))
        assert len(paths) == 0


# ──────────────────────────────────────────────────────────────────────
# Tests for shortest_path_with_promoted_edges
# ──────────────────────────────────────────────────────────────────────

class TestShortestPathWithPromotedEdges:

    def test_basic_shortest_path(self):
        g = nx.DiGraph()
        g.add_edge("A", "B", capacity=0)
        g.add_edge("B", "C", capacity=0)
        path, cost = shortest_path_with_promoted_edges(g, "A", "C", [], weight_attr="capacity")
        assert path == ["A", "B", "C"]
        assert cost == 0

    def test_no_path_returns_none(self):
        g = nx.DiGraph()
        g.add_edge("A", "B", capacity=0)
        g.add_edge("C", "D", capacity=0)
        path, cost = shortest_path_with_promoted_edges(g, "A", "D", [], weight_attr="capacity")
        assert path is None
        assert cost == float('inf')

    def test_prefers_lower_weight(self):
        g = nx.DiGraph()
        g.add_edge("A", "B", capacity=0)
        g.add_edge("B", "C", capacity=0)
        g.add_edge("A", "C", capacity=10)  # direct but heavy
        path, cost = shortest_path_with_promoted_edges(g, "A", "C", [], weight_attr="capacity")
        assert path == ["A", "B", "C"]
        assert cost == 0

    def test_with_promoted_edges(self):
        g = nx.DiGraph()
        # Two equal-weight paths: A->B->D and A->C->D
        g.add_edge("A", "B", capacity=0)
        g.add_edge("B", "D", capacity=0)
        g.add_edge("A", "C", capacity=0)
        g.add_edge("C", "D", capacity=0)
        # Promote A->C edge
        path, cost = shortest_path_with_promoted_edges(
            g, "A", "D", promoted_edges=[("A", "C")], weight_attr="capacity")
        assert path == ["A", "C", "D"]

    def test_single_node_path(self):
        g = nx.DiGraph()
        g.add_node("A")
        path, cost = shortest_path_with_promoted_edges(g, "A", "A", [], weight_attr="capacity")
        assert path == ["A"]


# ──────────────────────────────────────────────────────────────────────
# Tests for add_double_edges_null_redispatch
# ──────────────────────────────────────────────────────────────────────

class TestAddDoubleEdgesNullRedispatch:

    def test_doubles_gray_zero_capacity(self):
        g = nx.MultiDiGraph()
        g.add_edge("A", "B", color="gray", capacity=0., name="line_AB")
        edges_to_double, edges_added = add_double_edges_null_redispatch(g)

        assert "line_AB" in edges_to_double
        assert "line_AB" in edges_added
        # Reverse edge B->A should exist now
        assert g.has_edge("B", "A")

    def test_does_not_double_nonzero_capacity(self):
        g = nx.MultiDiGraph()
        g.add_edge("A", "B", color="gray", capacity=5., name="line_AB")
        edges_to_double, edges_added = add_double_edges_null_redispatch(g)

        assert len(edges_to_double) == 0
        assert not g.has_edge("B", "A")

    def test_does_not_double_non_gray(self):
        g = nx.MultiDiGraph()
        g.add_edge("A", "B", color="blue", capacity=0., name="line_AB")
        edges_to_double, edges_added = add_double_edges_null_redispatch(g)

        assert len(edges_to_double) == 0

    def test_with_different_color_init(self):
        g = nx.MultiDiGraph()
        g.add_edge("A", "B", color="blue", capacity=0., name="line_AB")
        edges_to_double, edges_added = add_double_edges_null_redispatch(g, color_init="blue")

        assert "line_AB" in edges_to_double
        assert g.has_edge("B", "A")

    def test_preserves_edge_attributes(self):
        g = nx.MultiDiGraph()
        g.add_edge("A", "B", color="gray", capacity=0., name="line_AB", style="dashed")
        edges_to_double, edges_added = add_double_edges_null_redispatch(g)

        reverse_edge = edges_added["line_AB"]
        attrs = g.edges[reverse_edge]
        assert attrs["name"] == "line_AB"
        assert attrs["color"] == "gray"
        assert attrs["capacity"] == 0.
        assert attrs["style"] == "dashed"

    def test_multiple_qualifying_edges(self):
        g = nx.MultiDiGraph()
        g.add_edge("A", "B", color="gray", capacity=0., name="l1")
        g.add_edge("C", "D", color="gray", capacity=0., name="l2")
        g.add_edge("E", "F", color="gray", capacity=5., name="l3")  # should NOT be doubled
        edges_to_double, edges_added = add_double_edges_null_redispatch(g)

        assert len(edges_to_double) == 2
        assert "l1" in edges_to_double
        assert "l2" in edges_to_double
        assert "l3" not in edges_to_double

    def test_only_no_dir_flag(self):
        g = nx.MultiDiGraph()
        g.add_edge("A", "B", color="gray", capacity=0., name="l1", dir="none")
        g.add_edge("C", "D", color="gray", capacity=0., name="l2")  # no dir attr
        edges_to_double, edges_added = add_double_edges_null_redispatch(g, only_no_dir=True)

        assert "l1" in edges_to_double
        assert "l2" not in edges_to_double

    def test_empty_graph(self):
        g = nx.MultiDiGraph()
        edges_to_double, edges_added = add_double_edges_null_redispatch(g)
        assert len(edges_to_double) == 0
        assert len(edges_added) == 0


# ──────────────────────────────────────────────────────────────────────
# Tests for remove_unused_added_double_edge
# ──────────────────────────────────────────────────────────────────────

class TestRemoveUnusedAddedDoubleEdge:

    def test_removes_unused_double_edges(self):
        g = nx.MultiDiGraph()
        g.add_edge("A", "B", color="gray", capacity=0., name="l1")
        g.add_edge("C", "D", color="gray", capacity=0., name="l2")
        edges_to_double, edges_added = add_double_edges_null_redispatch(g)

        # Mark none as kept
        edges_to_keep = set()
        g = remove_unused_added_double_edge(g, edges_to_keep, edges_to_double, edges_added)

        # Added double edges should be removed
        assert not g.has_edge("B", "A")
        assert not g.has_edge("D", "C")
        # Original edges should still be present
        assert g.has_edge("A", "B")
        assert g.has_edge("C", "D")

    def test_keeps_recolored_double_edges(self):
        g = nx.MultiDiGraph()
        g.add_edge("A", "B", color="gray", capacity=0., name="l1")
        edges_to_double, edges_added = add_double_edges_null_redispatch(g)

        # Recolor the added double edge (simulate it being kept)
        added_edge = edges_added["l1"]
        g.edges[added_edge]["color"] = "blue"

        edges_to_keep = {added_edge}
        g = remove_unused_added_double_edge(g, edges_to_keep, edges_to_double, edges_added)

        # The recolored double edge should remain
        assert g.has_edge("B", "A")
        # The original gray edge should be removed (replaced by the kept double)
        assert not g.has_edge("A", "B")

    def test_empty_edges_to_keep(self):
        g = nx.MultiDiGraph()
        g.add_edge("A", "B", color="gray", capacity=0., name="l1")
        edges_to_double, edges_added = add_double_edges_null_redispatch(g)
        initial_edges_count = g.number_of_edges()

        g = remove_unused_added_double_edge(g, set(), edges_to_double, edges_added)

        # Only original edges should remain
        assert g.number_of_edges() == initial_edges_count - len(edges_to_double)


# ──────────────────────────────────────────────────────────────────────
# Tests for ConstrainedPath
# ──────────────────────────────────────────────────────────────────────

class TestConstrainedPath:

    def test_e_amont_returns_amont_edges(self):
        amont = [("A", "B", 0), ("B", "C", 0)]
        cp = ConstrainedPath(amont, ("C", "D", 0), [("D", "E", 0)])
        assert cp.e_amont() == amont

    def test_e_aval_returns_aval_edges(self):
        aval = [("D", "E", 0), ("E", "F", 0)]
        cp = ConstrainedPath([("A", "B", 0)], ("C", "D", 0), aval)
        assert cp.e_aval() == aval

    def test_n_amont_with_edges(self):
        amont = [("A", "B", 0), ("B", "C", 0)]
        cp = ConstrainedPath(amont, ("C", "D", 0), [])
        nodes = cp.n_amont()
        assert nodes == ["A", "B", "C"]

    def test_n_aval_with_edges(self):
        aval = [("D", "E", 0), ("E", "F", 0)]
        cp = ConstrainedPath([], ("C", "D", 0), aval)
        nodes = cp.n_aval()
        assert nodes == ["D", "E", "F"]

    def test_n_amont_empty_returns_constrained_source(self):
        cp = ConstrainedPath([], ("X", "Y", 0), [])
        assert cp.n_amont() == ["X"]

    def test_n_aval_empty_returns_constrained_target(self):
        cp = ConstrainedPath([], ("X", "Y", 0), [])
        assert cp.n_aval() == ["Y"]

    def test_full_n_constrained_path(self):
        amont = [("A", "B", 0)]
        constrained = ("B", "C", 0)
        aval = [("C", "D", 0)]
        cp = ConstrainedPath(amont, constrained, aval)
        full = cp.full_n_constrained_path()
        assert full == ["A", "B", "C", "D"]

    def test_full_n_constrained_path_deduplicates(self):
        amont = [("A", "B", 0), ("B", "C", 0)]
        constrained = ("C", "D", 0)
        aval = [("D", "C", 0)]  # C appears again
        cp = ConstrainedPath(amont, constrained, aval)
        full = cp.full_n_constrained_path()
        # C should appear only once
        assert full.count("C") == 1

    def test_repr(self):
        cp = ConstrainedPath([("A", "B", 0)], ("B", "C", 0), [("C", "D", 0)])
        repr_str = repr(cp)
        assert "ConstrainedPath" in repr_str
        assert "A" in repr_str


# ──────────────────────────────────────────────────────────────────────
# Tests for detect_edges_to_keep
# ──────────────────────────────────────────────────────────────────────

class _FakeOverFlowGraph:
    """Minimal mock to call detect_edges_to_keep which is a method of OverFlowGraph."""
    def __init__(self):
        self.g = nx.MultiDiGraph()

    detect_edges_to_keep = OverFlowGraph.detect_edges_to_keep


class TestDetectEdgesToKeep:

    def test_no_edges_of_interest_returns_empty(self):
        """When edges_of_interest doesn't overlap with g_c, return empty sets."""
        obj = _FakeOverFlowGraph()
        g_c = nx.MultiDiGraph()
        g_c.add_edge("A", "B", color="gray", capacity=0., name="l1")

        # edges_of_interest refers to edges not in g_c
        fake_interest = {("X", "Y", 0)}
        rec, non_rec = obj.detect_edges_to_keep(g_c, ["A"], ["B"], fake_interest)
        assert rec == set()
        assert non_rec == set()

    def test_no_source_nodes_in_gc_returns_empty(self):
        obj = _FakeOverFlowGraph()
        g_c = nx.MultiDiGraph()
        g_c.add_edge("A", "B", color="gray", capacity=0., name="l1")

        edge_key = list(g_c.edges(keys=True))[0]
        rec, non_rec = obj.detect_edges_to_keep(
            g_c, ["X"], ["B"], {edge_key})
        assert rec == set()
        assert non_rec == set()

    def test_no_target_nodes_in_gc_returns_empty(self):
        obj = _FakeOverFlowGraph()
        g_c = nx.MultiDiGraph()
        g_c.add_edge("A", "B", color="gray", capacity=0., name="l1")

        edge_key = list(g_c.edges(keys=True))[0]
        rec, non_rec = obj.detect_edges_to_keep(
            g_c, ["A"], ["Y"], {edge_key})
        assert rec == set()
        assert non_rec == set()

    def test_finds_reconnectable_edge_on_path(self):
        """Edge of interest on a simple path between source and target."""
        obj = _FakeOverFlowGraph()
        g_c = nx.MultiDiGraph()
        g_c.add_edge("S", "M", color="gray", capacity=0., name="line_SM")
        g_c.add_edge("M", "T", color="gray", capacity=0., name="line_disconnect")

        edge_disconnect = ("M", "T", 0)
        edges_of_interest = {edge_disconnect}

        rec, non_rec = obj.detect_edges_to_keep(
            g_c, {"S"}, {"T"}, edges_of_interest)

        assert len(rec) > 0
        assert edge_disconnect in rec

    def test_non_reconnectable_edge_classified_correctly(self):
        """Non-reconnectable edges on path should go to non_reconnectable set."""
        obj = _FakeOverFlowGraph()
        g_c = nx.MultiDiGraph()
        g_c.add_edge("S", "M", color="gray", capacity=0., name="line_SM")
        g_c.add_edge("M", "T", color="gray", capacity=0., name="line_nr")

        edge_nr = ("M", "T", 0)
        edges_of_interest = {edge_nr}
        non_reconnectable_edges = [edge_nr]

        rec, non_rec = obj.detect_edges_to_keep(
            g_c, {"S"}, {"T"}, edges_of_interest,
            non_reconnectable_edges=non_reconnectable_edges)

        assert edge_nr in non_rec
        assert edge_nr not in rec

    def test_source_equals_target_set(self):
        """When source_nodes == target_nodes, finds paths between different nodes in the set."""
        obj = _FakeOverFlowGraph()
        g_c = nx.MultiDiGraph()
        g_c.add_edge("A", "B", color="gray", capacity=0., name="line_disc")
        g_c.add_edge("B", "C", color="gray", capacity=0., name="line_BC")

        edge_disc = ("A", "B", 0)
        edges_of_interest = {edge_disc}
        node_set = {"A", "C"}

        rec, non_rec = obj.detect_edges_to_keep(
            g_c, node_set, node_set, edges_of_interest)

        assert edge_disc in rec

    def test_max_path_length_filter(self):
        """Paths longer than max_null_flow_path_length are excluded."""
        obj = _FakeOverFlowGraph()
        g_c = nx.MultiDiGraph()
        # Build chain: 1->2->3->4->5->6->7->8->9
        for i in range(1, 9):
            g_c.add_edge(str(i), str(i+1), color="gray", capacity=0., name=f"l{i}")

        # Edge of interest at the end
        edge_interest = ("8", "9", 0)
        edges_of_interest = {edge_interest}

        # max_null_flow_path_length=3 should exclude the 9-node path
        rec, non_rec = obj.detect_edges_to_keep(
            g_c, {"1"}, {"9"}, edges_of_interest,
            max_null_flow_path_length=3)

        assert len(rec) == 0

    def test_no_incident_interest_returns_empty(self):
        """When no source/target has incident edges of interest, skip all pairs."""
        obj = _FakeOverFlowGraph()
        g_c = nx.MultiDiGraph()
        g_c.add_edge("A", "B", color="gray", capacity=0., name="l1")
        g_c.add_edge("B", "C", color="gray", capacity=0., name="l2")
        g_c.add_edge("C", "D", color="gray", capacity=0., name="l_disc")

        # Edge of interest is far from sources/targets
        edge_interest = ("C", "D", 0)
        rec, non_rec = obj.detect_edges_to_keep(
            g_c, {"A"}, {"B"}, {edge_interest})

        # edge_interest is not incident to A or B, so should be empty
        assert len(rec) == 0

    def test_negative_capacities_flipped(self):
        """Negative capacities should be flipped to positive for Dijkstra."""
        obj = _FakeOverFlowGraph()
        g_c = nx.MultiDiGraph()
        g_c.add_edge("S", "T", color="gray", capacity=-3., name="line_disc")

        edge_disc = ("S", "T", 0)
        rec, non_rec = obj.detect_edges_to_keep(
            g_c, {"S"}, {"T"}, {edge_disc})

        # After the call, capacity should be flipped to positive
        assert g_c.edges[edge_disc]["capacity"] == 3.

    def test_shortest_path_preferred(self):
        """When multiple paths exist, shorter ones are preferred for edge selection."""
        obj = _FakeOverFlowGraph()
        g_c = nx.MultiDiGraph()
        # Short path: S -> M -> T (2 edges)
        g_c.add_edge("S", "M", color="gray", capacity=0., name="l_short1")
        g_c.add_edge("M", "T", color="gray", capacity=0., name="l_disc_short")
        # Long path: S -> X -> Y -> T (3 edges)
        g_c.add_edge("S", "X", color="gray", capacity=0., name="l_long1")
        g_c.add_edge("X", "Y", color="gray", capacity=0., name="l_disc_long")
        g_c.add_edge("Y", "T", color="gray", capacity=0., name="l_long3")

        edge_short = ("M", "T", 0)
        edge_long = ("X", "Y", 1)

        edges_of_interest = {edge_short, edge_long}

        rec, non_rec = obj.detect_edges_to_keep(
            g_c, {"S"}, {"T"}, edges_of_interest)

        # Both disconnect edges should be found (different paths)
        assert edge_short in rec


# ──────────────────────────────────────────────────────────────────────
# Tests for shortest_path_min_weight_then_hops
# ──────────────────────────────────────────────────────────────────────

class TestShortestPathMinWeightThenHops:

    def test_basic_path_through_mandatory_edge(self):
        g = nx.DiGraph()
        g.add_edge("A", "B", weight=1)
        g.add_edge("B", "C", weight=1)
        g.add_edge("C", "D", weight=1)
        path, cost = shortest_path_min_weight_then_hops(
            g, "A", "D", mandatory_edge=("B", "C"), weight_attr="weight")
        assert "B" in path
        assert "C" in path
        assert cost == 3

    def test_no_path_returns_none(self):
        g = nx.DiGraph()
        g.add_edge("A", "B", weight=1)
        g.add_edge("C", "D", weight=1)
        path, cost = shortest_path_min_weight_then_hops(
            g, "A", "D", mandatory_edge=("A", "B"), weight_attr="weight")
        assert path is None
        assert cost == float('inf')

    def test_multigraph_with_key(self):
        g = nx.MultiDiGraph()
        k0 = g.add_edge("A", "B", weight=5)
        k1 = g.add_edge("A", "B", weight=1)
        g.add_edge("B", "C", weight=1)
        path, cost = shortest_path_min_weight_then_hops(
            g, "A", "C", mandatory_edge=("A", "B", k1), weight_attr="weight")
        assert path is not None
        assert cost == 2  # weight of key k1 (1) + B->C (1)


# ──────────────────────────────────────────────────────────────────────
# Integration-style tests for add_relevant_null_flow_lines_all_paths
# with hand-crafted graphs (no data file dependencies)
# ──────────────────────────────────────────────────────────────────────

class TestAddRelevantNullFlowLinesSetupIntegration:

    def test_setup_null_flow_styles_sets_dotted_for_non_reconnectable(self):
        """_setup_null_flow_styles should set dotted style on non-reconnectable edges."""
        obj = _FakeOverFlowGraph()
        obj.g = nx.MultiDiGraph()
        obj.g.add_edge("A", "B", color="gray", capacity=0., name="nr_line", style="solid")
        obj.g.add_edge("C", "D", color="gray", capacity=0., name="re_line", style="solid")

        obj._setup_null_flow_styles = OverFlowGraph._setup_null_flow_styles.__get__(obj)
        result = obj._setup_null_flow_styles(["re_line"], ["nr_line"])

        # nr_line should be dotted
        edge_nr = ("A", "B", 0)
        assert obj.g.edges[edge_nr]["style"] == "dotted"
        assert obj.g.edges[edge_nr].get("dir") == "none"

        # re_line should be dashed
        edge_re = ("C", "D", 0)
        assert obj.g.edges[edge_re]["style"] == "dashed"

    def test_setup_returns_combined_lines(self):
        """_setup_null_flow_styles should return combined list."""
        obj = _FakeOverFlowGraph()
        obj.g = nx.MultiDiGraph()
        obj.g.add_edge("A", "B", color="gray", capacity=0., name="line1")
        obj.g.add_edge("C", "D", color="gray", capacity=0., name="line2")

        obj._setup_null_flow_styles = OverFlowGraph._setup_null_flow_styles.__get__(obj)
        result = obj._setup_null_flow_styles(["line1"], ["line2"])

        assert "line1" in result
        assert "line2" in result
