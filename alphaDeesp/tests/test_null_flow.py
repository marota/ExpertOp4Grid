"""Unit tests for :mod:`alphaDeesp.core.graphs.null_flow` and for the
null-flow-line helpers on :class:`OverFlowGraph`."""

import networkx as nx

from alphaDeesp.core.graphsAndPaths import (
    OverFlowGraph,
    add_double_edges_null_redispatch,
    remove_unused_added_double_edge,
)
from alphaDeesp.tests.graphs_test_helpers import (
    FakeOverFlowGraph,
    NullFlowHelperHost,
)


# ──────────────────────────────────────────────────────────────────────
# add_double_edges_null_redispatch
# ──────────────────────────────────────────────────────────────────────

class TestAddDoubleEdgesNullRedispatch:

    def test_doubles_gray_zero_capacity(self):
        g = nx.MultiDiGraph()
        g.add_edge("A", "B", color="gray", capacity=0., name="line_AB")
        edges_to_double, edges_added = add_double_edges_null_redispatch(g)

        assert "line_AB" in edges_to_double
        assert "line_AB" in edges_added
        assert g.has_edge("B", "A")

    def test_does_not_double_nonzero_capacity(self):
        g = nx.MultiDiGraph()
        g.add_edge("A", "B", color="gray", capacity=5., name="line_AB")
        edges_to_double, _ = add_double_edges_null_redispatch(g)
        assert len(edges_to_double) == 0
        assert not g.has_edge("B", "A")

    def test_does_not_double_non_gray(self):
        g = nx.MultiDiGraph()
        g.add_edge("A", "B", color="blue", capacity=0., name="line_AB")
        edges_to_double, _ = add_double_edges_null_redispatch(g)
        assert len(edges_to_double) == 0

    def test_with_different_color_init(self):
        g = nx.MultiDiGraph()
        g.add_edge("A", "B", color="blue", capacity=0., name="line_AB")
        edges_to_double, _ = add_double_edges_null_redispatch(g, color_init="blue")
        assert "line_AB" in edges_to_double
        assert g.has_edge("B", "A")

    def test_preserves_edge_attributes(self):
        g = nx.MultiDiGraph()
        g.add_edge("A", "B", color="gray", capacity=0., name="line_AB", style="dashed")
        _, edges_added = add_double_edges_null_redispatch(g)

        attrs = g.edges[edges_added["line_AB"]]
        assert attrs["name"] == "line_AB"
        assert attrs["color"] == "gray"
        assert attrs["capacity"] == 0.
        assert attrs["style"] == "dashed"

    def test_multiple_qualifying_edges(self):
        g = nx.MultiDiGraph()
        g.add_edge("A", "B", color="gray", capacity=0., name="l1")
        g.add_edge("C", "D", color="gray", capacity=0., name="l2")
        g.add_edge("E", "F", color="gray", capacity=5., name="l3")  # not doubled
        edges_to_double, _ = add_double_edges_null_redispatch(g)
        assert len(edges_to_double) == 2
        assert "l1" in edges_to_double and "l2" in edges_to_double
        assert "l3" not in edges_to_double

    def test_only_no_dir_flag(self):
        g = nx.MultiDiGraph()
        g.add_edge("A", "B", color="gray", capacity=0., name="l1", dir="none")
        g.add_edge("C", "D", color="gray", capacity=0., name="l2")  # no dir
        edges_to_double, _ = add_double_edges_null_redispatch(g, only_no_dir=True)
        assert "l1" in edges_to_double
        assert "l2" not in edges_to_double

    def test_empty_graph(self):
        edges_to_double, edges_added = add_double_edges_null_redispatch(nx.MultiDiGraph())
        assert len(edges_to_double) == 0
        assert len(edges_added) == 0


# ──────────────────────────────────────────────────────────────────────
# remove_unused_added_double_edge
# ──────────────────────────────────────────────────────────────────────

class TestRemoveUnusedAddedDoubleEdge:

    def test_removes_unused_double_edges(self):
        g = nx.MultiDiGraph()
        g.add_edge("A", "B", color="gray", capacity=0., name="l1")
        g.add_edge("C", "D", color="gray", capacity=0., name="l2")
        edges_to_double, edges_added = add_double_edges_null_redispatch(g)

        g = remove_unused_added_double_edge(g, set(), edges_to_double, edges_added)

        assert not g.has_edge("B", "A")
        assert not g.has_edge("D", "C")
        assert g.has_edge("A", "B")
        assert g.has_edge("C", "D")

    def test_keeps_recolored_double_edges(self):
        g = nx.MultiDiGraph()
        g.add_edge("A", "B", color="gray", capacity=0., name="l1")
        edges_to_double, edges_added = add_double_edges_null_redispatch(g)

        added_edge = edges_added["l1"]
        g.edges[added_edge]["color"] = "blue"

        g = remove_unused_added_double_edge(g, {added_edge}, edges_to_double, edges_added)

        assert g.has_edge("B", "A")  # recoloured — kept
        assert not g.has_edge("A", "B")  # original gray replaced

    def test_empty_edges_to_keep(self):
        g = nx.MultiDiGraph()
        g.add_edge("A", "B", color="gray", capacity=0., name="l1")
        edges_to_double, edges_added = add_double_edges_null_redispatch(g)
        initial = g.number_of_edges()

        g = remove_unused_added_double_edge(g, set(), edges_to_double, edges_added)
        assert g.number_of_edges() == initial - len(edges_to_double)


# ──────────────────────────────────────────────────────────────────────
# _setup_null_flow_styles
# ──────────────────────────────────────────────────────────────────────

class TestSetupNullFlowStyles:

    def test_sets_dotted_for_non_reconnectable(self):
        obj = FakeOverFlowGraph()
        obj.g.add_edge("A", "B", color="gray", capacity=0., name="nr_line", style="solid")
        obj.g.add_edge("C", "D", color="gray", capacity=0., name="re_line", style="solid")

        obj._setup_null_flow_styles = OverFlowGraph._setup_null_flow_styles.__get__(obj)
        obj._setup_null_flow_styles(["re_line"], ["nr_line"])

        edge_nr = ("A", "B", 0)
        assert obj.g.edges[edge_nr]["style"] == "dotted"
        assert obj.g.edges[edge_nr].get("dir") == "none"

        edge_re = ("C", "D", 0)
        assert obj.g.edges[edge_re]["style"] == "dashed"

    def test_returns_combined_lines(self):
        obj = FakeOverFlowGraph()
        obj.g.add_edge("A", "B", color="gray", capacity=0., name="line1")
        obj.g.add_edge("C", "D", color="gray", capacity=0., name="line2")

        obj._setup_null_flow_styles = OverFlowGraph._setup_null_flow_styles.__get__(obj)
        result = obj._setup_null_flow_styles(["line1"], ["line2"])

        assert "line1" in result
        assert "line2" in result


# ──────────────────────────────────────────────────────────────────────
# Null-flow helper methods used by add_relevant_null_flow_lines
# ──────────────────────────────────────────────────────────────────────

class TestNullFlowHelpers:

    def test_prepare_edge_sets_keeps_connex_lines(self):
        obj = NullFlowHelperHost()
        obj.g = nx.MultiDiGraph()
        obj.g.add_edge("A", "B", color="coral", capacity=1., name="coloured_line")
        obj.g.add_edge("B", "C", color="gray", capacity=0., name="connex_line")
        obj.g.add_edge("X", "Y", color="gray", capacity=0., name="far_line")

        sets = obj._prepare_null_flow_edge_sets(["connex_line", "far_line"], [])
        considered_names = {obj.g.edges[e]["name"]
                            for e in sets["edges_non_connected_lines_to_consider"]}
        assert "connex_line" in considered_names
        assert "far_line" not in considered_names

    def test_build_gray_components_drops_coloured_edges(self):
        obj = NullFlowHelperHost()
        obj.g = nx.MultiDiGraph()
        obj.g.add_edge("A", "B", color="coral", capacity=1., name="red")
        obj.g.add_edge("C", "D", color="gray", capacity=0., name="gray_line1")
        obj.g.add_edge("D", "E", color="gray", capacity=0., name="gray_line2")

        components = obj._build_gray_components()
        assert len(components) == 1
        assert {d["name"] for _, _, d in components[0].edges(data=True)} == {
            "gray_line1", "gray_line2"}

    def test_structural_info_extracts_red_and_cpath_nodes(self):
        class _FakeCP:
            def n_amont(self):
                return ["A", "B"]

            def n_aval(self):
                return ["C"]

        class _FakeRedLoops:
            class Path:
                shape = (0,)  # empty

        class _FakeStructured:
            constrained_path = _FakeCP()
            red_loops = _FakeRedLoops()

        info = OverFlowGraph._structural_info_for_null_flow(_FakeStructured())
        assert info["node_red_paths"] == []
        assert info["node_amont_constrained_path"] == ["A", "B"]
        assert info["node_aval_constrained_path"] == ["C"]

    def test_apply_recoloring_paints_blue_for_blue_only(self):
        obj = NullFlowHelperHost()
        obj.g = nx.MultiDiGraph()
        obj.g.add_edge("A", "B", color="gray", capacity=0., name="l1")
        edge = ("A", "B", 0)
        obj._apply_null_flow_recoloring(
            target_path="blue_only",
            edges_to_keep={edge},
            edges_non_reconnectable=set(),
            edges_to_double={},
            edges_double_added={},
        )
        assert obj.g.edges[edge]["color"] == "blue"

    def test_apply_recoloring_blue_to_red_respects_sign(self):
        obj = NullFlowHelperHost()
        obj.g = nx.MultiDiGraph()
        obj.g.add_edge("A", "B", color="gray", capacity=-1., name="neg")
        obj.g.add_edge("C", "D", color="gray", capacity=1., name="pos")
        edge_neg = ("A", "B", 0)
        edge_pos = ("C", "D", 0)
        obj._apply_null_flow_recoloring(
            target_path="blue_to_red",
            edges_to_keep={edge_neg, edge_pos},
            edges_non_reconnectable=set(),
            edges_to_double={},
            edges_double_added={},
        )
        assert obj.g.edges[edge_neg]["color"] == "blue"
        assert obj.g.edges[edge_pos]["color"] == "coral"
