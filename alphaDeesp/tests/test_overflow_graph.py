"""Unit tests for :class:`OverFlowGraph` behaviour: component filtering,
node collapsing, penwidth scaling, and the ``detect_edges_to_keep``
path search + its four internal helpers."""

import pandas as pd
import networkx as nx

from alphaDeesp.core.graphsAndPaths import OverFlowGraph
from alphaDeesp.tests.graphs_test_helpers import (
    DetectEdgesHelperHost,
    FakeOverFlowGraph,
    make_ofg_with_graph,
)


# ──────────────────────────────────────────────────────────────────────
# keep_overloads_components
# ──────────────────────────────────────────────────────────────────────

def _edge_colors(g):
    return {(u, v, k): d["color"] for u, v, k, d in g.edges(keys=True, data=True)}


class TestKeepOverloadsComponents:
    """Components without any ``black`` edge should be greyed out; components
    containing at least one black edge are left untouched."""

    def test_component_with_overload_is_kept(self):
        g = nx.MultiDiGraph()
        g.add_edge(0, 1, color="black", capacity=-5.)
        g.add_edge(1, 2, color="blue", capacity=-3.)
        ofg = make_ofg_with_graph(g)

        ofg.keep_overloads_components()

        colors = _edge_colors(ofg.g)
        assert colors[(0, 1, 0)] == "black"
        assert colors[(1, 2, 0)] == "blue"

    def test_component_without_overload_becomes_gray(self):
        g = nx.MultiDiGraph()
        g.add_edge(0, 1, color="blue", capacity=-5.)
        g.add_edge(1, 2, color="coral", capacity=3.)
        ofg = make_ofg_with_graph(g)

        ofg.keep_overloads_components()

        colors = _edge_colors(ofg.g)
        assert colors[(0, 1, 0)] == "gray"
        assert colors[(1, 2, 0)] == "gray"

    def test_multiple_components_mixed(self):
        """Two components; only the one with a black edge should survive."""
        g = nx.MultiDiGraph()
        # component 1 (0-1-2) has a black edge
        g.add_edge(0, 1, color="black", capacity=-5.)
        g.add_edge(1, 2, color="blue", capacity=-3.)
        # component 2 (10-11) does not
        g.add_edge(10, 11, color="coral", capacity=4.)
        ofg = make_ofg_with_graph(g)

        ofg.keep_overloads_components()
        colors = _edge_colors(ofg.g)
        assert colors[(0, 1, 0)] == "black"
        assert colors[(1, 2, 0)] == "blue"
        assert colors[(10, 11, 0)] == "gray"

    def test_already_gray_edges_stay_gray(self):
        g = nx.MultiDiGraph()
        g.add_edge(0, 1, color="black", capacity=-5.)
        g.add_edge(1, 2, color="gray", capacity=0.)
        ofg = make_ofg_with_graph(g)

        ofg.keep_overloads_components()
        assert _edge_colors(ofg.g)[(1, 2, 0)] == "gray"

    def test_empty_graph(self):
        ofg = make_ofg_with_graph(nx.MultiDiGraph())
        ofg.keep_overloads_components()
        assert ofg.g.number_of_nodes() == 0

    def test_all_gray_graph(self):
        g = nx.MultiDiGraph()
        g.add_edge(0, 1, color="gray", capacity=0.)
        g.add_edge(1, 2, color="gray", capacity=0.)
        ofg = make_ofg_with_graph(g)

        ofg.keep_overloads_components()
        assert set(_edge_colors(ofg.g).values()) == {"gray"}

    def test_single_black_edge_component(self):
        g = nx.MultiDiGraph()
        g.add_edge(0, 1, color="black", capacity=-5.)
        ofg = make_ofg_with_graph(g)

        ofg.keep_overloads_components()
        assert _edge_colors(ofg.g)[(0, 1, 0)] == "black"

    def test_single_blue_edge_component_becomes_gray(self):
        g = nx.MultiDiGraph()
        g.add_edge(0, 1, color="blue", capacity=-5.)
        ofg = make_ofg_with_graph(g)

        ofg.keep_overloads_components()
        assert _edge_colors(ofg.g)[(0, 1, 0)] == "gray"

    def test_parallel_edges_component_with_overload(self):
        g = nx.MultiDiGraph()
        g.add_edge(0, 1, color="black", capacity=-5.)
        g.add_edge(0, 1, color="blue", capacity=-3.)
        g.add_edge(1, 2, color="coral", capacity=4.)
        ofg = make_ofg_with_graph(g)

        ofg.keep_overloads_components()
        colors = _edge_colors(ofg.g)
        assert colors[(0, 1, 0)] == "black"
        assert colors[(0, 1, 1)] == "blue"
        assert colors[(1, 2, 0)] == "coral"

    def test_parallel_edges_component_without_overload(self):
        g = nx.MultiDiGraph()
        g.add_edge(0, 1, color="blue", capacity=-5.)
        g.add_edge(0, 1, color="coral", capacity=3.)
        ofg = make_ofg_with_graph(g)

        ofg.keep_overloads_components()
        colors = _edge_colors(ofg.g)
        assert colors[(0, 1, 0)] == "gray"
        assert colors[(0, 1, 1)] == "gray"

    def test_gray_edge_between_components_does_not_bridge(self):
        """Gray edges are removed before connectivity detection so the two
        sides are treated as separate components."""
        g = nx.MultiDiGraph()
        g.add_edge(0, 1, color="black", capacity=-5.)
        g.add_edge(1, 2, color="gray", capacity=0.)  # would bridge if kept
        g.add_edge(2, 3, color="coral", capacity=3.)
        ofg = make_ofg_with_graph(g)

        ofg.keep_overloads_components()
        colors = _edge_colors(ofg.g)
        assert colors[(0, 1, 0)] == "black"
        assert colors[(2, 3, 0)] == "gray"

    def test_three_components_only_middle_has_overload(self):
        g = nx.MultiDiGraph()
        g.add_edge(0, 1, color="blue", capacity=-5.)  # c1
        g.add_edge(10, 11, color="black", capacity=-5.)  # c2
        g.add_edge(10, 12, color="coral", capacity=3.)   # c2
        g.add_edge(20, 21, color="coral", capacity=3.)   # c3
        ofg = make_ofg_with_graph(g)

        ofg.keep_overloads_components()
        colors = _edge_colors(ofg.g)
        assert colors[(0, 1, 0)] == "gray"
        assert colors[(10, 11, 0)] == "black"
        assert colors[(10, 12, 0)] == "coral"
        assert colors[(20, 21, 0)] == "gray"

    def test_idempotent(self):
        """Running twice should produce the same colouring."""
        g = nx.MultiDiGraph()
        g.add_edge(0, 1, color="black", capacity=-5.)
        g.add_edge(10, 11, color="blue", capacity=-5.)
        ofg = make_ofg_with_graph(g)

        ofg.keep_overloads_components()
        colors_after_first = _edge_colors(ofg.g)
        ofg.keep_overloads_components()
        colors_after_second = _edge_colors(ofg.g)
        assert colors_after_first == colors_after_second


# ──────────────────────────────────────────────────────────────────────
# collapse_red_loops
# ──────────────────────────────────────────────────────────────────────

class TestCollapseRedLoops:

    def test_node_in_coral_loop_is_collapsed(self):
        g = nx.MultiDiGraph()
        g.add_node("N1", shape="oval")
        g.add_edge("N1", "N2", color="coral")
        ofg = make_ofg_with_graph(g)
        ofg.collapse_red_loops()
        assert ofg.g.nodes["N1"]["shape"] == "point"

    def test_hub_is_not_collapsed(self):
        g = nx.MultiDiGraph()
        g.add_node("N1", shape="diamond")
        g.add_edge("N1", "N2", color="coral")
        ofg = make_ofg_with_graph(g)
        ofg.collapse_red_loops()
        assert ofg.g.nodes["N1"]["shape"] == "diamond"

    def test_node_with_peripheries_not_collapsed(self):
        g = nx.MultiDiGraph()
        g.add_node("N1", shape="oval", peripheries=2)
        g.add_edge("N1", "N2", color="coral")
        ofg = make_ofg_with_graph(g)
        ofg.collapse_red_loops()
        assert ofg.g.nodes["N1"]["shape"] == "oval"

    def test_node_with_non_coral_edge_not_collapsed(self):
        g = nx.MultiDiGraph()
        g.add_node("N1", shape="oval")
        g.add_edge("N1", "N2", color="coral")
        g.add_edge("N1", "N3", color="blue")
        ofg = make_ofg_with_graph(g)
        ofg.collapse_red_loops()
        assert ofg.g.nodes["N1"]["shape"] == "oval"

    def test_node_with_dashed_edge_not_collapsed(self):
        g = nx.MultiDiGraph()
        g.add_node("N1", shape="oval")
        g.add_edge("N1", "N2", color="coral", style="dashed")
        ofg = make_ofg_with_graph(g)
        ofg.collapse_red_loops()
        assert ofg.g.nodes["N1"]["shape"] == "oval"


# ──────────────────────────────────────────────────────────────────────
# Penwidth scaling on edge construction
# ──────────────────────────────────────────────────────────────────────

def _basic_topo(n_nodes):
    return {
        "nodes": {
            "are_prods": [False] * n_nodes,
            "are_loads": [False] * n_nodes,
            "prods_values": [0.0] * n_nodes,
            "loads_values": [0.0] * n_nodes,
        },
        "edges": {"idx_or": [0], "idx_ex": [1], "init_flows": [0.0]},
    }


class TestEdgeColor:
    """``OverFlowGraph._edge_color`` maps a (row-index, flow, gray flag,
    cut-lines) tuple to the final rendered colour."""

    def test_cut_line_is_black(self):
        assert OverFlowGraph._edge_color(2, 5.0, False, [2]) == "black"
        assert OverFlowGraph._edge_color(0, -5.0, False, [0]) == "black"

    def test_cut_line_beats_gray_flag(self):
        """Even if the row is flagged as an insignificant (gray) edge,
        a cut line still renders black."""
        assert OverFlowGraph._edge_color(1, 0.1, True, [1]) == "black"

    def test_gray_edge_when_flagged(self):
        assert OverFlowGraph._edge_color(0, 3.0, True, []) == "gray"

    def test_negative_flow_is_blue(self):
        assert OverFlowGraph._edge_color(0, -1.0, False, []) == "blue"

    def test_positive_flow_is_coral(self):
        assert OverFlowGraph._edge_color(0, 1.0, False, []) == "coral"

    def test_zero_flow_falls_through_to_coral(self):
        """Zero is not <0, so falls into the coral branch when not gray."""
        assert OverFlowGraph._edge_color(0, 0.0, False, []) == "coral"


class TestRecolorAmbiguousAsBlue:
    """``_recolor_ambiguous_as_blue`` finds simple cycles from ``sources``
    back to ``sources`` inside the search graph and recolours every
    non-{blue, black} edge on those cycles to blue on ``self.g``."""

    def test_recolors_coral_edges_on_cycle(self):
        from alphaDeesp.tests.graphs_test_helpers import FakeOverFlowGraph

        obj = FakeOverFlowGraph()
        # self.g: A (amont) -> M -> B (amont). Edges coral.
        # With both A and B in `sources`, the utility finds path A->M->B.
        obj.g.add_edge("A", "M", color="coral", capacity=1., name="AM")
        obj.g.add_edge("M", "B", color="coral", capacity=1., name="MB")

        g_c = nx.MultiDiGraph()
        for u, v, d in obj.g.edges(data=True):
            g_c.add_edge(u, v, **d)

        OverFlowGraph._recolor_ambiguous_as_blue(obj, g_c, ["A", "B"])

        colors = nx.get_edge_attributes(obj.g, "color")
        assert set(colors.values()) == {"blue"}

    def test_leaves_blue_and_black_untouched(self):
        from alphaDeesp.tests.graphs_test_helpers import FakeOverFlowGraph

        obj = FakeOverFlowGraph()
        # An already-blue edge and an already-black edge on a path A -> M -> B
        obj.g.add_edge("A", "M", color="black", capacity=1., name="AM")
        obj.g.add_edge("M", "B", color="blue", capacity=1., name="MB")

        g_c = nx.MultiDiGraph()
        for u, v, d in obj.g.edges(data=True):
            g_c.add_edge(u, v, **d)

        OverFlowGraph._recolor_ambiguous_as_blue(obj, g_c, ["A", "B"])

        colors = list(nx.get_edge_attributes(obj.g, "color").values())
        # No ambiguous (non-blue/non-black) edges, so nothing changes.
        assert colors.count("black") == 1
        assert colors.count("blue") == 1

    def test_no_cycle_leaves_graph_unchanged(self):
        from alphaDeesp.tests.graphs_test_helpers import FakeOverFlowGraph

        obj = FakeOverFlowGraph()
        obj.g.add_edge("A", "B", color="coral", capacity=1., name="AB")

        g_c = nx.MultiDiGraph()
        g_c.add_edge("A", "B", color="coral", capacity=1., name="AB")

        OverFlowGraph._recolor_ambiguous_as_blue(obj, g_c, ["A"])

        # No simple cycle A -> ... -> A, so nothing to recolour.
        assert obj.g.edges[("A", "B", 0)]["color"] == "coral"


class TestOverFlowGraphScaling:

    def test_linear_scaling(self):
        df = pd.DataFrame({
            "idx_or": [0, 1, 2],
            "idx_ex": [1, 2, 0],
            "delta_flows": [1000.0, 100.0, 10.0],
            "gray_edges": [False, False, False],
            "line_name": ["L1", "L2", "L3"],
        })
        ofg = OverFlowGraph(_basic_topo(3), [], df)
        penwidths = {data["name"]: data["penwidth"] for _, _, data in ofg.g.edges(data=True)}
        # max 1000 → target_max_penwidth 15.0 → scale = 0.015.
        # Floor = max(1 MW × 0.015, 10 % × 15) = max(0.015, 1.5) = 1.5, so
        # L2 sits exactly on the floor and L3 (10 MW → 0.15 raw) is
        # clamped up to keep it visible without zoom.
        assert penwidths["L1"] == 15.0
        assert abs(penwidths["L2"] - 1.5) < 1e-5
        assert abs(penwidths["L3"] - 1.5) < 1e-5

    def test_min_penwidth_clamping(self):
        df = pd.DataFrame({
            "idx_or": [0], "idx_ex": [1],
            "delta_flows": [0.0], "gray_edges": [False],
            "line_name": ["L1"],
        })
        ofg = OverFlowGraph(_basic_topo(2), [], df)
        penwidth = list(ofg.g.edges(data=True))[0][2]["penwidth"]
        # All-zero flow: scaling_factor falls back to 1.0, so the floor is
        # max(1.0 MW, 10 % of 15) = 1.5.
        assert penwidth == 1.5


# ──────────────────────────────────────────────────────────────────────
# detect_edges_to_keep (full method)
# ──────────────────────────────────────────────────────────────────────

class TestDetectEdgesToKeep:

    def test_no_edges_of_interest_returns_empty(self):
        """``edges_of_interest`` disjoint from ``g_c`` short-circuits."""
        obj = FakeOverFlowGraph()
        g_c = nx.MultiDiGraph()
        g_c.add_edge("A", "B", color="gray", capacity=0., name="l1")
        rec, non_rec = obj.detect_edges_to_keep(g_c, ["A"], ["B"], {("X", "Y", 0)})
        assert rec == set() and non_rec == set()

    def test_no_source_nodes_in_gc_returns_empty(self):
        obj = FakeOverFlowGraph()
        g_c = nx.MultiDiGraph()
        g_c.add_edge("A", "B", color="gray", capacity=0., name="l1")
        edge_key = list(g_c.edges(keys=True))[0]
        rec, non_rec = obj.detect_edges_to_keep(g_c, ["X"], ["B"], {edge_key})
        assert rec == set() and non_rec == set()

    def test_no_target_nodes_in_gc_returns_empty(self):
        obj = FakeOverFlowGraph()
        g_c = nx.MultiDiGraph()
        g_c.add_edge("A", "B", color="gray", capacity=0., name="l1")
        edge_key = list(g_c.edges(keys=True))[0]
        rec, non_rec = obj.detect_edges_to_keep(g_c, ["A"], ["Y"], {edge_key})
        assert rec == set() and non_rec == set()

    def test_finds_reconnectable_edge_on_path(self):
        obj = FakeOverFlowGraph()
        g_c = nx.MultiDiGraph()
        g_c.add_edge("S", "M", color="gray", capacity=0., name="line_SM")
        g_c.add_edge("M", "T", color="gray", capacity=0., name="line_disconnect")
        edge = ("M", "T", 0)
        rec, _ = obj.detect_edges_to_keep(g_c, {"S"}, {"T"}, {edge})
        assert edge in rec

    def test_non_reconnectable_edge_classified_correctly(self):
        obj = FakeOverFlowGraph()
        g_c = nx.MultiDiGraph()
        g_c.add_edge("S", "M", color="gray", capacity=0., name="line_SM")
        g_c.add_edge("M", "T", color="gray", capacity=0., name="line_nr")
        edge = ("M", "T", 0)
        rec, non_rec = obj.detect_edges_to_keep(
            g_c, {"S"}, {"T"}, {edge}, non_reconnectable_edges=[edge])
        assert edge in non_rec and edge not in rec

    def test_source_equals_target_set(self):
        obj = FakeOverFlowGraph()
        g_c = nx.MultiDiGraph()
        g_c.add_edge("A", "B", color="gray", capacity=0., name="line_disc")
        g_c.add_edge("B", "C", color="gray", capacity=0., name="line_BC")
        edge = ("A", "B", 0)
        rec, _ = obj.detect_edges_to_keep(g_c, {"A", "C"}, {"A", "C"}, {edge})
        assert edge in rec

    def test_max_path_length_filter(self):
        """Paths longer than ``max_null_flow_path_length`` are excluded."""
        obj = FakeOverFlowGraph()
        g_c = nx.MultiDiGraph()
        for i in range(1, 9):
            g_c.add_edge(str(i), str(i + 1), color="gray", capacity=0., name=f"l{i}")
        rec, _ = obj.detect_edges_to_keep(
            g_c, {"1"}, {"9"}, {("8", "9", 0)}, max_null_flow_path_length=3)
        assert len(rec) == 0

    def test_no_incident_interest_returns_empty(self):
        obj = FakeOverFlowGraph()
        g_c = nx.MultiDiGraph()
        g_c.add_edge("A", "B", color="gray", capacity=0., name="l1")
        g_c.add_edge("B", "C", color="gray", capacity=0., name="l2")
        g_c.add_edge("C", "D", color="gray", capacity=0., name="l_disc")
        # edge of interest ("C","D",0) is not incident to sources A or targets B
        rec, _ = obj.detect_edges_to_keep(g_c, {"A"}, {"B"}, {("C", "D", 0)})
        assert len(rec) == 0

    def test_negative_capacities_flipped(self):
        obj = FakeOverFlowGraph()
        g_c = nx.MultiDiGraph()
        g_c.add_edge("S", "T", color="gray", capacity=-3., name="line_disc")
        edge = ("S", "T", 0)
        obj.detect_edges_to_keep(g_c, {"S"}, {"T"}, {edge})
        assert g_c.edges[edge]["capacity"] == 3.

    def test_shortest_path_preferred(self):
        obj = FakeOverFlowGraph()
        g_c = nx.MultiDiGraph()
        # Short: S -> M -> T (2 edges)
        g_c.add_edge("S", "M", color="gray", capacity=0., name="l_short1")
        g_c.add_edge("M", "T", color="gray", capacity=0., name="l_disc_short")
        # Long: S -> X -> Y -> T (3 edges)
        g_c.add_edge("S", "X", color="gray", capacity=0., name="l_long1")
        g_c.add_edge("X", "Y", color="gray", capacity=0., name="l_disc_long")
        g_c.add_edge("Y", "T", color="gray", capacity=0., name="l_long3")
        edge_short = ("M", "T", 0)
        edge_long = ("X", "Y", 1)
        rec, _ = obj.detect_edges_to_keep(
            g_c, {"S"}, {"T"}, {edge_short, edge_long})
        assert edge_short in rec


# ──────────────────────────────────────────────────────────────────────
# Internal helpers of detect_edges_to_keep
# ──────────────────────────────────────────────────────────────────────

class TestDetectEdgesHelpers:

    def test_prepare_returns_none_when_no_edges_of_interest_in_gc(self):
        obj = DetectEdgesHelperHost()
        g_c = nx.MultiDiGraph()
        g_c.add_edge("A", "B", name="l1", capacity=0.)
        assert obj._prepare_detect_edges_inputs(
            g_c, ["A"], ["B"], edges_of_interest={("X", "Y", 0)},
            non_reconnectable_edges=[], depth_edges_search=2) is None

    def test_prepare_flips_negative_capacities(self):
        obj = DetectEdgesHelperHost()
        g_c = nx.MultiDiGraph()
        g_c.add_edge("A", "B", name="l1", capacity=-4.)
        prepared = obj._prepare_detect_edges_inputs(
            g_c, ["A"], ["B"], edges_of_interest={("A", "B", 0)},
            non_reconnectable_edges=[], depth_edges_search=2)
        assert prepared is not None
        assert g_c.edges[("A", "B", 0)]["capacity"] == 4.

    def test_prepare_returns_none_when_source_or_target_missing(self):
        obj = DetectEdgesHelperHost()
        g_c = nx.MultiDiGraph()
        g_c.add_edge("A", "B", name="l1", capacity=0.)
        assert obj._prepare_detect_edges_inputs(
            g_c, ["Z"], ["B"], edges_of_interest={("A", "B", 0)},
            non_reconnectable_edges=[], depth_edges_search=2) is None

    def test_compute_sssp_paths_caches_per_source(self):
        obj = DetectEdgesHelperHost()
        g_c = nx.MultiDiGraph()
        g_c.add_edge("S", "M", name="sm", capacity=0.)
        g_c.add_edge("M", "T", name="mt", capacity=0.)
        prepared = obj._prepare_detect_edges_inputs(
            g_c, ["S"], ["T"], edges_of_interest={("M", "T", 0)},
            non_reconnectable_edges=[], depth_edges_search=2)
        sssp = obj._compute_sssp_paths(g_c, prepared, {("M", "T", 0)})
        assert sssp["S"]["T"] == ["S", "M", "T"]

    def test_collect_paths_filters_by_max_length(self):
        obj = DetectEdgesHelperHost()
        g_c = nx.MultiDiGraph()
        for i in range(1, 9):
            g_c.add_edge(str(i), str(i + 1), name=f"l{i}", capacity=0.)
        prepared = obj._prepare_detect_edges_inputs(
            g_c, ["1"], ["9"], edges_of_interest={("8", "9", 0)},
            non_reconnectable_edges=[], depth_edges_search=10)
        sssp = obj._compute_sssp_paths(g_c, prepared, {("8", "9", 0)})
        paths = obj._collect_paths_of_interest(g_c, prepared, sssp, max_null_flow_path_length=3)
        assert paths == []

    def test_classify_routes_non_reconnectable_to_the_right_set(self):
        obj = DetectEdgesHelperHost()
        g_c = nx.MultiDiGraph()
        g_c.add_edge("S", "M", name="sm", capacity=0.)
        g_c.add_edge("M", "T", name="mt", capacity=0.)
        edge_mt = ("M", "T", 0)
        prepared = obj._prepare_detect_edges_inputs(
            g_c, ["S"], ["T"], edges_of_interest={edge_mt},
            non_reconnectable_edges=[edge_mt], depth_edges_search=2)
        sssp = obj._compute_sssp_paths(g_c, prepared, {edge_mt})
        paths = obj._collect_paths_of_interest(g_c, prepared, sssp, max_null_flow_path_length=7)
        rec, non_rec = obj._classify_paths_by_reconnectability(prepared, paths)
        assert edge_mt in non_rec and edge_mt not in rec
