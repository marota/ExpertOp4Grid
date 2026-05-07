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
# extra_lines_to_cut: operator-supplied extras keep their natural flow
# colour, are flagged ``is_extra_cut`` (alongside ``constrained``), and
# stay out of the ``is_overload`` / ``is_monitored`` semantic layers.
# ──────────────────────────────────────────────────────────────────────


def _three_line_df():
    """L1 positive overload, L2 negative extra-cut, L3 healthy positive."""
    return pd.DataFrame({
        "idx_or": [0, 1, 2],
        "idx_ex": [1, 2, 0],
        "delta_flows": [1000.0, -100.0, 50.0],
        "gray_edges": [False, False, False],
        "line_name": ["L1", "L2", "L3"],
    })


def _edge_by_name(g, name):
    for u, v, k, data in g.edges(keys=True, data=True):
        if data.get("name") == name:
            return (u, v, k), data
    raise AssertionError(f"edge {name!r} not found")


class TestExtraLinesToCut:

    def test_default_extras_is_empty(self):
        ofg = OverFlowGraph(_basic_topo(3), [0, 1], _three_line_df())
        assert ofg.extra_lines_cut == set()

    def test_extras_stored_as_set(self):
        ofg = OverFlowGraph(
            _basic_topo(3), [0, 1], _three_line_df(),
            extra_lines_to_cut=[1, 1],
        )
        assert ofg.extra_lines_cut == {1}

    def test_extras_keep_natural_flow_colour(self):
        """An extra-cut line never gets the black overload colour — it
        keeps coral / blue based on its delta-flow polarity."""
        ofg = OverFlowGraph(
            _basic_topo(3), [0, 1], _three_line_df(),
            extra_lines_to_cut=[1],
        )
        _, l1 = _edge_by_name(ofg.g, "L1")  # in lines_to_cut, NOT extra
        _, l2 = _edge_by_name(ofg.g, "L2")  # in lines_to_cut AND extra
        _, l3 = _edge_by_name(ofg.g, "L3")  # not cut at all
        assert l1["color"] == "black"
        assert l2["color"] == "blue"   # natural — delta_flows = -100
        assert l3["color"] == "coral"  # natural — delta_flows = +50

    def test_extras_are_constrained_and_flagged(self):
        ofg = OverFlowGraph(
            _basic_topo(3), [0, 1], _three_line_df(),
            extra_lines_to_cut=[1],
        )
        _, l1 = _edge_by_name(ofg.g, "L1")
        _, l2 = _edge_by_name(ofg.g, "L2")
        _, l3 = _edge_by_name(ofg.g, "L3")
        # Real overload: constrained, not extra.
        assert l1.get("constrained") is True
        assert "is_extra_cut" not in l1
        # Extra cut: both flags set so downstream layers can find it.
        assert l2.get("constrained") is True
        assert l2.get("is_extra_cut") is True
        # Untouched line carries neither flag.
        assert "constrained" not in l3
        assert "is_extra_cut" not in l3

    def test_extras_skipped_in_overload_and_monitored(self):
        """``highlight_significant_line_loading`` must not stamp
        ``is_overload`` / ``is_monitored`` on extras, must not yellow-tint
        their colour, but should still annotate the edge label."""
        ofg = OverFlowGraph(
            _basic_topo(3), [0, 1], _three_line_df(),
            extra_lines_to_cut=[1],
        )
        ofg.highlight_significant_line_loading({
            "L1": {"before": 110, "after": 80},
            "L2": {"before": 90, "after": 0},
            "L3": {"before": 75, "after": 60},
        })
        _, l1 = _edge_by_name(ofg.g, "L1")
        _, l2 = _edge_by_name(ofg.g, "L2")
        _, l3 = _edge_by_name(ofg.g, "L3")

        # L1 is a real overload: yellow-tinted, both flags set.
        assert l1["color"] == '"black:yellow:black"'
        assert l1.get("is_overload") is True
        assert l1.get("is_monitored") is True

        # L2 is the extra cut: keeps natural blue (no yellow tint), no
        # is_overload / is_monitored, but the loading annotation still
        # fires so the operator sees how their choice materialises.
        assert l2["color"] == "blue"
        assert l2.get("is_overload") is None
        assert l2.get("is_monitored") is None
        assert "90% → <B>0%</B>" in l2["label"]

        # L3 is a low-margin line (not overload, not extra).
        assert l3["color"] == '"coral:yellow:coral"'
        assert l3.get("is_overload") is None
        assert l3.get("is_monitored") is True

    def test_legacy_behaviour_when_no_extras(self):
        """Without ``extra_lines_to_cut`` the contingency lines render
        black and get tagged as overloads — the legacy contract."""
        ofg = OverFlowGraph(_basic_topo(3), [0], _three_line_df())
        ofg.highlight_significant_line_loading({
            "L1": {"before": 110, "after": 80},
        })
        _, l1 = _edge_by_name(ofg.g, "L1")
        assert l1["color"] == '"black:yellow:black"'
        assert l1.get("is_overload") is True
        assert l1.get("is_extra_cut") is None


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


# ──────────────────────────────────────────────────────────────────────
# Source-of-truth attribute tagging — feeds the interactive HTML viewer
# layer toggles (hubs, red-loops, constrained path, overloads, monitored).
# ──────────────────────────────────────────────────────────────────────

class TestSetHubsShapeAttributeFlag:
    def test_is_hub_flag_is_set_on_hub_nodes_only(self):
        g = nx.MultiDiGraph()
        g.add_node("A", shape="oval")
        g.add_node("B", shape="oval")
        g.add_node("C", shape="oval")
        ofg = make_ofg_with_graph(g)
        ofg.set_hubs_shape(["A", "C"], shape_hub="diamond")
        assert ofg.g.nodes["A"]["is_hub"] is True
        assert ofg.g.nodes["B"]["is_hub"] is False
        assert ofg.g.nodes["C"]["is_hub"] is True
        assert ofg.g.nodes["A"]["shape"] == "diamond"


class TestCollapseRedLoopsIsPurelyVisual:
    """``collapse_red_loops`` is now a purely visual heuristic (point
    shape vs oval). Semantic ``in_red_loop`` tagging is handled by
    :meth:`tag_red_loops` which consumes the source-of-truth list
    from the recommender's
    ``Structured_Overload_Distribution_Graph.get_dispatch_edges_nodes``.
    """

    def test_collapse_does_not_set_in_red_loop(self):
        g = nx.MultiDiGraph()
        g.add_node("N1", shape="oval")
        g.add_node("N2", shape="oval")
        g.add_edge("N1", "N2", color="coral", name="line_1")
        ofg = make_ofg_with_graph(g)
        ofg.collapse_red_loops()
        # N1 (purely-coral) collapses visually but no semantic flag.
        assert ofg.g.nodes["N1"]["shape"] == "point"
        assert "in_red_loop" not in ofg.g.nodes["N1"]
        assert "in_red_loop" not in ofg.g.nodes["N2"]
        for _, _, _, data in ofg.g.edges(keys=True, data=True):
            assert "in_red_loop" not in data

    def test_collapse_does_not_set_in_red_loop_for_blue_only(self):
        g = nx.MultiDiGraph()
        g.add_node("N1", shape="oval")
        g.add_node("N2", shape="oval")
        g.add_edge("N1", "N2", color="blue")
        ofg = make_ofg_with_graph(g)
        ofg.collapse_red_loops()
        assert "in_red_loop" not in ofg.g.nodes["N1"]
        assert "in_red_loop" not in ofg.g.nodes["N2"]


class TestHighlightSignificantLineLoadingFlags:
    def _make_graph_with_named_edges(self):
        g = nx.MultiDiGraph()
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B", name="line_overload", color="black", label="100")
        g.add_edge("B", "C", name="line_monitored", color="coral", label="50")
        g.add_edge("A", "C", name="line_quiet", color="gray", label="10")
        return g

    def test_overload_flag_on_black_edges(self):
        g = self._make_graph_with_named_edges()
        ofg = make_ofg_with_graph(g)
        ofg.highlight_significant_line_loading({
            "line_overload": {"before": 95, "after": 110},
            "line_monitored": {"before": 78, "after": 92},
        })
        # Find edges by name and check flags.
        edge_flags = {
            data.get("name"): {
                "is_overload": data.get("is_overload"),
                "is_monitored": data.get("is_monitored"),
            }
            for _, _, _, data in ofg.g.edges(keys=True, data=True)
        }
        # Overloads are a strict subset of monitored / low-margin
        # lines: every entry in dict_line_loading is monitored, and
        # the black ones are additionally overloads.
        assert edge_flags["line_overload"]["is_overload"] is True
        assert edge_flags["line_overload"]["is_monitored"] is True
        assert edge_flags["line_monitored"]["is_monitored"] is True
        assert edge_flags["line_monitored"]["is_overload"] is None
        # Untagged line keeps neither flag.
        assert edge_flags["line_quiet"]["is_overload"] is None
        assert edge_flags["line_quiet"]["is_monitored"] is None


class TestTagConstrainedPath:
    def test_tags_edges_by_name_and_nodes_by_identity(self):
        g = nx.MultiDiGraph()
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B", name="L1")
        g.add_edge("B", "C", name="L2")
        g.add_edge("A", "C", name="L3")
        ofg = make_ofg_with_graph(g)
        ofg.tag_constrained_path(
            lines_constrained_path=["L1", "L2"],
            nodes_constrained_path=["A", "B"],
        )
        edges_on = {
            data.get("name"): data.get("on_constrained_path")
            for _, _, _, data in ofg.g.edges(keys=True, data=True)
        }
        assert edges_on["L1"] is True
        assert edges_on["L2"] is True
        assert edges_on["L3"] is None
        assert ofg.g.nodes["A"].get("on_constrained_path") is True
        assert ofg.g.nodes["B"].get("on_constrained_path") is True
        assert ofg.g.nodes["C"].get("on_constrained_path") is None

    def test_no_op_when_inputs_empty(self):
        g = nx.MultiDiGraph()
        g.add_node("A")
        g.add_edge("A", "A", name="loop")
        ofg = make_ofg_with_graph(g)
        ofg.tag_constrained_path(None, None)
        ofg.tag_constrained_path([], [])
        for _, _, _, data in ofg.g.edges(keys=True, data=True):
            assert "on_constrained_path" not in data
        assert "on_constrained_path" not in ofg.g.nodes["A"]


# ──────────────────────────────────────────────────────────────────────
# Layer-toggle bug fixes (v2): hubs auto-membership, broader red loops,
# coral filtering on constrained path, no-op on coral-only constrained
# path entries.
# ──────────────────────────────────────────────────────────────────────

class TestSetHubsShapeAlsoTagsRedLoopAndConstrainedPath:
    """Hubs are by definition both on the constrained path AND
    inside red-loop paths — those flags must be set alongside `is_hub`.
    """

    def test_hubs_get_on_constrained_path_flag(self):
        g = nx.MultiDiGraph()
        g.add_node("HUB", shape="oval")
        g.add_node("OTHER", shape="oval")
        ofg = make_ofg_with_graph(g)
        ofg.set_hubs_shape(["HUB"], shape_hub="diamond")
        assert ofg.g.nodes["HUB"].get("on_constrained_path") is True
        assert "on_constrained_path" not in ofg.g.nodes["OTHER"]

    def test_hubs_get_in_red_loop_flag(self):
        g = nx.MultiDiGraph()
        g.add_node("HUB", shape="oval")
        g.add_node("OTHER", shape="oval")
        ofg = make_ofg_with_graph(g)
        ofg.set_hubs_shape(["HUB"], shape_hub="diamond")
        assert ofg.g.nodes["HUB"].get("in_red_loop") is True
        assert "in_red_loop" not in ofg.g.nodes["OTHER"]


class TestTagRedLoops:
    """``tag_red_loops`` propagates the source-of-truth lists from
    ``Structured_Overload_Distribution_Graph.get_dispatch_edges_nodes(
    only_loop_paths=True)`` onto graph attributes. The viewer's
    "Red-loop paths" layer reads those flags directly — there is no
    heuristic involved.
    """

    def test_tags_only_lines_in_provided_list(self):
        g = nx.MultiDiGraph()
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B", name="loop_line", color="coral")
        g.add_edge("B", "C", name="exit_line", color="coral")
        ofg = make_ofg_with_graph(g)
        ofg.tag_red_loops(
            lines_red_loops=["loop_line"],
            nodes_red_loops=["A", "B"],
        )
        edges_on = {
            data["name"]: data.get("in_red_loop")
            for _, _, _, data in ofg.g.edges(keys=True, data=True)
        }
        assert edges_on["loop_line"] is True
        # exit_line is NOT in the source-of-truth list → not tagged
        # (this is the user-reported CHALOY633 invariant).
        assert edges_on["exit_line"] is None
        assert ofg.g.nodes["A"].get("in_red_loop") is True
        assert ofg.g.nodes["B"].get("in_red_loop") is True
        assert "in_red_loop" not in ofg.g.nodes["C"]

    def test_no_op_when_inputs_empty(self):
        g = nx.MultiDiGraph()
        g.add_node("A")
        g.add_edge("A", "A", name="self_loop", color="coral")
        ofg = make_ofg_with_graph(g)
        ofg.tag_red_loops(None, None)
        ofg.tag_red_loops([], [])
        for _, _, _, data in ofg.g.edges(keys=True, data=True):
            assert "in_red_loop" not in data
        assert "in_red_loop" not in ofg.g.nodes["A"]

    def test_tags_compound_color_edges_when_in_source_list(self):
        # A monitored coral edge ("coral:yellow:coral") that the
        # recommender included in the dispatch loop list MUST be
        # tagged — name match is colour-agnostic. (The previous
        # heuristic-based logic already handled compound colours;
        # this test pins the explicit-list contract too.)
        g = nx.MultiDiGraph()
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B", name="L_MON", color='"coral:yellow:coral"')
        ofg = make_ofg_with_graph(g)
        ofg.tag_red_loops(lines_red_loops=["L_MON"], nodes_red_loops=["A", "B"])
        edge_data = list(ofg.g.edges(keys=True, data=True))[0][3]
        assert edge_data.get("in_red_loop") is True

    def test_chalop6_chalop3_style_exit_branch_is_NOT_tagged(self):
        """Regression for the user-reported CHALOP6→CHALOP3 case:
        the recommender's ``get_dispatch_edges_nodes(only_loop_paths
        =True)`` does NOT include such transformer "exit" branches —
        because their endpoints are not in any cycle path. The
        explicit-list approach therefore leaves them un-tagged.
        """
        g = nx.MultiDiGraph()
        g.add_node("CHALOP6")
        g.add_node("CHALOP3")
        g.add_node("LOUHAP3")
        g.add_edge("CHALOP6", "CHALOP3", name="CHALOY633", color="coral")
        g.add_edge("CHALOP6", "CHALOP3", name="CHALOY631", color="coral")
        g.add_edge("CHALOP6", "CHALOP3", name="CHALOY632", color="coral")
        g.add_edge("CHALOP3", "LOUHAP3", name="CHALOL31LOUHA",
                   color="coral", style="dashed")
        ofg = make_ofg_with_graph(g)
        # Recommender returns an empty dispatch loop list because none
        # of these nodes participate in a true cycle path.
        ofg.tag_red_loops(lines_red_loops=[], nodes_red_loops=[])
        for _, _, _, data in ofg.g.edges(keys=True, data=True):
            assert "in_red_loop" not in data, (
                f"edge {data['name']} wrongly tagged in_red_loop"
            )
        for n in ("CHALOP6", "CHALOP3", "LOUHAP3"):
            assert "in_red_loop" not in ofg.g.nodes[n], (
                f"node {n} wrongly tagged in_red_loop"
            )


class TestTagConstrainedPathSkipsCoralEdges:
    """The constrained path is, by definition, the network of black
    (overloaded) and blue (negative-flow) edges. Coral edges that share
    a name with a constrained-path entry (because the
    ``MultiDiGraph`` carries both flow directions of one physical
    line) must NOT end up tagged.
    """

    def test_coral_edge_with_matching_name_is_skipped(self):
        g = nx.MultiDiGraph()
        g.add_node("A")
        g.add_node("B")
        # Same `name` for both directions: blue (negative) + coral (positive).
        g.add_edge("A", "B", name="L1", color="blue")
        g.add_edge("B", "A", name="L1", color="coral")
        ofg = make_ofg_with_graph(g)
        ofg.tag_constrained_path(lines_constrained_path=["L1"])
        flagged_colors = [
            data.get("color")
            for _, _, _, data in ofg.g.edges(keys=True, data=True)
            if data.get("on_constrained_path")
        ]
        assert flagged_colors == ["blue"]

    def test_compound_color_string_with_coral_base_is_skipped(self):
        # After `highlight_significant_line_loading` the `color` may be
        # a graphviz compound `"coral:yellow:coral"`. The split-on-':'
        # heuristic must still classify it as coral and skip it.
        g = nx.MultiDiGraph()
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B", name="L1", color='"coral:yellow:coral"')
        g.add_edge("B", "A", name="L1", color="black")
        ofg = make_ofg_with_graph(g)
        ofg.tag_constrained_path(lines_constrained_path=["L1"])
        flagged = [
            (data.get("color"), data.get("on_constrained_path"))
            for _, _, _, data in ofg.g.edges(keys=True, data=True)
        ]
        # The black one is tagged, the compound-coral is skipped.
        assert ('"coral:yellow:coral"', None) in [(c, t) for c, t in flagged] \
               or ('"coral:yellow:coral"', None) in flagged
        assert any(c == "black" and t is True for c, t in flagged)
        assert all(not (c == '"coral:yellow:coral"' and t) for c, t in flagged)

    def test_black_and_blue_edges_with_matching_name_are_tagged(self):
        g = nx.MultiDiGraph()
        g.add_node("A")
        g.add_node("B")
        g.add_node("C")
        g.add_edge("A", "B", name="L1", color="black")
        g.add_edge("B", "C", name="L2", color="blue")
        ofg = make_ofg_with_graph(g)
        ofg.tag_constrained_path(lines_constrained_path=["L1", "L2"])
        for _, _, _, data in ofg.g.edges(keys=True, data=True):
            assert data.get("on_constrained_path") is True
