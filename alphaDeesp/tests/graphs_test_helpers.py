"""Shared test helpers for the `test_graph_*` suite.

Holds the small hand-crafted graph fixtures and the minimal mock classes
that route calls to ``OverFlowGraph`` methods without invoking its
``__init__``. Tests across multiple files reuse these so each focused
test module can stay short.
"""

import networkx as nx
from alphaDeesp.core.graphsAndPaths import OverFlowGraph


# ──────────────────────────────────────────────────────────────────────
# Hand-crafted graph fixtures
# ──────────────────────────────────────────────────────────────────────

def make_colored_multidigraph():
    """Small MultiDiGraph with mixed edge colours.

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


def make_linear_multidigraph():
    """Linear 5-node gray MultiDiGraph suitable for null-flow tests."""
    g = nx.MultiDiGraph()
    g.add_edge(1, 2, color="gray", capacity=0., name="e1")
    g.add_edge(2, 3, color="gray", capacity=0., name="e2")
    g.add_edge(3, 4, color="gray", capacity=0., name="e3")
    g.add_edge(4, 5, color="gray", capacity=0., name="e4")
    return g


def make_branching_multidigraph():
    """Branching gray MultiDiGraph with one disconnected edge.

      S --s1--> A --a1--> B --b1--> T
                A --a2--> C --c1--> T     (a2 == disconnected line)
    """
    g = nx.MultiDiGraph()
    g.add_edge("S", "A", color="gray", capacity=0., name="s1")
    g.add_edge("A", "B", color="gray", capacity=0., name="a1")
    g.add_edge("B", "T", color="gray", capacity=0., name="b1")
    g.add_edge("A", "C", color="gray", capacity=0., name="a2_disconnected")
    g.add_edge("C", "T", color="gray", capacity=0., name="c1")
    return g


def make_detect_edges_graph():
    """Two-hop gray MultiDiGraph for ``detect_edges_to_keep`` tests."""
    g = nx.MultiDiGraph()
    g.add_edge("SRC1", "MID", color="gray", capacity=0., name="line_SM")
    g.add_edge("MID", "TGT1", color="gray", capacity=0., name="line_disconnect")
    return g


# ──────────────────────────────────────────────────────────────────────
# Minimal mocks for invoking OverFlowGraph methods
# ──────────────────────────────────────────────────────────────────────

class FakeOverFlowGraph:
    """Minimal stand-in for ``OverFlowGraph`` that skips ``__init__``.

    Exposes the bound-method access used by tests on
    ``detect_edges_to_keep`` and its helpers.
    """

    def __init__(self):
        self.g = nx.MultiDiGraph()

    detect_edges_to_keep = OverFlowGraph.detect_edges_to_keep
    _prepare_detect_edges_inputs = OverFlowGraph._prepare_detect_edges_inputs
    _compute_sssp_paths = OverFlowGraph._compute_sssp_paths
    _collect_paths_of_interest = OverFlowGraph._collect_paths_of_interest
    _classify_paths_by_reconnectability = OverFlowGraph._classify_paths_by_reconnectability


class DetectEdgesHelperHost:
    """Exposes ``detect_edges_to_keep`` internal helpers without any state."""
    _prepare_detect_edges_inputs = OverFlowGraph._prepare_detect_edges_inputs
    _compute_sssp_paths = OverFlowGraph._compute_sssp_paths
    _collect_paths_of_interest = OverFlowGraph._collect_paths_of_interest
    _classify_paths_by_reconnectability = OverFlowGraph._classify_paths_by_reconnectability


class NullFlowHelperHost:
    """Exposes the ``add_relevant_null_flow_lines`` helpers without state."""
    _prepare_null_flow_edge_sets = OverFlowGraph._prepare_null_flow_edge_sets
    _build_gray_components = OverFlowGraph._build_gray_components
    _structural_info_for_null_flow = OverFlowGraph._structural_info_for_null_flow
    _apply_null_flow_recoloring = OverFlowGraph._apply_null_flow_recoloring


def make_ofg_with_graph(g):
    """Build a :class:`FakeOverFlowGraph` with ``g`` already attached and
    bind the ``OverFlowGraph`` colouring methods tests typically call."""
    obj = FakeOverFlowGraph()
    obj.g = g
    obj.keep_overloads_components = OverFlowGraph.keep_overloads_components.__get__(obj)
    obj.collapse_red_loops = OverFlowGraph.collapse_red_loops.__get__(obj)
    return obj
