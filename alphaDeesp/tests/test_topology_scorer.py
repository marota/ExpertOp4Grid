"""Unit tests for :mod:`alphaDeesp.core.topology_scorer`.

Tests the scoring helpers exposed by TopologyScorerMixin using lightweight
host objects backed by hand-crafted NetworkX graphs and mock element data.
"""

import pytest
import networkx as nx
import numpy as np
import pandas as pd

from alphaDeesp.core.topology_scorer import TopologyScorerMixin
from alphaDeesp.core.elements import (
    Consumption,
    ExtremityLine,
    OriginLine,
    Production,
)


# ──────────────────────────────────────────────────────────────────────
# Minimal host class (uses mixin without AlphaDeesp __init__)
# ──────────────────────────────────────────────────────────────────────

class _ScorerHost(TopologyScorerMixin):
    """Minimal concrete class exposing TopologyScorerMixin methods."""

    def __init__(self, g, simulator_data, distribution_graph=None, debug=False):
        self.g = g
        self.simulator_data = simulator_data
        self.g_distribution_graph = distribution_graph
        self.debug = debug


# ──────────────────────────────────────────────────────────────────────
# get_prod_conso_sum
# ──────────────────────────────────────────────────────────────────────

class TestGetProdConsoSum:

    def _host(self, elements, node=1):
        return _ScorerHost(nx.MultiDiGraph(), {"substations_elements": {node: elements}})

    def test_production_on_bus(self):
        elements = [Production(busbar_id=0, value=10.0)]
        host = self._host(elements)
        assert host.get_prod_conso_sum(1, 0, [0]) == 10.0

    def test_consumption_on_bus(self):
        elements = [Consumption(busbar_id=1, value=5.0)]
        host = self._host(elements)
        assert host.get_prod_conso_sum(1, 1, [1]) == -5.0

    def test_mixed_elements_on_same_bus(self):
        elements = [Production(busbar_id=0, value=10.0), Consumption(busbar_id=0, value=3.0)]
        host = self._host(elements)
        assert host.get_prod_conso_sum(1, 0, [0, 0]) == pytest.approx(7.0)

    def test_element_on_different_bus_not_counted(self):
        elements = [Production(busbar_id=0, value=10.0), Production(busbar_id=1, value=5.0)]
        host = self._host(elements)
        # only bus 0 element counted
        assert host.get_prod_conso_sum(1, 0, [0, 1]) == pytest.approx(10.0)

    def test_no_elements_returns_zero(self):
        host = self._host([])
        assert host.get_prod_conso_sum(1, 0, []) == 0

    def test_origin_line_ignored(self):
        elements = [OriginLine(busbar_id=0, end_substation_id=2)]
        host = self._host(elements)
        assert host.get_prod_conso_sum(1, 0, [0]) == 0


# ──────────────────────────────────────────────────────────────────────
# get_bus_id_from_edge
# ──────────────────────────────────────────────────────────────────────

class TestGetBusIdFromEdge:

    def _host(self, elements, node=1):
        return _ScorerHost(nx.MultiDiGraph(), {"substations_elements": {node: elements}})

    def test_origin_line_out_edge(self):
        elements = [OriginLine(busbar_id=0, end_substation_id=2)]
        host = self._host(elements, node=1)
        # edge from node 1 to 2 (out-edge); topo_vect[0] = 0 → bus 0
        edge = (1, 2, 0)
        assert host.get_bus_id_from_edge(1, edge, [0]) == 0

    def test_extremity_line_in_edge(self):
        elements = [ExtremityLine(busbar_id=1, start_substation_id=3)]
        host = self._host(elements, node=1)
        edge = (3, 1, 0)
        assert host.get_bus_id_from_edge(1, edge, [1]) == 1

    def test_parallel_edge_key_selects_second(self):
        elements = [
            OriginLine(busbar_id=0, end_substation_id=2),
            OriginLine(busbar_id=1, end_substation_id=2),
        ]
        host = self._host(elements, node=1)
        # key=1 → second OriginLine → bus_id 1
        edge = (1, 2, 1)
        assert host.get_bus_id_from_edge(1, edge, [0, 1]) == 1

    def test_no_match_returns_none(self):
        elements = [OriginLine(busbar_id=0, end_substation_id=9)]
        host = self._host(elements, node=1)
        edge = (1, 2, 0)  # edge to node 2 but element connects to 9
        assert host.get_bus_id_from_edge(1, edge, [0]) is None


# ──────────────────────────────────────────────────────────────────────
# is_connected_to_cpath
# ──────────────────────────────────────────────────────────────────────

class TestIsConnectedToCpath:

    def _host(self):
        return _ScorerHost(nx.MultiDiGraph(), {"substations_elements": {}})

    def test_negative_blue_edge_is_connected(self):
        host = self._host()
        edge = ("A", "B", 0)
        color_attrs = {edge: "blue"}
        label_attrs = {edge: "-5.0"}
        assert host.is_connected_to_cpath(color_attrs, label_attrs, "A", edge, False) is True

    def test_negative_black_edge_is_connected(self):
        host = self._host()
        edge = ("A", "B", 0)
        assert host.is_connected_to_cpath(
            {edge: "black"}, {edge: "-1.0"}, "A", edge, False) is True

    def test_positive_blue_edge_not_connected(self):
        host = self._host()
        edge = ("A", "B", 0)
        assert host.is_connected_to_cpath(
            {edge: "blue"}, {edge: "5.0"}, "A", edge, False) is False

    def test_single_node_flag_blocks_connection(self):
        host = self._host()
        edge = ("A", "B", 0)
        assert host.is_connected_to_cpath(
            {edge: "blue"}, {edge: "-5.0"}, "A", edge, isSingleNode=True) is False

    def test_gray_edge_not_connected(self):
        host = self._host()
        edge = ("A", "B", 0)
        assert host.is_connected_to_cpath(
            {edge: "gray"}, {edge: "-5.0"}, "A", edge, False) is False


# ──────────────────────────────────────────────────────────────────────
# _collect_flows_on_bus
# ──────────────────────────────────────────────────────────────────────

class TestCollectFlowsOnBus:

    def _host_with_node(self, node, neighbor, elements, topo_vect):
        """Build a graph with one in-edge and one out-edge at node."""
        g = nx.MultiDiGraph()
        g.add_edge(neighbor, node, capacity=-3.0, label="-3.0")
        g.add_edge(node, neighbor, capacity=5.0, label="5.0")
        sim_data = {"substations_elements": {node: elements}}
        return _ScorerHost(g, sim_data), topo_vect

    def test_in_negative_classified_correctly(self):
        elem_in = ExtremityLine(busbar_id=0, start_substation_id=99)
        elem_out = OriginLine(busbar_id=0, end_substation_id=99)
        host, tv = self._host_with_node(1, 99, [elem_in, elem_out], [0, 0])
        label_attrs = nx.get_edge_attributes(host.g, "label")
        result = host._collect_flows_on_bus(host.g, 1, 0, tv, label_attrs)
        assert result["in_neg"] == [3.0]
        assert result["out_pos"] == [5.0]
        assert result["in_pos"] == []
        assert result["out_neg"] == []

    def test_only_bus1_edges_counted_when_bus0_requested(self):
        elem_in = ExtremityLine(busbar_id=1, start_substation_id=99)
        elem_out = OriginLine(busbar_id=1, end_substation_id=99)
        host, tv = self._host_with_node(1, 99, [elem_in, elem_out], [1, 1])
        label_attrs = nx.get_edge_attributes(host.g, "label")
        result = host._collect_flows_on_bus(host.g, 1, 0, tv, label_attrs)
        assert result["in_neg"] == []
        assert result["out_pos"] == []


# ──────────────────────────────────────────────────────────────────────
# _score_not_connected_to_cpath
# ──────────────────────────────────────────────────────────────────────

class TestScoreNotConnectedToCpath:

    def test_balanced_negative_flows_score_zero(self):
        g = nx.MultiDiGraph()
        g.add_edge("X", 1, capacity=-4.0, label="-4.0", color="blue")
        g.add_edge(1, "X", capacity=-4.0, label="-4.0", color="blue")
        elements = [
            ExtremityLine(busbar_id=0, start_substation_id="X"),
            OriginLine(busbar_id=0, end_substation_id="X"),
        ]
        host = _ScorerHost(g, {"substations_elements": {1: elements}})
        label_attrs = nx.get_edge_attributes(g, "label")
        score = host._score_not_connected_to_cpath(g, 1, [0, 0], label_attrs)
        assert score == pytest.approx(0.0)

    def test_unbalanced_returns_min_abs_diff(self):
        g = nx.MultiDiGraph()
        g.add_edge("X", 1, capacity=-6.0, label="-6.0", color="blue")
        g.add_edge(1, "X", capacity=-2.0, label="-2.0", color="blue")
        elements = [
            ExtremityLine(busbar_id=0, start_substation_id="X"),
            OriginLine(busbar_id=0, end_substation_id="X"),
        ]
        host = _ScorerHost(g, {"substations_elements": {1: elements}})
        label_attrs = nx.get_edge_attributes(g, "label")
        score = host._score_not_connected_to_cpath(g, 1, [0, 0], label_attrs)
        # bus0: |6 - 2| = 4; bus1: |0 - 0| = 0; min = 0
        assert score == pytest.approx(0.0)
