"""Unit tests for the shortest-path helpers in
:mod:`alphaDeesp.core.graphs.shortest_paths`."""

import networkx as nx

from alphaDeesp.core.graphsAndPaths import (
    shortest_path_with_promoted_edges,
    shortest_path_min_weight_then_hops,
)


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
        assert cost == float("inf")

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
        g.add_edge("A", "B", capacity=0)
        g.add_edge("B", "D", capacity=0)
        g.add_edge("A", "C", capacity=0)
        g.add_edge("C", "D", capacity=0)
        path, _ = shortest_path_with_promoted_edges(
            g, "A", "D", promoted_edges=[("A", "C")], weight_attr="capacity")
        assert path == ["A", "C", "D"]

    def test_single_node_path(self):
        g = nx.DiGraph()
        g.add_node("A")
        path, _ = shortest_path_with_promoted_edges(g, "A", "A", [], weight_attr="capacity")
        assert path == ["A"]


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
        assert cost == float("inf")

    def test_multigraph_with_key(self):
        g = nx.MultiDiGraph()
        g.add_edge("A", "B", weight=5)
        k1 = g.add_edge("A", "B", weight=1)
        g.add_edge("B", "C", weight=1)
        path, cost = shortest_path_min_weight_then_hops(
            g, "A", "C", mandatory_edge=("A", "B", k1), weight_attr="weight")
        assert path is not None
        assert cost == 2  # key k1 (1) + B->C (1)
