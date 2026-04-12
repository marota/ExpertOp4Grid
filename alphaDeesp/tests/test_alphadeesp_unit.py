"""
Unit tests for the helper methods introduced while decomposing
``rank_current_topo_at_node_x`` and ``apply_new_topo_to_graph`` in
``alphaDeesp/core/alphadeesp.py``.

The helpers under test do not exercise the heavier ``AlphaDeesp.__init__``
pipeline; the tests build a minimal hand-crafted graph + simulator data and
invoke the methods directly through a mock object that exposes the unbound
methods of :class:`~alphaDeesp.core.alphadeesp.AlphaDeesp`.
"""

import networkx as nx

from alphaDeesp.core.alphadeesp import AlphaDeesp
from alphaDeesp.core.elements import (
    Consumption,
    ExtremityLine,
    OriginLine,
    Production,
)
from alphaDeesp.core.twin_nodes import (
    TWIN_NODE_OFFSET,
    is_twin_node_id,
    original_substation_id,
    twin_node_id,
)


# ──────────────────────────────────────────────────────────────────────
# Twin-node id scheme
# ──────────────────────────────────────────────────────────────────────


class TestTwinNodeIds:

    def test_twin_and_original_roundtrip(self):
        assert original_substation_id(twin_node_id(7)) == 7
        assert original_substation_id(twin_node_id(0)) == 0
        assert original_substation_id(twin_node_id(1234)) == 1234

    def test_is_twin_node_id_discriminates(self):
        assert is_twin_node_id(twin_node_id(0)) is True
        assert is_twin_node_id(twin_node_id(99)) is True
        assert is_twin_node_id(0) is False
        assert is_twin_node_id(99) is False
        assert is_twin_node_id("not-an-int") is False

    def test_twin_ids_are_disjoint_from_realistic_substation_ids(self):
        # Any substation id below the offset must stay below the offset.
        for sub_id in [0, 1, 14, 118, 999, 5_000]:
            assert sub_id < TWIN_NODE_OFFSET
            assert twin_node_id(sub_id) >= TWIN_NODE_OFFSET

    def test_twin_node_id_rejects_negative(self):
        import pytest
        with pytest.raises(ValueError):
            twin_node_id(-1)

    def test_twin_node_id_rejects_too_large(self):
        import pytest
        with pytest.raises(ValueError):
            twin_node_id(TWIN_NODE_OFFSET)


# ──────────────────────────────────────────────────────────────────────
# Mock host for rank_current_topo_at_node_x helpers
# ──────────────────────────────────────────────────────────────────────


class _FakeConstrainedPath:
    def __init__(self, amont, aval):
        self._amont = list(amont)
        self._aval = list(aval)

    def n_amont(self):
        return self._amont

    def n_aval(self):
        return self._aval


class _FakeRedLoops:
    def __init__(self, paths):
        self.Path = paths


class _FakeStructured:
    def __init__(self, amont=(), aval=(), red_loop_paths=()):
        self._cp = _FakeConstrainedPath(amont, aval)
        self._loops = _FakeRedLoops(list(red_loop_paths))

    def get_constrained_path(self):
        return self._cp

    def get_loops(self):
        return self._loops


class _RankHost:
    """Minimal object exposing the helpers under test."""
    rank_current_topo_at_node_x = AlphaDeesp.rank_current_topo_at_node_x
    _pick_interesting_bus_id = AlphaDeesp._pick_interesting_bus_id
    _collect_flows_on_bus = AlphaDeesp._collect_flows_on_bus
    _score_amont = AlphaDeesp._score_amont
    _score_aval = AlphaDeesp._score_aval
    _score_in_red_loop = AlphaDeesp._score_in_red_loop
    _score_not_connected_to_cpath = AlphaDeesp._score_not_connected_to_cpath
    get_bus_id_from_edge = AlphaDeesp.get_bus_id_from_edge
    get_prod_conso_sum = AlphaDeesp.get_prod_conso_sum
    is_connected_to_cpath = AlphaDeesp.is_connected_to_cpath

    def __init__(self, simulator_data, amont=(), aval=(), red_loop_paths=()):
        self.debug = False
        self.simulator_data = simulator_data
        self.g_distribution_graph = _FakeStructured(amont, aval, red_loop_paths)


def _line_to(end_substation_id, flow_value):
    """Build an OriginLine element wired up for the helpers."""
    return OriginLine(busbar_id=0, end_substation_id=end_substation_id,
                      flow_value=[flow_value])


def _line_from(start_substation_id, flow_value):
    return ExtremityLine(busbar_id=0, start_substation_id=start_substation_id,
                         flow_value=[flow_value])


def _build_amont_node_graph():
    """
    Build a tiny graph around a node 1 that sits in "amont" of a constrained
    edge running 1 -> 99 (with a negative flow so it counts as a constrained
    edge for the amont-scoring branch).

    Topology at node 1: two OriginLines to nodes 2 and 99.
    """
    g = nx.MultiDiGraph()
    # node 1 -> node 2 outgoing, positive flow (drains on busbar 0)
    g.add_edge(1, 2, label="5", color="coral")
    # node 1 -> node 99 outgoing, negative flow (connected to cpath)
    g.add_edge(1, 99, label="-3", color="blue")
    # incoming flow from node 3 into node 1 (positive -> in_pos)
    g.add_edge(3, 1, label="4", color="coral")
    # incoming negative flow from node 4
    g.add_edge(4, 1, label="-2", color="coral")
    return g


def _amont_simulator_data():
    return {
        "substations_elements": {
            1: [
                _line_to(end_substation_id=2, flow_value=5),    # busbar 0 in topo
                _line_to(end_substation_id=99, flow_value=-3),  # busbar 0 in topo
                _line_from(start_substation_id=3, flow_value=4),   # busbar 0
                _line_from(start_substation_id=4, flow_value=-2),  # busbar 0
            ]
        }
    }


# ──────────────────────────────────────────────────────────────────────
# Tests for the extracted helpers
# ──────────────────────────────────────────────────────────────────────


class TestCollectFlowsOnBus:

    def test_partitions_in_and_out_flows_by_sign(self):
        host = _RankHost(_amont_simulator_data(), amont=[1])
        g = _build_amont_node_graph()
        label_attrs = nx.get_edge_attributes(g, "label")
        topo_vect = [0, 0, 0, 0]  # all on busbar 0

        flows = host._collect_flows_on_bus(g, 1, 0, topo_vect, label_attrs)

        assert sorted(flows["in_pos"]) == [4.0]
        assert sorted(flows["in_neg"]) == [2.0]
        assert sorted(flows["out_pos"]) == [5.0]
        assert sorted(flows["out_neg"]) == [3.0]  # absolute value of -3


class TestPickInterestingBusId:

    def test_picks_twin_bus_when_connected_to_cpath(self):
        host = _RankHost(_amont_simulator_data(), amont=[1])
        g = _build_amont_node_graph()
        color_attrs = nx.get_edge_attributes(g, "color")
        label_attrs = nx.get_edge_attributes(g, "label")
        topo_vect = [0, 0, 0, 0]

        # The out-edge 1->99 is negative and blue so it's "connected to cpath";
        # the helper must take the other bus id (1 - 0 = 1).
        interesting = host._pick_interesting_bus_id(
            g, node=1, topo_vect=topo_vect, isSingleNode=False,
            is_score_specific_substation=True,
            color_attrs=color_attrs, label_attrs=label_attrs,
            direction="amont")
        assert interesting == 1


class TestScoreAmont:

    def test_score_amont_equals_in_neg_plus_max_pos_plus_injection(self):
        host = _RankHost(_amont_simulator_data(), amont=[1])
        g = _build_amont_node_graph()
        color_attrs = nx.get_edge_attributes(g, "color")
        label_attrs = nx.get_edge_attributes(g, "label")
        topo_vect = [0, 0, 0, 0]

        # With all elements on busbar 0 and interesting_bus_id == 1 (because
        # of the "connected to cpath" twin-node rule), the collected flows on
        # busbar 1 are empty → score == diff_sums == 0.
        score = host._score_amont(
            g, node=1, topo_vect=topo_vect, isSingleNode=False,
            is_score_specific_substation=True,
            color_attrs=color_attrs, label_attrs=label_attrs)
        assert score == 0.0


class TestScoreNotConnectedToCpath:

    def test_score_is_min_imbalance_between_busbars(self):
        """
        Build a node 10 with equal ingoing/outgoing negative flow on busbar 0
        and unbalanced flow on busbar 1; the score must be ``min(score_0, score_1) == 0``.
        """
        g = nx.MultiDiGraph()
        g.add_edge(11, 10, label="-3")  # in neg on bus 0
        g.add_edge(10, 12, label="-3")  # out neg on bus 0 (balanced)
        g.add_edge(13, 10, label="-5")  # in neg on bus 1
        # no outgoing negative flow on bus 1 → imbalance 5

        simulator_data = {
            "substations_elements": {
                10: [
                    _line_from(start_substation_id=11, flow_value=-3),  # bus 0
                    _line_to(end_substation_id=12, flow_value=-3),      # bus 0
                    _line_from(start_substation_id=13, flow_value=-5),  # bus 1
                ]
            }
        }
        host = _RankHost(simulator_data)
        label_attrs = nx.get_edge_attributes(g, "label")
        topo_vect = [0, 0, 1]

        score = host._score_not_connected_to_cpath(g, 10, topo_vect, label_attrs)
        assert score == 0.0


class TestScoreInRedLoop:

    def test_returns_zero_for_single_node_topology(self):
        """When topo_vect has only one busbar value, the helper short-circuits to 0."""
        g = nx.MultiDiGraph()
        g.add_edge(7, 8, label="1", color="coral")
        simulator_data = {
            "substations_elements": {7: [_line_to(end_substation_id=8, flow_value=1)]}
        }
        host = _RankHost(simulator_data)
        color_attrs = nx.get_edge_attributes(g, "color")
        label_attrs = nx.get_edge_attributes(g, "label")

        score = host._score_in_red_loop(g, 7, topo_vect=[0], color_attrs=color_attrs,
                                        label_attrs=label_attrs)
        assert score == 0.0


class TestRankDispatcher:

    def test_dispatcher_routes_to_amont_when_in_amont(self):
        """Integration of the dispatcher itself — verify it reaches _score_amont."""
        host = _RankHost(_amont_simulator_data(), amont=[1])
        g = _build_amont_node_graph()
        score = host.rank_current_topo_at_node_x(
            g, node=1, topo_vect=[0, 0, 0, 0],
            is_score_specific_substation=True)
        # Same as the amont test above — should be 0.0.
        assert score == 0.0

    def test_dispatcher_routes_to_isolated_branch(self):
        """Node not in amont / aval / any red loop path."""
        g = nx.MultiDiGraph()
        g.add_edge(100, 101, label="-1")
        simulator_data = {
            "substations_elements": {
                100: [_line_to(end_substation_id=101, flow_value=-1)]
            }
        }
        host = _RankHost(simulator_data, amont=[], aval=[], red_loop_paths=[])
        score = host.rank_current_topo_at_node_x(
            g, node=100, topo_vect=[0],
            is_score_specific_substation=True)
        # Nothing is connected to the (empty) constrained path or red loops;
        # the isolated branch returns min(|in-out|, |in-out|) over the two
        # busbars — here only bus 0 has a negative out edge (1), bus 1 empty.
        assert score == 0.0


# ──────────────────────────────────────────────────────────────────────
# Tests for apply_new_topo_to_graph helpers
# ──────────────────────────────────────────────────────────────────────


class _ApplyHost:
    _compute_prod_load_per_bus = staticmethod(AlphaDeesp._compute_prod_load_per_bus)
    _classify_bus = staticmethod(AlphaDeesp._classify_bus)
    _add_bus_nodes = AlphaDeesp._add_bus_nodes
    _reconnect_bus_edges = AlphaDeesp._reconnect_bus_edges


class TestComputeProdLoadPerBus:

    def test_accumulates_per_busbar(self):
        elements = [
            Production(busbar_id=0, value=5.0),
            Production(busbar_id=1, value=3.0),
            Consumption(busbar_id=0, value=2.0),
            Consumption(busbar_id=1, value=1.5),
            Consumption(busbar_id=1, value=0.5),
        ]
        prod, load = AlphaDeesp._compute_prod_load_per_bus(elements)
        assert prod == {0: 5.0, 1: 3.0}
        assert load == {0: 2.0, 1: 2.0}

    def test_ignores_lines(self):
        elements = [
            _line_to(end_substation_id=9, flow_value=10),
            Production(busbar_id=0, value=4.0),
        ]
        prod, load = AlphaDeesp._compute_prod_load_per_bus(elements)
        assert prod == {0: 4.0}
        assert load == {}


class TestClassifyBus:

    def test_prod_beats_load_when_positive(self):
        kind, value = AlphaDeesp._classify_bus(0, {0: 5.0}, {0: 3.0})
        assert kind == "prod"
        assert value == 2.0

    def test_load_wins_when_negative_balance(self):
        kind, value = AlphaDeesp._classify_bus(1, {1: 2.0}, {1: 6.0})
        assert kind == "load"
        assert value == -4.0

    def test_only_prod(self):
        kind, value = AlphaDeesp._classify_bus(0, {0: 5.0}, {})
        assert kind == "prod"
        assert value == 5.0

    def test_only_load(self):
        kind, value = AlphaDeesp._classify_bus(0, {}, {0: 2.0})
        assert kind == "load"
        assert value == 2.0

    def test_empty_returns_none(self):
        kind, value = AlphaDeesp._classify_bus(0, {}, {})
        assert kind is None
        assert value == 0


class TestAddBusNodes:

    def test_adds_twin_node_for_busbar1(self):
        host = _ApplyHost()
        g = nx.MultiDiGraph()
        prod = {0: 3.0}
        load = {1: 4.0}
        host._add_bus_nodes(g, bus_ids={0, 1}, prod=prod, load=load,
                            node_to_change=5, new_node_id=twin_node_id(5))
        assert 5 in g.nodes
        assert twin_node_id(5) in g.nodes
        assert g.nodes[5]["prod_or_load"] == "prod"
        assert g.nodes[twin_node_id(5)]["prod_or_load"] == "load"

    def test_white_node_when_neither_prod_nor_load(self):
        host = _ApplyHost()
        g = nx.MultiDiGraph()
        host._add_bus_nodes(g, bus_ids={0}, prod={}, load={},
                            node_to_change=5, new_node_id=twin_node_id(5))
        assert g.nodes[5]["fillcolor"] == "#ffffff"
