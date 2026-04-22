"""TopologyScorerMixin: topology-ranking score helpers for AlphaDeesp.

Extracted from ``alphaDeesp/core/alphadeesp.py`` to keep per-file LOC and
average cyclomatic complexity within A-grade bounds.

The mixin assumes the concrete class provides:
    self.g                   — the overflow MultiDiGraph
    self.debug               — bool debug flag
    self.g_distribution_graph — Structured_Overload_Distribution_Graph
    self.simulator_data      — {"substations_elements": {node: [element, ...]}}
"""

import logging
from math import fabs
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from alphaDeesp.core.elements import Consumption, ExtremityLine, OriginLine, Production

logger = logging.getLogger(__name__)


class TopologyScorerMixin:
    """Scoring helpers for topology ranking; mixed into AlphaDeesp."""

    def rank_current_topo_at_node_x(
        self,
        graph: nx.MultiDiGraph,
        node: int,
        isSingleNode: bool = False,
        topo_vect: Optional[List[int]] = None,
        is_score_specific_substation: bool = True,
    ) -> Any:
        """Dispatch topology scoring based on node location relative to constrained path."""
        if topo_vect is None:
            topo_vect = [0, 0, 1, 1, 1]

        color_attrs = nx.get_edge_attributes(graph, "color")
        label_attrs = nx.get_edge_attributes(graph, "label")

        constrained_path = self.g_distribution_graph.get_constrained_path()
        red_loops = self.g_distribution_graph.get_loops()

        if node in constrained_path.n_amont():
            return self._score_amont(
                graph, node, topo_vect, isSingleNode,
                is_score_specific_substation, color_attrs, label_attrs)

        if node in constrained_path.n_aval():
            return self._score_aval(
                graph, node, topo_vect, isSingleNode,
                is_score_specific_substation, color_attrs, label_attrs)

        red_loop_nodes = {n for loop_nodes in red_loops.Path for n in loop_nodes}
        if node in red_loop_nodes:
            return self._score_in_red_loop(
                graph, node, topo_vect, color_attrs, label_attrs)

        return self._score_not_connected_to_cpath(graph, node, topo_vect, label_attrs)

    def _pick_interesting_bus_id(
        self,
        graph: nx.MultiDiGraph,
        node: int,
        topo_vect: List[int],
        isSingleNode: bool,
        is_score_specific_substation: bool,
        color_attrs: Dict[Any, Any],
        label_attrs: Dict[Any, Any],
        direction: str,
    ) -> int:
        """Return the bus id to score for amont/aval branches.

        When *is_score_specific_substation* is True, find an out/in-edge
        connected to the constrained path and pick the opposite bus (twin
        node). Otherwise pick the bus with the largest negative flow.
        """
        get_edges = graph.out_edges if direction == "amont" else graph.in_edges
        neg_edges = graph.in_edges if direction == "amont" else graph.out_edges

        if is_score_specific_substation:
            for edge in get_edges(node, keys=True):
                if self.is_connected_to_cpath(color_attrs, label_attrs, node, edge, isSingleNode):
                    return abs(self.get_bus_id_from_edge(node, edge, topo_vect) - 1)
            return 0

        caps_bus0 = [float(label_attrs[e]) for e in neg_edges(node, keys=True)
                     if self.get_bus_id_from_edge(node, e, topo_vect) == 0]
        caps_bus1 = [float(label_attrs[e]) for e in neg_edges(node, keys=True)
                     if self.get_bus_id_from_edge(node, e, topo_vect) == 1]
        neg0 = fabs(sum(x for x in caps_bus0 if x < 0))
        neg1 = fabs(sum(x for x in caps_bus1 if x < 0))
        return 1 if neg1 > neg0 else 0

    def _collect_flows_on_bus(
        self,
        graph: nx.MultiDiGraph,
        node: int,
        bus_id: int,
        topo_vect: List[int],
        label_attrs: Dict[Any, Any],
    ) -> Dict[str, List[float]]:
        """Partition in/out flows at *node* on *bus_id* into four sign buckets."""
        in_pos, in_neg, out_pos, out_neg = [], [], [], []

        for edge in graph.in_edges(node, keys=True):
            if self.get_bus_id_from_edge(node, edge, topo_vect) != bus_id:
                continue
            value = float(label_attrs[edge])
            if value < 0:
                in_neg.append(fabs(value))
            else:
                in_pos.append(value)

        for edge in graph.out_edges(node, keys=True):
            if self.get_bus_id_from_edge(node, edge, topo_vect) != bus_id:
                continue
            value = float(label_attrs[edge])
            if value > 0:
                out_pos.append(value)
            else:
                out_neg.append(fabs(value))

        return {"in_pos": in_pos, "in_neg": in_neg, "out_pos": out_pos, "out_neg": out_neg}

    def _score_amont(
        self,
        graph: nx.MultiDiGraph,
        node: int,
        topo_vect: List[int],
        isSingleNode: bool,
        is_score_specific_substation: bool,
        color_attrs: Dict[Any, Any],
        label_attrs: Dict[Any, Any],
    ) -> Any:
        """Score a node upstream (amont) of the constrained path."""
        if self.debug:
            logger.debug("||||||| node [%s] is in_Amont of constrained_edge", node)

        bus_id = self._pick_interesting_bus_id(
            graph, node, topo_vect, isSingleNode, is_score_specific_substation,
            color_attrs, label_attrs, direction="amont")
        flows = self._collect_flows_on_bus(graph, node, bus_id, topo_vect, label_attrs)
        diff_sums = self.get_prod_conso_sum(node, bus_id, topo_vect)
        max_pos = max(sum(flows["out_pos"]), sum(flows["in_pos"]))

        if is_score_specific_substation:
            final_score = np.around(sum(flows["in_neg"]) + max_pos + diff_sums, decimals=2)
        else:
            final_score = np.around(
                sum(flows["in_neg"]) - np.around(sum(flows["out_neg"])) + sum(flows["out_pos"]),
                decimals=2)

        if self.debug:
            logger.debug("AMONT diff=%s max_pos=%s in_neg=%s score=%s",
                         diff_sums, max_pos, flows["in_neg"], final_score)
        return final_score

    def _score_aval(
        self,
        graph: nx.MultiDiGraph,
        node: int,
        topo_vect: List[int],
        isSingleNode: bool,
        is_score_specific_substation: bool,
        color_attrs: Dict[Any, Any],
        label_attrs: Dict[Any, Any],
    ) -> Any:
        """Score a node downstream (aval) of the constrained path."""
        if self.debug:
            logger.debug("||||||| node [%s] is in_Aval of constrained_edge", node)

        bus_id = self._pick_interesting_bus_id(
            graph, node, topo_vect, isSingleNode, is_score_specific_substation,
            color_attrs, label_attrs, direction="aval")
        flows = self._collect_flows_on_bus(graph, node, bus_id, topo_vect, label_attrs)
        max_pos = max(sum(flows["out_pos"]), sum(flows["in_pos"]))
        diff_sums = -self.get_prod_conso_sum(node, bus_id, topo_vect)

        if is_score_specific_substation:
            final_score = np.around(sum(flows["out_neg"]) + max_pos + diff_sums, decimals=2)
        else:
            final_score = np.around(
                sum(flows["out_neg"]) - np.around(sum(flows["in_neg"])) + sum(flows["in_pos"]),
                decimals=2)

        if self.debug:
            logger.debug("AVAL diff=%s score=%s", diff_sums, final_score)
        return final_score

    def _score_in_red_loop(
        self,
        graph: nx.MultiDiGraph,
        node: int,
        topo_vect: List[int],
        color_attrs: Dict[Any, Any],
        label_attrs: Dict[Any, Any],
    ) -> Any:
        """Score a node belonging to a red loop path."""
        if not (1 in topo_vect and 0 in topo_vect):
            return 0.0

        input_red_delta_by_bus = {0: 0.0, 1: 0.0}
        for edge in graph.in_edges(node, keys=True):
            if color_attrs[edge] != "coral":
                continue
            edge_bus_id = self.get_bus_id_from_edge(node, edge, topo_vect)
            if edge_bus_id in (0, 1):
                input_red_delta_by_bus[edge_bus_id] += float(label_attrs[edge])

        biggest_bus = 1 if input_red_delta_by_bus[1] >= input_red_delta_by_bus[0] else 0
        input_red_delta = input_red_delta_by_bus[biggest_bus]

        output_red_delta = 0.0
        for edge in graph.out_edges(node, keys=True):
            if self.get_bus_id_from_edge(node, edge, topo_vect) != biggest_bus:
                continue
            if color_attrs[edge] == "coral":
                output_red_delta += float(label_attrs[edge])

        injection = -self.get_prod_conso_sum(node, biggest_bus, topo_vect)
        return np.around(min(output_red_delta, input_red_delta) + injection, decimals=2)

    def _score_not_connected_to_cpath(
        self,
        graph: nx.MultiDiGraph,
        node: int,
        topo_vect: List[int],
        label_attrs: Dict[Any, Any],
    ) -> Any:
        """Score a node not connected to the constrained path or any red loop."""
        logger.debug("||||||| node [%s] not connected to constrained_edge path.", node)

        in_neg_by_bus: Dict[int, List[float]] = {0: [], 1: []}
        out_neg_by_bus: Dict[int, List[float]] = {0: [], 1: []}

        for edge in graph.in_edges(node, keys=True):
            value = float(label_attrs[edge])
            if value < 0:
                bus_id = self.get_bus_id_from_edge(node, edge, topo_vect)
                if bus_id in in_neg_by_bus:
                    in_neg_by_bus[bus_id].append(fabs(value))

        for edge in graph.out_edges(node, keys=True):
            value = float(label_attrs[edge])
            if value < 0:
                bus_id = self.get_bus_id_from_edge(node, edge, topo_vect)
                if bus_id in out_neg_by_bus:
                    out_neg_by_bus[bus_id].append(fabs(value))

        score_1 = fabs(sum(in_neg_by_bus[0]) - sum(out_neg_by_bus[0]))
        score_2 = fabs(sum(in_neg_by_bus[1]) - sum(out_neg_by_bus[1]))
        final_score = np.around(min(score_1, score_2), decimals=2)

        if self.debug:
            logger.debug("in_neg=%s out_neg=%s score=%s", in_neg_by_bus, out_neg_by_bus, final_score)
        return final_score

    def get_prod_conso_sum(self, node: int, interesting_bus_id: int, topo_vect: List[int]) -> float:
        """Sum of production minus consumption at *node* on *interesting_bus_id*."""
        total = 0
        elements = self.simulator_data["substations_elements"][node]
        for element, bus_id in zip(elements, topo_vect):
            if bus_id == interesting_bus_id:
                if isinstance(element, Consumption):
                    total -= element.value
                elif isinstance(element, Production):
                    total += element.value
        return total

    def get_bus_id_from_edge(
        self, node: int, edge: Tuple[Any, Any, Any], topo_vect: List[int]
    ) -> Optional[int]:
        """Return the busbar id at *node* on which *edge* is connected."""
        target_extremity = edge[0] if edge[0] != node else edge[1]
        elements = self.simulator_data["substations_elements"][node]
        paralel_edge_count_id = edge[2]
        paralel_edge_counter = 0

        for element, bus_id in zip(elements, topo_vect):
            if isinstance(element, OriginLine):
                if element.end_substation_id == target_extremity:
                    if paralel_edge_counter == paralel_edge_count_id:
                        return bus_id
                    paralel_edge_counter += 1
            elif isinstance(element, ExtremityLine):
                if element.start_substation_id == target_extremity:
                    if paralel_edge_counter == paralel_edge_count_id:
                        return bus_id
                    paralel_edge_counter += 1
        return None

    def is_connected_to_cpath(
        self,
        all_edges_color_attributes: Dict[Any, Any],
        all_edges_xlabel_attributes: Dict[Any, Any],
        node: int,
        edge: Tuple[Any, Any, Any],
        isSingleNode: bool,
    ) -> bool:
        """True when *edge* is a negative-valued blue/black edge (connected to cpath)."""
        edge_color = all_edges_color_attributes[edge]
        edge_value = all_edges_xlabel_attributes[edge]
        result = (
            float(edge_value) < 0
            and edge_color in ("blue", "black")
            and not isSingleNode
        )
        if result and self.debug:
            logger.debug("Node [%s] → twin node selected (edge connected to cpath).", node)
        return result
