# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of ExpertOp4Grid, an expert system approach to solve flow congestions in power grids

""" This file is the main file for the Expert Agent called AlphaDeesp """
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import pandas as pd
import itertools
import numpy as np

from alphaDeesp.core.graphsAndPaths import Structured_Overload_Distribution_Graph
from alphaDeesp.core.elements import (
    Consumption,
    ExtremityLine,
    OriginLine,
    Production,
)
from alphaDeesp.core.twin_nodes import twin_node_id
from math import fabs

logger = logging.getLogger(__name__)


class AlphaDeesp:  # AKA SOLVER
    def __init__(self, _g: nx.MultiDiGraph, df_of_g: pd.DataFrame, simulator_data: Optional[Dict[str, Any]] = None, substation_in_cooldown: Optional[List[int]] = None, debug: bool = False) -> None:
        # used for postprocessing
        self.bag_of_graphs = {}
        self.debug = debug
        self.boolean_dump_data_to_file = False

        # data from Simulator Class
        self.simulator_data = simulator_data

        self.data = {}
        self.g = _g  # here the g is the overflow graph
        self.df = df_of_g
        self.initial_graph = self.g.copy()
        # we cannot play with those substations so no need to compute simulations
        self.substation_in_cooldown = substation_in_cooldown if substation_in_cooldown is not None else []

        #Compute the overload distribution graph (constrained path, loops, hubs)
        self.g_distribution_graph = Structured_Overload_Distribution_Graph(self.g)

        # adds min cut_values to self.g_distribution_graph.red_loops
        self.rank_red_loops()
        self.rankedLoopBuses = self.rank_loop_buses(self.g, self.df)

        # classify nodes into 4 categories (hubs / c_path / loop / aval)
        self.structured_topological_actions = self.identify_routing_buses()
        self.ranked_combinations = self.compute_best_topologies()

        if self.boolean_dump_data_to_file:
            self.ranked_combinations[0].to_csv("./result_ranked_combinations.csv", index=True)

    def get_ranked_combinations(self) -> List[pd.DataFrame]:
        return self.ranked_combinations

    def compute_best_topologies(self) -> List[pd.DataFrame]:
        """
        inputs: selected_ranked_nodes (after having identified routing buses, ie 4 categories)
        :return: pd.DataFrame containing all topologies scored.
        """
        selected_ranked_nodes = []
        for key_indice in list(self.structured_topological_actions.keys()):
            res = self.structured_topological_actions[key_indice]
            if res is not None:
                for elem in res:
                    selected_ranked_nodes.append(elem)

        res_container = []
        for node in selected_ranked_nodes:
            if node in self.substation_in_cooldown:
                logger.info("substation %s is in cooldown and no action can be performed on it for now", node)
                continue

            all_combinations = self.compute_all_combinations(node)
            if len(all_combinations) != 0:
                ranked_combinations = self.rank_topologies(all_combinations, self.g, node)
                best_topologies = self.clean_and_sort_best_topologies(ranked_combinations)
                res_container.append(best_topologies)

        return res_container

    def clean_and_sort_best_topologies(self, best_topologies: pd.DataFrame) -> pd.DataFrame:
        """This function cleans the Dataframe best_topologies;
        it deletes rows with XX, and sorts the Dataframe. In order to achieve this we have to set_index first."""
        best_topologies = best_topologies.set_index("score")
        best_topologies = best_topologies.drop("XX", axis=0)
        best_topologies = best_topologies.sort_values("score", ascending=False)

        return best_topologies

    def compute_all_combinations(self, node: int) -> List[Any]:
        """ Given a node, returns all possible combinations of a node configuration.
        ex: [001], [010], [100], [101], [011]... etc..."""
        node_configuration_elements = self.simulator_data["substations_elements"][node]
        n_elements = len(node_configuration_elements)
        node_configuration = [node_configuration_elements[i].busbar_id for i in range(n_elements)]
        node_configuration_sym = [0 if node_configuration[i] == 1 else 1 for i in range(n_elements)]

        if n_elements == 0 or n_elements == 1:
            raise ValueError("Cannot generate combinations out of a configuration with len = 1 or 2")
        if n_elements == 2:
            return [(1, 1), (0, 0)]

        allcomb = [list(i) for i in itertools.product([0, 1], repeat=n_elements)]

        # prods and loads are listed first in the substation; find where they end
        nProds_loads = 0
        for element in node_configuration_elements:
            if isinstance(element, (Production, Consumption)):
                nProds_loads += 1
            else:
                break

        # We break the busbar symmetry by fixing comb[0] == 0 and also drop
        # topologies that would isolate all prods/loads on a single busbar.
        return [c for c in allcomb if self.legal_comb(
            c, nProds_loads, n_elements, node_configuration, node_configuration_sym)]

    def legal_comb(self, comb: List[int], nProd_loads: int, n_elements: int, node_configuration: List[int], node_configuration_sym: List[int]) -> bool:
        sum_comb=np.sum(comb)
        busBar_prods_loads=set(comb[0:nProd_loads])
        busBar_lines = set(comb[nProd_loads:])

        areProdsLoadsIsolated=False
        if(nProd_loads>=2) and (sum_comb != 1) and (sum_comb != n_elements - 1):
            busbar_diff=set(busBar_prods_loads)-set(busBar_lines)
            if(len(busbar_diff)!=0):
                areProdsLoadsIsolated=True

        legal_condition=((comb[0] == 0) & (comb != node_configuration) & (comb != node_configuration_sym) &
        (sum_comb != 1) & (sum_comb != n_elements - 1) & (areProdsLoadsIsolated==False))

        return legal_condition

    def rank_topologies(self, all_combinations: List[Any], graph: nx.MultiDiGraph, node_to_change: int) -> pd.DataFrame:
        """Score every candidate topology at ``node_to_change`` and return a
        DataFrame with columns ``score``, ``topology``, ``node``. The first
        row is a sentinel ``"XX"`` later dropped by
        :meth:`clean_and_sort_best_topologies`."""
        scores_data = [["XX", ["X", "X", "X"], "X"]]

        for topo in all_combinations:
            is_single_node_topo = np.all(np.array(topo) == 0) or np.all(np.array(topo) == 1)
            score = self.rank_current_topo_at_node_x(self.g, node_to_change, is_single_node_topo, topo)
            if self.debug:
                logger.debug("** RESULTS ** new topo [%s] on node [%s] has a score: [%s]", topo, node_to_change, score)
            scores_data.append([score, topo, node_to_change])

        return pd.DataFrame(columns=["score", "topology", "node"], data=scores_data)

    # WARNING: does not work yet when you go back from two nodes to one node at a given substation? Basically one node will be not connected?
    def apply_new_topo_to_graph(self, graph: nx.MultiDiGraph, new_topology: List[int], node_to_change: int) -> Tuple[nx.MultiDiGraph, Dict[Any, Any]]:
        """
        Apply a busbar reassignment to ``graph``.

        The function is decomposed into four small helpers:
          - :meth:`_gather_edge_colors` memoises existing edge colors so they
            survive the node rebuild step;
          - :meth:`_compute_prod_load_per_bus` accumulates production and
            consumption per target busbar;
          - :meth:`_add_bus_nodes` (re)creates the styled graph nodes for the
            original substation and its twin;
          - :meth:`_reconnect_bus_edges` wires every line element back onto
            the right busbar node with the right reported flow.
        """
        if self.debug:
            logger.debug("====================================== apply new topo to graph ======================================")
            logger.debug(" new topology applied = [%s] to node: [%s]", new_topology, node_to_change)
            logger.debug("======================================================================================================")

        bus_ids = set(new_topology)
        assert ((len(bus_ids) != 0) & (len(bus_ids) <= 2))

        internal_repr_dict = dict(self.simulator_data["substations_elements"])
        new_node_id = twin_node_id(node_to_change)

        element_types = self.simulator_data["substations_elements"][node_to_change]
        assert len(element_types) == len(new_topology)

        color_edges = self._gather_edge_colors()

        if 1 in new_topology:  # keeping only busbar 0 would leave the old node in place
            graph.remove_node(node_to_change)

        # Propagate busbar assignments back onto the internal element objects
        for internal_elem, element in zip(internal_repr_dict[node_to_change], new_topology):
            internal_elem.busbar_id = element

        prod, load = self._compute_prod_load_per_bus(internal_repr_dict[node_to_change])

        self._add_bus_nodes(graph, bus_ids, prod, load, node_to_change, new_node_id)
        self._reconnect_bus_edges(
            graph, new_topology, element_types, node_to_change, new_node_id, color_edges)

        name = str(node_to_change) + "_" + "".join(str(e) for e in new_topology)
        self.bag_of_graphs[name] = graph
        return graph, internal_repr_dict

    # ------------------------------------------------------------------
    # Helpers for apply_new_topo_to_graph
    # ------------------------------------------------------------------

    def _gather_edge_colors(self) -> Dict[Tuple[Any, Any, Any], Any]:
        """
        Snapshot the color of every edge of ``self.g`` before the graph is
        mutated. For edges marked as ``swapped`` in ``self.df`` we also store
        the reversed key so lookups survive the direction change.
        """
        color_edges = {}
        for u, v, idx, color in self.g.edges(data="color", keys=True):
            condition = list(self.df.query(
                "idx_or == " + str(u) + " & idx_ex == " + str(v))["swapped"])[0]
            color_edges[(u, v, idx)] = color
            if condition:
                color_edges[(v, u, idx)] = color
        return color_edges

    @staticmethod
    def _compute_prod_load_per_bus(elements: Iterable[Any]) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        Sum production and consumption values per ``busbar_id`` for the
        given iterable of ``elements``. Returns ``(prod, load)`` dicts where
        each key is a busbar id and each value is the absolute total.
        """
        prod, load = {}, {}
        for element in elements:
            bus = element.busbar_id
            if isinstance(element, Production):
                prod[bus] = prod.get(bus, 0) + fabs(element.value)
            elif isinstance(element, Consumption):
                load[bus] = load.get(bus, 0) + fabs(element.value)
        return prod, load

    @staticmethod
    def _classify_bus(bus_id: int, prod: Dict[int, float], load: Dict[int, float]) -> Tuple[Optional[str], float]:
        """
        Return ``(kind, prod_minus_load)`` for ``bus_id`` where ``kind`` is
        ``"prod"``, ``"load"`` or ``None`` (neither production nor load).
        """
        has_prod = bus_id in prod
        has_load = bus_id in load
        if has_prod and has_load:
            diff = prod[bus_id] - load[bus_id]
            return ("prod" if diff > 0 else "load"), diff
        if has_prod:
            return "prod", prod[bus_id]
        if has_load:
            return "load", load[bus_id]
        return None, 0

    def _add_bus_nodes(self, graph: nx.MultiDiGraph, bus_ids: Iterable[int], prod: Dict[int, float], load: Dict[int, float], node_to_change: int, new_node_id: Any) -> None:
        """
        Re-add the (up to two) graph nodes corresponding to the new topology,
        styled by their net prod/load balance.
        """
        for bus_id in bus_ids:
            node_label = new_node_id if bus_id == 1 else node_to_change
            kind, prod_minus_load = self._classify_bus(bus_id, prod, load)

            if kind == "prod":
                graph.add_node(node_label, pin=True, prod_or_load="prod",
                               value=str(prod_minus_load),
                               style="filled", fillcolor="#f30000")
            elif kind == "load":
                graph.add_node(node_label, pin=True, prod_or_load="load",
                               value=str(-prod_minus_load),
                               style="filled", fillcolor="#478fd0")
            else:  # neither prod nor load — white
                graph.add_node(node_label, pin=True, prod_or_load="load",
                               value=str(prod_minus_load),
                               style="filled", fillcolor="#ffffff")

    def _reconnect_bus_edges(self, graph: nx.MultiDiGraph, new_topology: List[int], element_types: List[Any],
                             node_to_change: int, new_node_id: Any, color_edges: Dict[Tuple[Any, Any, Any], Any]) -> None:
        """
        Re-add every line edge between the (possibly split) substation and
        its neighbours, using the memoised ``color_edges`` for styling.
        """
        for element, element_type in zip(new_topology, element_types):
            if not isinstance(element_type, (OriginLine, ExtremityLine)):
                continue

            reported_flow = element_type.flow_value[0]
            penwidth = fabs(reported_flow) / 10 or 0.1

            # Which end of the substation does the edge live on?
            local_node = new_node_id if element == 1 else node_to_change
            if element not in (0, 1):
                raise ValueError("Error element has to belong to either Busbar 1 or 2")

            if isinstance(element_type, OriginLine):
                color = color_edges[(node_to_change, element_type.end_substation_id, 0)]
                graph.add_edge(local_node, element_type.end_substation_id,
                               capacity=float("%.2f" % reported_flow),
                               label="%.2f" % reported_flow,
                               color=color, fontsize=10,
                               penwidth="%.2f" % penwidth)
            else:  # ExtremityLine
                color = color_edges[(element_type.start_substation_id, node_to_change, 0)]
                graph.add_edge(element_type.start_substation_id, local_node,
                               capacity=float("%.2f" % reported_flow),
                               label="%.2f" % reported_flow,
                               color=color, fontsize=10,
                               penwidth="%.2f" % penwidth)

    def rank_current_topo_at_node_x(self, graph: nx.MultiDiGraph, node: int, isSingleNode: bool = False, topo_vect: Optional[List[int]] = None,
                                    is_score_specific_substation: bool = True) -> Any:
        """
        Rank a candidate topology at ``node`` by scoring how much it relieves
        the overloaded constrained path.

        This is the orchestrator: it classifies ``node`` relative to the
        constrained path / red loops and dispatches to one of four branch
        helpers (``_score_amont``, ``_score_aval``, ``_score_in_red_loop``,
        ``_score_not_connected_to_cpath``). The score-computation semantics
        live entirely in those helpers.
        """
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

        return self._score_not_connected_to_cpath(
            graph, node, topo_vect, label_attrs)

    # ------------------------------------------------------------------
    # Helpers for rank_current_topo_at_node_x
    # ------------------------------------------------------------------

    def _pick_interesting_bus_id(self, graph: nx.MultiDiGraph, node: int, topo_vect: List[int], isSingleNode: bool,
                                 is_score_specific_substation: bool,
                                 color_attrs: Dict[Any, Any], label_attrs: Dict[Any, Any], direction: str) -> int:
        """
        Return the bus id on which to score edges for amont/aval branches.

        ``direction`` is ``"amont"`` (pick out-edge connected to cpath) or
        ``"aval"`` (pick in-edge connected to cpath).

        When ``is_score_specific_substation`` is True we look for an edge that
        is *not* connected to the constrained path and take the opposite bus
        id — this is the "twin node" logic. Otherwise we pick the bus that
        carries the largest negative flow (in-edges for amont, out-edges for
        aval) so the comparison is meaningful across substations.
        """
        get_edges = graph.out_edges if direction == "amont" else graph.in_edges
        neg_edges = graph.in_edges if direction == "amont" else graph.out_edges

        if is_score_specific_substation:
            for edge in get_edges(node, keys=True):
                if self.is_connected_to_cpath(
                        color_attrs, label_attrs, node, edge, isSingleNode):
                    return abs(self.get_bus_id_from_edge(node, edge, topo_vect) - 1)
            return 0

        # Pick the bus carrying the largest negative flow on the relevant side
        caps_bus0 = [float(label_attrs[edge]) for edge in neg_edges(node, keys=True)
                     if self.get_bus_id_from_edge(node, edge, topo_vect) == 0]
        caps_bus1 = [float(label_attrs[edge]) for edge in neg_edges(node, keys=True)
                     if self.get_bus_id_from_edge(node, edge, topo_vect) == 1]
        neg0 = fabs(sum(x for x in caps_bus0 if x < 0))
        neg1 = fabs(sum(x for x in caps_bus1 if x < 0))
        return 1 if neg1 > neg0 else 0

    def _collect_flows_on_bus(self, graph: nx.MultiDiGraph, node: int, bus_id: int, topo_vect: List[int], label_attrs: Dict[Any, Any]) -> Dict[str, List[float]]:
        """
        Partition in/out flows incident to ``node`` on ``bus_id`` into the
        four (positive, negative) × (in, out) buckets.

        Returns a dict with keys ``in_pos``, ``in_neg``, ``out_pos``,
        ``out_neg``; each maps to a list of floats. Negative values are
        returned as absolute values (matching the original code).
        """
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

        return {"in_pos": in_pos, "in_neg": in_neg,
                "out_pos": out_pos, "out_neg": out_neg}

    def _score_amont(self, graph: nx.MultiDiGraph, node: int, topo_vect: List[int], isSingleNode: bool,
                     is_score_specific_substation: bool, color_attrs: Dict[Any, Any], label_attrs: Dict[Any, Any]) -> Any:
        """Score a node that sits in "amont" (upstream) of the constrained path."""
        if self.debug:
            logger.debug("||||||||||||||||||||||||||| node [%s] is in_Amont of constrained_edge", node)

        interesting_bus_id = self._pick_interesting_bus_id(
            graph, node, topo_vect, isSingleNode, is_score_specific_substation,
            color_attrs, label_attrs, direction="amont")

        flows = self._collect_flows_on_bus(
            graph, node, interesting_bus_id, topo_vect, label_attrs)

        diff_sums = self.get_prod_conso_sum(node, interesting_bus_id, topo_vect)
        max_pos_in_or_out = max(sum(flows["out_pos"]), sum(flows["in_pos"]))

        # we want to push ingoing negative and production towards the red path
        if is_score_specific_substation:
            final_score = np.around(
                sum(flows["in_neg"]) + max_pos_in_or_out + diff_sums, decimals=2)
        else:
            final_score = np.around(
                sum(flows["in_neg"]) - np.around(sum(flows["out_neg"])) + sum(flows["out_pos"]),
                decimals=2)

        if self.debug:
            logger.debug("AMONT")
            logger.debug("diff_sums = %s", diff_sums)
            logger.debug("type(diff_sums) = %s", type(diff_sums))
            logger.debug("max_pos_in_or_out_flows = %s", max_pos_in_or_out)
            logger.debug("in negative flows = %s", flows["in_neg"])
            logger.debug("out negative flows = %s", flows["out_neg"])
            logger.debug("out positive flow = %s", flows["out_pos"])
            logger.debug("Final score = %s", final_score)

        return final_score

    def _score_aval(self, graph: nx.MultiDiGraph, node: int, topo_vect: List[int], isSingleNode: bool,
                    is_score_specific_substation: bool, color_attrs: Dict[Any, Any], label_attrs: Dict[Any, Any]) -> Any:
        """Score a node that sits in "aval" (downstream) of the constrained path."""
        if self.debug:
            logger.debug("||||||||||||||||||||||||||| node [%s] is in_Aval of constrained_edge", node)

        interesting_bus_id = self._pick_interesting_bus_id(
            graph, node, topo_vect, isSingleNode, is_score_specific_substation,
            color_attrs, label_attrs, direction="aval")

        flows = self._collect_flows_on_bus(
            graph, node, interesting_bus_id, topo_vect, label_attrs)

        max_pos_in_or_out = max(sum(flows["out_pos"]), sum(flows["in_pos"]))

        if self.debug:
            logger.debug("out_negative_flows = %s", flows["out_neg"])
            logger.debug("in_negative_flows = %s", flows["in_neg"])
            logger.debug("out_positive_flows = %s", flows["out_pos"])
            logger.debug("in_positive_flows = %s", flows["in_pos"])
            logger.debug("sum out neg = %s", sum(flows["out_neg"]))
            logger.debug("sum in  neg = %s", sum(flows["in_neg"]))
            logger.debug("sum out pos = %s", sum(flows["out_pos"]))
            logger.debug("sum in  pos = %s", sum(flows["in_pos"]))
            logger.debug("max_pos_in_or_out_flows = %s", max_pos_in_or_out)

        # on the twin node (not connected to cpath) we want to attract loads,
        # not productions, hence the sign flip on diff_sums.
        diff_sums = -self.get_prod_conso_sum(node, interesting_bus_id, topo_vect)
        if is_score_specific_substation:
            final_score = np.around(
                sum(flows["out_neg"]) + max_pos_in_or_out + diff_sums, decimals=2)
        else:
            final_score = np.around(
                sum(flows["out_neg"]) - np.around(sum(flows["in_neg"])) + sum(flows["in_pos"]),
                decimals=2)

        if self.debug:
            logger.debug("AVAL")
            logger.debug("diff_sums = %s", diff_sums)
            logger.debug("Final score = %s", final_score)
            logger.debug("type(final_score) = %s", type(final_score))

        return final_score

    def _score_in_red_loop(self, graph: nx.MultiDiGraph, node: int, topo_vect: List[int], color_attrs: Dict[Any, Any], label_attrs: Dict[Any, Any]) -> Any:
        """
        Score a node that belongs to a red loop path.

        Only meaningful for 2-busbar topologies (``0`` and ``1`` both present
        in ``topo_vect``); returns ``0.0`` otherwise, matching the original
        control flow.
        """
        if not (1 in topo_vect and 0 in topo_vect):
            return 0.0

        # Pick the bus with the biggest ingoing red (coral) delta flow
        input_red_delta_by_bus = {0: 0.0, 1: 0.0}
        for edge in graph.in_edges(node, keys=True):
            if color_attrs[edge] != "coral":
                continue
            edge_bus_id = self.get_bus_id_from_edge(node, edge, topo_vect)
            if edge_bus_id in (0, 1):
                input_red_delta_by_bus[edge_bus_id] += float(label_attrs[edge])

        if input_red_delta_by_bus[1] >= input_red_delta_by_bus[0]:
            biggest_bus = 1
        else:
            biggest_bus = 0
        input_red_delta = input_red_delta_by_bus[biggest_bus]

        # On that bus, sum outgoing red flow (we want flow to transit through it)
        output_red_delta = 0.0
        for edge in graph.out_edges(node, keys=True):
            if self.get_bus_id_from_edge(node, edge, topo_vect) != biggest_bus:
                continue
            if color_attrs[edge] != "coral":
                continue
            output_red_delta += float(label_attrs[edge])

        min_pos_in_or_out = min(output_red_delta, input_red_delta)
        injection = -self.get_prod_conso_sum(node, biggest_bus, topo_vect)
        return np.around(min_pos_in_or_out + injection, decimals=2)

    def _score_not_connected_to_cpath(self, graph: nx.MultiDiGraph, node: int, topo_vect: List[int], label_attrs: Dict[Any, Any]) -> Any:
        """
        Score a node that is not on the constrained path or any red loop.
        The score is the smaller imbalance between the two candidate busbars.
        """
        logger.debug("||||||||||||||||||||||||||| node [%s] is not connected to a path to the constrained_edge.", node)

        in_neg_by_bus = {0: [], 1: []}
        out_neg_by_bus = {0: [], 1: []}

        for edge in graph.in_edges(node, keys=True):
            value = float(label_attrs[edge])
            if value >= 0:
                continue
            bus_id = self.get_bus_id_from_edge(node, edge, topo_vect)
            if bus_id in in_neg_by_bus:
                in_neg_by_bus[bus_id].append(fabs(value))

        for edge in graph.out_edges(node, keys=True):
            value = float(label_attrs[edge])
            if value >= 0:
                continue
            bus_id = self.get_bus_id_from_edge(node, edge, topo_vect)
            if bus_id in out_neg_by_bus:
                out_neg_by_bus[bus_id].append(fabs(value))

        score_1 = fabs(sum(in_neg_by_bus[0]) - sum(out_neg_by_bus[0]))
        score_2 = fabs(sum(in_neg_by_bus[1]) - sum(out_neg_by_bus[1]))
        final_score = np.around(min(score_1, score_2), decimals=2)

        if self.debug:
            logger.debug("in negative flows node 1 = %s", in_neg_by_bus[0])
            logger.debug("out negative flows node 1 = %s", out_neg_by_bus[0])
            logger.debug("in negative flows node 2 = %s", in_neg_by_bus[1])
            logger.debug("out negative flows node 2 = %s", out_neg_by_bus[1])
            logger.debug("Final score = %s", final_score)

        return final_score

    def get_prod_conso_sum(self, node: int, interesting_bus_id: int, topo_vect: List[int]) -> float:
        total = 0
        elements = self.simulator_data["substations_elements"][node]
        for element, bus_id in zip(elements, topo_vect):
            if bus_id == interesting_bus_id:
                if isinstance(element, Consumption):
                    total = total - element.value
                elif isinstance(element, Production):
                    total = total + element.value
        return total

    def get_bus_id_from_edge(self, node: int, edge: Tuple[Any, Any, Any], topo_vect: List[int]) -> Optional[int]:
        """
        Knowing that topo_vect is applied on given node, returns on which bus_id is connected a given edge

        :param node:
        :param edge:
        :param topo_vect:
        :return: an int representing the bus_id on which the edge is connected to node
        """

        # Get edge substation id extremity (the other one than edge)
        target_extremity = edge[0]
        if target_extremity == node:
            target_extremity = edge[1]

        # Get elements connected to the substation (given by "node") - iterate to find whic one corresponds to edge - return corresponding bus_id
        elements = self.simulator_data["substations_elements"][node]
        paralel_edge_count_id = edge[2]
        paralel_edge_counter = 0

        for element, bus_id in zip(elements, topo_vect):
            if isinstance(element, OriginLine):
                if element.end_substation_id == target_extremity:
                    if paralel_edge_counter == paralel_edge_count_id:
                        return bus_id
                    else:  # in that case there are parallel edges between the two nodes, so deal with that
                        paralel_edge_counter += 1
            elif isinstance(element, ExtremityLine):
                if element.start_substation_id == target_extremity:
                    if paralel_edge_counter == paralel_edge_count_id:
                        return bus_id
                    else:  # in that case there are parallel edges between the two nodes, so deal with that
                        paralel_edge_counter += 1

    def is_connected_to_cpath(self, all_edges_color_attributes: Dict[Any, Any], all_edges_xlabel_attributes: Dict[Any, Any], node: int, edge: Tuple[Any, Any, Any], isSingleNode: bool) -> bool:
        edge_color = all_edges_color_attributes[edge]
        edge_value = all_edges_xlabel_attributes[edge]
        # if there is a outgoing negative blue or black edge this means we are connected to cpath.
        # therefore change to bus ID 1
        bool = (float(edge_value) < 0 and (edge_color == "blue" or edge_color == "black") and not isSingleNode)
        if bool and self.debug:
            logger.debug("######################################################")
            logger.debug("Node [%s] is not connected to cpath. Twin node selected...", node)
            logger.debug("######################################################")
        return bool

    def sort_hubs(self, hubs: Optional[List[Any]]) -> Optional[pd.DataFrame]:
        """Return a DataFrame of ``hubs`` sorted by largest absolute incident
        delta-flow (max of sum of ingoing vs outgoing). ``None`` if no hubs."""
        if not hubs:
            return None

        flows = []
        for node in hubs:
            ingoing, outgoing = [], []
            for _, row in self.df.iterrows():
                if row["idx_or"] == node:
                    outgoing.append(fabs(row["delta_flows"]))
                if row["idx_ex"] == node:
                    ingoing.append(fabs(row["delta_flows"]))
            flows.append(max(sum(ingoing), sum(outgoing)))

        df = pd.DataFrame({"hubs": hubs, "max_flows": flows})
        df.sort_values("max_flows", ascending=False, inplace=True)
        return df

    def identify_routing_buses(self) -> Dict[int, Any]:
        """Split interesting nodes into 4 categories:

        1. hubs, sorted by incident delta-flow
        2. other constrained-path nodes
        3. intermediate red-loop buses, sorted by strength measure
        4. aval nodes not already covered above
        """
        hubs = self.g_distribution_graph.get_hubs()
        df_sorted_hubs = self.sort_hubs(hubs)
        if df_sorted_hubs is None:
            return {}

        category1 = list(df_sorted_hubs["hubs"])
        constrained_path = self.g_distribution_graph.get_constrained_path()
        category2 = set(constrained_path.full_n_constrained_path()) - set(category1)

        sorted_loop_buses = sorted(self.rankedLoopBuses.items(), key=lambda x: x[1], reverse=True)
        category3 = list({bus for bus, _ in sorted_loop_buses})
        category4 = set(constrained_path.n_aval()) - (set(category1) | category2 | set(category3))

        return {1: category1, 2: category2, 3: category3, 4: category4}

    def rank_loop_buses(self, graph: nx.MultiDiGraph, df_initial_flows: pd.DataFrame) -> Dict[Any, float]:
        """
        Score each intermediate bus of every red loop by
        ``(non_red_inflow + local_production) * red_inflow_delta``.

        The heavier the upstream feed and the larger the red redispatch
        through the bus, the more "routing-relevant" it is.
        """
        color_attrs = nx.get_edge_attributes(graph, "color")
        label_attrs = nx.get_edge_attributes(graph, "label")

        strength_by_bus: Dict[Any, float] = {}
        red_loops = self.g_distribution_graph.get_loops()
        for _, loop in red_loops.iterrows():
            for bus in loop.Path:
                if bus == loop.Source or bus == loop.Target:
                    continue
                strength_by_bus[bus] = self._bus_loop_strength(
                    bus, df_initial_flows, color_attrs, label_attrs)
        return strength_by_bus

    def _bus_loop_strength(self, bus: Any, df_initial_flows: pd.DataFrame,
                           color_attrs: Dict[Any, Any], label_attrs: Dict[Any, Any]) -> float:
        """Compute the red-loop strength measure for a single intermediate bus."""
        red_delta_in = 0.0
        non_red_in = 0.0
        for edge in self.g.in_edges(bus, keys=True):
            if color_attrs[edge] == "coral":
                red_delta_in += float(label_attrs[edge])
            else:
                other = edge[0] if edge[0] != bus else edge[1]
                non_red_in += self._initial_inflow_between(df_initial_flows, other, bus)
        total_in = non_red_in + self._local_production_at_bus(bus)
        return total_in * red_delta_in

    def _local_production_at_bus(self, bus: Any) -> float:
        """Sum production values attached to ``bus`` (0 if none)."""
        total = 0.0
        for element in self.simulator_data["substations_elements"][bus]:
            if isinstance(element, Production):
                total += element.value
        return total

    @staticmethod
    def _initial_inflow_between(df_initial_flows: pd.DataFrame, source: Any, target: Any) -> float:
        """
        Look up ``|init_flow|`` for the line carrying power into ``target``
        from ``source`` in the initial (pre-cut) flow DataFrame.

        Handles both naming conventions: the line may be stored oriented
        ``source -> target`` with a positive flow, or oriented the other way
        with a negative flow. Returns 0 if no matching row is found.
        """
        nodes_or = df_initial_flows["idx_or"]
        nodes_ex = df_initial_flows["idx_ex"]
        flows = df_initial_flows["init_flows"]
        for i in range(len(nodes_or)):
            flow = flows[i]
            if flow >= 0 and nodes_or[i] == source and nodes_ex[i] == target:
                return float(np.abs(flow))
            if flow <= 0 and nodes_or[i] == target and nodes_ex[i] == source:
                return float(np.abs(flow))
        return 0.0

    def rank_red_loops(self) -> None:
        """Annotate ``self.g_distribution_graph.get_loops()`` in place with
        ``min_cut_values`` and ``min_cut_edges`` for each row, using
        networkx's min-cut on a weighted DiGraph flattening of the coral
        MultiDiGraph."""
        red_only_multi = self.g_distribution_graph.g_only_red_components
        red_digraph = self.to_DiGraph(red_only_multi)

        cut_values: List[Any] = []
        cut_sets: List[Any] = []

        red_loops = self.g_distribution_graph.get_loops()
        for _, row in red_loops.iterrows():
            cut_value, (reachable, non_reachable) = nx.minimum_cut(
                red_digraph, row["Source"], row["Target"])
            cutset = {
                (u, v) for u in reachable for v in red_only_multi[u] if v in non_reachable
            }
            cut_values.append(cut_value)
            cut_sets.append(next(iter(cutset)))

        red_loops["min_cut_values"] = cut_values
        red_loops["min_cut_edges"] = cut_sets

    def to_DiGraph(self, gM: nx.MultiDiGraph) -> nx.DiGraph:
        """Flatten a MultiDiGraph to a DiGraph by summing parallel edge
        capacities (required for ``nx.minimum_cut``)."""
        G = nx.DiGraph()
        for u, v, _, data in gM.edges(data=True, keys=True):
            w = data.get("capacity", 1.0)
            if G.has_edge(u, v):
                G[u][v]["capacity"] += w
            else:
                G.add_edge(u, v, capacity=w)
        return G

    def filter_constrained_path(self, path_to_filter: Any) -> List[Any]:
        """This function gets rid of duplicates, tuples, arrays, and return a single clean ordered array"""
        set_constrained_path = []
        for edge in path_to_filter:
            for node in edge:
                if isinstance(node, tuple):
                    for n in node:
                        if n not in set_constrained_path:
                            set_constrained_path.append(n)
                else:
                    if node not in set_constrained_path:
                        set_constrained_path.append(node)
        return set_constrained_path


class AlphaDeesp_warmStart(AlphaDeesp):
    """Skip the expensive pipeline; trust the caller to supply a pre-built
    overflow graph and distribution graph. Used when caller already holds a
    valid :class:`Structured_Overload_Distribution_Graph` from a previous run."""

    def __init__(self, g: nx.MultiDiGraph, g_distribution_graph: Any, simulator_data: Optional[Dict[str, Any]] = None, debug: bool = False) -> None:
        self.bag_of_graphs = {}
        self.debug = debug
        self.boolean_dump_data_to_file = False
        self.g = g
        self.g_distribution_graph = g_distribution_graph
        self.simulator_data = simulator_data