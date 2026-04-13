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

        # check that line extemity does not have only load or productions: otherwise there is either node merging to do or nothing else

        #Compute the overload distribution graph (constrained path, loops, hubs)
        self.g_distribution_graph=Structured_Overload_Distribution_Graph(self.g)

        # this function takes the dataFrame self.red_loops and adds the min cut_values to it.
        self.rank_red_loops()

        self.rankedLoopBuses = self.rank_loop_buses(self.g, self.df)

        # here we classify nodes into 4 categories
        self.structured_topological_actions = self.identify_routing_buses()  # it is a dict
        # print("#########################################################################")
        # print("structured_top_actions =", self.structured_topological_actions)
        # print("#########################################################################")

        self.ranked_combinations = self.compute_best_topologies()

        # for ranked_comb in self.ranked_combinations:
        # print("---------------------------")
        # print(ranked_comb)

        if self.boolean_dump_data_to_file:
            self.ranked_combinations[0].to_csv("./result_ranked_combinations.csv", index=True)

        # The order to get structures is:
        # constrained_path
        # hubs
        # parallel loops and loops

        # self.simulate_network_change(self.ranked_combinations)

    def load2(self, observation: Any, line_to_cut: int) -> None:
        """@:arg observation: a pypownet observation,
        line_to_cut: line to cut for overload graph"""

    def simulate_network_change(self, ranked_combinations: Any) -> None:
        """This function takes a dataFrame ranked_combinations and computes new changes with Pypownet"""
        pass

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
            if (len(all_combinations) != 0):
                ranked_combinations = self.rank_topologies(all_combinations, self.g, node)
                # print(ranked_combinations)

                # best_topologies = best_topologies.append(ranked_combinations)
                # pd.concat([best_topologies, *ranked_combinations])

                # print("\n##############################################################################")
                # print("##########............BEST_TOPOLOGIES COMPUTED............####################")
                # print("##############################################################################")

                # best_topologies = self.clean_and_sort_best_topologies(best_topologies)
                best_topologies = self.clean_and_sort_best_topologies(ranked_combinations)
                res_container.append(best_topologies)
            # # print(best_topologies)

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

        # ## check that current topology is not in this list
        # TO DO: manage the fact that a substation can already be in 2 nodes. How do you get the node configuration and everything?
        node_configuration_elements = self.simulator_data["substations_elements"][node]
        n_elements = len(node_configuration_elements)
        node_configuration = [node_configuration_elements[i].busbar_id for i in range(n_elements)]
        node_configuration_sym = [0 if node_configuration[i] == 1 else 1 for i in range(n_elements)]

        # print("Inside compute_all_comb : for node [{}], node_configuration = {}".format(node, node_configuration))
        if n_elements == 0 or n_elements == 1:
            raise ValueError("Cannot generate combinations out of a configuration with len = 1 or 2")
        elif n_elements == 2:
            return [(1, 1), (0, 0)]
        else:
            l = [0, 1]
            allcomb = [list(i) for i in itertools.product(l, repeat=n_elements)]

            #we also want to filter combs that only have prods and loads connected to a node
            nProds_loads=0
            for element in node_configuration_elements:
                if isinstance(element, Production) or isinstance(element, Consumption):
                    nProds_loads+=1
                else:
                    break

            # we get rid of symetrical topologies by fixing the first element to busbar 0.
            # ideally if first element is not connected, we should fix the first connected element
            # Also a node should also have 2 elements connected to it, we filter that as well
            uniqueComb = [allcomb[i] for i in range(len(allcomb)) if self.legal_comb(allcomb[i],nProds_loads,n_elements,node_configuration,node_configuration_sym)]
                          #if (allcomb[i][0] == 0) & (allcomb[i] != node_configuration) & (allcomb[i] != node_configuration_sym) &
                          #(np.sum(allcomb[i]) != 1) & (np.sum(allcomb[i]) != n_elements - 1) &
                          #]  # we get rid of symetrical topologies by fixing the first element to busbar 0. ideally if first element is not connected, we should fix the first connected element


        return uniqueComb

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
        """==> ultimate goal: This function returns a DF with topologies ranked
        for the moment:
            takes a topo,
            apply it to graph,
            compute score,
            add to df
            next
        """

        ranked_combinations_columns = ["score", "topology", "node"]
        scores_data = [["XX", ["X", "X", "X"], "X"]]

        # ===========================================
        # print("\nNOEUD "+str(node_to_change))
        # print("number of topo to test "+str(len(all_combinations)))

        for i, topo in enumerate(all_combinations):
            # WARNING the internal_repr is not used further in the code. It is not up to date with the new_graph.
            # Only the original one.
            isSingleNodeTopo = ((np.all(np.array(topo) == 0)) or (np.all(np.array(topo) == 1)))
            score = self.rank_current_topo_at_node_x(self.g, node_to_change, isSingleNodeTopo, topo)
            if self.debug:
                logger.debug("** RESULTS ** new topo [%s] on node [%s] has a score: [%s]", topo, node_to_change, score)
            scores_data.append([score, topo, node_to_change])

            # =======================================================
            # if i % 10000 ==0:
              #   print("Done: "+str(i)+" topos")

        ranked_combinations = pd.DataFrame(columns = ranked_combinations_columns, data = scores_data)

        # =================================================
        # ranked_combinations.to_csv("NEW_rank_topologies_l2rpn_2019_node_"+str(node_to_change)+".csv", sep = ';', decimal = ',')
        return ranked_combinations

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
        # creates a DATAFRAME and sort it, returns the sorted hubs
        # print("================= sort_hubs =================")
        if hubs:
            df = pd.DataFrame()
            df["hubs"] = hubs

            # now for each node in hubs, get the max abs(ingoing or outgoing) flow
            flows = []

            for node in hubs:
                flow_compute_ingoing = []
                flow_compute_outgoing = []

                # print("node = ", node)

                for i, row in self.df.iterrows():
                    if row["idx_or"] == node:
                        flow_compute_outgoing.append(fabs(row["delta_flows"]))

                    if row["idx_ex"] == node:
                        flow_compute_ingoing.append(fabs(row["delta_flows"]))

                max_result = max(sum(flow_compute_ingoing), sum(flow_compute_outgoing))
                # print("all flows = ", max_result)
                flows.append(max_result)

            df["max_flows"] = flows
            df.sort_values("max_flows", ascending=False, inplace=True)
            # print(df)

            return df

        # else:
        #   raise ValueError("There are no hubs")

    def identify_routing_buses(self) -> Dict[int, Any]:
        """Categories 1 to 4
        1. Hubs
        2. all nodes that belong to c_path
        3. On //path
        4. Over Da on c_path"""

        # ALGO
        # get all nodes from c_path, loops, //paths, {set of all those nodes}
        # for nodes in interesting_nodes:
        #   classify to category 1, 2, 3, 4.
        hubs= self.g_distribution_graph.get_hubs()
        df_sorted_hubs = self.sort_hubs(hubs)
        if df_sorted_hubs is None:
            return {}
        else:
            category1 = list(df_sorted_hubs["hubs"])

            constrained_path = self.g_distribution_graph.get_constrained_path()
            set_category2 = set(constrained_path.full_n_constrained_path()) - set(category1)

            sort_redLoopBuses = sorted(self.rankedLoopBuses.items(), key=lambda x: x[1], reverse=True)
            category3 = list(set([sort_redLoopBuses[i][0] for i in range(len(sort_redLoopBuses))]))  # set()  # @TODO
            set_category4 = set(constrained_path.n_aval()) - (set(category1) | set_category2 | set(category3))

            d = {1: category1, 2: set_category2, 3: category3, 4: set_category4}
        return d

    def rank_loop_buses(self, graph: nx.MultiDiGraph, df_initial_flows: pd.DataFrame) -> Dict[Any, float]:
        # self.g => overflow graph
        all_edges_color_attributes = nx.get_edge_attributes(graph, "color")  # dict[edge]
        all_edges_xlabel_attributes = nx.get_edge_attributes(graph, "label")

        Strength_Bus_dic = {}
        red_loops = self.g_distribution_graph.get_loops()
        for index, loop in red_loops.iterrows():
            # for loop in self.red_loops:
            for bus in loop.Path:
                if (bus != loop.Source) & (bus != loop.Target):
                    nNode = 1
                    # TO DO:we should know if bus is 1 or 2 nodes
                    if (nNode == 1):
                        strength_measure = 0  # it will be the product of production and in_flows
                        sumInRedDeltaFlows = 0
                        sumInFlowsNotRed = 0

                        LocalProduction = 0
                        # LocalProduction = float(all_nodes_value_attributes[node])  # it is actually the sum of injections. To get only production, need to use self.simulator_data["substations_elements"][node]

                        for element in self.simulator_data["substations_elements"][bus]:
                            if isinstance(element, Production):
                                LocalProduction += (element.value)

                        for edge in self.g.in_edges(bus,keys=True):
                            edge_deltaflow_value = float(all_edges_xlabel_attributes[edge])
                            edge_color = all_edges_color_attributes[edge]
                            if (edge_color == "coral"):
                                sumInRedDeltaFlows += edge_deltaflow_value
                            else:  # we need to retrieve the initial flow from df_initial_flows
                                source, target,idx = edge

                                otherBus = source
                                if otherBus == bus:
                                    otherBus = target

                                nodes_or = df_initial_flows["idx_or"]
                                nodes_ex = df_initial_flows["idx_ex"]
                                # indexEdge_inDf=-1

                                for i in range(len(nodes_or)):
                                    flowValue = df_initial_flows["init_flows"][i]
                                    if ((flowValue >= 0) & (nodes_or[i] == otherBus) & (nodes_ex[i] == bus)):  # we are only looking for input flows
                                        sumInFlowsNotRed += np.abs(flowValue)
                                        break
                                    elif ((flowValue <= 0) & (nodes_or[i] == bus) & (nodes_ex[i] == otherBus)):
                                        sumInFlowsNotRed += np.abs(flowValue)
                                        break

                        sumInFlowsNotRed += LocalProduction
                        strength_measure = sumInFlowsNotRed * sumInRedDeltaFlows
                        Strength_Bus_dic[bus] = strength_measure
        return Strength_Bus_dic

    def rank_red_loops(self) -> None:
        cut_values = []
        cut_sets = []  # contains the edges that ended up having the minimum cut_values
        g_red_DiGraph=self.to_DiGraph(self.g_distribution_graph.g_only_red_components)#self.g_only_red_components)#necessary to be able to compute minimum_cut

        red_loops = self.g_distribution_graph.get_loops()
        for i, row in red_loops.iterrows():#red_loops.iterrows():
            source = row["Source"]
            target = row["Target"]

            # print("=============== source: {}, target: {}".format(source, target))

            #cut_value, partition = nx.minimum_cut(self.g_only_red_components, source, target)
            cut_value, partition = nx.minimum_cut(g_red_DiGraph, source, target)
            reachable, non_reachable = partition
            # print("cut_value: {}, partition: {}".format(cut_value, partition))

            # info from doc - ‘partition’ here is a tuple with the two sets of nodes that define the minimum cut.
            # You can compute the cut set of edges that induce the minimum cut as follows:
            cutset = set()
            for u, nbrs in ((n, self.g_distribution_graph.g_only_red_components[n]) for n in reachable):
                cutset.update((u, v) for v in nbrs if v in non_reachable)
            # print("sorted(cutset) = ", sorted(cutset))

            cut_values.append(cut_value)
            cut_sets.append(list(cutset)[0])

        # print("cut_values = ", cut_values)

        red_loops["min_cut_values"] = cut_values
        red_loops["min_cut_edges"] = cut_sets
        # print("======================= cut_values added =======================")
        # print(self.red_loops)

    # create weighted digraph from MultiDiGraph
    def to_DiGraph(self, gM: nx.MultiDiGraph) -> nx.DiGraph:
        G = nx.DiGraph()
        for u, v,idx, data in gM.edges(data=True,keys=True):
            w = data['capacity'] if 'capacity' in data else 1.0
            if G.has_edge(u, v):
                G[u][v]['capacity'] += w
            else:
                G.add_edge(u, v, capacity=w)
        return G

    def compute_meaningful_structures(self) -> None:
        self.data["constrained_path"] = self.get_constrained_path()

    def get_adjacency_matrix(self, g: nx.MultiDiGraph) -> None:
        logger.debug("adjacency matrix full = ")
        for line in nx.generate_adjlist(g):
            logger.debug("%s", line)

    def get_loop_paths(self) -> None:
        pass

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

    def isAntenna(self) -> None:
        pass

    def write_g(self, g: nx.MultiDiGraph) -> None:
        """This saves file g"""
        nx.write_edgelist(self.g, "./alphaDeesp/tmp/save.graph")
        # print("file saved")
        pass

    def read_g(self) -> None:
        pass


class AlphaDeesp_warmStart(AlphaDeesp):
    def __init__(self, g: nx.MultiDiGraph, g_distribution_graph: Any, simulator_data: Optional[Dict[str, Any]] = None, debug: bool = False) -> None:
        # used for postprocessing
        self.bag_of_graphs = {}
        self.debug = debug
        self.boolean_dump_data_to_file = False

        # data from Simulator Class
        self.g=g
        #Compute the overload distribution graph (constrained path, loops, hubs)
        self.g_distribution_graph=g_distribution_graph
        self.simulator_data=simulator_data