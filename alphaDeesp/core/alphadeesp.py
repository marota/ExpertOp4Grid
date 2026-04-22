# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of ExpertOp4Grid, an expert system approach to solve flow congestions in power grids

"""AlphaDeesp: main orchestrator for topology-action scoring and ranking."""

import logging
from typing import Any, Dict, List, Optional, Tuple

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
from alphaDeesp.core.topology_scorer import TopologyScorerMixin
from alphaDeesp.core.topo_applicator import TopoApplicatorMixin

logger = logging.getLogger(__name__)


class AlphaDeesp(TopologyScorerMixin, TopoApplicatorMixin):
    """Expert system solver for power-grid overload topological actions."""

    def __init__(
        self,
        _g: nx.MultiDiGraph,
        df_of_g: pd.DataFrame,
        simulator_data: Optional[Dict[str, Any]] = None,
        substation_in_cooldown: Optional[List[int]] = None,
        debug: bool = False,
    ) -> None:
        self.bag_of_graphs: Dict[str, Any] = {}
        self.debug = debug
        self.boolean_dump_data_to_file = False

        self.simulator_data = simulator_data

        self.data: Dict[str, Any] = {}
        self.g = _g
        self.df = df_of_g
        self.initial_graph = self.g.copy()
        self.substation_in_cooldown = substation_in_cooldown if substation_in_cooldown is not None else []

        self.g_distribution_graph = Structured_Overload_Distribution_Graph(self.g)

        self.rank_red_loops()
        self.rankedLoopBuses = self.rank_loop_buses(self.g, self.df)

        self.structured_topological_actions = self.identify_routing_buses()
        self.ranked_combinations = self.compute_best_topologies()

        if self.boolean_dump_data_to_file:
            self.ranked_combinations[0].to_csv("./result_ranked_combinations.csv", index=True)

    def get_ranked_combinations(self) -> List[pd.DataFrame]:
        return self.ranked_combinations

    def compute_best_topologies(self) -> List[pd.DataFrame]:
        """Score every candidate topology and return ranked DataFrames per node."""
        selected_ranked_nodes = []
        for key_indice in list(self.structured_topological_actions.keys()):
            res = self.structured_topological_actions[key_indice]
            if res is not None:
                for elem in res:
                    selected_ranked_nodes.append(elem)

        res_container = []
        for node in selected_ranked_nodes:
            if node in self.substation_in_cooldown:
                logger.info("substation %s is in cooldown — skipping.", node)
                continue

            all_combinations = self.compute_all_combinations(node)
            if len(all_combinations) != 0:
                ranked_combinations = self.rank_topologies(all_combinations, self.g, node)
                best_topologies = self.clean_and_sort_best_topologies(ranked_combinations)
                res_container.append(best_topologies)

        return res_container

    def clean_and_sort_best_topologies(self, best_topologies: pd.DataFrame) -> pd.DataFrame:
        """Drop the sentinel 'XX' row and sort by score descending."""
        best_topologies = best_topologies.set_index("score")
        best_topologies = best_topologies.drop("XX", axis=0)
        best_topologies = best_topologies.sort_values("score", ascending=False)
        return best_topologies

    def compute_all_combinations(self, node: int) -> List[Any]:
        """Return all legal busbar-assignment combinations for *node*."""
        node_configuration_elements = self.simulator_data["substations_elements"][node]
        n_elements = len(node_configuration_elements)
        node_configuration = [node_configuration_elements[i].busbar_id for i in range(n_elements)]
        node_configuration_sym = [0 if node_configuration[i] == 1 else 1 for i in range(n_elements)]

        if n_elements == 0 or n_elements == 1:
            raise ValueError("Cannot generate combinations out of a configuration with len = 1 or 2")
        if n_elements == 2:
            return [(1, 1), (0, 0)]

        allcomb = [list(i) for i in itertools.product([0, 1], repeat=n_elements)]

        nProds_loads = 0
        for element in node_configuration_elements:
            if isinstance(element, (Production, Consumption)):
                nProds_loads += 1
            else:
                break

        return [c for c in allcomb if self.legal_comb(
            c, nProds_loads, n_elements, node_configuration, node_configuration_sym)]

    def legal_comb(
        self,
        comb: List[int],
        nProd_loads: int,
        n_elements: int,
        node_configuration: List[int],
        node_configuration_sym: List[int],
    ) -> bool:
        """Return True when *comb* is a legal busbar assignment."""
        sum_comb = np.sum(comb)
        busBar_prods_loads = set(comb[0:nProd_loads])
        busBar_lines = set(comb[nProd_loads:])

        areProdsLoadsIsolated = False
        if (nProd_loads >= 2) and (sum_comb != 1) and (sum_comb != n_elements - 1):
            busbar_diff = set(busBar_prods_loads) - set(busBar_lines)
            if len(busbar_diff) != 0:
                areProdsLoadsIsolated = True

        return bool(
            (comb[0] == 0)
            & (comb != node_configuration)
            & (comb != node_configuration_sym)
            & (sum_comb != 1)
            & (sum_comb != n_elements - 1)
            & (not areProdsLoadsIsolated)
        )

    def rank_topologies(
        self,
        all_combinations: List[Any],
        graph: nx.MultiDiGraph,
        node_to_change: int,
    ) -> pd.DataFrame:
        """Score every candidate topology and return a DataFrame with sentinel row."""
        scores_data = [["XX", ["X", "X", "X"], "X"]]

        for topo in all_combinations:
            is_single_node_topo = np.all(np.array(topo) == 0) or np.all(np.array(topo) == 1)
            score = self.rank_current_topo_at_node_x(self.g, node_to_change, is_single_node_topo, topo)
            if self.debug:
                logger.debug("topo %s on node %s → score %s", topo, node_to_change, score)
            scores_data.append([score, topo, node_to_change])

        return pd.DataFrame(columns=["score", "topology", "node"], data=scores_data)

    def sort_hubs(self, hubs: Optional[List[Any]]) -> Optional[pd.DataFrame]:
        """Sort hubs by largest absolute incident delta-flow; None if no hubs."""
        if not hubs:
            return None

        flows = []
        for node in hubs:
            ingoing, outgoing = [], []
            for _, row in self.df.iterrows():
                if row["idx_or"] == node:
                    outgoing.append(abs(row["delta_flows"]))
                if row["idx_ex"] == node:
                    ingoing.append(abs(row["delta_flows"]))
            flows.append(max(sum(ingoing), sum(outgoing)))

        df = pd.DataFrame({"hubs": hubs, "max_flows": flows})
        df.sort_values("max_flows", ascending=False, inplace=True)
        return df

    def identify_routing_buses(self) -> Dict[int, Any]:
        """Classify nodes into 4 categories: hubs / c_path / loop / aval."""
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

    def rank_loop_buses(
        self, graph: nx.MultiDiGraph, df_initial_flows: pd.DataFrame
    ) -> Dict[Any, float]:
        """Score each intermediate bus of every red loop by (non_red_inflow + local_production) * red_inflow_delta."""
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

    def _bus_loop_strength(
        self,
        bus: Any,
        df_initial_flows: pd.DataFrame,
        color_attrs: Dict[Any, Any],
        label_attrs: Dict[Any, Any],
    ) -> float:
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
        """Sum production values attached to *bus* (0 if none)."""
        total = 0.0
        for element in self.simulator_data["substations_elements"][bus]:
            if isinstance(element, Production):
                total += element.value
        return total

    @staticmethod
    def _initial_inflow_between(
        df_initial_flows: pd.DataFrame, source: Any, target: Any
    ) -> float:
        """Return |init_flow| for the line carrying power into *target* from *source*."""
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
        """Annotate self.g_distribution_graph loops with min_cut_values and min_cut_edges."""
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
        """Flatten a MultiDiGraph to a DiGraph by summing parallel edge capacities."""
        G = nx.DiGraph()
        for u, v, _, data in gM.edges(data=True, keys=True):
            w = data.get("capacity", 1.0)
            if G.has_edge(u, v):
                G[u][v]["capacity"] += w
            else:
                G.add_edge(u, v, capacity=w)
        return G

    def filter_constrained_path(self, path_to_filter: Any) -> List[Any]:
        """Flatten a nested edge-path structure to a deduplicated ordered node list."""
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
    """Skip the expensive pipeline; caller supplies a pre-built distribution graph."""

    def __init__(
        self,
        g: nx.MultiDiGraph,
        g_distribution_graph: Any,
        simulator_data: Optional[Dict[str, Any]] = None,
        debug: bool = False,
    ) -> None:
        self.bag_of_graphs: Dict[str, Any] = {}
        self.debug = debug
        self.boolean_dump_data_to_file = False
        self.g = g
        self.g_distribution_graph = g_distribution_graph
        self.simulator_data = simulator_data
