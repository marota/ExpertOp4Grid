"""GraphConsolidationMixin: graph consolidation and disambiguation for OverFlowGraph.

Extracted from ``alphaDeesp/core/graphs/overflow_graph.py`` to keep per-file
LOC and average cyclomatic complexity within A-grade bounds.

The mixin assumes the concrete class provides:
    self.g               — the overflow MultiDiGraph
    self.float_precision — format string (e.g. "%.2f")
"""

import logging
from typing import Any, Iterable, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from alphaDeesp.core.graphs.graph_utils import (
    all_simple_edge_paths_multi,
    delete_color_edges,
)
from alphaDeesp.core.graphs.structured_overload_graph import (
    Structured_Overload_Distribution_Graph,
)

logger = logging.getLogger(__name__)


class GraphConsolidationMixin:
    """Graph consolidation and flow-direction helpers; mixed into OverFlowGraph."""

    def consolidate_constrained_path(
        self,
        constrained_path_nodes_amont: List[Any],
        constrained_path_nodes_aval: List[Any],
        constrained_path_edges: List[Any],
        ignore_null_edges: bool = True,
    ) -> None:
        """Extend the blue constrained path to cover below-threshold edges."""
        g_base = delete_color_edges(self.g, "coral")
        if ignore_null_edges:
            init_capacity = nx.get_edge_attributes(g_base, "capacity")
            g_base.remove_edges_from([e for e, c in init_capacity.items() if c == 0.])
        g_base.remove_edges_from(constrained_path_edges)

        g_amont = g_base.copy()
        g_amont.remove_nodes_from(constrained_path_nodes_aval)
        g_aval = g_base
        g_aval.remove_nodes_from(constrained_path_nodes_amont)

        for g_c, sources in ((g_amont, constrained_path_nodes_amont),
                             (g_aval, constrained_path_nodes_aval)):
            self._recolor_ambiguous_as_blue(g_c, sources)

    def _recolor_ambiguous_as_blue(
        self, g_c: nx.MultiDiGraph, sources: Iterable[Any]
    ) -> None:
        """Recolour non-{blue, black} edges on cycles within g_c to blue on self.g."""
        paths = list(all_simple_edge_paths_multi(g_c, sources, sources))
        if not paths:
            return
        colors = nx.get_edge_attributes(g_c, 'color')
        edges_to_recolor: Set[Any] = set()
        for path in paths:
            if any(colors[edge] not in ("blue", "black") for edge in path):
                edges_to_recolor.update(path)
        updates = {edge: {"color": "blue"} for edge in edges_to_recolor
                   if colors[edge] not in ("blue", "black")}
        nx.set_edge_attributes(self.g, updates)

    def reverse_edges(self, edge_path_names: List[str], target_color: str) -> None:
        """Reverse edge directions and flip capacities for named edges."""
        graph_edge_names = nx.get_edge_attributes(self.g, 'name')
        edges_path = [edge for edge, name in graph_edge_names.items() if name in edge_path_names]
        path_subgraph = self.g.edge_subgraph(edges_path)
        current_colors = nx.get_edge_attributes(path_subgraph, 'color')

        new_colors = {e: {"color": target_color} for e in current_colors}
        nx.set_edge_attributes(self.g, new_colors)

        reduced_capacities_dict = nx.get_edge_attributes(path_subgraph, "capacity")
        new_attributes_dict = {
            e: {"capacity": -cap, "label": self.float_precision % -cap}
            for e, cap in reduced_capacities_dict.items()
            if current_colors[e] != target_color
        }
        nx.set_edge_attributes(self.g, new_attributes_dict)

        edges_to_reverse = list(new_attributes_dict.keys())
        path_subgraph_to_reverse = path_subgraph.edge_subgraph(edges_to_reverse)
        self.g.add_edges_from([(e[1], e[0], e[2]) for e in path_subgraph_to_reverse.edges(data=True)])
        self.g.remove_edges_from(edges_to_reverse)

    def reverse_blue_edges_in_looppaths(self, constrained_path: List[Any]) -> None:
        """Reverse blue edges outside constrained paths so they push flows outward."""
        g_without_pos = delete_color_edges(self.g, "coral")
        g_without_pos.remove_nodes_from(constrained_path)

        capacities_dict = nx.get_edge_attributes(g_without_pos, "capacity")
        g_without_pos.remove_edges_from(
            [e for e, cap in capacities_dict.items() if cap > -1])

        current_colors = nx.get_edge_attributes(g_without_pos, 'color')
        new_colors = {e: {"color": "coral"} for e, c in current_colors.items() if c != "gray"}
        nx.set_edge_attributes(g_without_pos, new_colors)

        reduced_caps = nx.get_edge_attributes(g_without_pos, "capacity")
        new_attrs = {
            e: {"capacity": -cap, "label": self.float_precision % -cap}
            for e, cap in reduced_caps.items() if cap != 0
        }
        nx.set_edge_attributes(g_without_pos, new_attrs)

        self.g.add_edges_from([(e[1], e[0], e[2]) for e in g_without_pos.edges(data=True)])
        self.g.remove_edges_from(g_without_pos.edges)

    def consolidate_loop_path(
        self,
        hub_sources: Iterable[Any],
        hub_targets: Iterable[Any],
        ignore_null_edges: bool = True,
    ) -> None:
        """Recolour gray edges on loop paths between hubs to coral."""
        all_edges_to_recolor = []
        g_without_blue = delete_color_edges(self.g, "blue")

        if ignore_null_edges:
            init_capacity = nx.get_edge_attributes(g_without_blue, "capacity")
            g_without_blue.remove_edges_from(
                [e for e, cap in init_capacity.items() if cap == 0.])

        for source, target in zip(hub_sources, hub_targets):
            for path in nx.all_simple_edge_paths(g_without_blue, source, target):
                all_edges_to_recolor += path

        all_edges_to_recolor = set(all_edges_to_recolor)
        current_colors = nx.get_edge_attributes(self.g, 'color')
        edge_attrs = {
            edge: {"color": "coral"}
            for edge in g_without_blue.edges
            if edge in all_edges_to_recolor and current_colors[edge] == "gray"
        }
        nx.set_edge_attributes(self.g, edge_attrs)

    def consolidate_graph(
        self,
        structured_graph: Any,
        non_connected_lines_to_ignore: List[Any] = [],
        no_desambiguation: bool = False,
    ) -> None:
        """Consolidate overflow graph knowing structural elements from StructuredOverflowGraph."""
        edge_names = nx.get_edge_attributes(self.g, 'name')
        edges_to_remove = [
            e for e, name in edge_names.items() if name in non_connected_lines_to_ignore
        ]
        edges_to_remove_data = [
            (u, v, data) for u, v, data in self.g.edges(data=True)
            if data["name"] in non_connected_lines_to_ignore
        ]
        self.g.remove_edges_from(edges_to_remove)

        structured_graph, hubs_paths = self._run_consolidation_loop(structured_graph)

        if not no_desambiguation:
            ambiguous_edge_paths, ambiguous_node_paths = self.identify_ambiguous_paths(structured_graph)
            for ambiguous_edge_path, ambiguous_node_path in zip(ambiguous_edge_paths, ambiguous_node_paths):
                path_type = self.desambiguation_type_path(ambiguous_node_path, structured_graph)
                self.reverse_edges(ambiguous_edge_path, "coral" if path_type == "loop_path" else "blue")

        self.consolidate_loop_path(hubs_paths.Source, hubs_paths.Target)
        self.g.add_edges_from(edges_to_remove_data)

    def _run_consolidation_loop(self, structured_graph: Any) -> Tuple[Any, Any]:
        """Iterate constrained-path consolidation until hub count stabilises."""
        hubs_paths = structured_graph.find_loops()[["Source", "Target"]].drop_duplicates()
        n_hub_paths = hubs_paths.shape[0]
        n_hubs_init = 0

        while n_hubs_init != n_hub_paths:
            n_hubs_init = n_hub_paths
            cp = structured_graph.constrained_path
            cp_edges = cp.aval_edges + [cp.constrained_edge] + cp.amont_edges
            self.consolidate_constrained_path(cp.n_amont(), cp.n_aval(), cp_edges)
            structured_graph = Structured_Overload_Distribution_Graph(self.g)
            hubs_paths = structured_graph.find_loops()[["Source", "Target"]].drop_duplicates()
            n_hub_paths = hubs_paths.shape[0]

        return structured_graph, hubs_paths

    def identify_ambiguous_paths(
        self, structured_graph: Any
    ) -> Tuple[List[Any], List[Any]]:
        """Return edge/node paths containing both red and blue edges."""
        g_amb = structured_graph.g_without_gray_and_c_edge
        edge_names = nx.get_edge_attributes(structured_graph.g_without_gray_and_c_edge, 'name')

        lines_constrained_path, _, _other_blue, _ = structured_graph.get_constrained_edges_nodes()
        lines_dispatch, _ = structured_graph.get_dispatch_edges_nodes()

        edges_to_remove = [
            e for e, name in edge_names.items()
            if name in lines_constrained_path + lines_dispatch
        ]
        g_amb.remove_edges_from(edges_to_remove)

        ambiguous_edge_paths, ambiguous_node_paths = [], []
        for c in nx.weakly_connected_components(g_amb):
            if self._is_ambiguous_component(g_amb, c):
                ambiguous_node_paths.append(c)
                ambiguous_edge_paths.append(
                    list(nx.get_edge_attributes(g_amb.subgraph(c), "name").values()))

        return ambiguous_edge_paths, ambiguous_node_paths

    @staticmethod
    def _is_ambiguous_component(g_amb: nx.MultiDiGraph, component: set) -> bool:
        """True when component has ≥2 nodes and contains both blue and coral edges."""
        if len(component) < 2:
            return False
        comp_colors = np.unique(list(
            nx.get_edge_attributes(g_amb.subgraph(component), "color").values()))
        return "blue" in comp_colors and "coral" in comp_colors and len(comp_colors) == 2

    def desambiguation_type_path(
        self, ambiguous_node_path: Iterable[Any], structured_graph: Any
    ) -> str:
        """Classify an ambiguous path as 'constrained_path' or 'loop_path'."""
        cp = structured_graph.constrained_path
        nodes_in_cp = [n for n in ambiguous_node_path if n in cp.full_n_constrained_path()]

        if len(nodes_in_cp) < 2:
            return "loop_path"

        connects_amont = any(n in cp.n_amont() for n in nodes_in_cp)
        connects_aval = any(n in cp.n_aval() for n in nodes_in_cp)
        return "loop_path" if (connects_amont and connects_aval) else "constrained_path"
