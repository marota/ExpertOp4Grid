"""OverFlowGraph: coloured overflow-redispatch graph.

Subclasses :class:`PowerFlowGraph`, :class:`NullFlowGraphMixin`, and
:class:`GraphConsolidationMixin`. The null-flow and consolidation logic live
in the mixins to keep per-file complexity within A-grade bounds.
"""

import logging
from math import fabs
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from alphaDeesp.core.printer import Printer
from alphaDeesp.core.graphs.power_flow_graph import PowerFlowGraph
from alphaDeesp.core.graphs.null_flow_graph import NullFlowGraphMixin
from alphaDeesp.core.graphs.graph_consolidation import GraphConsolidationMixin
from alphaDeesp.core.graphs.graph_utils import delete_color_edges

logger = logging.getLogger(__name__)

# Penwidth thresholds used by build_edges_from_df.
# The floor is dynamic: at least the width equivalent to 1 MW of flow
# (``scaling_factor`` applied to 1.0), and at least 10 % of the largest
# rendered penwidth, so low / zero-flow edges (reconnectable,
# non-reconnectable, null-flow) remain visible without zooming.
_TARGET_MAX_PENWIDTH = 15.0
_MIN_PENWIDTH_FLOW_MW = 1.0
_MIN_PENWIDTH_FRACTION = 0.10


class OverFlowGraph(NullFlowGraphMixin, GraphConsolidationMixin, PowerFlowGraph):
    """A coloured graph of grid overflow redispatch."""

    def __init__(
        self,
        topo: Dict[str, Any],
        lines_to_cut: List[int],
        df_overflow: pd.DataFrame,
        layout: Optional[List[Tuple[float, float]]] = None,
        float_precision: str = "%.2f",
    ) -> None:
        if "line_name" not in df_overflow.columns:
            df_overflow["line_name"] = [
                str(idx_or) + "_" + str(idx_ex) + "_" + str(i)
                for i, (idx_or, idx_ex) in df_overflow[["idx_or", "idx_ex"]].iterrows()
            ]

        self.df = df_overflow
        super().__init__(topo, lines_to_cut, layout, float_precision)

    def build_graph(self) -> None:
        """Create the NetworkX MultiDiGraph from the overflow DataFrame."""
        g = nx.MultiDiGraph()
        self.build_nodes(
            g,
            self.topo["nodes"]["are_prods"],
            self.topo["nodes"]["are_loads"],
            self.topo["nodes"]["prods_values"],
            self.topo["nodes"]["loads_values"],
        )
        self.build_edges_from_df(g, self.lines_cut)
        self.g = g

    def build_edges_from_df(self, g: nx.MultiDiGraph, lines_to_cut: List[int]) -> None:
        """Add one coloured edge per row of self.df to g."""
        max_abs_flow = self.df["delta_flows"].abs().max()
        scaling_factor = _TARGET_MAX_PENWIDTH / max_abs_flow if max_abs_flow > 0 else 1.0
        min_penwidth = max(
            _MIN_PENWIDTH_FLOW_MW * scaling_factor,
            _MIN_PENWIDTH_FRACTION * _TARGET_MAX_PENWIDTH,
        )

        cols = ("idx_or", "idx_ex", "delta_flows", "gray_edges", "line_name")
        for i, (origin, extremity, reported_flow, gray_edge, line_name) in enumerate(
                zip(*(self.df[c] for c in cols))):
            self._add_overflow_edge(
                g, origin, extremity, reported_flow, line_name,
                color=self._edge_color(i, reported_flow, gray_edge, lines_to_cut),
                scaling_factor=scaling_factor,
                min_penwidth=min_penwidth,
                is_constrained=(i in lines_to_cut))

    @staticmethod
    def _edge_color(
        index: int, reported_flow: float, gray_edge: bool, lines_to_cut: List[int]
    ) -> str:
        """Map a row to its edge colour: black → gray → blue/coral."""
        if index in lines_to_cut:
            return "black"
        if gray_edge:
            return "gray"
        return "blue" if reported_flow < 0 else "coral"

    def _add_overflow_edge(
        self,
        g: nx.MultiDiGraph,
        origin: Any,
        extremity: Any,
        reported_flow: float,
        line_name: str,
        color: str,
        scaling_factor: float,
        min_penwidth: float,
        is_constrained: bool,
    ) -> None:
        """Add a single styled overflow edge to g."""
        fp = self.float_precision
        penwidth = max(float(fp % (fabs(reported_flow) * scaling_factor)), min_penwidth)
        attrs = {
            "capacity": float(fp % reported_flow),
            "label": fp % reported_flow,
            "color": color,
            "fontsize": 10,
            "penwidth": penwidth,
            "name": line_name,
        }
        if is_constrained:
            attrs["constrained"] = True
        g.add_edge(origin, extremity, **attrs)

    def keep_overloads_components(self) -> None:
        """Recolour to gray edges in components that contain no overloaded (black) edge."""
        g_coloured = delete_color_edges(self.g, "gray")
        components = list(nx.weakly_connected_components(g_coloured))

        for component_nodes in components:
            subgraph = g_coloured.subgraph(component_nodes)
            has_overload = any(color == "black" for _, _, color in subgraph.edges(data="color"))
            if not has_overload:
                for u, v, key in self.g.edges(keys=True):
                    if u in component_nodes and v in component_nodes:
                        if self.g[u][v][key].get("color") != "gray":
                            self.g[u][v][key]["color"] = "gray"

    def set_hubs_shape(self, hubs: Iterable[Any], shape_hub: str = "circle") -> None:
        """Distinguish hub nodes with a custom shape."""
        dict_shapes = {node: "oval" for node in self.g.nodes}
        for hub in hubs:
            dict_shapes[hub] = shape_hub
        nx.set_node_attributes(self.g, dict_shapes, "shape")

    def highlight_swapped_flows(self, lines_swapped: List[Any]) -> None:
        """Draw lines whose flow direction has swapped in a tapered style."""
        edge_names = nx.get_edge_attributes(self.g, "name")
        swapped_edges = [edge for edge, name in edge_names.items() if name in lines_swapped]
        for attr_name, value in (("style", "tapered"), ("dir", "both"), ("arrowtail", "none")):
            nx.set_edge_attributes(self.g, {edge: value for edge in swapped_edges}, attr_name)

    def highlight_significant_line_loading(self, dict_line_loading: Dict[Any, Any]) -> None:
        """Augment edge labels with loading rates for monitored lines."""
        edge_names = nx.get_edge_attributes(self.g, "name")
        edge_colors = nx.get_edge_attributes(self.g, "color")
        edge_x_labels = nx.get_edge_attributes(self.g, "label")
        label_font_color = {edge: "black" for edge in edge_names.keys()}
        color_label_highlight = "darkred"

        for edge, edge_name in edge_names.items():
            if edge_name not in dict_line_loading:
                continue
            current_x_label = edge_x_labels[edge]
            current_edge_color = edge_colors[edge]
            before = dict_line_loading[edge_name]["before"]
            after = dict_line_loading[edge_name]["after"]

            if current_edge_color == "black":
                edge_x_labels[edge] = f'< {current_x_label} <BR/>  <B>{before}%</B>  → {after}%>'
            else:
                edge_x_labels[edge] = f'< {current_x_label} <BR/>  {before}% → <B>{after}%</B> >'

            label_font_color[edge] = color_label_highlight
            edge_colors[edge] = f'"{current_edge_color}:yellow:{current_edge_color}"'

        nx.set_edge_attributes(self.g, edge_x_labels, "label")
        nx.set_edge_attributes(self.g, label_font_color, "fontcolor")
        nx.set_edge_attributes(self.g, edge_colors, "color")

    def plot(
        self,
        layout: Optional[List[Any]],
        rescale_factor: Optional[float] = None,
        allow_overlap: bool = True,
        fontsize: Optional[int] = None,
        node_thickness: int = 3,
        save_folder: str = "",
        without_gray_edges: bool = False,
    ) -> Any:
        printer = Printer(save_folder)
        g = self.g

        if without_gray_edges:
            layout_dict = {n: c for n, c in zip(g.nodes, layout)} if layout is not None else None
            g = delete_color_edges(g, "gray")
            if layout_dict is not None:
                layout = [layout_dict[node] for node in g.nodes]

        kwargs = dict(rescale_factor=rescale_factor, fontsize=fontsize,
                      node_thickness=node_thickness, name="g_overflow_print")
        if save_folder == "":
            return printer.plot_graphviz(g, layout, allow_overlap=allow_overlap, **kwargs)
        printer.display_geo(g, layout, **kwargs)
        return None

    def rename_nodes(self, mapping: Dict[Any, Any]) -> None:
        self.g = nx.relabel_nodes(self.g, mapping, copy=True)
        self.df["idx_or"] = [mapping[idx_or] for idx_or in self.df["idx_or"]]
        self.df["idx_ex"] = [mapping[idx_or] for idx_or in self.df["idx_ex"]]

    def collapse_red_loops(self) -> None:
        """Collapse purely-coral, non-hub nodes to point shapes."""
        shapes = nx.get_node_attributes(self.g, "shape")
        peripheries = nx.get_node_attributes(self.g, "peripheries")
        edge_colors = nx.get_edge_attributes(self.g, "color")
        edge_styles = nx.get_edge_attributes(self.g, "style")

        nodes_to_collapse = {}
        for node in self.g.nodes:
            if shapes.get(node) != "oval":
                continue
            if node in peripheries and peripheries[node] >= 2:
                continue
            all_edges = list(self.g.in_edges(node, keys=True)) + list(self.g.out_edges(node, keys=True))
            if all_edges and self._all_edges_coral_no_dash(all_edges, edge_colors, edge_styles):
                nodes_to_collapse[node] = "point"

        nx.set_node_attributes(self.g, nodes_to_collapse, "shape")

    @staticmethod
    def _all_edges_coral_no_dash(
        all_edges: List[Any],
        edge_colors: Dict[Any, str],
        edge_styles: Dict[Any, str],
    ) -> bool:
        """Return True when all edges are coral and none are dashed/dotted."""
        for edge in all_edges:
            if edge_colors.get(edge) != "coral":
                return False
            if edge_styles.get(edge, "") in ("dashed", "dotted"):
                return False
        return True
