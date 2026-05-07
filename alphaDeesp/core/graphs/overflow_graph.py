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
        extra_lines_to_cut: Optional[Iterable[int]] = None,
    ) -> None:
        if "line_name" not in df_overflow.columns:
            df_overflow["line_name"] = [
                str(idx_or) + "_" + str(idx_ex) + "_" + str(i)
                for i, (idx_or, idx_ex) in df_overflow[["idx_or", "idx_ex"]].iterrows()
            ]

        self.df = df_overflow
        # Subset of ``lines_to_cut`` that the caller wants the cut-analysis
        # to treat like overloads (so they get the same black/constrained
        # styling and feed the structured-overload graph the same way) but
        # WITHOUT being classified as overloads in the viewer's
        # ``Overloads`` layer / ``is_overload`` flag.  Used by callers who
        # want the recommender to find actions that prevent flow increase
        # on otherwise-healthy lines (ExpertAgent's ``additionalLinesToCut``
        # semantic).  ``None`` / empty means "no extras" — every cut line
        # is a true overload, preserving the legacy behaviour.
        self.extra_lines_cut = set(extra_lines_to_cut or [])
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
        # Operator-selected extras must NOT be coloured black: black is the
        # visual signal for "overload contingency line" used by both the
        # ``Overloads`` layer and the structured-overload analyser.  We
        # therefore strip extras from the cut list passed to ``_edge_color``
        # so they keep their natural flow polarity colour (coral / blue).
        # They stay marked ``is_constrained`` and ``is_extra_cut`` so the
        # downstream layers can still find them by flag.
        cut_for_colour = [idx for idx in lines_to_cut if idx not in self.extra_lines_cut]
        for i, (origin, extremity, reported_flow, gray_edge, line_name) in enumerate(
                zip(*(self.df[c] for c in cols))):
            self._add_overflow_edge(
                g, origin, extremity, reported_flow, line_name,
                color=self._edge_color(i, reported_flow, gray_edge, cut_for_colour),
                scaling_factor=scaling_factor,
                min_penwidth=min_penwidth,
                is_constrained=(i in lines_to_cut),
                is_extra_cut=(i in self.extra_lines_cut))

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
        is_extra_cut: bool = False,
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
        if is_extra_cut:
            # Operator-supplied extra cut — gets the black/constrained
            # styling like a real overload but the viewer's "Overloads"
            # layer must skip it.  ``highlight_significant_line_loading``
            # respects this flag when stamping ``is_overload``.
            attrs["is_extra_cut"] = True
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
        """Distinguish hub nodes with a custom shape.

        Also stamps a ``is_hub`` boolean node attribute (source-of-truth flag)
        so downstream consumers — notably the interactive HTML viewer — can
        identify hubs without reinterpreting the visual shape.

        Hubs are by definition both **on the constrained path** AND
        **inside red-loop paths** (they are the converging substations
        feeding the overloaded lines, surrounded by positive-flow
        redispatch loops). Those flags are propagated here so a viewer
        layer toggle stays consistent regardless of which other tagging
        method (``tag_constrained_path`` / ``collapse_red_loops``) has
        already run.
        """
        dict_shapes = {node: "oval" for node in self.g.nodes}
        hubs_set = set(hubs)
        for hub in hubs_set:
            dict_shapes[hub] = shape_hub
        nx.set_node_attributes(self.g, dict_shapes, "shape")
        nx.set_node_attributes(
            self.g, {node: (node in hubs_set) for node in self.g.nodes}, "is_hub"
        )
        if hubs_set:
            nx.set_node_attributes(
                self.g, {h: True for h in hubs_set}, "on_constrained_path"
            )
            nx.set_node_attributes(
                self.g, {h: True for h in hubs_set}, "in_red_loop"
            )

    def highlight_swapped_flows(self, lines_swapped: List[Any]) -> None:
        """Draw lines whose flow direction has swapped in a tapered style."""
        edge_names = nx.get_edge_attributes(self.g, "name")
        swapped_edges = [edge for edge, name in edge_names.items() if name in lines_swapped]
        for attr_name, value in (("style", "tapered"), ("dir", "both"), ("arrowtail", "none")):
            nx.set_edge_attributes(self.g, {edge: value for edge in swapped_edges}, attr_name)

    def highlight_significant_line_loading(self, dict_line_loading: Dict[Any, Any]) -> None:
        """Augment edge labels with loading rates for monitored lines.

        Also stamps source-of-truth flags so the interactive viewer can
        toggle them as semantic layers without scraping the compound
        ``"X:yellow:X"`` colour:

        * ``is_monitored=True`` — every edge in ``dict_line_loading``
          (i.e. every line whose loading rate is high enough to be
          flagged as a "low-margin" line by the recommender).
        * ``is_overload=True`` — the strict subset of monitored edges
          that are overloaded contingency lines (current colour was
          ``black`` before the highlight). Overloads are therefore a
          subset of low-margin lines, not a disjoint category.
        """
        edge_names = nx.get_edge_attributes(self.g, "name")
        edge_colors = nx.get_edge_attributes(self.g, "color")
        edge_x_labels = nx.get_edge_attributes(self.g, "label")
        edge_extra_cut = nx.get_edge_attributes(self.g, "is_extra_cut")
        label_font_color = {edge: "black" for edge in edge_names.keys()}
        color_label_highlight = "darkred"

        is_overload_attrs: Dict[Any, bool] = {}
        is_monitored_attrs: Dict[Any, bool] = {}

        for edge, edge_name in edge_names.items():
            if edge_name not in dict_line_loading:
                continue
            current_x_label = edge_x_labels[edge]
            current_edge_color = edge_colors[edge]
            before = dict_line_loading[edge_name]["before"]
            after = dict_line_loading[edge_name]["after"]
            is_extra = bool(edge_extra_cut.get(edge, False))

            # Every entry in dict_line_loading is a monitored / low-
            # margin line; the black ones are additionally overloads.
            # Operator-selected extras (``is_extra_cut``) are kept out
            # of both flags so the viewer's ``Overloads`` and
            # ``Low margin lines`` layers reflect the recommender's
            # detected state, not user-supplied targets.
            if not is_extra:
                is_monitored_attrs[edge] = True
                if current_edge_color == "black":
                    edge_x_labels[edge] = f'< {current_x_label} <BR/>  <B>{before}%</B>  → {after}%>'
                    is_overload_attrs[edge] = True
                else:
                    edge_x_labels[edge] = f'< {current_x_label} <BR/>  {before}% → <B>{after}%</B> >'
                edge_colors[edge] = f'"{current_edge_color}:yellow:{current_edge_color}"'
            else:
                # Extras keep their natural flow colour; only the
                # ``before → 0%`` annotation surfaces the cut so the
                # operator sees how their choice materialises.
                edge_x_labels[edge] = f'< {current_x_label} <BR/>  {before}% → <B>{after}%</B> >'

            label_font_color[edge] = color_label_highlight

        nx.set_edge_attributes(self.g, edge_x_labels, "label")
        nx.set_edge_attributes(self.g, label_font_color, "fontcolor")
        nx.set_edge_attributes(self.g, edge_colors, "color")
        if is_overload_attrs:
            nx.set_edge_attributes(self.g, is_overload_attrs, "is_overload")
        if is_monitored_attrs:
            nx.set_edge_attributes(self.g, is_monitored_attrs, "is_monitored")

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
        """Collapse purely-coral, non-hub nodes to point shapes.

        This is purely a visual heuristic for the rendered graph
        (point markers vs ovals). The semantic ``in_red_loop`` flag is
        no longer derived from this collapse — it is set explicitly by
        :meth:`tag_red_loops` from the recommender's
        ``get_dispatch_edges_nodes(only_loop_paths=True)`` source-of-
        truth list.
        """
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

    def tag_red_loops(
        self,
        lines_red_loops: Optional[Iterable[str]] = None,
        nodes_red_loops: Optional[Iterable[Any]] = None,
    ) -> None:
        """Tag the source-of-truth ``in_red_loop`` flag on the edges
        and nodes that form the dispatch loop paths identified upstream
        by ``Structured_Overload_Distribution_Graph.get_dispatch_edges_
        nodes(only_loop_paths=True)``.

        This replaces the previous ad-hoc heuristics (collapse-based,
        connected-component-based) that produced false positives on
        coral "exit" branches such as the CHALOP6→CHALOP3 transformers
        the user reported. The recommender already computes the actual
        cycle paths from the structured analysis — we just propagate
        them as graph attributes.

        Edges are matched by their ``name`` attribute (the line name
        used everywhere else in the codebase). Nodes are matched by
        identity.
        """
        if lines_red_loops:
            wanted = set(lines_red_loops)
            edge_names = nx.get_edge_attributes(self.g, "name")
            edge_attrs = {
                edge: True for edge, name in edge_names.items() if name in wanted
            }
            if edge_attrs:
                nx.set_edge_attributes(self.g, edge_attrs, "in_red_loop")
        if nodes_red_loops:
            wanted_nodes = set(nodes_red_loops)
            node_attrs = {
                node: True for node in self.g.nodes if node in wanted_nodes
            }
            if node_attrs:
                nx.set_node_attributes(self.g, node_attrs, "in_red_loop")

    def tag_constrained_path(
        self,
        lines_constrained_path: Optional[Iterable[str]] = None,
        nodes_constrained_path: Optional[Iterable[Any]] = None,
    ) -> None:
        """Tag the source-of-truth ``on_constrained_path`` flag on the
        edges and nodes that form the constrained path identified
        upstream by the recommender's distribution graph analysis.

        Edges are matched by their ``name`` attribute (the line name
        used everywhere else in the codebase). Nodes are matched by
        identity in the graph.

        **Coral edges are skipped** even when their ``name`` matches a
        constrained-path entry. The constrained path is, by definition,
        the network of black (overloaded) and blue (negative-flow)
        edges that funnel current into the overloads. The overflow
        ``MultiDiGraph`` may carry both flow directions of a single
        physical line under the same ``name`` — only the negative one
        is on the constrained path; including the coral counterpart
        would surface positive-overflow edges in the layer toggle and
        confuse the operator.
        """
        if lines_constrained_path:
            wanted = set(lines_constrained_path)
            edge_names = nx.get_edge_attributes(self.g, "name")
            edge_colors = nx.get_edge_attributes(self.g, "color")
            edge_attrs: Dict[Any, bool] = {}
            for edge, name in edge_names.items():
                if name not in wanted:
                    continue
                color = edge_colors.get(edge, "")
                base_color = (
                    color.split(":", 1)[0].strip().strip('"').lower()
                    if isinstance(color, str)
                    else ""
                )
                if base_color == "coral":
                    continue
                edge_attrs[edge] = True
            if edge_attrs:
                nx.set_edge_attributes(self.g, edge_attrs, "on_constrained_path")
        if nodes_constrained_path:
            wanted_nodes = set(nodes_constrained_path)
            node_attrs = {
                node: True for node in self.g.nodes if node in wanted_nodes
            }
            if node_attrs:
                nx.set_node_attributes(self.g, node_attrs, "on_constrained_path")

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
