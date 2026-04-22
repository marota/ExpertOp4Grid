"""OverFlowGraph: coloured overflow-redispatch graph.

Subclasses :class:`PowerFlowGraph`. Most of the mass of the original
``graphsAndPaths`` monolith lived here: this module keeps the class body
identical while delegating helpers and the structured-graph builder to
sibling modules.
"""

import logging
from math import fabs
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from alphaDeesp.core.printer import Printer
from alphaDeesp.core.graphs.power_flow_graph import PowerFlowGraph
from alphaDeesp.core.graphs.structured_overload_graph import (
    Structured_Overload_Distribution_Graph,
)
from alphaDeesp.core.graphs.graph_utils import (
    all_simple_edge_paths_multi,
    delete_color_edges,
    find_multidigraph_edges_by_name,
    nodepath_to_edgepath,
)
from alphaDeesp.core.graphs.null_flow import (
    add_double_edges_null_redispatch,
    remove_unused_added_double_edge,
)

logger = logging.getLogger(__name__)

# Penwidth thresholds used by :meth:`OverFlowGraph.build_edges_from_df`
_TARGET_MAX_PENWIDTH = 15.0
_MIN_PENWIDTH = 0.1


class OverFlowGraph(PowerFlowGraph):
    """
    A coloured graph of grid overflow redispatch, displaying the delta flows before and after disconnecting the overloaded lines
    """

    def __init__(self, topo: Dict[str, Any], lines_to_cut: List[int], df_overflow: pd.DataFrame, layout: Optional[List[Tuple[float, float]]] = None, float_precision: str = "%.2f") -> None:
        """
        Parameters
        ----------

        topo: :class:`dict`
            dictionnary of two dictionnaries edges and nodes, to represent the grid topologie. edges have attributes "init_flows" representing the power flowing, as well as "idx_or","idx_ex"
             for substation extremities
             Nodes have attributes "are_prods","are_loads" if nodes have any productions or any load, as well as "prods_values","load_values" array enumerating the prod and load values at this node.

        lines_to_cut: ``array``
            ids of lines disconnected

        df_overflow: :class:``pd.Dataframe``
            pandas dataframe of deltaflows after disconnecting the overloaded lines. see create_df in simulation.py. One row per powerline
            columns: idx_or, idx_ex, init_flows, new_flows, delta_flows, gray_edges (for unsignificant delta_flows below a threshold)

        """
        if "line_name" not in df_overflow.columns:
            df_overflow["line_name"]=[str(idx_or)+"_"+str(idx_ex)+"_"+str(i) for i, (idx_or,idx_ex) in df_overflow[["idx_or","idx_ex"]].iterrows()]

        self.df = df_overflow
        super().__init__(topo, lines_to_cut,layout,float_precision)

    def build_graph(self) -> None:
        """This method creates the NetworkX Graph of the overflow redispatch """
        g = nx.MultiDiGraph()
        self.build_nodes(g, self.topo["nodes"]["are_prods"], self.topo["nodes"]["are_loads"],
                    self.topo["nodes"]["prods_values"], self.topo["nodes"]["loads_values"])

        self.build_edges_from_df(g, self.lines_cut)

        # print("WE ARE IN BUILD GRAPH FROM DATA FRAME ===========")
        # all_edges_label_attributes = nx.get_edge_attributes(g, "label")  # dict[edge]
        # print("all_edges_label_attributes = ", all_edges_label_attributes)

        self.g=g
        #self.add_double_edges_null_redispatch()

    def build_edges_from_df(self, g: nx.MultiDiGraph, lines_to_cut: List[int]) -> None:
        """Add one coloured edge per row of ``self.df`` to ``g``.

        The penwidth is scaled linearly so the largest absolute delta-flow
        maps to :data:`_TARGET_MAX_PENWIDTH`, with :data:`_MIN_PENWIDTH` as
        the floor for near-zero flows. Colour follows :meth:`_edge_color`."""
        max_abs_flow = self.df["delta_flows"].abs().max()
        scaling_factor = _TARGET_MAX_PENWIDTH / max_abs_flow if max_abs_flow > 0 else 1.0

        cols = ("idx_or", "idx_ex", "delta_flows", "gray_edges", "line_name")
        for i, (origin, extremity, reported_flow, gray_edge, line_name) in enumerate(
                zip(*(self.df[c] for c in cols))):
            self._add_overflow_edge(
                g, origin, extremity, reported_flow, line_name,
                color=self._edge_color(i, reported_flow, gray_edge, lines_to_cut),
                scaling_factor=scaling_factor,
                is_constrained=(i in lines_to_cut))

    @staticmethod
    def _edge_color(index: int, reported_flow: float, gray_edge: bool, lines_to_cut: List[int]) -> str:
        """Map a row to its edge colour: black (cut line) → gray (insignificant)
        → blue (negative flow) → coral (positive flow)."""
        if index in lines_to_cut:
            return "black"
        if gray_edge:
            return "gray"
        return "blue" if reported_flow < 0 else "coral"

    def _add_overflow_edge(self, g: nx.MultiDiGraph, origin: Any, extremity: Any,
                           reported_flow: float, line_name: str, color: str,
                           scaling_factor: float, is_constrained: bool) -> None:
        """Add a single styled overflow edge to ``g``."""
        fp = self.float_precision
        penwidth = max(float(fp % (fabs(reported_flow) * scaling_factor)), _MIN_PENWIDTH)
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
        """
        Filter the graph to only keep components that contain overloaded (black) edges.

        For the coloured graph (graph without grey edges), detect connected components
        that do not include any overloaded edges (black colour) and recolour all their
        edges to grey so they are no longer considered significant.
        """
        # Build the coloured graph: remove grey edges to get only significant ones
        g_coloured = delete_color_edges(self.g, "gray")

        # Find weakly connected components of the coloured graph
        components = list(nx.weakly_connected_components(g_coloured))

        for component_nodes in components:
            # Get the subgraph for this component
            subgraph = g_coloured.subgraph(component_nodes)

            # Check if the component contains any black (overloaded) edge
            has_overload = any(
                color == "black"
                for _, _, color in subgraph.edges(data="color")
            )

            if not has_overload:
                # Recolour all edges of this component to grey in the original graph
                for u, v, key in self.g.edges(keys=True):
                    if u in component_nodes and v in component_nodes:
                        if self.g[u][v][key].get("color") != "gray":
                            self.g[u][v][key]["color"] = "gray"

    def consolidate_constrained_path(self, constrained_path_nodes_amont: List[Any], constrained_path_nodes_aval: List[Any], constrained_path_edges: List[Any], ignore_null_edges: bool = True) -> None:
        """
        Extend the constrained (blue) path to cover edges that were discarded
        because their delta flow was below threshold but are actually part of
        the path. Works separately on the amont and aval sides of the
        constrained edge so an amont extension never crosses into aval territory.
        """
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

    def _recolor_ambiguous_as_blue(self, g_c: nx.MultiDiGraph, sources: Iterable[Any]) -> None:
        """For every simple path from ``sources`` back to ``sources`` inside
        ``g_c`` that contains a non-{blue, black} edge, recolour every
        non-{blue, black} edge of the path to ``blue`` on ``self.g``."""
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

        graph_adge_names = nx.get_edge_attributes(self.g, 'name')
        edges_path=[edge for edge, name in graph_adge_names.items() if name in edge_path_names]

        path_subgraph = self.g.edge_subgraph(edges_path)
        current_colors = nx.get_edge_attributes(path_subgraph, 'color')

        if target_color == "coral":
            new_colors = {e: {"color": "coral"} for e, color
                          in current_colors.items()}
        else:
            new_colors = {e: {"color": "blue"} for e, color
                          in current_colors.items()}
        nx.set_edge_attributes(self.g, new_colors)
        # reversing capacities (with negative values) and direction for edges for which we changed color
        reduced_capacities_dict = nx.get_edge_attributes(path_subgraph, "capacity")
        new_attributes_dict = {e: {"capacity": -capacity, "label": self.float_precision % -capacity} for e, capacity
                               in reduced_capacities_dict.items() if current_colors[e]!=target_color}
        nx.set_edge_attributes(self.g, new_attributes_dict)

        edges_to_reverse=list(new_attributes_dict.keys())
        path_subgraph_to_reverse = path_subgraph.edge_subgraph(edges_to_reverse)
        self.g.add_edges_from([(edge[1], edge[0], edge[2]) for edge in path_subgraph_to_reverse.edges(data=True)])
        self.g.remove_edges_from(edges_to_reverse)

    def reverse_blue_edges_in_looppaths(self, constrained_path: List[Any]) -> None:
        """
        Reverse blue edges that are not on the constrained paths, and that should be regarded as edges on which we are pushing
        the flows

        Parameters
        ----------

        constrained_path: ``list``
            list of nodes that areon the constrained path

        """
        g_without_pos_edges = delete_color_edges(self.g, "coral")
        #g_only_blue_components = delete_color_edges(g_without_pos_edges, "gray")
        #g_only_blue_components.remove_nodes_from(constrained_path)
        g_without_pos_edges.remove_nodes_from(constrained_path)

        #edges that have positive capacities, and significant enough (more than 1MW delta_flow) among gray edges should not be touched
        capacities_dict = nx.get_edge_attributes(g_without_pos_edges, "capacity")
        g_without_pos_edges.remove_edges_from([edge for edge,capacity in capacities_dict.items() if capacity>-1])

        #modifies blue edges (reverse them and color them red) that are not on constrained path
        #on the graph
        #changing colors only for significative flows (non gray) here
        current_colors = nx.get_edge_attributes(g_without_pos_edges, 'color')

        new_colors = {e: {"color": "coral"} for e, color
                               in current_colors.items() if color!="gray"}
        nx.set_edge_attributes(g_without_pos_edges, new_colors)

        # reversing capacities (with negative values) and direction for all edges here
        reduced_capacities_dict = nx.get_edge_attributes(g_without_pos_edges, "capacity")
        new_attributes_dict = {e: {"capacity": -capacity, "label": self.float_precision % -capacity} for e, capacity
                               in reduced_capacities_dict.items() if capacity!=0}
        nx.set_edge_attributes(g_without_pos_edges, new_attributes_dict)

        self.g.add_edges_from([(edge[1], edge[0], edge[2]) for edge in g_without_pos_edges.edges(data=True)])
        self.g.remove_edges_from(g_without_pos_edges.edges)

    def consolidate_loop_path(self, hub_sources: Iterable[Any], hub_targets: Iterable[Any], ignore_null_edges: bool = True) -> None:
        """
        Consolidate constrained red path for some edges that were discarded with lower values but are actually on the path
        knowing the hubs in the SuscturedOverflowGraph
        WARNING: prefer to reverse blue edges first with reverse_blue_edges_in_looppaths, to get a better result

        Parameters
        ----------

        hub_sources: ``array``
            list of nodes that are hubs and sources of loop paths in the structured graph

        hub_targets: ``array``
            list of nodes that are hubs and targets of loop paths in the structured graph

        """
        all_edges_to_recolor = []

        # we capture all edges with negative value that we find in between the two hubs (source and target)
        # this is important for graphs with double or triple edges for instance between nodes
        g_without_blue_edges = delete_color_edges(self.g, "blue")

        if ignore_null_edges:
            init_capacity = nx.get_edge_attributes(g_without_blue_edges, "capacity")
            edges_to_remove_null_capacity = [edge for edge, capacity in
                                         init_capacity.items() if capacity == 0.]
            g_without_blue_edges.remove_edges_from(edges_to_remove_null_capacity)

        for source, target in zip(hub_sources, hub_targets):
            paths = nx.all_simple_edge_paths(g_without_blue_edges, source, target)
            for path in paths:
                all_edges_to_recolor += path

        all_edges_to_recolor=set(all_edges_to_recolor)

        current_colors = nx.get_edge_attributes(self.g, 'color')
        #all_edges_to_recolor=
        edge_attribues_to_set = {edge: {"color": "coral"} for i,edge in enumerate(g_without_blue_edges.edges) if edge in all_edges_to_recolor and current_colors[edge]=="gray"}
        nx.set_edge_attributes(self.g, edge_attribues_to_set)

    def set_hubs_shape(self, hubs: Iterable[Any], shape_hub: str = "circle") -> None:
        """
        Distinguish the shape of "hub" nodes to make them more visible

        Parameters
        ----------

        hub: ``array``
            list of nodes that are hubs in the structured graph

        shape_hub: ``str``
            shape type for drawing the hubs
        """
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
        """
        Highlight lines that could get overloaded and should be monitored. Edge label is augmented with change in loading rate
        before and after the constrained line cut

        WARNING: apply this at the end of the process before ploting the graph, as it changes the edge colors for these target lines,
        but colors are used in other part of the processing and if they are change before, this could create interferences

        Parameters
        ----------

        dict_line_loading: ``dict``
            dict of lines to monitor with "before" and "after" loading rate values

        """
        edge_names = nx.get_edge_attributes(self.g, "name")
        edge_colors = nx.get_edge_attributes(self.g, "color")
        edge_x_labels = nx.get_edge_attributes(self.g, "label")
        label_font_color = {edge: "black" for edge in edge_names.keys()}
        color_label_highlight = "darkred"  # "gold"

        for edge, edge_name in edge_names.items():
            if edge_name in dict_line_loading:
                current_x_lable = edge_x_labels[edge]
                current_edge_color = edge_colors[edge]

                # update edge labels for loaded lines with loading change
                if current_edge_color == "black":  # this is a constraint, highlight initial overloading rate
                    edge_x_labels[
                        edge] = f'< {current_x_lable} <BR/>  <B>{dict_line_loading[edge_name]["before"]}%</B>  → {dict_line_loading[edge_name]["after"]}%>'
                else:
                    edge_x_labels[
                        edge] = f'< {current_x_lable} <BR/>  {dict_line_loading[edge_name]["before"]}% → <B>{dict_line_loading[edge_name]["after"]}%</B> >'

                # update font color and edge color for highlighting
                label_font_color[edge] = color_label_highlight
                edge_colors[edge] = f'"{current_edge_color}:yellow:{current_edge_color}"'

        # edge_x_label=[edge_x_label+"\n "+str(dict_line_loading[edge_name["before"]])+"% -> "+str(dict_line_loading[edge_name["after"]])+"%" else edge_x_label for edge_name,edge_x_label in zip(edge_names,edge_x_label) if edge_name in dict_line_loading]
        nx.set_edge_attributes(self.g, edge_x_labels, "label")
        nx.set_edge_attributes(self.g, label_font_color, "fontcolor")
        nx.set_edge_attributes(self.g, edge_colors, "color")

    def plot(self, layout: Optional[List[Any]], rescale_factor: Optional[float] = None, allow_overlap: bool = True, fontsize: Optional[int] = None, node_thickness: int = 3, save_folder: str = "", without_gray_edges: bool = False) -> Any:
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

    def consolidate_graph(self, structured_graph: Any, non_connected_lines_to_ignore: List[Any] = [], no_desambiguation: bool = False) -> None:
        """
        Consolidate overflow graph knwoing structural elements from SuscturedOverflowGraph

        Parameters
        ----------

        structured_graph: ``SuscturedOverflowGraph``
            a structured graph with identified constrained path, hubs, loop paths

        """
        #remove temporarily edges
        # Get the names of the edges in the graph
        edge_names = nx.get_edge_attributes(self.g, 'name')
        edges_to_remove = [edge for edge, edge_name in edge_names.items() if
                           edge_name in non_connected_lines_to_ignore]
#
        edges_to_remove_data = [(edge_or, edge_ex, edge_properties) for edge_or, edge_ex, edge_properties in
                                self.g.edges(data=True) if
                                edge_properties["name"] in non_connected_lines_to_ignore]

        self.g.remove_edges_from(edges_to_remove)

        # consolider le chemin en contrainte avec la connaissance des hubs, en itérant une fois de plus
        n_hubs_init = 0
        hubs_paths = structured_graph.find_loops()[["Source", "Target"]].drop_duplicates()
        n_hub_paths = hubs_paths.shape[0]

        while n_hubs_init != n_hub_paths:
            n_hubs_init = n_hub_paths

            constrained_path = structured_graph.constrained_path
            nodes_amont = constrained_path.n_amont()
            nodes_aval = constrained_path.n_aval()
            constrained_path_edges = constrained_path.aval_edges + [
                constrained_path.constrained_edge] + constrained_path.amont_edges
            self.consolidate_constrained_path(nodes_amont, nodes_aval, constrained_path_edges)

            structured_graph = Structured_Overload_Distribution_Graph(self.g)

            hubs_paths = structured_graph.find_loops()[["Source", "Target"]].drop_duplicates()
            n_hub_paths = hubs_paths.shape[0]

        #recolor and reverse blue or red edges outside of constrained or loop paths
        if not no_desambiguation:
            ambiguous_edge_paths, ambiguous_node_paths = self.identify_ambiguous_paths(structured_graph)
            for ambiguous_edge_path, ambiguous_node_path in zip(ambiguous_edge_paths, ambiguous_node_paths):
                path_type=self.desambiguation_type_path(ambiguous_node_path, structured_graph)
                if path_type=="loop_path":
                    self.reverse_edges(ambiguous_edge_path,target_color="coral")
                else:
                    self.reverse_edges(ambiguous_edge_path, target_color="blue")

        #not needed anymore as more generic ambiguous path detection and correction above ?
        #constrained_path = structured_graph.constrained_path.full_n_constrained_path()
        #self.reverse_blue_edges_in_looppaths(constrained_path)

        # consolidate loop paths by recoloring gray edges that are significant enough and within a loop path
        self.consolidate_loop_path(hubs_paths.Source, hubs_paths.Target)

        #add back removed edges
        self.g.add_edges_from(edges_to_remove_data)

    def identify_ambiguous_paths(self, structured_graph: Any) -> Tuple[List[Any], List[Any]]:
        """
        Identify ambiguous paths in the structured graph.

        An ambiguous path is one that contains both red and blue edges. These paths need to be desambiguated
        to determine which color (red or blue) should be kept for further analysis.

        Parameters
        ----------
        structured_graph : Structured_Overload_Distribution_Graph
            A structured graph with identified constrained path, hubs, loop paths.

        Returns
        -------
        tuple
            A tuple containing two lists:
            - ambiguous_edge_paths: List of lists, where each inner list contains the names of edges in an ambiguous path.
            - ambiguous_node_paths: List of sets, where each set contains the nodes in an ambiguous path.
        """

        # Get the graph without gray and constrained edges
        g_red_blue_ambiguous = structured_graph.g_without_gray_and_c_edge

        # Get the names of the edges in the graph
        edge_names = nx.get_edge_attributes(structured_graph.g_without_gray_and_c_edge, 'name')

        # Identify lines that are part of the constrained path and dispatch path
        lines_constrained_path, nodes_constrained_path,other_blue_edges, other_blue_nodes  = structured_graph.get_constrained_edges_nodes()
        lines_dispatch, nodes_dispatch_path = structured_graph.get_dispatch_edges_nodes()

        # Remove edges that are part of the constrained path or dispatch path from the graph
        edges_to_remove = [edge for edge, edge_name in edge_names.items() if
                           edge_name in lines_constrained_path + lines_dispatch]
        g_red_blue_ambiguous.remove_edges_from(edges_to_remove)

        # Find weakly connected components in the graph
        weak_comps = nx.weakly_connected_components(g_red_blue_ambiguous)

        # Initialize lists to store ambiguous edge paths and node paths
        ambiguous_edge_paths = []
        ambiguous_node_paths = []

        # Iterate over each weakly connected component
        for c in weak_comps:
            # If the component has at least two nodes, it is considered ambiguous
            if len(c) >= 2:
                #check if two colors
                comp_colors=np.unique(list(nx.get_edge_attributes(g_red_blue_ambiguous.subgraph(c), "color").values()))
                if "blue" in comp_colors and "coral" in comp_colors and len(comp_colors)==2:#blue and coral
                    ambiguous_node_paths.append(c)
                    # Get the names of the edges in the subgraph of the component
                    ambiguous_edge_paths.append(list(nx.get_edge_attributes(g_red_blue_ambiguous.subgraph(c), "name").values()))

        return ambiguous_edge_paths, ambiguous_node_paths

    def desambiguation_type_path(self, ambiguous_node_path: Iterable[Any], structured_graph: Any) -> str:
        """
        Desambiguates the type of path for ambiguous nodes based on the structured graph.

        This method determines whether the ambiguous path is part of the constrained path or a loop path.
        It uses the structured graph to identify the nodes and edges that are part of the constrained path
        and loop paths. Based on this information, it classifies the ambiguous path as either a constrained
        path or a loop path.

        Parameters
        ----------
        ambiguous_node_path : list
            A list of nodes that are part of an ambiguous path. These nodes need to be classified as either
            part of the constrained path or a loop path.

        structured_graph : Structured_Overload_Distribution_Graph
            A structured graph with identified constrained path, hubs, and loop paths.

        Returns
        -------
        str
            A string indicating the type of path: "constrained_path" or "loop_path".
        """
        # Get the constrained path object from the structured graph
        constrained_path_object = structured_graph.constrained_path

        # Get the full list of nodes in the constrained path
        nodes_constrained_path = constrained_path_object.full_n_constrained_path()

        # Find nodes in the ambiguous path that are also in the constrained path
        path_nodes_in_c_path = [node for node in ambiguous_node_path if node in nodes_constrained_path]

        # If there are at least two nodes in the ambiguous path that are in the constrained path
        if len(path_nodes_in_c_path) >= 2:
            # Get the nodes upstream (amont) and downstream (aval) of the constrained path
            nodes_amont = constrained_path_object.n_amont()
            nodes_aval = constrained_path_object.n_aval()

            # Check if the ambiguous path connects to nodes upstream of the constrained path
            do_path_connect_amont = any([node in nodes_amont for node in path_nodes_in_c_path])

            # Check if the ambiguous path connects to nodes downstream of the constrained path
            do_path_connect_aval = any([node in nodes_aval for node in path_nodes_in_c_path])

            # If the ambiguous path connects to both upstream and downstream nodes, it is a loop path
            if do_path_connect_amont and do_path_connect_aval:
                return "loop_path"
            else:
                # Otherwise, it is part of the constrained path
                return "constrained_path"
        elif len(path_nodes_in_c_path)==1:#if the edge connected to the constrained path is red, we can consider it on a loop path
            return "loop_path"#"constrained_path"
        else:
            # If there are fewer than two nodes in the ambiguous path that are in the constrained path,
            # it is classified as a loop path
            return "loop_path"

    def rename_nodes(self, mapping: Dict[Any, Any]) -> None:
        self.g = nx.relabel_nodes(self.g, mapping, copy=True)
        self.df["idx_or"]=[mapping[idx_or] for idx_or in self.df["idx_or"]]
        self.df["idx_ex"] = [mapping[idx_or] for idx_or in self.df["idx_ex"]]

    def _setup_null_flow_styles(self, non_connected_lines: List[Any], non_reconnectable_lines: List[Any]) -> List[Any]:
        """Set ``style`` (dotted/dashed) and ``dir`` on every edge matching a
        non-connected or non-reconnectable line name. Returns the union of
        the two input lines so the caller can reuse it."""
        union_lines = list(set(non_connected_lines) | set(non_reconnectable_lines))

        edge_names = nx.get_edge_attributes(self.g, 'name')
        non_reconnectable_set = set(non_reconnectable_lines)
        non_connected_edges = {e for e, n in edge_names.items() if n in union_lines}
        non_reconnectable_edges = {e for e, n in edge_names.items() if n in non_reconnectable_set}
        reconnectable_edges = non_connected_edges - non_reconnectable_edges

        nx.set_edge_attributes(self.g, {e: {"style": "dotted"} for e in non_reconnectable_edges})
        nx.set_edge_attributes(self.g, {e: {"style": "dashed"} for e in reconnectable_edges})
        nx.set_edge_attributes(self.g, {e: "none" for e in non_reconnectable_edges}, "dir")
        return union_lines

    def add_relevant_null_flow_lines_all_paths(self, structured_graph: Any, non_connected_lines: List[Any], non_reconnectable_lines: List[Any] = []) -> None:
        """
        Make edges bi-directionnal when flow redispatch value is null

        Parameters
        ----------

        structured_graph: ``SuscturedOverflowGraph``
            a structured graph with identified constrained path, hubs, loop paths


        non_connected_lines: ``array``
            list of lines that are non connected but that could be reconnected and that we want to highlight if relevant

        """
        # One-time setup: styles and directions (idempotent across target_path iterations)
        non_connected_lines = self._setup_null_flow_styles(non_connected_lines, non_reconnectable_lines)

        # Pre-compute structural info that is the same for all target_paths
        node_red_paths = []
        if structured_graph.red_loops.Path.shape[0] != 0:
            node_red_paths = set(structured_graph.g_only_red_components.nodes)
        node_amont_constrained_path = structured_graph.constrained_path.n_amont()
        node_aval_constrained_path = structured_graph.constrained_path.n_aval()

        structural_info = {
            "node_red_paths": node_red_paths,
            "node_amont_constrained_path": node_amont_constrained_path,
            "node_aval_constrained_path": node_aval_constrained_path,
        }

        for target_path in ["blue_amont_aval", "red_only", "blue_to_red", "blue_only"]:
            self.add_relevant_null_flow_lines(structured_graph, non_connected_lines, non_reconnectable_lines,
                                              target_path=target_path,
                                              _skip_style_setup=True,
                                              _structural_info=structural_info)





    def add_relevant_null_flow_lines(self, structured_graph: Any, non_connected_lines: List[Any],
                                     non_reconnectable_lines: List[Any] = [], target_path: str = "blue_to_red",
                                     depth_reconnectable_edges_search: int = 2,
                                     max_null_flow_path_length: int = 7,
                                     _skip_style_setup: bool = False, _structural_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Make edges bi-directional when flow redispatch is null, recolor the
        relevant ones that could be of interest for analyzing or solving the
        problem, and get back to initial edges for the others.

        The orchestration is split across four helpers:

        - :meth:`_prepare_null_flow_edge_sets` computes the line / edge sets
          and "connex" candidates up front,
        - :meth:`_build_gray_components` materialises the gray-only weakly
          connected components used as search subgraphs,
        - :meth:`_detect_edges_for_target_path` runs the per-component
          ``detect_edges_to_keep`` calls for one of the four target-path
          strategies (``blue_only`` / ``blue_amont_aval`` / ``red_only`` /
          ``blue_to_red``),
        - :meth:`_apply_null_flow_recoloring` paints the resulting edges and
          cleans up unused doubled edges.

        Parameters
        ----------
        structured_graph: ``Structured_Overload_Distribution_Graph``
            a structured graph with identified constrained path, hubs, loop paths
        non_connected_lines: list[str]
            list of lines that are non connected but could be reconnected
        non_reconnectable_lines: list[str]
            list of lines that are non connected and cannot be reconnected
        target_path: str
            one of ``blue_only``, ``blue_amont_aval``, ``red_only``, ``blue_to_red``
        """
        if not _skip_style_setup:
            non_connected_lines = self._setup_null_flow_styles(
                non_connected_lines, non_reconnectable_lines)

        sets = self._prepare_null_flow_edge_sets(non_connected_lines, non_reconnectable_lines)

        # Make null-flow-redispatch lines bidirectional inside self.g (this
        # mutates self.g and returns the edges we added so we can roll back
        # the ones that turned out not to be useful at the end).
        edges_to_double, edges_double_added = add_double_edges_null_redispatch(self.g)

        # Refresh edge sets now that new edges exist
        edge_names = nx.get_edge_attributes(self.g, 'name')
        edges_non_connected_lines = {
            edge for edge, n in edge_names.items() if n in sets["non_connected_lines_set"]
        }
        edges_non_reconnectable_lines = {
            edge for edge, n in edge_names.items() if n in sets["non_reconnectable_lines_set"]
        }

        gray_components = self._build_gray_components()

        structural_info = _structural_info or self._structural_info_for_null_flow(structured_graph)

        edges_to_keep, edges_non_reconnectable = self._detect_edges_for_target_path(
            gray_components, target_path, structural_info,
            sets["edges_non_connected_lines_to_consider"],
            edges_non_connected_lines,
            edges_non_reconnectable_lines,
            depth_reconnectable_edges_search,
            max_null_flow_path_length,
        )

        self._apply_null_flow_recoloring(
            target_path, edges_to_keep, edges_non_reconnectable,
            edges_to_double, edges_double_added,
        )

    # ------------------------------------------------------------------
    # Helpers for add_relevant_null_flow_lines
    # ------------------------------------------------------------------

    def _prepare_null_flow_edge_sets(self, non_connected_lines: List[Any], non_reconnectable_lines: List[Any]) -> Dict[str, Any]:
        """
        Compute the input edge sets used by :meth:`add_relevant_null_flow_lines`:
        the plain line-name sets, plus ``edges_non_connected_lines_to_consider``
        which keeps only lines "connex" to at least one non-gray edge (i.e.
        lines whose reconnection could plausibly matter for the analysis).
        """
        non_connected_lines_set = set(non_connected_lines)
        non_reconnectable_lines_set = set(non_reconnectable_lines)
        edge_names = nx.get_edge_attributes(self.g, 'name')

        # Nodes with at least one non-gray edge
        edge_colors = nx.get_edge_attributes(self.g, 'color')
        nodes_coloured = set()
        for edge, color in edge_colors.items():
            if color != "gray":
                nodes_coloured.add(edge[0])
                nodes_coloured.add(edge[1])

        # Connex gray edge names — touching at least one coloured node
        edge_connex_names = set()
        for edge, color in edge_colors.items():
            if color == "gray" and (edge[0] in nodes_coloured or edge[1] in nodes_coloured):
                name = edge_names.get(edge)
                if name:
                    edge_connex_names.add(name)

        non_connected_lines_to_consider = non_connected_lines_set & edge_connex_names
        edges_non_connected_lines_to_consider = {
            edge for edge, n in edge_names.items() if n in non_connected_lines_to_consider
        }

        return {
            "non_connected_lines_set": non_connected_lines_set,
            "non_reconnectable_lines_set": non_reconnectable_lines_set,
            "edges_non_connected_lines_to_consider": edges_non_connected_lines_to_consider,
        }

    def _build_gray_components(self) -> List[Any]:
        """
        Build the list of weakly connected components consisting only of
        "gray" edges (i.e. non-coral/blue/black). Components are returned
        sorted by size ascending and each is a mutable copy, so the caller
        can freely drop positive- or negative-capacity edges before running
        path searches on them.
        """
        _EXCLUDED_COLORS = frozenset({"coral", "blue", "black"})
        g_only_gray_components = nx.MultiDiGraph()
        for u, v, k, data in self.g.edges(keys=True, data=True):
            if data.get("color") not in _EXCLUDED_COLORS:
                g_only_gray_components.add_edge(u, v, key=k, **data)
        g_only_gray_components.remove_nodes_from(list(nx.isolates(g_only_gray_components)))

        return [
            g_only_gray_components.subgraph(c).copy()
            for c in sorted(
                nx.weakly_connected_components(g_only_gray_components),
                key=len, reverse=False,
            )
        ]

    @staticmethod
    def _structural_info_for_null_flow(structured_graph: Any) -> Dict[str, Any]:
        """Extract the red/amont/aval node sets once per call."""
        node_red_paths = []
        if structured_graph.red_loops.Path.shape[0] != 0:
            node_red_paths = set(structured_graph.g_only_red_components.nodes)
        return {
            "node_red_paths": node_red_paths,
            "node_amont_constrained_path": structured_graph.constrained_path.n_amont(),
            "node_aval_constrained_path": structured_graph.constrained_path.n_aval(),
        }

    def _detect_edges_for_target_path(self, gray_components: List[Any], target_path: str, structural_info: Dict[str, Any],
                                      edges_non_connected_lines_to_consider: Set[Any],
                                      edges_non_connected_lines: Set[Any],
                                      edges_non_reconnectable_lines: Set[Any],
                                      depth_reconnectable_edges_search: int,
                                      max_null_flow_path_length: int) -> Tuple[Set[Any], Set[Any]]:
        """
        Per-component dispatch to :meth:`detect_edges_to_keep` based on the
        chosen ``target_path`` strategy. Each strategy picks a different
        source / target node set on the shared structural_info and may
        additionally prune positive- or negative-capacity edges from the
        mutable gray component before running the search.
        """
        node_red_paths = structural_info["node_red_paths"]
        node_amont = structural_info["node_amont_constrained_path"]
        node_aval = structural_info["node_aval_constrained_path"]

        edges_to_keep = set()
        edges_non_reconnectable = set()

        def _run(g_c, sources, targets):
            keep, non_rec = self.detect_edges_to_keep(
                g_c, sources, targets,
                edges_non_connected_lines, edges_non_reconnectable_lines,
                depth_edges_search=depth_reconnectable_edges_search,
                max_null_flow_path_length=max_null_flow_path_length)
            edges_to_keep.update(keep)
            edges_non_reconnectable.update(non_rec)

        for g_c in gray_components:
            if not edges_non_connected_lines_to_consider.intersection(set(g_c.edges)):
                continue

            if target_path == "blue_only":
                # Looking for blue (negative) edge paths — drop positive-capacity edges
                edges_to_remove = [
                    edge for edge, capacity in nx.get_edge_attributes(g_c, "capacity").items()
                    if capacity > 0.
                ]
                g_c.remove_edges_from(edges_to_remove)

                intersect_amont = set(g_c).intersection(node_amont)
                intersect_aval = set(g_c).intersection(node_aval)
                _run(g_c, intersect_amont, intersect_amont)
                _run(g_c, intersect_aval, intersect_aval)

            elif target_path == "blue_amont_aval":
                intersect_amont = set(g_c).intersection(node_amont)
                intersect_aval = set(g_c).intersection(node_aval)
                _run(g_c, intersect_amont, intersect_aval)

            elif target_path == "red_only":
                # Looking for red (positive) edge paths — drop negative-capacity edges
                edges_to_remove = [
                    edge for edge, capacity in nx.get_edge_attributes(g_c, "capacity").items()
                    if capacity < 0.
                ]
                g_c.remove_edges_from(edges_to_remove)

                intersect_red = set(g_c).intersection(node_red_paths)
                _run(g_c, intersect_red, intersect_red)

            elif target_path == "blue_to_red":
                intersect_amont = set(g_c).intersection(node_amont)
                intersect_aval = set(g_c).intersection(node_aval)
                intersect_red = set(g_c).intersection(node_red_paths)

                if intersect_amont:
                    _run(g_c, intersect_amont, intersect_red)
                if intersect_aval:
                    _run(g_c, intersect_red, intersect_aval)
                # Potential new loop path using disconnected lines
                if intersect_amont and intersect_aval:
                    _run(g_c, intersect_amont, intersect_aval)

        return edges_to_keep, edges_non_reconnectable

    def _apply_null_flow_recoloring(self, target_path: str, edges_to_keep: Set[Any], edges_non_reconnectable: Set[Any],
                                    edges_to_double: Dict[Any, Any], edges_double_added: Dict[Any, Any]) -> None:
        """
        Paint the detected edges and roll back the double-edges that were
        not used. Blue/red colours follow the ``target_path`` strategy; for
        ``blue_to_red`` we additionally look at the current capacity sign so
        negative edges stay blue even inside a red-tagged path.
        """
        if target_path == "blue_only":
            edge_attributes = {edge: {"color": "blue"} for edge in edges_to_keep}
        elif target_path == "blue_to_red":
            current_weights = nx.get_edge_attributes(self.g, 'capacity')
            edge_attributes = {edge: {"color": "coral"} for edge in edges_to_keep}
            edge_attributes.update({
                edge: {"color": "blue"}
                for edge in edges_to_keep
                if current_weights[edge] < 0
            })
        else:
            edge_attributes = {edge: {"color": "coral"} for edge in edges_to_keep}

        # Non-reconnectable edges that were still gray get marked dimgray
        edge_attributes.update({
            edge: {"color": "dimgray"}
            for edge in edges_non_reconnectable
            if self.g.edges[edge]["color"] == "gray"
        })

        nx.set_edge_attributes(self.g, edge_attributes)

        # Kept edges that came from the "double edge" trick should be drawn
        # without an arrow tip (they represent null-flow lines).
        doubled_edges = set(edges_to_double.values()) | set(edges_double_added.values())
        edge_dirs = {edge: "none" for edge in edges_to_keep.intersection(doubled_edges)}
        nx.set_edge_attributes(self.g, edge_dirs, "dir")

        # Roll back unused doubled edges
        self.g = remove_unused_added_double_edge(
            self.g, edges_to_keep, edges_to_double, edges_double_added)

    def detect_edges_to_keep(self, g_c: nx.MultiDiGraph, source_nodes: Iterable[Any], target_nodes: Iterable[Any], edges_of_interest: Set[Any],
                             non_reconnectable_edges: List[Any] = [], depth_edges_search: int = 2,
                             max_null_flow_path_length: int = 7) -> Tuple[Set[Any], Set[Any]]:
        """
        Detect edges in ``edges_of_interest`` that lie on a short path between
        ``source_nodes`` and ``target_nodes`` inside the subgraph ``g_c``.

        The work is split across five helpers:

        - :meth:`_prepare_detect_edges_inputs` handles every early-exit case,
          flips negative capacities, and collects the per-node metadata.
        - :meth:`_compute_sssp_paths` runs one single-source Dijkstra per
          source node with an incentivised weight function.
        - :meth:`_collect_paths_of_interest` materialises the shortest paths
          that actually traverse an edge of interest.
        - :meth:`_classify_paths_by_reconnectability` splits the collected
          paths into reconnectable / non-reconnectable edges.

        Returns
        -------
        (set, set)
            ``(reconnectable_edges, non_reconnectable_edges)`` — both sets of
            MultiDiGraph edge keys.
        """
        prepared = self._prepare_detect_edges_inputs(
            g_c, source_nodes, target_nodes, edges_of_interest,
            non_reconnectable_edges, depth_edges_search)
        if prepared is None:
            return set(), set()

        sssp_paths_cache = self._compute_sssp_paths(g_c, prepared, edges_of_interest)
        paths_of_interest = self._collect_paths_of_interest(
            g_c, prepared, sssp_paths_cache, max_null_flow_path_length)
        return self._classify_paths_by_reconnectability(prepared, paths_of_interest)

    # ------------------------------------------------------------------
    # Helpers for detect_edges_to_keep
    # ------------------------------------------------------------------

    def _prepare_detect_edges_inputs(self, g_c: nx.MultiDiGraph, source_nodes: Iterable[Any], target_nodes: Iterable[Any], edges_of_interest: Set[Any],
                                     non_reconnectable_edges: List[Any], depth_edges_search: int) -> Optional[Dict[str, Any]]:
        """
        Run the up-front bookkeeping shared by every branch of
        :meth:`detect_edges_to_keep`:

        - filter ``edges_of_interest`` / ``source_nodes`` / ``target_nodes``
          down to what exists in ``g_c``,
        - flip the sign of any negative-capacity edge (single pass),
        - pre-compute which nodes have an incident edge of interest,
        - pre-compute per-node BFS results for the edge-name search.

        Returns ``None`` if any early-exit condition is met; otherwise a
        dictionary with the cached structures for downstream helpers.
        """
        g_c_edge_names_dict = nx.get_edge_attributes(g_c, "name")

        edges_of_interest_in_gc = edges_of_interest & set(g_c_edge_names_dict.keys())
        if not edges_of_interest_in_gc:
            return None

        edge_names_of_interest = {g_c_edge_names_dict[edge] for edge in edges_of_interest_in_gc}
        non_reconnectable_edges_names = {
            g_c_edge_names_dict[edge] for edge in non_reconnectable_edges
            if edge in g_c_edge_names_dict
        }

        # Flip negative capacities once (single pass)
        new_attributes_dict = {
            e: {"capacity": -capacity}
            for e, capacity in nx.get_edge_attributes(g_c, "capacity").items()
            if capacity < 0
        }
        if new_attributes_dict:
            nx.set_edge_attributes(g_c, new_attributes_dict)

        source_nodes_in_gc = [s for s in source_nodes if s in g_c]
        target_nodes_in_gc = [t for t in target_nodes if t in g_c]
        if not source_nodes_in_gc or not target_nodes_in_gc:
            return None

        unique_nodes = set(source_nodes_in_gc) | set(target_nodes_in_gc)
        node_has_incident_interest = {}
        for node in unique_nodes:
            incident = set(g_c.out_edges(node, keys=True)) | set(g_c.in_edges(node, keys=True))
            node_has_incident_interest[node] = bool(incident & edges_of_interest_in_gc)

        if not any(node_has_incident_interest.values()):
            return None

        bfs_cache = {
            node: find_multidigraph_edges_by_name(
                g_c, node, edge_names_of_interest,
                depth=depth_edges_search, name_attr="name")
            for node in unique_nodes
        }

        targets_with_bfs = frozenset(t for t in target_nodes_in_gc if bfs_cache[t])
        any_target_has_interest = any(node_has_incident_interest[t] for t in target_nodes_in_gc)

        return {
            "g_c_edge_names_dict": g_c_edge_names_dict,
            "edges_of_interest_in_gc": edges_of_interest_in_gc,
            "edge_names_of_interest": edge_names_of_interest,
            "non_reconnectable_edges_names": non_reconnectable_edges_names,
            "source_nodes_in_gc": source_nodes_in_gc,
            "target_nodes_in_gc": target_nodes_in_gc,
            "node_has_incident_interest": node_has_incident_interest,
            "bfs_cache": bfs_cache,
            "targets_with_bfs": targets_with_bfs,
            "any_target_has_interest": any_target_has_interest,
        }

    def _compute_sssp_paths(self, g_c: nx.MultiDiGraph, prepared: Dict[str, Any], edges_of_interest: Set[Any]) -> Dict[Any, Any]:
        """
        Run single-source Dijkstra once per source node with a weight function
        that massively favours low-capacity edges and gently promotes edges
        of interest when capacities tie. Returns a dict
        ``source_node -> {target_node -> path_nodes}``.
        """
        HUGE_MULTIPLIER = 1_000_000_000
        NORMAL_HOP_COST = 100
        PROMOTED_HOP_COST = 33
        promoted_set = set(edges_of_interest)

        def incentivized_weight(u, v, attr):
            real_weight = attr.get("capacity", 0)
            if real_weight < 0:
                raise ValueError("Negative weights not allowed.")
            is_promoted = (u, v) in promoted_set
            hop_cost = PROMOTED_HOP_COST if is_promoted else NORMAL_HOP_COST
            return (real_weight * HUGE_MULTIPLIER) + hop_cost

        bfs_cache = prepared["bfs_cache"]
        targets_with_bfs = prepared["targets_with_bfs"]
        node_has_incident_interest = prepared["node_has_incident_interest"]
        any_target_has_interest = prepared["any_target_has_interest"]

        sssp_paths_cache = {}
        for source_node in set(prepared["source_nodes_in_gc"]):
            if not node_has_incident_interest[source_node] and not any_target_has_interest:
                continue
            if not bfs_cache[source_node] and not targets_with_bfs:
                continue
            try:
                sssp_paths_cache[source_node] = nx.single_source_dijkstra_path(
                    g_c, source_node, weight=incentivized_weight)
            except Exception:
                sssp_paths_cache[source_node] = {}
        return sssp_paths_cache

    def _collect_paths_of_interest(self, g_c: nx.MultiDiGraph, prepared: Dict[str, Any], sssp_paths_cache: Dict[Any, Any], max_null_flow_path_length: int) -> List[Any]:
        """
        Walk the (source, target) product and materialise the paths whose
        node-count is within ``max_null_flow_path_length`` and which actually
        traverse at least one edge of interest. Paths are returned sorted by
        length so the downstream classifier can dedupe greedily.
        """
        source_nodes_in_gc = prepared["source_nodes_in_gc"]
        target_nodes_in_gc = prepared["target_nodes_in_gc"]
        bfs_cache = prepared["bfs_cache"]
        targets_with_bfs = prepared["targets_with_bfs"]
        node_has_incident_interest = prepared["node_has_incident_interest"]
        edges_of_interest_in_gc = prepared["edges_of_interest_in_gc"]

        paths_of_interest = []
        for source_node in source_nodes_in_gc:
            if source_node not in sssp_paths_cache:
                continue
            source_paths = sssp_paths_cache[source_node]
            source_has_bfs = bool(bfs_cache[source_node])
            source_has_interest = node_has_incident_interest[source_node]

            for target_node in target_nodes_in_gc:
                if source_node == target_node:
                    continue
                if not source_has_interest and not node_has_incident_interest[target_node]:
                    continue
                if not source_has_bfs and target_node not in targets_with_bfs:
                    continue

                path_nodes = source_paths.get(target_node)
                if not path_nodes or len(path_nodes) > max_null_flow_path_length:
                    continue
                path = nodepath_to_edgepath(g_c, path_nodes, with_keys=True)
                if any(edge in edges_of_interest_in_gc for edge in path):
                    paths_of_interest.append(path)

        paths_of_interest.sort(key=len)
        return paths_of_interest

    def _classify_paths_by_reconnectability(self, prepared: Dict[str, Any], paths_of_interest: List[Any]) -> Tuple[Set[Any], Set[Any]]:
        """
        Greedy dedupe over the sorted ``paths_of_interest``: each edge name
        is attributed to the *first* (shortest) path that uses it, and the
        path as a whole is classified as non-reconnectable iff any of its
        newly-attributed edges is in the non-reconnectable edge-name set.
        """
        g_c_edge_names_dict = prepared["g_c_edge_names_dict"]
        edge_names_of_interest = prepared["edge_names_of_interest"]
        non_reconnectable_edges_names = prepared["non_reconnectable_edges_names"]

        edge_names_already_found = set()
        edges_to_keep_reconnectable = []
        edges_to_keep_non_reconnectable = []

        for path in paths_of_interest:
            fresh_edges = {
                edge for edge in path
                if g_c_edge_names_dict[edge] not in edge_names_already_found
            }
            fresh_edge_names = {g_c_edge_names_dict[edge] for edge in fresh_edges}

            if not (fresh_edge_names & edge_names_of_interest):
                continue

            if fresh_edge_names & non_reconnectable_edges_names:
                edges_to_keep_non_reconnectable += fresh_edges
            else:
                edges_to_keep_reconnectable += fresh_edges
            edge_names_already_found |= fresh_edge_names

        return set(edges_to_keep_reconnectable), set(edges_to_keep_non_reconnectable)

    def collapse_red_loops(self) -> None:
        """
        Collapse nodes that are purely part of "red loops" (coral-only edges) into point shapes.

        A node is collapsed when all of the following conditions are met:
        - All edges connected to the node (both incoming and outgoing) are "coral" coloured
        - The node shape is simply "oval" (default shape, not a hub)
        - The node has no "peripheries" attribute set (no electrical node number)
        - None of the connected edges have "dashed" or "dotted" style
        """
        shapes = nx.get_node_attributes(self.g, "shape")
        peripheries = nx.get_node_attributes(self.g, "peripheries")
        edge_colors = nx.get_edge_attributes(self.g, "color")
        edge_styles = nx.get_edge_attributes(self.g, "style")

        nodes_to_collapse = {}

        for node in self.g.nodes:
            # Check shape is simply "oval"
            if shapes.get(node) != "oval":
                continue

            # Check no peripheries attribute
            if node in peripheries and peripheries[node]>=2:
                continue

            # Get all edges connected to this node (in and out)
            in_edges = list(self.g.in_edges(node, keys=True))
            out_edges = list(self.g.out_edges(node, keys=True))
            all_edges = in_edges + out_edges

            # Node must have at least one edge
            if not all_edges:
                continue

            # Check all edges are coral and none are dashed/dotted
            all_coral = True
            for edge in all_edges:
                if edge_colors.get(edge) != "coral":
                    all_coral = False
                    break
                style = edge_styles.get(edge, "")
                if style in ("dashed", "dotted"):
                    all_coral = False
                    break

            if all_coral and all_edges:
                nodes_to_collapse[node] = "point"

        nx.set_node_attributes(self.g, nodes_to_collapse, "shape")
