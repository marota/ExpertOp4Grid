"""Structured_Overload_Distribution_Graph: extract constrained path,
loop paths and hubs from a raw overflow graph.
"""

import logging
from typing import Any, List, Optional, Tuple

import networkx as nx
import pandas as pd
import rustworkx as rx

from alphaDeesp.core.graphs.constrained_path import ConstrainedPath
from alphaDeesp.core.graphs.graph_utils import delete_color_edges
from alphaDeesp.core.graphs.null_flow import (
    add_double_edges_null_redispatch,
    remove_unused_added_double_edge,
)

logger = logging.getLogger(__name__)


class Structured_Overload_Distribution_Graph:
    """
    Staring from a raw overload distribution graph with color edges, this class identifies the underlying path structure in terms of constrained path, loop paths and hub nodes
    """
    def __init__(self, g: nx.MultiDiGraph, possible_hubs: Optional[List[Any]] = None) -> None:
        """
        Parameters
        ----------

        g: :class:`nx:MultiDiGraph`
            a raw graph from OverflowGraph

        """
        self.g_init=g
        self.g_without_pos_edges = delete_color_edges(self.g_init, "coral") #graph without loop path that have positive/red-coloured weight edges
        self.g_only_blue_components = delete_color_edges(self.g_without_pos_edges, "gray")
        self.g_only_blue_components = delete_color_edges(self.g_only_blue_components, "dimgray")#also delete those edges of non reconnectable lines that we would want to visualize but is not an operational path in the structured path

        self.g_without_constrained_edge = delete_color_edges(self.g_init, "black")
        self.g_without_gray_and_c_edge = delete_color_edges(self.g_without_constrained_edge, "gray")
        self.g_without_gray_and_c_edge = delete_color_edges(self.g_without_gray_and_c_edge, "dimgray")
        self.g_only_red_components = delete_color_edges(self.g_without_gray_and_c_edge, "blue")#graph with only loop path that have positive/red-coloured weight edges

        self.constrained_path= self.find_constrained_path() #constrained path that contains the constrained edges and their connected component of blue edges
        self.type=""#
        if possible_hubs is not None:#in case we already have a subset of candidates, for instance when we already built a first Overload graph and are consolidating it
            self.hubs=possible_hubs
        else:
            self.hubs=[]
        self.red_loops = self.find_loops() #parallel path to the constrained path on which flow can be rerouted
        self.hubs = self.find_hubs() #specific nodes at substations connecting loop paths to constrained path. This is where flow can be most easily rerouted

    def get_amont_blue_edges(self, g: nx.MultiDiGraph, node: Any) -> List[Any]:
        """
        From a given node, get blue edges (with negative overflow redispatch) that are above this node

        Parameters
        ----------

        g: :class:`nx:MultiDiGraph`
            an overflow redispatch networkx graph

        node: int
            node of interest

        Returns
        ----------

        res: ``array`` int
            ordered list of edges

        """
        res = []
        for e in nx.edge_dfs(g, node, orientation="reverse"):
            if g.edges[(e[0], e[1],e[2])]["color"] == "blue":
                res.append((e[0], e[1],e[2]))
        return res

    def get_aval_blue_edges(self, g: nx.MultiDiGraph, node: Any) -> List[Any]:
        """
        From a given node, get blue edges (with negative overflow redispatch) that are after this node

        Parameters
        ----------

        g: :class:`nx:MultiDiGraph`
            an overflow redispatch networkx graph

        node: int
            node of interest

        Returns
        ----------

        res: ``array`` int
            ordered list of edges

        """
        res = []
        # print("debug AlphaDeesp get aval blue edges")
        # print(list(nx.edge_dfs(g, node, orientation="original")))
        for e in nx.edge_dfs(g, node, orientation="original"):
            if g.edges[(e[0], e[1],e[2])]["color"] == "blue":
                res.append((e[0], e[1],e[2]))
        return res


    def find_hubs(self) -> List[Any]:
        """
        "A hub (carrefour_electrique) has a constrained_path and positiv reports"

        Returns
        ----------

        res: list int
            a list of nodes that are detected as hubs
        """
        g = self.g_without_constrained_edge
        hubs = []

        if self.constrained_path is not None:
            logger.debug("In get_hubs(): constrained_path = %s", self.constrained_path)
        else:
            e_amont, constrained_edge, e_aval = self.get_constrained_path()
            self.constrained_path = ConstrainedPath(e_amont, constrained_edge, e_aval)

        # for nodes in aval, if node has RED inputs (ie incoming flows) then it is a hub
        for node in self.constrained_path.n_aval():
            in_edges = list(g.in_edges(node,keys=True))
            for e in in_edges:
                if g.edges[e]["color"] == "coral":
                    hubs.append(node)
                    break

        # for nodes in amont, if node has RED outputs (ie outgoing flows) then it is a hub
        for node in self.constrained_path.n_amont():
            out_edges = list(g.out_edges(node,keys=True))
            for e in out_edges:
                if g.edges[e]["color"] == "coral":
                    hubs.append(node)
                    break

        # print("get_hubs = ", hubs)
        return hubs

    def get_hubs(self) -> List[Any]:
        return self.hubs

    def find_loops(self) -> pd.DataFrame:

        """This function returns all parallel paths. After discussing with Antoine, start with the most "en Aval" node,
        and walk in reverse for loops and parallel path returns a dict with all data

        Returns
        ----------

        res: pd.DataFrame
            a dataframe with rows representing each detected path, with column attibutes "Source, Target, Path" with Path representing a list of nodes
        """

        attr_edge_direction=nx.get_edge_attributes(self.g_only_red_components, "dir")
        if len(attr_edge_direction)!=0:
            # add edges to make simple paths work for no direction edges
            edges_to_double, edges_double_added = add_double_edges_null_redispatch(self.g_only_red_components,color_init="coral",only_no_dir=True)

        # print("==================== In function get_loops ====================")
        g = self.g_only_red_components
        c_path_n = self.constrained_path.full_n_constrained_path()
        if len(self.hubs)!=0:#already some insights of possible hubs
            c_path_n=self.hubs

        # --- 1. PRE-PROCESSING (Rustworkx) ---
        # Convert NetworkX graph to Rustworkx for 50x speedup
        rx_graph = rx.networkx_converter(g)

        # Map Node Names (Strings) -> Node Indices (Integers)
        nodes_list = list(g.nodes())
        node_map = {node: i for i, node in enumerate(nodes_list)}

        all_loop_paths = []

        # --- 2. SEARCH LOOP ---
        # We iterate efficiently
        for i in range(len(c_path_n)):
            for j in range(len(c_path_n) - 1, i, -1):
                src_name = c_path_n[i]
                tgt_name = c_path_n[j]

                # Ensure nodes exist in the graph to avoid crashes
                if src_name in node_map and tgt_name in node_map:
                    s_idx = node_map[src_name]
                    t_idx = node_map[tgt_name]

                    # Rustworkx: Find all simple paths (FAST)
                    # cutoff=10 is crucial to prevent hanging on large grids
                    paths_indices = rx.all_simple_paths(rx_graph, s_idx, t_idx, min_depth=1)#, cutoff=10)

                    # Convert Indices -> Names
                    # We extend the main list directly
                    paths_names = [[nodes_list[idx] for idx in p] for p in paths_indices]
                    all_loop_paths.extend(paths_names)

        # --- 3. OPTIMIZED DATAFRAME CREATION ---
        # Instead of iterating and appending, we build lists directly.
        # This assumes 'all_loop_paths' is a list of lists: [['A', 'B'], ['C', 'D']]

        if not all_loop_paths:
            # Handle empty case to avoid errors
            data_for_df = {"Source": [], "Target": [], "Path": []}
        else:
            # List comprehensions are significantly faster than .append() loop
            data_for_df = {
                "Source": [p[0] for p in all_loop_paths],
                "Target": [p[-1] for p in all_loop_paths],
                "Path": all_loop_paths
            }

        # --- 4. GRAPH CLEANUP (Your original logic) ---
        if len(attr_edge_direction) != 0:
            # remove added edges that made simple paths working for no direction edges
            self.g_only_red_components = remove_unused_added_double_edge(
                self.g_only_red_components,
                set(edges_to_double.values()),
                edges_to_double,
                edges_double_added
            )

        return pd.DataFrame(data_for_df)

    def get_loops(self) -> pd.DataFrame:
        return self.red_loops

    def find_constrained_path(self) -> "ConstrainedPath":
        """Find and return the constrained path

         Returns
        ----------

        res: :class:`ConstrainedPath`
            a constrained path object
        """
        constrained_edge = None
        edge_list = nx.get_edge_attributes(self.g_only_blue_components, "color")
        for edge, color in edge_list.items():
            if "black" in color:
                constrained_edge = edge
        amont_edges = self.get_amont_blue_edges(self.g_only_blue_components, constrained_edge[0])
        aval_edges = self.get_aval_blue_edges(self.g_only_blue_components, constrained_edge[1])

        return ConstrainedPath(amont_edges,constrained_edge,aval_edges)

    def get_constrained_path(self) -> "ConstrainedPath":
        return self.constrained_path

    def get_constrained_edges_nodes(self) -> Tuple[List[Any], List[Any], List[Any], List[Any]]:
        """
        This function identifies the constrained path within the distribution graph.

        Parameters:
        g_distribution_graph (Structured_Overload_Distribution_Graph): The structured overload distribution graph.

        Returns:
        tuple: A tuple containing two lists:
               - edges_constrained_path: List of edges that are part of the constrained path.
               - nodes_constrained_path: List of nodes that are part of the constrained path.
        """
        constrained_path_object = self.constrained_path#self.find_constrained_path()
        nodes_constrained_path = constrained_path_object.full_n_constrained_path()
        edges_constrained_path = []

        edge_names = nx.get_edge_attributes(self.g_init, 'name')
        edges_constrained_path += [edge_name for edge, edge_name in edge_names.items() if
                                   edge in constrained_path_object.amont_edges]
        edges_constrained_path += [edge_name for edge, edge_name in edge_names.items() if
                                   edge in constrained_path_object.aval_edges]

        if type(constrained_path_object.constrained_edge) is list:
            edges_constrained_path += [edge_name for edge, edge_name in edge_names.items() if
                                       edge in constrained_path_object.constrained_edge]
        else:
            edges_constrained_path.append([edge_name for edge, edge_name in edge_names.items() if
                                           edge == constrained_path_object.constrained_edge][0])

        g_blue=self.g_only_blue_components.copy()
        g_blue.remove_edges_from(edges_constrained_path)

        other_blue_edges=list(g_blue.edges())
        other_blue_nodes=[node for node in g_blue.nodes() if node not in nodes_constrained_path]

        return list(set(edges_constrained_path)), nodes_constrained_path, other_blue_edges, other_blue_nodes

    def get_dispatch_edges_nodes(self, only_loop_paths: bool = True) -> Tuple[List[Any], List[Any]]:
        """
        This function identifies the dispatch path within the distribution graph.

        Parameters:
        g_distribution_graph (Structured_Overload_Distribution_Graph): The structured overload distribution graph.

        Returns:
        tuple: A tuple containing two lists:
               - lines_redispatch: List of lines that are part of the dispatch path.
               - list_nodes_dispatch_path: List of nodes that are part of the dispatch path.
        """
        lines_redispatch=[]
        list_nodes_dispatch_path=[]
        g_red = self.g_only_red_components

        if only_loop_paths:
            list_nodes_dispatch_path = list(set(self.red_loops.Path.sum()))#list(set(self.find_loops()["Path"].sum()))
        else:
            list_nodes_dispatch_path=list(g_red.nodes)

        edge_names_red = nx.get_edge_attributes(g_red, 'name')
        lines_redispatch=[edge_name for edge, edge_name in edge_names_red.items() if
                                (edge[0] in list_nodes_dispatch_path) and (edge[1] in list_nodes_dispatch_path)]

        return lines_redispatch, list_nodes_dispatch_path
