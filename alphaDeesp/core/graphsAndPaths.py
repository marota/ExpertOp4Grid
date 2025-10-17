
import pandas as pd
import networkx as nx
from networkx.exception import NetworkXNoPath
from math import fabs
from alphaDeesp.core.printer import Printer
import numpy as np

default_voltage_colors={400:"red",225:"darkgreen",90:"gold",63:"purple",20:"pink",24:"pink",15:"pink",10:"pink",33:"pink",}#[400., 225.,  63.,  24.,  20.,  33.,  10.]

class PowerFlowGraph:
    """
    A coloured graph of current grid state with productions, consumptions and topology
    """

    def __init__(self, topo,lines_cut,layout=None,float_precision="%.2f"):
        """
        Parameters
        ----------

        topo: :class:`dict`
            dictionnary of two dictionnaries edges and nodes, to represent the grid topologie. edges have attributes "init_flows" representing the power flowing, as well as "idx_or","idx_ex"
             for substation extremities
             Nodes have attributes "are_prods","are_loads" if nodes have any productions or any load, as well as "prods_values","load_values" array enumerating the prod and load values at this node.

        lines_cut: ``array``
            ids of lines disconnected

        float_precision: "str"
            Significant digits for dispalyed values at edges. In the form of "%.2f"
        """
        self.topo=topo
        self.lines_cut=lines_cut
        self.layout=layout
        self.float_precision=float_precision
        self.build_graph()
        #self.g=self.build_powerflow_graph()

    def build_graph(self):
        """This method creates the NetworkX Graph of the grid state"""
        g = nx.MultiDiGraph()

        # Get the id of lines that are disconnected from network
        # lines_cut = np.argwhere(obs.line_status == False)[:, 0]
        topo=self.topo

        # Get the whole topology information
        idx_or = topo["edges"]['idx_or']
        idx_ex = topo["edges"]['idx_ex']
        are_prods = topo["nodes"]['are_prods']
        are_loads = topo["nodes"]['are_loads']
        prods_values = topo["nodes"]['prods_values']
        loads_values = topo["nodes"]['loads_values']
        current_flows = topo["edges"]['init_flows']

        # =========================================== NODE PART ===========================================
        self.build_nodes(g, are_prods, are_loads, prods_values, loads_values)
        # =========================================== EDGE PART ===========================================
        self.build_edges(g, idx_or, idx_ex, edge_weights=current_flows)
        #return g
        self.g=g

    def build_nodes(self,g, are_prods, are_loads, prods_values, loads_values,debug=False):
        """
        Create nodes in graph for current grid state

        Parameters
        ----------

        g: :class:`nx:MultiDiGraph`
            a networkx graph to which to add edges

        are_prods: ``array`` boolean
            if there are productions at each node

        are_loads: ``array`` boolean
            if there are cosnumptions at each node

        prods_values: ``array`` float
            the production values at each node

        loads_values: ``array`` float
            the consumption values at each node

        """
        # =========================================== NODE PART ===========================================
        # print(f"There are {len(are_loads)} nodes")
        prods_iter, loads_iter = iter(prods_values), iter(loads_values)
        i = 0
        # We color the nodes depending if they are production or consumption
        for is_prod, is_load in zip(are_prods, are_loads):
            prod = next(prods_iter) if is_prod else 0.
            load = next(loads_iter) if is_load else 0.
            prod_minus_load = prod - load
            if debug:
                print(f"Node n°[{i}] : Production value: [{prod}] - Load value: [{load}] ")
            if prod_minus_load > 0:  # PROD
                g.add_node(i, pin=True, prod_or_load="prod", value=str(prod_minus_load), style="filled",
                           fillcolor="coral")#orange#ff8000 #f30000")  # red color
            elif prod_minus_load < 0:  # LOAD
                g.add_node(i, pin=True, prod_or_load="load", value=str(prod_minus_load), style="filled",
                           fillcolor="lightblue")#"#478fd0")  # blue color
            else:  # WHITE COLOR
                g.add_node(i, pin=True, prod_or_load="load", value=str(prod_minus_load), style="filled",
                           fillcolor="#ffffed")  # white color
            i += 1

    def build_edges(self,g, idx_or, idx_ex, edge_weights):

        """
        Create edges in graph for current grid state

        Parameters
        ----------

        g: :class:`nx:MultiDiGraph`
            a networkx graph to which to add edges

        idx_or: ``array`` int
            first extremity of edge for each edge

        idx_ex: ``array`` int
            second extremity of edge for each edge

        edge_weights: ``array`` float
            the flow value for each edge

        gtype: ``str``
            if we want a powerflow graph or

        """

        #if gtype is "powerflow":
        for origin, extremity, weight_value in zip(idx_or, idx_ex, edge_weights):
            # origin += 1
            # extremity += 1
            penwidth = fabs(weight_value) / 10
            min_penwidth=0.1
            if penwidth == 0.0:
                penwidth = min_penwidth

            if weight_value >= 0:
                g.add_edge(origin, extremity, label=self.float_precision% weight_value, color="gray", fontsize=10,
                           penwidth=max(float(self.float_precision % penwidth),min_penwidth))
            else:
                g.add_edge(extremity, origin, label=self.float_precision % fabs(weight_value), color="gray", fontsize=10,
                           penwidth=max(float(self.float_precision % penwidth),min_penwidth))


    def get_graph(self):
        """
        Returns the NetworkX graph representing the current state of the power flow.

        Returns
        -------
        :class:`nx:MultiDiGraph`
            The NetworkX graph.
        """
        return self.g

    def set_voltage_level_color(self, voltage_levels_dict, voltage_colors=default_voltage_colors):
        """
        Sets the voltage level color for each node in the graph based on the provided voltage levels dictionary.

        Parameters
        ----------
        voltage_levels_dict : dict
            A dictionary mapping node IDs to their respective voltage levels.
        voltage_colors : dict, optional
            A dictionary mapping voltage levels to their corresponding colors. Defaults to `default_voltage_colors`.

        Notes
        -----
        This method updates the 'color' attribute of each node in the graph based on the voltage levels provided.
        """
        voltage_levels_colors_dict = {node: voltage_colors[voltage_levels_dict[node]] for node in self.g}

        nx.set_node_attributes(self.g, voltage_levels_colors_dict, "color")

    def set_electrical_node_number(self, nodal_number_dict):
        """
        Sets the electrical node number for each node in the graph based on the provided nodal number dictionary.

        Parameters
        ----------
        nodal_number_dict : dict
            A dictionary mapping node IDs to their respective electrical node numbers.

        Notes
        -----
        This method updates the 'peripheries' attribute of each node in the graph based on the nodal numbers provided.
        """
        peripheries_dict = {node: nodal_number_dict[node] for node in self.g}

        nx.set_node_attributes(self.g, peripheries_dict, "peripheries")

    def plot(self, save_folder, name, state="before", sim=None):
        """
        Plots the graph using the Printer class.

        Parameters
        ----------
        save_folder : str
            The folder where the plot will be saved.
        name : str
            The name of the plot.
        state : str, optional
            The state of the simulation to plot. Defaults to "before".
        sim : object, optional
            The simulator object, which may have a plot method. Defaults to None.

        Notes
        -----
        If a simulator object is provided and it has a plot method, this method will use the simulator's plot method.
        Otherwise, it will use the Printer class to display the graph.
        """
        printer = Printer(save_folder)

        # In case the simulator also provides a plot function, use it
        if sim is not None and hasattr(sim, 'plot'):
            output_name = printer.create_namefile("geo", name=name, type="base")
            if state == "before":
                obs = sim.obs
            else:
                obs = sim.obs_linecut
            sim.plot(obs, save_file_path=output_name[1])
        else:
            if self.layout:
                printer.display_geo(self.g, self.layout, name=name)

class OverFlowGraph(PowerFlowGraph):
    """
    A coloured graph of grid overflow redispatch, displaying the delta flows before and after disconnecting the overloaded lines
    """

    def __init__(self, topo,lines_to_cut,df_overflow,layout=None,float_precision="%.2f"):
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

    def build_graph(self):
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

    def _get_edge_attributes(self, reported_flow, gray_edge, line_is_cut):
        """Helper method to determine edge attributes based on flow data."""
        penwidth = max(fabs(reported_flow) / 10, 0.1)
        attributes = {
            "capacity": float(self.float_precision % reported_flow),
            "label": self.float_precision % reported_flow,
            "fontsize": 10,
            "penwidth": float(self.float_precision % penwidth)
        }

        if line_is_cut:
            attributes.update({"color": "black", "constrained": True})
        elif gray_edge:
            attributes["color"] = "gray"
        elif reported_flow < 0:
            attributes["color"] = "blue"
        else:
            attributes["color"] = "coral"

        return attributes

    def build_edges_from_df(self, g, lines_to_cut):
        """Create edges in graph for overflow redispatch."""
        for i, row in self.df.iterrows():
            origin = row["idx_or"]
            extremity = row["idx_ex"]
            line_name = row["line_name"]

            attributes = self._get_edge_attributes(
                row["delta_flows"],
                row["gray_edges"],
                i in lines_to_cut
            )
            attributes["name"] = line_name
            g.add_edge(origin, extremity, **attributes)

    def _find_edges_to_recolor_on_path(self, g_subgraph, path_nodes):
        """Finds gray edges on paths between nodes on a constrained path section."""
        edges_to_recolor = set()
        paths = list(all_simple_edge_paths_multi(g_subgraph, path_nodes, path_nodes))
        current_colors = nx.get_edge_attributes(g_subgraph, 'color')

        for path in paths:
            path_colors = {current_colors.get(edge) for edge in path}
            if path_colors - {"blue", "black"}: # If there are non-blue/black edges
                edges_to_recolor.update(path)
        return edges_to_recolor

    def consolidate_constrained_path(self, constrained_path_nodes_amont, constrained_path_nodes_aval, constrained_path_edges, ignore_null_edges=True):
        """Consolidate constrained blue path for some edges that were discarded."""
        g_to_consolidate = delete_color_edges(self.g, "coral")

        if ignore_null_edges:
            null_capacity_edges = [e for e, cap in nx.get_edge_attributes(g_to_consolidate, "capacity").items() if cap == 0.0]
            g_to_consolidate.remove_edges_from(null_capacity_edges)

        g_to_consolidate.remove_edges_from(constrained_path_edges)

        # Handle amont path
        g_amont = g_to_consolidate.copy()
        g_amont.remove_nodes_from(constrained_path_nodes_aval)
        amont_edges_to_recolor = self._find_edges_to_recolor_on_path(g_amont, constrained_path_nodes_amont)

        # Handle aval path
        g_aval = g_to_consolidate.copy()
        g_aval.remove_nodes_from(constrained_path_nodes_amont)
        aval_edges_to_recolor = self._find_edges_to_recolor_on_path(g_aval, constrained_path_nodes_aval)

        all_edges_to_recolor = amont_edges_to_recolor.union(aval_edges_to_recolor)
        current_colors = nx.get_edge_attributes(self.g, 'color')
        edge_attributes_to_set = {
            edge: {"color": "blue"}
            for edge in all_edges_to_recolor
            if current_colors.get(edge) not in ["blue", "black"]
        }
        nx.set_edge_attributes(self.g, edge_attributes_to_set)

    #def consolidate_constrained_path(self, constrained_path_nodes_amont,constrained_path_nodes_aval,constrained_path_edges,ignore_null_edges=True):#hub_sources,hub_targets):
    #    """
    #    Consolidate constrained blue path for some edges that were discarded with lower values but are actually on the path
    #    knowing the hubs in the SuscturedOverflowGraph
#
    #    Parameters
    #    ----------
#
    #    hub_sources: ``array``
    #        list of nodes that are hubs and sources of loop paths in the structured graph
#
    #    hub_targets: ``array``
    #        list of nodes that are hubs and targets of loop paths in the structured graph
#
    #    """



        #g_without_pos_edges = delete_color_edges(self.g, "coral")
        #current_colors = nx.get_edge_attributes(self.g, 'color')
#
        #init_capacity = nx.get_edge_attributes(g_without_pos_edges, "capacity")
        #edges_to_remove_positive_capacity = [edge for edge, capacity in
        #                             init_capacity.items() if capacity >0. ]
        #g_without_pos_edges.remove_edges_from(edges_to_remove_positive_capacity)

        #Reasoning flawed in the end, we don't want to fin new contsrained path from amont to aval of the constraint, but only amont and only aval
        #for source, target in zip(hub_sources, hub_targets):
        #    paths = nx.all_simple_edge_paths(g_without_pos_edges, source, target)
#
        #    for path in paths:
        #        path_color = set([current_colors[edge] for edge in path])
        #        has_edge_to_recolor=len(path_color - set({"blue","black"}))!=0
        #        if has_edge_to_recolor:
        #            all_edges_to_recolor += path
#
        #all_edges_to_recolor=set(all_edges_to_recolor)
#
#
        #current_weights=nx.get_edge_attributes(self.g, 'capacity') #############################
        #edge_attribues_to_set = {edge: {"color": "blue"} for edge in all_edges_to_recolor if current_colors[edge] not in ["blue","black"] and float(current_weights[edge])!=0}
        #nx.set_edge_attributes(self.g, edge_attribues_to_set)
#
        ##########
        ##correction: reverse edges with positive values
        #current_capacities = nx.get_edge_attributes(self.g, 'capacity')
        #edges_to_correct=[edge for edge in all_edges_to_recolor if current_capacities[edge]>0]
        #reverse_edges=[(edge_ex,edge_or,edge_properties) for edge_or,edge_ex,edge_properties in self.g.edges(data=True) if edge_properties["color"]=="blue" and edge_properties["capacity"]>0]
        #self.g.add_edges_from(reverse_edges)
        #self.g.remove_edges_from(edges_to_correct)
#
        ##correct capacity values with opposite value after reversing edge
        #current_capacities = nx.get_edge_attributes(self.g, 'capacity')
        #current_colors = nx.get_edge_attributes(self.g, 'color')
        #edge_attribues_to_set = {edge: {"capacity": -capacity,"label":str(-capacity)}
        #                         for edge,color,capacity in zip(self.g.edges,current_colors.values(),current_capacities.values()) if
        #                         capacity>0 and color=="blue"}
        #nx.set_edge_attributes(self.g, edge_attribues_to_set)

        ############
        #for null flow redispatch, if connected to nodes on blue path, reverse it and make it blue for it to belong there
        #blue_edges=[edge for edge in self.g.edges if current_colors[edge]=="blue"]
        #nodes_blue_path=self.g.edge_subgraph(blue_edges).nodes

        #overall_constrained_graph=self.g.subgraph(nodes_constrained_path)
#
        #current_capacities = nx.get_edge_attributes(overall_constrained_graph, 'capacity')
        #current_colors = nx.get_edge_attributes(overall_constrained_graph, 'color')
#
        #edges_non_constrained_path_yet=[edge for edge,color in current_colors.items() if color not in ["blue","black"]]
        #edges_non_constrained_path_yet_with_properties=[(edge_or,edge_ex,edge_properties) for edge_or,edge_ex,edge_properties in overall_constrained_graph.edges(data=True) if edge_properties["color"] not in ["blue","black"]]
#
        #if len(edges_non_constrained_path_yet)!=0:
        #    #reverse edges for red edges
        #    edges_to_correct=[edge for edge,capacity in current_capacities.items() if capacity>0]
        #    reverse_edges=[(edge_ex,edge_or,edge_properties) for edge_or,edge_ex,edge_properties in edges_non_constrained_path_yet_with_properties if edge_properties["capacity"]>0]
        #    self.g.add_edges_from(reverse_edges)
        #    self.g.remove_edges_from(edges_to_correct)
#
        #    #update of this after reversing edges
        #    overall_constrained_graph=self.g.subgraph(nodes_constrained_path)
        #    current_colors = nx.get_edge_attributes(overall_constrained_graph, 'color')
        #    current_capacities = nx.get_edge_attributes(overall_constrained_graph, 'capacity')
        #    edges_non_constrained_path_yet = [edge for edge, color in current_colors.items() if
        #                                      color not in ["blue", "black"]]
#
        #    #set new attribute, in particular blue color
        #    edge_attributes_to_set = {edge: {"capacity": -abs(current_capacities[edge]),"label":str(-abs(current_capacities[edge])),"color":"blue"}
        #                             for edge in edges_non_constrained_path_yet}
        #    nx.set_edge_attributes(self.g, edge_attributes_to_set)
#
        #print("ok")

    def reverse_edges(self, edge_path_names,target_color):

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

    def reverse_blue_edges_in_looppaths(self, constrained_path):
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

    def consolidate_loop_path(self, hub_sources,hub_targets,ignore_null_edges=True):
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

    def set_hubs_shape(self, hubs,shape_hub="circle"):
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

    def highlight_swapped_flows(self, lines_swapped):
        """
        Highlight lines with "tappered" style on edge that have seen their flows swapped in the overflow graph to be aware of that

        Parameters
        ----------

        lines_swapped: ``list``
            list of lines whose flow direction has swapped

        """
        edge_names = nx.get_edge_attributes(self.g, "name")
        edge_styles={edge:"tapered" for edge, edge_name in edge_names.items() if edge_name in lines_swapped}
        edge_dirs = {edge: "both" for edge, edge_name in edge_names.items() if edge_name in lines_swapped}
        edge_tails = {edge: "none" for edge, edge_name in edge_names.items() if edge_name in lines_swapped}

        nx.set_edge_attributes(self.g, edge_styles, "style")
        nx.set_edge_attributes(self.g, edge_dirs, "dir")
        nx.set_edge_attributes(self.g, edge_tails, "arrowtail")

    def highlight_significant_line_loading(self, dict_line_loading):
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

    def plot(self,layout,rescale_factor=None,allow_overlap=True,fontsize=None,node_thickness=3,save_folder="",without_gray_edges=False):
        printer=Printer(save_folder)
        g=self.g

        if layout is not None:
            layout_dict = {n: coord for n, coord in zip(g.nodes, layout)}

        if without_gray_edges:
            g=delete_color_edges(g, "gray")
            kept_nodes=g.nodes

            if layout is not None:
                layout=[layout_dict[node] for node in kept_nodes]# for node, coord in layout_dict.items() if node in kept_nodes]

        if save_folder=="":
            output_graphviz_svg=printer.plot_graphviz(g, layout,rescale_factor=rescale_factor,allow_overlap=allow_overlap,fontsize=fontsize,node_thickness=node_thickness, name="g_overflow_print")
            return output_graphviz_svg
        else:
            printer.display_geo(g, layout,rescale_factor=rescale_factor,fontsize=fontsize,node_thickness=node_thickness, name="g_overflow_print")
            return None

    def _temporarily_remove_ignored_lines(self, lines_to_ignore):
        """Finds and removes edges that should be ignored during consolidation."""
        if not lines_to_ignore:
            return []

        edges_to_remove_data = [
            (u, v, data) for u, v, data in self.g.edges(data=True)
            if data.get("name") in lines_to_ignore
        ]

        # We only need the edge keys (u, v, key) to remove them
        edges_to_remove = [(u, v, data.get('key')) for u, v, data in edges_to_remove_data]
        self.g.remove_edges_from(edges_to_remove)

        return edges_to_remove_data

    def _iteratively_consolidate_constrained_path(self, structured_graph):
        """
        Repeatedly consolidates the constrained (blue) path until the graph structure stabilizes.
        """
        n_hubs_init = 0
        # Assuming find_loops() returns a DataFrame with "Source" and "Target" columns
        hubs_paths = structured_graph.find_loops()[["Source", "Target"]].drop_duplicates()
        n_hub_paths = hubs_paths.shape[0]

        while n_hubs_init != n_hub_paths:
            n_hubs_init = n_hub_paths

            cp = structured_graph.constrained_path
            constrained_path_edges = cp.aval_edges + [cp.constrained_edge] + cp.amont_edges
            self.consolidate_constrained_path(cp.n_amont(), cp.n_aval(), constrained_path_edges)

            # Re-evaluate the graph structure after consolidation
            structured_graph = Structured_Overload_Distribution_Graph(self.g)
            hubs_paths = structured_graph.find_loops()[["Source", "Target"]].drop_duplicates()
            n_hub_paths = hubs_paths.shape[0]

        return structured_graph

    def _desambiguate_paths(self, structured_graph):
        """Identifies and resolves ambiguous paths with mixed blue/coral colors."""
        ambiguous_edge_paths, ambiguous_node_paths = self.identify_ambiguous_paths(structured_graph)

        for edge_path, node_path in zip(ambiguous_edge_paths, ambiguous_node_paths):
            path_type = self.desambiguation_type_path(node_path, structured_graph)
            target_color = "coral" if path_type == "loop_path" else "blue"
            self.reverse_edges(edge_path, target_color=target_color)

    def consolidate_graph(self, structured_graph, non_connected_lines_to_ignore=[], no_desambiguation=False):
        """
        (Refactored) Consolidates the overflow graph using its structural elements.
        """
        # 1. Temporarily remove any specified lines from the graph
        ignored_edges_data = self._temporarily_remove_ignored_lines(non_connected_lines_to_ignore)

        # 2. Iteratively consolidate the main constrained (blue) path
        final_structured_graph = self._iteratively_consolidate_constrained_path(structured_graph)

        # 3. Resolve ambiguous (mixed-color) paths if requested
        if not no_desambiguation:
            self._desambiguate_paths(final_structured_graph)

        # 4. Consolidate the loop (coral) paths
        hubs_paths = final_structured_graph.red_loops[["Source", "Target"]].drop_duplicates()
        if not hubs_paths.empty:
            self.consolidate_loop_path(hubs_paths.Source, hubs_paths.Target)

        # 5. Add back the lines that were temporarily removed
        if ignored_edges_data:
            self.g.add_edges_from(ignored_edges_data)

    def identify_ambiguous_paths(self, structured_graph):
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

    def desambiguation_type_path(self, ambiguous_node_path, structured_graph):
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

    def rename_nodes(self,mapping):
        self.g = nx.relabel_nodes(self.g, mapping, copy=True)
        self.df["idx_or"]=[mapping[idx_or] for idx_or in self.df["idx_or"]]
        self.df["idx_ex"] = [mapping[idx_or] for idx_or in self.df["idx_ex"]]

    def add_relevant_null_flow_lines_all_paths(self, structured_graph, non_connected_lines,non_reconnectable_lines=[]):
        """
        Make edges bi-directionnal when flow redispatch value is null

        Parameters
        ----------

        structured_graph: ``SuscturedOverflowGraph``
            a structured graph with identified constrained path, hubs, loop paths


        non_connected_lines: ``array``
            list of lines that are non connected but that could be reconnected and that we want to highlight if relevant

        """
        self.add_relevant_null_flow_lines(structured_graph, non_connected_lines, non_reconnectable_lines,
                                          target_path="blue_amont_aval")
        self.add_relevant_null_flow_lines(structured_graph, non_connected_lines, non_reconnectable_lines,
                                          target_path="red_only")
        self.add_relevant_null_flow_lines(structured_graph, non_connected_lines, non_reconnectable_lines,
                                          target_path="blue_to_red")
        self.add_relevant_null_flow_lines(structured_graph, non_connected_lines, non_reconnectable_lines,
                                          target_path="blue_only")

        # =================================================================
        # REFACTORED METHODS START HERE
        # =================================================================

    def _style_and_categorize_null_lines(self, non_connected_lines, non_reconnectable_lines):
        """
        Applies styles to null-flow lines and categorizes them.

        Returns a dictionary containing sets of edges for different categories.
        """
        all_null_lines = list(set(non_connected_lines + non_reconnectable_lines))
        edge_names = nx.get_edge_attributes(self.g, 'name')

        categorized_edges = {
            "all": {e for e, name in edge_names.items() if name in all_null_lines},
            "non_reconnectable": {e for e, name in edge_names.items() if name in non_reconnectable_lines}
        }
        categorized_edges["reconnectable"] = categorized_edges["all"] - categorized_edges["non_reconnectable"]

        # Apply styles based on category
        nx.set_edge_attributes(self.g, {e: "dotted" for e in categorized_edges["non_reconnectable"]}, name="style")
        nx.set_edge_attributes(self.g, {e: "dashed" for e in categorized_edges["reconnectable"]}, name="style")
        nx.set_edge_attributes(self.g, {e: "none" for e in categorized_edges["non_reconnectable"]}, name="dir")

        return categorized_edges

    def _find_relevant_gray_components(self):
        """
        Finds and returns weakly connected components of the gray-edge subgraph.
        """
        g_no_red = delete_color_edges(self.g, "coral")
        g_only_gray_components = delete_color_edges(delete_color_edges(g_no_red, "blue"), "black")

        return [
            g_only_gray_components.subgraph(c).copy()
            for c in sorted(nx.weakly_connected_components(g_only_gray_components), key=len, reverse=False)
        ]

    def _process_gray_component(self, g_component, structured_graph, categorized_edges, target_path):
        """
        Processes a single gray component to find paths for potential reconnection.

        Returns a tuple of (edges_to_keep, edges_non_reconnectable_in_path).
        """
        component_edges = set(g_component.edges)
        # Consider only null-flow lines that are within the current gray component
        null_lines_in_component = categorized_edges["all"].intersection(component_edges)

        if not null_lines_in_component:
            return set(), set()

        # Define source and target node sets based on the target_path strategy
        sg = structured_graph
        cp = sg.constrained_path
        nodes = {
            "amont": cp.n_amont(),
            "aval": cp.n_aval(),
            "red": set(sg.g_only_red_components.nodes()) if not sg.red_loops.empty else set()
        }

        component_nodes = set(g_component.nodes)
        intersections = {key: component_nodes.intersection(val) for key, val in nodes.items()}

        edges_to_keep = set()
        non_reconnectable_in_path = set()

        path_definitions = {
            "blue_only": [
                (intersections["amont"], intersections["amont"]),
                (intersections["aval"], intersections["aval"])
            ],
            "blue_amont_aval": [
                (intersections["amont"], intersections["aval"])
            ],
            "red_only": [
                (intersections["red"], intersections["red"])
            ],
            "blue_to_red": [
                (intersections["amont"], intersections["red"]),
                (intersections["red"], intersections["aval"]),
                (intersections["amont"], intersections["aval"])
            ]
        }

        # Pre-process component for pathfinding based on target color
        if target_path in ["blue_only", "blue_amont_aval"]:
            edges_to_remove = {e for e, cap in nx.get_edge_attributes(g_component, "capacity").items() if cap > 0}
            g_component.remove_edges_from(edges_to_remove)
        elif target_path == "red_only":
            edges_to_remove = {e for e, cap in nx.get_edge_attributes(g_component, "capacity").items() if cap < 0}
            g_component.remove_edges_from(edges_to_remove)

        # Detect paths for the current strategy
        for sources, targets in path_definitions.get(target_path, []):
            if sources and targets:
                keep, non_recon = self.detect_edges_to_keep(
                    g_component, sources, targets,
                    categorized_edges["all"], categorized_edges["non_reconnectable"]
                )
                edges_to_keep.update(keep)
                non_reconnectable_in_path.update(non_recon)

        return edges_to_keep, non_reconnectable_in_path

    def _apply_reconnection_colors_and_cleanup(self, edges_to_keep, non_reconnectable_in_path, doubled_edges,
                                               target_path):
        """Applies final colors to kept edges and removes unused doubled edges."""

        # Determine color based on target_path
        if target_path == "blue_only":
            color_map = {e: {"color": "blue"} for e in edges_to_keep}
        elif target_path == "blue_to_red":
            current_weights = nx.get_edge_attributes(self.g, 'capacity')
            color_map = {e: {"color": "coral"} for e in edges_to_keep}
            color_map.update({e: {"color": "blue"} for e in edges_to_keep if current_weights.get(e, 0) < 0})
        else:  # red_only or blue_amont_aval (which implies a new loop)
            color_map = {e: {"color": "coral"} for e in edges_to_keep}

        # Non-reconnectable lines have a distinct color
        color_map.update({
            e: {"color": "dimgray"}
            for e in non_reconnectable_in_path
            if self.g.edges[e]["color"] == "gray"
        })

        nx.set_edge_attributes(self.g, color_map)

        # Set direction to none for zero-capacity edges that were kept
        edges_to_double_set = set(doubled_edges['original'].values()).union(set(doubled_edges['added'].values()))
        kept_zero_edges = edges_to_keep.intersection(edges_to_double_set)
        nx.set_edge_attributes(self.g, {e: "none" for e in kept_zero_edges}, name="dir")

        # Cleanup unused doubled edges
        self.g = remove_unused_added_double_edge(self.g, edges_to_keep, doubled_edges['original'],
                                                 doubled_edges['added'])

    def add_relevant_null_flow_lines(self,structured_graph,non_connected_lines,non_reconnectable_lines=[],target_path="blue_to_red"):
        """
        Make edges bi-directionnal when flow redispatch value is null, recolor the relevant ones that could be of interest
        for analyzing the problem or solving it, and get back to initial edges for the other

        Parameters
        ----------

        structured_graph: ``SuscturedOverflowGraph``
            a structured graph with identified constrained path, hubs, loop paths


        non_connected_lines: ``array``
            list of lines that are non connected but that could be reconnected and that we want to highlight if relevant

        target_path: ``str``
            target path on which to highlight disconnected lines. either blue_only,blue_amont_aval, red_only, blue_to_red

        non_connected_lines: ``array``
            list of lines that are non connected and cannot be reconnected
        """

        # 1. Style and categorize null-flow lines
        categorized_edges = self._style_and_categorize_null_lines(non_connected_lines, non_reconnectable_lines)

        # 2. Make null-flow edges bi-directional for pathfinding
        edges_to_double, edges_double_added = add_double_edges_null_redispatch(self.g)
        doubled_edges = {'original': edges_to_double, 'added': edges_double_added}

        # 3. Find relevant gray subgraphs to analyze
        gray_components = self._find_relevant_gray_components()

        # 4. Process each gray component to find reconnection paths
        all_edges_to_keep = set()
        all_non_reconnectable_in_path = set()

        for g_component in gray_components:
            kept, non_recon = self._process_gray_component(
                g_component, structured_graph, categorized_edges, target_path
            )
            all_edges_to_keep.update(kept)
            all_non_reconnectable_in_path.update(non_recon)

        # 5. Apply colors to kept edges and clean up the graph
        self._apply_reconnection_colors_and_cleanup(
            all_edges_to_keep, all_non_reconnectable_in_path, doubled_edges, target_path
        )

    def _find_shortest_paths_between_sets(self, g_pathfinding, sources, targets):
        """Finds all shortest paths between any source and any target node."""
        all_paths = []
        for source in sources:
            for target in targets:
                if source == target:
                    continue
                try:
                    paths_nodes_iter = nx.all_shortest_paths(g_pathfinding, source=source, target=target,
                                                             weight="capacity")
                    for p_nodes in paths_nodes_iter:
                        all_paths.append(nodepath_to_edgepath(g_pathfinding, p_nodes, with_keys=True))
                except (NetworkXNoPath, nx.NodeNotFound):
                    continue
        return all_paths

        # TODO plutot parcourir les edges of interest ?
        # for edge_of_interest in edges_of_interest:
        #    if edge_of_interest in g_c.edges:
        #        node1=edge_of_interest[0]
        #        node2=edge_of_interest[1]
        #        source_node=None
        #        node_continuing_path=None
        #        if node1 in source_nodes:
        #            source_node=node1
        #            node_continuing_path=node2
        #        elif node2 in source_nodes:
        #            source_node=node2
        #            node_continuing_path = node1

    #
    #        if source_node:
    #            g_c.remove_edges_from(edge_of_interest) #on supprime l'edge pour les traitements futurs ci-dessous
    #            for target_node in target_nodes:

    def _categorize_edges_from_paths(self, paths_of_interest, g_c_edge_names_dict, edge_names_of_interest,
                                     non_reconnectable_edges_names):
        """Sorts paths by length and categorizes their edges as reconnectable or non-reconnectable."""
        sorted_paths = sorted(paths_of_interest, key=len, reverse=False)
        edge_names_already_found_in_path = set()
        edges_to_keep_reconnectable = set()
        edges_to_keep_non_reconnectable = set()

        for path in sorted_paths:
            path_edges_not_already_found = {edge for edge in path if
                                            g_c_edge_names_dict.get(edge) not in edge_names_already_found_in_path}
            path_edge_names_not_already_found = {g_c_edge_names_dict.get(edge) for edge in path_edges_not_already_found}

            if path_edge_names_not_already_found.intersection(edge_names_of_interest):
                found_remaining_non_reconnectable_edges_names = path_edge_names_not_already_found.intersection(
                    non_reconnectable_edges_names)
                if found_remaining_non_reconnectable_edges_names:
                    edges_to_keep_non_reconnectable.update(path_edges_not_already_found)
                else:
                    edges_to_keep_reconnectable.update(path_edges_not_already_found)

                edge_names_already_found_in_path.update(path_edge_names_not_already_found)

        return edges_to_keep_reconnectable, edges_to_keep_non_reconnectable

    def detect_edges_to_keep(self, g_c, source_nodes, target_nodes, edges_of_interest, non_reconnectable_edges=[]):
        """
        detect edges in edges of interest that belongs to gthe subgraph and are on a path between source nodes and target nodes

        Parameters
        ----------

        g_c: ``Networkx Graph``
            a networkx subgraph of gray edges to possibly recolor


        source_nodes: ``array`` str
            list of nodes, that belong to either constrained path or red loops, from which to find a path

        target_nodes: ``array`` str
            list of nodes, that belong to either constrained path or red loops, to which to find a path

        Returns
        ----------
        res: ``set`` str
            set of edges of interest found on paths and to be recoloured
        """
        # 1. Prepare a safe graph copy for pathfinding locally
        g_pathfinding = g_c.copy()
        new_attributes = {
            e: {"capacity": abs(capacity)}
            for e, capacity in nx.get_edge_attributes(g_pathfinding, "capacity").items()
        }
        nx.set_edge_attributes(g_pathfinding, new_attributes)

        # 2. Find all relevant shortest paths
        all_paths = self._find_shortest_paths_between_sets(g_pathfinding, source_nodes, target_nodes)

        # 3. Filter paths to keep only those containing an edge of interest
        g_c_edge_names_dict = nx.get_edge_attributes(g_c, "name")
        edge_names_of_interest = {g_c_edge_names_dict.get(edge) for edge in edges_of_interest if
                                  edge in g_c_edge_names_dict}

        paths_of_interest = []
        for path in all_paths:
            path_edge_names = {g_c_edge_names_dict.get(edge) for edge in path}
            if path_edge_names.intersection(edge_names_of_interest):
                paths_of_interest.append(path)

        # 4. Categorize the edges from the filtered paths
        non_reconnectable_edges_names = {g_c_edge_names_dict.get(edge) for edge in non_reconnectable_edges if
                                         edge in g_c_edge_names_dict}

        return self._categorize_edges_from_paths(
            paths_of_interest, g_c_edge_names_dict, edge_names_of_interest, non_reconnectable_edges_names
        )


class ConstrainedPath:

    """
    A connected path of lines that includes the overloaded lines and for which the overflow redipsatch is negative.
    This can be regarded as the main path through which the flows get in and out of the overloaded lines.
    Hence flows should be pushed on other path to relieve the overloads.
    """

    def __init__(self, amont_edges, constrained_edge, aval_edges):
        print("Constrained path created")
        self.amont_edges = amont_edges #lines which flow goes into the overloaded lines
        self.constrained_edge = constrained_edge #overloaded lines
        self.aval_edges = aval_edges #lines which flow comes from the overloaded lines

    def n_amont(self) -> list:
        """Returns a list of nodes that are in "amont" """
        return from_edges_get_nodes(self.amont_edges, "amont", self.constrained_edge)

    def n_aval(self):
        """Returns a list of nodes that are in "aval" """
        return from_edges_get_nodes(self.aval_edges, "aval", self.constrained_edge)

    def e_amont(self):
        """Returns a list of edges that are in "amont" """
        return self.amont_edges

    def e_aval(self):
        """Returns a list of edges that are in "aval" """
        return self.aval_edges

    def filter_constrained_path_for_nodes(self):
        """
        This filters the constrained_path_lists and creates a uniq ordered list that represents the constrained_path

        Returns
        ----------

        res: ``array`` int
            ordered list of nodes on the constrained path

        """
        set_constrained_path = []
        for path in [self.amont_edges, self.constrained_edge, self.aval_edges]:
            if isinstance(path, tuple):
                edge = path
                for n in edge[0:2]:
                    if n not in set_constrained_path:
                        set_constrained_path.append(n)
            else:
                for edge in path:
                    if isinstance(edge, tuple):
                        for n in edge[0:2]:
                            if n not in set_constrained_path:
                                set_constrained_path.append(n)

        return set_constrained_path

    def full_n_constrained_path(self):
        return self.filter_constrained_path_for_nodes()

    def __repr__(self):
        return "################################################################\n" \
               "ConstrainedPath = %s \nDetails: (amont: %s, constrained_edge: %s, aval: %s)\n" \
               "################################################################" % (
                   self.full_n_constrained_path(), self.amont_edges, self.constrained_edge, self.aval_edges)

class Structured_Overload_Distribution_Graph:
    """
    Staring from a raw overload distribution graph with color edges, this class identifies the underlying path structure in terms of constrained path, loop paths and hub nodes
    """
    def __init__(self,g):
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
        self.red_loops = self.find_loops() #parallel path to the constrained path on which flow can be rerouted
        self.hubs = self.find_hubs() #specific nodes at substations connecting loop paths to constrained path. This is where flow can be most easily rerouted

    def get_amont_blue_edges(self, g, node):
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

    def get_aval_blue_edges(self, g, node):
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


    def find_hubs(self):
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
            print("In get_hubs(): c = ")
            print(self.constrained_path)
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

    def get_hubs(self):
        return self.hubs

    def find_loops(self):

        """This function returns all parallel paths. After discussing with Antoine, start with the most "en Aval" node,
        and walk in reverse for loops and parallel path returns a dict with all data

        Returns
        ----------

        res: pd.DataFrame
            a dataframe with rows representing each detected path, with column attibutes "Source, Target, Path" with Path representing a list of nodes
        """
        #TODO: could be augmented by consolidating here red paths that could include blue edges that are not on the constrained path (by reverting them artificially here)
        attr_edge_direction=nx.get_edge_attributes(self.g_only_red_components, "dir")
        if len(attr_edge_direction)!=0:
            # add edges to make simple paths work for no direction edges
            edges_to_double, edges_double_added = add_double_edges_null_redispatch(self.g_only_red_components,color_init="coral",only_no_dir=True)

        # print("==================== In function get_loops ====================")
        g = self.g_only_red_components
        c_path_n = self.constrained_path.full_n_constrained_path()
        all_loop_paths = {}
        ii = 0

        for i in range(len(c_path_n)):
            for j in reversed(range(len(c_path_n))):
                if i < j:
                    # # print(i, j)
                    # # print("we compare paths from source: {} to target: {}".format(c_path_n[i], c_path_n[j]))
                    node_source=c_path_n[i]
                    node_target = c_path_n[j]
                    if (node_source in g.nodes) and  (node_target in g.nodes):
                        try:
                            res = nx.all_simple_paths(g, node_source, node_target)#nx.all_shortest_paths(g, node_source, node_target)
                            for p in res:
                                # print("path = ", p)
                                all_loop_paths[ii] = p
                                ii += 1
                        except nx.NetworkXNoPath:
                            print("shortest path between {0} and {1} failed".format(c_path_n[i], c_path_n[j]))

        # print("### Print in get_loops ###, all_loop_paths")
        # pprint.pprint(all_loop_paths)

        data_for_df = {"Source": [], "Target": [], "Path": []}
        for path in list(all_loop_paths.keys()):
            data_for_df["Source"].append(all_loop_paths[path][0])
            data_for_df["Target"].append(all_loop_paths[path][-1])
            data_for_df["Path"].append(all_loop_paths[path])

        # pprint.pprint(data_for_df)
        if len(attr_edge_direction)!=0:
            #remove added edges that made simple paths working for no direction edges
            self.g_only_red_components = remove_unused_added_double_edge(self.g_only_red_components, set(edges_to_double.values()),
                                                                         edges_to_double, edges_double_added)

        return pd.DataFrame.from_dict(data_for_df)

    def get_loops(self):
        return self.red_loops

    def find_constrained_path(self):
        """Find and return the constrained path

         Returns
        ----------

        res: :class:`ConstrainedPath`
            a constrained path object
        """
        constrained_edge = None
        tmp_constrained_path = []
        edge_list = nx.get_edge_attributes(self.g_only_blue_components, "color")
        for edge, color in edge_list.items():
            if "black" in color:
                constrained_edge = edge
        amont_edges = self.get_amont_blue_edges(self.g_only_blue_components, constrained_edge[0])
        aval_edges = self.get_aval_blue_edges(self.g_only_blue_components, constrained_edge[1])

        return ConstrainedPath(amont_edges,constrained_edge,aval_edges)

    def get_constrained_path(self):
        return self.constrained_path

    def get_constrained_edges_nodes(self):
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

        edges_to_remove=[edge for edge, edge_name in edge_names.items() if edge_name in edges_constrained_path]
        g_blue.remove_edges_from(edges_to_remove)

        other_blue_edges=list(g_blue.edges())
        other_blue_nodes=[node for node in g_blue.nodes() if node not in nodes_constrained_path]

        return list(set(edges_constrained_path)), nodes_constrained_path, other_blue_edges, other_blue_nodes

    def get_dispatch_edges_nodes(self,only_loop_paths=True):
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
            if len(self.find_loops()["Path"])!=0:
                list_nodes_dispatch_path = list(set(self.find_loops()["Path"].sum()))
            else:
                list_nodes_dispatch_path=[]
        else:
            list_nodes_dispatch_path=list(g_red.nodes)

        edge_names_red = nx.get_edge_attributes(g_red, 'name')
        lines_redispatch=[edge_name for edge, edge_name in edge_names_red.items() if
                                (edge[0] in list_nodes_dispatch_path) and (edge[1] in list_nodes_dispatch_path)]

        return lines_redispatch, list_nodes_dispatch_path


def from_edges_get_nodes(edges, amont_or_aval: str, constrained_edge):
    """edges is a list of tuples"""
    if edges:
        nodes = []
        for e in edges:
            for node in e[:2]:
                if node not in nodes:
                    nodes.append(node)
        return nodes
    elif amont_or_aval == "amont":
        return [constrained_edge[0]]
    elif amont_or_aval == "aval":
        return [constrained_edge[1]]
    else:
        raise ValueError("Error in function from_edges_get_nodes")

def delete_color_edges(_g, edge_color):
    """
    Returns a copy of a graph without edges of a given color. Gray for instance, with values below a threshold of significance

    From a given node, get blue edges (with negative overflow redispatch) that are above this node

    Parameters
    ----------

    _g: :class:`nx:MultiDiGraph`
        an overflow redispatch networkx graph

    edge_color: ``str``
        color of edges to delete from graoh

    Returns
    ----------

    res: :class:`nx:MultiDiGraph`
        the graph without edges for the targeted color

    """
    g = _g.copy()

    TargetColor_edges = []
    i = 1
    for u, v,idx, color in g.edges(data="color",keys=True):
        if color == edge_color:
            TargetColor_edges.append((i, (u, v,idx)))
        i += 1

    # delete from graph gray edges
    # this extracts the (u,v) from pos_edges
    if TargetColor_edges:
        g.remove_edges_from(list(zip(*TargetColor_edges))[1])
        g.remove_nodes_from(list(nx.isolates(g)))
    return g

def nodepath_to_edgepath(G, node_path, with_keys=False):
    """Convert a list of nodes into a list of edges for Graph/MultiGraph."""
    edges = []
    for u, v in zip(node_path[:-1], node_path[1:]):
        if with_keys and G.is_multigraph():
            # take the first key by default
            #k = next(iter(G[u][v].keys()))
            #take all keys
            for k in G[u][v].keys():
                edges.append((u, v, k))
        else:
            edges.append((u, v))
    return edges

def incident_edges(G, node, data=True, keys=False):
    if keys and G.is_multigraph():
        out_e = G.out_edges(node, keys=True, data=data)
        in_e  = G.in_edges(node, keys=True, data=data)
    else:
        out_e = G.out_edges(node, data=data)
        in_e  = G.in_edges(node, data=data)
    return list(out_e) + list(in_e)

def all_simple_edge_paths_multi(G, sources, targets, cutoff=None):
    """
    Yield all simple edge paths between multiple sources and targets.

    Parameters
    ----------
    G : nx.Graph / nx.DiGraph / nx.MultiDiGraph
        Graph object.
    sources : iterable
        Set/list of source nodes.
    targets : iterable
        Set/list of target nodes.
    cutoff : int, optional
        Maximum path length.

    Yields
    ------
    path : list of edges (u, v) or (u, v, key) for multigraphs
    """
    for s in sources:
        for t in targets:
            if s != t and s in G and t in G:
                try:
                    for path in nx.all_simple_edge_paths(G, s, t, cutoff=cutoff):
                        yield path
                except nx.NetworkXNoPath:
                    continue

def remove_unused_added_double_edge(g, edges_to_keep, edges_to_double, edges_double_added):

    """
    Make edges bi-directionnal when flow redispatch value is null

    Parameters
    ----------
     g: NetworkX graph
      graph on which to remove edges
     edges_to_keep: ``set`` str
        set of edges of interest found on paths and to be recoloured

    edges_to_double: ``dict`` str: networkx edge
        original set of edges that has been doubled, with line names as key and edge as value

    edges_double_added: ``dict`` str: networkx edge
        new set of edges that doubles the original ones in the other direction, with line name as key and edge as value

    Return
    ----------------
    g: NetworkX graph
      graph on which edges where removed
    """
    name_edges_to_keep = nx.get_edge_attributes(g.edge_subgraph(edges_to_keep), "name").values()

    # for initial edges that has not been recoloured but for which the added double edge has been, remove those initial edges
    edges_to_remove = [edge for name, edge in edges_to_double.items() if
                       name in name_edges_to_keep and g.edges[edge]["color"] == "gray"]
    edge_names_to_remove = [name for name, edge in edges_to_double.items() if
                            name in name_edges_to_keep and g.edges[edge]["color"] == "gray"]

    # for added double edges that has not been recoloured, remove them
    edges_to_remove += [edge for name, edge in edges_double_added.items() if name not in edge_names_to_remove]
    assert (len(edges_to_remove) == len(edges_to_double))
    g.remove_edges_from(edges_to_remove)

    return g

def add_double_edges_null_redispatch(g,color_init="gray",only_no_dir=False):
    """
    Make edges bi-directionnal when flow redispatch value is null

    Parameters
    -------------
    g: NetworkX graph
      graph on which to add edges

    only_no_dir: bool
        condition to restrict edge doubling at no_dir case

    Returns
    ----------
    edges_to_double_name_dict: ``dict`` str: networkx edge
        original set of edges that has been doubled, with line names as key and edge as value

    edges_added_name_dict: ``dict`` str: networkx edge
        new set of edges that doubles the original ones in the other direction, with line name as key and edge as value

    """
    init_edges_names=nx.get_edge_attributes(g, "name")
    init_colors=nx.get_edge_attributes(g, "color").values()
    init_capacity = nx.get_edge_attributes(g, "capacity").values()
    no_dir_edges =nx.get_edge_attributes(g, "dir")

    edges_to_double_name_dict={name:edge for edge,name,color,capacity in zip(init_edges_names.keys(),init_edges_names.values(),init_colors,init_capacity) if color==color_init and capacity==0. and (not only_no_dir or edge in no_dir_edges)}

    edges_to_double_data=[ (edge_or, edge_ex, edge_properties) for edge_or,edge_ex,edge_properties in g.edges(data=True) if edge_properties["color"]==color_init and edge_properties["capacity"]==0. and (not only_no_dir or "dir" in edge_properties)]
    edges_to_add_data=[(edge_ex, edge_or, edge_properties) for edge_or,edge_ex,edge_properties in edges_to_double_data]
    g.add_edges_from(edges_to_add_data)

    new_edges_names=nx.get_edge_attributes(g, "name")#.keys()
    only_new_edges=set(new_edges_names.keys()) - set(init_edges_names.keys())
    edges_added_name_dict={name:edge for edge,name in new_edges_names.items() if edge in only_new_edges}

    assert(set(edges_to_double_name_dict.keys())==set(edges_added_name_dict.keys()))
    return edges_to_double_name_dict,edges_added_name_dict