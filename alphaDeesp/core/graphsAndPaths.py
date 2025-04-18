
import pandas as pd
import networkx as nx
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

    def build_edges_from_df(self, g, lines_to_cut):
        """
        Create edges in graph for overflow redispatch

        Parameters
        ----------

        g: :class:`nx:MultiDiGraph`
            a networkx graph to which to add edges

        lines_to_cut: ``array`` int
            list of lines in overflow that are getting disconnected

        """

        i = 0
        for origin, extremity, reported_flow, gray_edge, line_name in zip(self.df["idx_or"], self.df["idx_ex"],
                                                               self.df["delta_flows"], self.df["gray_edges"],self.df["line_name"]):
            penwidth = fabs(reported_flow) / 10
            min_penwidth=0.1
            if penwidth == 0.0:
                penwidth = min_penwidth
            if i in lines_to_cut:
                g.add_edge(origin, extremity, capacity=float(self.float_precision % reported_flow), label=self.float_precision % reported_flow,
                           color="black", fontsize=10, penwidth=max(float(self.float_precision % penwidth),min_penwidth),
                           constrained=True, name=line_name)#style="dotted, setlinewidth(2)"
            elif gray_edge:  # Gray
                g.add_edge(origin, extremity, capacity=float(self.float_precision % reported_flow), label=self.float_precision % reported_flow,
                           color="gray", fontsize=10, penwidth=max(float(self.float_precision % penwidth),min_penwidth),name=line_name)
            elif reported_flow < 0:  # Blue
                g.add_edge(origin, extremity, capacity=float(self.float_precision % reported_flow), label=self.float_precision % reported_flow,
                           color="blue", fontsize=10, penwidth=max(float(self.float_precision % penwidth),min_penwidth),name=line_name)
            else:  # > 0  # Red
                g.add_edge(origin, extremity, capacity=float(self.float_precision % reported_flow), label=self.float_precision % reported_flow,
                           color="coral",#orange"#ff8000"#"coral",
                           fontsize=10, penwidth=max(float(self.float_precision % penwidth),min_penwidth),name=line_name)#"#ff8000")#orange
            i += 1
        #nx.set_edge_attributes(g, {e:self.df["line_name"][i] for i,e in enumerate(g.edges)}, name="name")


    def consolidate_constrained_path(self, hub_sources,hub_targets):
        """
        Consolidate constrained blue path for some edges that were discarded with lower values but are actually on the path
        knowing the hubs in the SuscturedOverflowGraph

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
        g_without_pos_edges = delete_color_edges(self.g, "coral")
        for source, target in zip(hub_sources, hub_targets):
            paths = nx.all_simple_edge_paths(g_without_pos_edges, source, target)
            for path in paths:
                all_edges_to_recolor += path

        all_edges_to_recolor=set(all_edges_to_recolor)

        current_colors = nx.get_edge_attributes(self.g, 'color')
        current_weights=nx.get_edge_attributes(self.g, 'capacity') #############################
        edge_attribues_to_set = {edge: {"color": "blue"} for edge in all_edges_to_recolor if current_colors[edge]!="black" and float(current_weights[edge])!=0}
        nx.set_edge_attributes(self.g, edge_attribues_to_set)

        #########
        #correction: reverse edges with positive values
        current_capacities = nx.get_edge_attributes(self.g, 'capacity')
        edges_to_correct=[edge for edge in all_edges_to_recolor if current_capacities[edge]>0]
        reverse_edges=[(edge_ex,edge_or,edge_properties) for edge_or,edge_ex,edge_properties in self.g.edges(data=True) if edge_properties["color"]=="blue" and edge_properties["capacity"]>0]
        self.g.add_edges_from(reverse_edges)
        self.g.remove_edges_from(edges_to_correct)

        #correct capacity values with opposite value after reversing edge
        current_capacities = nx.get_edge_attributes(self.g, 'capacity')
        current_colors = nx.get_edge_attributes(self.g, 'color')
        edge_attribues_to_set = {edge: {"capacity": -capacity,"label":str(-capacity)}
                                 for edge,color,capacity in zip(self.g.edges,current_colors.values(),current_capacities.values()) if
                                 capacity>0 and color=="blue"}
        nx.set_edge_attributes(self.g, edge_attribues_to_set)

        ############
        #for null flow redispatch, if connected to nodes on blue path, reverse it and make it blue for it to belong there
        blue_edges=[edge for edge in self.g.edges if current_colors[edge]=="blue"]
        nodes_blue_path=self.g.edge_subgraph(blue_edges).nodes

        overall_constrained_graph=self.g.subgraph(nodes_blue_path)

        current_capacities = nx.get_edge_attributes(overall_constrained_graph, 'capacity')
        current_colors = nx.get_edge_attributes(overall_constrained_graph, 'color')
        edges_to_correct = [edge for edge,capacity,color in zip(overall_constrained_graph.edges,current_capacities.values(),current_colors.values()) if capacity==0 and color!="blue"]
        reverse_edges = [(edge_ex, edge_or, edge_properties) for edge_or, edge_ex, edge_properties in
                         overall_constrained_graph.edges(data=True) if
                         edge_properties["color"] != "blue" and edge_properties["capacity"] == 0]

        self.g.add_edges_from(reverse_edges)
        self.g.remove_edges_from(edges_to_correct)


        print("ok")

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

    def consolidate_loop_path(self, hub_sources,hub_targets):
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

    def consolidate_graph(self, structured_graph):
        """
        Consolidate overflow graph knwoing structural elements from SuscturedOverflowGraph

        Parameters
        ----------

        structured_graph: ``SuscturedOverflowGraph``
            a structured graph with identified constrained path, hubs, loop paths

        """

        # consolider le chemin en contrainte avec la connaissance des hubs, en itérant une fois de plus
        n_hubs_init = 0
        hubs_paths = structured_graph.find_loops()[["Source", "Target"]].drop_duplicates()
        n_hub_paths = hubs_paths.shape[0]

        while n_hubs_init != n_hub_paths:
            n_hubs_init = n_hub_paths

            self.consolidate_constrained_path(hubs_paths.Source, hubs_paths.Target)
            structured_graph = Structured_Overload_Distribution_Graph(self.g)

            hubs_paths = structured_graph.find_loops()[["Source", "Target"]].drop_duplicates()
            n_hub_paths = hubs_paths.shape[0]

        #recolor and reverse blue or red edges outside of constrained or loop paths
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
        lines_constrained_path, nodes_constrained_path = structured_graph.get_constrained_edges_nodes()
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
        else:
            # If there are fewer than two nodes in the ambiguous path that are in the constrained path,
            # it is classified as a loop path
            return "loop_path"

    def rename_nodes(self,mapping):
        self.g = nx.relabel_nodes(self.g, mapping, copy=True)
        self.df["idx_or"]=[mapping[idx_or] for idx_or in self.df["idx_or"]]
        self.df["idx_ex"] = [mapping[idx_or] for idx_or in self.df["idx_ex"]]

    def add_double_edges_null_redispatch(self):
        """
        Make edges bi-directionnal when flow redispatch value is null

        Returns
        ----------
        edges_to_double_name_dict: ``dict`` str: networkx edge
            original set of edges that has been doubled, with line names as key and edge as value

        edges_added_name_dict: ``dict`` str: networkx edge
            new set of edges that doubles the original ones in the other direction, with line name as key and edge as value

        """
        init_edges_names=nx.get_edge_attributes(self.g, "name")
        init_colors=nx.get_edge_attributes(self.g, "color").values()
        init_capacity = nx.get_edge_attributes(self.g, "capacity").values()
        edges_to_double_name_dict={name:edge for edge,name,color,capacity in zip(init_edges_names.keys(),init_edges_names.values(),init_colors,init_capacity) if color=="gray" and capacity==0.}

        edges_to_double_data=[ (edge_or, edge_ex, edge_properties) for edge_or,edge_ex,edge_properties in self.g.edges(data=True) if edge_properties["color"]=="gray" and edge_properties["capacity"]==0.]
        edges_to_add_data=[(edge_ex, edge_or, edge_properties) for edge_or,edge_ex,edge_properties in edges_to_double_data]
        self.g.add_edges_from(edges_to_add_data)

        new_edges_names=nx.get_edge_attributes(self.g, "name")#.keys()
        only_new_edges=set(new_edges_names.keys()) - set(init_edges_names.keys())
        edges_added_name_dict={name:edge for edge,name in new_edges_names.items() if edge in only_new_edges}

        assert(set(edges_to_double_name_dict.keys())==set(edges_added_name_dict.keys()))
        return edges_to_double_name_dict,edges_added_name_dict

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
        self.add_relevant_null_flow_lines(structured_graph, non_connected_lines,non_reconnectable_lines, target_path="red_only")
        self.add_relevant_null_flow_lines(structured_graph,non_connected_lines,non_reconnectable_lines,target_path="blue_to_red")
        self.add_relevant_null_flow_lines(structured_graph, non_connected_lines,non_reconnectable_lines, target_path="blue_only")


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
            target path on which to highlight disconnected lines. either blue_only, red_only, blue_to_red

        non_connected_lines: ``array``
            list of lines that are non connected and cannot be reconnected
        """
        edges_to_double, edges_double_added=self.add_double_edges_null_redispatch()#making null flow redispatch lines bidirectionnal

        non_connected_lines=list(set(non_connected_lines+non_reconnectable_lines))#make sure we consider them all in the first place, and differentiate their case after

        edge_names = nx.get_edge_attributes(self.g, 'name')
        edges_non_connected_lines = set(
            [edge for edge, edge_name in edge_names.items() if edge_name in non_connected_lines])

        edges_non_reconnectable_lines= set([edge for edge, edge_name in edge_names.items() if edge_name in non_reconnectable_lines])

        # detect connected components for gray edges among which we will try to detect
        # some non_connected_lines of interest, that link constrained path and loop paths
        g_only_gray_components = delete_color_edges(structured_graph.g_without_pos_edges, "blue")
        g_only_gray_components = delete_color_edges(g_only_gray_components, "black")
        g_only_blue_components = delete_color_edges(structured_graph.g_without_pos_edges, "gray")

        S = [g_only_gray_components.subgraph(c).copy() for c in nx.weakly_connected_components(g_only_gray_components)]

        #between nodes on the constrained path and on the loop paths, detect edge paths that pass through our lines of interest

        #recover possible target and source nodes on constrained paths and loop paths
        edges_to_keep = set()
        edges_non_reconnectable=set()
        node_red_paths=[]
        if structured_graph.red_loops.Path.shape[0]!=0:
            node_red_paths = set(structured_graph.red_loops.Path.sum())
        node_amont_constrained_path = structured_graph.constrained_path.n_amont()
        node_aval_constrained_path = structured_graph.constrained_path.n_aval()

        for g_c in S:
            #detect new edges with null-flow to highlight on constrained path
            if target_path=="blue_only":
                nodes_interest=structured_graph.constrained_path.full_n_constrained_path()

                intersect_constrained_path_amont = set(g_c).intersection(set(node_amont_constrained_path))
                intersect_constrained_path_aval = set(g_c).intersection(set(node_aval_constrained_path))

                #only look at edges that connect "amont" path on one side "aval" path on the other side.
                #edges that would connect "amont" and "aval" path should be rather considered as a new loop path and tagged blue
                edges_to_keep_path, edges_non_reconnectable_path=self.detect_edges_to_keep(g_c, intersect_constrained_path_amont, intersect_constrained_path_amont, edges_non_connected_lines,edges_non_reconnectable_lines)
                edges_to_keep.update(edges_to_keep_path)
                edges_non_reconnectable.update(edges_non_reconnectable_path)

                edges_to_keep_path, edges_non_reconnectable_path=self.detect_edges_to_keep(g_c, intersect_constrained_path_aval, intersect_constrained_path_aval, edges_non_connected_lines,edges_non_reconnectable_lines)
                edges_to_keep.update(edges_to_keep_path)
                edges_non_reconnectable.update(edges_non_reconnectable_path)

            # detect new edges with null-flow to highlight on red paths
            elif target_path=="red_only":
                intersect_red_path = set(g_c).intersection(set(node_red_paths))
                edges_to_keep_path, edges_non_reconnectable_path=self.detect_edges_to_keep(g_c, intersect_red_path, intersect_red_path, edges_non_connected_lines,edges_non_reconnectable_lines)
                edges_to_keep.update(edges_to_keep_path)
                edges_non_reconnectable.update(edges_non_reconnectable_path)

            # detect new edges with null-flow to highlight in between constrained path and red paths
            elif target_path=="blue_to_red":

                intersect_constrained_path_amont = set(g_c).intersection(set(node_amont_constrained_path))
                intersect_constrained_path_aval = set(g_c).intersection(set(node_aval_constrained_path))
                intersect_red_path = set(g_c).intersection(set(node_red_paths))

                # look for edges from constrained path ("amont", before the constraint) to red_path
                if len(intersect_constrained_path_amont) != 0:
                    edges_to_keep_path, edges_non_reconnectable_path=self.detect_edges_to_keep(g_c, intersect_constrained_path_amont, intersect_red_path,
                                                              edges_non_connected_lines,edges_non_reconnectable_lines)
                    edges_to_keep.update(edges_to_keep_path)
                    edges_non_reconnectable.update(edges_non_reconnectable_path)
                # look for edges from red_path to constrained path ("aval", after the constraint)
                if len(intersect_constrained_path_aval) != 0:
                    edges_to_keep_path, edges_non_reconnectable_path =self.detect_edges_to_keep(g_c, intersect_red_path, intersect_constrained_path_aval,
                                             edges_non_connected_lines,edges_non_reconnectable_lines)
                    edges_to_keep.update(edges_to_keep_path)
                    edges_non_reconnectable.update(edges_non_reconnectable_path)


                #look for a new loop path that could exist with disconnected lines
                if len(intersect_constrained_path_amont)!=0 and len(intersect_constrained_path_aval) != 0:
                    edges_to_keep_path, edges_non_reconnectable_path=self.detect_edges_to_keep(g_c, intersect_constrained_path_amont, intersect_constrained_path_aval,
                                                  edges_non_connected_lines,edges_non_reconnectable_lines)
                    edges_to_keep.update(edges_to_keep_path)
                    edges_non_reconnectable.update(edges_non_reconnectable_path)


        #color those edges in blue or red
        if target_path=="blue_only":
            edge_attribues_to_set = {edge: {"color": "blue"} for edge in edges_to_keep}
        elif target_path=="blue_to_red":
            current_weights = nx.get_edge_attributes(self.g, 'capacity')
            edge_attribues_to_set = {edge: {"color": "coral"} for edge in edges_to_keep}
            #mark negative edges as blue
            edge_attribues_to_set.update({edge: {"color": "blue"} for edge in edges_to_keep if current_weights[edge]<0})
        else:
            edge_attribues_to_set = {edge: {"color": "coral"} for edge in edges_to_keep}

        #make special case for non reconnectable lines
        edge_attribues_to_set.update({edge: {"color": "dimgray"} for edge in edges_non_reconnectable})

        nx.set_edge_attributes(self.g, edge_attribues_to_set)

        # represent null-flow edges with new colors as dashed lines
        edges_non_connected_lines_displayed=edges_to_keep.intersection(edges_non_connected_lines)
        edge_attribues_to_set = {edge: {"style": "dashed"} for edge in edges_non_connected_lines_displayed}
        # make special case for non reconnectable lines
        edges_non_reconnectable_lines_displayed = edges_non_reconnectable.intersection(edges_non_reconnectable_lines)
        edge_attribues_to_set.update(
            {edge: {"style": "dotted"} for edge in edges_non_reconnectable_lines_displayed})#only for actually disconnected lines

        nx.set_edge_attributes(self.g, edge_attribues_to_set)

        #also make no direction for non reconnetable edges
        edge_dirs = {edge: "none" for edge in edges_non_reconnectable}
        nx.set_edge_attributes(self.g, edge_dirs, "dir")

        #remove added double edges not used
        self.remove_unused_added_double_edge(edges_to_keep,edges_to_double, edges_double_added)

    def detect_edges_to_keep(self,g_c, source_nodes, target_nodes, edges_of_interest,non_reconnectable_edges=[]):
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
        edges_to_keep_reconnectable = []
        edges_to_keep_non_reconnectable=[]
        for source_node in source_nodes:
            for target_node in target_nodes:

                edge_paths = list(nx.all_simple_edge_paths(g_c, source=source_node, target=target_node))
                for path in edge_paths:
                    if len(set(path).intersection(set(edges_of_interest))) != 0:
                        if len(set(path).intersection(set(non_reconnectable_edges))) != 0:
                            edges_to_keep_non_reconnectable += path
                        else:
                            edges_to_keep_reconnectable += path
        return set(edges_to_keep_reconnectable),set(edges_to_keep_non_reconnectable)

    def remove_unused_added_double_edge(self, edges_to_keep, edges_to_double, edges_double_added):

        """
        Make edges bi-directionnal when flow redispatch value is null

        Parameters
        ----------
         edges_to_keep: ``set`` str
            set of edges of interest found on paths and to be recoloured

        edges_to_double: ``dict`` str: networkx edge
            original set of edges that has been doubled, with line names as key and edge as value

        edges_double_added: ``dict`` str: networkx edge
            new set of edges that doubles the original ones in the other direction, with line name as key and edge as value

        """
        name_edges_to_keep = nx.get_edge_attributes(self.g.edge_subgraph(edges_to_keep), "name").keys()

        # for added double edges that has not been recoloured, remove them
        edges_to_remove=[edge for name,edge in edges_double_added.items() if self.g.edges[edge]["color"]=="gray"]

        # for initial edges that has not been recoloured but for which the added double edge has been, remove those initial edges
        edges_to_remove+=[edge for name,edge in edges_to_double.items() if name in name_edges_to_keep and self.g.edges[edge]["color"]=="gray"]
        self.g.remove_edges_from(edges_to_remove)

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
            if color == "black":
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

        return list(set(edges_constrained_path)), nodes_constrained_path

    def get_dispatch_edges_nodes(self):
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
        if len(g_red.nodes)!=0 and self.find_loops()["Path"].shape[0]!=0:
            list_nodes_dispatch_path = list(set(self.find_loops()["Path"].sum()))

            edge_names_red = nx.get_edge_attributes(g_red, 'name')

            lines_redispatch = [edge_name for edge, edge_name in edge_names_red.items() if
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