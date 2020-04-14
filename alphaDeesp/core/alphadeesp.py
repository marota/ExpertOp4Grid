""" This file is the main file for the Expert Agent called AlphaDeesp """
import networkx as nx
import pandas as pd
import itertools
import pprint
import numpy as np

from alphaDeesp.core.constrainedPath import ConstrainedPath
from alphaDeesp.core.elements import *
from math import fabs, ceil
import subprocess


# from . import Printer


class AlphaDeesp:  # AKA SOLVER
    def __init__(self, _g, df_of_g, printer=None, custom_layout=None, simulator_data=None, debug=False):
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
        self.printer = printer
        self.custom_layout = custom_layout

        self.g_without_pos_edges = self.delete_color_edges(self.g, "red")
        self.g_only_blue_components = self.delete_color_edges(self.g_without_pos_edges, "gray")
        self.g_without_constrained_edge = self.delete_color_edges(self.g, "black")
        self.g_without_gray_and_c_edge = self.delete_color_edges(self.g_without_constrained_edge, "gray")
        self.g_only_red_components = self.delete_color_edges(self.g_without_gray_and_c_edge, "blue")

        # printer.display_geo(self.g_without_pos_edges, custom_layout, name="g_without_pos_edges")
        # printer.display_geo(self.g_only_blue_components, custom_layout, name="g_only_blue")
        # printer.display_geo(self.g_without_constrained_edge, custom_layout, name="g_without_black")

        e_amont, constrained_edge, e_aval = self.get_constrained_path()
        self.constrained_path = ConstrainedPath(e_amont, constrained_edge, e_aval)
        print("n_amont = ", self.constrained_path.n_amont())
        print("n_aval = ", self.constrained_path.n_aval())

        self.hubs = self.get_hubs()

        # red_loops is a dataFrame
        self.red_loops = self.get_loops()
        print("self.red_loops = ")
        print(self.red_loops)

        # this function takes the dataFrame self.red_loops and adds the min cut_values to it.
        self.rank_red_loops()

        # here we classify nodes into 4 categories
        self.structured_topological_actions = self.identify_routing_buses()  # it is a dict
        print("#########################################################################")
        print("structured_top_actions =", self.structured_topological_actions)
        print("#########################################################################")

        self.ranked_combinations = self.compute_best_topologies()

        for ranked_comb in self.ranked_combinations:
            print("---------------------------")
            print(ranked_comb)

        if self.boolean_dump_data_to_file:
            self.ranked_combinations[0].to_csv("./result_ranked_combinations.csv", index=True)

        # The order to get structures is:
        # constrained_path
        # hubs
        # parallel loops and loops

        # self.simulate_network_change(self.ranked_combinations)

    def load2(self, observation, line_to_cut: int):
        """@:arg observation: a pypownet observation,
        line_to_cut: line to cut for overload graph"""

    def simulate_network_change(self, ranked_combinations):
        """This function takes a dataFrame ranked_combinations and computes new changes with Pypownet"""
        pass

    def get_ranked_combinations(self):
        return self.ranked_combinations

    def compute_best_topologies(self):
        """
        inputs: selected_ranked_nodes (after having identified routing buses, ie 4 categories)
        :return: pd.DataFrame containing all topologies scored.
        """

        selected_ranked_nodes = []

        print("Nodes to explore in order are: ")
        for key_indice in list(self.structured_topological_actions.keys()):
            res = self.structured_topological_actions[key_indice]
            if res is not None:
                if res:
                    for elem in res:
                        selected_ranked_nodes.append(elem)
                print(self.structured_topological_actions[key_indice])

        print("selected ranked nodes = ", selected_ranked_nodes)

        # selected_ranked_nodes = [4]
        # selected_ranked_nodes = [5, 4, 12]

        # selected_ranked_nodes = [11]
        # ranked_combinations_structure_initiation = {
        #     "score": ["XX"],
        #     "topology": [["X", "X", "X"]],
        #     "node": ["X"]
        # }
        # best_topologies = pd.DataFrame(ranked_combinations_structure_initiation)

        res_container = []

        for node in selected_ranked_nodes:

            all_combinations = self.compute_all_combinations(node)
            # print("for node [{}], all combinations = {}".format(node, all_combinations))

            ranked_combinations = self.rank_topologies(all_combinations, self.g, node)
            print(ranked_combinations)

            # best_topologies = best_topologies.append(ranked_combinations)
            # pd.concat([best_topologies, *ranked_combinations])

            print("\n##############################################################################")
            print("##########............BEST_TOPOLOGIES COMPUTED............####################")
            print("##############################################################################")

            # best_topologies = self.clean_and_sort_best_topologies(best_topologies)
            best_topologies = self.clean_and_sort_best_topologies(ranked_combinations)
            res_container.append(best_topologies)
            # print(best_topologies)

        return res_container

    def clean_and_sort_best_topologies(self, best_topologies):
        """This function cleans the Dataframe best_topologies;
        it deletes rows with XX, and sorts the Dataframe. In order to achieve this we have to set_index first."""
        best_topologies = best_topologies.set_index("score")
        best_topologies = best_topologies.drop("XX", axis=0)
        best_topologies = best_topologies.sort_values("score", ascending=False)

        return best_topologies

    def compute_all_combinations(self, node):
        """ Given a node, returns all possible combinations of a node configuration.
        ex: [001], [010], [100], [101], [011]... etc..."""

        # ## check that current topology is not in this list
        # print("node = ", node)
        # print("type = ", type(node))
        # print("keys = ", self.simulator_data["substations_elements"])
        node_configuration = self.simulator_data["substations_elements"][node]
        print("Inside compute_all_comb : for node [{}], node_configuration = {}".format(node, node_configuration))
        length = len(node_configuration)
        if length == 0 or length == 1:
            raise ValueError("Cannot generate combinations out of a configuration with len = 1 or 2")
        elif length == 2:
            return [(1, 1), (0, 0)]
        # elif length == 3:
        #     return [(1, 1, 1), (0, 0, 0)]
        else:
            arg = [n for n in range(length)]
            # print("arg = ", arg)

            res = []
            external_i = 0

            for c in range(2, int(ceil(length/2) + 1)):
                # print("c = ", c)
                res_comb = list(itertools.combinations(arg, c))
                # print(res_comb)

                for pos in res_comb:
                    res.append(list(np.zeros(length, dtype=int)))
                    for p in pos:
                        res[external_i][p] = 1
                    external_i += 1

            return res

    def rank_topologies(self, all_combinations, graph, node_to_change: int):
        """==> ultimate goal: This function returns a DF with topologies ranked
        for the moment:
            takes a topo,
            apply it to graph,
            compute score,
            add to df
            next
        """
        # created dataframe here
        ranked_combinations_structure_initiation = {
            "score": ["XX"],
            "topology": [["X", "X", "X"]],
            "node": ["X"]
        }
        ranked_combinations = pd.DataFrame(ranked_combinations_structure_initiation)
        # print("\n########################### Creating DataFrame ranked_combinations ###########################")
        # print(ranked_combinations)
        # print("########################### XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX ###########################\n")

        # for debug purposes, we override all_combinations
        # all_combinations = [[1, 0, 0, 0, 0, 1]]

        # all_combinations = [[0, 0, 0, 1, 0, 1]]  # best topo for node 5 internalrepr

        # all_combinations = [[1, 1, 0, 0, 0]]
        # all_combinations = [[0, 0, 1, 1, 0]]

        # all_combinations = [[1, 0, 1, 0, 0, 1]]
        # all_combinations = [[1, 1, 0]]
        # print("---------- in rank topologies ----------")
        for topo in all_combinations:
            print("################################################################################")
            print("################################################################################")
            print("################################################################################")
            # NO EFFICIENT IF MANY NODES IN THE GRAPH, WE COULD ONLY EXTRACT THE NODE TO CHANGE
            g_copy = self.g.copy()

            # WARNING the internal_repr is not used further in the code. It is not up to date with the new_graph.
            # Only the original one.
            # the new_graph has new_topo, but has no simulated values from pypownet, only old values with new topo
            print("BEFOREEEEEEEEEEEEEEEEEEEEEEE")
            all_edges_xlabel_attributes = nx.get_edge_attributes(g_copy, "xlabel")  # dict[edge]
            print("all_edges_xlabel_attributes = ", all_edges_xlabel_attributes)

            new_graph, internal_repr = self.apply_new_topo_to_graph(g_copy, topo, node_to_change)

            print("AFTERRRRRRRRRRRRRRRRRRRRRRRRRR")
            all_edges_xlabel_attributes = nx.get_edge_attributes(new_graph, "xlabel")  # dict[edge]
            print("all_edges_xlabel_attributes = ", all_edges_xlabel_attributes)

            if nx.is_weakly_connected(new_graph):
                print("we are inside weakly connected")
                score = self.rank_current_topo_at_node_x(new_graph, node_to_change)
            else:
                print("\n=============================================================================")
                print("WARNING, GRAPH WITH TOPO {} IS NOT CONNECTED, WE SKIP IT".format(topo))
                print("=============================================================================")
                continue

            if self.debug:
                print("\n** RESULTS ** new topo [{}] on node [{}] has a score: [{}]\n".format(topo, node_to_change, score))
            score_data = [score, topo, node_to_change]

            # max_index == last row in dataframe ranked_combinations, to append next row
            max_index = ranked_combinations.shape[0]  # rows
            ranked_combinations.loc[max_index] = score_data
            # print(ranked_combinations)

            # now add the score to a DF with node and topo info

        return ranked_combinations

    def apply_new_topo_to_graph(self, graph: nx.DiGraph, new_topology, node_to_change: int):
        """given  a graph, a node_topoly and a node_id, this function applies the change to the graph
        :return new_graph, internal_repr_dict"""
        if self.debug:
            print("\n====================================== apply new topo to graph ======================================")
            print(" new topology applied = [{}] to node: [{}]".format(new_topology, node_to_change))
            print("======================================================================================================\n")

        # check if there are two nodes, there are 2 different values in new topo 0 and 1
        bus_ids = set(new_topology)
        assert(len(bus_ids) == 2)

        internal_repr_dict = dict(self.simulator_data["substations_elements"])
        # if self.debug:
            # print("MACHINE ADDRESSE self.simulator_data ======> ", hex(id(self.simulator_data["substations_elements"])))
            # print("MACHINE ADDRESSE self.simulator_data ======> ", hex(id(internal_repr_dict)))
            # print("INTERNAL REPR DICT BEFORE CHANGES")
            # pprint.pprint(internal_repr_dict)
#

        new_node_id = int("666" + str(node_to_change))

        element_types = self.simulator_data["substations_elements"][node_to_change]
        # it has to be the same, otherwise it does not make sense, ie, there is an error somewhere
        assert len(element_types) == len(new_topology)

        # BEFORE REMOVING, GET NEEDED INFORMATION ON EDGES: COLORS, WIDTH etc...
        color_edges = {}
        for u, v, color in self.g.edges(data="color"):
            # invert edges that have been marked as SWAPPED in DATAFRAME.
            condition = list(self.df.query("idx_or == " + str(u) + " & idx_ex == " + str(v))["swapped"])[0]
            color_edges[(u, v)] = color
            if condition:
                color_edges[(v, u)] = color
            else:
                color_edges[(u, v)] = color

        if 1 in new_topology:  # ie, if the topo is not [0, ... , 0]
            # first we delete the node_to_change ==> it deletes all edges for us
            graph.remove_node(node_to_change)

        # ################ PREPROCESSING NODE RECONSTRUCTION PART, IMPORTANT TO GET COLORS RIGHT
        i = 0
        current_node = []  # Busbar 0
        new_node = []  # Busbar 1
        # then, parsing element by element, reconnect the graph.
        for internal_elem, element, element_type in zip(internal_repr_dict[node_to_change], new_topology, element_types):
            if element == 0:
                # print("we were in 0")
                internal_elem.busbar_id = 0
            elif element == 1:
                # print("we were in 1")
                internal_elem.busbar_id = 1

        # WE RECONSTRUCT INTERNAL REPR
        # prod = {0: 0, 1: 0}  # busid:value
        # load = {0: 0, 1: 0}  # busid:value
        prod = {}
        load = {}

        # for element in internal_repr_dict[node_to_change]:
        #     if element.busbar_id == 0:
        #         if isinstance(element, Production):
        #             prod[0] += fabs(element.value)
        #
        #         elif isinstance(element, Consumption):
        #             load[0] += fabs(element.value)
        #
        #     elif element.busbar_id == 1:
        #         if isinstance(element, Production):
        #             prod[1] += fabs(element.value)
        #
        #         elif isinstance(element, Consumption):
        #             load[1] += fabs(element.value)

        for bus_id in bus_ids:
            for element in internal_repr_dict[node_to_change]:
                if element.busbar_id == bus_id:
                    if bus_id not in prod.keys():
                        if isinstance(element, Production):
                            prod[bus_id] = fabs(element.value)
                    else:
                        if isinstance(element, Production):
                            prod[bus_id] += fabs(element.value)

                    if bus_id not in load.keys():
                        if isinstance(element, Consumption):
                            load[bus_id] = fabs(element.value)
                    else:
                        if isinstance(element, Consumption):
                            load[bus_id] += fabs(element.value)

        # print("prod = ", prod)
        # print("load = ", load)
        node_type = {}

        # node_type = {}
        # for bus_id in bus_ids:
        #     for element in internal_repr_dict[node_to_change]:
        #         prod = 0
        #         prod_type = False
        #         load = 0
        #         load_type = False
        #         if element.busbar_id == bus_id:
        #             # check all
        #             if isinstance(element, Production):
        #                 node_type[bus_id] = "prod"

        for bus_id in bus_ids:  # [0, 1]
            if bus_id in prod.keys() and bus_id in load.keys():
                prod_minus_load = prod[bus_id] - load[bus_id]
                if prod_minus_load > 0:
                    node_type[bus_id] = "prod"
                else:
                    node_type[bus_id] = "load"
            elif bus_id in prod.keys():
                node_type[bus_id] = "prod"
            elif bus_id in load.keys():
                node_type[bus_id] = "load"

            # if self.debug:
            #     print("NODE TYPE = ")
            #     print(node_type)

            node_label = node_to_change
            prod_minus_load = 0
            if bus_id == 1:
                node_label = new_node_id

            # if bus_id in prod.keys() and bus_id in load.keys():
            #     prod_minus_load = prod[bus_id] - load[bus_id]
            # elif bus_id in prod.keys():
            #     prod_minus_load = prod[bus_id]
            # elif bus_id in load.keys():
            #     prod_minus_load = load[bus_id]

            if bus_id in node_type.keys():
                if node_type[bus_id] == "prod":  # PROD
                    prod_minus_load = prod[bus_id]
                    graph.add_node(node_label, pin=True, prod_or_load="prod", value=str(prod_minus_load),
                                   style="filled", fillcolor="#f30000")
                elif node_type[bus_id] == "load":
                    prod_minus_load = load[bus_id]
                    graph.add_node(node_label, pin=True, prod_or_load="load", value=str(-prod_minus_load),
                                   style="filled", fillcolor="#478fd0")
            else:  # WHITE
                graph.add_node(node_label, pin=True, prod_or_load="load", value=str(prod_minus_load),
                               style="filled", fillcolor="#ffffff")

        # prod_minus_load_0 = prod[0] - load[0]
        # prod_minus_load_1 = prod[1] - load[1]
        # prod_minus_load.append(prod[0] - load[0], prod_minus_load)

        # node busbar 0
        # if prod_minus_load_0 > 0:  # PROD
        #     graph.add_node(node_to_change, pin=True, prod_or_load="prod", style="filled", fillcolor="#f30000")
        # elif prod_minus_load_0 < 0:  # LOAD
        #     graph.add_node(node_to_change, pin=True, prod_or_load="load", style="filled", fillcolor="#478fd0")
        # elif prod_minus_load_0 == 0:  # WHITE
        #     graph.add_node(node_to_change, pin=True, prod_or_load="load", style="filled", fillcolor="#ffffff")

        # THEN FROM INTERNAL REPR WE RECREATE GRAPH

        # ############ ACTUAL NODE RECONSTRUCTION
        # Algo:
        # If len(current_node) == 2:
        # If Prod and Cons:
        #   [0]isProd - [1]isCons => get right color
        # elif [0]isProd:
        #   => add RED Node
        # elif [0]isCons:
        #   => add BLUE Node
        # elif [0] is Or or Ex then good
        #   => add WHITE Node
        # CURRENT_NODE ==> BusBar 0
        # if len(current_node) == 2:
        #     if isinstance(current_node[0], Production) and isinstance(current_node[1], Consumption):# if Prod and Cons
        #         # this means we have a production and a consumption.
        #         prod_minus_load = current_node[0].value - current_node[1].value
        #         if prod_minus_load > 0:  # PROD
        #             graph.add_node(node_to_change, pin=True, prod_or_load="prod", style="filled", fillcolor="#f30000")
        #         else:  # LOAD
        #             graph.add_node(node_to_change, pin=True, prod_or_load="load", style="filled", fillcolor="#478fd0")
        #     if isinstance(current_node[0], Production):
        #         graph.add_node(node_to_change, pin=True, prod_or_load="prod", style="filled", fillcolor="#f30000")
        #     elif isinstance(current_node[0], Consumption):
        #         graph.add_node(node_to_change, pin=True, prod_or_load="load", style="filled", fillcolor="#478fd0")
        #
        # elif len(current_node) == 1:
        #     if isinstance(current_node[0], Production):
        #         graph.add_node(node_to_change, pin=True, prod_or_load="prod", style="filled", fillcolor="#f30000")
        #     elif isinstance(current_node[0], Consumption):
        #         graph.add_node(node_to_change, pin=True, prod_or_load="load", style="filled", fillcolor="#478fd0")
        #
        # # NEW_NODE ==> BusBar 1
        # if len(new_node) == 2:
        #     if isinstance(new_node[0], Production) and isinstance(new_node[1], Consumption):
        #         # this means we have a production and a consumption.
        #         prod_minus_load = new_node[0].value - new_node[1].value
        #         if prod_minus_load > 0:  # PROD
        #             graph.add_node(new_node_id, pin=True, prod_or_load="prod", style="filled", fillcolor="#f30000")
        #         else: # LOAD
        #             graph.add_node(new_node_id, pin=True, prod_or_load="load", style="filled", fillcolor="#478fd0")
        #
        #     elif isinstance(new_node[0], Production):
        #         graph.add_node(new_node_id, pin=True, prod_or_load="prod", style="filled", fillcolor="#f30000")
        #     elif isinstance(new_node[0], Consumption):
        #         graph.add_node(new_node_id, pin=True, prod_or_load="load", style="filled", fillcolor="#478fd0")
        #
        # elif len(new_node) == 1:
        #     if isinstance(new_node[0], Production):
        #         graph.add_node(new_node_id, pin=True, prod_or_load="prod", style="filled", fillcolor="#f30000")
        #     elif isinstance(new_node[0], Consumption):
        #         graph.add_node(new_node_id, pin=True, prod_or_load="load", style="filled", fillcolor="#478fd0")

        # ################ EDGE RECONSTRUCTION PART

        i = 0
        # then, parsing element by element, reconnect the graph.
        for element, element_type in zip(new_topology, element_types):
            print("element = ", element)
            print("element type = ", element_type)
            reported_flow = None
            edge_color = None
            penwidth = None
            if isinstance(element_type, OriginLine) or isinstance(element_type, ExtremityLine):
                # print("element_type.flow_value=", element_type.flow_value)
                reported_flow = element_type.flow_value[0]
                # print("reported_flow = ", reported_flow)
                penwidth = fabs(reported_flow) / 10
                if penwidth == 0.0:
                    penwidth = 0.1

            if element == 1:  # connect FROM or TO node: 666XX
                if isinstance(element_type, OriginLine):
                    graph.add_edge(new_node_id, element_type.end_substation_id,
                                   capacity=float("%.2f" % reported_flow), xlabel="%.2f" % reported_flow,
                                   color=color_edges[(node_to_change, element_type.end_substation_id)],
                                   fontsize=10, penwidth="%.2f" % penwidth)

                elif isinstance(element_type, ExtremityLine):
                    graph.add_edge(element_type.start_substation_id, new_node_id,
                                   capacity=float("%.2f" % reported_flow), xlabel="%.2f" % reported_flow,
                                   color=color_edges[(element_type.start_substation_id, node_to_change)],
                                   fontsize=10, penwidth="%.2f" % penwidth)

            elif element == 0:  # connect from node node:_to_change
                if isinstance(element_type, OriginLine):
                    graph.add_edge(node_to_change, element_type.end_substation_id,
                                   capacity=float("%.2f" % reported_flow), xlabel="%.2f" % reported_flow,
                                   color=color_edges[(node_to_change, element_type.end_substation_id)],
                                   fontsize=10, penwidth="%.2f" % penwidth)

                elif isinstance(element_type, ExtremityLine):
                    graph.add_edge(element_type.start_substation_id, node_to_change,
                                   capacity=float("%.2f" % reported_flow), xlabel="%.2f" % reported_flow,
                                   color=color_edges[(element_type.start_substation_id, node_to_change)],
                                   fontsize=10, penwidth="%.2f" % penwidth)

            else:
                raise ValueError("Error element has to belong to either Busbar 1 or 2")

            i += 1

        # print("---------------- finished applying new topo to graph ----------------")
        # if node_to_change == 5 and new_topology == [1, 0, 0, 0, 0, 1]:
        #     name = "".join(str(e) for e in new_topology)
        #     name = str(node_to_change) + "_" + name
        #     self.printer.display_geo(graph, self.custom_layout, name=name)
        # if new_topology == [1, 0, 0, 0, 0, 1]:
        #     return

        name = "".join(str(e) for e in new_topology)
        name = str(node_to_change) + "_" + name
        self.bag_of_graphs[name] = graph
        # self.printer.display_geo(graph, self.custom_layout, name=name)

        return graph, internal_repr_dict

    def rank_current_topo_at_node_x(self, graph, node: int):
        """This function ranks current topology at node X"""

        # if self.debug:
        print("\n-------------------------------------------------------------------------------------------------")
        print("------------------------------- INSIDE RANK_CURRENT_TOPO_AT_NODE_X ------------------------------")
        print("-------------------------------------------------------------------------------------------------\n")

        final_score = 0.0
        all_nodes_value_attributes = nx.get_node_attributes(graph, "value")  # dict[node]
        # print("all_nodes_value_attributes = ", all_nodes_value_attributes)

        all_edges_color_attributes = nx.get_edge_attributes(graph, "color")  # dict[edge]
        # print("all_edges_color_attributes = ")
        # pprint.pprint(all_edges_color_attributes)

        all_edges_xlabel_attributes = nx.get_edge_attributes(graph, "xlabel")  # dict[edge]
        print("all_edges_xlabel_attributes = ", all_edges_xlabel_attributes)

        #  ########## IS IN AMONT ##########
        if node in self.constrained_path.n_amont():
        # if self.is_in_amont(graph, node):
            if self.debug:
                print("||||||||||||||||||||||||||| node [{}] is in_Amont of constrained_edge".format(node))
            in_negative_flows = []
            in_positive_flows = []
            out_positive_flows = []

            # pick the node that does not belong to cpath.
            # first identify the node that is not connected to cpath. either node or 666+node

            print("OUT EDGES = ", graph.out_edges(node))
            for edge in graph.out_edges(node):
                edge_color = all_edges_color_attributes[edge]
                edge_value = all_edges_xlabel_attributes[edge]
                # if there is a outgoing negative blue or black edge this means we are connected to cpath.
                # therefore change to twin node 666+node
                print("EDGE VALUE = ", edge_value)
                # if float(edge_value) < 0 and (edge_color == "blue" or edge_color == "black"):
                if edge_color == "blue" or edge_color == "black":
                    print("WE GOT IN THE IF")
                    if self.debug:
                        print("\n######################################################")
                        print("Node [{}] is not connected to cpath. Twin node selected...".format(node))
                        print("######################################################")
                    if "666" in str(node):
                        # TODO
                        # remove 666 from node
                        pass
                    else:
                        node = int("666" + str(node))

            # somme des reports négatifs entrants + sommes des reports positifs entrants
            # print("g in edges({}) = {}".format(node, list(graph.in_edges(node))))
            for edge in graph.in_edges(node):
                edge_flow_value = float(all_edges_xlabel_attributes[edge])
                if edge_flow_value < 0:
                    in_negative_flows.append(fabs(edge_flow_value))
                else:
                    in_positive_flows.append(edge_flow_value)

            # somme des reports positifs sortant +
            for edge in graph.out_edges(node):
                edge_flow_value = float(all_edges_xlabel_attributes[edge])
                if edge_flow_value > 0:
                    out_positive_flows.append(edge_flow_value)

            # if self.debug:
            print("incoming negative flows node [{}] = ".format(node))
            print(in_negative_flows)
            print("sum in_negative_flows = ", sum(in_negative_flows))
            print("sum in_positive_flows = ", sum(in_positive_flows))
            print("sum out_positive_flows = ", sum(out_positive_flows))

            # somme Productions - somme Consommation.
            # print("substations_elements = ", self.simulator_data["substations_elements"].keys())
            # print("res = ", self.simulator_data["substations_elements"][node])
            # sum_prod = 0.0
            # sum_cons = 0.0
            # for element in self.simulator_data["substations_elements"][node]:
            #     if isinstance(element, Production):
            #         sum_prod += element.value
            #     elif isinstance(element, Consumption):
            #         sum_cons += element.value
            # diff_sums = sum_prod - sum_cons

            diff_sums = float(all_nodes_value_attributes[node])
            max_pos_in_or_out_flows = max(sum(out_positive_flows), sum(in_positive_flows))
            final_score = np.around(sum(in_negative_flows) + max_pos_in_or_out_flows + diff_sums, decimals=2)
            # final_score = np.around(sum(in_negative_flows)[0] + max_pos_in_or_out_flows + diff_sums, decimals=2)[0]

            if self.debug:
                print("diff_sums = ", diff_sums)
                print(type(diff_sums))
                print("max_pos_in_or_out_flows = ", max_pos_in_or_out_flows)

                print("in negative flows = ", in_negative_flows)
                print("max_pos_in_or_out_flows = ", max_pos_in_or_out_flows)
                print("Final score = ", final_score)

            # print("------------------ for node {} ---------------- ".format(node))

        #  ########## IS IN AVAL ##########
        elif node in self.constrained_path.n_aval():
        # elif self.is_in_aval(graph, node):
            if self.debug:
                print("||||||||||||||||||||||||||| node [{}] is in_Aval of constrained_edge".format(node))

            out_negative_flows = []
            out_positive_flows = []
            in_positive_flows = []

            # first identify the node that is not connected to cpath. either node or 666+node
            for edge in graph.in_edges(node):
                edge_color = all_edges_color_attributes[edge]
                edge_value = all_edges_xlabel_attributes[edge]
                # if there is a incoming negative blue or black edge this means we are connected to cpath.
                # therefore change to twin node 666+node
                if float(edge_value) < 0 and (edge_color == "blue" or edge_color == "black"):
                    if self.debug:
                        print("\n######################################################")
                        print("Node [{}] is not connected to cpath. Twin node selected...".format(node))
                        print("######################################################")
                    if "666" in str(node):
                        # TODO
                        # remove 666 from node
                        pass
                    else:
                        node = int("666" + str(node))

            # somme des reports negatifs et positifs SORTANT
            for edge in graph.out_edges(node):
                edge_flow_value = float(all_edges_xlabel_attributes[edge])
                if edge_flow_value < 0:
                    out_negative_flows.append(fabs(edge_flow_value))
                else:
                    out_positive_flows.append(edge_flow_value)

            for edge in graph.in_edges(node):
                edge_flow_value = float(all_edges_xlabel_attributes[edge])
                if edge_flow_value > 0:
                    in_positive_flows.append(edge_flow_value)

            # MAX (somme des reports positifs ENTRANT ET SORTANT)
            max_pos_in_or_out_flows = max(sum(out_positive_flows), sum(in_positive_flows))

            if self.debug:
                print("out_negative_flows = ", out_negative_flows)
                print("out_positive_flows = ", out_positive_flows)
                print("in_positive_flows = ", in_positive_flows)
                print("sum out neg = ", sum(out_negative_flows))
                print("sum out pos = ", sum(out_positive_flows))
                print("sum in  pos = ", sum(in_positive_flows))
                print("max_pos_in_or_out_flows = ", max_pos_in_or_out_flows)

            # somme Productions - somme Consommation.
            # print("substations_elements = ", self.simulator_data["substations_elements"].keys())
            # pprint.pprint(self.simulator_data["substations_elements"][node])
            # sum_prod = 0.0
            # sum_cons = 0.0
            # for element in self.simulator_data["substations_elements"][node]:
            #     if isinstance(element, Production):
            #         sum_prod += element.value
            #     elif isinstance(element, Consumption):
            #         sum_cons += element.value
            #         break
            #

            # diff_sums = -(sumProd - sumCons)= sum_CONS - sum_PROD
            # sur le noeud choisi (non connecté au cpath) on souhaite y connecter des consommations et pas des
            # productions pour l'aval.
            diff_sums = -float(all_nodes_value_attributes[node])

            # print("sum_prod = ", sum_prod)
            # print("sum_cons = ", sum_cons)

            final_score = np.around(sum(out_negative_flows) + max_pos_in_or_out_flows + diff_sums, decimals=2)

            if self.debug:
                print("diff_sums = ", diff_sums)
                print("Final score = ", final_score)
                print(type(final_score))
                # node = int("666" + str(node))
                # for edge in graph.in_edges(node):
                #     print("in edge from node {} : {} ".format(node, edge))
                # for edge in graph.out_edges(node):
                #     print("out edge from node {} : {} ".format(node, edge))

        else:
            print("||||||||||||||||||||||||||| node [{}] is not connected to a path to the constrained_edge.".format(node))

        return final_score

    def is_in_aval(self, graph, node):   # in Aval of constrained_edge
        """ This functions check if node is in Aval of constrained_edge"""
        g = self.g
        aval_constrained_node = self.constrained_path.constrained_edge[1]
        if node == aval_constrained_node:
            return True
        # print("aval constrained_node = ", aval_constrained_node)
        # print("list successors = ", list(g.successors(aval_constrained_node)))
        if node in list(g.successors(aval_constrained_node)):
            return True
        else:
            return False

    def is_in_amont(self, graph, node):  # in Amont of constrained_edge
        # g = self.g
        g = graph
        amont_constrained_node = self.constrained_path.constrained_edge[0]
        print("constrained path = ", self.constrained_path)
        if node == amont_constrained_node:
            return True
        # print("amont constrained_node = ", amont_constrained_node)
        # print("list predecessors = ", list(g.predecessors(amont_constrained_node)))
        # print("list successor = ", list(g.successors(amont_constrained_node)))
        if node in list(g.predecessors(amont_constrained_node)):
            return True
        else:
            return False

    @staticmethod
    def is_in_aval_of_node_x(g, node, node_x):
        """This function returns a boolean True or False.
        True if node is in aval of node_x. False if not."""

        nodes_succ = set()
        successors = list(nx.edge_dfs(g, node_x))
        for p in successors:
            for t in p:
                if isinstance(t, int):
                    nodes_succ.add(t)
        print("successors = ", successors)
        print("nodes successors = ", nodes_succ)
        if node in nodes_succ:
            return True
        else:
            return False

    @staticmethod
    def is_in_amont_of_node_x(g, node, node_x):
        # print("amont constrained_node = ", amont_constrained_node)

        nodes_pred = set()
        predecessors = list(nx.edge_dfs(g, node_x, orientation="reverse"))
        for p in predecessors:
            for t in p:  # for tuple in predcessors
                if isinstance(t, int):
                    nodes_pred.add(t)

        print("predecessors = ", predecessors)
        print("nodes predcessors = ", nodes_pred)
        if node in nodes_pred:
            return True
        else:
            return False

        # # res = {t: s for s, t in nx.bfs_edges(g, n, reverse=True)}
        # # print("res =", res)
        # print("g.in_edges(n) = ", g.in_edges(n))
        # print("dict dfs_pred = ", dfs_pred)
        # dfs_suc = nx.dfs_successors(g, n)
        #
        # bfs_pred = nx.bfs_predecessors(g, n)
        # print("bfs pred = ", dict(bfs_pred))
        # # print("dfs pred = ", set(list(dfs_pred.values())))
        # # print("dfs_suc =", dfs_suc)
        #
        #
        # return 0
        # # if node in list(g.predecessors(node_x)):
        # #     return True
        # # else:
        # #     return False


    def sort_hubs(self, hubs):
        # creates a DATAFRAME and sort it, returns the sorted hubs
        print("================= sort_hubs =================")
        if hubs:
            df = pd.DataFrame()
            df["hubs"] = hubs

            # now for each node in hubs, get the max abs(ingoing or outgoing) flow
            flows = []

            for node in hubs:
                flow_compute_ingoing = []
                flow_compute_outgoing = []

                print("node = ", node)

                for i, row in self.df.iterrows():
                    # if row["idx_or"] == node or row["idx_ex"] == node:
                    #     flow_compute.append(fabs(row["delta_flows"]))

                    if row["idx_or"] == node:
                        flow_compute_outgoing.append(fabs(row["delta_flows"]))

                    if row["idx_ex"] == node:
                        flow_compute_ingoing.append(fabs(row["delta_flows"]))

                max_result = max(sum(flow_compute_ingoing), sum(flow_compute_outgoing))
                print("all flows = ", max_result)
                flows.append(max_result)

            df["max_flows"] = flows
            df.sort_values("max_flows", ascending=False, inplace=True)
            print(df)

            return df

        else:
            raise ValueError("There are no hubs")

    def identify_routing_buses(self):
        """Categories 1 to 4
        1. Hubs
        2. all nodes that belong to c_path
        3. On //path
        4. Over Da on c_path"""

        # ALGO
        # get all nodes from c_path, loops, //paths, {set of all those nodes}
        # for nodes in interesting_nodes:
        #   classify to category 1, 2, 3, 4.
        #

        df_sorted_hubs = self.sort_hubs(self.hubs)
        category1 = list(df_sorted_hubs["hubs"])
        set_category2 = set(self.constrained_path.full_n_constrained_path()) - set(category1)
        set_category3 = set()  # @TODO
        set_category4 = set(self.constrained_path.n_aval()) - (set(category1) | set_category2 | set_category3)

        d = {1: category1, 2: set_category2, 3: set_category3, 4: set_category4}

        # ways of prioritizing nodes, from category 1 to 4, and sorted list in each
        # category_1 = self.hubs, that belong to loops and then parallel
        # category_2 = self.constrained_path.full_n_constrained_path()
        # category_3 = self.parallel_paths # check for each node, to which substation it belongs, and if there already
        # exists at least 2 nodes
        # category_4 = self.constrained_path.n_aval()

        return d

    def rank_red_loops(self):
        cut_values = []
        cut_sets = []  # contains the edges that ended up having the minimum cut_values
        for i, row in self.red_loops.iterrows():
            source = row["Source"]
            target = row["Target"]
            p = row["Path"]

            print("=============== source: {}, target: {}".format(source, target))

            cut_value, partition = nx.minimum_cut(self.g_only_red_components, source, target)
            reachable, non_reachable = partition
            print("cut_value: {}, partition: {}".format(cut_value, partition))

            # info from doc - ‘partition’ here is a tuple with the two sets of nodes that define the minimum cut.
            # You can compute the cut set of edges that induce the minimum cut as follows:
            cutset = set()
            for u, nbrs in ((n, self.g_only_red_components[n]) for n in reachable):
                cutset.update((u, v) for v in nbrs if v in non_reachable)
            print("sorted(cutset) = ", sorted(cutset))

            cut_values.append(cut_value)
            cut_sets.append(list(cutset)[0])

        print("cut_values = ", cut_values)

        self.red_loops["min_cut_values"] = cut_values
        self.red_loops["min_cut_edges"] = cut_sets
        print("======================= cut_values added =======================")
        print(self.red_loops)
            # break

    def joke(self):
        print("Heard about the new restaurant called Karma ?...")
        print("....")
        print("There's no menu:")
        print("You get what you deserve.")

    def get_amont_blue_edges(self, g, node):
        res = []
        for e in nx.edge_dfs(g, node, orientation="reverse"):
            if g.edges[(e[0], e[1])]["color"] == "blue":
                res.append((e[0], e[1]))
        return res

    def get_aval_blue_edges(self, g, node):
        res = []
        print("debug AlphaDeesp get aval blue edges")
        print(list(nx.edge_dfs(g, node, orientation="original")))
        for e in nx.edge_dfs(g, node, orientation="original"):
            if g.edges[(e[0], e[1])]["color"] == "blue":
                res.append((e[0], e[1]))
        return res

    def delete_positive_edges(self, _g):
        """Returns a copy of g without positive edges"""
        g = _g.copy()
        # array containing the indices of edges with positive report flow
        pos_edges = []

        # get indices of positive edges
        i = 1
        for u, v, report in g.edges(data="xlabel"):
            if float(report) > 0:
                pos_edges.append((i, (u, v)))
            i += 1

        # delete from graph positive edges
        # this extracts the (u,v) from pos_edges
        # print("pos_edges test = ", list(zip(*pos_edges))[1])
        if pos_edges:
            g.remove_edges_from(list(zip(*pos_edges))[1])
        return g

    def delete_color_edges(self, _g, edge_color):
        """Returns a copy of g without gray edges"""
        g = _g.copy()

        gray_edges = []
        i = 1
        for u, v, color in g.edges(data="color"):
            if color == edge_color:
                gray_edges.append((i, (u, v)))
            i += 1

        # delete from graph gray edges
        # this extracts the (u,v) from pos_edges
        if gray_edges:
            g.remove_edges_from(list(zip(*gray_edges))[1])
        return g

    def from_edges_get_nodes(self, edges):
        """edges is a list of tuples"""
        nodes = []
        for e in edges:
            if isinstance(e, int):
                nodes.append(e)
            else:
                for node in e:
                    if node not in nodes:
                        nodes.append(node)
        return nodes

    def compute_meaningful_structures(self):
        self.data["constrained_path"] = self.get_constrained_path()

    def get_adjacency_matrix(self, g):
        print("adjacency matrix full = ")
        for line in nx.generate_adjlist(g):
            print(line)

    def get_loop_paths(self):
        pass

    def filter_constrained_path(self, path_to_filter):
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

    def get_blue_components(self):
        """return a list of sorted (by biggest len) components (sets of nodes)"""
        # g = _g.copy()

        # g_without_pos = self.delete_positive_edges(g)
        # g_only_blue_comp_left = self.delete_gray_edges(g_without_pos)

        if self.g_only_blue_components is not None:
            res = [(len(c), c) for c in
                   sorted(nx.weakly_connected_components(self.g_only_blue_components), key=len, reverse=True)]
        else:
            raise ValueError(
                "Error : self.g_only_blue_components graph was not properly created in __init__ function...")

        fres = list(filter(lambda x: x[0] > 1, res))  # this filters all components that are just one node

        return fres

    def get_constrained_path(self):
        """Return the constrained path"""
        constrained_edge = None
        tmp_constrained_path = []
        constrained_path = []

        edge_list = nx.get_edge_attributes(self.g_only_blue_components, "color")
        for edge, color in edge_list.items():
            if color == "black":
                constrained_edge = edge

        amont_edges = self.get_amont_blue_edges(self.g_only_blue_components, constrained_edge[0])
        aval_edges = self.get_aval_blue_edges(self.g_only_blue_components, constrained_edge[1])

        amont_nodes = self.from_edges_get_nodes(amont_edges)
        aval_nodes = self.from_edges_get_nodes(aval_edges)
        constrained_node = self.from_edges_get_nodes(constrained_edge)

        tmp_constrained_path.append(amont_edges)
        tmp_constrained_path.append(constrained_edge)
        tmp_constrained_path.append(aval_edges)

        # tmp_constrained_path.append(amont_nodes)
        # tmp_constrained_path.append(constrained_node)
        # tmp_constrained_path.append(aval_nodes)

        # print("constrained_path =", tmp_constrained_path)
        #
        # constrained_path = self.filter_constrained_path(tmp_constrained_path)
        # print("set constrained path = ", constrained_path)
        #
        # components = self.get_blue_components()
        # print("components = ", components)
        #
        # # here we check if there are multiple path from each component to the constrained_path?
        # for component in components:
        #     print(component[1])
        #     if component[1] == set(constrained_path):
        #         print("that comp is the constrained path = ", component)
        #     else:
        #         # HERE TEST THE PATHS ? ASK ANTOINE
        #         print("that comp is linked to nothing = ", component)

        return tmp_constrained_path

    def get_hubs(self):
        """A hub (carrefour_electrique) has a constrained_path and positiv reports"""
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
            in_edges = list(g.in_edges(node))
            for e in in_edges:
                if g.edges[e]["color"] == "red":
                    hubs.append(node)
                    break

        # for nodes in amont, if node has RED outputs (ie outgoing flows) then it is a hub
        for node in self.constrained_path.n_amont():
            out_edges = list(g.out_edges(node))
            for e in out_edges:
                if g.edges[e]["color"] == "red":
                    hubs.append(node)
                    break

        print("get_hubs = ", hubs)
        return hubs

    def get_loops(self):
        """This function returns all parallel paths. After discussing with Antoine, start with the most "en Aval" node,
        and walk in reverse for loops and parallel path returns a dict with all data """

        print("==================== In function get_loops ====================")
        g = self.g_only_red_components
        c_path_n = self.constrained_path.full_n_constrained_path()
        all_loop_paths = {}
        ii = 0

        for i in range(len(c_path_n)):
            for j in reversed(range(len(c_path_n))):
                if i < j:
                    # print(i, j)
                    # print("we compare paths from source: {} to target: {}".format(c_path_n[i], c_path_n[j]))
                    try:
                        res = nx.all_shortest_paths(g, c_path_n[i], c_path_n[j])
                        for p in res:
                            print("path = ", p)
                            all_loop_paths[ii] = p
                            ii += 1
                    except nx.NetworkXNoPath:
                        print("shortest path between {0} and {1} failed".format(c_path_n[i], c_path_n[j]))

        print("### Print in get_loops ###, all_loop_paths")
        pprint.pprint(all_loop_paths)

        data_for_df = {"Source": [], "Target": [], "Path": []}
        for path in list(all_loop_paths.keys()):
            data_for_df["Source"].append(all_loop_paths[path][0])
            data_for_df["Target"].append(all_loop_paths[path][-1])
            data_for_df["Path"].append(all_loop_paths[path])

        # pprint.pprint(data_for_df)

        return pd.DataFrame.from_dict(data_for_df)

    def get_color_path_from_node(self, g, node, _color: str, orientation="original"):
        """This function returns an array with "_color" edges predecessors and successors (depending on orientation)"""
        """orientation has to be : "original" or "reverse" """
        res = {0: []}
        i = 1
        for e in nx.edge_bfs(g, node, orientation=orientation):
            if g.edges[(e[0], e[1])]["color"] == _color:
                print(e)
                if not res[0]:  # if empty
                    res[0].append((e[0], e[1]))

                # if pred
                elif res[0][-1][0] == e[1]:
                    res[0].append((e[0], e[1]))

                # elif e[0] in res.values():

                else:
                    # we check in res, or new branch
                    found = False
                    for p in list(res.keys()):
                        print("we check res[p] = ", res[p])

                        path = list(res[p])
                        if e[1] == path[-1][0]:
                            res[p].append((e[0], e[1]))
                            print("we append e = ", e)
                            found = True
                            break

                    if not found:
                        # create new list
                        res[i] = [(e[0], e[1])]
                        print("we create new list for e = ", e)
                        i += 1

        i += 1
        # for p in list(res.keys()):
        #     res[p] = list(reversed(res[p]))

        return res

    def write_g(self, g):
        """This saves file g"""
        nx.write_edgelist(self.g, "./alphaDeesp/tmp/save.graph")
        print("file saved")
        pass

    def read_g(self):
        pass

########################################################################################################################
# ######################################### EXTERNAL COMMANDS ##########################################################
########################################################################################################################


def execute_command(command: str):
    """
    This function executes a command on the local machine, and fill self.output and self.error with results of
    command.
    @return True if command went through
    """

    # print("command = ", command)
    sub_p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = sub_p.communicate()
    exit_code = sub_p.returncode
    # pid = sub_p.pid

    output = stdout.decode()
    error = stderr.decode()

    print("--------------------\n output is:", output)
    print("--------------------\n stderr is:", error)
    print("--------------------\n exit code is:", exit_code)
    # print("--------------------\n pid is:", pid)

    if not error:
        # string error is empty
        return True
    else:
        # string error is full
        # print(f"Error {error}")
        return False
