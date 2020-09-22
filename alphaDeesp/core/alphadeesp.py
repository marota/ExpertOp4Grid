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

import os


# os.environ['PATH'] += os.pathsep + r'C:\Users\nmegel\graphviz-2.38\release\bin'


class AlphaDeesp:  # AKA SOLVER
    def __init__(self, _g, df_of_g, printer=None, custom_layout=None, simulator_data=None, substation_in_cooldown=[], debug=False):
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
        self.substation_in_cooldown = substation_in_cooldown  # we cannot play with those substations so no need to compute simulations

        # check that line extemity does not have only load or productions: otherwise there is either node merging to do or nothing else

        ranked_combinations_structure_initiation = {
            "score": ["XX"],
            "topology": [["X", "X", "X"]],
            "node": ["X"]
        }
        ranked_combinations = pd.DataFrame(ranked_combinations_structure_initiation)
        # otherwise proceed
        self.g_without_pos_edges = self.delete_color_edges(self.g, "red")
        self.g_only_blue_components = self.delete_color_edges(self.g_without_pos_edges, "gray")
        self.g_without_constrained_edge = self.delete_color_edges(self.g, "black")
        self.g_without_gray_and_c_edge = self.delete_color_edges(self.g_without_constrained_edge, "gray")
        self.g_only_red_components = self.delete_color_edges(self.g_without_gray_and_c_edge, "blue")

        e_amont, constrained_edge, e_aval = self.get_constrained_path()
        self.constrained_path = ConstrainedPath(e_amont, constrained_edge, e_aval)
        # print("n_amont = ", self.constrained_path.n_amont())
        # print("n_aval = ", self.constrained_path.n_aval())

        self.hubs = self.get_hubs()

        # red_loops is a dataFrame
        self.red_loops = self.get_loops()
        # print("self.red_loops = ")
        # print(self.red_loops)

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
        for key_indice in list(self.structured_topological_actions.keys()):
            res = self.structured_topological_actions[key_indice]
            if res is not None:
                for elem in res:
                    selected_ranked_nodes.append(elem)

        res_container = []
        for node in selected_ranked_nodes:
            if node in self.substation_in_cooldown:
                print("substation " + str(node) + " is in cooldown and no action can be performed on it for now")
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

    def legal_comb(self,comb,nProd_loads,n_elements,node_configuration,node_configuration_sym):
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

    def rank_topologies(self, all_combinations, graph, node_to_change: int):
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
                print("\n** RESULTS ** new topo [{}] on node [{}] has a score: [{}]\n".format(topo, node_to_change, score))
            scores_data.append([score, topo, node_to_change])

            # =======================================================
            # if i % 10000 ==0:
              #   print("Done: "+str(i)+" topos")

        ranked_combinations = pd.DataFrame(columns = ranked_combinations_columns, data = scores_data)

        # =================================================
        # ranked_combinations.to_csv("NEW_rank_topologies_l2rpn_2019_node_"+str(node_to_change)+".csv", sep = ';', decimal = ',')
        return ranked_combinations

    # WARNING: does not work yet when you go back from two nodes to one node at a given substation? Basically one node will be not connected?
    def apply_new_topo_to_graph(self, graph: nx.MultiDiGraph, new_topology, node_to_change: int):
        """given  a graph, a node_topoly and a node_id, this function applies the change to the graph
        :return new_graph, internal_repr_dict"""
        if self.debug:
            print("\n====================================== apply new topo to graph ======================================")
            print(" new topology applied = [{}] to node: [{}]".format(new_topology, node_to_change))
            print("======================================================================================================\n")

        # check if there are two nodes, there are 2 different values in new topo 0 and 1
        bus_ids = set(new_topology)
        # assert(len(bus_ids) == 2)#not necesarrily, it should be at least 1 and not more than 2
        assert ((len(bus_ids) != 0) & (len(bus_ids) <= 2))

        internal_repr_dict = dict(self.simulator_data["substations_elements"])
        new_node_id = int("666" + str(node_to_change))

        element_types = self.simulator_data["substations_elements"][node_to_change]
        # TO DO: you need to get all elements of the substation, especially if it is already in a 2 nodes topology and you want to merge the nodes
        # if (new_node_id in self.simulator_data["substations_elements"]):#the substation was already in a 2 node topology
        #    element_types+=self.simulator_data["substations_elements"][new_node_id]
        # it has to be the same, otherwise it does not make sense, ie, there is an error somewhere
        assert len(element_types) == len(new_topology)

        # BEFORE REMOVING, GET NEEDED INFORMATION ON EDGES: COLORS, WIDTH etc...
        color_edges = {}
        for u, v,idx, color in self.g.edges(data="color",keys=True):
            # invert edges that have been marked as SWAPPED in DATAFRAME.
            condition = list(self.df.query("idx_or == " + str(u) + " & idx_ex == " + str(v))["swapped"])[0]
            color_edges[(u, v,idx)] = color
            if condition:
                color_edges[(v, u,idx)] = color
            else:
                color_edges[(u, v,idx)] = color

        if 1 in new_topology:  # ie, if the topo is not [0, ... , 0]
            # first we delete the node_to_change ==> it deletes all edges for us
            graph.remove_node(node_to_change)

        # ################ PREPROCESSING NODE RECONSTRUCTION PART, IMPORTANT TO GET COLORS RIGHT
        i = 0
        current_node = []  # Busbar 0
        new_node = []  # Busbar 1
        # then, parsing element by element, reconnect the graph.
        for internal_elem, element, element_type in zip(internal_repr_dict[node_to_change], new_topology, element_types):
            internal_elem.busbar_id = element

        # WE RECONSTRUCT INTERNAL REPR
        # prod = {0: 0, 1: 0}  # busid:value
        # load = {0: 0, 1: 0}  # busid:value
        prod = {}
        load = {}
        for element in internal_repr_dict[node_to_change]:
            if element.busbar_id not in prod.keys():
                if isinstance(element, Production):
                    prod[element.busbar_id] = fabs(element.value)
            else:
                if isinstance(element, Production):
                    prod[element.busbar_id] += fabs(element.value)

            if element.busbar_id not in load.keys():
                if isinstance(element, Consumption):
                    load[element.busbar_id] = fabs(element.value)
            else:
                if isinstance(element, Consumption):
                    load[element.busbar_id] += fabs(element.value)

        node_type = {}
        for bus_id in bus_ids:  # [0, 1]
            node_label = node_to_change
            prod_minus_load = 0

            if bus_id in prod.keys() and bus_id in load.keys():
                prod_minus_load = prod[bus_id] - load[bus_id]
                if prod_minus_load > 0:
                    node_type[bus_id] = "prod"
                else:
                    node_type[bus_id] = "load"
            elif bus_id in prod.keys():
                node_type[bus_id] = "prod"
                prod_minus_load = prod[bus_id]
            elif bus_id in load.keys():
                node_type[bus_id] = "load"
                prod_minus_load = load[bus_id]

            if bus_id == 1:
                node_label = new_node_id
            if bus_id in node_type.keys():
                if node_type[bus_id] == "prod":  # PROD
                    # prod_minus_load = prod[bus_id]
                    graph.add_node(node_label, pin=True, prod_or_load="prod", value=str(prod_minus_load),
                                   style="filled", fillcolor="#f30000")
                elif node_type[bus_id] == "load":
                    # prod_minus_load = load[bus_id]
                    graph.add_node(node_label, pin=True, prod_or_load="load", value=str(-prod_minus_load),
                                   style="filled", fillcolor="#478fd0")
            else:  # WHITE
                graph.add_node(node_label, pin=True, prod_or_load="load", value=str(prod_minus_load),
                               style="filled", fillcolor="#ffffff")

        i = 0
        # then, parsing element by element, reconnect the graph.
        for element, element_type in zip(new_topology, element_types):
            # print("element = ", element)
            # print("element type = ", element_type)
            reported_flow = None
            edge_color = None
            penwidth = None
            if isinstance(element_type, OriginLine) or isinstance(element_type, ExtremityLine):
                # # print("element_type.flow_value=", element_type.flow_value)
                reported_flow = element_type.flow_value[0]
                # # print("reported_flow = ", reported_flow)
                penwidth = fabs(reported_flow) / 10
                if penwidth == 0.0:
                    penwidth = 0.1

            if element == 1:  # connect FROM or TO node: 666XX
                if isinstance(element_type, OriginLine):
                    graph.add_edge(new_node_id, element_type.end_substation_id,
                                   capacity=float("%.2f" % reported_flow), xlabel="%.2f" % reported_flow,
                                   color=color_edges[(node_to_change, element_type.end_substation_id,0)],
                                   fontsize=10, penwidth="%.2f" % penwidth)#we have a multiGraph, so we need to give a third index to color_edges, should be revised if reused

                elif isinstance(element_type, ExtremityLine):
                    graph.add_edge(element_type.start_substation_id, new_node_id,
                                   capacity=float("%.2f" % reported_flow), xlabel="%.2f" % reported_flow,
                                   color=color_edges[(element_type.start_substation_id, node_to_change,0)],
                                   fontsize=10, penwidth="%.2f" % penwidth)

            elif element == 0:  # connect from node node:_to_change
                if isinstance(element_type, OriginLine):
                    graph.add_edge(node_to_change, element_type.end_substation_id,
                                   capacity=float("%.2f" % reported_flow), xlabel="%.2f" % reported_flow,
                                   color=color_edges[(node_to_change, element_type.end_substation_id,0)],
                                   fontsize=10, penwidth="%.2f" % penwidth)

                elif isinstance(element_type, ExtremityLine):
                    graph.add_edge(element_type.start_substation_id, node_to_change,
                                   capacity=float("%.2f" % reported_flow), xlabel="%.2f" % reported_flow,
                                   color=color_edges[(element_type.start_substation_id, node_to_change,0)],
                                   fontsize=10, penwidth="%.2f" % penwidth)

            else:
                raise ValueError("Error element has to belong to either Busbar 1 or 2")

            i += 1

        name = "".join(str(e) for e in new_topology)
        name = str(node_to_change) + "_" + name
        self.bag_of_graphs[name] = graph
        return graph, internal_repr_dict

    def rank_current_topo_at_node_x(self, graph, node: int, isSingleNode=False, topo_vect=[0, 0, 1, 1, 1]):
        """This function ranks current topology at node X"""
        final_score = 0.0
        all_nodes_value_attributes = nx.get_node_attributes(graph, "value")  # dict[node]
        all_edges_color_attributes = nx.get_edge_attributes(graph, "color")  # dict[edge]
        all_edges_xlabel_attributes = nx.get_edge_attributes(graph, "xlabel")  # dict[edge]

        # ======================================
        # print('\nnoeud '+str(node)+' topo '+str(topo_vect))

        #  ########## IS IN AMONT ##########
        if node in self.constrained_path.n_amont():
            # ======================================
            #print("AMONT")
            if self.debug:
                print("||||||||||||||||||||||||||| node [{}] is in_Amont of constrained_edge".format(node))
            in_negative_flows = []
            in_positive_flows = []
            out_positive_flows = []

            interesting_bus_id = 0
            for edge in graph.out_edges(node,keys=True):
                if self.is_connected_to_cpath(all_edges_color_attributes, all_edges_xlabel_attributes, node, edge, isSingleNode):
                    # take the other bus id
                    interesting_bus_id = abs(self.get_bus_id_from_edge(node, edge, topo_vect) - 1)
                    break

            # somme des reports négatifs entrants + sommes des reports positifs entrants
            for edge in graph.in_edges(node,keys=True):
                edge_bus_id = self.get_bus_id_from_edge(node, edge, topo_vect)
                if edge_bus_id == interesting_bus_id: # MASK
                    edge_flow_value = float(all_edges_xlabel_attributes[edge])
                    if edge_flow_value < 0:
                        in_negative_flows.append(fabs(edge_flow_value))
                    else:
                        in_positive_flows.append(edge_flow_value)

            # somme des reports positifs sortant +
            for edge in graph.out_edges(node,keys=True):
                edge_bus_id = self.get_bus_id_from_edge(node, edge, topo_vect)
                if edge_bus_id == interesting_bus_id: # MASK
                    edge_flow_value = float(all_edges_xlabel_attributes[edge])
                    if edge_flow_value > 0:
                        out_positive_flows.append(edge_flow_value)

            diff_sums = self.get_prod_conso_sum(node, interesting_bus_id, topo_vect)
            max_pos_in_or_out_flows = max(sum(out_positive_flows), sum(in_positive_flows))
            final_score = np.around(sum(in_negative_flows) + max_pos_in_or_out_flows + diff_sums, decimals=2)
            if self.debug:
                print("AMONT")
                print("diff_sums = ", diff_sums)
                print(type(diff_sums))
                print("max_pos_in_or_out_flows = ", max_pos_in_or_out_flows)
                print("in negative flows = ", in_negative_flows)
                print("max_pos_in_or_out_flows = ", max_pos_in_or_out_flows)
                print("Final score = ", final_score)

        #  ########## IS IN AVAL ##########
        elif node in self.constrained_path.n_aval():
            # =============================================================
            #print("AVAL")
            if self.debug:
                print("||||||||||||||||||||||||||| node [{}] is in_Aval of constrained_edge".format(node))

            out_negative_flows = []
            out_positive_flows = []
            in_positive_flows = []

            interesting_bus_id = 0
            for edge in graph.in_edges(node,keys=True):
                if self.is_connected_to_cpath(all_edges_color_attributes, all_edges_xlabel_attributes, node, edge, isSingleNode):
                    # take the other bus id
                    interesting_bus_id = abs(self.get_bus_id_from_edge(node, edge, topo_vect) - 1)
                    break

            # somme des reports negatifs et positifs SORTANT
            for edge in graph.out_edges(node,keys=True):
                edge_bus_id = self.get_bus_id_from_edge(node, edge, topo_vect)
                if edge_bus_id == interesting_bus_id: # MASK
                    edge_flow_value = float(all_edges_xlabel_attributes[edge])
                    if edge_flow_value < 0:
                        out_negative_flows.append(fabs(edge_flow_value))
                    else:
                        out_positive_flows.append(edge_flow_value)

            for edge in graph.in_edges(node,keys=True):
                edge_bus_id = self.get_bus_id_from_edge(node, edge, topo_vect)
                if edge_bus_id == interesting_bus_id: # MASK
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
            # sur le noeud choisi (non connecté au cpath) on souhaite y connecter des consommations et pas des
            # productions pour l'aval.
            diff_sums = -self.get_prod_conso_sum(node, interesting_bus_id, topo_vect)
            final_score = np.around(sum(out_negative_flows) + max_pos_in_or_out_flows + diff_sums, decimals=2)

            if self.debug:
                print("AVAL")
                print("diff_sums = ", diff_sums)
                print("Final score = ", final_score)
                print(type(final_score))

        #  ########## IS IN Loop ##########
        # you want a node with the maximum output lines connected to the ingoing red loop edges, not connected to other ingoing edges
        elif node in set([x for loop in range(len(self.red_loops.Path)) for x in self.red_loops.Path[loop]]):
            # ========================================================================
            # print("AUTRE")
            node2 = int("666" + str(node))

            if 1 in topo_vect and 0 in topo_vect:  # need to be a 2 node topology
                # we find the node with the biggest red ingoing delta flow
                InputRedDeltaFlow_1 = 0
                InputRedDeltaFlow_2 = 0


                for edge in graph.in_edges(node,keys=True):
                    edge_bus_id = self.get_bus_id_from_edge(node, edge, topo_vect)
                    edge_color = all_edges_color_attributes[edge]
                    edge_value = float(all_edges_xlabel_attributes[edge])
                    if (edge_color == "red"):
                        if edge_bus_id == 0: # MASK
                            InputRedDeltaFlow_1 += edge_value
                        elif edge_bus_id == 1: # MASK
                            InputRedDeltaFlow_2 += edge_value

                Bus_BiggestInputDeltaFlow = 0
                InputRedDeltaFlow = InputRedDeltaFlow_1
                if (InputRedDeltaFlow_2 >= InputRedDeltaFlow_1):
                    Bus_BiggestInputDeltaFlow = 1
                    InputRedDeltaFlow = InputRedDeltaFlow_2
                ###
                # over node with BiggestInputDeltaFlow, we want as much ingoing and outgoing red flow possible, with least possible production
                OutputRedDeltaFlow = 0
                for edge in graph.out_edges(node,keys=True):
                    edge_bus_id = self.get_bus_id_from_edge(node, edge, topo_vect)
                    if edge_bus_id == Bus_BiggestInputDeltaFlow:
                        edge_color = all_edges_color_attributes[edge]
                        edge_value = float(all_edges_xlabel_attributes[edge])
                        if (edge_color == "red"):
                            OutputRedDeltaFlow += edge_value

                min_pos_in_or_out_flows = min(OutputRedDeltaFlow, InputRedDeltaFlow)
                injection = -self.get_prod_conso_sum(node, Bus_BiggestInputDeltaFlow, topo_vect)
                final_score = np.around(min_pos_in_or_out_flows + injection, decimals=2)
        else:
            print("||||||||||||||||||||||||||| node [{}] is not connected to a path to the constrained_edge.".format(node))

        #=====================================================================
        # print("SCORE   ---  "+str(final_score))
        # print('\n')
        return final_score

    def is_in_aval(self, graph, node):  # in Aval of constrained_edge
        """ This functions check if node is in Aval of constrained_edge"""
        g = self.g
        aval_constrained_node = self.constrained_path.constrained_edge[1]
        if node == aval_constrained_node:
            return True
        if node in list(g.successors(aval_constrained_node)):
            return True
        else:
            return False

    def get_prod_conso_sum(self, node, interesting_bus_id, topo_vect):
        total = 0
        elements = self.simulator_data["substations_elements"][node]
        for element, bus_id in zip(elements, topo_vect):
            if bus_id == interesting_bus_id:
                if isinstance(element, Consumption):
                    total = total - element.value
                elif isinstance(element, Production):
                    total = total + element.value
        return total

    def get_bus_id_from_edge(self, node, edge, topo_vect):
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
        for element, bus_id in zip(elements, topo_vect):
            if isinstance(element, OriginLine):
                if element.end_substation_id == target_extremity:
                    return bus_id
            elif isinstance(element, ExtremityLine):
                if element.start_substation_id == target_extremity:
                    return bus_id

    def is_connected_to_cpath(self, all_edges_color_attributes, all_edges_xlabel_attributes, node, edge, isSingleNode):
        edge_color = all_edges_color_attributes[edge]
        edge_value = all_edges_xlabel_attributes[edge]
        # if there is a outgoing negative blue or black edge this means we are connected to cpath.
        # therefore change to bus ID 1
        bool = (float(edge_value) < 0 and (edge_color == "blue" or edge_color == "black") and not isSingleNode)
        if bool and self.debug:
            print("\n######################################################")
            print("Node [{}] is not connected to cpath. Twin node selected...".format(node))
            print("######################################################")
        return bool

    def is_in_amont(self, graph, node):  # in Amont of constrained_edge
        # g = self.g
        g = graph
        amont_constrained_node = self.constrained_path.constrained_edge[0]
        # print("constrained path = ", self.constrained_path)
        if node == amont_constrained_node:
            return True
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
        # print("successors = ", successors)
        # print("nodes successors = ", nodes_succ)
        if node in nodes_succ:
            return True
        else:
            return False

    @staticmethod
    def is_in_amont_of_node_x(g, node, node_x):
        nodes_pred = set()
        predecessors = list(nx.edge_dfs(g, node_x, orientation="reverse"))
        for p in predecessors:
            for t in p:  # for tuple in predcessors
                if isinstance(t, int):
                    nodes_pred.add(t)

        # print("predecessors = ", predecessors)
        # print("nodes predcessors = ", nodes_pred)
        if node in nodes_pred:
            return True
        else:
            return False

    def sort_hubs(self, hubs):
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
        df_sorted_hubs = self.sort_hubs(self.hubs)
        if df_sorted_hubs is None:
            return {}
        else:
            category1 = list(df_sorted_hubs["hubs"])
            set_category2 = set(self.constrained_path.full_n_constrained_path()) - set(category1)

            sort_redLoopBuses = sorted(self.rankedLoopBuses.items(), key=lambda x: x[1], reverse=True)
            category3 = [sort_redLoopBuses[i][0] for i in range(len(sort_redLoopBuses))]  # set()  # @TODO
            set_category4 = set(self.constrained_path.n_aval()) - (set(category1) | set_category2 | set(category3))

            d = {1: category1, 2: set_category2, 3: category3, 4: set_category4}
        return d

    def rank_loop_buses(self, graph, df_initial_flows):
        # self.g => overflow graph
        all_nodes_value_attributes = nx.get_node_attributes(graph, "value")
        all_edges_color_attributes = nx.get_edge_attributes(graph, "color")  # dict[edge]
        all_edges_xlabel_attributes = nx.get_edge_attributes(graph, "xlabel")

        Strength_Bus_dic = {}
        for index, loop in self.red_loops.iterrows():
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
                            if (edge_color == "red"):
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
                                        indexEdge_inDf = i
                                        sumInFlowsNotRed += np.abs(flowValue)
                                        break
                                    elif ((flowValue <= 0) & (nodes_or[i] == bus) & (nodes_ex[i] == otherBus)):
                                        indexEdge_inDf = i
                                        sumInFlowsNotRed += np.abs(flowValue)
                                        break

                        sumInFlowsNotRed += LocalProduction
                        strength_measure = sumInFlowsNotRed * sumInRedDeltaFlows
                        Strength_Bus_dic[bus] = strength_measure
        return Strength_Bus_dic

    def rank_red_loops(self):
        cut_values = []
        cut_sets = []  # contains the edges that ended up having the minimum cut_values
        g_red_DiGraph=self.to_DiGraph(self.g_only_red_components)#necessary to be able to compute minimum_cut
        for i, row in self.red_loops.iterrows():
            source = row["Source"]
            target = row["Target"]
            p = row["Path"]

            # print("=============== source: {}, target: {}".format(source, target))

            #cut_value, partition = nx.minimum_cut(self.g_only_red_components, source, target)
            cut_value, partition = nx.minimum_cut(g_red_DiGraph, source, target)
            reachable, non_reachable = partition
            # print("cut_value: {}, partition: {}".format(cut_value, partition))

            # info from doc - ‘partition’ here is a tuple with the two sets of nodes that define the minimum cut.
            # You can compute the cut set of edges that induce the minimum cut as follows:
            cutset = set()
            for u, nbrs in ((n, self.g_only_red_components[n]) for n in reachable):
                cutset.update((u, v) for v in nbrs if v in non_reachable)
            # print("sorted(cutset) = ", sorted(cutset))

            cut_values.append(cut_value)
            cut_sets.append(list(cutset)[0])

        # print("cut_values = ", cut_values)

        self.red_loops["min_cut_values"] = cut_values
        self.red_loops["min_cut_edges"] = cut_sets
        # print("======================= cut_values added =======================")
        # print(self.red_loops)

    # create weighted digraph from MultiDiGraph
    def to_DiGraph(self,gM):
        G = nx.DiGraph()
        for u, v,idx, data in gM.edges(data=True,keys=True):
            w = data['capacity'] if 'capacity' in data else 1.0
            if G.has_edge(u, v):
                G[u][v]['capacity'] += w
            else:
                G.add_edge(u, v, capacity=w)
        return G

    def joke(self):
        print("Heard about the new restaurant called Karma ?...")
        print("....")
        print("There's no menu:")
        print("You get what you deserve.")

    def get_amont_blue_edges(self, g, node):
        res = []
        for e in nx.edge_dfs(g, node, orientation="reverse"):
            if g.edges[(e[0], e[1],e[2])]["color"] == "blue":
                res.append((e[0], e[1],e[2]))
        return res

    def get_aval_blue_edges(self, g, node):
        res = []
        # print("debug AlphaDeesp get aval blue edges")
        # print(list(nx.edge_dfs(g, node, orientation="original")))
        for e in nx.edge_dfs(g, node, orientation="original"):
            if g.edges[(e[0], e[1],e[2])]["color"] == "blue":
                res.append((e[0], e[1],e[2]))
        return res

    def delete_positive_edges(self, _g):
        """Returns a copy of g without positive edges"""
        g = _g.copy()
        # array containing the indices of edges with positive report flow
        pos_edges = []

        # get indices of positive edges
        i = 1
        for u, v, report in g.edges(data="xlabel",keys=True):
            if float(report) > 0:
                pos_edges.append((i, (u, v)))
            i += 1

        # delete from graph positive edges
        # this extracts the (u,v) from pos_edges
        # # print("pos_edges test = ", list(zip(*pos_edges))[1])
        if pos_edges:
            g.remove_edges_from(list(zip(*pos_edges))[1])
        return g

    def delete_color_edges(self, _g, edge_color):
        """Returns a copy of g without gray edges"""
        g = _g.copy()

        gray_edges = []
        i = 1
        for u, v,idx, color in g.edges(data="color",keys=True):
            if color == edge_color:
                gray_edges.append((i, (u, v,idx)))
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
        edge_list = nx.get_edge_attributes(self.g_only_blue_components, "color")
        for edge, color in edge_list.items():
            if color == "black":
                constrained_edge = edge
        amont_edges = self.get_amont_blue_edges(self.g_only_blue_components, constrained_edge[0])
        aval_edges = self.get_aval_blue_edges(self.g_only_blue_components, constrained_edge[1])
        tmp_constrained_path.append(amont_edges)
        tmp_constrained_path.append(constrained_edge)
        tmp_constrained_path.append(aval_edges)
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
            in_edges = list(g.in_edges(node,keys=True))
            for e in in_edges:
                if g.edges[e]["color"] == "red":
                    hubs.append(node)
                    break

        # for nodes in amont, if node has RED outputs (ie outgoing flows) then it is a hub
        for node in self.constrained_path.n_amont():
            out_edges = list(g.out_edges(node,keys=True))
            for e in out_edges:
                if g.edges[e]["color"] == "red":
                    hubs.append(node)
                    break

        # print("get_hubs = ", hubs)
        return hubs

    def get_loops(self):
        """This function returns all parallel paths. After discussing with Antoine, start with the most "en Aval" node,
        and walk in reverse for loops and parallel path returns a dict with all data """

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
                    try:
                        res = nx.all_shortest_paths(g, c_path_n[i], c_path_n[j])
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

    def get_color_path_from_node(self, g, node, _color: str, orientation="original"):
        """This function returns an array with "_color" edges predecessors and successors (depending on orientation)"""
        """orientation has to be : "original" or "reverse" """
        res = {0: []}
        i = 1
        for e in nx.edge_bfs(g, node, orientation=orientation):
            if g.edges[(e[0], e[1],e[2])]["color"] == _color:
                # print(e)
                if not res[0]:  # if empty
                    res[0].append((e[0], e[1],e[2]))

                # if pred
                elif res[0][-1][0] == e[1]:
                    res[0].append((e[0], e[1],e[2]))

                # elif e[0] in res.values():

                else:
                    # we check in res, or new branch
                    found = False
                    for p in list(res.keys()):
                        # print("we check res[p] = ", res[p])

                        path = list(res[p])
                        if e[1] == path[-1][0]:
                            res[p].append((e[0], e[1]))
                            # print("we append e = ", e)
                            found = True
                            break

                    if not found:
                        # create new list
                        res[i] = [(e[0], e[1])]
                        # print("we create new list for e = ", e)
                        i += 1

        i += 1
        # for p in list(res.keys()):
        #     res[p] = list(reversed(res[p]))

        return res

    def isAntenna(self):
        pass

    def write_g(self, g):
        """This saves file g"""
        nx.write_edgelist(self.g, "./alphaDeesp/tmp/save.graph")
        # print("file saved")
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

    # # print("command = ", command)
    sub_p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = sub_p.communicate()
    exit_code = sub_p.returncode
    # pid = sub_p.pid

    output = stdout.decode()
    error = stderr.decode()

    # print("--------------------\n output is:", output)
    # print("--------------------\n stderr is:", error)
    # print("--------------------\n exit code is:", exit_code)
    # # print("--------------------\n pid is:", pid)

    if not error:
        # string error is empty
        return True
    else:
        # string error is full
        # # print(f"Error {error}")
        return False
