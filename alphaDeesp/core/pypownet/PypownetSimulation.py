from math import fabs
import ast

import networkx as nx
import pypownet.environment
from pypownet.agent import *
from pypownet.environment import ElementType

from alphaDeesp.core.elements import *
from alphaDeesp.core.network import Network
from alphaDeesp.core.simulation import Simulation
from alphaDeesp.core.printer import Printer


class PypownetSimulation(Simulation):
    def __init__(self, env, obs, action_space, param_options=None, debug=False, ltc=[9], plot_folder = None,isScoreFromBackend=False):
        super().__init__()
        print("PypownetSimulation object created...")

        if not param_options or param_options is None:
            raise AttributeError("\nparam_options are empty or None, meaning the config file is not properly read.")

        self.save_bag = []
        self.debug = debug
        self.args_number_of_simulated_topos = param_options["totalnumberofsimulatedtopos"]
        self.args_inner_number_of_simulated_topos_per_node = param_options["numberofsimulatedtopospernode"]
        self.grid = None
        self.df = None
        self.topo = None  # a dict create in retrieve topology
        self.ltc = ltc
        self.param_options = param_options
        self.printer = Printer(plot_folder)
        #############################
        self.environment = env
        self.action_space = action_space
        # Run one step in the environment
        self.obs = obs
        self.obs_linecut = None
        self.isScoreFromBackend=isScoreFromBackend

        # Layout of the grid
        self.layout = self.compute_layout()

        print("HARD OVERFLOW = ", self.environment.game.hard_overflow_coefficient)
        print("")

        observation_space = self.environment.observation_space


        #############################
        # new structures to omit querying Pypownet, they are filled in LOAD function.
        # for each substation, we get an array with (Prod, Cons, Line) Objects, representing the actual configuration
        self.substations_elements = {}
        self.substation_to_node_mapping = {}
        self.internal_to_external_mapping = {}  # d[internal_id] = external_name_id
        self.external_to_internal_mapping = {}  # d[external_id] = internal_name_id
        print("current chronic name = ", self.environment.game.get_current_chronic_name())
        print(self.obs)
        self.load()

    def compute_layout(self):
        try:
            layout = self.param_options['CustomLayout']
            # Conversion from string to list
            layout = ast.literal_eval(layout)
        except:
            layout = [(-280, -81), (-100, -270), (366, -270), (366, -54), (-64, -54), (-64, 54), (366, 0),
                    (438, 0), (326, 54), (222, 108), (79, 162), (-152, 270), (-64, 270), (222, 216),
                    (-280, -151), (-100, -340), (366, -340), (390, -110), (-14, -104), (-184, 54), (400, -80),
                    (438, 100), (326, 140), (200, 8), (79, 12), (-152, 170), (-70, 200), (222, 200)]
        return layout

    def get_layout(self):
        return self.layout

    def get_substation_elements(self):
        return self.substations_elements

    def get_substation_to_node_mapping(self):
        return self.substation_to_node_mapping

    def get_internal_to_external_mapping(self):
        return self.internal_to_external_mapping

    def get_dataframe(self):
        """
        :return: pandas dataframe with topology information before and after line cutting
        """
        return self.df

    def plot_grid_beforecut(self):
        """
        Plots the grid with alphadeesp.printer API for Observation before lines have been cut
        :return: Figure
        """
        g_pow = self.build_powerflow_graph_beforecut()
        return self.plot_grid(g_pow, name = "g_pow")

    def plot_grid_aftercut(self):
        """
        Plots the grid with alphadeesp.printer API for Observation after lines have been cut
        :return: Figure
        """
        g_pow_prime = self.build_powerflow_graph_aftercut()
        return self.plot_grid(g_pow_prime, name = "g_pow_prime")

    def plot_grid_delta(self):
        """
        Plots the grid with alphadeesp.printer API for delta between Observations before and after lines have been cut
        :return: Figure
        """
        g_over = self.build_graph_from_data_frame(self.ltc)
        return self.plot_grid(g_over, name="g_overflow_print")

    def plot_grid_from_obs(self, obs, name):
        """
        Plots the grid with alphadeesp.printer API from given observation
        :return: Figure
        """
        # Pypownet needs to rebuild its internal structure to produce objects to plot
        self.load_from_observation(obs, self.ltc)
        g_over_detailed = self.build_detailed_graph_from_internal_structure(self.ltc)
        return self.plot_grid(g_over_detailed, name=name)

    def plot_grid(self, g, name):
        # Use printer API to plot (graphviz/neato)
        self.printer.display_geo(g, self.get_layout(), name=name)

    def isAntenna(self):
        """TODO"""
        return None

    def isDoubleLine(self):
        """TODO"""
        return None

    def getLinesAtSubAndBusbar(self):
        """TODO"""
        return None

    def get_overload_disconnection_topovec_subor(self, l):
        """TODO"""
        return None,None

    def get_reference_topovec_sub(self,sub):
        """TODO"""
        return None

    def get_substation_in_cooldown(self):
        """TODO"""
        return None

    def compute_new_network_changes(self, ranked_combinations):
        """this function takes a dataframe ranked_combinations,
         for each combination it computes a simulation step in Pypownet with action:
         change nodes topo(combination)"""

        print("\n##############################################################################")
        print("##########...........COMPUTE NEW NETWORK CHANGES..........####################")
        print("##############################################################################")

        # the function score creates a Dataframe with sorted score for each topo change.
        # FINISHED
        end_result_dataframe = self.create_end_result_empty_dataframe()
        j = 0
        for df in ranked_combinations:
            ii = 0
            if j == int(self.args_number_of_simulated_topos):
                break
            for i, row in df.iterrows():
                if ii == int(self.args_inner_number_of_simulated_topos_per_node):
                    break
                saved_obs = self.obs
                # target_node = row["node"] + 1
                internal_target_node = row["node"]
                target_node = self.internal_to_external_mapping[row["node"]]
                # target_node = row["node"]
                new_conf = row["topology"]
                print("ROW KEYS = ", row.keys())
                score_topo = i
                print("#####################################################################################")
                print("###########"" Compute new network changes on node [{}] with new topo [{}] ###########"
                      .format(target_node, new_conf))
                print("#####################################################################################")
                current_conf, types = self.obs.get_nodes_of_substation(target_node)

                # target configuration represents the action to be taken to get from curr_conf to ==> new_conf.
                target_configuration = get_differencial_topology(new_conf, current_conf)

                # load this topo to pypownet
                action_space = self.environment.action_space
                observation_space = self.environment.observation_space
                # Create template of action with no switch activated (do-nothing action)
                action = action_space.get_do_nothing_action(as_class_Action=True)
                # This function fills the "action" with correct values.
                action_space.set_substation_switches_in_action(action=action, substation_id=target_node,
                                                               new_values=target_configuration)

                # action_space.set_lines_status_switch_from_id(action=action, line_id=[9], new_switch_value=1)
                # Run one step in the environment
                raw_obs,action, reward, *_ = self.environment.simulate(action)

                # if obs is None, error in the simulation of the next step
                if raw_obs is None:
                    print("Pypownet simulation returnt a None... Cannot process results...")
                    continue

                # transform raw_obs into Observation object. (Useful for prints, for debugging)
                obs = observation_space.array_to_observation(raw_obs)
                obs.get_lines_capacity_usage()
                if self.debug:
                    print(obs)

                print("old obs addr = ", hex(id(saved_obs)))
                print("new obs addr = ", hex(id(obs)))

                # this is used to display graphs at the end. Check main.
                name = "".join(str(e) for e in new_conf)
                name = str(internal_target_node) + "_" + name

                # for flow, origin, ext, in zip(obs.active_flows_origin, obs.lines_or_nodes, obs.lines_ex_nodes)

                self.save_bag.append([name, obs])

                delta_flow = saved_obs.active_flows_origin[self.ltc[0]] - \
                             obs.active_flows_origin[self.ltc[0]]

                print("self.lines to cut[0] = ", self.ltc[0])
                print("self.obs.status line = ", self.obs.lines_status)
                print(" simulated obs  = ", obs.lines_status)
                print("saved flow = ", list(saved_obs.active_flows_origin.astype(int)))
                print("current flow = ", list(obs.active_flows_origin.astype(int)))
                print("deltaflows = ", (saved_obs.active_flows_origin - obs.active_flows_origin).astype(int))
                print("final delta_flow = ", delta_flow)

                # 1) having the new and old obs. Now how do we compare ?
                simulated_score, worsened_line_ids, redistribution_prod, redistribution_load, efficacity = \
                    self.observations_comparator(saved_obs, obs, score_topo, delta_flow)

                if (self.isScoreFromBackend) and (simulated_score==4):
                    # dans le cas ou on resoud bien les contraintes, on prend la reward L2RPN
                    efficacity = reward

                # The next three lines have to been done like this to properly have a python empty list if no lines
                # are worsened. This is important to save, read back, and compare a DATAFRAME
                worsened_line_ids = list(np.where(worsened_line_ids == 1))
                worsened_line_ids = worsened_line_ids[0]

                # further tricks to properly read back a saved dataframe
                if worsened_line_ids.size == 0:
                    worsened_line_ids = []
                elif isinstance(worsened_line_ids, np.ndarray):
                    worsened_line_ids = list(worsened_line_ids)


                score_data = [self.ltc[0],
                              saved_obs.active_flows_origin[self.ltc[0]],
                              obs.active_flows_origin[self.ltc[0]],
                              delta_flow,
                              worsened_line_ids,
                              redistribution_prod,
                              redistribution_load,
                              new_conf,
                              new_conf,#the pypownet conf is the same as alphadeesp
                              self.external_to_internal_mapping[target_node],
                              1,  # category hubs?
                              score_topo,
                              simulated_score,
                              efficacity]

                max_index = end_result_dataframe.shape[0]  # rows
                end_result_dataframe.loc[max_index] = score_data

                end_result_dataframe.to_csv("./END_RESULT_DATAFRAME.csv", index=True)
                ii += 1
                j += 1

        return end_result_dataframe

    def observations_comparator(self, old_obs, new_obs, score_topo, delta_flow):
        """This function takes two observations and extracts several information:
        - the flow reports in %
        - list of lines, on which the situation got worse ie, the line capacity diminished.
            easily accessible with => obs.get_lines_capacity_usage()
        - prod volume diff
        - cons volume diff
        - score_simule => categories => 0, 1, 2, 3, 4
        - Efficacity =>

        it returns an array with all data for end_result_dataframe creation
        """
        simulated_score = self.score_changes_between_two_observations(old_obs, new_obs,self.environment.game.n_timesteps_actionned_line_reactionable)

        worsened_line_ids = self.create_boolean_array_of_worsened_line_ids(old_obs, new_obs)

        redistribution_prod = np.sum(np.absolute(new_obs.active_productions - old_obs.active_productions))
        redistribution_load = np.sum(np.absolute(new_obs.active_loads - old_obs.active_loads))

        if simulated_score in [4, 3, 2]:  # success
            efficacity = fabs(delta_flow / new_obs.get_lines_capacity_usage()[self.ltc[0]])
            pass
        elif simulated_score in [0, 1]:  # failure
            efficacity = -fabs(delta_flow / new_obs.get_lines_capacity_usage()[self.ltc[0]])
            pass
        else:
            raise ValueError("Cannot compute efficacity, the score is wrong.")

        return simulated_score, worsened_line_ids, redistribution_prod, redistribution_load, efficacity

    def create_boolean_array_of_worsened_line_ids(self, old_obs, new_obs):
        """This function creates a boolean array of lines that got worse between two observations.
        @:return boolean numpy array [0..1]"""

        res = []
        n_lines=len(new_obs.get_lines_capacity_usage())
        n_timesteps_actionned_line_reactionable=self.environment.game.n_timesteps_actionned_line_reactionable

        old_rho=old_obs.get_lines_capacity_usage()
        new_rho=new_obs.get_lines_capacity_usage()

        old_time_reco=old_obs.timesteps_before_lines_reconnectable
        new_time_reco=new_obs.timesteps_before_lines_reconnectable
        #for old, new in zip(old_obs, new_obs):
        for l in range(n_lines):
            if fabs(new_rho[l]) > 1 and fabs(old_rho[l]) > 1 and fabs(new_rho[l]) > 1.05 * fabs(old_rho[l]):  # contrainte existante empiree
                res.append(1)
            elif fabs(new_rho[l]) > 1 > fabs(old_rho[l]):
                res.append(1)
            elif (new_time_reco[l] - old_time_reco[l]>n_timesteps_actionned_line_reactionable):
                res.append(1)
            else:
                res.append(0)

        return np.array(res)

    def score_changes_between_two_observations(self, old_obs, new_obs,nb_timestep_cooldown_line_param=0):
        """This function takes two observations and computes a score to quantify the change between old_obs and new_obs.
        @:return int between [0 and 4]
        4: if every overload disappeared
        3: if an overload disappeared without stressing the network
        2: if at least 30% of an overload was relieved
        1: if an overload was relieved but another appeared and got worse
        0: if no overloads were alleviated or if it resulted in some load shedding or production distribution.
        """
        old_number_of_overloads = 0
        new_number_of_overloads = 0
        boolean_overload_worsened = []
        boolean_constraint_30percent_relieved = []
        boolean_constraint_relieved = []
        boolean_overload_created = []
        boolean_line_cascading_disconnection = ((new_obs.timesteps_before_lines_reconnectable - old_obs.timesteps_before_lines_reconnectable) > nb_timestep_cooldown_line_param)

        old_obs_lines_capacity_usage = old_obs.get_lines_capacity_usage()
        new_obs_lines_capacity_usage = new_obs.get_lines_capacity_usage()
        # ################################### PREPROCESSING #####################################
        for elem in old_obs_lines_capacity_usage:
            if elem > 1.0:
                old_number_of_overloads += 1

        for elem in new_obs_lines_capacity_usage:
            if elem > 1.0:
                new_number_of_overloads += 1

        # preprocessing for score 3 and 2
        line_id = 0
        for old, new in zip(old_obs_lines_capacity_usage, new_obs_lines_capacity_usage):
            # preprocessing for score 3
            if (new > 1.05 * old) & (new > 1.0):  # if new > old > 1.0 it means it worsened an existing constraint
                boolean_overload_worsened.append(1)
            else:
                boolean_overload_worsened.append(0)

            # preprocessing for score 2
            if (old > 1.0) & (line_id in self.ltc):  # if old was an overload:
                surcharge = old - 1.0
                diff = old - new
                percentage_relieved = diff * 100 / surcharge
                if percentage_relieved > 30.0:
                    boolean_constraint_30percent_relieved.append(1)
                else:
                    boolean_constraint_30percent_relieved.append(0)
            else:
                boolean_constraint_30percent_relieved.append(0)

            # preprocessing for score 1
            if (old > 1.0 > new) & (line_id in self.ltc):
                boolean_constraint_relieved.append(1)
            else:
                boolean_constraint_relieved.append(0)

            if old < 1.0 < new:
                boolean_overload_created.append(1)
            else:
                boolean_overload_created.append(0)

            line_id += 1

        boolean_overload_worsened = np.array(boolean_overload_worsened)
        boolean_constraint_30percent_relieved = np.array(boolean_constraint_30percent_relieved)
        boolean_constraint_relieved = np.array(boolean_constraint_relieved)
        boolean_overload_created = np.array(boolean_overload_created)

        redistribution_prod = np.sum(np.absolute(new_obs.active_productions - old_obs.active_productions))

        cut_load_percent = np.sum(np.absolute(new_obs.active_loads - old_obs.active_loads))/np.sum(old_obs.active_loads)

        # ################################ END OF PREPROCESSING #################################
        # score 0 if no overloads were alleviated or if it resulted in some load shedding or production distribution.
        if old_number_of_overloads == 0:
            # print("return NaN: No overflow at initial state of grid")
            return float('nan')
        elif cut_load_percent > 0.01:  # (boolean_overload_relieved == 0).all()
            # print("return 0: no overloads were alleviated or some load shedding occured.")
            return 0

        # score 1 if overload was relieved but another one appeared and got worse
        elif (boolean_constraint_relieved == 1).any() and ((boolean_overload_created == 1).any() or
                                                           (boolean_overload_worsened == 1).any() or (boolean_line_cascading_disconnection).any() ):
            # print("return 1: our overload was relieved but another one appeared")
            return 1

        # 4: if every overload disappeared
        elif old_number_of_overloads > 0 and new_number_of_overloads == 0:
            # print("return 4: every overload disappeared")
            return 4

        # 3: if this overload disappeared without stressing the network, ie,
        # if an overload disappeared
        # and without worsening existing constraint
        # and no Loads that get cut
        elif (boolean_constraint_relieved == 1).any() and \
                (boolean_overload_worsened == 0).all():
            # and \ (new_obs.are_loads_cut == 0).all():
            # print("return 3: our overload disappeared without stressing the network")
            return 3

        # 2: if at least 30% of this overload was relieved
        elif (boolean_constraint_30percent_relieved == 1).any() and \
                (boolean_overload_worsened == 0).all():
            # print("return 2: at least 30% of our overload [{}] was relieved".format(
            #     np.where(boolean_overload_30percent_relieved == 1)[0]))
            return 2

        # score 0
        elif (boolean_constraint_30percent_relieved == 0).all() or \
                (boolean_overload_worsened == 1).any():
            return 0

        else:
            raise ValueError("Probleme with Scoring")

    def create_and_fill_internal_structures(self, obs, df):
        """This function fills multiple structures:
        self.substation_elements, self.substation_to_node_mapping, self.internal_to_external_mapping
        @:arg observation, df"""
        # ################ PART I : fill self.internal_to_external_mapping
        substations_list = list(obs.substations_ids.astype(int))
        # we create mapping from external representation to internal.
        for i, substation_id in enumerate(substations_list):
            self.internal_to_external_mapping[i] = substation_id

        if self.internal_to_external_mapping:
            self.external_to_internal_mapping = self.invert_dict_keys_values(self.internal_to_external_mapping)

        # prod values
        prod_nodes = obs.productions_substations_ids
        cons_nodes = obs.loads_substations_ids
        prod_values = obs.active_productions
        cons_values = obs.active_loads

        # ################ PART II : fill self.substation_elements
        for substation_id in self.internal_to_external_mapping.keys():
            elements_array = []
            external_substation_id = self.internal_to_external_mapping[substation_id]
            current_conf, types = obs.get_nodes_of_substation(external_substation_id)

            if self.debug:
                print("--------------- substation_id {}", substation_id)
                print(current_conf)
                print(types)
                print(types[0])

            # here we create arrays of substations_ids indicating the destination for OriginLine
            indexes_tmp_or = np.where(obs.lines_or_substations_ids == external_substation_id)
            indexes_dest_or = [obs.lines_ex_substations_ids[ind] for ind in indexes_tmp_or]
            # this iterator contains a LIST OF SUBSTATION_IDS, which correspond to "the other end" of element line OR
            iter_indexes_dest_or = iter(indexes_dest_or[0])

            # here we create arrays of substations_ids indicating the destination for ExtremityLine
            indexes_tmp_ex = np.where(obs.lines_ex_substations_ids == external_substation_id)
            indexes_dest_ex = [obs.lines_or_substations_ids[ind] for ind in indexes_tmp_ex]
            # this iterator contains a LIST OF SUBSTATION_IDS, which correspond to "the other end" of element line EX
            iter_indexes_dest_ex = iter(indexes_dest_ex[0])

            assert (len(current_conf) == len(types))

            for busbar, elem in zip(current_conf, types):
                if elem == ElementType.PRODUCTION:
                    value = None
                    if external_substation_id in prod_nodes:
                        res_index = np.where(prod_nodes == external_substation_id)
                        value = prod_values[res_index]
                    elements_array.append(Production(busbar, value))

                elif elem == ElementType.CONSUMPTION:
                    value = None
                    if external_substation_id in cons_nodes:
                        res_index = np.where(cons_nodes == external_substation_id)
                        value = cons_values[res_index]
                    elements_array.append(Consumption(busbar, value))

                elif elem == ElementType.ORIGIN_POWER_LINE:
                    dest = self.external_to_internal_mapping[int(next(iter_indexes_dest_or))]
                    flow_value = list(df.query("idx_or == " + str(substation_id) + " & idx_ex == " + str(dest))
                                      ["delta_flows"].round(decimals=2))
                    if flow_value:  # if not empty
                        elements_array.append(OriginLine(busbar, dest, flow_value))
                    else:  # else means the flow has been swapped. We must invert edge.
                        flow_value = list(df.query("idx_ex == " + str(substation_id) + " & idx_or == " + str(dest))
                                          ["delta_flows"].round(decimals=2))
                        swapped_condition = \
                            list(df.query("idx_ex == " + str(substation_id) + " & idx_or == " + str(dest))
                                 ["swapped"])[0]
                        # second swapped_condition for new_flows_swapped in self.topo
                        second_condition = \
                            list(df.query("idx_ex == " + str(substation_id) + " & idx_or == " + str(dest))
                                 ["new_flows_swapped"])[0]

                        # if both are true, two swaps = do nothing or both are false and we do nothing.
                        if (swapped_condition and second_condition) or (not swapped_condition and not second_condition):
                            elements_array.append(OriginLine(busbar, dest, flow_value))

                        # if one condition is true
                        elif swapped_condition or second_condition:
                            elements_array.append(ExtremityLine(busbar, dest, flow_value))

                        else:
                            raise ValueError("Problem with swap conditions")

                elif elem == ElementType.EXTREMITY_POWER_LINE:
                    dest = self.external_to_internal_mapping[int(next(iter_indexes_dest_ex))]
                    flow_value = list(df.query("idx_or == " + str(dest) + " & idx_ex == " + str(substation_id))
                                      ["delta_flows"].round(decimals=2))

                    if flow_value:  # if not empty
                        elements_array.append(ExtremityLine(busbar, dest, flow_value))
                    else:
                        flow_value = list(df.query("idx_ex == " + str(dest) + " & idx_or == " + str(substation_id))
                                          ["delta_flows"].round(decimals=2))

                        swapped_condition = \
                            list(df.query("idx_ex == " + str(dest) + " & idx_or == " + str(substation_id))
                                 ["swapped"])[0]
                        # second swapped_condition for new_flows_swapped in self.topo
                        second_condition = \
                            list(df.query("idx_ex == " + str(dest) + " & idx_or == " + str(substation_id))
                                 ["new_flows_swapped"])[0]

                        # if both are true, two swaps = do nothing or both are false and we do nothing.
                        if (swapped_condition and second_condition) or (not swapped_condition and not second_condition):
                            elements_array.append(ExtremityLine(busbar, dest, flow_value))
                        # if one condition is true
                        elif swapped_condition or second_condition:
                            elements_array.append(OriginLine(busbar, dest, flow_value))
                        else:
                            raise ValueError("Problem with swap conditions")

            self.substations_elements[substation_id] = elements_array

    def load(self):
        self.load_from_observation(self.obs, self.ltc)

    def load_from_observation(self, observation, lines_to_cut: list):
        # first, load information into a data frame
        self.ltc = lines_to_cut
        # d is a dict containing topology
        d = self.extract_topo_from_obs(observation)
        self.topo = d
        df = self.create_df(d, lines_to_cut)
        self.df = df
        print("DF From load2")
        print(df)
        self.create_and_fill_internal_structures(observation, df)

    def build_graph_from_data_frame(self, lines_to_cut):
        """This function creates a graph G from a DataFrame"""
        g = nx.MultiDiGraph()
        build_nodes(g, self.topo["nodes"]["are_prods"], self.topo["nodes"]["are_loads"],
                    self.topo["nodes"]["prods_values"], self.topo["nodes"]["loads_values"])

        self.build_edges_from_df(g, lines_to_cut)

        # print("WE ARE IN BUILD GRAPH FROM DATA FRAME ===========")
        # all_edges_xlabel_attributes = nx.get_edge_attributes(g, "xlabel")  # dict[edge]
        # print("all_edges_xlabel_attributes = ", all_edges_xlabel_attributes)

        return g

    def build_detailed_graph_from_internal_structure(self, lines_to_cut):
        """This function create a detailed graph from internal self structures as self.substations_elements..."""
        g = nx.MultiDiGraph()
        network = Network(self.substations_elements)
        print("Network = ", network)
        build_nodes_v2(g, network.nodes_prod_values)
        build_edges_v2(g, network.substation_id_busbar_id_node_id_mapping, self.substations_elements)
        print("This graph is weakly connected : ", nx.is_weakly_connected(g))
        return g

    def change_nodes_configurations(self, new_configuration, node_id):
        """Changes pypownet's internal graph network by changing node : node_id by applying new_configuration"""

        action_space = self.environment.action_space
        action = action_space.get_do_nothing_action(as_class_Action=True)

        for new_conf, id_node in zip(new_configuration, node_id):
            action_space.set_substation_switches_in_action(action, id_node, new_conf)
        raw_simulated_obs, *_ = self.environment.simulate(action)
        # if obs is None, error in the simulation of the next step
        if raw_simulated_obs is None:
            raise ValueError("\n\nPypownet simulation returned a None... Cannot process results...\n")
        # transform raw_obs into Observation object. (Useful for prints, for debugging)
        obs = self.environment.observation_space.array_to_observation(raw_simulated_obs)
        return obs

    @staticmethod
    def extract_topo_from_obs(obs):
        """This function, takes an obs an returns a dict with all topology information"""
        d = {
            "edges": {},
            "nodes": {}
        }
        # print("lines_cut = ", lines_cut)
        nodes_ids = obs.substations_ids
        # print("obs substations_ids = ", nodes_ids)
        idx_or = [int(x - 1) for x in obs.lines_or_substations_ids]
        idx_ex = [int(x - 1) for x in obs.lines_ex_substations_ids]
        prods_ids = obs.productions_substations_ids
        loads_ids = obs.loads_substations_ids
        are_prods = [node_id in prods_ids for node_id in nodes_ids]
        are_loads = [node_id in loads_ids for node_id in nodes_ids]
        prods_values = obs.active_productions
        loads_values = obs.active_loads
        current_flows = obs.active_flows_origin
        d["edges"]["idx_or"] = [x for x in idx_or]
        d["edges"]["idx_ex"] = [x for x in idx_ex]
        d["edges"]["init_flows"] = current_flows
        d["nodes"]["are_prods"] = are_prods
        d["nodes"]["are_loads"] = are_loads
        d["nodes"]["prods_values"] = prods_values
        d["nodes"]["loads_values"] = loads_values
        return d

    def build_powerflow_graph_beforecut(self):
        """
        Builds a graph of the grid and its powerflow before the lines are cut
        :return: NetworkX Graph of representing the grid
        """
        g = self.build_powerflow_graph(self.obs)
        return g

    def build_powerflow_graph_aftercut(self):
        """
        Builds a graph of the grid and its powerflow after the lines have been cut
        :return: NetworkX Graph of representing the grid
        """
        g = self.build_powerflow_graph(self.obs_linecut)
        return g


    def build_powerflow_graph(self, obs):
        """This function takes a pypownet Observation and returns a NetworkX Graph"""
        g = nx.MultiDiGraph()
        lines_cut = np.argwhere(obs.lines_status == 0)
        nodes_ids = obs.substations_ids
        idx_or = [int(x - 1) for x in obs.lines_or_substations_ids]
        idx_ex = [int(x - 1) for x in obs.lines_ex_substations_ids]
        prods_ids = obs.productions_substations_ids
        loads_ids = obs.loads_substations_ids
        are_prods = [node_id in prods_ids for node_id in nodes_ids]
        are_loads = [node_id in loads_ids for node_id in nodes_ids]
        prods_values = obs.active_productions
        loads_values = obs.active_loads
        current_flows = obs.active_flows_origin
        if self.debug:
            print("============================= FUNCTION build_powerflow_graph 2 =============================")
            print("self.idx_or = ", idx_or)
            print("self.idx_ex = ", idx_ex)
            print("Nodes that are prods =", are_prods)
            print("Nodes that are loads =", are_loads)
            print("prods_values = ", prods_values)
            print("loads_values = ", loads_values)
            print("current_flows = ", current_flows)
        # =========================================== NODE PART ===========================================
        build_nodes(g, are_prods, are_loads, prods_values, loads_values, debug=self.debug)
        # =========================================== EDGE PART ===========================================
        build_edges(g, idx_or, idx_ex, edge_weights=current_flows, debug=self.debug,
                    gtype="powerflow", lines_cut=lines_cut)
        return g

    def cut_lines_and_recomputes_flows(self, ids: list):
        """This functions cuts lines: [ids], simulates and returns new line flows"""
        action_space = self.environment.action_space
        action = action_space.get_do_nothing_action(as_class_Action=True)
        for line_id in ids:
            action_space.set_lines_status_switch_from_id(action=action, line_id=line_id, new_switch_value=1)
        raw_simulated_obs = self.environment.simulate(action)
        if raw_simulated_obs[0] is None:
            raise ValueError("The simulation step of Pypownet returnt a None... Something")
        obs = self.environment.observation_space.array_to_observation(raw_simulated_obs[0])
        self.obs_linecut = obs

        return obs.active_flows_origin

    def build_edges_from_df(self, g, lines_to_cut):
        i = 0
        for origin, extremity, reported_flow, gray_edge in zip(self.df["idx_or"], self.df["idx_ex"],
                                                               self.df["delta_flows"], self.df["gray_edges"]):
            penwidth = fabs(reported_flow) / 10
            if penwidth == 0.0:
                penwidth = 0.1
            if i in lines_to_cut:
                g.add_edge(origin, extremity, capacity=float("%.2f" % reported_flow), xlabel="%.2f" % reported_flow,
                           color="black", style="dotted, setlinewidth(2)", fontsize=10, penwidth="%.2f" % penwidth,
                           constrained=True)
            elif gray_edge:  # Gray
                g.add_edge(origin, extremity, capacity=float("%.2f" % reported_flow), xlabel="%.2f" % reported_flow,
                           color="gray", fontsize=10, penwidth="%.2f" % penwidth)
            elif reported_flow < 0:  # Blue
                g.add_edge(origin, extremity, capacity=float("%.2f" % reported_flow), xlabel="%.2f" % reported_flow,
                           color="blue", fontsize=10, penwidth="%.2f" % penwidth)
            else:  # > 0  # Red
                g.add_edge(origin, extremity, capacity=float("%.2f" % reported_flow), xlabel="%.2f" % reported_flow,
                           color="red", fontsize=10, penwidth="%.2f" % penwidth)
            i += 1


def build_nodes(g, are_prods, are_loads, prods_values, loads_values, debug=False):
    # =========================================== NODE PART ===========================================
    print(f"There are {len(are_loads)} nodes")
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
                       fillcolor="#f30000")  # red color
        elif prod_minus_load < 0:  # LOAD
            g.add_node(i, pin=True, prod_or_load="load", value=str(prod_minus_load), style="filled",
                       fillcolor="#478fd0")  # blue color
        else:  # WHITE COLOR
            g.add_node(i, pin=True, prod_or_load="load", value=str(prod_minus_load), style="filled",
                       fillcolor="#ffffed")  # white color
        i += 1


def build_nodes_v2(g, nodes_prod_values: list):
    """nodes_prod_values is a list of tuples, (graphical_node_id, prod_cons_total_value)
        prod_cons_total_value is a float.
        If the value is positive then it is a Production, if negative it is a Consumption
    """
    print("IN FUNCTION BUILD NODES V2222222222", nodes_prod_values)
    for data in nodes_prod_values:
        print("data = ", data)
        i = int(data[0])
        if data[1] is None or data[1] == "XXX":
            prod_minus_load = 0.0  # It will end up as a white node
        else:
            prod_minus_load = data[1]
        print("prod_minus_load = ", prod_minus_load)
        if prod_minus_load > 0:  # PROD
            g.add_node(i, pin=True, prod_or_load="prod", value=str(prod_minus_load), style="filled",
                       fillcolor="#f30000")  # red color
        elif prod_minus_load < 0:  # LOAD
            g.add_node(i, pin=True, prod_or_load="load", value=str(prod_minus_load), style="filled",
                       fillcolor="#478fd0")  # blue color
        else:  # WHITE COLOR
            g.add_node(i, pin=True, prod_or_load="load", value=str(prod_minus_load), style="filled",
                       fillcolor="#ffffed")  # white color
        i += 1


def build_edges_v2(g, substation_id_busbar_id_node_id_mapping, substations_elements):
    print("\nWE ARE IN BUILD EDGES V2")
    substation_ids = sorted(list(substations_elements.keys()))
    # loops through each substation, and creates an edge from (
    for substation_id in substation_ids:
        print("\nSUBSTATION ID = ", substation_id)
        for element in substations_elements[substation_id]:
            print(element)
            origin = None
            extremity = None
            if isinstance(element, OriginLine):
                # origin = substation_id
                origin = int(substation_id_busbar_id_node_id_mapping[substation_id][element.busbar_id])
                extremity = int(element.end_substation_id)
                # check if extremity on busbar1, if it is,
                # check with the substation substation_id_busbar_id_node_id_mapping dic what "graphical" node it is
                print("substations_elements[extremity] = ", substations_elements[extremity])
                for elem in substations_elements[extremity]:
                    # if this true, we are talking about correct edge
                    if isinstance(elem, ExtremityLine) and elem.flow_value == element.flow_value:
                        if elem.busbar == 1:
                            extremity = substation_id_busbar_id_node_id_mapping[extremity][1]
                reported_flow = element.flow_value
            elif origin is None or extremity is None:
                continue
            # in case we get on an element that is Production or Consumption
            else:
                continue
            print("origin = ", origin)
            print("extremity = ", extremity)
            print("reported_flow = ", reported_flow)
            pen_width = fabs(reported_flow[0]) / 10.0
            if pen_width < 0.01:
                pen_width = 0.1
            print(f"#################### Edge created : ({origin}, {extremity}), with flow = {reported_flow},"
                  f" pen_width = {pen_width} >>>")
            if reported_flow[0] > 0:  # RED
                g.add_edge(origin, extremity, capacity=float(reported_flow[0]), xlabel=reported_flow[0], color="red",
                           penwidth="%.2f" % pen_width)
            else:  # BLUE
                g.add_edge(origin, extremity, capacity=float(reported_flow[0]), xlabel=reported_flow[0], color="blue",
                           penwidth="%.2f" % pen_width)
            g.add_edge(origin, extremity, capacity=float(reported_flow[0]), xlabel=reported_flow[0])


def build_edges(g, idx_or, idx_ex, edge_weights, gtype, gray_edges=None, lines_cut=None, debug=False,
                initial_flows=None):
    if gtype is "powerflow":
        for origin, extremity, weight_value in zip(idx_or, idx_ex, edge_weights):
            # origin += 1
            # extremity += 1
            pen_width = fabs(weight_value) / 10
            if pen_width == 0.0:
                pen_width = 0.1

            if weight_value >= 0:
                g.add_edge(origin, extremity, xlabel="%.2f" % weight_value, color="gray", fontsize=10,
                           penwidth="%.2f" % pen_width)
            else:
                g.add_edge(extremity, origin, xlabel="%.2f" % fabs(weight_value), color="gray", fontsize=10,
                           penwidth="%.2f" % pen_width)

    elif gtype is "overflow" and initial_flows is not None:
        i = 0
        for origin, extremity, reported_flow, initial_flow, gray_edge in zip(idx_or, idx_ex, edge_weights,
                                                                             initial_flows, gray_edges):
            # origin += 1
            # extremity += 1
            penwidth = fabs(reported_flow) / 10
            if penwidth == 0.0:
                penwidth = 0.1
            if i in lines_cut:
                g.add_edge(origin, extremity, xlabel="%.2f" % reported_flow, color="black",
                           style="dotted, setlinewidth(2)", fontsize=10, penwidth="%.2f" % penwidth,
                           constrained=True)
            elif gray_edge:  # Gray
                if reported_flow <= 0 and fabs(reported_flow) > 2 * fabs(initial_flow):
                    g.add_edge(extremity, origin, xlabel="%.2f" % reported_flow, color="gray", fontsize=10,
                               penwidth="%.2f" % penwidth)
                else:  # positive
                    g.add_edge(origin, extremity, xlabel="%.2f" % reported_flow, color="gray", fontsize=10,
                               penwidth="%.2f" % penwidth)
            elif reported_flow < 0:  # Blue
                if fabs(reported_flow) > 2 * fabs(initial_flow):
                    g.add_edge(extremity, origin, xlabel="%.2f" % reported_flow, color="blue", fontsize=10,
                               penwidth="%.2f" % penwidth)
                else:
                    g.add_edge(origin, extremity, xlabel="%.2f" % reported_flow, color="blue", fontsize=10,
                               penwidth="%.2f" % penwidth)
            else:  # > 0  # Red
                g.add_edge(origin, extremity, xlabel="%.2f" % reported_flow, color="red", fontsize=10,
                           penwidth="%.2f" % penwidth)
            i += 1
    else:
        raise RuntimeError("Graph's GType not understood, cannot build_edges!")


def get_differencial_topology(new_conf, old_conf):
    """new - old, for elem in result, if elem -1, then put one"""
    assert (len(new_conf) == len(old_conf))
    res = []

    for elemNew, elemOld in zip(new_conf, old_conf):
        r = elemNew - elemOld
        if r < 0:
            res.append(1)
        else:
            res.append(int(r))
    return res
