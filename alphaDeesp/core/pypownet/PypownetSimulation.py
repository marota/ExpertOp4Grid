import pprint
import sys
import numpy as np
from math import fabs
import pandas as pd


# sys.path.append("/home/mozgawamar/Documents/pypownet_last_version/pypownet/")
sys.path.append("/home/mozgawamar/Documents/pypownet-master")
import pypownet.environment
from pypownet.agent import *
from pypownet.environment import ElementType

import networkx as nx

from alphaDeesp.core.simulation import Simulation
from alphaDeesp.core.elements import *
from alphaDeesp.core.network import Network


class PypownetSimulation(Simulation):
    def __init__(self, param_options=None, debug=False, ltc=9, param_folder=None):
        super().__init__()
        print("PypownetSimulation object created...")

        if not param_options or param_options is None:
            raise AttributeError("\nparam_options are empty or None, meaning the config file is not properly read.")

        if not param_folder or param_folder is None:
            raise AttributeError("\nThe parameters folder for Pypownet is empty or None.")

        # parameters_folder = "./alphaDeesp/ressources/parameters/default14"
        # parameters_folder = "/home/mozgawamar/Documents/Libs/pypownet_fork/pypownet/parameters/default14"
        # parameters_folder = "/home/mozgawamar/Documents/Libs/pypownet_fork/pypownet/parameters/default14_static"
        parameters_folder = param_folder
        game_level = "level0"
        chronic_looping_mode = 'natural'
        chronic_starting_id = 0

        game_over_mode = 'easy'
        without_overflow_cuttof = True

        render_bool = True

        self.save_bag = []
        self.debug = debug
        self.args_number_of_simulated_topos = param_options["totalnumberofsimulatedtopos"]
        self.args_inner_number_of_simulated_topos_per_node = param_options["numberofsimulatedtopospernode"]

        self.grid = None
        self.df = None
        self.topo = None  # a dict create in retrieve topology
        self.lines_to_cut = None
        self.param_options = param_options

        #############################
        self.environment = pypownet.environment.RunEnv(parameters_folder=parameters_folder, game_level=game_level,
                                                       chronic_looping_mode=chronic_looping_mode,
                                                       start_id=chronic_starting_id,
                                                       game_over_mode=game_over_mode,
                                                       without_overflow_cutoff=without_overflow_cuttof)
        print("HARD OVERFLOW = ", self.environment.game.hard_overflow_coefficient)
        print("")

        action_space = self.environment.action_space
        observation_space = self.environment.observation_space

        # Create do_nothing action.
        action_do_nothing = action_space.get_do_nothing_action()

        # Run one step in the environment
        # raw_obs, *_ = self.environment.step(action_do_nothing)

        raw_simulated_obs = self.environment.simulate(action_do_nothing)
        self.obs = self.environment.observation_space.array_to_observation(raw_simulated_obs[0])
        # transform raw_obs into Observation object. (Useful for prints, for debugging)
        #############################

        # new structures to omit querying Pypownet, they are filled in LOAD function.
        # for each substation, we get an array with (Prod, Cons, Line) Objects, representing the actual configuration
        self.substations_elements = {}
        self.substation_to_node_mapping = {}
        self.internal_to_external_mapping = {}  # d[internal_id] = external_name_id
        self.external_to_internal_mapping = {}  # d[external_id] = internal_name_id
        print("current chronic name = ", self.environment.game.get_current_chronic_name())
        print(self.obs)
        self.load(self.obs, ltc)

    def get_layout(self):
        return [(-280, -81), (-100, -270), (366, -270), (366, -54), (-64, -54), (-64, 54), (366, 0),
                (438, 0), (326, 54), (222, 108), (79, 162), (-152, 270), (-64, 270), (222, 216),
                (-280, -151), (-100, -340), (366, -340), (390, -110), (-14, -104), (-184, 54), (400, -80),
                (438, 100), (326, 140), (200, 8), (79, 12), (-152, 170), (-70, 200), (222, 200)]

    def get_substation_elements(self):
        return self.substations_elements

    def get_substation_to_node_mapping(self):
        return self.substation_to_node_mapping

    def get_internal_to_external_mapping(self):
        return self.internal_to_external_mapping

    def compute_new_network_changes(self, ranked_combinations):
        """this function takes a dataframe ranked_combinations,
         for each combination it computes a simulation step in Pypownet with action:
         change nodes topo(combination)"""

        print("\n##############################################################################")
        print("##########...........COMPUTE NEW NETWORK CHANGES..........####################")
        print("##############################################################################")

        # for each node_combination
        # pypownet.step(node_combination)
        # new_obs = pypow.getobs
        #

        # then alphadeesp.score(new_obs)
        # the function score creates a Dataframe with sorted score for each topo change.
        # FINISHED
        final_results = pd.DataFrame()

        # df = ranked_combinations[0]

        # print("df in compute new network change")
        # print(df)

        end_result_dataframe = self.create_end_result_empty_dataframe()

        new_node = True

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
                raw_obs, *_ = self.environment.simulate(action)

                # if obs is None, error in the simulation of the next step
                if raw_obs is None:
                    print("Pypownet simulation returnt a None... Cannot process results...")
                    continue

                # transform raw_obs into Observation object. (Useful for prints, for debugging)
                obs = observation_space.array_to_observation(raw_obs)
                obs.get_lines_capacity_usage()
                if self.debug:
                    print(obs)

                # print(obs.active_flows_origin)
                # print(saved_obs.active_flows_origin)
                # assert(obs.active_flows_origin == saved_obs.active_flows_origin)

                print("old obs addr = ", hex(id(saved_obs)))
                print("new obs addr = ", hex(id(obs)))

                # this is used to display graphs at the end. Check main.
                name = "".join(str(e) for e in new_conf)
                name = str(internal_target_node) + "_" + name

                # for flow, origin, ext, in zip(obs.active_flows_origin, obs.lines_or_nodes, obs.lines_ex_nodes)

                self.save_bag.append([name, obs])

                delta_flow = saved_obs.active_flows_origin[self.lines_to_cut[0]] - \
                             obs.active_flows_origin[self.lines_to_cut[0]]

                print("self.lines to cut[0] = ", self.lines_to_cut[0])
                print("self.obs.status line = ", self.obs.lines_status)
                print(" simulated obs  = ", obs.lines_status)
                print("saved flow = ", list(saved_obs.active_flows_origin.astype(int)))
                print("current flow = ", list(obs.active_flows_origin.astype(int)))
                print("deltaflows = ", (saved_obs.active_flows_origin - obs.active_flows_origin).astype(int))
                print("final delta_flow = ", delta_flow)

                # 1) having the new and old obs. Now how do we compare ?
                simulated_score, worsened_line_ids, redistribution_prod, redistribution_load, efficacity = \
                    self.observations_comparator(saved_obs, obs, score_topo, delta_flow)

                # The next three lines have to been done like this to properly have a python empty list if no lines
                # are worsened. This is important to save, read back, and compare a DATAFRAME
                worsened_line_ids = list(np.where(worsened_line_ids == 1))
                worsened_line_ids = worsened_line_ids[0]

                # further tricks to properly read back a saved dataframe
                if worsened_line_ids.size == 0:
                    worsened_line_ids = []
                elif isinstance(worsened_line_ids, np.ndarray):
                    worsened_line_ids = list(worsened_line_ids)

                # if empty
                # if not worsened_line_ids:
                #     worsened_line_ids = []

                # print(type(worsened_line_ids[0]))
                # print("length = ", len(worsened_line_ids))
                # if len(worsened_line_ids) == 2:
                #     print(type(worsened_line_ids[1]))

                score_data = [self.lines_to_cut[0],
                              saved_obs.active_flows_origin[self.lines_to_cut[0]],
                              obs.active_flows_origin[self.lines_to_cut[0]],
                              delta_flow,
                              worsened_line_ids,
                              redistribution_prod,
                              redistribution_load,
                              new_conf,
                              self.external_to_internal_mapping[target_node],
                              1,  # category hubs?
                              score_topo,
                              simulated_score,
                              efficacity]

                max_index = end_result_dataframe.shape[0]  # rows
                end_result_dataframe.loc[max_index] = score_data

                end_result_dataframe.to_csv("./END_RESULT_DATAFRAME.csv", index=True)
                # print("--------------------------------------------------------------------------------------------")
                # print("----------------------------------- END RESULT DATAFRAME -----------------------------------")
                # print("--------------------------------------------------------------------------------------------")
                # print(end_result_dataframe)

                # self.prin
                # return

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
        # if self.debug:
        #     print("==========================================================================")
        #     print("===================== INSIDE OBSERVATIONS COMPARATOR =====================")
        #     print("==========================================================================")
        #     print("old obs = ")
        #     print(old_obs)
        #
        #     print("new obs = ")
        #     print(new_obs)

        simulated_score = self.score_changes_between_two_observations(old_obs, new_obs)

        worsened_line_ids = self.create_boolean_array_of_worsened_line_ids(old_obs, new_obs)

        redistribution_prod = np.sum(np.absolute(new_obs.active_productions - old_obs.active_productions))
        redistribution_load = np.sum(np.absolute(new_obs.active_loads - old_obs.active_loads))

        if simulated_score in [4, 3, 2]:  # success
            efficacity = fabs(delta_flow / new_obs.get_lines_capacity_usage()[self.lines_to_cut[0]])
            pass
        elif simulated_score in [0, 1]:  # failure
            efficacity = -fabs(delta_flow / new_obs.get_lines_capacity_usage()[self.lines_to_cut[0]])
            pass
        else:
            raise ValueError("Cannot compute efficacity, the score is wrong.")

        return simulated_score, worsened_line_ids, redistribution_prod, redistribution_load, efficacity

    def create_boolean_array_of_worsened_line_ids(self, old_obs, new_obs):
        """This function creates a boolean array of lines that got worse between two observations.
        @:return boolean numpy array [0..1]"""

        res = []

        for old, new in zip(old_obs.get_lines_capacity_usage(), new_obs.get_lines_capacity_usage()):
            if fabs(new) > 1 and fabs(old) > 1 and fabs(new) > 1.05 * fabs(old):  # contrainte existante empiree
                res.append(1)

            # elif fabs(new) > 1 and fabs(old) < 1:  #  nouvelle contrainte
            elif fabs(new) > 1 > fabs(old):
                res.append(1)
            else:
                res.append(0)

        return np.array(res)

    def score_changes_between_two_observations(self, old_obs, new_obs):
        """This function takes two observations and computes a score to quantify the change between old_obs and new_obs.
        @:return int between [0 and 4]
        4: if every overload disappeared
        3: if an overload disappeared without stressing the network
        2: if at least 30% of an overload was relieved
        1: if an overload was relieved but another appeared and got worse
        0: if no overloads were alleviated or if it resulted in some load shedding or production distribution.
        """
        # print("==========================================================================")
        # print("==================== INSIDE SCORE CHANGES BETWEEN 2 OBS ==================")
        # print("==========================================================================")
        old_number_of_overloads = 0
        new_number_of_overloads = 0
        # if new > old > 1.0 then 1 else 0
        boolean_constraint_worsened = []
        # if an overload has been relieved and the % is > 30% then 1 else 0
        boolean_overload_30percent_relieved = []
        boolean_overload_relieved = []
        boolean_overload_created = []

        old_obs_lines_capacity_usage = old_obs.get_lines_capacity_usage()
        new_obs_lines_capacity_usage = new_obs.get_lines_capacity_usage()
        # print("old obs capacity usage = ", old_obs_lines_capacity_usage)
        # print("new obs capacity usage = ", new_obs_lines_capacity_usage)

        # test for rank 3 works for node 5 and max conf simulated = 5, we get some simulated score 3 and 4.
        # old_obs_lines_capacity_usage[9] = 1.2
        # new_obs_lines_capacity_usage[9] = 0.9
        # old_obs_lines_capacity_usage[10] = 1.5
        # test for rank 2, unquote above and below
        # old_obs_lines_capacity_usage[0] = 1.9
        # new_obs_lines_capacity_usage[0] = 2.0

        # test for rank 2
        # old_obs_lines_capacity_usage[9] = 1.2
        # new_obs_lines_capacity_usage[9] = 0.83

        # test for rank 1
        # old_obs_lines_capacity_usage[9] = 1.2
        # new_obs_lines_capacity_usage[9] = 0.9

        # ################################### PREPROCESSING #####################################
        for elem in old_obs_lines_capacity_usage:
            if elem > 1.0:
                old_number_of_overloads += 1

        for elem in new_obs_lines_capacity_usage:
            if elem > 1.0:
                new_number_of_overloads += 1

        # print("new_obs_lines_capacity_usage = ", new_obs_lines_capacity_usage)
        # print("old_obs_lines_capacity_usage = ", old_obs_lines_capacity_usage)

        # preprocessing for score 3 and 2
        for old, new in zip(old_obs_lines_capacity_usage, new_obs_lines_capacity_usage):
            # preprocessing for score 3
            if new > 1.05 * old > 1.0:  # if new > old > 1.0 it means it worsened an existing constraint
                boolean_constraint_worsened.append(1)
            else:
                boolean_constraint_worsened.append(0)

            # preprocessing for score 2
            if old > 1.0:  # if old was an overload:
                surcharge = old - 1.0
                diff = old - new
                percentage_relieved = diff * 100 / surcharge

                # if self.debug:
                # print("old capa usage = ", old)
                # print("new capa usage = ", new)
                # print("diff = ", diff)
                # print("percentage relieved = ", percentage_relieved)

                if percentage_relieved > 30.0:
                    boolean_overload_30percent_relieved.append(1)
                else:
                    boolean_overload_30percent_relieved.append(0)
            else:
                boolean_overload_30percent_relieved.append(0)

            # preprocessing for score 1
            if old > 1.0 > new:
                boolean_overload_relieved.append(1)
            else:
                boolean_overload_relieved.append(0)

            if old < 1.0 < new:
                boolean_overload_created.append(1)
            else:
                boolean_overload_created.append(0)

        boolean_constraint_worsened = np.array(boolean_constraint_worsened)
        boolean_overload_30percent_relieved = np.array(boolean_overload_30percent_relieved)
        boolean_overload_relieved = np.array(boolean_overload_relieved)
        boolean_overload_created = np.array(boolean_overload_created)

        redistribution_prod = np.sum(np.absolute(new_obs.active_productions - old_obs.active_productions))
        redistribution_load = np.sum(np.absolute(new_obs.active_loads - old_obs.active_loads))

        # print("redistribution_prod", redistribution_prod)
        # print("redistribution_load", redistribution_load)
        #
        # print("old_obs.active_productions  = ", list(old_obs.active_productions))
        # print("new_obs.active_productions  = ", list(new_obs.active_productions))
        # print("old_obs.active_loads        = ", list(old_obs.active_loads))
        # print("new_obs.active_loads        = ", list(new_obs.active_loads))
        #
        # print("boolean_overload_relieved   = ", boolean_overload_relieved)
        # print("boolean_overload_created    = ", boolean_overload_created)
        # print("boolean_constraint_worsened = ", boolean_constraint_worsened)
        # print("boolean_overload_30%_reliev = ", boolean_overload_30percent_relieved)
        # print("old_number_of_overloads     = ", old_number_of_overloads)
        # print("new_number_of_overloads     = ", new_number_of_overloads)
        # print("new_obs.are_loads_cut       = ", new_obs.are_loads_cut)
        # print("Thermal limits = ", old_obs.thermal_limits)

        # ################################ END OF PREPROCESSING #################################
        # if old_number_of_overloads == 0:
        #     print("return -1: there were previously no overloads.")
        #     return -1

        # score 0 if no overloads were alleviated or if it resulted in some load shedding or production distribution.
        if redistribution_load > 0 or (new_obs.are_loads_cut == 1).any() or (new_obs.are_productions_cut == 1).any():
            print("return 0: no overloads were alleviated or some load shedding occured.")
            return 0

        # score 1 if overload was relieved but another one appeared and got worse
        elif (boolean_overload_relieved == 1).any() and ((boolean_overload_created == 1).any() or
                                                         (boolean_constraint_worsened == 1).any()):
            print("return 1: an overload was relieved but another one appeared")
            return 1

        # 4: if every overload disappeared
        elif old_number_of_overloads > 0 and new_number_of_overloads == 0:
            print("return 4: every overload disappeared")
            return 4

        # 3: if an overload disappeared without stressing the network, ie,
        # if an overload disappeared
        # and without worsening existing constraint
        # and no Loads that get cut
        elif new_number_of_overloads < old_number_of_overloads and \
                (boolean_constraint_worsened == 0).all() and \
                (new_obs.are_loads_cut == 0).all():
            print("return 3: an overload disappeared without stressing the network")
            return 3

        # 2: if at least 30% of an overload was relieved
        elif (boolean_overload_30percent_relieved == 1).any():
            print("return 2: at least 30% of line [{}] was relieved".format(
                np.where(boolean_overload_30percent_relieved == 1)[0]))
            return 2

        # score 0
        elif (boolean_overload_30percent_relieved == 0).all():
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
            self.external_to_internal_mapping = invert_dict_keys_values(self.internal_to_external_mapping)

        # prod values
        prod_nodes = obs.productions_substations_ids
        cons_nodes = obs.loads_substations_ids
        prod_values = obs.active_productions
        cons_values = obs.active_loads

        # self.cons_values =

        # if self.debug:
        #     print("internal to external mapping")
        #     pprint.pprint(self.internal_to_external_mapping)
        #     print("external to internal mapping")
        #     pprint.pprint(self.external_to_internal_mapping)
        #     print("-----------------------------------------------")
        #     print("prod nodes = ", prod_nodes)
        #     print("prod values = ", prod_values)
        #     print("cons nodes = ", cons_nodes)
        #     print("cons values = ", cons_values)
        #     print("lines_or_nodes = ", obs.lines_or_substations_ids)
        #     print("lines_ex_nodes = ", obs.lines_ex_substations_ids)
        #     print("line_flows = ", list(self.df["delta_flows"].round(decimals=2)))

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

            # flow_value = 0
            # print("test query DATAFRAME")
            # print(list(self.df.query("idx_or == " + str(substation_id) + " & idx_ex == 4")
            #            ["delta_flows"].round(decimals=2)))

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

                        # if not swapped_condition:
                        #     elements_array.append(OriginLine(busbar, dest, flow_value))
                        # else:
                        #     elements_array.append(ExtremityLine(busbar, dest, flow_value))

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

        # if self.debug:
        # print("from Load2, create_and_fill_internal_structure, substations_elements = ")
        # pprint.pprint(self.substations_elements)

    def load(self, observation, lines_to_cut: list):
        # first, load information into a data frame
        self.lines_to_cut = lines_to_cut

        # d is a dict containing topology
        d = self.extract_topo_from_obs(observation)
        self.topo = d

        df = self.create_df(d, lines_to_cut)
        self.df = df
        print("DF From load2")
        print(df)

        # this creates and fills
        # self.substation_elements, self.substation_to_node_mapping, self.internal_to_external_mapping
        self.create_and_fill_internal_structures(observation, df)

    def build_graph_from_data_frame(self, lines_to_cut):
        """This function creates a graph G from a DataFrame"""
        g = nx.DiGraph()
        build_nodes(g, self.topo["nodes"]["are_prods"], self.topo["nodes"]["are_loads"],
                    self.topo["nodes"]["prods_values"], self.topo["nodes"]["loads_values"])

        self.build_edges_from_df(g, lines_to_cut)

        # print("WE ARE IN BUILD GRAPH FROM DATA FRAME ===========")
        # all_edges_xlabel_attributes = nx.get_edge_attributes(g, "xlabel")  # dict[edge]
        # print("all_edges_xlabel_attributes = ", all_edges_xlabel_attributes)

        return g, self.df

    def build_detailed_graph_from_internal_structure(self, lines_to_cut):
        """This function create a detailed graph from internal self structures as self.substations_elements..."""

        custom_layout = [(-280, -81), (-100, -270), (366, -270), (366, -54), (-64, -54), (-64, 54), (366, 0),
                         (438, 0), (326, 54), (222, 108), (79, 162), (-152, 270), (-64, 270), (222, 216),
                         (-280, -151), (-100, -340), (366, -340), (390, -110), (-14, -104), (-184, 54), (400, -80),
                         (438, 100), (326, 140), (200, 8), (79, 12), (-152, 170), (-70, 200), (222, 200)]
        # nodes = self.substations_elements.keys()
        #
        # are_prods = np.zeros(len(nodes))
        # are_loads = np.zeros(len(nodes))
        # prod_values = np.zeros(len(nodes))
        # load_values = np.zeros(len(nodes))

        # #############################################################################################################
        # ############## FIRST PART, KNOW HOW MANY DISTINCT NODES WE HAVE, busbar0 and busbar 1 == 2 nodes
        # #############################################################################################################

        # var used to know if a node is using his "second" busbar
        # twin_node_needed = False
        #
        # for substation_id in nodes:
        #     twin_node_needed = False
        #
        #     for configuration in self.substations_elements[substation_id]:
        #         # substation_id = busbar 0, 666+(substation_id) = busbar 1
        #         # so here we detect if per configuration, there are 2 busbars needed, cad, add substation_id(666+id)
        #         if configuration.busbar_id == 1.0:
        #             twin_node_needed = True
        #
        #
        # nb_prods = 0
        # nb_loads = 0

        # main loop over all the network
        # for substation_id in nodes:
        #     print("Node ID = ", substation_id)
        #     for element in self.substations_elements[substation_id]:
        #         print(element)
        #
        #
        #         node_info = {
        #
        #         }

        # print("are prods = ", are_prods)
        # print("are loads = ", are_loads)
        # print("prod_values = ", prod_values)
        # print("load_values = ", load_values)

        # build_nodes(g, are_prods, are_loads, prod_values, load_values, debug=True)
        # print("Nodes")
        # print(g.nodes())

        g = nx.DiGraph()

        network = Network(self.substations_elements)
        print("Network = ", network)

        build_nodes_v2(g, network.nodes_prod_values)
        build_edges_v2(g, network.substation_id_busbar_id_node_id_mapping, self.substations_elements)
        print("This graph is weakly connected : ", nx.is_weakly_connected(g))
        # if not (nx.is_weakly_connected(g)):
        #     raise ValueError("\n\nWe don't allow disconnected graphs to be displayed or computed")

        # p = Printer()
        # p.display_geo(g, custom_layout, name="build_detailed_graph_examples")

        # for node in network.nodes_prod_values:
        #     print(node)

        return g

    def change_nodes_configurations(self, new_configuration, node_id):
        """Changes pypownet's internal graph network by changing node : node_id by applying new_configuration"""

        action_space = self.environment.action_space
        action = action_space.get_do_nothing_action(as_class_Action=True)

        for new_conf, id_node in zip(new_configuration, node_id):
            action_space.set_substation_switches_in_action(action, id_node, new_conf)

        # raw_simulated_obs = self.environment.simulate(action)
        raw_simulated_obs, *_ = self.environment.simulate(action)
        # if obs is None, error in the simulation of the next step
        if raw_simulated_obs is None:
            raise ValueError("\n\nPypownet simulation returned a None... Cannot process results...\n")
            # print("Pypownet simulation returnt a None... Cannot process results...")

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
        lines_cut = np.argwhere(obs.lines_status == 0)
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

        # print("extract topo from obs")
        # pprint.pprint(d)

        return d

    def build_powerflow_graph(self, obs):
        """This function takes a pypownet Observation and returns a NetworkX Graph"""
        g = nx.DiGraph()

        # print("======================================== BUILD POWERFLOW GRAPH2 ")
        # print("lines status = ", obs.lines_status)
        # print(obs)
        # print("Object type obs = ", type(obs))
        lines_cut = np.argwhere(obs.lines_status == 0)
        # print("lines_cut = ", lines_cut)
        # idx_or = obs.lines_or_nodes
        # idx_ex = obs.lines_ex_nodes
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

    # def build_overflow_graph(self, grid, lines_cut, param_options):
    #     """This function takes a pypownet grid and returns a NetworkX Graph"""
    #     print("we'll use var ThresholdReportOfLine = ", param_options["ThresholdReportOfLine"])
    #
    #     g = nx.DiGraph()
    #
    #     # first we extract the flows
    #     initial_flows = grid.extract_flows_a()
    #     initial_flows = grid.mpc["branch"][:, 13]
    #     # as we are creating an overflow graph we need to cut line then recompute flows
    #     new_flows = self.cut_lines_and_recomputes_flows(grid, lines_cut)
    #
    #     # retrieve topology
    #     mpcbus = grid.mpc['bus']
    #     mpcgen = grid.mpc['gen']
    #     half_nodes_ids = mpcbus[:len(mpcbus) // 2, 0]
    #     node_to_substation = lambda x: int(float(str(x).replace('666', '')))
    #     # intermediate step to get idx_or and idx_ex
    #     nodes_or_ids = np.asarray(list(map(node_to_substation, grid.mpc['branch'][:, 0])))
    #     nodes_ex_ids = np.asarray(list(map(node_to_substation, grid.mpc['branch'][:, 1])))
    #     # origin
    #     idx_or = [np.where(half_nodes_ids == or_id)[0][0] for or_id in nodes_or_ids]
    #     # extremeties
    #     idx_ex = [np.where(half_nodes_ids == ex_id)[0][0] for ex_id in nodes_ex_ids]
    #
    #     # retrieve loads and prods
    #     nodes_ids = mpcbus[:, 0]
    #     prods_ids = mpcgen[:, 0]
    #     are_prods = np.logical_or([node_id in prods_ids for node_id in nodes_ids[:len(nodes_ids) // 2]],
    #                               [node_id in prods_ids for node_id in nodes_ids[len(nodes_ids) // 2:]])
    #     are_loads = np.logical_or(grid.are_loads[:len(mpcbus) // 2], grid.are_loads[len(nodes_ids) // 2:])
    #     prods_values = grid.mpc['gen'][:, 1]
    #     loads_values = grid.mpc['bus'][grid.are_loads, 2]
    #     lines_por_values = grid.mpc['branch'][:, 13]
    #
    #     lines_cut = np.argwhere(grid.get_lines_status() == 0)
    #
    #     delta_flows = new_flows - initial_flows
    #
    #     # finally we compute the edges to be displayed in gray. IE: if report line is < ThresholdReportOfLine*MAX_OVER
    #     gray_edges = []  # boolean array
    #     # print("type delta flows = ", type(delta_flows))
    #     # print("max = ", max(delta_flows))
    #     max_overload = max(delta_flows) * float(param_options["ThresholdReportOfLine"])
    #     print("max overload = ", max_overload)
    #     for edge_value in delta_flows:
    #         if fabs(edge_value) < max_overload:
    #             gray_edges.append(True)
    #         else:
    #             gray_edges.append(False)
    #     print("gray edges = ", gray_edges)
    #
    #     if self.debug:
    #         print("============================= FUNCTION build_overflow_graph =============================")
    #         print("self.idx_or = ", idx_or)
    #         print("self.idx_ex = ", idx_ex)
    #         print("self.lines_por_values = ", lines_por_values)
    #         print("Nodes that are prods =", are_prods)
    #         print("Nodes that are loads =", are_loads)
    #         print("prods_values = ", prods_values)
    #         print("loads_values = ", loads_values)
    #         print("initial_flows = ", initial_flows)
    #         print("new_flows = ", new_flows)
    #         print("delta_flows = ", delta_flows)
    #
    #     # =========================================== NODE PART ===========================================
    #     build_nodes(g, are_prods, are_loads, prods_values, loads_values, debug=self.debug)
    #     # =========================================== EDGE PART ===========================================
    #     build_edges(g, idx_or, idx_ex, gray_edges=gray_edges, edge_weights=delta_flows, debug=self.debug,
    #                 initial_flows=initial_flows, gtype="overflow", lines_cut=lines_cut)
    #
    #     return g

    def cut_lines_and_recomputes_flows(self, ids: list):
        """This functions cuts lines: [ids], simulates and returns new line flows"""
        # print("DEBUG IN CUT LINES AND RECOMPUTE FLOWS")
        # print("LINES STATUS ==================",  self.obs.lines_status)
        # print("THERMAL LIMITS = ", self.obs.thermal_limits)
        # print("capacity usages ", self.obs.get_lines_capacity_usage())
        # print("flows in ampere = ", self.obs.ampere_flows)
        # print("DEBUG IN LINE TO CUT IDS = ", ids)
        action_space = self.environment.action_space
        action = action_space.get_do_nothing_action(as_class_Action=True)
        for line_id in ids:
            # print("we switch line id to ")
            action_space.set_lines_status_switch_from_id(action=action, line_id=line_id, new_switch_value=1)

        raw_simulated_obs = self.environment.simulate(action)
        # print("type = ", type(raw_simulated_obs[0]))
        if raw_simulated_obs[0] is None:
            raise ValueError("The simulation step of Pypownet returnt a None... Something")
        obs = self.environment.observation_space.array_to_observation(raw_simulated_obs[0])
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
            reported_flow = None

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

                # if substations_elements[extremity].busbar_id == 1:
                #     extremity = substation_id_busbar_id_node_id_mapping[extremity][1]

                reported_flow = element.flow_value

            # elif isinstance(element, ExtremityLine):
            #     origin = element.start_substation_id
            #     extremity = substation_id_busbar_id_node_id_mapping[substation_id][element.busbar_id]
            #     reported_flow = element.flow_value

            elif origin is None or extremity is None:
                continue

            # in case we get on an element that is Production or Consumption
            else:
                continue

            print("origin = ", origin)
            print("extremity = ", extremity)
            print("reported_flow = ", reported_flow)

            # if isinstance(element, OriginLine):
            #     print("origin mapped = ", substation_id_busbar_id_node_id_mapping[origin][element.busbar_id])
            # if isinstance(element, ExtremityLine):
            #     print("extremity mapped = ", substation_id_busbar_id_node_id_mapping[extremity][element.busbar_id])

            # if origin == "6660":
            #     print("WE SKIP")
            #     continue

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
    if debug:
        ar = list(zip(idx_or, idx_ex, edge_weights))
        # print(" ==== Build_edges debug === : ZIP OF DEATH = ")
        # pprint.pprint(ar)

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

                # if reported_flow >= 0:
                #     g.add_edge(origin, extremity, xlabel="%.2f" % reported_flow, color="blue", fontsize=10,
                #                penwidth="%.2f" % penwidth)
                # else:
                # g.add_edge(extremity, origin, xlabel="%.2f" % reported_flow, color="blue", fontsize=10,
                #            penwidth="%.2f" % penwidth)
            else:  # > 0  # Red
                # if reported_flow >= 0:
                g.add_edge(origin, extremity, xlabel="%.2f" % reported_flow, color="red", fontsize=10,
                           penwidth="%.2f" % penwidth)
                # else:
                #     g.add_edge(extremity, origin, xlabel="%.2f" % reported_flow, color="red", fontsize=10,
            #                penwidth="%.2f" % penwidth)
            i += 1
    else:
        raise RuntimeError("Graph's GType not understood, cannot build_edges!")


# class RTESimulation(Simulation):
#     def __init__(self):
#         super().__init__()


def invert_dict_keys_values(d):
    return dict([(v, k) for k, v in d.items()])


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
