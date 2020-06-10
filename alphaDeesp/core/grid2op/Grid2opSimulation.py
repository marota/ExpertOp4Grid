from pprint import pprint

import numpy as np
from math import fabs
import networkx as nx
from grid2op.dtypes import dt_int

from alphaDeesp.core.simulation import Simulation
from alphaDeesp.core.elements import OriginLine, Consumption, Production, ExtremityLine


class Grid2opSimulation(Simulation):
    def get_layout(self):
        return [(-280, -81), (-100, -270), (366, -270), (366, -54), (-64, -54), (-64, 54), (366, 0),
                (438, 0), (326, 54), (222, 108), (79, 162), (-152, 270), (-64, 270), (222, 216),
                (-280, -151), (-100, -340), (366, -340), (390, -110), (-14, -104), (-184, 54), (400, -80),
                (438, 100), (326, 140), (200, 8), (79, 12), (-152, 170), (-70, 200), (222, 200)]

    def __init__(self, config, obs, action_space, ltc=9, plot_helper=None):
        super().__init__()

        # Get Grid2op objects
        self.obs = obs
        self.obs_linecut = None
        self.plot_helper = plot_helper
        self.action_space = action_space

        # Get Alphadeesp configuration
        self.ltc = ltc
        self.param_options = config
        self.args_number_of_simulated_topos = config["totalnumberofsimulatedtopos"]
        self.args_inner_number_of_simulated_topos_per_node = config["numberofsimulatedtopospernode"]

        print("Number of generators of the powergrid: {}".format(self.obs.n_gen))
        print("Number of loads of the powergrid: {}".format(self.obs.n_load))
        print("Number of powerline of the powergrid: {}".format(self.obs.n_line))
        print("Number of elements connected to each substations in the powergrid: {}".format(self.obs.sub_info))
        print("Total number of elements: {}".format(self.obs.dim_topo))
        self.internal_to_external_mapping = {}
        self.external_to_internal_mapping = {}
        self.substations_elements = {}

        # Compute data structure representing grid an dtopology
        self.topo = self.extract_topo_from_obs(self.obs)
        self.topo_linecut = None
        self.df = self.create_df(self.topo, ltc)
        self.create_and_fill_internal_structures(self.obs, self.df)

    def get_substation_elements(self):
        return self.substations_elements

    def get_substation_to_node_mapping(self):
        pass

    def get_internal_to_external_mapping(self):
        return self.internal_to_external_mapping

    @staticmethod
    def merge_two_dicts(x, y):
        z = x.copy()   # start with x's keys and values
        z.update(y)    # modifies z with y's keys and values & returns None
        return z

    def get_action_from_topo(self, substation_id, new_conf, obs):
        final_dict = {}
        i = 0
        objects = obs.get_obj_connect_to(substation_id=substation_id)
        for gen_id in objects['generators_id']:
            if "generators_id" not in final_dict:
                final_dict["generators_id"] = []
            final_dict["generators_id"].append((gen_id, new_conf[i]))
            i += 1
        for load_id in objects['loads_id']:
            if "loads_id" not in final_dict:
                final_dict["loads_id"] = []
            final_dict["loads_id"].append((load_id, new_conf[i]))
            i += 1
        for line_id in objects['lines_or_id']:
            if "lines_or_id" not in final_dict:
                final_dict["lines_or_id"] = []
            final_dict["lines_or_id"].append((line_id, new_conf[i]))
            i += 1
        for line_id in objects['lines_ex_id']:
            if "lines_ex_id" not in final_dict:
                final_dict["lines_ex_id"] = []
            final_dict["lines_ex_id"].append((line_id, new_conf[i]))
            i += 1
        print(final_dict)
        return self.action_space({"set_bus": final_dict})

    def compute_new_network_changes(self, ranked_combinations):
        """
        This function takes a dataframe ranked_combinations,
        For each combination it computes a simulation step in Grid2op by following the given combinations
        Number of tested combinations and topo per node is given in alphadeesp parameters
        :returns pandas.DataFrame with results of simulations
        """
        print("\n##############################################################################")
        print("##########...........COMPUTE NEW NETWORK CHANGES..........####################")
        print("##############################################################################")
        end_result_dataframe = self.create_end_result_empty_dataframe()
        j = 0
        for df in ranked_combinations:
            ii = 0
            if j == int(self.args_number_of_simulated_topos):
                break
            for i, row in df.iterrows():
                if ii == int(self.args_inner_number_of_simulated_topos_per_node):
                    break
                obs = self.obs
                # target_node = row["node"] + 1
                internal_target_node = row["node"]
                # target_node = row["node"]
                new_conf = np.array([n + 1 for n in row["topology"]])
                score_topo = i
                print("###########"" Compute new network changes on node [{}] with new topo [{}] ###########"
                      .format(internal_target_node, new_conf))
                action = self.get_action_from_topo(internal_target_node, new_conf, obs)
                virtual_obs, reward, done, info = self.obs.simulate(action)
                self.plot_grid(virtual_obs).show()
                # Same as in Pypownet, this is not what we would want though, as we do the work for only one ltc
                only_line = self.ltc[0]
                line_state_before = obs.state_of(line_id=only_line)
                line_state_after = virtual_obs.state_of(line_id=only_line)
                flow_before = line_state_before["origin"]["p"]
                flow_after = line_state_after["origin"]["p"]
                delta_flow = flow_before - flow_after
                worsened_line_ids = self.create_boolean_array_of_worsened_line_ids(obs, virtual_obs)
                # TODO
                simulated_score = 4  # self.score_changes_between_two_observations(obs, virtual_obs)
                redistribution_prod = np.sum(np.absolute(virtual_obs.prod_p - obs.prod_p))
                redistribution_load = np.sum(np.absolute(virtual_obs.load_p - obs.load_p))
                if simulated_score in [4, 3, 2]:  # success
                    efficacity = fabs(delta_flow / virtual_obs.rho[self.ltc[0]])
                else:  # failure
                    efficacity = -fabs(delta_flow / virtual_obs.rho[self.ltc[0]])
                score_data = [only_line,
                              flow_before,
                              flow_after,
                              delta_flow,
                              worsened_line_ids,
                              redistribution_prod,
                              redistribution_load,
                              new_conf,
                              internal_target_node,
                              1,  # category hubs?
                              score_topo,
                              simulated_score,
                              efficacity]
                print(score_data)
                max_index = end_result_dataframe.shape[0]  # rows
                end_result_dataframe.loc[max_index] = score_data
                ii += 1
                j += 1
        end_result_dataframe.to_csv("./END_RESULT_DATAFRAME.csv", index=True)
        return end_result_dataframe

    @staticmethod
    def create_boolean_array_of_worsened_line_ids(old_obs, new_obs):
        res = []
        for old, new in zip(old_obs.rho, new_obs.rho):
            if new > 1 and old > 1 and new > 1.05 * old:
                res.append(1)
            elif new > 1 > old:
                res.append(1)
            else:
                res.append(0)
        res = np.array(res)
        res = list(np.where(res == 1))
        res = res[0]
        if res.size == 0:
            res = []
        elif isinstance(res, np.ndarray):
            res = list(res)
        return res

    def create_and_fill_internal_structures(self, obs, df):
        """This function fills multiple structures:
        self.substation_elements, self.substation_to_node_mapping, self.internal_to_external_mapping
        @:arg observation, df"""
        # ################ PART I : fill self.internal_to_external_mapping
        substations_list = list(obs.name_sub)
        # we create mapping from external representation to internal.
        for i, substation_id in enumerate(substations_list):
            self.internal_to_external_mapping[i] = substation_id
        if self.internal_to_external_mapping:
            self.external_to_internal_mapping = self.invert_dict_keys_values(self.internal_to_external_mapping)

        # ################ PART II : fill self.substation_elements
        for substation_id in self.internal_to_external_mapping.keys():
            elements_array = []
            objects = obs.get_obj_connect_to(substation_id=substation_id)
            for gen_id in objects['generators_id']:
                gen_state = obs.state_of(gen_id=gen_id)
                elements_array.append(Production(gen_state['bus'], gen_state['p']))
            for load_id in objects['loads_id']:
                load_state = obs.state_of(load_id=load_id)
                elements_array.append(Consumption(load_state['bus'], load_state['p']))
            for line_id in objects['lines_or_id']:
                line_state = obs.state_of(line_id=line_id)
                orig = line_state['origin']
                ext = line_state['extremity']
                dest = ext['sub_id']
                elements_array.append(self.get_model_obj_from_or(self.df, substation_id, dest, orig['bus']))
            for line_id in objects['lines_ex_id']:
                line_state = obs.state_of(line_id=line_id)
                orig = line_state['origin']
                ext = line_state['extremity']
                dest = orig['sub_id']
                elements_array.append(self.get_model_obj_from_ext(self.df, substation_id, dest, ext['bus']))
            self.substations_elements[substation_id] = elements_array
        pprint(self.substations_elements)

    @staticmethod
    def extract_topo_from_obs(obs):
        """This function, takes an obs an returns a dict with all topology information"""
        d = {
            "edges": {},
            "nodes": {}
        }
        nsub = obs.n_sub
        nodes_ids = list(range(nsub))
        idx_or = obs.line_or_to_subid
        idx_ex = obs.line_ex_to_subid
        prods_ids = obs.gen_to_subid
        loads_ids = obs.load_to_subid
        are_prods = [node_id in prods_ids for node_id in nodes_ids]
        are_loads = [node_id in loads_ids for node_id in nodes_ids]
        current_flows = obs.p_or  # Flow at the origin of power line is taken

        # Repartition of prod and load in substations
        prods_values = obs.prod_p
        loads_values = obs.load_p
        gens_ordered_by_subid = np.argsort(obs.gen_to_subid)
        loads_ordered_by_subid = np.argsort(obs.load_to_subid)
        prods_values = prods_values[gens_ordered_by_subid]
        loads_values = loads_values[loads_ordered_by_subid]

        # Store topo in dictionary
        d["edges"]["idx_or"] = [x for x in idx_or]
        d["edges"]["idx_ex"] = [x for x in idx_ex]
        d["edges"]["init_flows"] = current_flows
        d["nodes"]["are_prods"] = are_prods
        d["nodes"]["are_loads"] = are_loads
        d["nodes"]["prods_values"] = prods_values
        d["nodes"]["loads_values"] = loads_values

        # Debug
        for key in d.keys():
            print(key)
            for key2 in d[key].keys():
                print(key2)
                print(d[key][key2])
        return d

    def cut_lines_and_recomputes_flows(self, ids: list):
        """This functions cuts lines: [ids], simulates and returns new line flows"""

        # Set action which disconects the specified lines (by ids)
        deconexion_action = self.action_space({"set_line_status": [(id_, -1) for id_ in ids]})
        obs_linecut, reward, done, info = self.obs.simulate(deconexion_action)

        # Storage of new observation to access features in other function
        self.obs_linecut = obs_linecut
        self.topo_linecut = self.extract_topo_from_obs(self.obs_linecut)

        # Get new flow simulated
        new_flow = self.obs_linecut.p_or

        # Graph building
        # self.g_pow_prime = self.build_powerflow_graph(self.obs_cutted)

        return new_flow

    def build_powerflow_graph_beforecut(self):
        """
        Builds a graph of the grid and its powerflow before the lines are cut
        :return: NetworkX Graph of representing the grid
        """
        g = build_powerflow_graph(self.topo, self.obs)
        return g

    def build_powerflow_graph_aftercut(self):
        """
        Builds a graph of the grid and its powerflow after the lines have been cut
        :return: NetworkX Graph of representing the grid
        """
        g = build_powerflow_graph(self.topo_linecut, self.obs_linecut)
        return g

    def get_dataframe(self):
        """
        :return: pandas dataframe with topology information before and after line cutting
        """
        return self.df

    def build_graph_from_data_frame(self, lines_to_cut):
        """This function creates a graph G from a DataFrame"""
        g = nx.DiGraph()
        build_nodes(g, self.topo["nodes"]["are_prods"], self.topo["nodes"]["are_loads"],
                    self.topo["nodes"]["prods_values"], self.topo["nodes"]["loads_values"])

        self.build_edges_from_df(g, lines_to_cut)

        # print("WE ARE IN BUILD GRAPH FROM DATA FRAME ===========")
        # all_edges_xlabel_attributes = nx.get_edge_attributes(g, "xlabel")  # dict[edge]
        # print("all_edges_xlabel_attributes = ", all_edges_xlabel_attributes)

        return g

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

    def plot_grid_beforecut(self):
        """
        Plots the grid with Grid2op PlotHelper for Observations, before lines are cut
        :return: Figure
        """
        return self.plot_grid(self.obs)

    def plot_grid_aftercut(self):
        """
        Plots the grid with Grid2op PlotHelper for Observations, after lines have been cut
        :return: Figure
        """
        return self.plot_grid(self.obs_linecut)

    def plot_grid(self, obs):
        fig_obs = self.plot_helper.plot_obs(obs, line_info='p')
        return fig_obs


def build_powerflow_graph(topo, obs):
    """This function takes a Grid2op Observation and returns a NetworkX Graph"""
    g = nx.DiGraph()

    # Get the id of lines that are disconnected from network
    lines_cut = np.argwhere(obs.line_status == False)[:, 0]

    # Get the whole topology information
    idx_or = topo["edges"]['idx_or']
    idx_ex = topo["edges"]['idx_ex']
    are_prods = topo["nodes"]['are_prods']
    are_loads = topo["nodes"]['are_loads']
    prods_values = topo["nodes"]['prods_values']
    loads_values = topo["nodes"]['loads_values']
    current_flows = topo["edges"]['init_flows']

    # =========================================== NODE PART ===========================================
    build_nodes(g, are_prods, are_loads, prods_values, loads_values, debug=False)
    # =========================================== EDGE PART ===========================================
    build_edges(g, idx_or, idx_ex, edge_weights=current_flows, debug=False,
                gtype="powerflow", lines_cut=lines_cut)
    return g


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

    @staticmethod
    def score_changes_between_two_observations(old_obs, new_obs):
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
        boolean_constraint_worsened = []
        boolean_overload_30percent_relieved = []
        boolean_overload_relieved = []
        boolean_overload_created = []

        old_obs_lines_capacity_usage = old_obs.rho
        new_obs_lines_capacity_usage = new_obs.rho
        # ################################### PREPROCESSING #####################################
        for elem in old_obs_lines_capacity_usage:
            if elem > 1.0:
                old_number_of_overloads += 1

        for elem in new_obs_lines_capacity_usage:
            if elem > 1.0:
                new_number_of_overloads += 1

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

        redistribution_prod = np.sum(np.absolute(new_obs.prod_p - old_obs.prod_p))
        redistribution_load = np.sum(np.absolute(new_obs.load_p - old_obs.load_p))

        # ################################ END OF PREPROCESSING #################################
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
