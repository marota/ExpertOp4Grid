from pprint import pprint
import ast

import numpy as np
from math import fabs
import networkx as nx
from grid2op.dtypes import dt_int

from grid2op.PlotGrid import PlotMatplot

from alphaDeesp.core.simulation import Simulation
from alphaDeesp.core.network import Network
from alphaDeesp.core.elements import OriginLine, Consumption, Production, ExtremityLine
from alphaDeesp.core.printer import Printer


class Grid2opSimulation(Simulation):
    def compute_layout(self):
        try:
            layout = self.param_options['CustomLayout']
            # Conversion from string to list
            layout = ast.literal_eval(layout)
            print("WARNING : A CustomLayout has been given in config.ini. This layout will be set for the simulator")
        except:
            try:
                # Grid2op Layout if exists
                layout = list(self.obs.grid_layout.values())
                print("WARNING : No CustomLayout has been given in config.ini. The grid_layout in Grid2Op structure will be set for the simulator")
            except:
                layout = [(-280, -81), (-100, -270), (366, -270), (366, -54), (-64, -54), (-64, 54), (366, 0),
                          (438, 0), (326, 54), (222, 108), (79, 162), (-152, 270), (-64, 270), (222, 216),
                          (-280, -151), (-100, -340), (366, -340), (390, -110), (-14, -104), (-184, 54), (400, -80),
                          (438, 100), (326, 140), (200, 8), (79, 12), (-152, 170), (-70, 200), (222, 200)]
                print("WARNING : No CustomLayout has been given in config.ini and no grid_layout has been found in Grid2op data. Default layout is set and might cause plotting errors : "+str(layout))
        return layout

    def get_layout(self):
        return self.layout

    def __init__(self, obs, action_space, observation_space, param_options=None, debug = False, ltc=[9],other_ltc=[], plot=False, plot_folder = None,reward_type=None):
        super().__init__()

        # Get Grid2op objects
        if ltc is None:
            ltc = [9]
        if plot: # Manual mode
            self.plot_folder = plot_folder
            self.printer = Printer(plot_folder)
        self.obs = obs
        self.obs_linecut = None
        self.action_space = action_space
        self.observation_space = observation_space
        self.plot_helper = self.get_plot_helper()
        self.no_overflow_disc = self.obs._obs_env.no_overflow_disconnection # Keep it in memory to activate and deactivate during computation steps

        # Get Alphadeesp configuration
        self.ltc = ltc
        self.other_ltc = other_ltc
        self.substation_in_cooldown=self.get_substation_in_cooldown()
        self.param_options = param_options
        self.reward_type=reward_type
        self.args_number_of_simulated_topos = param_options["totalnumberofsimulatedtopos"]
        self.args_inner_number_of_simulated_topos_per_node = param_options["numberofsimulatedtopospernode"]

        print("Number of generators of the powergrid: {}".format(self.obs.n_gen))
        print("Number of loads of the powergrid: {}".format(self.obs.n_load))
        print("Number of powerline of the powergrid: {}".format(self.obs.n_line))
        print("Number of elements connected to each substations in the powergrid: {}".format(self.obs.sub_info))
        print("Total number of elements: {}".format(self.obs.dim_topo))
        self.internal_to_external_mapping = {}
        self.external_to_internal_mapping = {}
        self.substations_elements = {}

        # Layout of the grid
        self.layout = self.compute_layout()

        # Compute data structure representing grid an dtopology
        self.topo = None
        self.topo_linecut = None
        self.df = None
        self.load()
        self.save_bag = []

    def load(self):
        self.load_from_observation(self.obs, self.ltc+self.other_ltc)

    def load_from_observation(self, obs, linesToDisconnect):
        #self.obs = obs
        self.topo = self.extract_topo_from_obs(obs)
        self.topo_linecut = None
        self.df = self.create_df(self.topo, linesToDisconnect)
        self.internal_to_external_mapping = {}
        self.external_to_internal_mapping = {}
        self.substations_elements = {}
        self.create_and_fill_internal_structures(obs, self.df)

    def get_substation_elements(self):
        return self.substations_elements

    def get_substation_in_cooldown(self):
        return [i for i in range(self.obs.n_sub) if (self.obs.time_before_cooldown_sub[i]>=1)]

    def get_reference_topovec_sub(self,sub):
        nelements=self.obs.sub_info[sub]
        topovec=[0 for i in range(nelements)]
        return topovec

    def get_overload_disconnection_topovec_subor(self,l):
        sub_or=self.obs.line_or_to_subid[l]
        position_at_sub=self.obs.line_or_to_sub_pos[l]
        current_topo_vec=self.obs.state_of(substation_id=sub_or)['topo_vect']

        new_topo_vec=current_topo_vec
        new_topo_vec[position_at_sub]=-1#to get line disconnection
        return sub_or,new_topo_vec

    def get_substation_to_node_mapping(self):
        pass

    def get_internal_to_external_mapping(self):
        return self.internal_to_external_mapping

    def get_plot_helper(self):
        plot_helper = PlotMatplot(self.observation_space)
        return plot_helper

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
        # print(final_dict)
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
        actions = []
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
                alphaDeesp_Internal_topo=np.array([n for n in row["topology"]])

                new_conf = np.array([n + 1 for n in row["topology"]])
                if(len(alphaDeesp_Internal_topo)==1):#this is a line to disconnect, not a topology to change
                    l=alphaDeesp_Internal_topo[0]
                    #sub_id,new_conf_grid2op=self.get_overload_disconnection_topovec_subor(l)
                    new_conf=[l]
                    #action = self.action_space({"set_bus": {"substations_id": [(sub_id, new_conf_grid2op)] }})


                score_topo = i
                print("###########"" Compute new network changes on node [{}] with new topo [{}] ###########"
                      .format(internal_target_node, new_conf))


                if(len(alphaDeesp_Internal_topo)==1):#this is a line to disconnect, not a topology to change
                    new_conf_grid2op=[l]
                    new_conf=[l]
                    action = self.action_space({"set_line_status": [(l, -1)]})
                else:
                    action = self.get_action_from_topo(internal_target_node, new_conf, obs)
                    new_conf_grid2op = list(action.effect_on(substation_id=internal_target_node)[
                                                'set_bus'])  # grid2op conf is different from alphadeesp conf, because the elements are ordered differently
                actions.append(action)


                # virtual_obs, reward, done, info = self.obs.simulate(action)
                virtual_obs, reward, done, info = self.obs.simulate(action, time_step = 0)

                score_data=self.compute_one_network_change_score_data(obs,virtual_obs,done,info,new_conf,internal_target_node,alphaDeesp_Internal_topo,new_conf_grid2op,score_topo)
                #print(score_data)
                max_index = end_result_dataframe.shape[0]  # rows
                end_result_dataframe.loc[max_index] = score_data
                ii += 1
                j += 1

        end_result_dataframe.to_csv("./END_RESULT_DATAFRAME.csv", index=True)

        # Case there are no hubs --> action do nothing
        if len(actions) == 0:
            actions = [self.action_space()]
        return end_result_dataframe, actions

    def compute_one_network_change_score_data(self, obs,virtual_obs,done,info,new_conf,internal_target_node,alphaDeesp_Internal_topo,new_conf_grid2op,score_topo):
        # Same as in Pypownet, this is not what we would want though, as we do the work for only one ltc
        only_line = self.ltc[0]
        line_state_before = obs.state_of(line_id=only_line)
        line_state_after = virtual_obs.state_of(line_id=only_line)
        flow_before = line_state_before["origin"]["p"]
        flow_after = line_state_after["origin"]["p"]
        delta_flow = flow_before - flow_after

        if done:  # Game over: no need to compute further operations
            worsened_line_ids = []
            simulated_score = 0
            redistribution_prod = float('nan')
            redistribution_load = float('nan')
            efficacity = float('nan')

        else:
            # Fill save bag with observations for further analysis (detailed graph)
            name = "".join(str(e) for e in alphaDeesp_Internal_topo)#"".join(str(e) for e in new_conf)
            name = str(internal_target_node) + "_" + name
            self.save_bag.append([name, virtual_obs])
            worsened_line_ids = self.create_boolean_array_of_worsened_line_ids(obs, virtual_obs,
                                                                               self.obs._obs_env.parameters.NB_TIMESTEP_COOLDOWN_LINE)
            simulated_score = score_changes_between_two_observations(self.ltc, obs, virtual_obs,
                                                                     self.obs._obs_env.parameters.NB_TIMESTEP_COOLDOWN_LINE)

            # update simulated score to 0 in case
            # our line got disconnected, starting a cascading failure
            if (bool(info['disc_lines'][self.ltc])):#other line disconnections are already accounted in worsened lines
            #if (info['disc_lines'].any()):
                simulated_score = 0

            redistribution_prod = np.sum(np.absolute(virtual_obs.prod_p - obs.prod_p))

            TotalProd = np.nansum(virtual_obs.prod_p)
            Losses = np.nansum(np.abs(virtual_obs.p_or + virtual_obs.p_ex))
            ExpectedNewLoad = TotalProd - Losses
            redistribution_load = (np.sum(virtual_obs.load_p) - ExpectedNewLoad)  # / np.sum(old_obs.load_p)

            if simulated_score in [4, 3, 2, 1]:  # success
                efficacity = fabs(delta_flow / virtual_obs.rho[self.ltc[0]])
                if (self.reward_type is not None) and (self.reward_type in info["rewards"]):  # & (simulated_score==4):
                    # dans le cas ou on resoud bien les contraintes, on prend la reward L2RPN
                    efficacity = info["rewards"][self.reward_type]
            else:  # failure
                efficacity = -fabs(delta_flow / virtual_obs.rho[self.ltc[0]])

        # To store in data frame
        score_data = [only_line,
                      flow_before,
                      flow_after,
                      delta_flow,
                      worsened_line_ids,
                      redistribution_prod,
                      redistribution_load,
                      alphaDeesp_Internal_topo,  # alphaDeesp internal topology format
                      new_conf_grid2op,
                      # new_conf,#we prefer to have the backend conf definition, rather than alphadeesp one
                      internal_target_node,
                      1,  # category hubs?
                      score_topo,
                      simulated_score,
                      efficacity]
        # print(score_data)
        return score_data

    @staticmethod
    def create_boolean_array_of_worsened_line_ids(old_obs, new_obs,nb_timestep_cooldown_line_param):
        res = []
        n_lines=len(new_obs.rho)

        #for old, new in zip(old_obs, new_obs):
        for l in range(n_lines):
            if new_obs.rho[l] > 1 and old_obs.rho[l] > 1 and new_obs.rho[l] > 1.05 * old_obs.rho[l]:
                res.append(1)
            elif new_obs.rho[l] > 1 > old_obs.rho[l]:
                res.append(1)
            elif ((new_obs.time_before_cooldown_line[l] - old_obs.time_before_cooldown_line[l])>nb_timestep_cooldown_line_param): #line got into cascading failure
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
                elements_array.append(Production(gen_state['bus']-1, gen_state['p']))
            for load_id in objects['loads_id']:
                load_state = obs.state_of(load_id=load_id)
                elements_array.append(Consumption(load_state['bus']-1, load_state['p']))
            for line_id in objects['lines_or_id']:
                line_state = obs.state_of(line_id=line_id)
                orig = line_state['origin']
                ext = line_state['extremity']
                dest = ext['sub_id']
                elements_array.append(self.get_model_obj_from_or(self.df, substation_id, dest, orig['bus']-1))
            for line_id in objects['lines_ex_id']:
                line_state = obs.state_of(line_id=line_id)
                orig = line_state['origin']
                ext = line_state['extremity']
                dest = orig['sub_id']
                elements_array.append(self.get_model_obj_from_ext(self.df, substation_id, dest, ext['bus']-1))
            self.substations_elements[substation_id] = elements_array
        # pprint(self.substations_elements)

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
        # for key in d.keys():
        #     print(key)
        #     for key2 in d[key].keys():
        #         print(key2)
        #         print(d[key][key2])
        return d

    def cut_lines_and_recomputes_flows(self, ids: list):
        """This functions cuts lines: [ids], simulates and returns new line flows"""

        # First, set parameter to avoid disconnection
        self.obs._obs_env._parameters.NO_OVERFLOW_DISCONNECTION = True
        self.obs._obs_env._no_overflow_disconnection=True
        max_line_actions=self.obs._obs_env.parameters.MAX_LINE_STATUS_CHANGED

        self.obs._obs_env._parameters.MAX_LINE_STATUS_CHANGED = 999

        # Set action which disconects the specified lines (by ids)
        deconexion_action = self.action_space({"set_line_status": [(id_, -1) for id_ in ids]})
        obs_linecut, reward, done, info = self.obs.simulate(deconexion_action, time_step = 0) #, time_step = 0)
        # Storage of new observation to access features in other function
        self.obs_linecut = obs_linecut
        self.topo_linecut = self.extract_topo_from_obs(self.obs_linecut)

        # Get new flow simulated
        new_flow = self.obs_linecut.p_or

        # Graph building
        # self.g_pow_prime = self.build_powerflow_graph(self.obs_cutted)

        # Finaly, reset previous parameter
        self.obs._obs_env._no_overflow_disconnection = self.no_overflow_disc
        self.obs._obs_env._parameters.NO_OVERFLOW_DISCONNECTION = self.no_overflow_disc
        self.obs._obs_env._parameters.MAX_LINE_STATUS_CHANGED=max_line_actions

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

    def isAntenna(self):
        linesAtBusbar_dic = self.getLinesAtSubAndBusbar()

        for sub in linesAtBusbar_dic.keys():
            linesAtBusbar = linesAtBusbar_dic[sub]
            if len(linesAtBusbar) <= 1:
                return sub  # this is an Antenna
        return None

    def isDoubleLine(self):
        ltc = self.ltc[0]
        obs = self.obs

        sub_or = int(obs.line_or_to_subid[ltc])
        sub_ex = int(obs.line_ex_to_subid[ltc])

        linesOr_atSubOr = list(obs.get_obj_connect_to(substation_id=sub_or)['lines_or_id'])
        linesEx_atSubOr = list(obs.get_obj_connect_to(substation_id=sub_or)['lines_ex_id'])

        lines_atSubOr=linesOr_atSubOr+linesEx_atSubOr

        linesOr_atSubEx = list(obs.get_obj_connect_to(substation_id=sub_ex)['lines_or_id'])
        linesEx_atSubEx = list(obs.get_obj_connect_to(substation_id=sub_ex)['lines_ex_id'])

        lines_atSubEx = linesOr_atSubEx+linesEx_atSubEx

        Common_lines=[l for l in lines_atSubEx if (l in lines_atSubOr) and (l != ltc)]

        if(len(Common_lines)==0):
            return None
        else:
            return Common_lines


    def getLinesAtSubAndBusbar(self):
        ltc = self.ltc[0]
        obs=self.obs
        linesAtBusbar_dic = {}

        sub_or = int(obs.line_or_to_subid[ltc])
        sub_ex = int(obs.line_ex_to_subid[ltc])

        busBarOr = obs.state_of(line_id=ltc)['origin']['bus']
        busBarEx = obs.state_of(line_id=ltc)['extremity']['bus']

        # we should check if another line is connected with our line of interest. Otherwise it is an antenna
        # For SubOr
        linesOr_atSubOr = obs.get_obj_connect_to(substation_id=sub_or)['lines_or_id']
        linesEx_atSubOr = obs.get_obj_connect_to(substation_id=sub_or)['lines_ex_id']

        linesAtBusbarOr = [l for l in linesOr_atSubOr if obs.state_of(line_id=l)['origin']['bus'] == busBarOr]
        linesAtBusbarOr += [l for l in linesEx_atSubOr if obs.state_of(line_id=l)['extremity']['bus'] == busBarOr]

        # For SubEx
        linesOr_atSubEx = obs.get_obj_connect_to(substation_id=sub_ex)['lines_or_id']
        linesEx_atSubEx = obs.get_obj_connect_to(substation_id=sub_ex)['lines_ex_id']

        linesAtBusbarEx = [l for l in linesOr_atSubEx if obs.state_of(line_id=l)['origin']['bus'] == busBarEx]
        linesAtBusbarEx += [l for l in linesEx_atSubEx if obs.state_of(line_id=l)['extremity']['bus'] == busBarEx]

        linesAtBusbar_dic[sub_or] = linesAtBusbarOr
        linesAtBusbar_dic[sub_ex] = linesAtBusbarEx
        return linesAtBusbar_dic

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

        # Reduce busbar_ids by -1 (Grid2op: 1,2 / Pypownet: 0,1)
        # for key in self.substations_elements.keys():
        #     for elt in self.substations_elements[key]:
        #         elt.busbar_id -= 1

        network = Network(self.substations_elements)

        # print("Network = ", network)
        build_nodes_v2(g, network.nodes_prod_values)
        build_edges_v2(g, network.substation_id_busbar_id_node_id_mapping, self.substations_elements)
        # print("This graph is weakly connected : ", nx.is_weakly_connected(g))
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
        return self.plot_grid(self.obs, name = "g_pow")

    def plot_grid_aftercut(self):
        """
        Plots the grid with Grid2op PlotHelper for Observations, after lines have been cut
        :return: Figure
        """
        return self.plot_grid(self.obs_linecut, name = "g_pow_prime")

    def plot_grid_delta(self):
        """
        Plots the grid with alphadeesp.printer API for delta between Observations before and after lines have been cut
        :return: Figure
        """
        return self.plot_grid(None, name="g_overflow_print")

    def plot_grid_from_obs(self, obs, name):
        """
        Plots the grid with Grid2op PlotHelper from given observation
        :return: Figure
        """
        return self.plot_grid(obs, name=name)

    def plot_grid(self, obs, name):
        type_ = "results"
        if name in ["g_pow", "g_overflow_print", "g_pow_prime"]:
            type_ = "base"

        if name == "g_overflow_print":  # Use printer API to plot g_over (graphviz/neato)
            g_over = self.build_graph_from_data_frame(self.ltc)
            self.printer.display_geo(g_over, self.get_layout(), name=name)
        else:   # Use grid2op plot functionalities to plot all other graphs
            output_name = self.printer.create_namefile("geo", name = name, type = type_)
            fig_obs = self.plot_helper.plot_obs(obs, line_info='p')
            fig_obs.savefig(output_name[1])

    def change_nodes_configurations(self, new_configurations, node_ids, env):
        change = []
        for (conf, node) in zip(new_configurations, node_ids):
            change.append((node, conf))
        action = self.action_space({"set_bus": {"substations_id": change}})
        new_obs, reward, done, info = env.step(action)
        self.obs = new_obs
        self.load()
        return new_obs


def build_powerflow_graph(topo, obs):
    """This function takes a Grid2op Observation and returns a NetworkX Graph"""
    g = nx.MultiDiGraph()

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
                       fillcolor="#f30000")  # red color
        elif prod_minus_load < 0:  # LOAD
            g.add_node(i, pin=True, prod_or_load="load", value=str(prod_minus_load), style="filled",
                       fillcolor="#478fd0")  # blue color
        else:  # WHITE COLOR
            g.add_node(i, pin=True, prod_or_load="load", value=str(prod_minus_load), style="filled",
                       fillcolor="#ffffed")  # white color
        i += 1


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

def build_nodes_v2(g, nodes_prod_values: list):
    """nodes_prod_values is a list of tuples, (graphical_node_id, prod_cons_total_value)
        prod_cons_total_value is a float.
        If the value is positive then it is a Production, if negative it is a Consumption
    """
    # print("IN FUNCTION BUILD NODES V2222222222", nodes_prod_values)
    for data in nodes_prod_values:
        # print("data = ", data)
        i = int(data[0])
        if data[1] is None or data[1] == "XXX":
            prod_minus_load = 0.0  # It will end up as a white node
        else:
            prod_minus_load = data[1]
        # print("prod_minus_load = ", prod_minus_load)
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
    # print("\nWE ARE IN BUILD EDGES V2")
    substation_ids = sorted(list(substations_elements.keys()))
    # loops through each substation, and creates an edge from (
    for substation_id in substation_ids:
        # print("\nSUBSTATION ID = ", substation_id)
        for element in substations_elements[substation_id]:
            # print(element)
            origin = None
            extremity = None
            if isinstance(element, OriginLine):
                # origin = substation_id
                origin = int(substation_id_busbar_id_node_id_mapping[substation_id][element.busbar_id])
                extremity = int(element.end_substation_id)
                # check if extremity on busbar1, if it is,
                # check with the substation substation_id_busbar_id_node_id_mapping dic what "graphical" node it is
                # print("substations_elements[extremity] = ", substations_elements[extremity])
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
            # print("origin = ", origin)
            # print("extremity = ", extremity)
            # print("reported_flow = ", reported_flow)
            pen_width = fabs(reported_flow[0]) / 10.0
            if pen_width < 0.01:
                pen_width = 0.1
            # print(f"#################### Edge created : ({origin}, {extremity}), with flow = {reported_flow},"
            #       f" pen_width = {pen_width} >>>")
            if reported_flow[0] > 0:  # RED
                g.add_edge(origin, extremity, capacity=float(reported_flow[0]), xlabel=reported_flow[0], color="red",
                           penwidth="%.2f" % pen_width)
            else:  # BLUE
                g.add_edge(origin, extremity, capacity=float(reported_flow[0]), xlabel=reported_flow[0], color="blue",
                           penwidth="%.2f" % pen_width)
            g.add_edge(origin, extremity, capacity=float(reported_flow[0]), xlabel=reported_flow[0])

#TO DO: check is line is disconnected in new_obs

def score_changes_between_two_observations(ltc, old_obs, new_obs,nb_timestep_cooldown_line_param=0):
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
    boolean_line_cascading_disconnection = ((new_obs.time_before_cooldown_line-old_obs.time_before_cooldown_line)>nb_timestep_cooldown_line_param)



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
    line_id=0
    for old, new in zip(old_obs_lines_capacity_usage, new_obs_lines_capacity_usage):
        # preprocessing for score 3
        if (new > 1.05 * old) & (new > 1.0):  # if new > old > 1.0 it means it worsened an existing constraint
            boolean_overload_worsened.append(1)
        else:
            boolean_overload_worsened.append(0)

        # preprocessing for score 2
        if (old > 1.0) & (line_id in ltc) :  # if old was an overload:
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
        if (old > 1.0 > new) & (line_id in ltc):
            boolean_constraint_relieved.append(1)
        else:
            boolean_constraint_relieved.append(0)

        if old < 1.0 < new:
            boolean_overload_created.append(1)
        else:
            boolean_overload_created.append(0)

        line_id+=1

    boolean_overload_worsened = np.array(boolean_overload_worsened)
    boolean_constraint_30percent_relieved = np.array(boolean_constraint_30percent_relieved)
    boolean_constraint_relieved = np.array(boolean_constraint_relieved)
    boolean_overload_created = np.array(boolean_overload_created)

    redistribution_prod = np.sum(np.absolute(new_obs.prod_p - old_obs.prod_p))
    #redistribution_load = np.sum(np.absolute(new_obs.load_p - old_obs.load_p))#not exact in Grid2op if load are disconnected

    TotalProd=np.nansum(new_obs.prod_p)
    Losses=np.nansum(np.abs(new_obs.p_or+new_obs.p_ex))
    ExpectedNewLoad=TotalProd-Losses
    cut_load_percent=(np.sum(old_obs.load_p)-ExpectedNewLoad)/np.sum(old_obs.load_p)


    # ################################ END OF PREPROCESSING #################################
    # score 0 if no overloads were alleviated or if it resulted in some load shedding or production distribution.
    if old_number_of_overloads == 0:
        # print("return NaN: No overflow at initial state of grid")
        return float('nan')
    elif cut_load_percent > 0.01: # (boolean_overload_relieved == 0).all()
        # print("return 0: no overloads were alleviated or some load shedding occured.")
        return 0

    # score 1 if overload was relieved but another one appeared and got worse
    elif (boolean_constraint_relieved == 1).any() and ((boolean_overload_created == 1).any() or
                                                     (boolean_overload_worsened == 1).any() or (boolean_line_cascading_disconnection).any()):
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
