import numpy as np
from math import fabs
import networkx as nx

from alphaDeesp.core.simulation import Simulation
from alphaDeesp.core.elements import OriginLine, Consumption, Production, ExtremityLine


class Grid2opSimulation(Simulation):
    def get_layout(self):
        pass

    def get_substation_elements(self):
        pass

    def get_substation_to_node_mapping(self):
        pass

    def get_internal_to_external_mapping(self):
        pass

    def __init__(self, config, obs, action_space, ltc=9, plot_helper = None):
        super().__init__()
        self.obs = obs
        self.obs_linecut = None

        self.plot_helper = plot_helper
        self.action_space = action_space
        self.param_options = config
        print("Number of generators of the powergrid: {}".format(self.obs.n_gen))
        print("Number of loads of the powergrid: {}".format(self.obs.n_load))
        print("Number of powerline of the powergrid: {}".format(self.obs.n_line))
        print("Number of elements connected to each substations in the powergrid: {}".format(self.obs.sub_info))
        print("Total number of elements: {}".format(self.obs.dim_topo))
        self.internal_to_external_mapping = {}
        self.external_to_internal_mapping = {}
        self.substations_elements = {}

        self.topo = self.extract_topo_from_obs(self.obs)
        self.topo_linecut = None

        self.df = self.create_df(self.topo, ltc)
        #self.create_and_fill_internal_structures(self.obs, self.df)

    def create_and_fill_internal_structures(self, obs, df):
        """This function fills multiple structures:
        self.substation_elements, self.substation_to_node_mapping, self.internal_to_external_mapping
        @:arg observation, df"""
        # ################ PART I : fill self.internal_to_external_mapping
        substations_list = list(obs.name_sub)
        print(substations_list)

        # we create mapping from external representation to internal.
        for i, substation_id in enumerate(substations_list):
            self.internal_to_external_mapping[i] = substation_id
        if self.internal_to_external_mapping:
            self.external_to_internal_mapping = self.invert_dict_keys_values(self.internal_to_external_mapping)

        # ################ PART II : fill self.substation_elements
        for substation_id in self.internal_to_external_mapping.keys():
            print(substation_id)
            elements_array = []
            external_substation_id = self.internal_to_external_mapping[substation_id]
            objects = obs.get_obj_connect_to(substation_id=substation_id)

            for load_id in objects.loads_id:
                load_state = obs.state_of(load_id=load_id)
                elements_array.append(Consumption(load_state.bus, load_state.p))
            for gen_id in objects.generators_id:
                gen_state = obs.state_of(gen_id=gen_id)
                elements_array.append(Production(gen_state.bus, gen_state.p))
            for line_id in objects.lin:
                line_state = obs.state_of(line_id=line_id)
                orig = line_state.origin
                ext = line_state.extremity
                dest = orig.sub_id
                if dest == substation_id:
                    dest = ext.sub_id
                elements_array.append(OriginLine(orig.bus, dest, orig.p))
                elements_array.append(ExtremityLine(ext.bus, dest, ext.p))
            self.substations_elements[substation_id] = elements_array
            print(self.substations_elements)

    def extract_topo_from_obs(self, obs):
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
        # for line in ids:
        #     self.backend._disconnect_line(line)
        # self.backend.runpf()
        # new_flow = self.backend.get_line_flow()

        # Set action which disconects the specified lines (by ids)
        print({"set_line_status": [(id_, -1) for id_ in ids]})
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

    def build_powerflow_graph(self, mode = 'before_cutting'):
        """This function takes a Grid2op Observation and returns a NetworkX Graph"""
        g = nx.DiGraph()

        if mode == 'before_cutting':
            topo = self.topo
            obs = self.obs
        elif mode == 'after_cutting':
            topo = self.topo_linecut
            obs = self.obs_linecut
        else:
            print("Mode can be either 'before_cutting' or 'after_cutting' default has been set to 'before_cutting")
            topo = self.topo
            obs = self.obs

        # Get the id of lines that are disconnected from network
        lines_cut = np.argwhere(obs.line_status == False)[:,0]

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

    def plot_grid(self, before_removal = True, after_removal = False):
        if before_removal:
            fig_obs = self.plot_helper.plot_obs(self.obs, line_info = 'p')



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