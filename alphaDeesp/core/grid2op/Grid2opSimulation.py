import numpy as np

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

    def __init__(self, config, obs, backend, ltc=9, plot_helper = None):
        super().__init__()
        self.obs = obs
        self.plot_helper = plot_helper
        self.backend = backend
        self.param_options = config
        print("Number of generators of the powergrid: {}".format(self.obs.n_gen))
        print("Number of loads of the powergrid: {}".format(self.obs.n_load))
        print("Number of powerline of the powergrid: {}".format(self.obs.n_line))
        print("Number of elements connected to each substations in the powergrid: {}".format(self.obs.sub_info))
        print("Total number of elements: {}".format(self.obs.dim_topo))
        self.internal_to_external_mapping = {}
        self.external_to_internal_mapping = {}
        self.substations_elements = {}

        topo = self.extract_topo_from_obs()
        self.df = self.create_df(topo, ltc)
        self.create_and_fill_internal_structures(self.obs, self.df)

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

    def extract_topo_from_obs(self):
        """This function, takes an obs an returns a dict with all topology information"""
        d = {
            "edges": {},
            "nodes": {}
        }
        nsub = self.obs.n_sub
        nodes_ids = list(range(nsub))
        idx_or = self.obs.line_or_to_subid
        idx_ex = self.obs.line_ex_to_subid
        prods_ids = self.obs.gen_to_subid
        loads_ids = self.obs.load_to_subid
        are_prods = [node_id in prods_ids for node_id in nodes_ids]
        are_loads = [node_id in loads_ids for node_id in nodes_ids]
        current_flows = self.obs.p_or  # Flow at the origin of power line is taken

        # Repartition of prod and load in substations
        prods_values = self.obs.prod_p
        loads_values = self.obs.load_p
        gens_ordered_by_subid = np.argsort(self.obs.gen_to_subid)
        loads_ordered_by_subid = np.argsort(self.obs.load_to_subid)
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
        for line in ids:
            self.backend._disconnect_line(line)
        self.backend.runpf()
        new_flow = self.backend.get_line_flow()

        # self.g_pow_prime = self.build_powerflow_graph(self.obs_cutted)

        print("Lines in overflow after cutting line " + str(ids))
        print(np.where(self.backend.get_line_overflow()))

        return new_flow

    def build_powerflow_graph(self, raw_data):
        pass

    def plot_grid(self, before_removal = True, after_removal = False):
        if before_removal:
            fig_obs = self.plot_helper.plot_obs(self.obs, line_info = 'p')
