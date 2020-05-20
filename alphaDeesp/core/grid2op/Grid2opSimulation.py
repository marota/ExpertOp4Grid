import numpy as np

from alphaDeesp.core.simulation import Simulation
from alphaDeesp.core.grid2op.Grid2opObservationLoader import Grid2opObservationLoader
import grid2op
from grid2op.Chronics import ChangeNothing

class Grid2opSimulation(Simulation):
    def get_layout(self):
        pass

    def get_substation_elements(self):
        pass

    def get_substation_to_node_mapping(self):
        pass

    def get_internal_to_external_mapping(self):
        pass

    def __init__(self, param_options = None, parameter_folder = None, mode = 'manuel', ltc = 9):
        super().__init__()
        self.param_options = param_options
        self.parameter_folder = parameter_folder
        self.mode = mode
        self.init_timestep = 0  # TODO: make this timestep configurable

        if mode == "manuel":
            loader = Grid2opObservationLoader(self.parameter_folder)
            self.obs, self.backend =  loader.get_observation(timestep=self.init_timestep)
            self.obs_cutted = None
        elif mode == "auto":
            print("Mode Auto still to be developed")
            # TODO: load observation and backend from Agent Environment for auto mode

        print("Number of generators of the powergrid: {}".format(self.obs.n_gen))
        print("Number of loads of the powergrid: {}".format(self.obs.n_load))
        print("Number of powerline of the powergrid: {}".format(self.obs.n_line))
        print("Number of elements connected to each substations in the powergrid: {}".format(self.obs.sub_info))
        print("Total number of elements: {}".format(self.obs.dim_topo))

        topo = self.extract_topo_from_obs()
        self.df = self.create_df(topo, ltc)


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
        prods_values = self.obs.prod_p
        loads_values = self.obs.load_p
        current_flows = self.obs.p_or # Flow at the origin of power line is taken
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
        self.backend._disconnect_line(ids)
        self.backend .runpf()
        new_flow = self.backend.get_line_flow()

        # self.g_pow_prime = self.build_powerflow_graph(self.obs_cutted)

        print("Lines in overflow after cutting line "+str(ids))
        print(np.where(self.backend.get_line_overflow()))

        return new_flow

    def build_powerflow_graph(self, raw_data):
        pass
