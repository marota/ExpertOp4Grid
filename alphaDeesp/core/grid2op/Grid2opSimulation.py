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

    def __init__(self, parameter_folder, mode = 'manuel'):
        super().__init__()
        self.parameter_folder = parameter_folder
        self.mode = mode
        self.init_timestep = 15  # TODO: make this timestep configurable

        if mode == "manuel":
            loader = Grid2opObservationLoader(self.parameter_folder)
            self.obs =  loader.get_observation(timestep=self.init_timestep)
        elif mode == "auto":
            print("Mode Auto still to be developed")
        # TODO: load an agent observation for auto mode

        print("Number of generators of the powergrid: {}".format(self.obs.n_gen))
        print("Number of loads of the powergrid: {}".format(self.obs.n_load))
        print("Number of powerline of the powergrid: {}".format(self.obs.n_line))
        print("Number of elements connected to each substations in the powergrid: {}".format(self.obs.sub_info))
        print("Total number of elements: {}".format(self.obs.dim_topo))

    def build_powerflow_graph(self, raw_data):
        pass

    def cut_lines_and_recomputes_flows(self, ids: list):
        pass
