from alphaDeesp.core.simulation import Simulation


class Grid2opSimulation(Simulation):
    def get_layout(self):
        pass

    def get_substation_elements(self):
        pass

    def get_substation_to_node_mapping(self):
        pass

    def get_internal_to_external_mapping(self):
        pass

    def __init__(self, parameter_folder):
        super().__init__()
        self.parameter_fodler = parameter_folder

    def build_powerflow_graph(self, raw_data):
        pass

    def cut_lines_and_recomputes_flows(self, ids: list):
        pass
