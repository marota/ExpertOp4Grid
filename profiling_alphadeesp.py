import cProfile
import pstats

import os
import io
import argparse
import configparser
import numpy as np

from alphaDeesp.core.printer import shell_print_project_header
from alphaDeesp.expert_operator import expert_operator

from alphaDeesp.core.alphadeesp import AlphaDeesp
from alphaDeesp.core.printer import Printer

from alphaDeesp.core.grid2op.Grid2opSimulation import Grid2opSimulation
from alphaDeesp.core.grid2op.Grid2opObservationLoader import Grid2opObservationLoader

### PARAMETERS

# Grid and chronic
#parameters_folder = "./alphaDeesp/ressources/parameters/l2rpn_2019"
#chronicscenario = "a"
# parameters_folder = "./alphaDeesp/ressources/parameters/rte_case14_realistic"
# chronicscenario = "000"
parameters_folder = r'C:\Users\nmegel\data_grid2op\l2rpn_wcci_2020'
chronicscenario = "Scenario_february_069"

# Other parameters to config
timestep = 100
difficulty = None
debug = False
config = {"DEFAULT":{
            "totalnumberofsimulatedtopos": 50,
            "numberofsimulatedtopospernode": 10,
            "maxUnusedLines": 3,
            "ratioToReconsiderFlowDirection": 0.75,
            "ratioToKeepLoop": 0.25,
            "ThersholdMinPowerOfLoop": 0.1,
            "ThresholdReportOfLine": 0.2}
        }
ltc = [9]
snapshot = False
plot_folder = ""

### EXECUTION

# Loading Grid2op object
loader = Grid2opObservationLoader(parameters_folder, difficulty = difficulty)
env, obs, action_space = loader.get_observation(chronic_scenario= chronicscenario, timestep=timestep)
observation_space = env.observation_space

# Loading Grid2op Simulator
sim = Grid2opSimulation(obs, action_space, observation_space, param_options=config["DEFAULT"], debug=debug,
                         ltc=ltc, plot=snapshot, plot_folder = plot_folder)

# Generating objects
custom_layout = sim.get_layout()
printer = None
df_of_g = sim.get_dataframe()
g_over = sim.build_graph_from_data_frame(ltc)
g_pow = sim.build_powerflow_graph_beforecut()
g_pow_prime = sim.build_powerflow_graph_aftercut()
simulator_data = {"substations_elements": sim.get_substation_elements(),
                  "substation_to_node_mapping": sim.get_substation_to_node_mapping(),
                  "internal_to_external_mapping": sim.get_internal_to_external_mapping()}

# Launching Alphadeesp core
print("Overflow at lines: "+str(np.where(sim.obs.rho>1)))
alphadeesp = AlphaDeesp(g_over, df_of_g, custom_layout, printer, simulator_data,sim.substation_in_cooldown, debug = debug)

# Launching Alphadeesp core with Profiler
# pr = cProfile.Profile()
# pr.enable()
# # Beginning of profiling
# alphadeesp = AlphaDeesp(g_over, df_of_g, custom_layout, printer, simulator_data,sim.substation_in_cooldown, debug = debug)
# # End of profiling
# pr.disable()
#
# # Writting profiling results
# s = io.StringIO()
# ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
# ps.strip_dirs().print_stats()
# with open('test.txt', 'w+') as f:
#     f.write(s.getvalue())

# ranked_combinations = alphadeesp.get_ranked_combinations()
