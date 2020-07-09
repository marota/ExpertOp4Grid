import grid2op
from grid2op.Parameters import Parameters

import alphaDeesp.main as expertOp4grid
from alphaDeesp.core.grid2op.Grid2opSimulation import Grid2opSimulation


# =============================================================================================================
# Some parameters for the grid configuration
grid_folder = "./alphaDeesp/ressources/parameters/l2rpn_2019_ltc_9"
timestep = 4

# Some configuration for ExpertOp4Grid
config = {"totalnumberofsimulatedtopos":30,
          "numberofsimulatedtopospernode":10,
          "maxUnusedLines":3,
          "ratioToReconsiderFlowDirection": 0.75,
          "ratioToKeepLoop":0.25,
          "ThersholdMinPowerOfLoop":0.1,
          "ThresholdReportOfLine":0.2
          }
lines_to_cut = [9]
plot = False
debug = False

# =============================================================================================================
# Load a Grid2op Environment
custom_params = Parameters()
custom_params.NO_OVERFLOW_DISCONNECTION = True
env = grid2op.make(grid_folder, param = custom_params)

# Go to a timestep by doing nothing
observation_space = env.observation_space
action_space = env.action_space
do_nothing = action_space()
obs = None
if timestep == 0:
    obs = env.get_obs()
for i in range(1,timestep+1):
    obs, reward, done, info = env.step(do_nothing)

# =============================================================================================================
# Call ExpertOp4Grid
simulator = Grid2opSimulation(obs, action_space, observation_space, param_options=config, debug = debug, ltc=lines_to_cut)
ranked_combinations, expert_system_results = expertOp4grid.expert_operator(simulator, plot=plot, debug = debug)
# =============================================================================================================


