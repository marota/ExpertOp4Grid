import grid2op
from grid2op.Parameters import Parameters

class Grid2opObservationLoader:
    def __init__(self, parameter_folder):
        self.parameter_folder = parameter_folder
        self.custom_params = Parameters()
        self.custom_params.NO_OVERFLOW_DISCONNECTION = True
        self.env = grid2op.make(self.parameter_folder, param = self.custom_params)


    def get_observation(self, chronic_scenario = "", timestep = 0):
        # Method fast_forward_chronics doesnt work properly
        # self.env.fast_forward_chronics(nb_timestep= timestep)
        # obs = self.env.get_obs()

        # TODO: chopper le bon scenario

        # So we prefer just doing nothing during timesteps to skip
        if timestep == 0:
            obs = self.env.get_obs()
        for i in range(1,timestep+1):
            obs, reward, done, info = self.env.step(self.env.action_space())

        # Get backend which will be used to simulate network after modifying its configuration
        #backend = self.env.backend

        # Get action space to enable action generation for simulation
        action_space = self.env.action_space
        return self.env, obs, action_space

