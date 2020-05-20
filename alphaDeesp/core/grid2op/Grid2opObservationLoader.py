import grid2op

class Grid2opObservationLoader():
    def __init__(self, parameter_folder):
        self.parameter_folder = parameter_folder
        self.env = grid2op.make(self.parameter_folder)

    def get_observation(self, timestep = 0):
        self.env.fast_forward_chronics(nb_timestep= timestep)
        obs = self.env.get_obs()
        backend = self.env.backend
        return obs, backend
