import grid2op
from grid2op.PlotGrid import PlotMatplot

class Grid2opObservationLoader():
    def __init__(self, parameter_folder):
        self.parameter_folder = parameter_folder
        self.env = grid2op.make(self.parameter_folder)

    def get_observation(self, timestep = 5):
        # Method fast_forward_chronics doesnt work properly
        # self.env.fast_forward_chronics(nb_timestep= timestep)
        # obs = self.env.get_obs()

        # So we prefer just doing nothing during timesteps to skip
        if timestep == 0:
            obs = self.env.get_obs()
        for i in range(1,timestep+1):
            obs, reward, done, info = self.env.step(self.env.action_space())
        # get backend which will be used to simulate network after modifying its configuration
        backend = self.env.backend
        return obs, backend

    def get_plot_helper(self):
        plot_helper = PlotMatplot(self.env.observation_space)
        return plot_helper
