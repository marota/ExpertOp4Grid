import pypownet.environment
from pypownet.agent import *
from pypownet.environment import ElementType

class PypownetObservationLoader:
    def __init__(self, param_folder):

        if not param_folder or param_folder is None:
            raise AttributeError("\nThe parameters folder for Pypownet is empty or None.")
        parameters_folder = param_folder

        game_level = "level0"
        chronic_looping_mode = 'natural'
        chronic_starting_id = 0
        game_over_mode = 'easy'
        without_overflow_cuttof = True

        self.env = pypownet.environment.RunEnv(parameters_folder=parameters_folder, game_level=game_level,
                                                       chronic_looping_mode=chronic_looping_mode,
                                                       start_id=chronic_starting_id,
                                                       game_over_mode=game_over_mode,
                                                       without_overflow_cutoff=without_overflow_cuttof)



    def get_observation(self, timestep = 0):
        # Get action space
        action_space = self.env.action_space

        # Create do_nothing action.
        action_do_nothing = action_space.get_do_nothing_action()

        # Run one step in the environment
        raw_simulated_obs = self.env.simulate(action_do_nothing)
        obs = self.env.observation_space.array_to_observation(raw_simulated_obs[0])

        # TODO: go to timestep

        return self.env, obs, action_space

