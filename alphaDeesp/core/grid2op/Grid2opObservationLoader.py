import os
import grid2op
from grid2op.Parameters import Parameters

class Grid2opObservationLoader:
    def __init__(self, parameter_folder, difficulty = None):
        self.parameter_folder = parameter_folder
        #self.custom_params = Parameters()
        #self.custom_params.NO_OVERFLOW_DISCONNECTION = False
        #self.custom_params.HARD_OVERFLOW_THRESHOLD = 9999999
        #self.custom_params.NB_TIMESTEP_OVERFLOW_ALLOWED = 9999999
        
        try:
            from lightsim2grid.LightSimBackend import LightSimBackend
            backend = LightSimBackend()
        except:
            from grid2op.Backend import PandaPowerBackend
            backend = PandaPowerBackend()
            print("You might need to install the LightSimBackend (provisory name) to gain massive speed up")

        if difficulty is None:
            # By default, easy difficulty is set through parameters
            custom_params = Parameters()
            custom_params.NO_OVERFLOW_DISCONNECTION = False
            custom_params.HARD_OVERFLOW_THRESHOLD = 9999999
            custom_params.NB_TIMESTEP_OVERFLOW_ALLOWED = 9999999
            self.env = grid2op.make(self.parameter_folder, backend=backend, param=custom_params)
        elif difficulty not in ["0","1","2","competition"]:
            raise ValueError("Difficulty in config.ini should be either None or 0,1,2 or competition")
        else:
            self.env = grid2op.make(self.parameter_folder, backend=backend, difficulty=difficulty)

    def get_observation(self, chronic_scenario = None, timestep = 0):

        # If an int is provided, chronic_scenario is string by default, so it has to be converted
        try:
            chronic_scenario = int(chronic_scenario)
            print("INFO - An integer has been provided as chronic scenario - looking for the chronic folder in this position")
        except:
            if chronic_scenario is None:
                print("INFO - No value has been provided for chronic scenario - the first chronic folder will be chosen")
            else:
                print("INFO - A string value has been provided as chronic scenario - searching for a chronic folder with name "+str(chronic_scenario))

        # Go to desired chronic scenario (if None, first scenario will be taken)
        if chronic_scenario is not None:
            if type(chronic_scenario) is str:
                found_id = self.search_chronic_num_from_name(chronic_scenario)
            elif type(chronic_scenario) is int:
                found_id = chronic_scenario
                scenario_name = self.search_chronic_name_from_num(found_id)
                print("INFO - the name of the loaded Grid2op scenario is : " + str(scenario_name))
            if found_id is not None:
                self.env.set_id(found_id)
                self.env.reset()
            else: # if name not found
                raise ValueError("Chronic scenario name: "+chronic_scenario+" not found in folder")


        # Method fast_forward_chronics doesnt work properly
        if timestep >0:
            self.env.fast_forward_chronics(nb_timestep= timestep)
        obs = self.env.get_obs()
        
        # So we prefer just doing nothing during timesteps to skip
        # if timestep == 0:
        #     obs = self.env.get_obs()
        # for i in range(1,timestep+1):
        #     obs, reward, done, info = self.env.step(self.env.action_space())
        #     if done:
        #         raise ValueError("A Game Over occured at timestep number "+str(i)+" while acting with donothing actions")

        # Get backend which will be used to simulate network after modifying its configuration
        #backend = self.env.backend

        # Get action space to enable action generation for simulation
        action_space = self.env.action_space

        return self.env, obs, action_space

    def search_chronic_name_from_num(self, num):
        for id, sp in enumerate(self.env.chronics_handler.real_data.subpaths):
            chronic_scenario = os.path.basename(sp)
            if id == num:
                break
        return chronic_scenario

    def search_chronic_num_from_name(self, scenario_name):
        found_id = None
        # Search scenario with provided name
        for id, sp in enumerate(self.env.chronics_handler.real_data.subpaths):
            sp_end = os.path.basename(sp)
            if sp_end == scenario_name:
                found_id = id
        return found_id



