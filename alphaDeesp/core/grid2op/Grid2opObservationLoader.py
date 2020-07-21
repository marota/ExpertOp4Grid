import grid2op
from grid2op.Parameters import Parameters

class Grid2opObservationLoader:
    def __init__(self, parameter_folder, difficulty = 0):
        self.parameter_folder = parameter_folder
        self.custom_params = Parameters()
        self.custom_params.NO_OVERFLOW_DISCONNECTION = False
        self.custom_params.HARD_OVERFLOW_THRESHOLD = 9999999
        self.custom_params.NB_TIMESTEP_OVERFLOW_ALLOWED = 9999999
        
        try:
            from lightsim2grid.LightSimBackend import LightSimBackend
            backend = LightSimBackend()
        except:
            from grid2op.Backend import PandaPowerBackend
            backend = PandaPowerBackend()
            print("You might need to install the LightSimBackend (provisory name) to gain massive speed up")
        
        self.env = grid2op.make(self.parameter_folder,backend=backend, param = self.custom_params)
        #self.env = grid2op.make(self.parameter_folder, backend=backend, difficulty=difficulty)


    def get_observation(self, chronic_scenario = 0, timestep = 0):

        # Go to desired chronic scenario
        if chronic_scenario > 0:
            self.env.set_id(chronic_scenario)
            self.env.reset()

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

