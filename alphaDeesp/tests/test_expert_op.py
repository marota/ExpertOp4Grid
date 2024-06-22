import unittest
import configparser
import os, shutil
import numpy as np
from alphaDeesp.expert_operator import expert_operator
from alphaDeesp.main import generate_plot_folders
from alphaDeesp.core.grid2op.Grid2opObservationLoader import Grid2opObservationLoader
from alphaDeesp.core.grid2op.Grid2opSimulation import Grid2opSimulation

def build_sim():
    config = configparser.ConfigParser()
    config.read("./alphaDeesp/tests/resources_for_tests_grid2op/config_for_tests.ini")
    param_folder = "./alphaDeesp/tests/resources_for_tests_grid2op/l2rpn_2019_ltc_9"

    loader = Grid2opObservationLoader(param_folder)
    env, obs, action_space = loader.get_observation()
    observation_space = env.observation_space

    ltc=[9]
    chronicscenario=0
    timestep=0

    try:
        plot_base_folder = config["DEFAULT"]["outputPath"]  # Case there is a grid path given in config.ini
    except:  # Default load l2rpn_2019 in packages data
        print("No outputPath in config.ini: generating outputs in current folder")
        plot_base_folder = "output"
    plot_folder = generate_plot_folders(plot_base_folder, ltc,chronicscenario,timestep, config)

    sim = Grid2opSimulation(obs, action_space, observation_space, param_options=config["DEFAULT"], debug=False,
                                 ltc=[9],plot=True, plot_folder = plot_folder)
    return sim,env

def get_folder_empty(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


class TestExpertOp(unittest.TestCase):

    def test_expertOp(self):
        # test results: ranked_combinations, expert_system_results, actions
        sim,env=build_sim()

        graphs_folder=os.path.join(sim.plot_folder,"Base graph")
        plot_results_folder=os.path.join(sim.plot_folder,"Result graph")
        get_folder_empty(graphs_folder)
        get_folder_empty(plot_results_folder)
        expert_operator(sim, plot=True)#we test if plots are properly generated

        print("check if files generated")
        filenames=os.listdir(graphs_folder)

        assert len(filenames)==4
        assert np.any(["g_overflow" in filename for filename in filenames])
        assert np.any([".dot" in filename for filename in filenames])
        assert np.any(["g_pow" in filename for filename in filenames])
        assert np.any(["g_pow_prime" in filename for filename in filenames])

        filenames = os.listdir(plot_results_folder)
        assert len(filenames) == 42

