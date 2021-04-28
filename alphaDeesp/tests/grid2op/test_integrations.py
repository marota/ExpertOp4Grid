import configparser

from alphaDeesp.core.alphadeesp import AlphaDeesp
import pandas as pd
from pathlib import Path
import ast
import numpy as np
import pandas as pd
import time

from alphaDeesp.core.grid2op.Grid2opObservationLoader import Grid2opObservationLoader
from alphaDeesp.core.grid2op.Grid2opSimulation import Grid2opSimulation


custom_layout = [(-280, -81), (-100, -270), (366, -270), (366, -54), (-64, -54), (-64, 54), (366, 0),
                 (438, 0), (326, 54), (222, 108), (79, 162), (-152, 270), (-64, 270), (222, 216),
                 (-280, -151), (-100, -340), (366, -340), (390, -110), (-14, -104), (-184, 54), (400, -80),
                 (438, 100), (326, 140), (200, 8), (79, 12), (-152, 170), (-70, 200), (222, 200)]


def build_sim(ltc, param_folder, config_file = "./alphaDeesp/tests/resources_for_tests_grid2op/config_for_tests.ini",
              chronic_scenario = None, timestep = 0,modified_thermal_Limit=None):
    config = configparser.ConfigParser()
    config.read(config_file)

    loader = Grid2opObservationLoader(param_folder)
    env, obs, action_space = loader.get_observation(timestep = timestep, chronic_scenario=chronic_scenario)
    if(modified_thermal_Limit):
        env._thermal_limit_a[ltc]=modified_thermal_Limit
    observation_space = env.observation_space
    sim = Grid2opSimulation(obs, action_space, observation_space, param_options=config["DEFAULT"], debug=False,
                            ltc=[ltc])
                            #, plot=True, plot_folder="./alphaDeesp/tests/output_tempo")
    return sim


def test_save_red_dataframe():
    """Simple test, where a pandas.DataFrame gets saved, read and compared"""

    df = pd.DataFrame([[10.123123, -12.123123], [20.0123210, 91.1111]])
    df.to_csv("./test_read.csv")

    path_file = Path.cwd() / "test_read.csv"
    saved_df = pd.read_csv(path_file, index_col=0)

    print("They two dataframes are equal: ", are_dataframes_equal(df, saved_df))


def test_round_random_tests():
    arr = [12.312312, 11.094329]

    new_arr = [round(elem, 2) for elem in arr]
    print(new_arr)
    assert (new_arr == [12.31, 11.09])


def are_dataframes_equal(df1, df2):
    """Function that compares row by row, then field after field, 2 dataframes.
    If each value is identical, then they are equal

    Returns True if identical, False otherwise
    """

    generated_bag = []
    saved_bag = []

    for i, row in df1.iterrows():
        generated_bag.append(list(row))

    for i, row in df2.iterrows():
        saved_bag.append(list(row))

    # to properly compare the results, we round all floats to .XX 2 decimals
    generated_bag_rounded = []
    saved_bag_rounded = []

    for row in generated_bag:
        row_tab = []

        for elem in row:
            # print("elem = ", elem)
            # print("type = ", type(elem))
            if isinstance(elem, float):
                elem = round(elem, 2)

            if type(elem).__module__ == np.__name__:  # Has been identified as numpy array : has to be converted in list
                elem = elem.tolist()
            if type(elem) != list:
                if pd.isnull(elem): # NaN cant be equal to anything, even NaN themselves
                    elem = str(elem)
            row_tab.append(elem)

        generated_bag_rounded.append(row_tab)

    print("SEPARATOR ============================================================================================")

    for row in saved_bag:
        row_tab = []

        for elem in row:
            if isinstance(elem, float):
                elem = round(elem, 2)
            elif isinstance(elem, str):
                # string evaluation are used for arrays in string form, they get transformed into arrays
                elem = ast.literal_eval(elem)

            if type(elem).__module__ == np.__name__:  # Has been identified as numpy array : has to be converted in list
                elem = elem.tolist()
            if type(elem) != list:
                if pd.isnull(elem): # NaN cant be equal to anything, even NaN themselves
                    elem = str(elem)

            row_tab.append(elem)


        saved_bag_rounded.append(row_tab)

    print(generated_bag_rounded)
    print(saved_bag_rounded)

    return generated_bag_rounded == saved_bag_rounded


def test_integration_dataframe_results_with_line_9_cut():
    """
    In the initial state of the network, all substations are on busbar1
    Line 9 is between Node 4 and 5 [internal node ID indexing]
    Test
    """
    # import os
    # os.chdir('../../../')

    ltc = 9
    param_folder = "./alphaDeesp/tests/resources_for_tests_grid2op/l2rpn_2019_ltc_9"

    sim = build_sim(ltc, param_folder)
    df_of_g = sim.get_dataframe()
    g_over = sim.build_graph_from_data_frame([ltc])
    g_pow = sim.build_powerflow_graph_beforecut()
    g_pow_prime = sim.build_powerflow_graph_aftercut()
    simulator_data = {"substations_elements": sim.get_substation_elements(),
                      "substation_to_node_mapping": sim.get_substation_to_node_mapping(),
                      "internal_to_external_mapping": sim.get_internal_to_external_mapping()}
    # create AlphaDeesp
    alphadeesp = AlphaDeesp(g_over, df_of_g, simulator_data=simulator_data)
    ranked_combinations = alphadeesp.get_ranked_combinations()
    expert_system_results, actions = sim.compute_new_network_changes(ranked_combinations)

    #expert_system_results.to_csv("alphaDeesp/tests/resources_for_tests_grid2op/END_RESULT_DATAFRAME_G2OP_LTC9_9CAPA_230_generated.csv")

    path_to_saved_end_result_dataframe = \
        Path.cwd() / "alphaDeesp/tests/resources_for_tests_grid2op/END_RESULT_DATAFRAME_G2OP_LTC9_9CAPA_230.csv"

    #expert_system_results.to_csv(path_to_saved_end_result_dataframe)
    saved_df = pd.read_csv(path_to_saved_end_result_dataframe, index_col=0)

    # List understandable format
    saved_df["Internal Topology applied "] = saved_df["Internal Topology applied "].str.replace(" ", ",")

    #print("The two dataframes are equal: ", are_dataframes_equal(expert_system_results, saved_df))
    assert are_dataframes_equal(expert_system_results, saved_df)


def test_integration_dataframe_results_with_line_8_cut():
    """
    In the initial state of the network, all substations are on busbar1
    Line 8 is between Node 4 and 5 [internal node ID indexing]
    Test
    """

    # import os
    # os.chdir('../../../')

    ltc = 8
    param_folder = "./alphaDeesp/tests/resources_for_tests_grid2op/l2rpn_2019_ltc_8"

    sim = build_sim(ltc, param_folder)
    df_of_g = sim.get_dataframe()
    g_over = sim.build_graph_from_data_frame([ltc])
    g_pow = sim.build_powerflow_graph_beforecut()
    g_pow_prime = sim.build_powerflow_graph_aftercut()
    simulator_data = {"substations_elements": sim.get_substation_elements(),
                      "substation_to_node_mapping": sim.get_substation_to_node_mapping(),
                      "internal_to_external_mapping": sim.get_internal_to_external_mapping()}
    # create AlphaDeesp
    alphadeesp = AlphaDeesp(g_over, df_of_g, simulator_data=simulator_data)
    ranked_combinations = alphadeesp.get_ranked_combinations()
    expert_system_results, actions = sim.compute_new_network_changes(ranked_combinations)

    path_to_saved_end_result_dataframe = \
        Path.cwd() / "alphaDeesp/tests/resources_for_tests_grid2op/END_RESULT_DATAFRAME_G2OP_LTC8_8CAPA_88.csv"

    # expert_system_results.to_csv(path_to_saved_end_result_dataframe)
    saved_df = pd.read_csv(path_to_saved_end_result_dataframe, index_col=0)

    # List understandable format
    saved_df["Internal Topology applied "] = saved_df["Internal Topology applied "].str.replace(" ", ",")

    #print("The two dataframes are equal: ", are_dataframes_equal(expert_system_results, saved_df))
    assert are_dataframes_equal(expert_system_results, saved_df)


def test_integration_dataframe_results_with_modified_substation4():
    """
    In the initial state of the network, all substations are on busbar1
    Line 9 is between Node 4 and 5 [internal node ID indexing]
    Test
    """

    # import os
    # os.chdir('../../../')

    timestep = 5
    ltc = 8
    param_folder = "./alphaDeesp/tests/resources_for_tests_grid2op/l2rpn_2019_ltc_8_modify_substation_4"
    config = configparser.ConfigParser()
    config.read("./alphaDeesp/tests/resources_for_tests_grid2op/config_for_tests.ini")

    ## Read Grid2op environment at timestep
    loader = Grid2opObservationLoader(param_folder)
    env, obs, action_space = loader.get_observation(timestep=timestep)

    ## Modify buses
    action = action_space({"set_bus": {'lines_ex_id': [(1, 2)], "lines_or_id": [(9, 2)]}})
    new_obs, reward, done, info = env.step(action)

    ## Build simulator and generate objects for alphadeesp
    sim = Grid2opSimulation(new_obs, action_space, env.observation_space, param_options=config["DEFAULT"], debug=False,
                            ltc=[ltc])
    df_of_g = sim.get_dataframe()
    g_over = sim.build_graph_from_data_frame([ltc])
    g_pow = sim.build_powerflow_graph_beforecut()
    g_pow_prime = sim.build_powerflow_graph_aftercut()
    simulator_data = {"substations_elements": sim.get_substation_elements(),
                      "substation_to_node_mapping": sim.get_substation_to_node_mapping(),
                      "internal_to_external_mapping": sim.get_internal_to_external_mapping()}

    ## Launch AlphaDeesp and get expert results
    alphadeesp = AlphaDeesp(g_over, df_of_g, simulator_data=simulator_data)
    ranked_combinations = alphadeesp.get_ranked_combinations()
    expert_system_results, actions = sim.compute_new_network_changes(ranked_combinations)
    # =============

    # Read desired results
    path_to_saved_end_result_dataframe = \
        Path.cwd() / "alphaDeesp/tests/resources_for_tests_grid2op/END_RESULT_DATAFRAME_G2OP_MODIFIED_SUBSTATION4.csv"
    #expert_system_results.to_csv(path_to_saved_end_result_dataframe)
    saved_df = pd.read_csv(path_to_saved_end_result_dataframe, index_col=0)

    ## Properly compare the two dataframes
    # List understandable format
    saved_df["Internal Topology applied "] = saved_df["Internal Topology applied "].str.replace(" ", ",")
    #print("The two dataframes are equal: ", are_dataframes_equal(expert_system_results, saved_df))
    assert are_dataframes_equal(expert_system_results, saved_df)

def test_integration_dataframe_results_with_case_14_realistic():
    """
    In the initial state of the network, all substations are on busbar1
    Test
    """

    # import os
    # os.chdir('../../../')

    ltc = 4
    chronic_scenario = "000"
    timestep = 518
    param_folder = "./alphaDeesp/ressources/parameters/rte_case14_realistic" # We go directly in the folder to avoid double storing of "heavy" data
    config_file = "./alphaDeesp/tests/resources_for_tests_grid2op/config_for_tests.ini"

    sim = build_sim(ltc, param_folder, config_file = config_file, timestep=timestep, chronic_scenario=chronic_scenario)
    df_of_g = sim.get_dataframe()
    g_over = sim.build_graph_from_data_frame([ltc])
    g_pow = sim.build_powerflow_graph_beforecut()
    g_pow_prime = sim.build_powerflow_graph_aftercut()

    simulator_data = {"substations_elements": sim.get_substation_elements(),
                      "substation_to_node_mapping": sim.get_substation_to_node_mapping(),
                      "internal_to_external_mapping": sim.get_internal_to_external_mapping()}
    # create AlphaDeesp
    alphadeesp = AlphaDeesp(g_over, df_of_g, simulator_data=simulator_data)
    ranked_combinations = alphadeesp.get_ranked_combinations()
    expert_system_results, actions = sim.compute_new_network_changes(ranked_combinations)

    # expert_system_results.to_csv("alphaDeesp/tests/resources_for_tests_grid2op/END_RESULT_DATAFRAME_G2OP_CASE14_REALISTIC_generated.csv")

    path_to_saved_end_result_dataframe = \
        Path.cwd() / "alphaDeesp/tests/resources_for_tests_grid2op/END_RESULT_DATAFRAME_G2OP_CASE14_REALISTIC.csv"

    saved_df = pd.read_csv(path_to_saved_end_result_dataframe, index_col=0)

    # List understandable format
    saved_df["Internal Topology applied "] = saved_df["Internal Topology applied "].str.replace(" ", ",")

    #print("The two dataframes are equal: ", are_dataframes_equal(expert_system_results, saved_df))
    assert are_dataframes_equal(expert_system_results, saved_df)

def test_integration_dataframe_results_no_hubs():
    """
    In the initial state of the network, all substations are on busbar1
    No hubs are detected by alphadeesp - result should be an empty dataframe
    Test
    """

    # import os
    # os.chdir('../../../')

    ltc = 9
    chronic_scenario = "i"
    timestep = 1
    param_folder = "./alphaDeesp/tests/resources_for_tests_grid2op/l2rpn_2019_nohubs"
    config_file = "./alphaDeesp/tests/resources_for_tests_grid2op/config_for_tests.ini"

    sim = build_sim(ltc, param_folder, config_file = config_file, timestep=timestep, chronic_scenario=chronic_scenario)
    df_of_g = sim.get_dataframe()
    g_over = sim.build_graph_from_data_frame([ltc])
    g_pow = sim.build_powerflow_graph_beforecut()
    g_pow_prime = sim.build_powerflow_graph_aftercut()

    simulator_data = {"substations_elements": sim.get_substation_elements(),
                      "substation_to_node_mapping": sim.get_substation_to_node_mapping(),
                      "internal_to_external_mapping": sim.get_internal_to_external_mapping()}
    # create AlphaDeesp
    alphadeesp = AlphaDeesp(g_over, df_of_g, simulator_data=simulator_data)
    ranked_combinations = alphadeesp.get_ranked_combinations()
    expert_system_results, actions = sim.compute_new_network_changes(ranked_combinations)

    # expert_system_results.to_csv("alphaDeesp/tests/resources_for_tests_grid2op/END_RESULT_DATAFRAME_G2OP_NO_HUBS_generated.csv")
    # expert_system_results.to_csv(r'D:\RTE\ExpertOp4Grid\5 - Résultats\2021_Timestep0_errors\generated_df.csv')

    path_to_saved_end_result_dataframe = \
        Path.cwd() / "alphaDeesp/tests/resources_for_tests_grid2op/END_RESULT_DATAFRAME_G2OP_NO_HUBS.csv"

    saved_df = pd.read_csv(path_to_saved_end_result_dataframe, index_col=0)

    # List understandable format
    saved_df["Internal Topology applied "] = saved_df["Internal Topology applied "].str.replace(" ", ",")

    #print("The two dataframes are equal: ", are_dataframes_equal(expert_system_results, saved_df))
    assert are_dataframes_equal(expert_system_results, saved_df)

def test_integration_l2rpn_wcci_2020_computation_time():
    """
    In the initial state of the network, all substations are on busbar1
    Test
    """

    # import os
    # os.chdir('../../../')

    # Time threshold
    max_elapsed_time = 60 # seconds

    # Configuration
    ltc = 13
    chronic_scenario = "Scenario_february_069"
    timestep = 100
    param_folder = "./alphaDeesp/tests/resources_for_tests_grid2op/l2rpn_wcci_2020"
    config_file = "./alphaDeesp/tests/resources_for_tests_grid2op/config_for_tests.ini"
    sim = build_sim(ltc, param_folder, config_file = config_file, timestep=timestep, chronic_scenario=chronic_scenario)

    # Starting time
    start = time.time()

    # Simulation objects
    df_of_g = sim.get_dataframe()
    g_over = sim.build_graph_from_data_frame([ltc])
    g_pow = sim.build_powerflow_graph_beforecut()
    g_pow_prime = sim.build_powerflow_graph_aftercut()
    simulator_data = {"substations_elements": sim.get_substation_elements(),
                      "substation_to_node_mapping": sim.get_substation_to_node_mapping(),
                      "internal_to_external_mapping": sim.get_internal_to_external_mapping()}

    # create AlphaDeesp
    printer = None
    custom_layout = sim.get_layout()
    alphadeesp = AlphaDeesp(g_over, df_of_g, custom_layout, printer, simulator_data, sim.substation_in_cooldown, debug=False)

    # End time
    elapsed_time = time.time() - start
    print("the computation time is: " + str(elapsed_time))
    assert (elapsed_time <= max_elapsed_time)

def test_double_lines_wcci_2020():
    """
    In the initial state of the network, all substations are on busbar1
    Test
    """

    # import os
    # os.chdir('../../../')

    # Time threshold
    max_elapsed_time = 60 # seconds

    # Configuration
    ltc = 27
    chronic_scenario = "Scenario_february_069"
    timestep = 100
    param_folder = "./alphaDeesp/tests/resources_for_tests_grid2op/l2rpn_wcci_2020"
    config_file = "./alphaDeesp/tests/resources_for_tests_grid2op/config_for_tests.ini"

    modified_thermal_Limit=150#the flow in the line is about 158amps
    sim = build_sim(ltc, param_folder, config_file = config_file, timestep=timestep, chronic_scenario=chronic_scenario,modified_thermal_Limit=modified_thermal_Limit)

    # Starting time
    start = time.time()

    # Simulation objects
    df_of_g = sim.get_dataframe()
    g_over = sim.build_graph_from_data_frame([ltc])

    if(len(df_of_g)!=g_over.number_of_edges()):
        print("some edges were not properly added to the graph")
        assert(len(df_of_g)==g_over.number_of_edges())
    g_pow = sim.build_powerflow_graph_beforecut()
    g_pow_prime = sim.build_powerflow_graph_aftercut()
    simulator_data = {"substations_elements": sim.get_substation_elements(),
                      "substation_to_node_mapping": sim.get_substation_to_node_mapping(),
                      "internal_to_external_mapping": sim.get_internal_to_external_mapping()}

    # create AlphaDeesp
    printer = None
    custom_layout = sim.get_layout()
    alphadeesp = AlphaDeesp(g_over, df_of_g, custom_layout, printer, simulator_data, sim.substation_in_cooldown, debug=False)

    # End time

    print("AlphaDeesp succeeded for an overflow graph with double lines")

