import configparser
from alphaDeesp.core.printer import Printer
from alphaDeesp.core.pypownet.PypownetObservationLoader import PypownetObservationLoader
from alphaDeesp.core.pypownet.PypownetSimulation import PypownetSimulation
from alphaDeesp.core.alphadeesp import AlphaDeesp
import pandas as pd
from pathlib import Path
import ast


custom_layout = [(-280, -81), (-100, -270), (366, -270), (366, -54), (-64, -54), (-64, 54), (366, 0),
                 (438, 0), (326, 54), (222, 108), (79, 162), (-152, 270), (-64, 270), (222, 216),
                 (-280, -151), (-100, -340), (366, -340), (390, -110), (-14, -104), (-184, 54), (400, -80),
                 (438, 100), (326, 140), (200, 8), (79, 12), (-152, 170), (-70, 200), (222, 200)]


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
                row_tab.append(round(elem, 2))
            else:
                row_tab.append(elem)

        generated_bag_rounded.append(row_tab)

    print("SEPARATOR ============================================================================================")

    for row in saved_bag:
        row_tab = []

        for elem in row:
            # print("elem = ", elem)
            # print("type = ", type(elem))
            if isinstance(elem, float):
                row_tab.append(round(elem, 2))
            elif isinstance(elem, str):
                # string evaluation are used for arrays in string form, they get transformed into arrays
                row_tab.append(ast.literal_eval(elem))
            else:
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

    #import os
    #os.chdir('../../../')

    # Some manual configuration
    line_to_cut = [9]
    timestep = 0
    debug = False
    param_folder = "./alphaDeesp/tests/ressources_for_tests/default14_static_line9"

    # read config file and parameters folder for Pypownet
    config = configparser.ConfigParser()
    config.read("./alphaDeesp/tests/ressources_for_tests/config_for_tests.ini")

    # run Pypownet
    loader = PypownetObservationLoader(param_folder)
    env, obs, action_space = loader.get_observation(timestep=timestep)
    sim = PypownetSimulation(
        env, obs, action_space, param_options = config["DEFAULT"], debug = debug, ltc = line_to_cut
    )

    # create NetworkX graph g_over, and DataFrame df_of_g
    df_of_g = sim.get_dataframe()
    g_over = sim.build_graph_from_data_frame(line_to_cut)

    simulator_data = {"substations_elements": sim.substations_elements,
                      "substation_to_node_mapping": sim.substation_to_node_mapping,
                      "internal_to_external_mapping": sim.internal_to_external_mapping}

    # create AlphaDeesp
    alphadeesp = AlphaDeesp(g_over, df_of_g, simulator_data=simulator_data)

    ranked_combinations = alphadeesp.get_ranked_combinations()

    expert_system_results = sim.compute_new_network_changes(ranked_combinations)
    # This removes the first XXX line (used to construct initial dataframe structure)
    expert_system_results = expert_system_results.drop(0, axis=0)

    # print("--------------------------------------------------------------------------------------------")
    # print("----------------------------------- END RESULT DATAFRAME -----------------------------------")
    # print("--------------------------------------------------------------------------------------------")
    # print(expert_system_results)

    path_to_saved_end_result_dataframe = \
        Path.cwd() / "alphaDeesp/tests/ressources_for_tests/END_RESULT_DATAFRAME_for_test_with_line_9_cut.csv"

    #expert_system_results.to_csv("./alphaDeesp/tests/ressources_for_tests/END_RESULT_DATAFRAME_for_test_with_line_9_cut.csv")

    saved_df = pd.read_csv(path_to_saved_end_result_dataframe, index_col=0)
    saved_df = saved_df.drop(0, axis=0)


    print("--------------------------------------------------------------------------------------------")
    print("-------------------------------- SAVED END RESULT DATAFRAME --------------------------------")
    print("--------------------------------------------------------------------------------------------")
    print(saved_df)

    print("The two dataframes are equal: ", are_dataframes_equal(expert_system_results, saved_df))


def test_integration_dataframe_results_with_line_8_cut():
    """
    In the initial state of the network, all substations are on busbar1
    Line 9 is between Node 3 and 8 [internal node ID indexing]
    Test
    """

    #import os
    #os.chdir('../../../')

    # Some manual configuration
    line_to_cut = [8]
    timestep = 0
    debug = False
    param_folder = "./alphaDeesp/tests/ressources_for_tests/default14_static_line8"

    # read config file and parameters folder for Pypownet
    config = configparser.ConfigParser()
    config.read("./alphaDeesp/tests/ressources_for_tests/config_for_tests.ini")

    # run Pypownet
    loader = PypownetObservationLoader(param_folder)
    env, obs, action_space = loader.get_observation(timestep=timestep)
    sim = PypownetSimulation(
        env, obs, action_space, param_options=config["DEFAULT"], debug=debug, ltc=line_to_cut
    )

    # create NetworkX graph g_over, and DataFrame df_of_g
    df_of_g = sim.get_dataframe()
    g_over = sim.build_graph_from_data_frame(line_to_cut)

    simulator_data = {"substations_elements": sim.substations_elements,
                      "substation_to_node_mapping": sim.substation_to_node_mapping,
                      "internal_to_external_mapping": sim.internal_to_external_mapping}

    # create AlphaDeesp
    alphadeesp = AlphaDeesp(g_over, df_of_g, simulator_data=simulator_data)

    ranked_combinations = alphadeesp.get_ranked_combinations()

    expert_system_results = sim.compute_new_network_changes(ranked_combinations)
    # This removes the first XXX line (used to construct initial dataframe structure)
    expert_system_results = expert_system_results.drop(0, axis=0)

    # print("--------------------------------------------------------------------------------------------")
    # print("----------------------------------- END RESULT DATAFRAME -----------------------------------")
    # print("--------------------------------------------------------------------------------------------")
    # print(expert_system_results)

    path_to_saved_end_result_dataframe = \
        Path.cwd() / "alphaDeesp/tests/ressources_for_tests/END_RESULT_DATAFRAME_for_test_with_line_8_cut.csv"

    # expert_system_results.to_csv("./alphaDeesp/tests/ressources_for_tests/END_RESULT_DATAFRAME_for_test_with_line_9_cut.csv")

    saved_df = pd.read_csv(path_to_saved_end_result_dataframe, index_col=0)
    saved_df = saved_df.drop(0, axis=0)

    print("--------------------------------------------------------------------------------------------")
    print("-------------------------------- SAVED END RESULT DATAFRAME --------------------------------")
    print("--------------------------------------------------------------------------------------------")
    print(saved_df)

    print("The two dataframes are equal: ", are_dataframes_equal(expert_system_results, saved_df))
