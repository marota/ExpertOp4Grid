import configparser

from alphaDeesp.core.alphadeesp import AlphaDeesp
import pandas as pd
from pathlib import Path
import ast

from alphaDeesp.core.grid2op.Grid2opObservationLoader import Grid2opObservationLoader
from alphaDeesp.core.grid2op.Grid2opSimulation import Grid2opSimulation


custom_layout = [(-280, -81), (-100, -270), (366, -270), (366, -54), (-64, -54), (-64, 54), (366, 0),
                 (438, 0), (326, 54), (222, 108), (79, 162), (-152, 270), (-64, 270), (222, 216),
                 (-280, -151), (-100, -340), (366, -340), (390, -110), (-14, -104), (-184, 54), (400, -80),
                 (438, 100), (326, 140), (200, 8), (79, 12), (-152, 170), (-70, 200), (222, 200)]


def build_sim():
    config = configparser.ConfigParser()
    config.read("./alphaDeesp/tests/resources_for_tests_grid2op/config_for_tests.ini")
    param_folder = "./alphaDeesp/tests/resources_for_tests_grid2op/l2rpn_2019_ltc_9"

    loader = Grid2opObservationLoader(param_folder)
    env, obs, action_space = loader.get_observation()
    sim = Grid2opSimulation(config["DEFAULT"], env, obs, action_space, [9], plot_helper=loader.get_plot_helper())
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
                row_tab.append(round(elem, 2))
            else:
                row_tab.append(elem)

        generated_bag_rounded.append(row_tab)

    print("SEPARATOR ============================================================================================")

    for row in saved_bag:
        row_tab = []

        for elem in row:
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
    sim = build_sim()
    df_of_g = sim.get_dataframe()
    g_over = sim.build_graph_from_data_frame([9])
    g_pow = sim.build_powerflow_graph_beforecut()
    g_pow_prime = sim.build_powerflow_graph_aftercut()
    simulator_data = {"substations_elements": sim.get_substation_elements(),
                      "substation_to_node_mapping": sim.get_substation_to_node_mapping(),
                      "internal_to_external_mapping": sim.get_internal_to_external_mapping()}
    # create AlphaDeesp
    alphadeesp = AlphaDeesp(g_over, df_of_g, simulator_data=simulator_data)
    ranked_combinations = alphadeesp.get_ranked_combinations()
    expert_system_results = sim.compute_new_network_changes(ranked_combinations)
    # This removes the first XXX line (used to construct initial dataframe structure)
    expert_system_results = expert_system_results.drop(0, axis=0)
    path_to_saved_end_result_dataframe = \
        Path.cwd() / "alphaDeesp/tests/resources_for_tests_grid2op/END_RESULT_DATAFRAME_for_test_with_line_9_cut.csv"

    saved_df = pd.read_csv(path_to_saved_end_result_dataframe, index_col=0)

    print("The two dataframes are equal: ", are_dataframes_equal(expert_system_results, saved_df))


def test_integration_dataframe_results_with_line_8_cut():
    """
    In the initial state of the network, all substations are on busbar1
    Line 9 is between Node 3 and 8 [internal node ID indexing]
    Test
    """
    line_to_cut = [8]

    # read config file and parameters folder for Pypownet
    config = configparser.ConfigParser()
    config.read("./alphaDeesp/tests/ressources_for_tests/config_for_tests.ini")
    param_folder = "./alphaDeesp/tests/ressources_for_tests/default14_static_line8"

    # run Pypownet
    sim = PypownetSimulation(
        config["DEFAULT"], debug=False, param_folder=param_folder
    )

    print(sim.obs.thermal_limits) # Check thermal limits

    # retrieve Topology as an Observation object from Pypownet
    _current_observation = sim.obs
    # load Observation into Simulation Class
    sim.load(_current_observation, line_to_cut)
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
    path_to_saved_end_result_dataframe = \
        Path.cwd() / "alphaDeesp/tests/ressources_for_tests/END_RESULT_DATAFRAME_for_test_with_line_8_cut.csv"
    saved_df = pd.read_csv(path_to_saved_end_result_dataframe, index_col=0)
    print("The two dataframes are equal: ", are_dataframes_equal(expert_system_results, saved_df))


