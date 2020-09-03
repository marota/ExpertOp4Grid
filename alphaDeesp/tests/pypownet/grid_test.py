"""This file contains tests for creating overflow graph. ie,
graph with delta flows, current_flows - flows_after_closing_line"""

import configparser
import networkx as nx
from alphaDeesp.core.printer import Printer
from alphaDeesp.core.network import Network
from alphaDeesp.core.pypownet.PypownetSimulation import PypownetSimulation
from alphaDeesp.core.pypownet.PypownetObservationLoader import PypownetObservationLoader
import numpy as np

def build_sim(ltc):
    config = configparser.ConfigParser()
    config.read("./alphaDeesp/tests/ressources_for_tests/config_for_tests.ini")
    param_folder = "./alphaDeesp/ressources/parameters/default14_static"
    timestep = 0

    loader = PypownetObservationLoader(param_folder)
    env, obs, action_space = loader.get_observation(timestep)
    sim = PypownetSimulation(env, obs, action_space, param_options=config["DEFAULT"], debug=False,
                             ltc=[ltc])
    return sim

def test_powerflow_graph():
    custom_layout = [(-280, -81), (-100, -270), (366, -270), (366, -54), (-64, -54), (-64, 54), (366, 0),
                     (438, 0), (326, 54), (222, 108), (79, 162), (-152, 270), (-64, 270), (222, 216),
                     (-280, -151), (-100, -340), (366, -340), (390, -110), (-14, -104), (-184, 54), (400, -80),
                     (438, 100), (326, 140), (200, 8), (79, 12), (-152, 170), (-70, 200), (222, 200)]
    config = configparser.ConfigParser()
    config.read("./alphaDeesp/tests/ressources_for_tests/config_for_tests.ini")
    param_folder = "./alphaDeesp/ressources/parameters/default14_static"

    sim = build_sim(ltc = 9)

    g_pow = sim.build_powerflow_graph(sim.obs)

    path_to_saved_graph = "./alphaDeesp/tests/ressources_for_tests/saved_graphs/gpow_geo_save_2019-08-05_18-26_0.dot"
    saved_g = nx.drawing.nx_pydot.read_dot(path_to_saved_graph)

    for e1, e2 in zip(saved_g.edges(data="xlabel"), g_pow.edges(data="xlabel")):
        assert (float(e1[2]) == float(e2[2]))

    print("g_over and saved_g are isomorphic: ", nx.is_isomorphic(g_pow, saved_g))
    assert (nx.is_isomorphic(g_pow, saved_g))

def test_overflow_grid():
    """This function, given the input folder in test/path,
    it computes the overflow graph and compares it with the saved graph: saved_overflow_graph.dot"""
    custom_layout = [(-280, -81), (-100, -270), (366, -270), (366, -54), (-64, -54), (-64, 54), (366, 0),
                     (438, 0), (326, 54), (222, 108), (79, 162), (-152, 270), (-64, 270), (222, 216),
                     (-280, -151), (-100, -340), (366, -340), (390, -110), (-14, -104), (-184, 54), (400, -80),
                     (438, 100), (326, 140), (200, 8), (79, 12), (-152, 170), (-70, 200), (222, 200)]

    sim = build_sim(ltc = 9)

    _grid = sim.environment.game.grid
    _current_observation = sim.obs

    sim.load_from_observation(_current_observation, [9])

    df_of_g = sim.get_dataframe()
    g_over = sim.build_graph_from_data_frame([9])

    path_to_saved_graph = "./alphaDeesp/tests/ressources_for_tests/saved_graphs/g_overflow_print_geo_2020-09-03_12-20.dot"
        #"./alphaDeesp/tests/ressources_for_tests/saved_graphs/overflow_graph_example_geo_2019-11-06_16-40_0_.dot"

    saved_g = nx.drawing.nx_pydot.read_dot(path_to_saved_graph)


    for e1 in saved_g.edges(data="xlabel"):
        for e2 in g_over.edges(data="xlabel"):
            if int(e1[0]) == int(e2[0]) and int(e1[1]) == int(e2[1]):
                saved_flow = float(e1[2][1: -1])
                current_flow = float(e2[2])
                assert (saved_flow == current_flow)

    print("g_over and saved_g are isomorphic: ", nx.is_isomorphic(g_over, saved_g))
    #assert (nx.is_isomorphic(g_over, saved_g))


def test_constrained_path():
    """Given the input folder : default_14_static,
    we must get constrained path : cpath = 4, 5, 6"""


def test_new_flows_after_network_cut():
    """After closing one electric line, the flows should change on the network."""
    custom_layout = [(-280, -81), (-100, -270), (366, -270), (366, -54), (-64, -54), (-64, 54), (366, 0),
                     (438, 0), (326, 54), (222, 108), (79, 162), (-152, 270), (-64, 270), (222, 216),
                     (-280, -151), (-100, -340), (366, -340), (390, -110), (-14, -104), (-184, 54), (400, -80),
                     (438, 100), (326, 140), (200, 8), (79, 12), (-152, 170), (-70, 200), (222, 200)]

    print("Test new flows")
    sim = build_sim(ltc = 9)
    print("flows = ", sim.obs.active_flows_origin.astype(int))

    new_flows = sim.cut_lines_and_recomputes_flows([9])

    print("diff flows = ")
    print((new_flows - sim.obs.active_flows_origin).astype(int))
    result = (new_flows - sim.obs.active_flows_origin).astype(int)
    print(result != 0.0)
    print("there are {} non zero elements".format(np.count_nonzero(result)))

    # only the line that got cut should be at zero.
    assert (np.count_nonzero(result) == 19)

    # g_over df_of_g = sim.buildgra
    #
    # printer = Printer()
    # printer.display_geo(g_over, custom_layout, name="test_graph")


def extract_topology():
    """This function takes an observation and tests the function: extract_topo_from_obs"""
    print("Test extract_topology")
    sim = build_sim(ltc = 9)
    print("flows = ", sim.obs.active_flows_origin.astype(int))


def test_isgraph_connected():
    """This function test if a graph is connexe"""
    # Three different types of graph for tests
    # ============ Graph g1 ============
    #        5
    #         \>
    # 1->-2->-3->-4
    #      \>
    #      6
    # printer = Printer()

    g1 = nx.DiGraph()
    g1.add_edges_from([(1, 2), (2, 3), (2, 6), (3, 4), (5, 3)])
    # printer.display_geo(g1, name="test_graph_connected")
    print("graph g1 is connected: ", nx.is_weakly_connected(g1))
    assert (nx.is_weakly_connected(g1))
    g1.remove_edge(2, 3)
    print("After removing edge graph g1 is not_connected: ", nx.is_weakly_connected(g1))
    assert (not nx.is_weakly_connected(g1))
    # printer.display_geo(g1, name="test_graph")


def test_detailed_graph_with_node_0_split_in_two():
    """
    From an observation, avec using load2()
    basic graph with zero nodes split in two, has 14 nodes
    with node 0 split in two, we should have 15 nodes
    :return:
    """
    sim = build_sim(ltc = 9)

    _grid = sim.environment.game.grid
    _current_observation = sim.obs

    # new_configuration = [[1, 1, 0], [0, 0, 0, 0, 1]]
    new_configuration = [[0, 1, 0]]
    # node_id = [1, 5]
    node_id = [1]
    # new_configuration = [[1, 1, 1], [1, 1, 1, 1, 1]]
    # node_id = [1, 5]
    _current_observation = sim.change_nodes_configurations(new_configuration, node_id)
    # print(type(_current_observation))
    print(_current_observation)

    sim.load_from_observation(_current_observation, [9])
    # #################################################### PUT THE ABOVE IN A FIXTURE PYTEST ?

    # g_over, df_of_g = sim.build_graph_from_data_frame([9])
    # g_over_detailed = sim.build_detailed_graph_from_internal_structure([9])

    network = Network(sim.substations_elements)
    print("There are {} graphical nodes ".format(network.get_graphical_number_of_nodes()))
    assert (network.get_graphical_number_of_nodes() == 15)


def test_from_dict_substations_elements_14_nodes_111_on_node_0_and_11111_on_node_5_should_have_16_nodes():
    sim = build_sim(ltc = 9)

    _current_observation = sim.obs

    new_configuration = [[1, 1, 1], [1, 1, 1, 1, 1]]
    node_id = [1, 5]
    _current_observation = sim.change_nodes_configurations(new_configuration, node_id)

    sim.load_from_observation(_current_observation, [9])

    network = Network(sim.substations_elements)
    assert (network.get_graphical_number_of_nodes() == 16)


def test_from_dict_substations_elements_14_nodes_111_on_node_0_and_11000_on_node_5_should_have_16_nodes():
    sim = build_sim(ltc = 9)

    _current_observation = sim.obs

    new_configuration = [[1, 1, 1], [1, 1, 0, 0, 0]]
    node_id = [1, 5]
    _current_observation = sim.change_nodes_configurations(new_configuration, node_id)

    sim.load_from_observation(_current_observation, [9])

    network = Network(sim.substations_elements)
    print("There are {} graphical nodes ".format(network.get_graphical_number_of_nodes()))
    assert (network.get_graphical_number_of_nodes() == 16)


def test_from_dict_substations_elements_14_nodes_110_on_node_0_and_11000_on_node_5_should_have_16_nodes():
    # read config file and parameters folder for Pypownet
    config = configparser.ConfigParser()
    config.read("./alphaDeesp/tests/ressources_for_tests/config_for_tests.ini")
    param_folder = "./alphaDeesp/ressources/parameters/default14_static"

    # run Pypownet
    sim = build_sim(ltc = 9)

    # retrieve Topology as an Observation object from Pypownet
    _current_observation = sim.obs

    # new configuration creation
    new_configuration = [[1, 1, 0], [1, 1, 0, 0, 0]]
    node_ids = [1, 5]

    # change network Topology and retrieve an Obersation object from Pypownet
    _current_observation = sim.change_nodes_configurations(new_configuration, node_ids)

    sim.load_from_observation(_current_observation, [9])
    #
    print("sim.substations_elements =", sim.substations_elements)
    network = Network(sim.substations_elements)
    assert (network.get_graphical_number_of_nodes() == 16)

# test_powerflow_graph()
# test_overflow_grid()
# test_new_flows_after_network_cut()
# test_isgraph_connected()
# test_detailed_graph_with_node_0_split_in_two()
# test_integration_dataframe_results_with_line_9_cut()
# test_save_red_dataframe()
# test_round_random_tests()
