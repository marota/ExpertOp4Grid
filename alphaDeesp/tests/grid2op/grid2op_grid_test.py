"""This file contains tests for creating overflow graph. ie,
graph with delta flows, current_flows - flows_after_closing_line"""

import numpy as np
import configparser
import networkx as nx
from alphaDeesp.core.network import Network
from alphaDeesp.core.grid2op.Grid2opObservationLoader import Grid2opObservationLoader
from alphaDeesp.core.grid2op.Grid2opSimulation import Grid2opSimulation, build_powerflow_graph
from alphaDeesp.core.alphadeesp import AlphaDeesp



def build_sim():
    config = configparser.ConfigParser()
    config.read("./alphaDeesp/tests/resources_for_tests_grid2op/config_for_tests.ini")
    param_folder = "./alphaDeesp/tests/resources_for_tests_grid2op/l2rpn_2019_ltc_9"

    loader = Grid2opObservationLoader(param_folder)
    env, obs, action_space = loader.get_observation()
    observation_space = env.observation_space
    sim = Grid2opSimulation(obs, action_space, observation_space, param_options=config["DEFAULT"], debug=False,
                                 ltc=[9])
    return sim, env


def test_powerflow_graph():
    sim, env = build_sim()

    g_pow = build_powerflow_graph(sim.topo, sim.obs)
    print(sim.df)
    path_to_saved_graph = "./alphaDeesp/tests/resources_for_tests_grid2op/saved_graphs/g_pow.dot"
    saved_g = nx.drawing.nx_pydot.read_dot(path_to_saved_graph)
    
    for e1 in saved_g.edges(data="xlabel"):
        for e2 in g_pow.edges(data="xlabel"):
            if int(e1[0]) == int(e2[0]) and int(e1[1]) == int(e2[1]):
                saved_flow = float(e1[2])
                current_flow = float(e2[2])
                assert (saved_flow == current_flow)

    print("g_over and saved_g are isomorphic: ", nx.is_isomorphic(g_pow, saved_g))
    #assert (nx.is_isomorphic(g_pow, saved_g))


def test_overflow_grid():
    """This function, given the input folder in test/path,
    it computes the overflow graph and compares it with the saved graph: saved_overflow_graph.dot"""
    sim, env = build_sim()
    g_over = sim.build_graph_from_data_frame([9])
    path_to_saved_graph = "./alphaDeesp/tests/resources_for_tests_grid2op/saved_graphs/g_over.dot"

    saved_g = nx.drawing.nx_pydot.read_dot(path_to_saved_graph)

    for e1 in saved_g.edges(data="xlabel"):
        for e2 in g_over.edges(data="xlabel"):
            if int(e1[0]) == int(e2[0]) and int(e1[1]) == int(e2[1]):
                saved_flow = float(e1[2])
                current_flow = float(e2[2])
                assert (saved_flow == current_flow)

    print("g_over and saved_g are isomorphic: ", nx.is_isomorphic(g_over, saved_g))


def test_new_flows_after_network_cut():
    """After closing one electric line, the flows should change on the network."""
    sim, env = build_sim()

    new_flows = sim.cut_lines_and_recomputes_flows([9])

    print("diff flows = ")
    print((new_flows - sim.obs.p_or).astype(int))
    result = (new_flows - sim.obs.p_or).astype(int)
    print(result != 0.0)
    print("there are {} non zero elements".format(np.count_nonzero(result)))

    # only the line that got cut should be at zero.
    assert (np.count_nonzero(result) == 19)


def extract_topology():
    """This function takes an observation and tests the function: extract_topo_from_obs"""
    print("Test extract_topology")
    sim = build_sim()
    print("flows = ", sim.obs.p_or.astype(int))


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


def test_detailed_graph_with_node_0_split_in_two():
    """
    From an observation, avec using load2()
    basic graph with zero nodes split in two, has 14 nodes
    with node 0 split in two, we should have 15 nodes
    :return:
    """
    sim, env = build_sim()
    new_configuration = [[2, 2, 2, 1, 1]]
    node_id = [4]
    _current_observation = sim.change_nodes_configurations(new_configuration, node_id, env)
    print(_current_observation)

    network = Network(sim.substations_elements)
    print("There are {} graphical nodes ".format(network.get_graphical_number_of_nodes()))
    assert (network.get_graphical_number_of_nodes() == 15)

def test_apply_topo():
    """
    Load configuration, modifiy topo on substation 4 with grid2op API, and generate graphs
    Then apply new topo with alphadeesp core function
    :return:
    """
    # import os
    # os.chdir('../../../')

    config = configparser.ConfigParser()
    config.read("./alphaDeesp/tests/resources_for_tests_grid2op/config_for_tests.ini")
    param_folder = "./alphaDeesp/tests/resources_for_tests_grid2op/l2rpn_2019_ltc_9"
    ltc = [9]

    loader = Grid2opObservationLoader(param_folder)
    env, obs, action_space = loader.get_observation()

    sub_id = 4
    new_configuration = [2, 2, 2, 1, 1] # Grid2op notation
    new_conf_alphadeesp_1 = [1, 1, 0, 0, 0] # ALphadeesp notation
    new_conf_alphadeesp_2 = [0, 0, 0, 0, 0] # ALphadeesp notation

    action_def = {"set_bus": {"substations_id": [(sub_id, new_configuration)]}}
    action = env.action_space(action_def)
    osb, _reward, _done, _info = env.step(action)

    observation_space = env.observation_space
    sim = Grid2opSimulation(obs, action_space, observation_space, param_options=config["DEFAULT"], debug=False,
                            ltc=ltc)

    # Get data from simulator representing the grid before and after line cutting, and topologies. Then initialize ALphadeesp
    df_of_g = sim.get_dataframe()
    g_over = sim.build_graph_from_data_frame(ltc)
    g_pow = sim.build_powerflow_graph_beforecut()
    g_pow_prime = sim.build_powerflow_graph_aftercut()
    simulator_data = {"substations_elements": sim.get_substation_elements(),
                      "substation_to_node_mapping": sim.get_substation_to_node_mapping(),
                      "internal_to_external_mapping": sim.get_internal_to_external_mapping()}
    printer = None
    custom_layout = sim.get_layout()
    debug = False
    g_copy = g_over.copy() # Very important step ! otherwise, apply_new_topo will change g_over and it won't be chainable
    alphadeesp = AlphaDeesp(g_over, df_of_g, custom_layout, printer, simulator_data, sim.substation_in_cooldown, debug=debug)


    # Apply topo twice with Alphadeesp core on this Network and check if applied correctly
    new_graph, internal_repr = alphadeesp.apply_new_topo_to_graph(g_copy, new_topology=new_conf_alphadeesp_1,
                                                                  node_to_change=sub_id)
    for i, busbar_id in enumerate(new_conf_alphadeesp_1):
        assert (internal_repr[sub_id][i].busbar_id == busbar_id)

    new_graph, internal_repr = alphadeesp.apply_new_topo_to_graph(g_copy, new_topology=new_conf_alphadeesp_2,
                                                                  node_to_change=sub_id)
    for i, busbar_id in enumerate(new_conf_alphadeesp_2):
        assert (internal_repr[sub_id][i].busbar_id == busbar_id)



