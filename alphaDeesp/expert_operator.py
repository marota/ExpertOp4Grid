#!/usr/bin/python3

from alphaDeesp.core.alphadeesp import AlphaDeesp
from alphaDeesp.core.printer import Printer


def expert_operator(sim, plot=False, debug=False):
    # ====================================================================
    # Load the simulator given desired environment and config.ini

    ltc = sim.ltc
    custom_layout = sim.get_layout()
    printer = None
    if plot:
        printer = Printer()

    # ====================================================================
    # Simulation of Expert results with simulator and alphadeesp

    # Get data representing the grid before and after line cutting, and topologies
    df_of_g = sim.get_dataframe()
    g_over = sim.build_graph_from_data_frame(ltc)
    g_pow = sim.build_powerflow_graph_beforecut()
    g_pow_prime = sim.build_powerflow_graph_aftercut()
    simulator_data = {"substations_elements": sim.get_substation_elements(),
                      "substation_to_node_mapping": sim.get_substation_to_node_mapping(),
                      "internal_to_external_mapping": sim.get_internal_to_external_mapping()}

    if plot:
        # Common plot API
        sim.plot_grid_beforecut()
        sim.plot_grid_aftercut()
        sim.plot_grid_delta()

    # Launch alphadeesp core
    alphadeesp = AlphaDeesp(g_over, df_of_g, custom_layout, printer, simulator_data, debug = debug)
    ranked_combinations = alphadeesp.get_ranked_combinations()

    # Expert results --> end dataframe
    expert_system_results, actions = sim.compute_new_network_changes(ranked_combinations)
    print("--------------------------------------------------------------------------------------------")
    print("----------------------------------- END RESULT DATAFRAME -----------------------------------")
    print("--------------------------------------------------------------------------------------------")
    print(expert_system_results)

    # Plot option
    if plot:
        for elem in sim.save_bag:  # elem[0] = name, elem[1] = graph
            name = elem[0]
            simulated_obs = elem[1]
            sim.plot_grid_from_obs(simulated_obs, name)

    return ranked_combinations, expert_system_results, actions