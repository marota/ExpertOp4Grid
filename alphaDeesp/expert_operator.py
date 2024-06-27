#!/usr/bin/python3

from alphaDeesp.core.alphadeesp import AlphaDeesp
import pandas as pd
import os
from alphaDeesp.core.graphsAndPaths import OverFlowGraph,PowerFlowGraph


def expert_operator(sim, plot=False, debug=False):
    # ====================================================================
    # Load the simulator given desired environment and config.ini

    ltc = sim.ltc

    # ====================================================================
    # Simulation of Expert results with simulator and alphadeesp

    # Get data representing the grid before and after line cutting, and topologies
    df_of_g = sim.get_dataframe()
    g_over =  OverFlowGraph(sim.topo, ltc, df_of_g)#sim.build_graph_from_data_frame(ltc)
    #g_pow = PowerFlowGraph(sim.topo, sim.lines_outaged)#.g sim.build_powerflow_graph_beforecut()
    #g_pow_prime = PowerFlowGraph(sim.topo_linecut, sim.lines_outaged_cut) #sim.build_powerflow_graph_aftercut()
    simulator_data = {"substations_elements": sim.get_substation_elements(),
                      "substation_to_node_mapping": sim.get_substation_to_node_mapping(),
                      "internal_to_external_mapping": sim.get_internal_to_external_mapping()}

    if plot:
        # Common plot API
        PowerFlowGraph(sim.topo, sim.lines_outaged).plot(sim.plot_folder,name="g_pow",state="before",sim=sim)#grid state plot before overload disconnection
        PowerFlowGraph(sim.topo_linecut, sim.lines_outaged_cut).plot(sim.plot_folder, name="g_pow_prime", state="after", sim=sim)#grid state plot after overload disconnection
        g_over.plot(layout=None,save_folder=sim.plot_folder)#g_over.plot(sim.layout,sim.plot_folder)

    #check if problem is not simply an antenna
    isAntenna_Sub=sim.isAntenna()
    isDoubleLine = sim.isDoubleLine()
    if isDoubleLine is not None:
        print("check")

    # Launch alphadeesp core
    if isAntenna_Sub is None:
        alphadeesp = AlphaDeesp(g_over.get_graph(), df_of_g, simulator_data,sim.substation_in_cooldown, debug = debug)
        ranked_combinations = alphadeesp.get_ranked_combinations()
    else:
        ranked_combinations = []
        ranked_combinations.append(pd.DataFrame({
            "score": 1,
            "topology": [sim.get_reference_topovec_sub(isAntenna_Sub)],
            "node": isAntenna_Sub
        }))


    # Expert results --> end dataframe
    expert_system_results, actions = sim.compute_new_network_changes(ranked_combinations)
    print("--------------------------------------------------------------------------------------------")
    print("----------------------------------- END RESULT DATAFRAME -----------------------------------")
    print("--------------------------------------------------------------------------------------------")
    print(expert_system_results)

    # Plot option
    if plot:
        save_folder = os.path.join(sim.plot_folder, "Result graph")
        for elem in sim.save_bag:  # elem[0] = name, elem[1] = graph
            name = elem[0]
            simulated_obs = elem[1]
            save_file_path=os.path.join(save_folder,name)
            if hasattr(sim, 'plot'):
                sim.plot(simulated_obs, save_file_path)#def plot(self,obs,save_file_path)

    return ranked_combinations, expert_system_results, actions


