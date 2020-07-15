#!/usr/bin/python3
__author__ = "MarcM"

import argparse
import configparser

from alphaDeesp.core.alphadeesp import AlphaDeesp
from alphaDeesp.core.pypownet.PypownetSimulation import PypownetSimulation
from alphaDeesp.core.pypownet.PypownetObservationLoader import PypownetObservationLoader
from alphaDeesp.core.printer import Printer
from alphaDeesp.core.printer import shell_print_project_header
from alphaDeesp.core.grid2op.Grid2opSimulation import Grid2opSimulation
from alphaDeesp.core.grid2op.Grid2opObservationLoader import Grid2opObservationLoader
from alphaDeesp.expert_operator import expert_operator

def main():
    # ###############################################################################################################
    # Read parameters provided by manual mode (Shell - config.ini - param_folder for the grid)
    shell_print_project_header()

    parser = argparse.ArgumentParser(description="Expert System")
    parser.add_argument("-d", "--debug", type=int,
                        help="If 1, prints additional information for debugging purposes. If 0, doesn't print any info", default = 0)
    parser.add_argument("-s", "--snapshot", type=int,
                        help="If 1, displays the main overflow graph at step i, ie, delta_flows_graph, diff between "
                             "flows before and after cutting the constrained line. If 0, doesn't display the graphs", default = 1)
    # nargs '+' == 1 or more.
    # nargs '*' == 0 or more.
    # nargs '?' == 0 or 1.
    # ltc for: Lines To Cut
    parser.add_argument("-l", "--ltc", nargs="+", type=int,
                        help="List of integers representing the lines to cut", default = [9])
    parser.add_argument("-t", "--timestep", type=int,
                        help="ID of the timestep to use, starting from 0. Default is 0, i.e. the first time step will be considered", default = 0)
    parser.add_argument("-c", "--chronicscenario", type=int,
                        help="ID of chronic scenario to consider, starting from 0. By default, the first available chronic scenario will be chosen, i.e. ID 0",
                        default=0)

    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read("./alphaDeesp/config.ini")
    print("#### PARAMETERS #####")
    for key in config["DEFAULT"]:
        print("key: {} = {}".format(key, config['DEFAULT'][key]))
    print("#### ########## #####\n")

    if args.ltc is None or len(args.ltc) != 1:
        raise ValueError("Input arg error, --ltc, for the moment, we allow cutting only one line.\n\nPlease select"
                         " one line to cut ex: python3 -m alphaDeesp.main -l 9")

    if args.snapshot > 1:
        raise ValueError("Input arg error, --snapshot, options are 0 or 1")

    if args.debug > 1:
        raise ValueError("Input arg error, --debug, options are 0 or 1")


    print("-------------------------------------")
    print(f"Working on lines: {args.ltc} ")
    print("-------------------------------------\n")


    # ###############################################################################################################
    # Use Loaders API to load simulator environment in manual mode at desired timestep
    sim = None
    args.debug = bool(args.debug)
    args.snapshot = bool(args.snapshot)

    if config["DEFAULT"]["simulatorType"] == "Pypownet":
        print("We init Pypownet Simulation")
        parameters_folder = config["DEFAULT"]["gridPath"]
        loader = PypownetObservationLoader(parameters_folder)
        env, obs, action_space = loader.get_observation(args.timestep)
        sim = PypownetSimulation(env, obs, action_space, param_options=config["DEFAULT"], debug=args.debug,
                                 ltc=args.ltc)
    elif config["DEFAULT"]["simulatorType"] == "Grid2OP":
        print("We init Grid2OP Simulation")
        parameters_folder = config["DEFAULT"]["gridPath"]
        loader = Grid2opObservationLoader(parameters_folder)
        env, obs, action_space = loader.get_observation(chronic_scenario= args.chronicscenario, timestep=args.timestep)
        observation_space = env.observation_space
        sim = Grid2opSimulation(obs, action_space, observation_space, param_options=config["DEFAULT"], debug=args.debug,
                                 ltc=args.ltc, plot=args.snapshot)

    elif config["DEFAULT"]["simulatorType"] == "RTE":
        print("We init RTE Simulation")
        # sim = RTESimulation()
        return
    else:
        print("Error simulator Type in config.ini not recognized...")

    # ====================================================================
    # ====================================================================
    # BELOW PART SHOULD BE UNAWARE OF WETHER WE WORK WITH RTE OR PYPOWNET

    # df_of_g is a pandas_DataFrame
    g_over, df_of_g = sim.build_graph_from_data_frame(args.ltc)
    
    printer = Printer()
    if args.snapshot:
        printer.display_geo(g_over, custom_layout, name='overload')
    #
    # g_pow = sim.build_powerflow_graph(_grid)
    g_pow = sim.build_powerflow_graph(_current_observation)
    #
    # g_over = sim.build_overflow_graph(_grid, [9], config["DEFAULT"])
    #

    # printer.display_geo(g_pow, custom_layout, name="save_for_tests")

    # if args.snapshot:
    #     printer.display_geo(g_over, custom_layout, name="overflow_graph_example")

    simulator_data = {"substations_elements": sim.substations_elements,
                      "substation_to_node_mapping": sim.substation_to_node_mapping,
                      "internal_to_external_mapping": sim.internal_to_external_mapping}

    alphadeesp = AlphaDeesp(g_over, df_of_g, printer, custom_layout, simulator_data,
                            debug=args.debug)  # here instead of giving printer,

    ranked_combinations = alphadeesp.get_ranked_combinations()

    # expert_system_results is a DATAFRAME containing lots of information about the work done.

    expert_system_results = sim.compute_new_network_changes(ranked_combinations)
    print("--------------------------------------------------------------------------------------------")
    print("----------------------------------- END RESULT DATAFRAME -----------------------------------")
    print("--------------------------------------------------------------------------------------------")
    print(expert_system_results)

    # print("\n--------------------------------------- POST PROCESSING DEBUG ---------------------------------------\n")
    # save_bag is filled in function sim.compute_new_network_changes()
    if args.snapshot:
        # print("sim save bag")
        # print(sim.save_bag)
        # print("alphadeesp.bag of graphs")
        # print(alphadeesp.bag_of_graphs)
        for elem in sim.save_bag:  # elem[0] = name, elem[1] = graph
            # g = alphaDeesp.bag_of_graphs[elem[1]]

            # for e in g.edges():
            #     print(e)
            #
            # res = nx.get_edge_attributes(g, "xlabel")
            # print(res)

            # CREATE A DICTIONARY FOR NETWORKX to set edge_attributes, we must have a DICTIONNARY
            # { (u, v): {"xlabel"

            # now here apply new values from simulation into new graph with new topo
            # g = elem[1]
            # print(g)
            # printer.display_geo(g, custom_layout, name=elem[0])

            # _grid = sim.environment.game.grid
            # sim.load(elem[1], args.ltc)
            # g_over, df_of_g = sim.build_graph_from_data_frame()
            # printer.display_geo(g_over, custom_layout, name="overflow_graph_example")

            sim.load(elem[1], args.ltc)
            # g_over, df_of_g = sim.build_graph_from_data_frame(args.ltc)
            g_over_detailed = sim.build_detailed_graph_from_internal_structure(args.ltc)

            printer.display_geo(g_over_detailed, custom_layout, name=elem[0])

            # g_pow = sim.build_powerflow_graph(elem[1])
            # printer.display_geo(g_pow, custom_layout, name=elem[0])

    # print("substation to node mapping")
    # print(sim.substation_to_node_mapping)
    # print("internal to external mapping ")
    # print(sim.internal_to_external_mapping)

    # ###############################################################################################################
    # Call agent mode with possible plot and debug fonctionalities
    ranked_combinations, expert_system_results, action = expert_operator(sim, plot=args.snapshot)

    return ranked_combinations, expert_system_results, action

if __name__ == "__main__":
    main()
