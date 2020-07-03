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

def expert_operator(sim, plot = False, debug = False):


    # ====================================================================
    # Load the simulator given desired environment and config.ini

    ltc = sim.ltc
    custom_layout = sim.get_layout()
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
    expert_system_results = sim.compute_new_network_changes(ranked_combinations)
    print("--------------------------------------------------------------------------------------------")
    print("----------------------------------- END RESULT DATAFRAME -----------------------------------")
    print("--------------------------------------------------------------------------------------------")
    print(expert_system_results)

    # Plot option
    if plot:
        for elem in sim.save_bag:  # elem[0] = name, elem[1] = graph
            sim.plot_grid_from_obs(elem[1], elem[0])
            # sim.load_from_observation(elem[1], ltc)
            # g_over_detailed = sim.build_detailed_graph_from_internal_structure(ltc)
            # printer.display_geo(g_over_detailed, custom_layout, name=elem[0])


    return ranked_combinations, expert_system_results

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
                        help="ID of chronic scenario to consider, starting from 0. By default, the first available chronic scenario will be chosen, i.e. ID 0", default=0)

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
        sim = Grid2opSimulation(env, obs, action_space, param_options=config["DEFAULT"], debug=args.debug,
                                 ltc=args.ltc)

    elif config["DEFAULT"]["simulatorType"] == "RTE":
        print("We init RTE Simulation")
        # sim = RTESimulation()
        return
    else:
        print("Error simulator Type in config.ini not recognized...")

    # ###############################################################################################################
    # Call agent mode with possible plot and debug fonctionalities
    ranked_combinations, expert_system_results = expert_operator(sim, plot=args.snapshot, debug = args.debug)

    return ranked_combinations, expert_system_results

if __name__ == "__main__":
    main()
