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
        # Printer API (for both Grid2op and Pypownet)
        printer.display_geo(g_over, custom_layout, name="g_overflow_print")
        printer.display_geo(g_pow, custom_layout, name="g_pow")
        printer.display_geo(g_pow_prime, custom_layout, name="g_pow_prime")

        # Grid2op API (Grid2op only)
        # fig_before = sim.plot_grid_beforecut()
        # fig_before.show()
        # fig_after = sim.plot_grid_aftercut()
        # fig_after.show()

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
            sim.load_from_observation(elem[1], ltc)
            g_over_detailed = sim.build_detailed_graph_from_internal_structure(ltc)
            printer.display_geo(g_over_detailed, custom_layout, name=elem[0])

    return ranked_combinations, expert_system_results

def main():
    # ###############################################################################################################
    # Read parameters provided by manual mode (Shell - config.ini - param_folder for the grid)
    shell_print_project_header()

    parser = argparse.ArgumentParser(description="Expert System")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Prints additional information for debugging purposes")
    parser.add_argument("-s", "--snapshot", action="store_true",
                        help="Displays the main overflow graph at step i, ie, delta_flows_graph, diff between "
                             "flows before and after cutting the constrained line", default = False)
    # nargs '+' == 1 or more.
    # nargs '*' == 0 or more.
    # nargs '?' == 0 or 1.
    # ltc for: Lines To Cut
    parser.add_argument("-l", "--ltc", nargs="+", type=int,
                        help="List of integers representing the lines to cut", default = [9])
    parser.add_argument("-t", "--timestep", type=int,
                        help="Number of the timestep to use", default = 0)
    parser.add_argument("-g", "--gridpath",
                        help="Path to access to files representing a grid", default="")

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
    if args.gridpath is None:
        raise ValueError("Input arg error, --gridpath, please provide a path for access a grid configuration")

    print("-------------------------------------")
    print(f"Working on lines: {args.ltc} ")
    print("-------------------------------------\n")


    # ###############################################################################################################
    # Use Loaders API to load simulator environment in manual mode at desired timestep
    sim = None
    parameters_folder = args.gridpath
    if config["DEFAULT"]["simulatortype"] == "Pypownet":
        print("We init Pypownet Simulation")
        #parameters_folder = "./alphaDeesp/ressources/parameters/default14_static_ltc_9"
        loader = PypownetObservationLoader(parameters_folder)
        env, obs, action_space = loader.get_observation(args.timestep)
        sim = PypownetSimulation(env, obs, action_space, param_options=config["DEFAULT"], debug=args.debug,
                                 ltc=args.ltc)
    elif config["DEFAULT"]["simulatortype"] == "Grid2OP":
        print("We init Grid2OP Simulation")
        #parameters_folder = "./alphaDeesp/ressources/parameters/l2rpn_2019_ltc_9"
        loader = Grid2opObservationLoader(parameters_folder)
        env, obs, action_space = loader.get_observation(args.timestep)
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
