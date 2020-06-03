#!/usr/bin/python3
__author__ = "MarcM"

import argparse
import configparser
from alphaDeesp.core.alphadeesp import AlphaDeesp
from alphaDeesp.core.pypownet.PypownetSimulation import PypownetSimulation
from alphaDeesp.core.printer import Printer
from alphaDeesp.core.printer import shell_print_project_header
from alphaDeesp.core.grid2op.Grid2opSimulation import Grid2opSimulation
from alphaDeesp.core.grid2op.Grid2opObservationLoader import Grid2opObservationLoader


def main():
    shell_print_project_header()

    parser = argparse.ArgumentParser(description="Expert System")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Prints additional information for debugging purposes")
    parser.add_argument("-s", "--snapshot", action="store_true",
                        help="Displays the main overflow graph at step i, ie, delta_flows_graph, diff between "
                             "flows before and after cutting the constrained line")
    # nargs '+' == 1 or more.
    # nargs '*' == 0 or more.
    # nargs '?' == 0 or 1.
    # ltc for: Lines To Cut
    parser.add_argument("-l", "--ltc", nargs="+", type=int,
                        help="List of integers representing the lines to cut", default = [9])
    parser.add_argument("-t", "--timestep", type=int,
                        help="Number of the timestep to use", default = 0)

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

    print("-------------------------------------")
    print(f"Working on lines: {args.ltc} ")
    print("-------------------------------------\n")
    # ###############################################################################################################
    sim = None
    if config["DEFAULT"]["simulatortype"] == "Pypownet":
        parameters_folder = "./alphaDeesp/ressources/parameters/default14_static"
        sim = PypownetSimulation(config["DEFAULT"], args.debug, args.ltc, parameters_folder)
    elif config["DEFAULT"]["simulatortype"] == "Grid2OP":
        print("We init Grid2OP Simulation")
        parameters_folder = "./alphaDeesp/ressources/parameters/l2rpn_2019"
        loader = Grid2opObservationLoader(parameters_folder)
        obs, action_space = loader.get_observation(args.timestep)
        plot_helper = loader.get_plot_helper()
        sim = Grid2opSimulation(config["DEFAULT"], obs, action_space, args.ltc, plot_helper = plot_helper)
    elif config["DEFAULT"]["simulatorType"] == "RTE":
        print("We init RTE Simulation")
        # sim = RTESimulation(
        return
    else:
        print("Error simulator Type in config.ini not recognized...")
    custom_layout = sim.get_layout()

    printer = Printer()
    # ====================================================================
    # BELOW PART SHOULD BE UNAWARE OF WETHER WE WORK WITH RTE OR PYPOWNET

    ## Pypownet old way
    # Get data representing the grid before and after line cutting, and topologies
    # g_over, df_of_g = sim.build_graph_from_data_frame(args.ltc)
    # g_pow = sim.build_powerflow_graph(sim.obs)
    # g_pow_prime = sim.g_pow_prime

    # Plot the grids before and after line cutting
    # printer.display_geo(g_pow, custom_layout, name="g_pow_print")
    # printer.display_geo(g_over, custom_layout, name="g_overflow_print")
    # printer.display_geo(sim.g_pow_prime, custom_layout, name="g_pow_prime")


    ## Grid2op new way
    # Get data representing the grid before and after line cutting, and topologies
    df_of_g = sim.get_dataframe()
    g_over = sim.build_graph_from_data_frame(args.ltc)
    g_pow = sim.build_powerflow_graph_beforecut()
    g_pow_prime = sim.build_powerflow_graph_aftercut()

    # Plot the grids before and after line cutting
    # printer.display_geo(g_over, custom_layout, name="g_overflow_print") # Doesnt work
    fig_before = sim.plot_grid_beforecut()
    fig_before.show()
    fig_after = sim.plot_grid_aftercut()
    fig_after.show()


    # In common
    simulator_data = {"substations_elements": sim.get_substation_elements(),
                      "substation_to_node_mapping": sim.get_substation_to_node_mapping(),
                      "internal_to_external_mapping": sim.get_internal_to_external_mapping()}


    alphadeesp = AlphaDeesp(g_over, df_of_g, custom_layout, printer, simulator_data, debug=args.debug)
    ranked_combinations = alphadeesp.get_ranked_combinations()

    print("--------------------------------------------------------------------------------------------")
    print("----------------------------------- END RESULT DATAFRAME -----------------------------------")
    print("--------------------------------------------------------------------------------------------")
    # expert_system_results is a DATAFRAME containing lots of information about the work done.
    expert_system_results = sim.compute_new_network_changes(ranked_combinations)
    print(expert_system_results)

    # print("\n--------------------------------------- POST PROCESSING DEBUG ---------------------------------------\n")
    if args.snapshot:
        for elem in sim.save_bag:  # elem[0] = name, elem[1] = graph
            sim.load(elem[1], args.ltc)
            g_over_detailed = sim.build_detailed_graph_from_internal_structure(args.ltc)
            printer.display_geo(g_over_detailed, custom_layout, name=elem[0])

if __name__ == "__main__":
    main()
