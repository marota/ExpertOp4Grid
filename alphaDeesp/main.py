#!/usr/bin/python3
__author__ = "MarcM, NMegel, mjothy"

import os
import argparse
import configparser

from alphaDeesp.core.printer import shell_print_project_header
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
                             "flows before and after cutting the constrained line. If 0, doesn't display the graphs", default = 0)
    # nargs '+' == 1 or more.
    # nargs '*' == 0 or more.
    # nargs '?' == 0 or 1.
    # ltc for: Lines To Cut
    parser.add_argument("-l", "--ltc", nargs="+", type=int,
                        help="List of integers representing the lines to cut", default = [9])
    parser.add_argument("-t", "--timestep", type=int,
                        help="ID of the timestep to use, starting from 0. Default is 0, i.e. the first time step will be considered", default = 0)
    parser.add_argument("-c", "--chronicscenario", type=str,
                        help="Name of chronic scenario to consider, as stored in chronics folder. By default, the first available chronic scenario will be chosen",
                        default=None)

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

    if args.snapshot:
        plot_folder = "alphaDeesp/ressources/output"
        os.makedirs(plot_folder, exist_ok=True)
        gridName = config['DEFAULT']['gridPath'].split('/')[-1]
        plot_folder = os.path.join(plot_folder, gridName)
        os.makedirs(plot_folder, exist_ok=True)
        lineName = 'linetocut_'+str(args.ltc[0])
        plot_folder = os.path.join(plot_folder, lineName)
        os.makedirs(plot_folder, exist_ok=True)
        scenarioName = 'Scenario_'+str(args.chronicscenario)
        plot_folder = os.path.join(plot_folder, scenarioName)
        os.makedirs(plot_folder, exist_ok=True)
        timestepName = 'Timestep_' + str(args.timestep)
        plot_folder = os.path.join(plot_folder, timestepName)
        os.makedirs(plot_folder, exist_ok=True)
    else:
        plot_folder = None

    if config["DEFAULT"]["simulatorType"] == "Pypownet":
        print("We init Pypownet Simulation")
        from alphaDeesp.core.pypownet.PypownetSimulation import PypownetSimulation
        from alphaDeesp.core.pypownet.PypownetObservationLoader import PypownetObservationLoader

        parameters_folder = config["DEFAULT"]["gridPath"]
        loader = PypownetObservationLoader(parameters_folder)
        env, obs, action_space = loader.get_observation(args.timestep)
        sim = PypownetSimulation(env, obs, action_space, param_options=config["DEFAULT"], debug=args.debug,
                                 ltc=args.ltc)
    elif config["DEFAULT"]["simulatorType"] == "Grid2OP":
        print("We init Grid2OP Simulation")
        from alphaDeesp.core.grid2op.Grid2opSimulation import Grid2opSimulation
        from alphaDeesp.core.grid2op.Grid2opObservationLoader import Grid2opObservationLoader

        parameters_folder = config["DEFAULT"]["gridPath"]
        try:
            difficulty = str(config["DEFAULT"]["grid2opDifficulty"])
        except:
            print("Default difficulty level has been set to None")
            difficulty = None
        loader = Grid2opObservationLoader(parameters_folder, difficulty = difficulty)
        env, obs, action_space = loader.get_observation(chronic_scenario= args.chronicscenario, timestep=args.timestep)
        observation_space = env.observation_space
        sim = Grid2opSimulation(obs, action_space, observation_space, param_options=config["DEFAULT"], debug=args.debug,
                                 ltc=args.ltc, plot=args.snapshot, plot_folder = plot_folder)

    elif config["DEFAULT"]["simulatorType"] == "RTE":
        print("We init RTE Simulation")
        # sim = RTESimulation()
        return
    else:
        print("Error simulator Type in config.ini not recognized...")

    # ###############################################################################################################
    # Call agent mode with possible plot and debug fonctionalities
    ranked_combinations, expert_system_results, action = expert_operator(sim, plot=args.snapshot)

    return ranked_combinations, expert_system_results, action

if __name__ == "__main__":
    main()
