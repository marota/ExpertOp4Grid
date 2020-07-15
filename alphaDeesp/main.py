#!/usr/bin/python3
__author__ = "MarcM"

import argparse
import configparser
from alphaDeesp.core.alphadeesp import AlphaDeesp
from alphaDeesp.core.pypownet import PypownetSimulation
from alphaDeesp.core.printer import Printer
from alphaDeesp.core.printer import shell_print_project_header


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
                        help="List of integers representing the lines to cut")

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

    custom_layout = [(-280, -81), (-100, -270), (366, -270), (366, -54), (-64, -54), (-64, 54), (366, 0),
                     (438, 0), (326, 54), (222, 108), (79, 162), (-152, 270), (-64, 270), (222, 216),
                     (-280, -151), (-100, -340), (366, -340), (390, -110), (-14, -104), (-184, 54), (400, -80),
                     (438, 100), (326, 140), (200, 8), (79, 12), (-152, 170), (-70, 200), (222, 200)]

    custom_layout2 = {
        '6661': (-280, -151),
        '6662': (-100, -340),
        '6663': (366, -340),
        '6664': (390, -110),
        '6665': (-14, -74),
        '6666': (-184, 54),
        '6667': (400, -80),
        '6668': (438, 100),
        '6669': (326, 140),
        '66610': (200, 8),
        '66611': (79, 12),
        '66612': (-152, 170),
        '66613': (-70, 200),
        '66614': (222, 200)
    }
    axially_symetric = False
    if axially_symetric:
        x_inversed_layout = []
        for x in custom_layout:
            x_inversed_layout.append((x[0] * -1, x[1]))
        custom_layout = x_inversed_layout

    # ###############################################################################################################
    # ###############################################################################################################
    # ###############################################################################################################
    sim = None

    if config["DEFAULT"]["simulatortype"] == "Pypownet":
        parameters_folder = "./alphaDeesp/ressources/parameters/default14_static"
        sim = PypownetSimulation(config["DEFAULT"], args.debug, parameters_folder)

        print("current chronic name = ", sim.environment.game.get_current_chronic_name())

        _current_observation = sim.obs

        print(_current_observation)

        print("args ltc = ", args.ltc)
        print("type = ", type(args.ltc))

        if args.debug:
            print("current obs = ")
            print(_current_observation)

        sim.load(_current_observation, args.ltc)

    elif config["DEFAULT"]["simulatorType"] == "RTE":
        print("We init RTE Simulation")
        # sim = RTESimulation(
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


if __name__ == "__main__":
    main()
