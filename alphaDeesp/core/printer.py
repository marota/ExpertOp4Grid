"""This file contains all the possible displays for graph from NetworkX"""

import os
import pprint
import datetime
import networkx as nx

from alphaDeesp.core import alphadeesp

from pathlib import Path


class Printer:
    save_id_number = 0

    def __init__(self, output_path = None):
        if output_path is None:
            self.default_output_path = "alphaDeesp/ressources/output"
        else:
            self.default_output_path = output_path
        self.base_output_path = os.path.join(self.default_output_path,"Base graph")
        os.makedirs(self.base_output_path, exist_ok=True)
        self.results_output_path = os.path.join(self.default_output_path,"Result graph")
        os.makedirs(self.results_output_path, exist_ok=True)
        print("self.default output path = ", self.default_output_path)

    def display_geo(self, g, custom_layout=None, axial_symetry=False, save=False, name=None):
        """This function displays the graph g in a "geographical" way"""

        "filenames are pathlib.Paths objects"
        type_ = "results"
        if name in ["g_pow", "g_overflow_print", "g_pow_prime"]:
            type_ = "base"
        filename_dot, filename_pdf = self.create_namefile("geo", name=name, type = type_)

        dic_pos_attributes = {}
        if custom_layout is not None:
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
            assert isinstance(custom_layout, list) is True
            # we create a dictionary to add a position attribute to the nodes
            ii = 0
            for i, value in enumerate(custom_layout):
                # i += 1
                # print(f"i:{i} value:{value}")
                if i < len(custom_layout):
                    dic_pos_attributes[i] = {"pos": (str(value[0]) + ", " + str(value[1]) + "!")}
                else:
                    i = int("666" + str(ii))
                    dic_pos_attributes[i] = {"pos": (str(value[0]) + ", " + str(value[1]) + "!")}
                    ii += 1

            # we update the graph with some specific position
            nx.set_node_attributes(g, dic_pos_attributes)

        # print("############### debug in Printer ###############")
        # pprint.pprint(dic_pos_attributes)

        nx.drawing.nx_pydot.write_dot(g, filename_dot)

        if custom_layout is None:
            cmd_line = 'neato -Tpdf "' + str(filename_dot) + '" -o "' + str(filename_pdf) + '"'
        else:
            cmd_line = 'neato -n -Tpdf "' + str(filename_dot) + '" -o "' + str(filename_pdf) + '"'
        print("we print the cmd line = ", cmd_line)

        assert alphadeesp.execute_command(cmd_line)

        # os.system(cmd_line)

        cmd_line = f"evince {str(filename_pdf)} &"
        os.system(cmd_line)

        # try:
        #     alphadeesp.execute_command(cmd_line)
        # except RuntimeError:
        #     print("A problem while executing a command occured.")

        os.system(cmd_line)
        os.system("evince " + str(filename_pdf) + " &")

        # if save is False:
        #     assert(alphadeesp.execute_command(f"rm {filename_dot}"))
        #     assert(alphadeesp.execute_command(f"rm {filename_pdf}"))

        # if save is False:
        #     os.system("rm " + filename_dot)
        #     os.system("rm " + filename_pdf)

    def display_elec(self, g, save=False):
        pass

    def create_namefile(self, display_type, name=None, type = "results"):
        """return dot and pdf filenames"""
        # filename_dot = "graph_result_" + display_type + "_" + current_date + ".dot"
        # filename_pdf = "graph_result_" + display_type + "_" + current_date + ".pdf"
        current_date_no_filter = datetime.datetime.now()
        current_date = current_date_no_filter.strftime("%Y-%m-%d_%H-%M")
        if type == 'results':
            current_date += "_" + str(Printer.save_id_number) + "_"
            Printer.save_id_number += 1

        if name is None:
            name = ""
        print("name = ", name)
        filename_dot = name + "_" + display_type + "_" + current_date + ".dot"
        filename_pdf = name + "_" + display_type + "_" + current_date + ".pdf"

        if type == "results":
            output_path = self.results_output_path
        elif type == "base":
            output_path = self.base_output_path
        else:
            output_path = self.default_output_path
        hard_filename_dot = os.path.join(output_path,filename_dot)
        # hard_filename_dot = filename_dot
        hard_filename_pdf = os.path.join(output_path,filename_pdf)

        print("============================= FUNCTION create_namefile =============================")
        print("hard_filename = ", hard_filename_pdf)

        return hard_filename_dot, hard_filename_pdf


def shell_print_project_header():
    os.system("cat ./print_header.txt")
