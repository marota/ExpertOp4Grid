from abc import ABC, abstractmethod
from math import fabs
import numpy as np

import pandas as pd

from alphaDeesp.core.elements import ExtremityLine, OriginLine


class Simulation(ABC):
    """Abstract Class Simulation"""

    def __init__(self):
        super().__init__()


    @abstractmethod
    def cut_lines_and_recomputes_flows(self, ids: list):
        """network is the grid in pypownet, XX in RTE etc..."""

    @abstractmethod
    def isAntenna(self):
        """TODO"""
    @abstractmethod
    def isDoubleLine(self):
        """TODO"""

    @abstractmethod
    def getLinesAtSubAndBusbar(self):
        """TODO"""

    @abstractmethod
    def get_layout(self):
        """returns the layour of the graph in array of (x,y) form : [(x1,y1),(x2,y2)...]]"""

    @abstractmethod
    def get_substation_in_cooldown(self):
        """TODO"""

    @abstractmethod
    def get_substation_elements(self):
        """TODO"""

    @abstractmethod
    def get_substation_to_node_mapping(self):
        """TODO"""

    @abstractmethod
    def get_internal_to_external_mapping(self):
        """TODO"""

    @abstractmethod
    def get_dataframe(self):
        """TODO"""

    @abstractmethod
    def build_graph_from_data_frame(self, lines_to_cut: list):
        """TODO"""

    @abstractmethod
    def build_powerflow_graph_beforecut(self):
        """TODO"""

    @abstractmethod
    def get_reference_topovec_sub(self):
        """TODO"""

    @abstractmethod
    def get_overload_disconnection_topovec_subor(self):
        """TODO"""

    @abstractmethod
    def build_powerflow_graph_aftercut(self):
        """TODO"""

    @staticmethod
    def create_end_result_empty_dataframe():
        """This function creates initial structure for the dataframe"""

        end_result_dataframe_structure_initiation = {
            "overflow ID": [],
            "Flows before": [],
            "Flows after": [],
            "Delta flows": [],
            "Worsened line": [],
            "Prod redispatched": [],
            "Load redispatched": [],
            "Internal Topology applied ": [],
            "Topology applied": [],
            "Substation ID": [],
            "Rank Substation ID": [],
            "Topology score": [],
            "Topology simulated score": [],
            "Efficacity": [],
        }
        end_result_data_frame = pd.DataFrame(end_result_dataframe_structure_initiation)

        return end_result_data_frame

    def create_df(self, d: dict, line_to_cut: list):
        """arg: d represents a topology"""
        # HERE WE CREATE DATAFRAME
        df = pd.DataFrame(d["edges"])
        pd.set_option("display.float_format", lambda x: "%.3f" % x)

        # takes a dataframe and swaps branches init_flows < 0
        self.branch_direction_swaps(df)

        new_flows = self.cut_lines_and_recomputes_flows(line_to_cut)
        # print("new simulated flows = ", new_flows)

        # here we multiply by (-1) new flows that are reversed
        n_flows = []
        for f, swapped in zip(new_flows, df["swapped"]):
            if swapped:
                n_flows.append(f * -1)
            else:
                n_flows.append(f)

        df["new_flows"] = n_flows

        # if new_flows < 0, and abs(new) > abs(init) then True (we invert edge direction) else False
        new_flows_swapped = []

        for i, row in df.iterrows():
            # if newf < 0. and fabs(new_flows) > fabs(initf):
            if row["new_flows"] < 0 and fabs(row["new_flows"]) > fabs(row["init_flows"]):
                new_flows_swapped.append(True)
            else:
                new_flows_swapped.append(False)

        df["new_flows_swapped"] = new_flows_swapped

        delta_flo = []

        # now we add delta flows
        # report=abs(new_flows) - abs(init_flows) si le flux n'a pas change de direction
        # Si le flux a changé de direction, il y a 2 cas:
        # soit le nouveau flux est plus faible et dans ce cas, le report est négatif (on a déchargé la ligne) et le
        # report = -(abs(new_flows) + abs(init_flows))
        # sinon le report est positif et le report =
        # report = abs(new_flows) + abs(init_flows)
        for i, row in df.iterrows():
            if row["new_flows_swapped"]:
                delta_flo.append(fabs(row["new_flows"]) + fabs(row["init_flows"]))
                # here we swap origin and ext
                idx_or = row["idx_or"]
                df.at[i, "idx_or"] = row["idx_ex"]
                df.at[i, "idx_ex"] = idx_or
                df.at[i, "init_flows"] = fabs(row["init_flows"])
                # print(f"row #{i}, swapped idxor and idxer")
            elif (np.sign(row["new_flows"])!=np.sign(row["init_flows"])) and (row["new_flows"]!=0) and (row["init_flows"]!=0):#sign of 0 value is 0...
                delta_flo.append(-(fabs(row["new_flows"]) + fabs(row["init_flows"])))#negative flow dispacth in that case
            else:
                delta_flo.append(fabs(row["new_flows"]) - fabs(row["init_flows"]))

        # delta_flows = self.df["new_flows"].abs() - self.df["init_flows"].abs()
        df["delta_flows"] = delta_flo

        # small modification test to have multiple components
        # self.df.set_value(0, "delta_flows", -32.)
        # self.df.set_value(4, "delta_flows", -42.)
        # self.df.set_value(5, "delta_flows", -22.)

        # DO NOT USE SET_VALUE ANYMORE, USE DF.AT INSTEAD
        # df.at[5, "delta_flows"] = -22.

        # now we identify gray edges
        gray_edges = []
        ltc_report = df["delta_flows"].abs()[line_to_cut[0]]#pd.DataFrame.max(df["delta_flows"].abs())
        # print("max = ", max_report)
        max_overload = ltc_report * float(self.param_options["ThresholdReportOfLine"])
        # print("max overload = ", max_overload)
        for edge_value in df["delta_flows"]:
            if fabs(edge_value) < max_overload:
                gray_edges.append(True)
            else:
                gray_edges.append(False)
        # print("gray edges = ", gray_edges)
        df["gray_edges"] = gray_edges

        # if self.debug:
        # print("==== After gray_edges added IN FUNCTION CREATE DF ====")
        print(df)

        return df

    @staticmethod
    def branch_direction_swaps(df):
        """we parse self.df and invert branches init_flows < 0"""
        swapped = []
        for i, row in df.iterrows():
            # print("i {} row {}".format(i, row))
            # a = row["delta_flows"]
            # b = row["final_delta_flows"]
            # if np.sign(a) != np.sign(b):

            a = row["init_flows"]
            if a < 0 and a != 0.:
                # here we swap origin and ext
                idx_or = row["idx_or"]
                df.at[i, "idx_or"] = row["idx_ex"]
                df.at[i, "idx_ex"] = idx_or
                df.at[i, "init_flows"] = fabs(row["init_flows"])
                # print(f"row #{i}, swapped idxor and idxer")
                swapped.append(True)
            else:
                swapped.append(False)

        df["swapped"] = swapped

    @staticmethod
    def invert_dict_keys_values(d):
        return dict([(v, k) for k, v in d.items()])

    @staticmethod
    def get_model_obj_from_or(df, substation_id, dest, busbar):
        flow_value = list(df.query("idx_or == " + str(substation_id) + " & idx_ex == " + str(dest))
                          ["delta_flows"].round(decimals=2))
        if flow_value:  # if not empty
            return OriginLine(busbar, dest, flow_value)
        else:  # else means the flow has been swapped. We must invert edge.
            #POSSIBLY USELESS
            flow_value = list(df.query("idx_ex == " + str(substation_id) + " & idx_or == " + str(dest))
                              ["delta_flows"].round(decimals=2))
            swapped_condition = \
                list(df.query("idx_ex == " + str(substation_id) + " & idx_or == " + str(dest))
                     ["swapped"])[0]
            # second swapped_condition for new_flows_swapped in self.topo
            second_condition = \
                list(df.query("idx_ex == " + str(substation_id) + " & idx_or == " + str(dest))
                     ["new_flows_swapped"])[0]

            # if both are true, two swaps = do nothing or both are false and we do nothing.
            if (swapped_condition and second_condition) or (not swapped_condition and not second_condition):
                return OriginLine(busbar, dest, flow_value)

            # if one condition is true
            elif swapped_condition or second_condition:
                return ExtremityLine(busbar, dest, flow_value)

            else:
                raise ValueError("Problem with swap conditions")

    @staticmethod
    def get_model_obj_from_ext(df, substation_id, dest, busbar):
        flow_value = list(df.query("idx_or == " + str(dest) + " & idx_ex == " + str(substation_id))
                          ["delta_flows"].round(decimals=2))

        if flow_value:  # if not empty
            return ExtremityLine(busbar, dest, flow_value)
        else:
            #POSSIBLY USELESS
            flow_value = list(df.query("idx_ex == " + str(dest) + " & idx_or == " + str(substation_id))
                              ["delta_flows"].round(decimals=2))

            swapped_condition = \
                list(df.query("idx_ex == " + str(dest) + " & idx_or == " + str(substation_id))
                     ["swapped"])[0]
            # second swapped_condition for new_flows_swapped in self.topo
            second_condition = \
                list(df.query("idx_ex == " + str(dest) + " & idx_or == " + str(substation_id))
                     ["new_flows_swapped"])[0]

            # if both are true, two swaps = do nothing or both are false and we do nothing.
            if (swapped_condition and second_condition) or (not swapped_condition and not second_condition):
                return ExtremityLine(busbar, dest, flow_value)
            # if one condition is true
            elif swapped_condition or second_condition:
                return OriginLine(busbar, dest, flow_value)
            else:
                raise ValueError("Problem with swap conditions")