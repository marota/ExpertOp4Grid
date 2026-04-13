# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of ExpertOp4Grid, an expert system approach to solve flow congestions in power grids

import logging
from abc import ABC, abstractmethod
from math import fabs
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

import pandas as pd

from alphaDeesp.core.elements import Consumption, ExtremityLine, OriginLine, Production

logger = logging.getLogger(__name__)

# Convenience alias for the heterogeneous per-substation element lists.
SubstationElement = Union[Production, Consumption, OriginLine, ExtremityLine]


class Simulation(ABC):
    """Abstract Class Simulation"""

    #: Backend-provided runtime configuration (thresholds, layout, etc.)
    #: populated by concrete subclasses in their own ``__init__``.
    param_options: Dict[str, Any]
    #: Debug flag consulted by :meth:`create_df`; concrete subclasses set it.
    debug: bool

    def __init__(self) -> None:
        super().__init__()


    @abstractmethod
    def cut_lines_and_recomputes_flows(self, ids: List[int]) -> Sequence[float]:
        """Disconnect the lines identified by ``ids`` and re-run a power flow.

        Implementations must not mutate the long-lived simulation state
        beyond what is needed to compute the new flows (overload
        disconnection parameters, for example, should be restored before
        returning).

        :param ids: list of internal line ids (as used by the backend) to
            switch off before recomputing the flows.
        :returns: A sequence of post-cut line flows (``numpy.ndarray`` or
            similar), aligned on the backend's line ordering. The values
            are used by :meth:`create_df` to fill the ``new_flows`` column.
        """

    @abstractmethod
    def isAntenna(self) -> Optional[int]:
        """Return the substation id of an antenna attached to the overloaded line, if any.

        An "antenna" is a substation where the overloaded line is the only
        line connected at a given busbar; splitting such a substation cannot
        help relieving the overload and AlphaDeesp uses this information to
        prune candidate topologies.

        :returns: The external substation id of the antenna, or ``None`` if
            the overloaded line is not attached to an antenna.
        """

    @abstractmethod
    def isDoubleLine(self) -> Optional[List[int]]:
        """Return the list of parallel lines sharing the endpoints of the overloaded line.

        Two substations may be connected by more than one line ("double
        line"); AlphaDeesp needs to know this because topology actions on
        either endpoint behave differently from the single-line case.

        :returns: A list of backend line ids that run in parallel to the
            current overloaded line (``self.ltc[0]``), or ``None`` if there
            is no parallel line.
        """

    @abstractmethod
    def getLinesAtSubAndBusbar(self) -> Dict[Any, List[int]]:
        """Return the lines connected to each endpoint substation, grouped by busbar.

        Used by :meth:`isAntenna` and by the ranking step to count the
        degree of each busbar around the overloaded line.

        :returns: A ``dict`` keyed by ``(substation_id, busbar_id)`` (or a
            backend-specific equivalent) whose values are the list of line
            ids connected at that busbar.
        """

    @abstractmethod
    def get_layout(self) -> List[Tuple[float, float]]:
        """Return the 2D coordinates of each substation for plotting.

        :returns: A list of ``(x, y)`` tuples, one per substation, in the
            order used by the backend (``[(x1, y1), (x2, y2), ...]``).
        """

    @abstractmethod
    def get_substation_in_cooldown(self) -> List[int]:
        """Return substations that cannot be acted upon at the current timestep.

        Some backends (notably Grid2op) enforce a cooldown period after a
        topology change; substations still in cooldown must be excluded
        from the candidate set.

        :returns: A list of substation ids currently in cooldown.
        """

    @abstractmethod
    def get_substation_elements(self) -> Dict[int, List[SubstationElement]]:
        """Return the per-substation element model built from the observation.

        Each element is an instance of one of the classes in
        :mod:`alphaDeesp.core.elements` (``Production``, ``Consumption``,
        ``OriginLine``, ``ExtremityLine``). AlphaDeesp consumes this mapping
        to enumerate busbar configurations.

        :returns: A ``dict`` mapping substation id (int) to the list of
            element objects attached to that substation.
        """

    @abstractmethod
    def get_substation_to_node_mapping(self) -> Optional[Dict[int, Any]]:
        """Return the mapping from substation ids to overflow-graph node ids.

        Depending on the backend a substation may map to one or two nodes
        (one per busbar); this mapping is used to translate AlphaDeesp's
        internal graph back to substation-level actions.

        :returns: A ``dict`` keyed by substation id whose values are node
            ids in the overflow graph, or ``None`` if the backend uses a
            1-to-1 mapping and does not need the translation table.
        """

    @abstractmethod
    def get_internal_to_external_mapping(self) -> Dict[int, int]:
        """Return the translation table from internal node ids to backend ids.

        AlphaDeesp renumbers substations densely starting at 0 for its
        graph algorithms; this mapping is used to emit topology actions
        that reference the original backend ids.

        :returns: A ``dict`` keyed by internal (AlphaDeesp) node id whose
            values are the corresponding backend substation ids.
        """

    @abstractmethod
    def get_dataframe(self) -> pd.DataFrame:
        """Return the main topology + flow dataframe produced by :meth:`create_df`.

        The dataframe has one row per line of the grid and at least the
        columns ``idx_or``, ``idx_ex``, ``init_flows``, ``new_flows``,
        ``delta_flows``, ``swapped`` and ``gray_edges``. It is the primary
        input of the AlphaDeesp ranking step.

        :returns: A ``pandas.DataFrame`` representing the overloaded grid.
        """

    @abstractmethod
    def get_reference_topovec_sub(self, sub: int) -> List[int]:
        """Return the all-on-busbar-1 topology vector for substation ``sub``.

        AlphaDeesp uses this vector as the "do nothing" reference when
        ranking candidate busbar splits.

        :param sub: Backend substation id.
        :returns: A list of integers, one per element attached to ``sub``,
            all initialized to the reference busbar (typically 0 or 1
            depending on the backend convention).
        """

    @abstractmethod
    def get_overload_disconnection_topovec_subor(self, l: int) -> Tuple[int, List[int]]:
        """Return the topology vector that disconnects the origin side of line ``l``.

        This is used to simulate a line-opening action as a degenerate
        topology change on the ``origin`` substation.

        :param l: Backend line id (typically the overloaded line).
        :returns: A ``(substation_id, topo_vect)`` tuple where
            ``topo_vect`` has ``-1`` at the element position corresponding
            to the origin side of ``l`` (meaning "disconnected") and
            preserves the current assignment elsewhere.
        """

    @staticmethod
    def create_end_result_empty_dataframe() -> pd.DataFrame:
        """This function creates initial structure for the dataframe"""

        end_result_dataframe_structure_initiation: Dict[str, List[Any]] = {
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

    def create_df(self, d: Dict[str, Any], line_to_cut: List[int]) -> pd.DataFrame:
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

        if getattr(self, "debug", False):
            logger.debug("==== After gray_edges added IN FUNCTION CREATE DF ====")
            logger.debug("%s", df)

        return df

    @staticmethod
    def branch_direction_swaps(df: pd.DataFrame) -> None:
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
    def invert_dict_keys_values(d: Dict[Any, Any]) -> Dict[Any, Any]:
        return dict([(v, k) for k, v in d.items()])

    @staticmethod
    def get_model_obj_from_or(
        df_indexed: pd.DataFrame,
        substation_id: int,
        dest: int,
        busbar: int,
    ) -> Optional[Union[OriginLine, ExtremityLine]]:
        try:
            # Case 1: Direct Match
            val = df_indexed.loc[(substation_id, dest), 'delta_flows']

            # Handle duplicates: if multiple rows match, val is a Series
            if isinstance(val, pd.Series):
                val = val.iloc[0]  # Take the first one

            # Ensure it's a list for the constructor
            return OriginLine(busbar, dest, [val])

        except KeyError:
            # Case 2: Swapped Match
            try:
                row = df_indexed.loc[(dest, substation_id)]

                # --- FIX STARTS HERE ---
                # If we get a DataFrame (multiple matches), take the first row
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                # --- FIX ENDS HERE ---

                val = [row['delta_flows']]

                # Now 'row' is guaranteed to be a Series, so these are single scalars
                if row['swapped'] == row['new_flows_swapped']:
                    return OriginLine(busbar, dest, val)
                else:
                    return ExtremityLine(busbar, dest, val)

            except KeyError:
                return None

    @staticmethod
    def get_model_obj_from_ext(
        df_indexed: pd.DataFrame,
        substation_id: int,
        dest: int,
        busbar: int,
    ) -> Optional[Union[OriginLine, ExtremityLine]]:
        """
        Optimized version using Pandas MultiIndex, robust against Duplicate Rows.
        """
        try:
            # Case 1: Direct Match (idx_or == dest AND idx_ex == substation_id)
            # We look up the 'delta_flows' column directly
            val = df_indexed.loc[(dest, substation_id), 'delta_flows']

            # FIX 1: Handle if multiple rows match (returns Series instead of scalar)
            if isinstance(val, pd.Series):
                val = val.iloc[0]

            return ExtremityLine(busbar, dest, [val])

        except KeyError:
            # Case 2: Swapped Match (idx_or == substation_id AND idx_ex == dest)
            try:
                row = df_indexed.loc[(substation_id, dest)]

                # FIX 2: Handle if multiple rows match (returns DataFrame instead of Series)
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]  # Take the first row

                val = [row['delta_flows']]

                # Now 'row' is definitely a Series, so these comparisons result in a single Boolean
                # Logic: If swapped status matches new_flows_swapped -> It behaves like an ExtremityLine
                if row['swapped'] == row['new_flows_swapped']:
                    return ExtremityLine(busbar, dest, val)
                else:
                    return OriginLine(busbar, dest, val)

            except KeyError:
                # Case 3: Line not found in either direction
                return None