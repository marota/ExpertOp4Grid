"""PowerFlowGraph: a coloured current-state power-flow graph."""

import logging
from math import fabs
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from alphaDeesp.core.printer import Printer
from alphaDeesp.core.graphs.constants import default_voltage_colors

logger = logging.getLogger(__name__)


class PowerFlowGraph:
    """
    A coloured graph of current grid state with productions, consumptions and topology
    """

    def __init__(self, topo: Dict[str, Any], lines_cut: List[int], layout: Optional[List[Tuple[float, float]]] = None, float_precision: str = "%.2f") -> None:
        """
        Parameters
        ----------

        topo: :class:`dict`
            dictionnary of two dictionnaries edges and nodes, to represent the grid topologie. edges have attributes "init_flows" representing the power flowing, as well as "idx_or","idx_ex"
             for substation extremities
             Nodes have attributes "are_prods","are_loads" if nodes have any productions or any load, as well as "prods_values","load_values" array enumerating the prod and load values at this node.

        lines_cut: ``array``
            ids of lines disconnected

        float_precision: "str"
            Significant digits for dispalyed values at edges. In the form of "%.2f"
        """
        self.topo=topo
        self.lines_cut=lines_cut
        self.layout=layout
        self.float_precision=float_precision
        self.build_graph()
        #self.g=self.build_powerflow_graph()

    def build_graph(self) -> None:
        """This method creates the NetworkX Graph of the grid state"""
        g = nx.MultiDiGraph()

        # Get the id of lines that are disconnected from network
        # lines_cut = np.argwhere(obs.line_status == False)[:, 0]
        topo=self.topo

        # Get the whole topology information
        idx_or = topo["edges"]['idx_or']
        idx_ex = topo["edges"]['idx_ex']
        are_prods = topo["nodes"]['are_prods']
        are_loads = topo["nodes"]['are_loads']
        prods_values = topo["nodes"]['prods_values']
        loads_values = topo["nodes"]['loads_values']
        current_flows = topo["edges"]['init_flows']

        # =========================================== NODE PART ===========================================
        self.build_nodes(g, are_prods, are_loads, prods_values, loads_values)
        # =========================================== EDGE PART ===========================================
        self.build_edges(g, idx_or, idx_ex, edge_weights=current_flows)
        #return g
        self.g=g

    def build_nodes(self, g: nx.MultiDiGraph, are_prods: Any, are_loads: Any, prods_values: Any, loads_values: Any, debug: bool = False) -> None:
        """
        Create nodes in graph for current grid state

        Parameters
        ----------

        g: :class:`nx:MultiDiGraph`
            a networkx graph to which to add edges

        are_prods: ``array`` boolean
            if there are productions at each node

        are_loads: ``array`` boolean
            if there are cosnumptions at each node

        prods_values: ``array`` float
            the production values at each node

        loads_values: ``array`` float
            the consumption values at each node

        """
        # =========================================== NODE PART ===========================================
        # print(f"There are {len(are_loads)} nodes")
        prods_iter, loads_iter = iter(prods_values), iter(loads_values)
        i = 0
        # We color the nodes depending if they are production or consumption
        for is_prod, is_load in zip(are_prods, are_loads):
            prod = next(prods_iter) if is_prod else 0.
            load = next(loads_iter) if is_load else 0.
            prod_minus_load = prod - load
            if debug:
                logger.debug("Node n°[%s] : Production value: [%s] - Load value: [%s]", i, prod, load)
            if prod_minus_load > 0:  # PROD
                g.add_node(i, pin=True, prod_or_load="prod", value=str(prod_minus_load), style="filled",
                           fillcolor="coral")#orange#ff8000 #f30000")  # red color
            elif prod_minus_load < 0:  # LOAD
                g.add_node(i, pin=True, prod_or_load="load", value=str(prod_minus_load), style="filled",
                           fillcolor="lightblue")#"#478fd0")  # blue color
            else:  # WHITE COLOR
                g.add_node(i, pin=True, prod_or_load="load", value=str(prod_minus_load), style="filled",
                           fillcolor="#ffffed")  # white color
            i += 1

    def build_edges(self, g: nx.MultiDiGraph, idx_or: Any, idx_ex: Any, edge_weights: Any) -> None:

        """
        Create edges in graph for current grid state

        Parameters
        ----------

        g: :class:`nx:MultiDiGraph`
            a networkx graph to which to add edges

        idx_or: ``array`` int
            first extremity of edge for each edge

        idx_ex: ``array`` int
            second extremity of edge for each edge

        edge_weights: ``array`` float
            the flow value for each edge

        gtype: ``str``
            if we want a powerflow graph or

        """

        #if gtype is "powerflow":
        max_abs_flow = np.abs(np.array(edge_weights)).max()
        target_max_penwidth = 15.0
        # Determine the scaling factor
        if max_abs_flow > 0:
            scaling_factor = target_max_penwidth / max_abs_flow
        else:
            scaling_factor = 1.0

        for origin, extremity, weight_value in zip(idx_or, idx_ex, edge_weights):
            # origin += 1
            # extremity += 1
            penwidth = fabs(weight_value) * scaling_factor
            min_penwidth=0.1
            if penwidth == 0.0:
                penwidth = min_penwidth

            if weight_value >= 0:
                g.add_edge(origin, extremity, label=self.float_precision% weight_value, color="gray", fontsize=10,
                           penwidth=max(float(self.float_precision % penwidth),min_penwidth))
            else:
                g.add_edge(extremity, origin, label=self.float_precision % fabs(weight_value), color="gray", fontsize=10,
                           penwidth=max(float(self.float_precision % penwidth),min_penwidth))


    def get_graph(self) -> nx.MultiDiGraph:
        """
        Returns the NetworkX graph representing the current state of the power flow.

        Returns
        -------
        :class:`nx:MultiDiGraph`
            The NetworkX graph.
        """
        return self.g

    def set_voltage_level_color(self, voltage_levels_dict: Dict[Any, Any], voltage_colors: Dict[Any, str] = default_voltage_colors) -> None:
        """
        Sets the voltage level color for each node in the graph based on the provided voltage levels dictionary.

        Parameters
        ----------
        voltage_levels_dict : dict
            A dictionary mapping node IDs to their respective voltage levels.
        voltage_colors : dict, optional
            A dictionary mapping voltage levels to their corresponding colors. Defaults to `default_voltage_colors`.

        Notes
        -----
        This method updates the 'color' attribute of each node in the graph based on the voltage levels provided.
        """
        voltage_levels_colors_dict = {node: voltage_colors[voltage_levels_dict[node]] for node in self.g}

        nx.set_node_attributes(self.g, voltage_levels_colors_dict, "color")

    def set_electrical_node_number(self, nodal_number_dict: Dict[Any, Any]) -> None:
        """
        Sets the electrical node number for each node in the graph based on the provided nodal number dictionary.

        Parameters
        ----------
        nodal_number_dict : dict
            A dictionary mapping node IDs to their respective electrical node numbers.

        Notes
        -----
        This method updates the 'peripheries' attribute of each node in the graph based on the nodal numbers provided.
        """
        peripheries_dict = {node: nodal_number_dict[node] for node in self.g}

        nx.set_node_attributes(self.g, peripheries_dict, "peripheries")

    def plot(self, save_folder: str, name: str, state: str = "before", sim: Optional[Any] = None) -> None:
        """
        Plots the graph using the Printer class.

        Parameters
        ----------
        save_folder : str
            The folder where the plot will be saved.
        name : str
            The name of the plot.
        state : str, optional
            The state of the simulation to plot. Defaults to "before".
        sim : object, optional
            The simulator object, which may have a plot method. Defaults to None.

        Notes
        -----
        If a simulator object is provided and it has a plot method, this method will use the simulator's plot method.
        Otherwise, it will use the Printer class to display the graph.
        """
        printer = Printer(save_folder)

        # In case the simulator also provides a plot function, use it
        if sim is not None and hasattr(sim, 'plot'):
            output_name = printer.create_namefile("geo", name=name, type="base")
            if state == "before":
                obs = sim.obs
            else:
                obs = sim.obs_linecut
            sim.plot(obs, save_file_path=output_name[1])
        else:
            if self.layout:
                printer.display_geo(self.g, self.layout, name=name)
