"""TopoApplicatorMixin: busbar-split graph-mutation helpers for AlphaDeesp.

Extracted from ``alphaDeesp/core/alphadeesp.py`` to keep per-file LOC and
average cyclomatic complexity within A-grade bounds.

The mixin assumes the concrete class provides:
    self.g              — the overflow MultiDiGraph
    self.df             — the overflow DataFrame (has "swapped" column)
    self.debug          — bool debug flag
    self.simulator_data — {"substations_elements": {node: [element, ...]}}
    self.bag_of_graphs  — dict accumulating named topology graphs
"""

import logging
from math import fabs
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx

from alphaDeesp.core.elements import Consumption, ExtremityLine, OriginLine, Production
from alphaDeesp.core.twin_nodes import twin_node_id

logger = logging.getLogger(__name__)


class TopoApplicatorMixin:
    """Graph mutation helpers for busbar-split topology application."""

    def apply_new_topo_to_graph(
        self,
        graph: nx.MultiDiGraph,
        new_topology: List[int],
        node_to_change: int,
    ) -> Tuple[nx.MultiDiGraph, Dict[Any, Any]]:
        """Apply a busbar reassignment to *graph* and return (graph, internal_repr_dict)."""
        if self.debug:
            logger.debug("apply_new_topo: topo=%s node=%s", new_topology, node_to_change)

        bus_ids = set(new_topology)
        assert len(bus_ids) != 0 and len(bus_ids) <= 2

        internal_repr_dict = dict(self.simulator_data["substations_elements"])
        new_node_id = twin_node_id(node_to_change)
        element_types = self.simulator_data["substations_elements"][node_to_change]
        assert len(element_types) == len(new_topology)

        color_edges = self._gather_edge_colors()

        if 1 in new_topology:
            graph.remove_node(node_to_change)

        for internal_elem, element in zip(internal_repr_dict[node_to_change], new_topology):
            internal_elem.busbar_id = element

        prod, load = self._compute_prod_load_per_bus(internal_repr_dict[node_to_change])
        self._add_bus_nodes(graph, bus_ids, prod, load, node_to_change, new_node_id)
        self._reconnect_bus_edges(
            graph, new_topology, element_types, node_to_change, new_node_id, color_edges)

        name = str(node_to_change) + "_" + "".join(str(e) for e in new_topology)
        self.bag_of_graphs[name] = graph
        return graph, internal_repr_dict

    def _gather_edge_colors(self) -> Dict[Tuple[Any, Any, Any], Any]:
        """Snapshot edge colours before the graph is mutated; also store reversed keys for swapped edges."""
        color_edges = {}
        for u, v, idx, color in self.g.edges(data="color", keys=True):
            condition = list(self.df.query(
                "idx_or == " + str(u) + " & idx_ex == " + str(v))["swapped"])[0]
            color_edges[(u, v, idx)] = color
            if condition:
                color_edges[(v, u, idx)] = color
        return color_edges

    @staticmethod
    def _compute_prod_load_per_bus(
        elements: Iterable[Any],
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """Sum production and consumption per busbar_id; returns (prod, load) dicts."""
        prod: Dict[int, float] = {}
        load: Dict[int, float] = {}
        for element in elements:
            bus = element.busbar_id
            if isinstance(element, Production):
                prod[bus] = prod.get(bus, 0) + fabs(element.value)
            elif isinstance(element, Consumption):
                load[bus] = load.get(bus, 0) + fabs(element.value)
        return prod, load

    @staticmethod
    def _classify_bus(
        bus_id: int,
        prod: Dict[int, float],
        load: Dict[int, float],
    ) -> Tuple[Optional[str], float]:
        """Return (kind, net_value) where kind is 'prod', 'load', or None."""
        has_prod = bus_id in prod
        has_load = bus_id in load
        if has_prod and has_load:
            diff = prod[bus_id] - load[bus_id]
            return ("prod" if diff > 0 else "load"), diff
        if has_prod:
            return "prod", prod[bus_id]
        if has_load:
            return "load", load[bus_id]
        return None, 0

    def _add_bus_nodes(
        self,
        graph: nx.MultiDiGraph,
        bus_ids: Iterable[int],
        prod: Dict[int, float],
        load: Dict[int, float],
        node_to_change: int,
        new_node_id: Any,
    ) -> None:
        """Re-add the (up to two) graph nodes styled by net prod/load balance."""
        for bus_id in bus_ids:
            node_label = new_node_id if bus_id == 1 else node_to_change
            kind, prod_minus_load = self._classify_bus(bus_id, prod, load)
            if kind == "prod":
                graph.add_node(node_label, pin=True, prod_or_load="prod",
                               value=str(prod_minus_load), style="filled", fillcolor="#f30000")
            elif kind == "load":
                graph.add_node(node_label, pin=True, prod_or_load="load",
                               value=str(-prod_minus_load), style="filled", fillcolor="#478fd0")
            else:
                graph.add_node(node_label, pin=True, prod_or_load="load",
                               value=str(prod_minus_load), style="filled", fillcolor="#ffffff")

    def _reconnect_bus_edges(
        self,
        graph: nx.MultiDiGraph,
        new_topology: List[int],
        element_types: List[Any],
        node_to_change: int,
        new_node_id: Any,
        color_edges: Dict[Tuple[Any, Any, Any], Any],
    ) -> None:
        """Re-wire every line element onto the correct busbar with the original colour."""
        for element, element_type in zip(new_topology, element_types):
            if not isinstance(element_type, (OriginLine, ExtremityLine)):
                continue
            if element not in (0, 1):
                raise ValueError("element must be 0 or 1 (busbar id).")

            reported_flow = element_type.flow_value[0]
            penwidth = fabs(reported_flow) / 10 or 0.1
            local_node = new_node_id if element == 1 else node_to_change

            if isinstance(element_type, OriginLine):
                color = color_edges[(node_to_change, element_type.end_substation_id, 0)]
                graph.add_edge(local_node, element_type.end_substation_id,
                               capacity=float("%.2f" % reported_flow),
                               label="%.2f" % reported_flow,
                               color=color, fontsize=10,
                               penwidth="%.2f" % penwidth)
            else:
                color = color_edges[(element_type.start_substation_id, node_to_change, 0)]
                graph.add_edge(element_type.start_substation_id, local_node,
                               capacity=float("%.2f" % reported_flow),
                               label="%.2f" % reported_flow,
                               color=color, fontsize=10,
                               penwidth="%.2f" % penwidth)
