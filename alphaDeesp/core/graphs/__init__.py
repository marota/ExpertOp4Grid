"""alphaDeesp.core.graphs package.

Public surface re-exports everything the legacy ``graphsAndPaths`` module
used to expose. Existing import paths keep working through the shim in
:mod:`alphaDeesp.core.graphsAndPaths`.
"""

from alphaDeesp.core.graphs.constants import default_voltage_colors
from alphaDeesp.core.graphs.graph_utils import (
    all_simple_edge_paths_multi,
    delete_color_edges,
    find_multidigraph_edges_by_name,
    from_edges_get_nodes,
    incident_edges,
    nodepath_to_edgepath,
)
from alphaDeesp.core.graphs.null_flow import (
    add_double_edges_null_redispatch,
    remove_unused_added_double_edge,
)
from alphaDeesp.core.graphs.shortest_paths import (
    shortest_path_mandatory_and_promoted,
    shortest_path_min_weight_then_hops,
    shortest_path_with_promoted_edges,
)
from alphaDeesp.core.graphs.power_flow_graph import PowerFlowGraph
from alphaDeesp.core.graphs.constrained_path import ConstrainedPath
from alphaDeesp.core.graphs.structured_overload_graph import (
    Structured_Overload_Distribution_Graph,
)
from alphaDeesp.core.graphs.overflow_graph import OverFlowGraph

__all__ = [
    "default_voltage_colors",
    "PowerFlowGraph",
    "OverFlowGraph",
    "ConstrainedPath",
    "Structured_Overload_Distribution_Graph",
    "from_edges_get_nodes",
    "delete_color_edges",
    "nodepath_to_edgepath",
    "incident_edges",
    "all_simple_edge_paths_multi",
    "find_multidigraph_edges_by_name",
    "add_double_edges_null_redispatch",
    "remove_unused_added_double_edge",
    "shortest_path_min_weight_then_hops",
    "shortest_path_mandatory_and_promoted",
    "shortest_path_with_promoted_edges",
]
