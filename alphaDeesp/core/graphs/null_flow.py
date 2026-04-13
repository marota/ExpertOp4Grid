"""Double / un-double null-flow edges in an overflow graph."""

from typing import Any, Dict, Set, Tuple

import networkx as nx


def remove_unused_added_double_edge(g: nx.MultiDiGraph, edges_to_keep: Set[Any], edges_to_double: Dict[Any, Any], edges_double_added: Dict[Any, Any]) -> nx.MultiDiGraph:

    """
    Make edges bi-directionnal when flow redispatch value is null

    Parameters
    ----------
     g: NetworkX graph
      graph on which to remove edges
     edges_to_keep: ``set`` str
        set of edges of interest found on paths and to be recoloured

    edges_to_double: ``dict`` str: networkx edge
        original set of edges that has been doubled, with line names as key and edge as value

    edges_double_added: ``dict`` str: networkx edge
        new set of edges that doubles the original ones in the other direction, with line name as key and edge as value

    Return
    ----------------
    g: NetworkX graph
      graph on which edges where removed
    """
    # Get names of kept edges directly (replaces edge_subgraph creation)
    name_edges_to_keep = set()
    for edge in edges_to_keep:
        name = g.edges[edge].get("name") if g.has_edge(*edge) else None
        if name is not None:
            name_edges_to_keep.add(name)

    # for initial edges that has not been recoloured but for which the added double edge has been, remove those initial edges
    edges_to_remove = []
    edge_names_to_remove = set()
    for name, edge in edges_to_double.items():
        if name in name_edges_to_keep and g.edges[edge]["color"] == "gray":
            edges_to_remove.append(edge)
            edge_names_to_remove.add(name)

    # for added double edges that has not been recoloured, remove them
    edges_to_remove += [edge for name, edge in edges_double_added.items() if name not in edge_names_to_remove]
    assert (len(edges_to_remove) == len(edges_to_double))
    g.remove_edges_from(edges_to_remove)

    return g

def add_double_edges_null_redispatch(g: nx.MultiDiGraph, color_init: str = "gray", only_no_dir: bool = False) -> Tuple[Dict[Any, Any], Dict[Any, Any]]:
    """
    Make edges bi-directionnal when flow redispatch value is null

    Parameters
    -------------
    g: NetworkX graph
      graph on which to add edges

    only_no_dir: bool
        condition to restrict edge doubling at no_dir case

    Returns
    ----------
    edges_to_double_name_dict: ``dict`` str: networkx edge
        original set of edges that has been doubled, with line names as key and edge as value

    edges_added_name_dict: ``dict`` str: networkx edge
        new set of edges that doubles the original ones in the other direction, with line name as key and edge as value

    """
    # Single pass over edges to find those to double (replaces 4 nx.get_edge_attributes calls)
    to_double = []
    edges_to_double_name_dict = {}
    for u, v, k, data in g.edges(keys=True, data=True):
        if data.get("color") == color_init and data.get("capacity") == 0.:
            if not only_no_dir or "dir" in data:
                to_double.append((u, v, k, data))
                edges_to_double_name_dict[data["name"]] = (u, v, k)

    # Add reverse edges and directly track new edge keys
    edges_added_name_dict = {}
    for u, v, k, data in to_double:
        new_key = g.add_edge(v, u, **data)
        edges_added_name_dict[data["name"]] = (v, u, new_key)

    assert(set(edges_to_double_name_dict.keys())==set(edges_added_name_dict.keys()))
    return edges_to_double_name_dict,edges_added_name_dict
