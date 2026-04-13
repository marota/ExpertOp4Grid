"""Pure graph helper functions shared across the graphs package.

These helpers operate directly on NetworkX graphs and do not depend on
any of the graph *classes* defined in the package, which makes them
trivial to unit-test in isolation.
"""

import logging
from typing import Any, Iterable, List, Optional, Tuple

import networkx as nx

logger = logging.getLogger(__name__)


def from_edges_get_nodes(edges: Iterable[Any], amont_or_aval: str, constrained_edge: Any) -> List[Any]:
    """edges is a list of tuples"""
    if edges:
        nodes = []
        for e in edges:
            for node in e[:2]:
                if node not in nodes:
                    nodes.append(node)
        return nodes
    elif amont_or_aval == "amont":
        return [constrained_edge[0]]
    elif amont_or_aval == "aval":
        return [constrained_edge[1]]
    else:
        raise ValueError("Error in function from_edges_get_nodes")

def delete_color_edges(_g: nx.MultiDiGraph, edge_color: str) -> nx.MultiDiGraph:
    """
    Returns a copy of a graph without edges of a given color. Gray for instance, with values below a threshold of significance

    From a given node, get blue edges (with negative overflow redispatch) that are above this node

    Parameters
    ----------

    _g: :class:`nx:MultiDiGraph`
        an overflow redispatch networkx graph

    edge_color: ``str``
        color of edges to delete from graoh

    Returns
    ----------

    res: :class:`nx:MultiDiGraph`
        the graph without edges for the targeted color

    """
    g = _g.copy()

    TargetColor_edges = []
    i = 1
    for u, v,idx, color in g.edges(data="color",keys=True):
        if color == edge_color:
            TargetColor_edges.append((i, (u, v,idx)))
        i += 1

    # delete from graph gray edges
    # this extracts the (u,v) from pos_edges
    if TargetColor_edges:
        g.remove_edges_from(list(zip(*TargetColor_edges))[1])
        g.remove_nodes_from(list(nx.isolates(g)))
    return g

def nodepath_to_edgepath(G: Any, node_path: List[Any], with_keys: bool = False) -> List[Any]:
    """Convert a list of nodes into a list of edges for Graph/MultiGraph."""
    edges = []
    for u, v in zip(node_path[:-1], node_path[1:]):
        if with_keys and G.is_multigraph():
            # take the first key by default
            #k = next(iter(G[u][v].keys()))
            #take all keys
            for k in G[u][v].keys():
                edges.append((u, v, k))
        else:
            edges.append((u, v))
    return edges

def incident_edges(G: Any, node: Any, data: bool = True, keys: bool = False) -> List[Any]:
    if keys and G.is_multigraph():
        out_e = G.out_edges(node, keys=True, data=data)
        in_e  = G.in_edges(node, keys=True, data=data)
    else:
        out_e = G.out_edges(node, data=data)
        in_e  = G.in_edges(node, data=data)
    return list(out_e) + list(in_e)

def all_simple_edge_paths_multi(G: Any, sources: Iterable[Any], targets: Iterable[Any], cutoff: Optional[int] = None) -> Any:
    """
    Yield all simple edge paths between multiple sources and targets.

    Parameters
    ----------
    G : nx.Graph / nx.DiGraph / nx.MultiDiGraph
        Graph object.
    sources : iterable
        Set/list of source nodes.
    targets : iterable
        Set/list of target nodes.
    cutoff : int, optional
        Maximum path length.

    Yields
    ------
    path : list of edges (u, v) or (u, v, key) for multigraphs
    """
    for s in sources:
        for t in targets:
            if s != t and s in G and t in G:
                try:
                    for path in nx.all_simple_edge_paths(G, s, t, cutoff=cutoff):
                        yield path
                except nx.NetworkXNoPath:
                    continue

def find_multidigraph_edges_by_name(G: nx.MultiDiGraph, source_node: Any, target_names: Iterable[Any], depth: int = 2, name_attr: str = "name") -> List[Any]:
    """
    Traverses the MultiDiGraph using BFS up to 'depth'.
    For every connection (u, v) traversed, checks ALL parallel edges
    to see if their name is in 'target_names'.
    """
    # 1. Optimization: Set for O(1) lookup
    target_set = set(target_names)
    found_edges = []

    # 2. Lazy BFS Traversal
    # nx.bfs_edges yields (u, v) pairs representing the discovery path.
    # It yields (u, v) exactly once, even if there are multiple edges.
    for u, v in nx.bfs_edges(G, source_node, depth_limit=depth):

        # 3. Inspect ALL parallel edges between u and v
        # G[u][v] returns a dictionary of keys: {key1: {attr...}, key2: {attr...}}
        if G.has_edge(u, v):
            parallel_edges = G[u][v]

            for key, attributes in parallel_edges.items():
                edge_name = attributes.get(name_attr)

                # Check if this specific line is in our target list
                if edge_name in target_set:
                    found_edges.append(edge_name)

    return found_edges
