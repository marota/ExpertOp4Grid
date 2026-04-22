"""NullFlowGraphMixin: null-flow redispatch handling for OverFlowGraph.

Extracted from ``alphaDeesp/core/graphs/overflow_graph.py`` to keep per-file
LOC and average cyclomatic complexity within A-grade bounds.

The mixin assumes the concrete class provides:
    self.g               — the overflow MultiDiGraph
    self.float_precision — format string (e.g. "%.2f")
"""

import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx

from alphaDeesp.core.graphs.graph_utils import (
    all_simple_edge_paths_multi,
    find_multidigraph_edges_by_name,
    nodepath_to_edgepath,
)
from alphaDeesp.core.graphs.null_flow import (
    add_double_edges_null_redispatch,
    remove_unused_added_double_edge,
)

logger = logging.getLogger(__name__)


class NullFlowGraphMixin:
    """Null-flow redispatch helpers; mixed into OverFlowGraph."""

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def _setup_null_flow_styles(
        self,
        non_connected_lines: List[Any],
        non_reconnectable_lines: List[Any],
    ) -> List[Any]:
        """Set dotted/dashed styles; return union of both line lists."""
        union_lines = list(set(non_connected_lines) | set(non_reconnectable_lines))

        edge_names = nx.get_edge_attributes(self.g, 'name')
        non_reconnectable_set = set(non_reconnectable_lines)
        non_connected_edges = {e for e, n in edge_names.items() if n in union_lines}
        non_reconnectable_edges = {e for e, n in edge_names.items() if n in non_reconnectable_set}
        reconnectable_edges = non_connected_edges - non_reconnectable_edges

        nx.set_edge_attributes(self.g, {e: {"style": "dotted"} for e in non_reconnectable_edges})
        nx.set_edge_attributes(self.g, {e: {"style": "dashed"} for e in reconnectable_edges})
        nx.set_edge_attributes(self.g, {e: "none" for e in non_reconnectable_edges}, "dir")
        return union_lines

    def add_relevant_null_flow_lines_all_paths(
        self,
        structured_graph: Any,
        non_connected_lines: List[Any],
        non_reconnectable_lines: List[Any] = [],
    ) -> None:
        """Apply null-flow logic for all four target-path strategies."""
        non_connected_lines = self._setup_null_flow_styles(non_connected_lines, non_reconnectable_lines)

        structural_info = self._structural_info_for_null_flow(structured_graph)

        for target_path in ["blue_amont_aval", "red_only", "blue_to_red", "blue_only"]:
            self.add_relevant_null_flow_lines(
                structured_graph, non_connected_lines, non_reconnectable_lines,
                target_path=target_path,
                _skip_style_setup=True,
                _structural_info=structural_info)

    def add_relevant_null_flow_lines(
        self,
        structured_graph: Any,
        non_connected_lines: List[Any],
        non_reconnectable_lines: List[Any] = [],
        target_path: str = "blue_to_red",
        depth_reconnectable_edges_search: int = 2,
        max_null_flow_path_length: int = 7,
        _skip_style_setup: bool = False,
        _structural_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Make null-flow edges bidirectional and recolour relevant ones."""
        if not _skip_style_setup:
            non_connected_lines = self._setup_null_flow_styles(
                non_connected_lines, non_reconnectable_lines)

        sets = self._prepare_null_flow_edge_sets(non_connected_lines, non_reconnectable_lines)

        edges_to_double, edges_double_added = add_double_edges_null_redispatch(self.g)

        edge_names = nx.get_edge_attributes(self.g, 'name')
        edges_non_connected_lines = {
            edge for edge, n in edge_names.items() if n in sets["non_connected_lines_set"]
        }
        edges_non_reconnectable_lines = {
            edge for edge, n in edge_names.items() if n in sets["non_reconnectable_lines_set"]
        }

        gray_components = self._build_gray_components()

        structural_info = _structural_info or self._structural_info_for_null_flow(structured_graph)

        edges_to_keep, edges_non_reconnectable = self._detect_edges_for_target_path(
            gray_components, target_path, structural_info,
            sets["edges_non_connected_lines_to_consider"],
            edges_non_connected_lines,
            edges_non_reconnectable_lines,
            depth_reconnectable_edges_search,
            max_null_flow_path_length,
        )

        self._apply_null_flow_recoloring(
            target_path, edges_to_keep, edges_non_reconnectable,
            edges_to_double, edges_double_added,
        )

    # ------------------------------------------------------------------
    # Preparation helpers
    # ------------------------------------------------------------------

    def _prepare_null_flow_edge_sets(
        self,
        non_connected_lines: List[Any],
        non_reconnectable_lines: List[Any],
    ) -> Dict[str, Any]:
        """Compute the input edge sets used by add_relevant_null_flow_lines."""
        non_connected_lines_set = set(non_connected_lines)
        non_reconnectable_lines_set = set(non_reconnectable_lines)
        edge_names = nx.get_edge_attributes(self.g, 'name')
        edge_colors = nx.get_edge_attributes(self.g, 'color')

        nodes_coloured = set()
        for edge, color in edge_colors.items():
            if color != "gray":
                nodes_coloured.add(edge[0])
                nodes_coloured.add(edge[1])

        edge_connex_names = set()
        for edge, color in edge_colors.items():
            if color == "gray" and (edge[0] in nodes_coloured or edge[1] in nodes_coloured):
                name = edge_names.get(edge)
                if name:
                    edge_connex_names.add(name)

        non_connected_lines_to_consider = non_connected_lines_set & edge_connex_names
        edges_non_connected_lines_to_consider = {
            edge for edge, n in edge_names.items() if n in non_connected_lines_to_consider
        }

        return {
            "non_connected_lines_set": non_connected_lines_set,
            "non_reconnectable_lines_set": non_reconnectable_lines_set,
            "edges_non_connected_lines_to_consider": edges_non_connected_lines_to_consider,
        }

    def _build_gray_components(self) -> List[Any]:
        """Return sorted weakly-connected components of gray-only edges as mutable copies."""
        _EXCLUDED_COLORS = frozenset({"coral", "blue", "black"})
        g_only_gray = nx.MultiDiGraph()
        for u, v, k, data in self.g.edges(keys=True, data=True):
            if data.get("color") not in _EXCLUDED_COLORS:
                g_only_gray.add_edge(u, v, key=k, **data)
        g_only_gray.remove_nodes_from(list(nx.isolates(g_only_gray)))

        return [
            g_only_gray.subgraph(c).copy()
            for c in sorted(
                nx.weakly_connected_components(g_only_gray), key=len, reverse=False)
        ]

    @staticmethod
    def _structural_info_for_null_flow(structured_graph: Any) -> Dict[str, Any]:
        """Extract red/amont/aval node sets once per call."""
        node_red_paths: Any = []
        if structured_graph.red_loops.Path.shape[0] != 0:
            node_red_paths = set(structured_graph.g_only_red_components.nodes)
        return {
            "node_red_paths": node_red_paths,
            "node_amont_constrained_path": structured_graph.constrained_path.n_amont(),
            "node_aval_constrained_path": structured_graph.constrained_path.n_aval(),
        }

    # ------------------------------------------------------------------
    # Per-strategy edge detection
    # ------------------------------------------------------------------

    def _detect_edges_for_target_path(
        self,
        gray_components: List[Any],
        target_path: str,
        structural_info: Dict[str, Any],
        edges_non_connected_lines_to_consider: Set[Any],
        edges_non_connected_lines: Set[Any],
        edges_non_reconnectable_lines: Set[Any],
        depth_reconnectable_edges_search: int,
        max_null_flow_path_length: int,
    ) -> Tuple[Set[Any], Set[Any]]:
        """Per-component dispatch to detect_edges_to_keep for the chosen strategy."""
        node_red_paths = structural_info["node_red_paths"]
        node_amont = structural_info["node_amont_constrained_path"]
        node_aval = structural_info["node_aval_constrained_path"]

        edges_to_keep: Set[Any] = set()
        edges_non_reconnectable: Set[Any] = set()

        def _run(g_c: Any, sources: Any, targets: Any) -> None:
            keep, non_rec = self.detect_edges_to_keep(
                g_c, sources, targets,
                edges_non_connected_lines, edges_non_reconnectable_lines,
                depth_edges_search=depth_reconnectable_edges_search,
                max_null_flow_path_length=max_null_flow_path_length)
            edges_to_keep.update(keep)
            edges_non_reconnectable.update(non_rec)

        for g_c in gray_components:
            if not edges_non_connected_lines_to_consider.intersection(set(g_c.edges)):
                continue
            if target_path == "blue_only":
                self._run_blue_only(g_c, node_amont, node_aval, _run)
            elif target_path == "blue_amont_aval":
                self._run_blue_amont_aval(g_c, node_amont, node_aval, _run)
            elif target_path == "red_only":
                self._run_red_only(g_c, node_red_paths, _run)
            elif target_path == "blue_to_red":
                self._run_blue_to_red(g_c, node_amont, node_aval, node_red_paths, _run)

        return edges_to_keep, edges_non_reconnectable

    @staticmethod
    def _run_blue_only(g_c: Any, node_amont: Any, node_aval: Any, _run: Any) -> None:
        edges_to_remove = [
            e for e, c in nx.get_edge_attributes(g_c, "capacity").items() if c > 0.
        ]
        g_c.remove_edges_from(edges_to_remove)
        _run(g_c, set(g_c).intersection(node_amont), set(g_c).intersection(node_amont))
        _run(g_c, set(g_c).intersection(node_aval), set(g_c).intersection(node_aval))

    @staticmethod
    def _run_blue_amont_aval(g_c: Any, node_amont: Any, node_aval: Any, _run: Any) -> None:
        _run(g_c, set(g_c).intersection(node_amont), set(g_c).intersection(node_aval))

    @staticmethod
    def _run_red_only(g_c: Any, node_red_paths: Any, _run: Any) -> None:
        edges_to_remove = [
            e for e, c in nx.get_edge_attributes(g_c, "capacity").items() if c < 0.
        ]
        g_c.remove_edges_from(edges_to_remove)
        intersect_red = set(g_c).intersection(node_red_paths)
        _run(g_c, intersect_red, intersect_red)

    @staticmethod
    def _run_blue_to_red(
        g_c: Any, node_amont: Any, node_aval: Any, node_red_paths: Any, _run: Any
    ) -> None:
        intersect_amont = set(g_c).intersection(node_amont)
        intersect_aval = set(g_c).intersection(node_aval)
        intersect_red = set(g_c).intersection(node_red_paths)
        if intersect_amont:
            _run(g_c, intersect_amont, intersect_red)
        if intersect_aval:
            _run(g_c, intersect_red, intersect_aval)
        if intersect_amont and intersect_aval:
            _run(g_c, intersect_amont, intersect_aval)

    def _apply_null_flow_recoloring(
        self,
        target_path: str,
        edges_to_keep: Set[Any],
        edges_non_reconnectable: Set[Any],
        edges_to_double: Dict[Any, Any],
        edges_double_added: Dict[Any, Any],
    ) -> None:
        """Paint detected edges and roll back unused doubled edges."""
        if target_path == "blue_only":
            edge_attributes = {edge: {"color": "blue"} for edge in edges_to_keep}
        elif target_path == "blue_to_red":
            current_weights = nx.get_edge_attributes(self.g, 'capacity')
            edge_attributes = {edge: {"color": "coral"} for edge in edges_to_keep}
            edge_attributes.update({
                edge: {"color": "blue"}
                for edge in edges_to_keep
                if current_weights[edge] < 0
            })
        else:
            edge_attributes = {edge: {"color": "coral"} for edge in edges_to_keep}

        edge_attributes.update({
            edge: {"color": "dimgray"}
            for edge in edges_non_reconnectable
            if self.g.edges[edge]["color"] == "gray"
        })

        nx.set_edge_attributes(self.g, edge_attributes)

        doubled_edges = set(edges_to_double.values()) | set(edges_double_added.values())
        edge_dirs = {edge: "none" for edge in edges_to_keep.intersection(doubled_edges)}
        nx.set_edge_attributes(self.g, edge_dirs, "dir")

        self.g = remove_unused_added_double_edge(
            self.g, edges_to_keep, edges_to_double, edges_double_added)

    # ------------------------------------------------------------------
    # detect_edges_to_keep and helpers
    # ------------------------------------------------------------------

    def detect_edges_to_keep(
        self,
        g_c: nx.MultiDiGraph,
        source_nodes: Iterable[Any],
        target_nodes: Iterable[Any],
        edges_of_interest: Set[Any],
        non_reconnectable_edges: List[Any] = [],
        depth_edges_search: int = 2,
        max_null_flow_path_length: int = 7,
    ) -> Tuple[Set[Any], Set[Any]]:
        """Detect edges of interest on short paths between source and target nodes."""
        prepared = self._prepare_detect_edges_inputs(
            g_c, source_nodes, target_nodes, edges_of_interest,
            non_reconnectable_edges, depth_edges_search)
        if prepared is None:
            return set(), set()

        sssp_paths_cache = self._compute_sssp_paths(g_c, prepared, edges_of_interest)
        paths_of_interest = self._collect_paths_of_interest(
            g_c, prepared, sssp_paths_cache, max_null_flow_path_length)
        return self._classify_paths_by_reconnectability(prepared, paths_of_interest)

    def _prepare_detect_edges_inputs(
        self,
        g_c: nx.MultiDiGraph,
        source_nodes: Iterable[Any],
        target_nodes: Iterable[Any],
        edges_of_interest: Set[Any],
        non_reconnectable_edges: List[Any],
        depth_edges_search: int,
    ) -> Optional[Dict[str, Any]]:
        """Build edge + node bookkeeping for detect_edges_to_keep; None on early exit."""
        edge_result = self._preprocess_gc_edges(g_c, edges_of_interest, non_reconnectable_edges)
        if edge_result is None:
            return None
        gc_edge_names, interest_in_gc, edge_names_of_interest, non_reconnectable_names = edge_result

        node_result = self._preprocess_gc_nodes(
            g_c, source_nodes, target_nodes, interest_in_gc, edge_names_of_interest, depth_edges_search)
        if node_result is None:
            return None
        source_in_gc, target_in_gc, node_has_interest, bfs_cache, targets_with_bfs, any_target_has_interest = node_result

        return {
            "g_c_edge_names_dict": gc_edge_names,
            "edges_of_interest_in_gc": interest_in_gc,
            "edge_names_of_interest": edge_names_of_interest,
            "non_reconnectable_edges_names": non_reconnectable_names,
            "source_nodes_in_gc": source_in_gc,
            "target_nodes_in_gc": target_in_gc,
            "node_has_incident_interest": node_has_interest,
            "bfs_cache": bfs_cache,
            "targets_with_bfs": targets_with_bfs,
            "any_target_has_interest": any_target_has_interest,
        }

    @staticmethod
    def _preprocess_gc_edges(
        g_c: nx.MultiDiGraph,
        edges_of_interest: Set[Any],
        non_reconnectable_edges: List[Any],
    ) -> Optional[Tuple[Any, ...]]:
        """Filter edges to those in g_c, flip negative capacities. None if nothing found."""
        gc_edge_names = nx.get_edge_attributes(g_c, "name")
        interest_in_gc = edges_of_interest & set(gc_edge_names.keys())
        if not interest_in_gc:
            return None

        edge_names_of_interest = {gc_edge_names[e] for e in interest_in_gc}
        non_reconnectable_names = {
            gc_edge_names[e] for e in non_reconnectable_edges if e in gc_edge_names
        }

        neg_caps = {
            e: {"capacity": -c}
            for e, c in nx.get_edge_attributes(g_c, "capacity").items()
            if c < 0
        }
        if neg_caps:
            nx.set_edge_attributes(g_c, neg_caps)

        return gc_edge_names, interest_in_gc, edge_names_of_interest, non_reconnectable_names

    @staticmethod
    def _preprocess_gc_nodes(
        g_c: nx.MultiDiGraph,
        source_nodes: Iterable[Any],
        target_nodes: Iterable[Any],
        interest_in_gc: Set[Any],
        edge_names_of_interest: Set[str],
        depth: int,
    ) -> Optional[Tuple[Any, ...]]:
        """Filter source/target nodes, build incident-interest map and BFS cache. None on early exit."""
        source_in_gc = [s for s in source_nodes if s in g_c]
        target_in_gc = [t for t in target_nodes if t in g_c]
        if not source_in_gc or not target_in_gc:
            return None

        unique_nodes = set(source_in_gc) | set(target_in_gc)
        node_has_interest = {
            n: bool(
                (set(g_c.out_edges(n, keys=True)) | set(g_c.in_edges(n, keys=True)))
                & interest_in_gc
            )
            for n in unique_nodes
        }
        if not any(node_has_interest.values()):
            return None

        bfs_cache = {
            n: find_multidigraph_edges_by_name(
                g_c, n, edge_names_of_interest, depth=depth, name_attr="name")
            for n in unique_nodes
        }
        targets_with_bfs = frozenset(t for t in target_in_gc if bfs_cache[t])
        any_target_has_interest = any(node_has_interest[t] for t in target_in_gc)

        return source_in_gc, target_in_gc, node_has_interest, bfs_cache, targets_with_bfs, any_target_has_interest

    def _compute_sssp_paths(
        self,
        g_c: nx.MultiDiGraph,
        prepared: Dict[str, Any],
        edges_of_interest: Set[Any],
    ) -> Dict[Any, Any]:
        """Run single-source Dijkstra per source with an incentivised weight function."""
        HUGE_MULTIPLIER = 1_000_000_000
        NORMAL_HOP_COST = 100
        PROMOTED_HOP_COST = 33
        promoted_set = set(edges_of_interest)

        def incentivized_weight(u: Any, v: Any, attr: Dict[str, Any]) -> float:
            real_weight = attr.get("capacity", 0)
            if real_weight < 0:
                raise ValueError("Negative weights not allowed.")
            hop_cost = PROMOTED_HOP_COST if (u, v) in promoted_set else NORMAL_HOP_COST
            return (real_weight * HUGE_MULTIPLIER) + hop_cost

        bfs_cache = prepared["bfs_cache"]
        targets_with_bfs = prepared["targets_with_bfs"]
        node_has_incident_interest = prepared["node_has_incident_interest"]
        any_target_has_interest = prepared["any_target_has_interest"]

        sssp_paths_cache: Dict[Any, Any] = {}
        for source_node in set(prepared["source_nodes_in_gc"]):
            if not node_has_incident_interest[source_node] and not any_target_has_interest:
                continue
            if not bfs_cache[source_node] and not targets_with_bfs:
                continue
            try:
                sssp_paths_cache[source_node] = nx.single_source_dijkstra_path(
                    g_c, source_node, weight=incentivized_weight)
            except Exception:
                sssp_paths_cache[source_node] = {}
        return sssp_paths_cache

    def _collect_paths_of_interest(
        self,
        g_c: nx.MultiDiGraph,
        prepared: Dict[str, Any],
        sssp_paths_cache: Dict[Any, Any],
        max_null_flow_path_length: int,
    ) -> List[Any]:
        """Materialise (source, target) paths that traverse at least one edge of interest."""
        source_nodes_in_gc = prepared["source_nodes_in_gc"]
        target_nodes_in_gc = prepared["target_nodes_in_gc"]
        bfs_cache = prepared["bfs_cache"]
        targets_with_bfs = prepared["targets_with_bfs"]
        node_has_incident_interest = prepared["node_has_incident_interest"]
        edges_of_interest_in_gc = prepared["edges_of_interest_in_gc"]

        paths_of_interest = []
        for source_node in source_nodes_in_gc:
            if source_node not in sssp_paths_cache:
                continue
            source_paths = sssp_paths_cache[source_node]
            source_has_bfs = bool(bfs_cache[source_node])
            source_has_interest = node_has_incident_interest[source_node]

            for target_node in target_nodes_in_gc:
                if source_node == target_node:
                    continue
                if not source_has_interest and not node_has_incident_interest[target_node]:
                    continue
                if not source_has_bfs and target_node not in targets_with_bfs:
                    continue
                path_nodes = source_paths.get(target_node)
                if not path_nodes or len(path_nodes) > max_null_flow_path_length:
                    continue
                path = nodepath_to_edgepath(g_c, path_nodes, with_keys=True)
                if any(edge in edges_of_interest_in_gc for edge in path):
                    paths_of_interest.append(path)

        paths_of_interest.sort(key=len)
        return paths_of_interest

    def _classify_paths_by_reconnectability(
        self,
        prepared: Dict[str, Any],
        paths_of_interest: List[Any],
    ) -> Tuple[Set[Any], Set[Any]]:
        """Greedy dedupe: attribute each edge name to the first (shortest) path that uses it."""
        gc_edge_names_dict = prepared["g_c_edge_names_dict"]
        edge_names_of_interest = prepared["edge_names_of_interest"]
        non_reconnectable_edges_names = prepared["non_reconnectable_edges_names"]

        edge_names_already_found: Set[str] = set()
        edges_to_keep_reconnectable: List[Any] = []
        edges_to_keep_non_reconnectable: List[Any] = []

        for path in paths_of_interest:
            fresh_edges = {
                edge for edge in path
                if gc_edge_names_dict[edge] not in edge_names_already_found
            }
            fresh_edge_names = {gc_edge_names_dict[edge] for edge in fresh_edges}

            if not (fresh_edge_names & edge_names_of_interest):
                continue

            if fresh_edge_names & non_reconnectable_edges_names:
                edges_to_keep_non_reconnectable += fresh_edges
            else:
                edges_to_keep_reconnectable += fresh_edges
            edge_names_already_found |= fresh_edge_names

        return set(edges_to_keep_reconnectable), set(edges_to_keep_non_reconnectable)
