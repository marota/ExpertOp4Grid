# Changelog

## [0.3.0] - 2026-02-10

### New Features

- **Filter graph components without overloads** (PR #64): Added `keep_overloads_components()` method to `OverFlowGraph` that filters out connected components not containing any overloaded (black) edges by recolouring them to grey. This ensures only components with actual overloads are considered significant in further analysis.

- **Collapse coral-only loop nodes** (PR #65): Added `collapse_red_loops()` method to `OverFlowGraph` that collapses nodes purely part of coral-only loops into "point" shapes for cleaner visualization. A node is collapsed when all its edges are coral, its shape is "oval", it has no peripheries attribute, and none of its edges are dashed or dotted.

### Performance Improvements

- **Optimized null flow detection** (PR #63): Major refactoring and performance optimization of the null flow line detection pipeline:
  - Extracted `_setup_null_flow_styles()` to avoid redundant style setup across target path iterations
  - Pre-compute structural info (red loop nodes, constrained path endpoints) once and reuse across all target paths
  - Replace `delete_color_edges` graph copy operations with direct edge color computation for connex analysis
  - Build gray component graph in a single edge pass instead of 3 sequential copy+remove operations
  - Use single-source Dijkstra per source node instead of per source-target pair, reducing from O(S*T) to O(S) Dijkstra calls
  - Pre-filter edges of interest to component membership for early exits
  - Cache BFS results and incident edge lookups to eliminate redundant computation
  - Flip negative capacities once globally instead of per source-target pair
  - Optimized `remove_unused_added_double_edge` to avoid subgraph creation
  - Optimized `add_double_edges_null_redispatch` from 4 `nx.get_edge_attributes` calls to a single edge pass
  - Increased `max_null_flow_path_length` default from 5 to 7
  - Properly handle double edges when finding relevant null flow paths

### Tests

- Added 78 unit tests for `graphsAndPaths` helper functions and `detect_edges_to_keep` (PR #63), covering: `delete_color_edges`, `nodepath_to_edgepath`, `incident_edges`, `from_edges_get_nodes`, `find_multidigraph_edges_by_name`, `shortest_path_with_promoted_edges`, `all_simple_edge_paths_multi`, `add_double_edges_null_redispatch`, `remove_unused_added_double_edge`, `detect_edges_to_keep`, `ConstrainedPath` methods, and `shortest_path_min_weight_then_hops`
- Added 13 unit tests for `keep_overloads_components` (PR #64), covering: component with/without overloads, multiple mixed components, already-gray edges, empty graph, single edges, parallel edges, gray bridge not merging components, three components, and idempotency

## [0.2.8]

### Performance Improvements

- Speed up path findings with smarter graph algorithms and rustworkx (PR #61)
- Speed up AlphaDeesp representation creation: optimize `create_and_fill_internal_structures`, accelerate `get_model_obj_from_or` and `get_model_obj_from_ext` (PR #62)
- Added `rustworkx` as a dependency for faster graph operations

### Features

- Added ability to consider undirected edges in loop paths (PR #59)
- Extended topology scoring to reduced specified action space with warm-start support (PR #58)
- Exposed depth and breadth search parameters for reconnectable lines in upper functions

## [0.2.7]

- Highlight non-connected paths with additional blue edge path info (PR #57)
- Ambiguous path handling improvements (PR #55, #56)
- Highlight possible overloads (PR #53, #54)
- Display reconnectable lines (PR #52)
- Improved graphviz constrained path visualization (PR #51)
- Getting started guide and examples (PR #50)
- Refactored plotting (PR #47, #48)
