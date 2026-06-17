# Changelog

## [0.3.2.post4] - 2026-06-17

### New Features

- **Readable node names in the interactive overflow viewer** (PR #78): the
  viewer now uses each node's human-readable display label (e.g. a
  voltage-level name such as `Saucats 400kV`, set via the Graphviz `label`
  node attribute by the upstream recommender) instead of only the raw node
  id. Search matches on **both** the readable display name and the stable id;
  the hover tooltip and selection panel show the readable name as the header
  with the id underneath only when it differs. Node identity (SVG `<title>` /
  `data-name`) is unchanged, so selection, adjacency highlight and
  double-click SLD resolution keep using the stable id.

### Bug Fixes

- **Do not leak the Graphviz `\N` placeholder**: label-less nodes carry the
  Graphviz `\N` ("use node name") placeholder in their label attribute;
  `nodeDisplayName` now ignores any backslash escape and falls back to the
  id, so tooltips/search never surface a literal `\N`.

### Tests

- `test_interactive_html.py`: readable label surfaces as `data-attr-label`
  without losing the id; search consults both id and resolved display name;
  tooltip/selection skip the duplicated `label` attribute; label-less nodes
  fall back to the id (no `\N` leak).

## [0.3.2.post2] - 2026-05-07

### New Features

- **Operator-supplied "extra lines to cut"** (PR #76): `OverFlowGraph` accepts an `extra_lines_to_cut` argument naming a subset of `lines_to_cut` that the operator chose to disconnect to prevent flow increase elsewhere (ExpertAgent's `additionalLinesToCut` semantic). Those edges keep their natural flow colour (coral/blue) instead of black/yellow, are stamped `is_extra_cut=True` (alongside `constrained=True`) so downstream consumers can find them by flag, and are excluded from `is_overload` / `is_monitored` tagging in `highlight_significant_line_loading`. The before% → after% loading annotation still fires on extras so the operator sees how their cut materialises.
- **Interactive HTML viewer layer**: new "Extra lines to prevent flow increase" semantic layer (`semantic:is_extra_cut`) with a dashed-blue swatch, surfaced under the "Individual entities properties" section.

### Tests

- `TestExtraLinesToCut` in `test_overflow_graph.py` covers default empty extras, set storage, natural coral/blue colour, `is_extra_cut`/`constrained` flagging, exclusion from overload/monitored layers while still annotating the label, and the legacy "no extras" path.
- `test_layer_index_emits_extra_cut_layer_with_endpoints` verifies the new viewer layer (endpoint nodes, swatch, label, section).
 
## [0.3.0.post1] - 2026-03-10
 
### New Features
 
- **Standardized overflow graph arrow sizes** (PR #66): Implemented linear scaling for `penwidth` in `OverFlowGraph` based on absolute delta flow. This ensures arrow sizes are reasonable and readable even for very high delta flows, with a standardized maximum penwidth.
- **Improved arrow scaling in PowerFlowGraph**: Also applied linear scaling to `PowerFlowGraph` for consistent visualization across different graph types.
 
### Bug Fixes
 
- **Fixed `collapse_red_loops` logic**: Optimized the node collapse condition to ensure ALL connected edges are coral before collapsing a node into a "point" shape.
 
### Tests
 
- Added unit tests for arrow scaling and `collapse_red_loops` in `test_graphs_and_paths_unit.py`.
- Adapted `test_visualization_filtering.py` in the recommender repo to verify `collapse_red_loops` call.
 

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
