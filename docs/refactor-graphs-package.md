# Refactor: `graphsAndPaths` → `alphaDeesp.core.graphs` package

| Field  | Value                                                                    |
| ------ | ------------------------------------------------------------------------ |
| Status | Completed                                                                |
| Scope  | `alphaDeesp/core/graphsAndPaths.py`                                      |
| Branch | `claude/refactor-graph-module-fV6j5`                                     |
| Impact | Zero breaking change — legacy import path kept alive via a re-export shim |

## Motivation

`alphaDeesp/core/graphsAndPaths.py` had grown into a 2295-line monolith
holding four classes and a dozen module-level helpers. That shape made the
file hard to read, hard to navigate, and hard to test:

- a bug in, say, the null-flow doubling helper required scrolling past
  several hundred lines of unrelated class methods;
- unit tests against pure graph helpers had to import the entire module
  (pulling in `pandas`, `rustworkx`, `Printer`, the `OverFlowGraph`
  class, etc.) just to exercise `delete_color_edges`;
- adding a new helper or a new graph class meant editing the same giant
  file everyone else was editing — a recipe for merge conflicts;
- the public surface was implicit: there was no `__all__` and callers had
  to guess what was part of the API and what was internal.

The refactor splits the monolith into a focused package of
single-responsibility modules while **preserving behaviour byte-for-byte**.

## Design principles applied

- **Single Responsibility Principle.** Each sub-module holds exactly one
  class or one tightly-related group of helpers. Any given "reason to
  change" lands in exactly one file.
- **Strangler-fig / façade pattern.** `graphsAndPaths.py` is not deleted —
  it becomes a 49-line shim that re-exports everything from the new
  package. Every existing `from alphaDeesp.core.graphsAndPaths import X`
  keeps working. New code can progressively migrate to the direct import
  paths.
- **Explicit, acyclic dependency graph.** Sub-modules are layered so
  nothing imports upwards: `constants` → `graph_utils` / `null_flow` /
  `shortest_paths` → `constrained_path` → `structured_overload_graph` →
  `power_flow_graph` → `overflow_graph`.
- **Information hiding through layering.** Pure graph algorithms
  (`graph_utils`, `null_flow`, `shortest_paths`) have zero dependency on
  `Printer`, on `pandas.DataFrame`, or on power-flow domain concepts,
  which makes them trivial to unit-test with hand-built `nx.MultiDiGraph`
  fixtures.
- **Behaviour-preserving move, not rewrite.** Class and helper bodies
  were sliced byte-for-byte from the original file via a one-shot
  extraction script. No algorithmic change was introduced in the refactor
  itself — which is what makes it cheap to review and cheap to trust.
- **Explicit public surface.** Both the package `__init__` and the legacy
  shim define an explicit `__all__` listing the 16 exported names.
  Anything not in that list is internal.

## New layout

```text
alphaDeesp/core/graphs/
├── __init__.py                    # public re-exports + __all__
├── constants.py                   # default_voltage_colors
├── graph_utils.py                 # pure graph helpers
├── null_flow.py                   # null-flow edge doubling helpers
├── shortest_paths.py              # mandatory/promoted Dijkstra helpers
├── power_flow_graph.py            # PowerFlowGraph
├── constrained_path.py            # ConstrainedPath
├── structured_overload_graph.py   # Structured_Overload_Distribution_Graph
└── overflow_graph.py              # OverFlowGraph
alphaDeesp/core/graphsAndPaths.py  # backwards-compat shim
```

### Module contents

**`constants`** — `default_voltage_colors` mapping `kV → graphviz colour`.

**`graph_utils`** — pure helpers that operate on `networkx` graphs without
any knowledge of the surrounding power-system domain:

- `delete_color_edges`
- `nodepath_to_edgepath`
- `incident_edges`
- `all_simple_edge_paths_multi`
- `from_edges_get_nodes`
- `find_multidigraph_edges_by_name`

**`null_flow`** — helpers for making null-flow-redispatch edges
bidirectional and for rolling back unused doubled edges:

- `add_double_edges_null_redispatch`
- `remove_unused_added_double_edge`

**`shortest_paths`** — constrained shortest-path helpers built on Dijkstra
with a composite weight function (minimise physical weight, then favour
promoted edges, then minimise hop count):

- `shortest_path_min_weight_then_hops`
- `shortest_path_mandatory_and_promoted`
- `shortest_path_with_promoted_edges`

**`power_flow_graph`** — `PowerFlowGraph`, a coloured view of the current
grid state (productions, consumptions, flows) built from a topology dict.

**`constrained_path`** — `ConstrainedPath`, the main "constrained path"
object: amont edges, the overloaded edge itself, and the aval edges.

**`structured_overload_graph`** — `Structured_Overload_Distribution_Graph`,
takes a raw overflow graph and extracts its structural elements:
constrained path, loop paths, and hub nodes.

**`overflow_graph`** — `OverFlowGraph`, coloured overflow-redispatch graph.
Subclasses `PowerFlowGraph`. Holds most of the mass of the original file
(consolidation, null-flow line detection, ambiguous-path desambiguation,
plotting).

## Dependency graph

```text
constants  ────────────────────────────────────────────┐
                                                       │
graph_utils ──────┐                                    │
                  │                                    │
null_flow ────────┼──► structured_overload_graph       │
                  │          ▲                         │
shortest_paths    │          │                         │
                  │          │                         │
constrained_path ─┘          │                         │
                             │                         ▼
                             └──► overflow_graph ──► power_flow_graph
```

No sub-module imports anything from a module above it in this diagram,
which guarantees there are no import cycles. The layering is verified
automatically by `TestGraphsPackageStructure` in
`alphaDeesp/tests/test_graphs_package.py`.

## Backwards compatibility

Every consumer of the legacy module — `alphadeesp.py`,
`expert_operator.py`, `Grid2opSimulation.py`,
`Expert_rule_action_verification.py`, and every test file — keeps
working **unchanged**, because `alphaDeesp/core/graphsAndPaths.py`
re-exports all public names from `alphaDeesp.core.graphs`:

```python
# legacy (still works)
from alphaDeesp.core.graphsAndPaths import OverFlowGraph, ConstrainedPath

# preferred for new code
from alphaDeesp.core.graphs import OverFlowGraph, ConstrainedPath

# fine-grained (for unit tests of a single helper family)
from alphaDeesp.core.graphs.graph_utils import delete_color_edges
from alphaDeesp.core.graphs.null_flow import add_double_edges_null_redispatch
```

Instance identity is preserved across import paths: for every public
symbol `X`, `alphaDeesp.core.graphsAndPaths.X is alphaDeesp.core.graphs.X`.

## Migration guide

**Existing code.** No action required. The shim keeps the old import
path alive.

**New code and new tests.** Prefer importing from `alphaDeesp.core.graphs`
directly. When writing unit tests for a pure helper (for example
`delete_color_edges`), import it from its sub-module so the test pulls in
the smallest possible dependency surface:

```python
from alphaDeesp.core.graphs.graph_utils import delete_color_edges
```

**Extending the package.** Add a new class or helper to the most
appropriate sub-module. If it does not fit any existing module, create a
new one following the same layering rule (only import "downwards" in the
dependency graph). Remember to:

1. add the new symbol to `alphaDeesp/core/graphs/__init__.py`'s `__all__`
   and its re-export list;
2. mirror the addition in the legacy shim
   `alphaDeesp/core/graphsAndPaths.py` so the backwards-compat import
   path keeps working;
3. add tests in `alphaDeesp/tests/test_graphs_package.py` (for structural
   invariants) and in `alphaDeesp/tests/test_graphs_and_paths_unit.py`
   (for behaviour).

## Verification

The refactor is covered by two test files:

**`alphaDeesp/tests/test_graphs_and_paths_unit.py`** — pre-existing unit
tests for every helper and every major class method. 109 tests, all
passing against the new layout without any modification. This is the
primary "behaviour is preserved" guarantee.

**`alphaDeesp/tests/test_graphs_package.py`** — new test suite added
alongside the refactor. Covers:

- structural invariants of the refactor (shim parity, public-API
  stability, sub-module boundaries, import acyclicity);
- behavioural coverage of classes and helpers that were under-tested in
  the legacy file: `PowerFlowGraph` construction,
  `Structured_Overload_Distribution_Graph` end-to-end on a small
  hand-built graph, and `shortest_path_mandatory_and_promoted`.

Run both suites together with:

```bash
pytest alphaDeesp/tests/test_graphs_and_paths_unit.py \
       alphaDeesp/tests/test_graphs_package.py -v
```

## Metrics

| File                                    | Before | After |
| --------------------------------------- | -----: | ----: |
| `graphsAndPaths.py`                     |  2295  |   49  |
| `graphs/__init__.py`                    |    —   |   50  |
| `graphs/constants.py`                   |    —   |   13  |
| `graphs/graph_utils.py`                 |    —   |  148  |
| `graphs/null_flow.py`                   |    —   |   90  |
| `graphs/shortest_paths.py`              |    —   |  226  |
| `graphs/power_flow_graph.py`            |    —   |  242  |
| `graphs/constrained_path.py`            |    —   |   75  |
| `graphs/structured_overload_graph.py`   |    —   |  318  |
| `graphs/overflow_graph.py`              |    —   | 1284  |

The net +200 lines over the original monolith is made up of the added
per-file module docstrings and import blocks. No class or helper body was
rewritten.
