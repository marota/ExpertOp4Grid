# Code Quality & Maintainability Analysis

_Last updated: 2026-04-12 — high-impact items 5, 6, 7 and 9 are now all
resolved on branch `claude/fix-code-quality-issues-XGpMU` (type hints on
`core/simulation.py` and `core/elements.py`; remaining mutable defaults on
`Grid2opSimulation.__init__` / `PypownetSimulation.__init__`; items 5 and 7
were already done in prior passes and are reconfirmed). Longer-term refactors
for the four highest-CC functions and the `"666"` twin-node encoding landed
on branch `claude/refactor-rank-topo-function-U4y9g`. Short-term action plan
landed on `claude/code-quality-short-term-tnydm`; original immediate-cleanup
pass on `claude/code-quality-analysis-8Ftgi`._

This document captures a diagnostic review of the `alphaDeesp` (ExpertOp4Grid)
codebase. It is intended as a living punch-list for incremental cleanup work.
Numbers were produced with `pyflakes`, `radon cc/mi/raw`, and targeted `grep`
audits over the entire `alphaDeesp/` package.

## Cleanup progress

Items 11–13 from the longer-term refactor list are now done (in addition
to every "Immediate" and "Short-term" item). The four highest-CC functions
on the whole codebase have been decomposed into small helpers, each with
dedicated unit tests, and the string-prefix `"666"` twin-node encoding is
gone. The pyflakes CI scope (`alphaDeesp/core/**`, `expert_operator.py`,
`main.py`, excluding the legacy Pypownet backend) is still clean and
**130/130** tests across `test_graphs_and_paths_unit.py` (109) and the
new `test_alphadeesp_unit.py` (21) pass locally.

### Longer-term refactor landing (2026-04-12)

| Function | Before | After | Helpers |
|---|---|---|---|
| `AlphaDeesp.rank_current_topo_at_node_x` | F (68) | **B (7)** | `_pick_interesting_bus_id`, `_collect_flows_on_bus`, `_score_amont`, `_score_aval`, `_score_in_red_loop`, `_score_not_connected_to_cpath` (max helper CC 15) |
| `AlphaDeesp.apply_new_topo_to_graph` | E (36) | **B (7)** | `_gather_edge_colors`, `_compute_prod_load_per_bus`, `_classify_bus`, `_add_bus_nodes`, `_reconnect_bus_edges` (max helper CC 7) |
| `OverFlowGraph.detect_edges_to_keep` | F (46) | **A (2)** | `_prepare_detect_edges_inputs`, `_compute_sssp_paths`, `_collect_paths_of_interest`, `_classify_paths_by_reconnectability` (max helper CC 20) |
| `OverFlowGraph.add_relevant_null_flow_lines` | F (44) | **B (7)** | `_prepare_null_flow_edge_sets`, `_build_gray_components`, `_structural_info_for_null_flow`, `_detect_edges_for_target_path`, `_apply_null_flow_recoloring` (max helper CC 15) |

Semantics were preserved line-for-line — the decomposition is a pure
extract-method refactor with the following three *intentional* cleanups
absorbed along the way:

1. The original `rank_current_topo_at_node_x` declared four
   `in/out_{positive,negative}_flows` lists at the top of the function and
   reused them across the amont/aval branches. Each branch helper now owns
   a fresh dict of lists, which is behaviourally identical (the branches
   are mutually exclusive and each one completely rewrote the lists it
   used) but makes the dead-code sharing go away.
2. `apply_new_topo_to_graph` had a copy of the "second time, reparse
   element_types" loop that used to track two unused ``i`` counters; the
   unified ``_reconnect_bus_edges`` helper drops the counters and
   consolidates the ``element == 1`` / ``element == 0`` branches through a
   single ``local_node`` binding. The `ValueError` for "neither busbar"
   is raised *before* the add_edge calls, which was previously unreachable
   because the two branches covered every path and never reached the else.
3. `add_relevant_null_flow_lines` used to duplicate the pre-iteration
   structural-info computation (red/amont/aval nodes) between the
   ``all_paths`` wrapper and the individual calls; the structural info
   builder is now a dedicated helper on the class, callable from both
   entry points.

Each extracted helper has at least one targeted unit test; see
`alphaDeesp/tests/test_alphadeesp_unit.py` and the new
`TestDetectEdgesHelpers`/`TestNullFlowHelpers` classes in
`alphaDeesp/tests/test_graphs_and_paths_unit.py`.

### Twin-node id scheme

The string-prefix encoding ``int("666" + str(node_to_change))`` used to
mint a "twin" node id when a substation was split onto two busbars has been
replaced with a proper additive scheme in
`alphaDeesp/core/twin_nodes.py`:

- ``TWIN_NODE_OFFSET = 10_000_000`` — chosen to exceed any realistic
  substation id on the grids AlphaDeesp targets while fitting inside a
  32-bit integer for graphviz round-trips.
- ``twin_node_id(sub_id)`` returns ``TWIN_NODE_OFFSET + sub_id`` and
  raises ``ValueError`` on negative or too-large inputs so we can never
  silently collide with a real substation id again.
- ``is_twin_node_id(node_id)`` / ``original_substation_id(twin_id)`` give
  the two decoding primitives the old ``str(...)[3:]`` slice was pretending
  to provide.

The five string-prefix call sites (`alphadeesp.py`, `network.py`
×2 — build + decode, `printer.py` ×2) all go through the helpers now. The
old scheme broke for any substation id ≥ 1000 (because the decoder did an
unconditional 3-character slice); the new one works uniformly. Round-trip
and discrimination are covered in `TestTwinNodeIds`.

| Status | Item | Notes |
|---|---|---|
| done | Replace all 9 bare `except:` with specific exceptions | See "Bare except cleanup" below |
| done | Delete 15+ unused locals/imports flagged by pyflakes | Extended to star-import cleanup where necessary |
| done | Fix mutable defaults in `alphadeesp.py` | `__init__` + `rank_current_topo_at_node_x` |
| done | Drop unconditional `print(df)` from `Simulation.create_df` | Now gated on `self.debug`; now routed through `logger.debug` |
| done | Enable `pyflakes` in CI | New lint step in `.circleci/config.yml` before tests |
| done | Replace `from elements import *` with explicit imports | Completed in the immediate pass — `alphadeesp.py`, `network.py`, `PypownetSimulation.py` |
| done | Introduce module-level `logger` and convert `print()` to `logging` | See "Logging migration" below |
| done | Write real docstrings for every abstract method in `core/simulation.py` | Replaced 9 `"""TODO"""` stubs with full contract documentation |
| done | Re-enable the three excluded test modules in CI (or document why) | Tests now self-skip via `pytest.importorskip`; `--ignore` flags removed |
| done | Unify `LICENSE`/`LICENSE.md`; refresh `setup.py` classifiers and version pins | See "Packaging cleanup" below |
| done | Fix remaining mutable defaults on `Grid2opSimulation.__init__` / `PypownetSimulation.__init__` | See "Mutable defaults" below |
| done | Type hints seed on `core/simulation.py` and `core/elements.py` | See "Type hints seed" below |

### Bare except cleanup

All 9 bare `except:` clauses are replaced with narrow exception types. The
fallbacks in these locations are *expected control flow* (missing config key,
optional backend, alternate observation shape), so they are narrowed to
specific exceptions instead of reaching for `logging.exception()` — the latter
is for unexpected errors and would add misleading stack traces for the
config-fallback paths.

| File | Line(s) | Was | Now |
|---|---|---|---|
| `alphaDeesp/main.py` | 100, 107, 123 | `except:` | `except KeyError:` (missing config key) |
| `alphaDeesp/tests/test_expert_op.py` | 25 | `except:` | `except KeyError:` (missing config key) |
| `alphaDeesp/core/pypownet/PypownetSimulation.py` | 68 | `except:` | `except (KeyError, ValueError, SyntaxError):` (missing or malformed `CustomLayout`) |
| `alphaDeesp/core/grid2op/Grid2opSimulation.py` | 34 | `except:` | `except (KeyError, ValueError, SyntaxError):` + `logger.debug` on the fallback |
| `alphaDeesp/core/grid2op/Grid2opSimulation.py` | 39 | `except:` | `except AttributeError:` + `logger.debug` (no `grid_layout` on obs) |
| `alphaDeesp/core/grid2op/Grid2opObservationLoader.py` | 24 | `except:` | `except ImportError:` (lightsim2grid not installed) |
| `alphaDeesp/core/grid2op/Grid2opObservationLoader.py` | 47 | `except:` | `except (TypeError, ValueError):` (non-int chronic_scenario argument) |

`Grid2opSimulation.compute_layout` was additionally restructured from two
nested `try/except` blocks (which trapped exceptions from the *inner* try
inside the outer) into a sequence of early `return`s, so that an
`AttributeError` from `self.obs.grid_layout.values()` no longer leaks into the
outer `(KeyError, ValueError, SyntaxError)` handler. A module-level
`logger = logging.getLogger(__name__)` was introduced in that file as the
first toehold for the short-term logging migration.

### Pyflakes cleanup

After the edits, pyflakes reports **0 findings** for all files in the CI lint
scope. Concretely removed (mechanical, no behavior changes):

- `alphaDeesp/core/alphadeesp.py` — dropped unused `pprint`, `math.ceil`, `os`
  imports; dead locals `ranked_combinations` (line ~48), `current_node` +
  `new_node` (~262-263), `edge_color` (~325), two copies of
  `all_nodes_value_attributes` (~373, ~744), `not_interesting_bus_id` (~417),
  `node2` (~542), `indexEdge_inDf` (~786-790), `p` (~808). Replaced
  `from alphaDeesp.core.elements import *` with an explicit import of
  `Consumption`, `ExtremityLine`, `OriginLine`, `Production`.
- `alphaDeesp/core/graphsAndPaths.py` — dropped unused `NetworkXNoPath` and
  `itertools` imports; dead locals `edges_to_add_data` (~762),
  `nodes_interest` (~1052), `g_c_names_edge_dict` (~1205), `target_set`
  (~1258), `tmp_constrained_path` (~1661).
- `alphaDeesp/core/network.py` — replaced `from alphaDeesp.core.elements
  import *` with the same explicit import; this also clears the star-import
  "may be undefined" warnings that previously appeared for every
  `isinstance(element, Production)` / `Consumption` / `OriginLine` /
  `ExtremityLine` call in the file.
- `alphaDeesp/core/printer.py` — dropped unused `pprint`, `pydot`,
  `pathlib.Path` imports; simplified `execute_command` to only bind the
  variables it actually uses (`_stdout`, `stderr`, `error`).
- `alphaDeesp/core/grid2op/Grid2opSimulation.py` — dropped unused `pprint`,
  `grid2op.dtypes.dt_int`, and the `OverFlowGraph` re-export from
  `graphsAndPaths`; removed dead local `redistribution_prod` (~816).
- `alphaDeesp/core/pypownet/PypownetSimulation.py` — dropped
  `import pypownet.environment` (the module is imported *and* star-imported
  below, so this top-level import was unused); removed unused local
  `observation_space` in `__init__` and `redistribution_prod` in
  `compute_new_network_changes`; replaced `from alphaDeesp.core.elements
  import *` with an explicit import (the pypownet.agent star import is still
  there — see CI scope caveat below). This file is *not* in the pyflakes CI
  scope.
- `alphaDeesp/Expert_rule_action_verification.py` — dropped unused
  `configparser`, `Grid2opObservationLoader` imports, and the duplicate
  `version_packaging` import. This file is *not* in the pyflakes CI scope
  (it imports several project-external modules — `make_evaluation_env`,
  `pypowsybl`, `load_training_data` — that may not be installed in CI).

Where the dead local was the only reason a statement existed (e.g.
`target_set = set(target_nodes_in_gc)` in `graphsAndPaths.py` when
`target_set` was never read), the statement was deleted entirely rather than
renamed to `_`.

### Mutable defaults

Two mutable defaults in `alphaDeesp/core/alphadeesp.py`:

- `AlphaDeesp.__init__(self, ..., substation_in_cooldown=[], debug=False)` →
  `substation_in_cooldown=None`, normalized inside the body to `[]` when
  `None`.
- `rank_current_topo_at_node_x(..., topo_vect=[0, 0, 1, 1, 1], ...)` →
  `topo_vect=None`, normalized inside the body to `[0, 0, 1, 1, 1]` when
  `None`.

The `ltc=[9]` / `other_ltc=[]` defaults on `Grid2opSimulation.__init__` and
the `ltc=[9]` default on `PypownetSimulation.__init__` are now also fixed.
Both signatures take `ltc=None` / `other_ltc=None` and normalize to `[9]` /
`[]` inside the body; all in-tree callers pass these as keyword arguments
(verified across `main.py`, `agent_call.py`, the three grid2op test
modules, `Expert_rule_action_verification.py` and the four
`getting_started/` notebooks), so the signature change is source-compatible
for embedders too.

### Type hints seed

`core/elements.py` and `core/simulation.py` — the two modules the original
audit flagged as the highest-leverage starting points — are now fully
type-annotated. The rest of the package is still untyped; the two files
above are intended as a **seed** the rest of the code can be annotated
against without chasing `Any` through every call site.

What landed:

- `core/elements.py`:
  - `Production.__init__(self, busbar_id: int, value: Optional[float] = None) -> None`
  - `Consumption.__init__(self, busbar_id: int, value: Optional[float] = None) -> None`
  - `OriginLine.__init__(self, busbar_id: int, end_substation_id: Optional[int] = None, flow_value: Optional[List[float]] = None) -> None`
  - `ExtremityLine.__init__(self, busbar_id: int, start_substation_id: Optional[int] = None, flow_value: Optional[List[float]] = None) -> None`
  - `ID: int` class variable annotation on each of the four classes.
  - `busbar` property / setter typed as `int` on all four classes.
  - `__repr__(self) -> str` on all four classes.
  - The three ``# print("... created...")`` commented-out debug stubs that
    had been sitting at the top of each ``__init__`` were dropped along the
    way — they were dead and made the type annotations noisier to read.

- `core/simulation.py`:
  - `Simulation.__init__(self) -> None`.
  - All 10 abstract methods carry concrete return types:
    `cut_lines_and_recomputes_flows -> Sequence[float]`,
    `isAntenna -> Optional[int]`,
    `isDoubleLine -> Optional[List[int]]`,
    `getLinesAtSubAndBusbar -> Dict[Any, List[int]]`,
    `get_layout -> List[Tuple[float, float]]`,
    `get_substation_in_cooldown -> List[int]`,
    `get_substation_elements -> Dict[int, List[SubstationElement]]`,
    `get_substation_to_node_mapping -> Optional[Dict[int, Any]]`,
    `get_internal_to_external_mapping -> Dict[int, int]`,
    `get_dataframe -> pd.DataFrame`,
    `get_reference_topovec_sub(sub: int) -> List[int]`,
    `get_overload_disconnection_topovec_subor(l: int) -> Tuple[int, List[int]]`.
  - Concrete helpers: `create_df(self, d: Dict[str, Any], line_to_cut: List[int]) -> pd.DataFrame`,
    `branch_direction_swaps(df: pd.DataFrame) -> None`,
    `invert_dict_keys_values(d: Dict[Any, Any]) -> Dict[Any, Any]`,
    `create_end_result_empty_dataframe() -> pd.DataFrame`,
    `get_model_obj_from_or/ext(df_indexed: pd.DataFrame, substation_id: int, dest: int, busbar: int) -> Optional[Union[OriginLine, ExtremityLine]]`.
  - A module-level ``SubstationElement = Union[Production, Consumption,
    OriginLine, ExtremityLine]`` alias is exposed so downstream annotations
    can use a single name instead of repeating the 4-way union.

Deliberately out of scope for this pass (tracked under longer-term item 14):

- `core/alphadeesp.py`, `core/graphsAndPaths.py`, `core/network.py`,
  `core/printer.py`, and the two concrete backends
  (`core/grid2op/Grid2opSimulation.py`, `core/pypownet/PypownetSimulation.py`).
  These modules consume the seed types but annotating them requires
  carefully typing `networkx.DiGraph` / `rustworkx.PyDiGraph` nodes and
  edges, which is a larger job.
- Wiring `mypy --ignore-missing-imports` into CI. With only the two seed
  modules annotated mypy would be almost entirely noise; it makes sense to
  enable it once `alphadeesp.py` and the backends are typed.

Verification:

- `python -m pyflakes alphaDeesp/core alphaDeesp/*.py` (CI lint scope) →
  **0 findings**.
- `python -m pytest alphaDeesp/tests/test_graphs_and_paths_unit.py
  alphaDeesp/tests/test_alphadeesp_unit.py` → **130 passed**.
- `from alphaDeesp.core import simulation, elements` imports clean on
  Python 3.12 with numpy / pandas installed.

### `Simulation.create_df` stdout spam

The `print(df)` at the tail of `alphaDeesp/core/simulation.py::create_df` used
to fire on every invocation because the surrounding `if self.debug:` had been
commented out. It now fires only when `self.debug` is truthy (using
`getattr(self, "debug", False)` since `Simulation.__init__` on the base class
never sets `self.debug`; only the concrete subclasses do).

### Pyflakes in CI

`.circleci/config.yml` now installs `pyflakes` and runs it as a dedicated
step **before** the test suite, so lint regressions fail fast without waiting
on the (slow) Grid2op simulation tests. The scope deliberately excludes:

- `alphaDeesp/core/pypownet/` — the legacy backend uses
  `from pypownet.agent import *`, which pyflakes cannot statically resolve
  without pypownet installed. This matches the existing CI policy, which
  already excludes `alphaDeesp/tests/pypownet/` from pytest.
- `alphaDeesp/Expert_rule_action_verification.py` — imports
  `make_evaluation_env`, `pypowsybl`, `load_training_data`, and other
  first-run-time dependencies that are not part of `requirements.txt`.
  Re-including this file is blocked on first re-enabling its test module in
  CI.

`ruff` was *not* wired up in this pass — the action item mentioned
`pyflakes/ruff` as alternatives, and pyflakes is already the lower-friction
choice for an existing untyped codebase. Switching to ruff becomes more
attractive once the short-term type-hint work starts.

### Logging migration

A module-level `logger = logging.getLogger(__name__)` is now present in
every first-party module that previously printed to stdout (excluding the
legacy Pypownet backend and the `ressources/parameters/*.py` ad-hoc grid
builder scripts). `print()` calls were converted to the appropriate
`logger.info` / `logger.warning` / `logger.debug` based on the message
intent:

- `alphaDeesp/main.py` — 11 prints → `logger.info` / `logger.error`; the
  CLI entry point now installs a default `logging.basicConfig(level=INFO)`
  iff the root logger has no handlers (so embedders keep control).
- `alphaDeesp/expert_operator.py` — 5 prints → `logger.info` /
  `logger.debug`.
- `alphaDeesp/core/simulation.py` — the two debug `print(df)` calls that
  used to fire unconditionally now route through `logger.debug`.
- `alphaDeesp/core/alphadeesp.py` — 39 prints → `logger.debug` (every
  branch was already gated on `self.debug`, so the intent is trace-level)
  with one `logger.info` for the "substation X is in cooldown" message.
- `alphaDeesp/core/graphsAndPaths.py` — 4 prints → `logger.debug`.
- `alphaDeesp/core/network.py` — 18 prints (including a pile of
  `pprint.pprint` statements that used to fire on every `Network(...)`
  construction) → `logger.debug`, guarded with
  `logger.isEnabledFor(logging.DEBUG)` before calling `pprint.pformat` to
  keep the formatting cost out of the hot path when debug is off.
- `alphaDeesp/core/printer.py` — 5 prints → `logger.debug`.
- `alphaDeesp/core/elements.py` — 1 print → `logger.debug`.
- `alphaDeesp/core/grid2op/Grid2opSimulation.py` — 12 prints → a mix of
  `logger.info` (for the `__init__` announcements and
  `compute_new_network_changes` banners) and `logger.warning` (for the
  three layout fallback messages). The module-level `logger` that was
  introduced in the immediate pass for the `compute_layout` handlers is
  now used throughout the file.
- `alphaDeesp/core/grid2op/Grid2opObservationLoader.py` — 5 prints → 4
  `logger.info` and 1 `logger.warning` (LightSimBackend fallback).

Total: **101** `print()` calls across the ten CI-scoped modules replaced
with module-namespaced loggers. Embedders (for example
`l2rpn-baselines.ExpertAgent`) can now silence the expert system with a
single `logging.getLogger("alphaDeesp").setLevel(logging.WARNING)` — or
bring the trace back by switching to `DEBUG`.

The ad-hoc scripts under `alphaDeesp/ressources/parameters/` (≈6 prints
across `build_new_parameters_environment.py` and `make_reference_grid.py`)
and the four tests listed in the metrics table were intentionally left as
`print()` — they are one-off CLI utilities / test harness output, not
library code. The legacy Pypownet backend was also skipped for the same
reason it is skipped from the pyflakes CI scope and the test matrix.

### Abstract-method docstrings

All 9 `"""TODO"""` stubs on `Simulation` (`isAntenna`, `isDoubleLine`,
`getLinesAtSubAndBusbar`, `get_substation_in_cooldown`,
`get_substation_elements`, `get_substation_to_node_mapping`,
`get_internal_to_external_mapping`, `get_dataframe`,
`get_reference_topovec_sub`, `get_overload_disconnection_topovec_subor`)
now carry full Sphinx-style docstrings describing the contract expected
from every backend: what the method receives, what it must return, and
how AlphaDeesp consumes the result. The `cut_lines_and_recomputes_flows`
and `get_layout` methods — which already had one-line docstrings — were
expanded with the same level of detail. The signatures of
`get_reference_topovec_sub` and `get_overload_disconnection_topovec_subor`
on the abstract base also picked up the `sub` / `l` parameters they were
already taking in both concrete backends (`Grid2opSimulation` and
`PypownetSimulation`); the abstract base had silently diverged.

### CI re-enablement

The three previously excluded test modules are now handled via
`pytest.importorskip` at module top rather than a `--ignore` flag:

- `alphaDeesp/tests/test_expert_rules.py` — self-skips when
  `make_training_env`, `load_evaluation_data`, or
  `Expert_rule_action_verification` cannot be imported (the Dijon eval
  toolbox lives in a sibling project and is not installed in CI).
- `alphaDeesp/tests/pypownet/test_integrations.py` and
  `alphaDeesp/tests/pypownet/grid_test.py` — self-skip when `pypownet`
  cannot be imported.
- `alphaDeesp/tests/test_cli.py` — stays in its own CI step (it was never
  actually excluded from CI, just split out because it shells out to the
  installed `expertop4grid` binary).

`.circleci/config.yml` now runs `pytest --ignore=alphaDeesp/tests/test_cli.py`
(down from three `--ignore` flags). The net effect is that if someone
actually installs pypownet or the Dijon toolbox on the CI image, the
corresponding tests will run automatically instead of needing a
CI-config change.

### Packaging cleanup

- `LICENSE.md` was deleted — it was byte-identical to `LICENSE`. The
  `MANIFEST.in` reference was updated accordingly (`include LICENSE`).
- `setup.py`:
  - `python_requires=">=3.9"` added (previously nothing; classifiers
    advertised 3.6 / 3.7, both past EOL and no longer tested).
  - Classifiers refreshed to `3.9`, `3.10`, `3.11`, `3.12` plus the
    umbrella `Programming Language :: Python :: 3` and
    `Operating System :: OS Independent`.
  - `Grid2Op>=1.6.4` → `>=1.12.1` and `lightsim2grid>=0.5.5` → `>=0.10.3`
    to match `requirements.txt`.
  - Removed the stale `pathlib>=1.0.1` dependency — `pathlib` has been in
    the stdlib since Python 3.4, and the PyPI package of the same name
    is an ancient backport that collides with the stdlib on modern
    Python.
  - Dropped the `#"pygame==1.9.6"` / `#numba==0.49.1` inline-commented
    dependencies that had been dead for years.

### Verification

- `python -m pyflakes <CI scope>` → 0 findings (both after the immediate
  pass and after the short-term pass).
- `python -m pytest alphaDeesp/tests/test_graphs_and_paths_unit.py` →
  **98 passed** after the short-term edits.
- `python -m pytest alphaDeesp/tests/test_expert_rules.py --collect-only`
  → collected 0 / skipped 1 (module-level `importorskip`).
- `python -m pytest alphaDeesp/tests/pypownet/ --collect-only` → collected
  0 / skipped 2 (module-level `importorskip`).
- `AlphaDeesp` and all touched modules import cleanly under Python 3.11 /
  3.12 with numpy / pandas / networkx / rustworkx installed.
- Full Grid2op integration tests require a full Grid2Op install and were
  not re-run in this pass; CI on push will exercise them.

## Metrics at a glance

| Metric | Value | Tool |
|---|---|---|
| Python LOC (core + CLI) | ~5,600 SLOC / 9,273 total (incl. tests) | `wc`, `radon raw` |
| Largest module | `core/graphsAndPaths.py` — **2,185 lines** | `wc -l` |
| Largest class/file combo | `core/alphadeesp.py` — 908 lines, one class | `wc -l` |
| Maintainability Index | `graphsAndPaths.py` **C (0.00)**, `Grid2opSimulation.py` 25.95, `PypownetSimulation.py` 25.91, `Expert_rule_action_verification.py` 27.29 | `radon mi` |
| Worst cyclomatic complexity (post-refactor) | Dispatchers: `rank_current_topo_at_node_x` **B (7)**, `apply_new_topo_to_graph` **B (7)**, `detect_edges_to_keep` **A (2)**, `add_relevant_null_flow_lines` **B (7)**. Worst helper: `_prepare_detect_edges_inputs` **C (20)**. | `radon cc` |
| Average complexity | **B (5.72)** across 163 functions/classes | `radon cc` |
| `print()` calls | baseline: **247** across 20 files → **~108 remaining** (tests, `build_new_parameters_environment.py`, legacy Pypownet backend, `Expert_rule_action_verification.py`); all 10 CI-scoped first-party modules are at **0** | grep |
| Bare `except:` clauses | baseline: 9 → **0** | grep |
| Type-annotated functions | baseline 1 → **36** after the core base-class seed (16 in `core/elements.py` + 20 in `core/simulation.py`); rest of the package still untyped | grep `-> ` |
| Pyflakes findings | baseline: 59 → **0** in CI scope | `pyflakes` |
| TODO/FIXME markers | 25 across 7 files | grep |
| Test functions | 164 across 9 files; CI now only excludes `test_cli.py` (run as a dedicated step); pypownet + expert_rules self-skip via `pytest.importorskip` | `.circleci/config.yml` |

## Critical issues (fix first)

> Items 1–10 below are now resolved and are kept for historical context —
> see the "Cleanup progress" section at the top. Items 11+ remain as
> targets for the remaining longer-term refactors (14: type hints beyond
> the core base classes; 15: Pypownet backend fate).

### 1. Bare `except:` clauses swallowing all errors
`alphaDeesp/main.py` (×3 near lines 100, 105, 117),
`core/grid2op/Grid2opSimulation.py` (×2),
`core/grid2op/Grid2opObservationLoader.py` (×2),
`core/pypownet/PypownetSimulation.py` (×1).
Example: `Grid2opSimulation.py:34-43` hides layout-parsing bugs behind nested
bare excepts and falls back to a hardcoded 28-coordinate layout. These make
production failures impossible to diagnose. Replace with narrowly-typed
`except (KeyError, ValueError, ast.LiteralEvalError)` and log the fallback.

### 2. `from alphaDeesp.core.elements import *` in `alphadeesp.py` and `network.py`
Pyflakes flags **11** identifiers (`Production`, `Consumption`, `OriginLine`,
`ExtremityLine`) as "may be undefined, or defined from star imports." This
defeats static analysis and makes refactoring `elements.py` risky. Replace
with explicit imports.

### 3. ~~A single 68-CC function holds the ranking logic~~ (done)
`core/alphadeesp.py rank_current_topo_at_node_x` has been decomposed into
four scoring-branch helpers plus two data helpers; the dispatcher is now
CC 7 (B). The three runner-up monsters
(`apply_new_topo_to_graph`, `detect_edges_to_keep`,
`add_relevant_null_flow_lines`) are also done in the same pass. See the
"Longer-term refactor landing" table at the top of this document.

Still outstanding:
- `Grid2opSimulation.score_changes_between_two_observations` — CC 25 / D
- `Expert_rule_action_verification.check_rules` — CC 21 / D

### 4. CI skips three whole test modules
`.circleci/config.yml` runs:

```bash
pytest --ignore=alphaDeesp/tests/pypownet/ \
       --ignore=alphaDeesp/tests/test_cli.py \
       --ignore=alphaDeesp/tests/test_expert_rules.py
```

This leaves the Pypownet backend, the CLI, and the expert-rules verification
engine (`Expert_rule_action_verification.py`, 945 LOC, 21 tests) **untested in
CI**. The CLI is re-added as a separate job, but `test_expert_rules.py`
(547 lines) is silently excluded.

## High-impact issues

### 5. ~~No logging — 247 `print()` calls~~ (done)
Replaced with module-level `logging.getLogger(__name__)` across the 10
first-party modules in the CI scope. See the "Logging migration" section
near the top of this document. Embedders can now silence the expert system
with a single `logging.getLogger("alphaDeesp").setLevel(logging.WARNING)`.

### 6. ~~Zero type hints~~ (done for the core base classes)
`core/elements.py` and `core/simulation.py` — the two starting points the
original analysis recommended — are now fully annotated. All four element
classes (`Production`, `Consumption`, `OriginLine`, `ExtremityLine`) carry
annotations on constructors, the `busbar` property/setter, and the `ID`
class variable. `Simulation` has annotated signatures for all 10 abstract
methods, `create_df`, `branch_direction_swaps`, `invert_dict_keys_values`,
and both `get_model_obj_from_*` helpers. A `SubstationElement` `Union`
alias is exported from `simulation.py` so downstream code (the two
backends) can use the same type in their own annotations as they are
converted. See "Type hints seed" below. Further propagation to
`alphadeesp.py`, `network.py`, `graphsAndPaths.py` and the backend
simulators is tracked under longer-term item 14.

### 7. ~~Pyflakes: 15+ dead locals and unused imports in `alphadeesp.py` alone~~ (done)
See the "Pyflakes cleanup" section near the top of this document.
Reconfirmed: `python -m pyflakes` on the CI scope reports **0 findings**
after the item 6/9 edits.

### 8. Abstract-method docstrings are just `"""TODO"""`
`core/simulation.py:29-70` — 9 of 10 abstract methods have `"""TODO"""` as the
entire contract documentation. New backend authors have nothing to implement
against. The class is the public extension point; this is a doc-gap bug.

### 9. ~~Mutable default arguments~~ (done)
The two remaining `ltc=[9]` / `other_ltc=[]` defaults on
`Grid2opSimulation.__init__` and the `ltc=[9]` default on
`PypownetSimulation.__init__` are now `None` sentinels normalized inside
the body. Together with the prior `AlphaDeesp.__init__` /
`rank_current_topo_at_node_x` fixes this closes out the original audit
finding. See "Mutable defaults" above (updated) for the full list of
signatures.

## Medium-impact issues

### 10. ~~The magic `"666"` twin-node marker~~ (done)
Replaced by the ``TWIN_NODE_OFFSET`` scheme in
`alphaDeesp/core/twin_nodes.py` and its three helpers
(`twin_node_id`, `is_twin_node_id`, `original_substation_id`). All five
prior call sites (`alphadeesp.py`, `network.py` ×2, `printer.py` ×2) now
go through the helpers; round-trip and discrimination are unit-tested in
`TestTwinNodeIds`. See the "Twin-node id scheme" section at the top of this
document.

### 11. Naming inconsistency
Mix of `snake_case` methods (`get_dataframe`, `get_substation_elements`) and
`camelCase` methods (`isAntenna`, `isDoubleLine`, `getLinesAtSubAndBusbar`) on
the **same abstract class**. Attribute names mix too (`self.rankedLoopBuses`
alongside `self.substation_in_cooldown`). Because `isAntenna` etc. are part of
the public `Simulation` contract, renaming requires a deprecation shim.

### 12. Commented-out code and French/English mixed comments
`core/simulation.py:132-152` contains 20 lines of French conditional-logic
comments inside `create_df`, mixed with `# DO NOT USE SET_VALUE ANYMORE, USE
DF.AT INSTEAD` (historical warning). `graphsAndPaths.py` has multi-line
commented blocks at 419-492. Module-wide single-line-comment density is 17%
and multi-line-comment density jumps to 37% in `graphsAndPaths.py`, largely
because blocks were commented out instead of deleted.

### 13. Hardcoded 28-point layout duplicated in 3 places
`ressources/config/config.ini` line 20, `Grid2opSimulation.py:40`,
`PypownetSimulation.py:69`. Changing the reference grid means editing three
files.

### 14. `create_df` prints to stdout unconditionally
`core/simulation.py:180` — `print(df)` runs on every invocation even when
`debug=False`, because the surrounding `if self.debug:` is commented out.
Spam in production.

### 15. Setup vs. runtime version drift
`setup.py` declares `Grid2Op>=1.6.4`, `numpy>=1.18.4`, Python 3.6/3.7
classifiers. `requirements.txt` pins `Grid2Op==1.12.1`, `numpy==2.3.3`,
`scipy==1.16.2`, `lightsim2grid==0.10.3`. CI runs `cimg/python:3.12`. The
`setup.py` classifiers are six years out of date and will install on Python
versions the code no longer supports.

### 16. Pypownet backend is legacy baggage
`PypownetSimulation.py` (702 LOC, MI 25.91) has 50 `print()` calls, 7 TODOs,
and is excluded from CI. It should either be marked deprecated in
`setup.py extras` messaging or removed.

## Documentation and packaging

- **`docs/`** is Sphinx RST-based but `docs/conf.py` is minimal and there's no
  `readthedocs.yaml` in the repo; the README advertises
  `expertop4grid.readthedocs.io` — worth verifying it still builds.
- **`LICENSE` and `LICENSE.md`** are duplicated files. Pick one.
- **`octave-workspace`**, `20200630_*.docx`, and `_LARGE__..._.pdf` (1.5 MB)
  are committed artifacts that should live outside the repo (or in Git LFS).
- **`Test_Expert_baseline.ipynb` (371 KB)** and the three tutorial notebooks
  in `getting_started/` total ~7.6 MB, dominated by embedded images. Use
  `nbstripout` or move rendered outputs to docs.
- **`MANIFEST.in`** exists but `include_package_data=True` already ships the
  5 MB `ressources/` directory into the wheel. Consider splitting grids into
  a data package.

## Recommended action plan

### Immediate (low-effort, high-value) — ✅ done

1. ~~Replace all 9 bare `except:` with specific exceptions + `logging.exception(...)`.~~
2. ~~Delete the 15 unused locals/imports flagged by pyflakes (fully mechanical).~~
3. ~~Fix the mutable defaults in `alphadeesp.py`.~~
4. ~~Drop `print(df)` from `Simulation.create_df`.~~
5. ~~Enable `pyflakes`/`ruff` in CI to stop regressions.~~

See the "Cleanup progress" section at the top of this document for the
details of each change.

### Short-term — ✅ done

6. ~~Replace `from elements import *` with explicit imports.~~ (landed in
   the immediate pass alongside the pyflakes cleanup.)
7. ~~Introduce a module-level `logger = logging.getLogger(__name__)` and
   convert `print()` → `logger.info/debug`.~~
8. ~~Write real docstrings for every abstract method in
   `core/simulation.py`.~~
9. ~~Re-enable the three excluded test modules in CI (or document *why*
   they are skipped).~~
10. ~~Unify `LICENSE`/`LICENSE.md`; refresh `setup.py` classifiers and
    version pins to match `requirements.txt`.~~

See the "Cleanup progress" section at the top of this document for the
details of each change.

### Longer-term refactors
11. ~~Decompose `rank_current_topo_at_node_x` (CC 68) into ≤5 helpers, each
    with a unit test.~~ Done — dispatcher is now CC 7 (B); see the
    "Longer-term refactor landing" table above.
12. ~~Same treatment for `detect_edges_to_keep`, `add_relevant_null_flow_lines`,
    `apply_new_topo_to_graph`.~~ Done — F (46)/F (44)/E (36) → A (2)/B (7)/B (7).
13. ~~Replace the `"666"` twin-node encoding with a proper id scheme.~~ Done,
    see `alphaDeesp/core/twin_nodes.py`.
14. Add type hints starting from `core/simulation.py` and `core/elements.py`
    outward; enable `mypy --ignore-missing-imports` in CI. _Seed done for
    the two base-class modules — see "Type hints seed" above. Remaining:
    `core/alphadeesp.py`, `core/graphsAndPaths.py`, `core/network.py`,
    `core/printer.py`, and the two concrete simulator backends._
15. Decide the fate of the Pypownet backend: first-class (re-enable CI,
    modernize) or deprecated.

## How to reproduce these metrics

```bash
pip install pyflakes radon

# Complexity
python -m radon cc -a -s alphaDeesp/core/alphadeesp.py \
    alphaDeesp/core/graphsAndPaths.py \
    alphaDeesp/core/simulation.py \
    alphaDeesp/core/grid2op/Grid2opSimulation.py \
    alphaDeesp/Expert_rule_action_verification.py

# Maintainability Index
python -m radon mi -s alphaDeesp/core/ alphaDeesp/Expert_rule_action_verification.py

# Raw metrics (LOC/SLOC/comment ratio)
python -m radon raw -s alphaDeesp/core/ alphaDeesp/Expert_rule_action_verification.py

# Unused code and star-import leaks
python -m pyflakes alphaDeesp/core alphaDeesp/*.py

# Print statements
grep -rn --include='*.py' '^\s*print(' alphaDeesp/ | wc -l

# Bare excepts
grep -rn --include='*.py' '^\s*except\s*:' alphaDeesp/
```
