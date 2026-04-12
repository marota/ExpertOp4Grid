# Code Quality & Maintainability Analysis

_Last updated: 2026-04-12 — branch `claude/code-quality-analysis-8Ftgi`_

This document captures a diagnostic review of the `alphaDeesp` (ExpertOp4Grid)
codebase. It is intended as a living punch-list for incremental cleanup work.
Numbers were produced with `pyflakes`, `radon cc/mi/raw`, and targeted `grep`
audits over the entire `alphaDeesp/` package.

## Cleanup progress

The "Immediate" items from the action plan have been addressed. The pyflakes
CI scope (`alphaDeesp/core/**`, `expert_operator.py`, `main.py`, excluding the
legacy Pypownet backend) is now clean, and 98/98 tests in
`test_graphs_and_paths_unit.py` pass locally after the edits.

| Status | Item | Notes |
|---|---|---|
| done | Replace all 9 bare `except:` with specific exceptions | See "Bare except cleanup" below |
| done | Delete 15+ unused locals/imports flagged by pyflakes | Extended to star-import cleanup where necessary |
| done | Fix mutable defaults in `alphadeesp.py` | `__init__` + `rank_current_topo_at_node_x` |
| done | Drop unconditional `print(df)` from `Simulation.create_df` | Now gated on `self.debug` |
| done | Enable `pyflakes` in CI | New lint step in `.circleci/config.yml` before tests |

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
`PypownetSimulation.__init__` are **not** fixed yet — they are part of the
simulator public API and fixing them cleanly needs a careful scan of the
call sites. Tracked under the short-term action plan below.

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

### Verification

- `python -m pyflakes <CI scope>` → 0 findings.
- `python -m pytest alphaDeesp/tests/test_graphs_and_paths_unit.py` →
  **98 passed**.
- `AlphaDeesp` and all touched modules import cleanly under Python 3.12 with
  numpy / pandas / networkx / rustworkx installed.
- Full Grid2op integration tests require a full Grid2Op install and were not
  re-run in this pass; CI on push will exercise them.

## Metrics at a glance

| Metric | Value | Tool |
|---|---|---|
| Python LOC (core + CLI) | ~5,600 SLOC / 9,273 total (incl. tests) | `wc`, `radon raw` |
| Largest module | `core/graphsAndPaths.py` — **2,185 lines** | `wc -l` |
| Largest class/file combo | `core/alphadeesp.py` — 908 lines, one class | `wc -l` |
| Maintainability Index | `graphsAndPaths.py` **C (0.00)**, `Grid2opSimulation.py` 25.95, `PypownetSimulation.py` 25.91, `Expert_rule_action_verification.py` 27.29 | `radon mi` |
| Worst cyclomatic complexity | `AlphaDeesp.rank_current_topo_at_node_x` **F (68)**, `apply_new_topo_to_graph` **E (36)**, `OverFlowGraph.detect_edges_to_keep` **F (46)**, `add_relevant_null_flow_lines` **F (44)** | `radon cc` |
| Average complexity | **B (5.72)** across 163 functions/classes | `radon cc` |
| `print()` calls | **247** across 20 files (no logging framework) | grep |
| Bare `except:` clauses | **9** across 5 files | grep |
| Type-annotated functions | **1** across the entire codebase | grep `-> ` |
| Pyflakes findings | 59 (unused imports, unreachable locals, `from X import *` hiding symbols) | `pyflakes` |
| TODO/FIXME markers | 25 across 7 files | grep |
| Test functions | 164 across 9 files; CI excludes `pypownet/`, `test_cli.py`, `test_expert_rules.py` | `.circleci/config.yml` |

## Critical issues (fix first)

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

### 3. A single 68-CC function holds the ranking logic
`core/alphadeesp.py:382 rank_current_topo_at_node_x` — cyclomatic complexity
**68** (radon rates this "F — unstable"). It is ~250 lines, mixes busbar
bookkeeping, graph mutation, and French inline comments. Nearly impossible to
unit-test or modify safely. **This is the single most important refactor
target.**

Runners-up:
- `AlphaDeesp.apply_new_topo_to_graph` — CC 36 / E
- `OverFlowGraph.detect_edges_to_keep` — CC 46 / F
- `OverFlowGraph.add_relevant_null_flow_lines` — CC 44 / F
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

### 5. No logging — 247 `print()` calls
Distribution: `PypownetSimulation.py` (42), `alphadeesp.py` (39),
`Expert_rule_action_verification.py` (34), `main.py` (14),
`Grid2opSimulation.py` (12). There is no verbosity control; `-d/--debug` is
checked in only a few places. Embedders (e.g. l2rpn-baselines `ExpertAgent`)
cannot suppress output. Replace with `logging.getLogger(__name__)`.

### 6. Zero type hints
Exactly one `-> ` return annotation in the whole package. Given how much of
the code manipulates `pd.DataFrame`, `networkx.DiGraph`, and custom
`OriginLine`/`ExtremityLine` objects, this is the largest onboarding cost.
Start by annotating `core/simulation.py` (the abstract base) and
`core/elements.py` — they propagate outward.

### 7. Pyflakes: 15+ dead locals and unused imports in `alphadeesp.py` alone
Unused imports: `pprint`, `math.ceil`, `os`. Unused locals include
`ranked_combinations` (line 50), `current_node`/`new_node` (271-272),
`edge_color` (336), `all_nodes_value_attributes` (385, 758),
`not_interesting_bus_id` (428), `node2` (555), `p` (823), `indexEdge_inDf`
(805). This is abandoned refactoring. Similar patterns in `graphsAndPaths.py`
(5 dead locals), `printer.py` (3 unused imports + 2 dead locals),
`Expert_rule_action_verification.py` (redefinition of `version_packaging`).

### 8. Abstract-method docstrings are just `"""TODO"""`
`core/simulation.py:29-70` — 9 of 10 abstract methods have `"""TODO"""` as the
entire contract documentation. New backend authors have nothing to implement
against. The class is the public extension point; this is a doc-gap bug.

### 9. Mutable default arguments
`core/alphadeesp.py:28`:

```python
def __init__(self, _g, df_of_g, simulator_data=None,
             substation_in_cooldown=[], debug=False):
```

and `rank_current_topo_at_node_x(..., topo_vect=[0, 0, 1, 1, 1], ...)`.
Classic Python footgun that silently leaks state across calls.

## Medium-impact issues

### 10. The magic `"666"` twin-node marker
When splitting a substation into two busbars, the code creates a new node id
by string-prefixing `666`: `int("666" + str(node_to_change))`
(`alphadeesp.py` around line 245), and decodes it with
`int(str(substation_id)[3:])` (`network.py`, `printer.py`). This breaks for
any substation id ≥ 1000, is undocumented, and appears in at least 6 places.
Replace with a `TwinNodeId` dataclass or a disjoint numbering scheme.

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

### Short-term
6. Replace `from elements import *` with explicit imports.
7. Introduce a module-level `logger = logging.getLogger(__name__)` and
   convert `print()` → `logger.info/debug`.
8. Write real docstrings for every abstract method in `core/simulation.py`.
9. Re-enable the three excluded test modules in CI (or document *why* they
   are skipped).
10. Unify `LICENSE`/`LICENSE.md`; refresh `setup.py` classifiers and version
    pins to match `requirements.txt`.

### Longer-term refactors
11. Decompose `rank_current_topo_at_node_x` (CC 68) into ≤5 helpers, each
    with a unit test. This is the single biggest maintainability win.
12. Same treatment for `detect_edges_to_keep`, `add_relevant_null_flow_lines`,
    `apply_new_topo_to_graph`.
13. Replace the `"666"` twin-node encoding with a proper id scheme.
14. Add type hints starting from `core/simulation.py` and `core/elements.py`
    outward; enable `mypy --ignore-missing-imports` in CI.
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
