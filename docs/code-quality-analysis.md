# Code Quality & Maintainability Analysis

_Last updated: 2026-04-12 — branch `claude/code-quality-analysis-8Ftgi`_

This document captures a diagnostic review of the `alphaDeesp` (ExpertOp4Grid)
codebase. It is intended as a living punch-list for incremental cleanup work.
Numbers were produced with `pyflakes`, `radon cc/mi/raw`, and targeted `grep`
audits over the entire `alphaDeesp/` package.

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

### Immediate (low-effort, high-value)
1. Replace all 9 bare `except:` with specific exceptions + `logging.exception(...)`.
2. Delete the 15 unused locals/imports flagged by pyflakes (fully mechanical).
3. Fix the mutable defaults in `alphadeesp.py`.
4. Drop `print(df)` from `Simulation.create_df`.
5. Enable `pyflakes`/`ruff` in CI to stop regressions.

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
