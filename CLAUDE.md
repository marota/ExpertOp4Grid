# CLAUDE.md

This file gives Claude Code (claude.ai/code) working context for this repository.

## Project Overview

**ExpertOp4Grid** (package: `alphaDeesp`) is a Python expert system that solves
power-line overloads on an electrical grid using non-linear topological actions
(bus-bar splits, line switching). Given an overloaded line, it builds an
"influence graph" around the overload, ranks candidate substations/topologies,
simulates the top-ranked ones, and returns a score (0–4) indicating how well
each remediation fixes the overload.

It is the implementation of the paper *"Expert system for topological action
discovery in smart grids"*
(https://hal.archives-ouvertes.fr/hal-01897931/).

## Repository Layout

```
alphaDeesp/
├── main.py                         # CLI entry point (`expertop4grid`)
├── expert_operator.py              # Orchestrator: simulator -> AlphaDeesp -> results
├── Expert_rule_action_verification.py  # Rule-checking utilities for proposed actions
├── core/
│   ├── alphadeesp.py               # AlphaDeesp algorithm (ranking, topology exploration)
│   ├── graphsAndPaths.py           # OverFlowGraph, PowerFlowGraph, Structured_Overload_Distribution_Graph
│   ├── simulation.py               # Abstract Simulation base class + DataFrame plumbing
│   ├── network.py                  # Network / Substation model objects
│   ├── elements.py                 # Production, Consumption, OriginLine, ExtremityLine dataclasses
│   ├── printer.py                  # Graphviz/shell printing helpers
│   ├── grid2op/                    # Grid2op backend (Grid2opSimulation, Grid2opObservationLoader)
│   └── pypownet/                   # Pypownet backend (legacy / optional)
├── ressources/
│   ├── config/config.ini           # Default runtime config (thresholds, layout, simulator type)
│   └── parameters/                 # Built-in grids (l2rpn_2019, rte_case14_realistic, custom14, ...)
└── tests/                          # pytest suite (unit + integration, grid2op + pypownet)
docs/                               # Sphinx sources (RST)
getting_started/                    # Jupyter tutorial notebooks
.circleci/config.yml                # CI: pytest on cimg/python:3.12 with graphviz
```

## Architecture

```
                       ┌────────────────────┐
   config.ini + CLI -> │ alphaDeesp.main    │
                       └────────┬───────────┘
                                │ builds
                                ▼
              ┌────────────────────────────────────┐
              │ Grid2opSimulation / PypownetSimu.  │  (subclass of core.simulation.Simulation)
              └────────┬───────────────────────────┘
                       │ provides topology, dataframe, mappings
                       ▼
              ┌────────────────────────────────────┐
              │ expert_operator.expert_operator()  │
              └────────┬───────────────────────────┘
                       │
                       ├─► OverFlowGraph (graphsAndPaths.py)
                       ├─► AlphaDeesp.get_ranked_combinations()
                       └─► sim.compute_new_network_changes() -> end-result DataFrame
```

Key contract: any new simulator backend must implement the abstract methods of
`alphaDeesp/core/simulation.py::Simulation` (get_dataframe, isAntenna,
get_substation_elements, compute_new_network_changes, etc.).

## Common Commands

```bash
# Install (editable is fine; entry point `expertop4grid` is registered)
pip install -e .

# Run in manual mode on a built-in grid (cut line 9, timestep 0, scenario 0)
python -m alphaDeesp.main -l 9 -s 0 -c 0 -t 0
# or via the installed console script
expertop4grid -l 9 -s 0 -c 0 -t 0

# Flags:
#   -l/--ltc             lines to cut (single int for now)
#   -s/--snapshot        0|1 — render graphs to output/
#   -c/--chronicscenario chronic scenario id/name (default 0)
#   -t/--timestep        starting timestep (default 0)
#   -f/--fileconfig      alternate config.ini path
#   -d/--debug           0|1

# Run tests (CircleCI command, skipping pypownet + CLI + expert-rules suites)
pytest --ignore=alphaDeesp/tests/pypownet/ \
       --ignore=alphaDeesp/tests/test_cli.py \
       --ignore=alphaDeesp/tests/test_expert_rules.py

# CLI smoke test (runs separately in CI)
pytest alphaDeesp/tests/test_cli.py

# Full test suite with warning suppression (from README)
pytest --verbose --continue-on-collection-errors -p no:warnings
```

## Configuration (config.ini)

Defaults live in `alphaDeesp/ressources/config/config.ini`. Important keys:

- `simulatorType` — `Grid2OP` | `Pypownet` | `RTE`
- `gridPath` — folder containing the grid (defaults to packaged `l2rpn_2019`)
- `outputPath` — where snapshot plots are written
- `ThresholdReportOfLine`, `ThersholdMinPowerOfLoop`, `ratioToKeepLoop`,
  `ratioToReconsiderFlowDirection`, `maxUnusedLines`,
  `totalNumberOfSimulatedTopos`, `numberOfSimulatedToposPerNode` — tunables for
  the AlphaDeesp ranking and simulation step.

All of these can alternatively be supplied as a Python dict to the
`expert_operator()` API when embedding the system in another agent.

## Dependencies

Runtime: `Grid2Op`, `lightsim2grid`, `networkx`, `rustworkx`, `pandapower`,
`pandas`, `numpy`, `scipy`, `matplotlib`, `graphviz`, `pydot`, `Sphinx`,
`pytest`.

Optional: `pypownet>=2.2.0`, `oct2py`, `pypower` (for the legacy backend).

Python: targeted at 3.12 in CI; `setup.py` still advertises 3.6/3.7 (stale).

Graphviz **executables** must be on PATH for snapshot/plot mode (not just the
Python binding). On Debian/Ubuntu: `apt-get install graphviz`.

## Conventions and Gotchas

- The package directory is spelled **`alphaDeesp`** (lowercase-a) even though
  the PyPI name is `ExpertOp4Grid`. Imports use `from alphaDeesp.core...`.
- The `ressources/` directory keeps the French spelling — do not rename it; it
  is referenced by config paths and tests.
- `simulatorType = Grid2OP` (exact casing) in config.ini; `Pypownet` and `RTE`
  are alternative literals checked in `main.py`.
- Naming is mixed: public API uses both `snake_case` (`get_dataframe`) and
  `camelCase` (`isAntenna`, `isDoubleLine`, `getLinesAtSubAndBusbar`). Preserve
  existing names when editing to avoid breaking the abstract contract.
- `core/alphadeesp.py` and `core/network.py` rely on `from elements import *`;
  adding new element classes requires keeping that star import working.
- Several modules print directly to stdout; there is no logging framework
  wired up.
- Only one type-annotated function exists in the codebase — do not expect mypy
  to be useful without first adding annotations.

## Branch Policy for Claude Code Sessions

Development branch for code-quality work: **`claude/code-quality-analysis-8Ftgi`**.
Push all changes to that branch unless the user explicitly says otherwise.
Do not open PRs unless requested.
