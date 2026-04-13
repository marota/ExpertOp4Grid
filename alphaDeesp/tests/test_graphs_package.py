"""Tests for the ``alphaDeesp.core.graphs`` package refactor.

Complementary to ``test_graphs_and_paths_unit.py``:

* :class:`TestGraphsPackageStructure` checks the *structural* invariants
  of the refactor — shim parity, public API stability, sub-module
  boundaries, and import acyclicity. These tests guarantee that anyone
  who moves code around inside the package keeps the shim and the
  documented public surface honest.
* :class:`TestPowerFlowGraphConstruction`,
  :class:`TestStructuredOverloadDistributionGraph` and
  :class:`TestShortestPathMandatoryAndPromoted` provide behavioural
  coverage for three areas that were under-tested in the legacy unit
  file.
"""
import importlib
import pkgutil
import sys

import networkx as nx
import pytest

import alphaDeesp.core.graphs as graphs_pkg
import alphaDeesp.core.graphsAndPaths as graphs_shim
from alphaDeesp.core.graphs import (
    ConstrainedPath,
    OverFlowGraph,
    PowerFlowGraph,
    Structured_Overload_Distribution_Graph,
    shortest_path_mandatory_and_promoted,
)


# ---------------------------------------------------------------------------
# Structural invariants of the refactor
# ---------------------------------------------------------------------------

# The 16 public names the refactor commits to exporting, in the same order
# as ``alphaDeesp/core/graphs/__init__.py::__all__``.
EXPECTED_PUBLIC_NAMES = frozenset({
    "default_voltage_colors",
    "PowerFlowGraph",
    "OverFlowGraph",
    "ConstrainedPath",
    "Structured_Overload_Distribution_Graph",
    "from_edges_get_nodes",
    "delete_color_edges",
    "nodepath_to_edgepath",
    "incident_edges",
    "all_simple_edge_paths_multi",
    "find_multidigraph_edges_by_name",
    "add_double_edges_null_redispatch",
    "remove_unused_added_double_edge",
    "shortest_path_min_weight_then_hops",
    "shortest_path_mandatory_and_promoted",
    "shortest_path_with_promoted_edges",
})

# Where each public symbol physically lives. Structural tests lock these
# placements in so an accidental move gets caught by CI.
EXPECTED_SYMBOL_SUBMODULE = {
    "default_voltage_colors": "alphaDeesp.core.graphs.constants",
    "PowerFlowGraph": "alphaDeesp.core.graphs.power_flow_graph",
    "OverFlowGraph": "alphaDeesp.core.graphs.overflow_graph",
    "ConstrainedPath": "alphaDeesp.core.graphs.constrained_path",
    "Structured_Overload_Distribution_Graph":
        "alphaDeesp.core.graphs.structured_overload_graph",
    "from_edges_get_nodes": "alphaDeesp.core.graphs.graph_utils",
    "delete_color_edges": "alphaDeesp.core.graphs.graph_utils",
    "nodepath_to_edgepath": "alphaDeesp.core.graphs.graph_utils",
    "incident_edges": "alphaDeesp.core.graphs.graph_utils",
    "all_simple_edge_paths_multi": "alphaDeesp.core.graphs.graph_utils",
    "find_multidigraph_edges_by_name": "alphaDeesp.core.graphs.graph_utils",
    "add_double_edges_null_redispatch": "alphaDeesp.core.graphs.null_flow",
    "remove_unused_added_double_edge": "alphaDeesp.core.graphs.null_flow",
    "shortest_path_min_weight_then_hops": "alphaDeesp.core.graphs.shortest_paths",
    "shortest_path_mandatory_and_promoted": "alphaDeesp.core.graphs.shortest_paths",
    "shortest_path_with_promoted_edges": "alphaDeesp.core.graphs.shortest_paths",
}

EXPECTED_SUBMODULES = frozenset({
    "constants",
    "graph_utils",
    "null_flow",
    "shortest_paths",
    "power_flow_graph",
    "constrained_path",
    "structured_overload_graph",
    "overflow_graph",
})


class TestGraphsPackageStructure:
    """Lock in the refactor's structural contract."""

    def test_package_all_matches_expected_public_names(self):
        assert set(graphs_pkg.__all__) == EXPECTED_PUBLIC_NAMES

    def test_shim_all_matches_expected_public_names(self):
        assert set(graphs_shim.__all__) == EXPECTED_PUBLIC_NAMES

    def test_every_public_name_is_attached_to_package(self):
        for name in EXPECTED_PUBLIC_NAMES:
            assert hasattr(graphs_pkg, name), f"missing on package: {name}"

    def test_every_public_name_is_attached_to_shim(self):
        for name in EXPECTED_PUBLIC_NAMES:
            assert hasattr(graphs_shim, name), f"missing on shim: {name}"

    def test_shim_and_package_export_identical_objects(self):
        """``graphsAndPaths.X is graphs.X`` for every public symbol."""
        for name in EXPECTED_PUBLIC_NAMES:
            assert getattr(graphs_shim, name) is getattr(graphs_pkg, name), (
                f"shim/package identity mismatch for {name}"
            )

    def test_public_symbols_originate_from_expected_submodule(self):
        """Each public symbol lives where the refactor says it lives.

        Only classes and functions carry ``__module__`` — for module-level
        data (``default_voltage_colors``) we instead check membership in
        the expected sub-module's own namespace.
        """
        for name, expected_mod in EXPECTED_SYMBOL_SUBMODULE.items():
            obj = getattr(graphs_pkg, name)
            if hasattr(obj, "__module__"):
                assert obj.__module__ == expected_mod, (
                    f"{name} should live in {expected_mod}, "
                    f"found in {obj.__module__}"
                )
            else:
                sub = importlib.import_module(expected_mod)
                assert getattr(sub, name) is obj, (
                    f"{name} is not exported from {expected_mod}"
                )

    def test_all_expected_submodules_exist(self):
        actual = {
            m.name for m in pkgutil.iter_modules(graphs_pkg.__path__)
            if not m.ispkg
        }
        assert EXPECTED_SUBMODULES.issubset(actual), (
            f"missing submodules: {EXPECTED_SUBMODULES - actual}"
        )

    @pytest.mark.parametrize("submodule", sorted(EXPECTED_SUBMODULES))
    def test_submodule_is_importable_in_isolation(self, submodule):
        """Every sub-module imports cleanly on its own.

        Evicting everything under ``alphaDeesp.core.graphs*`` from
        ``sys.modules`` first ensures we catch circular imports that
        would otherwise be hidden by a warm cache.
        """
        to_drop = [
            name for name in sys.modules
            if name == "alphaDeesp.core.graphs"
            or name.startswith("alphaDeesp.core.graphs.")
            or name == "alphaDeesp.core.graphsAndPaths"
        ]
        for name in to_drop:
            sys.modules.pop(name, None)
        mod = importlib.import_module(f"alphaDeesp.core.graphs.{submodule}")
        assert mod.__name__ == f"alphaDeesp.core.graphs.{submodule}"

    def test_overflow_graph_is_subclass_of_power_flow_graph(self):
        assert issubclass(OverFlowGraph, PowerFlowGraph)

    def test_shim_re_imports_do_not_duplicate_classes(self):
        """Re-importing the shim must not create shadow copies of classes."""
        # Use import_module rather than reload: earlier tests may have
        # evicted the shim from sys.modules on purpose.
        shim = importlib.import_module("alphaDeesp.core.graphsAndPaths")
        pkg = importlib.import_module("alphaDeesp.core.graphs")
        assert shim.OverFlowGraph is pkg.OverFlowGraph


# ---------------------------------------------------------------------------
# Behavioural coverage — PowerFlowGraph construction
# ---------------------------------------------------------------------------

def _simple_powerflow_topo():
    """Minimal topology: node 0 prod, node 1 load, node 2 neutral.

        0 --(+5)--> 1     (positive flow, edge kept in order)
        1 --(-3)--> 2     (negative flow, edge gets reversed to 2 -> 1)
    """
    return {
        "nodes": {
            "are_prods":    [True,  False, False],
            "are_loads":    [False, True,  False],
            "prods_values": [10.0],
            "loads_values": [8.0],
        },
        "edges": {
            "idx_or":     [0, 1],
            "idx_ex":     [1, 2],
            "init_flows": [5.0, -3.0],
        },
    }


class TestPowerFlowGraphConstruction:
    """Construction-level tests for :class:`PowerFlowGraph`."""

    def test_builds_expected_node_count(self):
        pfg = PowerFlowGraph(_simple_powerflow_topo(), lines_cut=[])
        assert pfg.get_graph().number_of_nodes() == 3

    def test_nodes_are_coloured_by_prod_minus_load(self):
        pfg = PowerFlowGraph(_simple_powerflow_topo(), lines_cut=[])
        g = pfg.get_graph()
        # node 0: pure producer ⇒ coral
        assert g.nodes[0]["fillcolor"] == "coral"
        # node 1: pure load ⇒ lightblue
        assert g.nodes[1]["fillcolor"] == "lightblue"
        # node 2: neutral ⇒ the off-white colour
        assert g.nodes[2]["fillcolor"] == "#ffffed"

    def test_positive_flow_edge_keeps_direction(self):
        pfg = PowerFlowGraph(_simple_powerflow_topo(), lines_cut=[])
        g = pfg.get_graph()
        assert g.has_edge(0, 1)

    def test_negative_flow_edge_is_reversed(self):
        """A negative init flow from idx_or→idx_ex must be drawn in reverse."""
        pfg = PowerFlowGraph(_simple_powerflow_topo(), lines_cut=[])
        g = pfg.get_graph()
        assert g.has_edge(2, 1)
        assert not g.has_edge(1, 2)

    def test_edges_receive_gray_colour_by_default(self):
        pfg = PowerFlowGraph(_simple_powerflow_topo(), lines_cut=[])
        g = pfg.get_graph()
        for _, _, color in g.edges(data="color"):
            assert color == "gray"

    def test_set_electrical_node_number_sets_peripheries(self):
        pfg = PowerFlowGraph(_simple_powerflow_topo(), lines_cut=[])
        pfg.set_electrical_node_number({0: 1, 1: 2, 2: 1})
        g = pfg.get_graph()
        assert g.nodes[1]["peripheries"] == 2

    def test_set_voltage_level_color_applies_lookup(self):
        pfg = PowerFlowGraph(_simple_powerflow_topo(), lines_cut=[])
        pfg.set_voltage_level_color({0: 400, 1: 225, 2: 63})
        g = pfg.get_graph()
        # Uses the packaged ``default_voltage_colors`` mapping
        assert g.nodes[0]["color"] == "red"
        assert g.nodes[1]["color"] == "darkgreen"
        assert g.nodes[2]["color"] == "purple"


# ---------------------------------------------------------------------------
# Behavioural coverage — Structured_Overload_Distribution_Graph
# ---------------------------------------------------------------------------

def _make_structured_overload_input():
    """Hand-built overflow graph with one constraint and one loop path.

    ``A`` and ``B`` are amont nodes, ``C`` and ``D`` are aval nodes, and
    ``X`` is a side node that carries a coral "loop" bypass::

                   blue          black           blue
            A ──────────► B ──────────► C ──────────► D
             ╲                                         ╱
              ╲──── coral ─────► X ──── coral ───────╱

    Expected structural elements:
      * constrained_edge = ('B', 'C', 0)
      * constrained path = A → B → C → D
      * loop path       = A → X → D
      * hubs            = {A, D}
    """
    g = nx.MultiDiGraph()
    # constrained path (amont -> constraint -> aval)
    g.add_edge("A", "B", color="blue",  capacity=-5.0, name="line_AB")
    g.add_edge("B", "C", color="black", capacity=-10.0, name="line_BC",
               constrained=True)
    g.add_edge("C", "D", color="blue",  capacity=-5.0, name="line_CD")
    # loop path (coral bypass)
    g.add_edge("A", "X", color="coral", capacity=3.0, name="line_AX")
    g.add_edge("X", "D", color="coral", capacity=3.0, name="line_XD")
    return g


class TestStructuredOverloadDistributionGraph:
    """End-to-end tests for :class:`Structured_Overload_Distribution_Graph`."""

    def test_constructor_identifies_constrained_edge(self):
        sg = Structured_Overload_Distribution_Graph(_make_structured_overload_input())
        # constrained_edge is the (u, v, key) tuple carrying the black colour
        assert sg.constrained_path.constrained_edge[:2] == ("B", "C")

    def test_full_constrained_path_walks_amont_constraint_aval(self):
        sg = Structured_Overload_Distribution_Graph(_make_structured_overload_input())
        full = sg.constrained_path.full_n_constrained_path()
        assert full == ["A", "B", "C", "D"]

    def test_find_hubs_returns_amont_and_aval_endpoints(self):
        sg = Structured_Overload_Distribution_Graph(_make_structured_overload_input())
        hubs = set(sg.get_hubs())
        # A has a coral out-edge → amont hub; D has a coral in-edge → aval hub.
        # B and C cannot be hubs (no coral neighbour on the correct side).
        assert hubs == {"A", "D"}

    def test_loops_dataframe_contains_the_bypass_path(self):
        sg = Structured_Overload_Distribution_Graph(_make_structured_overload_input())
        loops = sg.get_loops()
        assert not loops.empty
        # The first path goes A → X → D via the two coral edges.
        assert ["A", "X", "D"] in loops["Path"].tolist()

    def test_get_constrained_edges_nodes_lists_named_lines(self):
        sg = Structured_Overload_Distribution_Graph(_make_structured_overload_input())
        edges, nodes, _other_edges, _other_nodes = sg.get_constrained_edges_nodes()
        assert set(edges) == {"line_AB", "line_BC", "line_CD"}
        assert nodes == ["A", "B", "C", "D"]

    def test_get_dispatch_edges_nodes_returns_loop_path_members(self):
        sg = Structured_Overload_Distribution_Graph(_make_structured_overload_input())
        lines, nodes = sg.get_dispatch_edges_nodes()
        assert set(lines) == {"line_AX", "line_XD"}
        assert set(nodes) == {"A", "X", "D"}


# ---------------------------------------------------------------------------
# Behavioural coverage — shortest_path_mandatory_and_promoted
# ---------------------------------------------------------------------------

class TestShortestPathMandatoryAndPromoted:
    """Dedicated tests for the under-covered mandatory+promoted helper."""

    def _make_three_branch_graph(self):
        """Three parallel paths from S to T, crossing a mandatory edge M1→M2::

               heavy                   heavy
          S ───────► M1 ───────► M2 ───────► T         (weights: 10, 10)
          S ──(1)─► X ──(1)─► M1                        (cheap, via X)
                                    M2 ──(1)─► Y ──(1)─► T   (cheap, via Y)
        """
        g = nx.DiGraph()
        # heavy direct stretches
        g.add_edge("S", "M1", weight=10.0)
        g.add_edge("M2", "T", weight=10.0)
        # mandatory middle edge
        g.add_edge("M1", "M2", weight=1.0)
        # cheap alternates (with a promotable edge)
        g.add_edge("S", "X", weight=1.0)
        g.add_edge("X", "M1", weight=1.0)
        g.add_edge("M2", "Y", weight=1.0)
        g.add_edge("Y", "T", weight=1.0)
        return g

    def test_path_traverses_mandatory_edge(self):
        g = self._make_three_branch_graph()
        path, _total = shortest_path_mandatory_and_promoted(
            g, "S", "T", mandatory_edge=("M1", "M2"),
            promoted_edges=[], weight_attr="weight",
        )
        assert path is not None
        # the (M1, M2) pair must appear somewhere in the path
        pair_in_path = any(
            (u, v) == ("M1", "M2")
            for u, v in zip(path[:-1], path[1:])
        )
        assert pair_in_path

    def test_minimises_physical_weight_over_hop_count(self):
        g = self._make_three_branch_graph()
        path, total = shortest_path_mandatory_and_promoted(
            g, "S", "T", mandatory_edge=("M1", "M2"),
            promoted_edges=[], weight_attr="weight",
        )
        # The cheap alternates (S→X→M1, M2→Y→T) both cost 2, versus 10
        # for the direct heavy stretches, so the routing helper must pick
        # the longer-but-lighter path.
        assert path == ["S", "X", "M1", "M2", "Y", "T"]
        assert total == pytest.approx(5.0)

    def test_promoted_edges_break_weight_ties_when_cost_is_equal(self):
        """With two alternates of identical physical weight, the promoted
        one is preferred."""
        g = nx.DiGraph()
        # mandatory middle
        g.add_edge("M1", "M2", weight=1.0)
        # two equal-weight amont stretches: S → A → M1 vs S → B → M1
        g.add_edge("S", "A", weight=1.0)
        g.add_edge("A", "M1", weight=1.0)
        g.add_edge("S", "B", weight=1.0)
        g.add_edge("B", "M1", weight=1.0)
        # aval: only one exit
        g.add_edge("M2", "T", weight=1.0)

        # Without promotion — Dijkstra picks *some* minimum path (ties may
        # go either way). We cannot assert which.
        # With promotion of the B-route, the helper must prefer it.
        path, total = shortest_path_mandatory_and_promoted(
            g, "S", "T", mandatory_edge=("M1", "M2"),
            promoted_edges=[("S", "B"), ("B", "M1")],
            weight_attr="weight",
        )
        assert path == ["S", "B", "M1", "M2", "T"]
        assert total == pytest.approx(4.0)

    def test_returns_none_when_no_path_exists(self):
        g = nx.DiGraph()
        g.add_edge("M1", "M2", weight=1.0)
        # S and T exist in the graph (so dijkstra does not raise
        # NodeNotFound) but are disconnected from the mandatory edge, so
        # there is no valid routing → the helper must return (None, inf).
        g.add_node("S")
        g.add_node("T")
        path, total = shortest_path_mandatory_and_promoted(
            g, "S", "T", mandatory_edge=("M1", "M2"),
            promoted_edges=[], weight_attr="weight",
        )
        assert path is None
        assert total == float("inf")


# ---------------------------------------------------------------------------
# Behavioural coverage — legacy-shim interop
# ---------------------------------------------------------------------------

class TestLegacyShimInterop:
    """Smoke test that code importing via the legacy shim still works."""

    def test_instance_built_via_shim_is_accepted_by_package_classes(self):
        # Re-import both sides *inside* the test: an earlier parametrized
        # test (``test_submodule_is_importable_in_isolation``) evicts
        # ``alphaDeesp.core.graphs*`` from sys.modules to catch import
        # cycles, which invalidates any class reference captured at
        # module load time. Fresh imports guarantee both names point at
        # the *current* class object in sys.modules.
        shim = importlib.import_module("alphaDeesp.core.graphsAndPaths")
        pkg = importlib.import_module("alphaDeesp.core.graphs")
        cp = shim.ConstrainedPath(
            [("A", "B", 0)], ("B", "C", 0), [("C", "D", 0)],
        )
        # Instance built via the shim import must be recognised by the
        # canonical class from the package.
        assert isinstance(cp, pkg.ConstrainedPath)
        assert cp.full_n_constrained_path() == ["A", "B", "C", "D"]
