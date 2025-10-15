import pytest
import networkx as nx
import numpy as np
from alphaDeesp.core.graphsAndPaths import PowerFlowGraph, default_voltage_colors

@pytest.fixture
def simple_topo():
    return {
        "nodes": {
            "are_prods": np.array([True, False, False]),
            "are_loads": np.array([False, True, False]),
            "prods_values": np.array([10.0]),
            "loads_values": np.array([5.0]),
        },
        "edges": {
            "idx_or": np.array([0, 1]),
            "idx_ex": np.array([1, 2]),
            "init_flows": np.array([8.0, -3.0]),
        },
    }

@pytest.fixture
def graph_obj(simple_topo):
    return PowerFlowGraph(topo=simple_topo, lines_cut=[])

# ---------- Initialization and build_graph ----------

def test_graph_is_created(graph_obj):
    g = graph_obj.get_graph()
    assert isinstance(g, nx.MultiDiGraph)
    assert len(g.nodes) == 3
    assert len(g.edges) == 2

def test_nodes_have_expected_attributes(graph_obj):
    g = graph_obj.get_graph()
    # Node 0 is prod (10 - 0)
    n0 = g.nodes[0]
    assert n0["prod_or_load"] == "prod"
    assert n0["fillcolor"] == "coral"
    # Node 1 is load (0 - 5)
    n1 = g.nodes[1]
    assert n1["prod_or_load"] == "load"
    assert n1["fillcolor"] == "lightblue"

def test_edges_have_correct_direction_and_labels(graph_obj):
    g = graph_obj.get_graph()
    edges = list(g.edges(data=True))
    # Flow 8.0 → edge 0→1
    e1 = [e for e in edges if e[0] == 0 and e[1] == 1][0]
    assert e1[2]["label"] == "%.2f" % 8.0
    # Flow -3.0 → edge reversed 2→1
    e2 = [e for e in edges if e[0] == 2 and e[1] == 1][0]
    assert e2[2]["label"] == "%.2f" % abs(-3.0)

# ---------- build_nodes edge cases ----------

def test_zero_prod_and_load_color():
    topo = {
        "nodes": {
            "are_prods": np.array([False]),
            "are_loads": np.array([False]),
            "prods_values": np.array([]),
            "loads_values": np.array([]),
        },
        "edges": {"idx_or": np.array([]), "idx_ex": np.array([]), "init_flows": np.array([])},
    }
    pf = PowerFlowGraph(topo=topo, lines_cut=[])
    g = pf.get_graph()
    n = g.nodes[0]
    assert n["fillcolor"] == "#ffffed"

# ---------- set_voltage_level_color ----------

def test_set_voltage_level_color(graph_obj):
    g = graph_obj.get_graph()
    voltage_levels = {0: 400, 1: 225, 2: 20}
    graph_obj.set_voltage_level_color(voltage_levels)
    for node, level in voltage_levels.items():
        assert g.nodes[node]["color"] == default_voltage_colors[level]

# ---------- set_electrical_node_number ----------

def test_set_electrical_node_number(graph_obj):
    g = graph_obj.get_graph()
    nodal_numbers = {0: 1, 1: 2, 2: 3}
    graph_obj.set_electrical_node_number(nodal_numbers)
    for node, per in nodal_numbers.items():
        assert g.nodes[node]["peripheries"] == per

# ---------- build_edges edge cases ----------

def test_build_edges_with_zero_flow():
    g = nx.MultiDiGraph()
    pf = PowerFlowGraph(
        topo={
            "nodes": {
                "are_prods": np.array([False, False]),
                "are_loads": np.array([False, False]),
                "prods_values": np.array([]),
                "loads_values": np.array([]),
            },
            "edges": {
                "idx_or": np.array([0]),
                "idx_ex": np.array([1]),
                "init_flows": np.array([0.0]),
            },
        },
        lines_cut=[]
    )
    g = pf.get_graph()
    e = list(g.edges(data=True))[0][2]
    assert float(e["penwidth"]) >= 0.1  # minimal width ensured

# ---------- plot behavior ----------

def test_plot_uses_printer(monkeypatch, graph_obj):
    called = {}

    class DummyPrinter:
        def __init__(self, folder): pass
        def display_geo(self, g, layout, name):
            called["used"] = True

    import alphaDeesp.core.graphsAndPaths as gp
    monkeypatch.setattr(gp, "Printer", DummyPrinter)

    graph_obj.layout = {"a": (0, 0)}
    graph_obj.plot(save_folder="tmp", name="test")
    assert called["used"]

# ---------- plot with sim that has plot() method ----------

def test_plot_uses_sim_plot(monkeypatch, graph_obj):
    called = {"plot_used": False, "create_namefile": False}

    class DummyPrinter:
        def __init__(self, folder): pass
        def create_namefile(self, *args, **kwargs):
            called["create_namefile"] = True
            return ("geo", "fake_path")

    class DummySim:
        def __init__(self):
            self.obs = {"dummy": 1}
            self.obs_linecut = {"dummy": 2}
        def plot(self, obs, save_file_path):
            called["plot_used"] = True
            called["obs_used"] = obs
            called["save_path"] = save_file_path

    import alphaDeesp.core.graphsAndPaths as gp
    monkeypatch.setattr(gp, "Printer", DummyPrinter)

    sim = DummySim()
    # Test default "before"
    graph_obj.plot(save_folder="tmp", name="test_sim", sim=sim)
    assert called["plot_used"]
    assert called["create_namefile"]
    assert called["obs_used"] == sim.obs
    assert called["save_path"] == "fake_path"

    # Test with state="after" (uses obs_linecut)
    called["plot_used"] = False
    graph_obj.plot(save_folder="tmp", name="test_sim_after", state="after", sim=sim)
    assert called["plot_used"]
    assert called["obs_used"] == sim.obs_linecut

# ---------- integration-like check ----------

def test_get_graph_is_consistent(graph_obj):
    g1 = graph_obj.get_graph()
    g2 = graph_obj.get_graph()
    assert g1 is g2  # same reference retained
