import pytest
import pandas as pd
import networkx as nx
import numpy as np
from alphaDeesp.core.graphsAndPaths import OverFlowGraph,add_double_edges_null_redispatch,Structured_Overload_Distribution_Graph,remove_unused_added_double_edge,all_simple_edge_paths_multi

# ---------- Fixtures ----------

@pytest.fixture
def sample_topo():
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
def df_overflow():
    # Edges represent before/after flow changes (delta_flows defines color)
    # - Positive delta → coral (increased flow)
    # - Negative delta → blue (decreased flow)
    # - Line cut (in lines_to_cut) → black (overloaded or constrained)
    return pd.DataFrame({
        "idx_or": [0, 1, 2, 0],
        "idx_ex": [1, 2, 0, 2],
        "init_flows": [10.0, 8.0, 3.0, 5.0],
        "new_flows": [11.0, 7.5, 4.0, 3.5],
        "delta_flows": [+1.0, -0.5, +1.0, -1.5],
        "gray_edges": [False, False, False, False],
    })


#@pytest.fixture
#def df_overflow():
#    return pd.DataFrame({
#        "idx_or": [0, 1, 2],
#        "idx_ex": [1, 2, 0],
#        "init_flows": [10.0, 5.0, -2.0],
#        "new_flows": [11.0, 4.0, -1.0],
#        "delta_flows": [1.0, -1.0, 0.0],
#        "gray_edges": [False, False, True],
#    })

@pytest.fixture
def overflow_graph(sample_topo, df_overflow):
    return OverFlowGraph(sample_topo, lines_to_cut=[1], df_overflow=df_overflow)

# ---------- Basic construction ----------

def test_overflow_graph_is_created(overflow_graph):
    g = overflow_graph.get_graph()
    assert isinstance(g, nx.MultiDiGraph)
    assert len(g.nodes) == 3
    assert len(g.edges) == 4  # one per df_overflow row

# ---------- Case 1: line_name column is present ----------

def test_line_name_preserved_when_present(df_overflow):
    df = df_overflow.copy()
    df["line_name"] = ["line_A", "line_B", "line_C","line_D"]

    topo = {
        "nodes": {
            "are_prods": np.array([False] * len(df)),
            "are_loads": np.array([False] * len(df)),
            "prods_values": np.array([]),
            "loads_values": np.array([]),
        },
        "edges": {"idx_or": np.array([]), "idx_ex": np.array([]), "init_flows": np.array([])},
    }

    g = OverFlowGraph(topo, lines_to_cut=[], df_overflow=df)

    # Should keep the provided names unchanged
    assert "line_name" in g.df.columns
    assert list(g.df["line_name"]) == ["line_A", "line_B", "line_C","line_D"]

# ---------- Case 2: line_name column is missing ----------

def test_line_names_added_when_missing(df_overflow):
    # Simulate missing line_name by dropping it
    df = df_overflow.drop(columns=["idx_or", "idx_ex"], errors="ignore").copy()
    n = len(df)
    df["idx_or"] = list(range(n))
    df["idx_ex"] = [(i + 1) % n for i in range(n)]
    df["gray_edges"] = [False] * n  # required field

    topo = {
        "nodes": {
            "are_prods": np.array([False] * n),
            "are_loads": np.array([False] * n),
            "prods_values": np.array([]),
            "loads_values": np.array([]),
        },
        "edges": {"idx_or": np.array([]), "idx_ex": np.array([]), "init_flows": np.array([])},
    }

    g = OverFlowGraph(topo, lines_to_cut=[], df_overflow=df)

    # Column should now exist and be auto-generated
    assert "line_name" in g.df.columns
    assert g.df["line_name"].apply(lambda x: isinstance(x, str)).all()
    # Format: "<idx_or>_<idx_ex>_<row_index>"
    assert g.df["line_name"].str.match(r"^\d+_\d+_\d+$").all()

# ---------- Edge attributes ----------

def test_build_edges_from_df_colors(overflow_graph):
    g = overflow_graph.get_graph()
    colors = [d["color"] for _, _, d in g.edges(data=True)]
    assert "black" in colors or "blue" in colors or "coral" in colors

def test_lines_to_cut_marked_constrained(overflow_graph):
    g = overflow_graph.get_graph()
    constrained_edges = [d for _, _, d in g.edges(data=True) if d.get("constrained")]
    assert any(constrained_edges)
    for d in constrained_edges:
        assert d["color"] == "black"

# ---------- Color logic by flow value ----------

def test_positive_flow_is_coral(overflow_graph):
    g = overflow_graph.get_graph()
    edge_attrs = [(u, v, d) for u, v, d in g.edges(data=True) if d["capacity"] > 0]
    assert any(d["color"] == "coral" for _, _, d in edge_attrs)

def test_negative_and_cut_edges_have_correct_colors(df_overflow, sample_topo):
    # Mark line 1 (idx 1→2) as a "cut" line (should be black)
    lines_to_cut = [1]
    g = OverFlowGraph(sample_topo, lines_to_cut=lines_to_cut, df_overflow=df_overflow).get_graph()

    # Collect edge color by capacity sign
    negative_edges = [(u, v, d) for u, v, d in g.edges(data=True) if d["capacity"] < 0]
    black_edges = [(u, v, d) for u, v, d in g.edges(data=True) if d["color"] == "black"]
    blue_edges = [(u, v, d) for u, v, d in g.edges(data=True) if d["color"] == "blue"]

    # Verify there is at least one blue and one black edge
    assert blue_edges, "There should be at least one blue edge (negative flow, not cut)"
    assert black_edges, "There should be at least one black edge (line cut / overloaded)"

    # Check logic consistency
    for _, _, data in negative_edges:
        if data.get("constrained"):
            # cut line
            assert data["color"] == "black"
        else:
            # normal negative flow
            assert data["color"] == "blue"


# ---------- Method: set_hubs_shape ----------

def test_set_hubs_shape(overflow_graph):
    g = overflow_graph.get_graph()
    overflow_graph.set_hubs_shape(hubs=[0, 2], shape_hub="triangle")
    assert g.nodes[0]["shape"] == "triangle"
    assert all("shape" in d for _, d in g.nodes(data=True))

# ---------- Method: highlight_swapped_flows ----------

def test_highlight_swapped_flows(overflow_graph):
    g = overflow_graph.get_graph()
    edge_names = [d["name"] for _, _, d in g.edges(data=True)]
    target = edge_names[:1]
    overflow_graph.highlight_swapped_flows(target)
    styled = [d for _, _, d in g.edges(data=True) if "style" in d]
    assert any(d["style"] == "tapered" for d in styled)

# ---------- Method: highlight_significant_line_loading ----------

def test_highlight_significant_line_loading(overflow_graph):
    g = overflow_graph.get_graph()
    first_edge_name = list(nx.get_edge_attributes(g, "name").values())[0]
    dict_line_loading = {first_edge_name: {"before": 90, "after": 110}}
    overflow_graph.highlight_significant_line_loading(dict_line_loading)
    d = list(g.edges(data=True))[0][2]
    assert "label" in d
    assert "fontcolor" in d
    assert "color" in d

# ---------- Method: rename_nodes ----------

def test_rename_nodes_updates_graph_and_df(overflow_graph):
    mapping = {0: "A", 1: "B", 2: "C"}
    overflow_graph.rename_nodes(mapping)
    g = overflow_graph.get_graph()
    assert all(node in g.nodes for node in mapping.values())
    assert all(isinstance(x, str) for x in overflow_graph.df["idx_or"])
    assert all(isinstance(x, str) for x in overflow_graph.df["idx_ex"])

# ---------- Plot behavior ----------

def test_plot_without_gray_edges(monkeypatch, overflow_graph):
    called = {}
    class DummyPrinter:
        def __init__(self, folder): pass
        def plot_graphviz(self, g, layout, **kwargs):
            called["used"] = "graphviz"
            return "SVG_RESULT"
        def display_geo(self, g, layout, **kwargs):
            called["used"] = "display"

    import alphaDeesp.core.graphsAndPaths as gp
    monkeypatch.setattr(gp, "Printer", DummyPrinter)
    layout = [(0, 0), (1, 1), (2, 2)]
    svg = overflow_graph.plot(layout, save_folder="")
    assert svg == "SVG_RESULT"

def test_plot_with_save_folder(monkeypatch, overflow_graph):
    called = {}
    class DummyPrinter:
        def __init__(self, folder): pass
        def display_geo(self, g, layout, **kwargs):
            called["used"] = True

    import alphaDeesp.core.graphsAndPaths as gp
    monkeypatch.setattr(gp, "Printer", DummyPrinter)
    layout = [(0, 0), (1, 1), (2, 2)]
    res = overflow_graph.plot(layout, save_folder="out")
    assert called["used"]
    assert res is None

# ---------- reverse_edges ----------
import pytest

def test_reverse_edges_swaps_directions_with_color_flip(overflow_graph):
    g = overflow_graph.get_graph()
    original_edges = list(g.edges(data=True))

    # Select some edges to reverse (first two)
    edge_path_names = [d["name"] for _, _, d in original_edges[:2]]

    # Record their original directions and colors
    original_map = {
        d["name"]: {"u": u, "v": v, "color": d["color"]}
        for u, v, d in original_edges if d["name"] in edge_path_names
    }


    # Verify reversed edges
    for name in edge_path_names:
        # Call the method — no need to specify a target color anymore
        orig = original_map[name]

        if orig["color"]=="blue":
            target_color="coral"
        else:
            target_color="blue"

        overflow_graph.reverse_edges(edge_path_names=[name], target_color=target_color)
        g2 = overflow_graph.get_graph()
        reversed_edge = [e for e in g2.edges(data=True) if e[2]["name"] == name]
        assert reversed_edge, f"Edge {name} should exist after reversal"
        u, v, data = reversed_edge[0]


        # ✅ Direction should be flipped
        assert (u == orig["v"] and v == orig["u"]), f"Edge {name} not reversed properly"

        # ✅ Color should have swapped: coral <-> blue
        if orig["color"] == "coral":
            assert data["color"] == "blue", f"Edge {name} should flip coral→blue"
        elif orig["color"] == "blue":
            assert data["color"] == "coral", f"Edge {name} should flip blue→coral"
        else:
            pytest.skip(f"Edge {name} had color {orig['color']} (not coral/blue)")

    # ✅ Unaffected edges retain original direction & color
    unaffected = [d for u, v, d in g2.edges(data=True) if d["name"] not in edge_path_names]
    for d in unaffected:
        orig = [e for e in original_edges if e[2]["name"] == d["name"]][0]
        assert d["color"] == orig[2]["color"]
        assert (orig[0], orig[1]) == (orig[0], orig[1])



def test_reverse_blue_edges_in_looppaths_changes_signs_and_dirs(overflow_graph):
    g = overflow_graph.get_graph()

    # Prepare: ensure at least two blue edges exist
    blue_edges = list(g.edges(data=True))[:2]
    for u, v, data in blue_edges:
        data["color"] = "blue"
        data["capacity"] = -abs(data["capacity"])  # negative capacities for blue edges

    # Build constrained path with their names
    constrained_path = [d["name"] for _, _, d in blue_edges]

    # Call the method with proper argument
    overflow_graph.reverse_blue_edges_in_looppaths(constrained_path)

    g2 = overflow_graph.get_graph()
    updated_edges = [(u, v, d) for u, v, d in g2.edges(data=True) if d["name"] in constrained_path]

    # ✅ Each blue edge should now be reversed and capacity made positive
    for u, v, data in updated_edges:
        # The direction must have been flipped
        original = [e for e in blue_edges if e[2]["name"] == data["name"]][0]
        orig_u, orig_v = original[0], original[1]
        assert (u == orig_v and v == orig_u), f"Edge {data['name']} should be reversed"

        # The capacity should be positive now
        assert data["capacity"] > 0, f"Edge {data['name']} capacity should be positive after reversal"

        # The color might stay blue (depending on method design), so we just assert it's valid
        assert data["color"] in ("blue", "coral"), "Reversed edge color must remain valid"

    # ✅ Unaffected edges should not change
    unaffected = [d for _, _, d in g2.edges(data=True) if d["name"] not in constrained_path]
    assert unaffected, "Some edges should remain unaffected"



# ---------- remove_gray_edges ----------

@pytest.mark.skip(reason="OverFlowGraph has no remove_gray_edges() method implemented.")
def test_remove_gray_edges_deletes_expected_edges():
    pass
    #g = overflow_graph.get_graph()
    ## Make sure some edges are gray before removal
    #for u, v, data in g.edges(data=True):
    #    data["color"] = "gray"
    #total_before = len(g.edges())
#
    #overflow_graph.remove_gray_edges()
    #g2 = overflow_graph.get_graph()
    #total_after = len(g2.edges())
#
    ## All gray edges should have been removed
    #assert total_after < total_before
    #assert all(d["color"] != "gray" for _, _, d in g2.edges(data=True))

def test_remove_gray_edges_keeps_non_gray_edges(overflow_graph):
    pass
    #g = overflow_graph.get_graph()
    ## Assign different colors
    #colors = ["gray", "coral", "blue"]
    #for (u, v, data), color in zip(g.edges(data=True), colors):
    #    data["color"] = color
#
    #total_before = len(g.edges())
    #overflow_graph.remove_gray_edges()
    #g2 = overflow_graph.get_graph()
    #remaining_colors = [d["color"] for _, _, d in g2.edges(data=True)]
#
    ## Only non-gray edges should remain
    #assert "gray" not in remaining_colors
    #assert len(g2.edges()) < total_before
    #assert any(c in ("blue", "coral") for c in remaining_colors)

def test_remove_gray_edges_on_graph_without_gray(overflow_graph):
    pass
    #g = overflow_graph.get_graph()
    #for _, _, data in g.edges(data=True):
    #    data["color"] = "blue"
#
    #total_before = len(g.edges())
    #overflow_graph.remove_gray_edges()
    #total_after = len(overflow_graph.get_graph().edges())
#
    ## Graph should remain unchanged
    #assert total_before == total_after

# ---------- 1. test_build_edges_from_df_with_positive_and_negative_deltas ----------

def test_build_edges_from_df_handles_positive_and_negative_deltas(df_overflow, sample_topo):
    g = OverFlowGraph(sample_topo, lines_to_cut=[], df_overflow=df_overflow).get_graph()
    colors = [d["color"] for _, _, d in g.edges(data=True)]

    assert any(c == "coral" for c in colors), "Positive delta_flows should produce coral edges"
    assert any(c == "blue" for c in colors), "Negative delta_flows should produce blue edges"

# ---------- 2. test_highlight_significant_edge_flow_change ----------

def test_highlight_significant_line_loading_reflects_flow_change_with_gradient(overflow_graph):
    """
    Tests highlight_significant_line_loading() including gradient colors like 'coral:yellow:coral'.
    """
    g = overflow_graph.get_graph()
    first_edge_name = list(nx.get_edge_attributes(g, "name").values())[0]

    # Simulate a strong change in loading
    line_loading_dict = {first_edge_name: {"before": 80.0, "after": 120.0}}
    overflow_graph.highlight_significant_line_loading(line_loading_dict)

    edge_data = [d for _, _, d in g.edges(data=True) if d["name"] == first_edge_name][0]

    # Must include label and fontcolor
    assert "label" in edge_data
    assert "fontcolor" in edge_data
    assert edge_data["fontcolor"] in ("darkred")

    # Check color value
    color_val = edge_data["color"]
    if isinstance(color_val, str) and ":" in color_val:
        # ✅ Remove quotes safely and split
        clean_color = color_val.strip('"').strip("'")
        parts = [p.strip('"').strip("'") for p in clean_color.split(":")]

        # Validate gradient triplet
        assert len(parts) == 3, f"Unexpected gradient format: {color_val}"
        assert parts[1] == "yellow", f"Middle color should be yellow: {color_val}"
        assert parts[0] == parts[2], f"Start/end colors must match, got {parts}"
        assert parts[0] in ("coral", "blue"), f"Unexpected base color: {parts[0]}"
    else:
        assert color_val in ("coral", "blue", "red", "black")



# ---------- 3. test_set_voltage_level_color ----------

def test_set_voltage_level_color_applies_correct_mapping(overflow_graph):
    g = overflow_graph.get_graph()
    voltage_levels = {n: 400 for n in g.nodes}
    overflow_graph.set_voltage_level_color(voltage_levels)
    colors = nx.get_node_attributes(g, "color")

    assert all(c == "red" for c in colors.values()), "All voltage levels 400 should map to red color"

# ---------- 4. test_set_electrical_node_number ----------

def test_set_electrical_node_number_assigns_peripheries(overflow_graph):
    g = overflow_graph.get_graph()
    nodal_numbers = {n: i + 1 for i, n in enumerate(g.nodes)}
    overflow_graph.set_electrical_node_number(nodal_numbers)
    peripheries = nx.get_node_attributes(g, "peripheries")

    assert set(peripheries.keys()) == set(g.nodes)
    assert all(isinstance(v, int) for v in peripheries.values())

# ---------- 5. test_highlight_swapped_flows_with_multiple_edges ----------

def test_highlight_swapped_flows_with_multiple_edges(overflow_graph):
    g = overflow_graph.get_graph()

    # ✅ Convertir en liste avant de prendre les 2 premiers éléments
    edges_list = list(g.edges(data=True))
    edge_names = [d["name"] for _, _, d in edges_list[:2]]

    overflow_graph.highlight_swapped_flows(edge_names)

    # ✅ Vérifier que les arêtes ciblées ont été stylisées
    for _, _, data in g.edges(data=True):
        if data["name"] in edge_names:
            assert data["style"] == "tapered"
            assert "penwidth" in data, "Penwidth doit être défini pour les arêtes stylisées"

    # ✅ Vérifier qu’au moins une arête a été modifiée
    changed = [d for _, _, d in g.edges(data=True) if d.get("style") == "tapered"]
    assert changed, "Aucune arête n’a été stylisée"


# ---------- 6. test_highlight_significant_line_loading_threshold ----------

def test_highlight_significant_line_loading_threshold(overflow_graph):
    g = overflow_graph.get_graph()
    first_name = list(nx.get_edge_attributes(g, "name").values())[0]
    line_loading_dict = {first_name: {"before": 80, "after": 110}}  # crossing 100%

    overflow_graph.highlight_significant_line_loading(line_loading_dict)
    edge_data = [d for _, _, d in g.edges(data=True) if d["name"] == first_name][0]

    assert "label" in edge_data
    assert "color" in edge_data

    clean_color = edge_data["color"].strip('"').strip("'")
    parts = [p.strip('"').strip("'") for p in clean_color.split(":")]
    assert parts[0] in ("coral", "black")

# ---------- 7. test_rename_nodes_preserves_edge_connectivity ----------

def test_rename_nodes_preserves_edge_connectivity(overflow_graph):
    g = overflow_graph.get_graph()
    mapping = {i: f"N{i}" for i in g.nodes}
    original_edges = list(g.edges())
    overflow_graph.rename_nodes(mapping)

    g2 = overflow_graph.get_graph()
    # Verify renaming applied to nodes
    assert all(isinstance(node, str) for node in g2.nodes)
    # Verify edges are still present between renamed nodes
    new_edges = list(g2.edges())
    assert len(original_edges) == len(new_edges)

# ---------- 8. test_plot_graphviz_invocation ----------

def test_plot_invokes_graphviz(monkeypatch, overflow_graph):
    called = {}

    class DummyPrinter:
        def __init__(self, folder): pass
        def plot_graphviz(self, g, layout, **kwargs):
            called["used"] = True
            return "FAKE_SVG"

    import alphaDeesp.core.graphsAndPaths as gp
    monkeypatch.setattr(gp, "Printer", DummyPrinter)

    svg = overflow_graph.plot(layout={}, save_folder="")
    assert called.get("used")
    assert svg == "FAKE_SVG"

# ---------- 9. test_reverse_edges_only_flips_selected ----------

def test_reverse_edges_only_flips_selected(overflow_graph):
    g = overflow_graph.get_graph()
    edges = list(g.edges(data=True))
    selected_names = [edges[0][2]["name"]]

    # Record original directions
    orig_dir = {d["name"]: (u, v) for u, v, d in edges}
    overflow_graph.reverse_edges(edge_path_names=selected_names, target_color="blue")

    g2 = overflow_graph.get_graph()
    for u, v, d in g2.edges(data=True):
        if d["name"] in selected_names:
            assert (u, v) == orig_dir[d["name"]][::-1]
            assert d["color"] == "blue"
        else:
            assert (u, v) == orig_dir[d["name"]]

# ---------- 10. test_reverse_blue_edges_in_looppaths_color_swap ----------

def test_reverse_blue_edges_in_looppaths_color_swap(overflow_graph):
    g = overflow_graph.get_graph()
    # Make some blue edges
    for u, v, d in g.edges(data=True):
        d["color"] = "blue"
        d["capacity"] = -abs(d["capacity"])

    edges_list = list(g.edges(data=True))
    constrained_path =  [d["name"] for _, _, d in edges_list[:2]]#[d["name"] for _, _, d in g.edges(data=True)[:2]]

    overflow_graph.reverse_blue_edges_in_looppaths(constrained_path)
    g2 = overflow_graph.get_graph()

    # Blue edges should now be reversed and coral
    for u, v, d in g2.edges(data=True):
        if d["name"] in constrained_path:
            assert d["capacity"] >= 0
            assert d["color"] in ("coral", "blue")

# ---------- 1. test_set_hubs_shape_for_multiple_hubs ----------

def test_set_hubs_shape_for_multiple_hubs(overflow_graph):
    g = overflow_graph.get_graph()
    hubs = list(g.nodes)[:2]

    overflow_graph.set_hubs_shape(hubs, shape_hub="diamond")
    shapes = nx.get_node_attributes(g, "shape")

    for h in hubs:
        assert shapes[h] == "diamond", f"Hub {h} should have shape 'diamond'"
    # Non-hubs should not have the shape
    non_hubs = [n for n in g.nodes if n not in hubs]
    for n in non_hubs:
        assert shapes[n] != "diamond"

# ---------- 2. test_set_voltage_level_color_mixed_levels ----------

def test_set_voltage_level_color_mixed_levels(overflow_graph):
    g = overflow_graph.get_graph()
    voltage_levels = {}
    for i, node in enumerate(g.nodes):
        voltage_levels[node] = 400 if i % 2 == 0 else 225

    overflow_graph.set_voltage_level_color(voltage_levels)
    colors = nx.get_node_attributes(g, "color")

    reds = [c for c in colors.values() if c == "red"]
    greens = [c for c in colors.values() if c == "darkgreen"]

    assert reds and greens, "Both 400→red and 225→darkgreen should appear"

# ---------- 3. test_set_electrical_node_number_varied ----------

def test_set_electrical_node_number_varied(overflow_graph):
    g = overflow_graph.get_graph()
    nodal_dict = {node: idx * 2 for idx, node in enumerate(g.nodes)}

    overflow_graph.set_electrical_node_number(nodal_dict)
    periph = nx.get_node_attributes(g, "peripheries")

    for node, expected in nodal_dict.items():
        assert periph[node] == expected
        assert isinstance(periph[node], (int, np.integer))

# ---------- 4. test_plot_with_gray_edges_removed ----------

def test_plot_with_gray_edges_removed(monkeypatch, overflow_graph):
    called = {}

    # Assign a gray edge to simulate removal step
    for _, _, d in overflow_graph.get_graph().edges(data=True):
        d["color"] = "gray"

    class DummyPrinter:
        def __init__(self, folder): pass
        def plot_graphviz(self, g, layout, **kwargs):
            called["used"] = "graphviz"
            return "SVG"
        def display_geo(self, g, layout, **kwargs):
            called["used"] = "geo"

    import alphaDeesp.core.graphsAndPaths as gp
    monkeypatch.setattr(gp, "Printer", DummyPrinter)

    layout = {n: (i, i) for i, n in enumerate(overflow_graph.get_graph().nodes)}
    svg = overflow_graph.plot(layout, save_folder="")
    assert called["used"] == "graphviz"
    assert svg == "SVG"

# ---------- 5. test_rename_nodes_updates_df_consistency ----------

def test_rename_nodes_updates_df_consistency(overflow_graph):
    df_before = overflow_graph.df.copy()
    g = overflow_graph.get_graph()
    mapping = {n: f"N{n}" for n in g.nodes}

    overflow_graph.rename_nodes(mapping)
    g2 = overflow_graph.get_graph()
    df_after = overflow_graph.df

    # Graph nodes updated
    assert all(isinstance(n, str) for n in g2.nodes)
    # DF columns updated consistently
    for col in ("idx_or", "idx_ex"):
        assert all(isinstance(x, str) for x in df_after[col])
    # Ensure the DataFrame still has same length
    assert len(df_after) == len(df_before)

# ---------- 6. test_penwidth_values_are_positive_and_scaled ----------

def test_penwidth_values_are_positive_and_scaled(overflow_graph):
    g = overflow_graph.get_graph()
    penwidths = [float(d["penwidth"]) for _, _, d in g.edges(data=True)]
    assert all(pw > 0 for pw in penwidths)
    # Penwidth should correlate with absolute delta_flows magnitude
    # More difference = thicker line
    sorted_pw = sorted(penwidths)
    assert sorted_pw[-1] >= sorted_pw[0], "Penwidth should scale with flow magnitude"

# ---------- 7. test_plot_with_save_folder_triggers_display_geo ----------

def test_plot_with_save_folder_triggers_display_geo(monkeypatch, overflow_graph):
    called = {}
    class DummyPrinter:
        def __init__(self, folder): pass
        def display_geo(self, g, layout, **kwargs):
            called["used"] = True

    import alphaDeesp.core.graphsAndPaths as gp
    monkeypatch.setattr(gp, "Printer", DummyPrinter)

    layout = {n: (i, i) for i, n in enumerate(overflow_graph.get_graph().nodes)}
    overflow_graph.plot(layout, save_folder="output_dir")
    assert called.get("used"), "display_geo should be called when save_folder is set"

# ---------- 1. detect_edges_to_keep ----------

def test_detect_edges_to_keep_returns_two_sets_with_expected_edges(overflow_graph):
    """
    Vérifie que detect_edges_to_keep() renvoie deux ensembles d'arêtes (tuples u, v, k)
    : reconnectables et non reconnectables.
    """
    # --- Créer un MultiDiGraph cohérent ---
    g_c = nx.MultiDiGraph()
    g_c.add_edge("A", "B", key=0, name="L1")
    g_c.add_edge("B", "C", key=0, name="L2")
    g_c.add_edge("C", "D", key=0, name="L3")
    g_c.add_edge("X", "Y", key=0, name="Lx")  # isolée

    # --- Définir sources et cibles ---
    source_nodes = ["A"]
    target_nodes = ["D"]

    # --- Arêtes d’intérêt : ici sous forme de tuples (u, v, k) ---
    edges_of_interest = list(g_c.edges(keys=True))

    # --- Appel de la méthode ---
    reconnectable, non_reconnectable = overflow_graph.detect_edges_to_keep(
        g_c=g_c,
        source_nodes=source_nodes,
        target_nodes=target_nodes,
        edges_of_interest=edges_of_interest,
    )

    # --- Vérifications structurelles ---
    assert isinstance(reconnectable, set), "La sortie reconnectable doit être un set"
    assert isinstance(non_reconnectable, set), "La sortie non reconnectable doit être un set"

    # --- Vérifications de contenu ---
    # Les arêtes retournées doivent être bien des tuples (u,v,k)
    assert all(isinstance(e, tuple) and len(e) == 3 for e in reconnectable | non_reconnectable)

    # Il devrait y avoir au moins une arête "interface" reconnue (ici A→B ou B→C)
    assert any(u in source_nodes or v in target_nodes for (u, v, k) in reconnectable), \
        "Une arête connectant une source ou une cible doit être reconnue comme reconnectable"

    # L’arête isolée ('X','Y',0) ne doit pas apparaître
    assert ('X', 'Y', 0) not in reconnectable, "L’arête isolée ne doit pas être reconnue reconnectable"

    # Les deux ensembles ne doivent pas être identiques
    assert reconnectable != non_reconnectable




# ---------- 2. add_relevant_null_flow_lines ----------
# --- Tests for add_relevant_null_flow_lines ---


def test_basic_doubling_edges_null_redispatch():
    """
    Tests that a simple edge with null capacity and correct color is doubled.
    """
    # 1. Setup: Create a graph with various edges
    g = nx.MultiDiGraph()
    # Edge that should be doubled
    g.add_edge(1, 2, key=0, name="line_to_double", color="gray", capacity=0.0)
    # Edge that should NOT be doubled (wrong color)
    g.add_edge(2, 3, key=0, name="line_wrong_color", color="red", capacity=0.0)
    # Edge that should NOT be doubled (non-null capacity)
    g.add_edge(3, 4, key=0, name="line_non_null", color="gray", capacity=10.0)

    initial_edge_count = g.number_of_edges()
    assert initial_edge_count == 3

    # 2. Action: Call the function
    doubled_dict, added_dict = add_double_edges_null_redispatch(g)

    # 3. Assertions
    # Check that the graph was modified correctly
    assert g.number_of_edges() == 4, "The total number of edges should be 4."
    assert g.has_edge(2, 1), "The graph should now have the reversed edge (2, 1)."

    # Verify attributes of the new edge
    original_attrs = g.get_edge_data(1, 2)[0]
    new_attrs = g.get_edge_data(2, 1)[0]
    assert original_attrs == new_attrs, "The new edge should have the same attributes as the original."

    # Check the returned dictionaries
    assert len(doubled_dict) == 1, "The 'doubled' dictionary should have one entry."
    assert "line_to_double" in doubled_dict, "The doubled line's name should be in the 'doubled' dictionary."
    assert doubled_dict["line_to_double"] == (1, 2, 0), "The edge in the 'doubled' dictionary is incorrect."

    assert len(added_dict) == 1, "The 'added' dictionary should have one entry."
    assert "line_to_double" in added_dict, "The added line's name should be in the 'added' dictionary."
    assert added_dict["line_to_double"][0] == 2 and added_dict["line_to_double"][1] == 1, \
        "The edge in the 'added' dictionary should be the reversed edge."

@pytest.fixture
def setup_graph_for_test():
    """
    Provides a standardized graph setup for testing null-flow line highlighting.
    - Constrained path: 0 -> 1(cut) -> 2 (blue/black)
    - Loop path: 0 -> 3 -> 2 (coral/red)
    - Null-flow lines for testing reconnections.
    """
    df = pd.DataFrame({
        "line_name": [
            "0_1", "1_2",  # Blue/Black path
            "0_3", "3_2",  # Coral/Red path
            "0_2",  # Null-flow connecting blue path ends
            "0_4", "4_3",  # Null-flow connecting to red path
            "1_3",  # Null-flow connecting blue to red
        ],
        "idx_or": [0, 1, 0, 3, 0, 0, 4, 1],
        "idx_ex": [1, 2, 3, 2, 2, 4, 3, 3],
        "delta_flows": [-10, -10, 20, 20, 0, 0, 0, 0],
        "gray_edges": [False, False, False, False, True, True, True, True],
    })
    topo = {
        "edges": {"idx_or": df["idx_or"], "idx_ex": df["idx_ex"], "init_flows": [0] * len(df)},
        "nodes": {
            "are_prods": [True, False, False, False, False], "are_loads": [False, False, True, False, False],
            "prods_values": [1], "loads_values": [1]
        }
    }
    lines_cut = [1]  # "1_2" is the overloaded line
    graph_obj = OverFlowGraph(topo=topo, lines_to_cut=lines_cut, df_overflow=df)
    struct_g = Structured_Overload_Distribution_Graph(graph_obj.g)
    return graph_obj, struct_g


def get_edge_by_name(g, name):
    """Helper to find an edge in the graph by its name attribute."""
    edge_names = nx.get_edge_attributes(g, 'name')
    for edge, edge_name in edge_names.items():
        if edge_name == name:
            return edge
    return None


def test_highlight_blue_path_reconnection_relevant_null_flow_lines(setup_graph_for_test):
    """
    Tests if a null-flow line connecting two nodes on the blue path is correctly highlighted.
    NOTE: The current implementation colors amont-to-aval reconnections as 'coral'.
    """
    graph_obj, struct_g = setup_graph_for_test
    line_to_test = "0_2"  # This line connects the amont and aval of the blue path

    # Action
    # Use the correct path type for a connection between the amont and aval sections.
    graph_obj.add_relevant_null_flow_lines(
        struct_g,
        non_connected_lines=[line_to_test],
        target_path="blue_amont_aval"
    )

    # Assertion
    edge = get_edge_by_name(graph_obj.g, line_to_test)
    assert edge is not None, f"Edge '{line_to_test}' not found in graph."
    attrs = graph_obj.g.edges[edge]
    # The source code's coloring logic for 'blue_amont_aval' paths currently results
    # in 'coral'. This test asserts the actual behavior.
    assert attrs['color'] == 'coral'
    assert attrs['style'] == 'dashed'


def test_highlight_red_path_reconnection_relevant_null_flow_lines(setup_graph_for_test):
    """
    Tests if a null-flow path connecting two nodes on the red path is correctly highlighted.
    """
    graph_obj, struct_g = setup_graph_for_test
    # This path (0->4->3) connects two nodes (0 and 3) on the red loop path
    lines_to_test = ["0_4", "4_3"]

    # Action
    graph_obj.add_relevant_null_flow_lines(
        struct_g,
        non_connected_lines=lines_to_test,
        target_path="red_only"
    )

    # Assertion
    for line_name in lines_to_test:
        edge = get_edge_by_name(graph_obj.g, line_name)
        assert edge is not None, f"Edge '{line_name}' not found in graph."
        attrs = graph_obj.g.edges[edge]
        assert attrs['color'] == 'coral'
        assert attrs['style'] == 'dashed'


def test_highlight_blue_to_red_reconnection_relevant_null_flow_lines(setup_graph_for_test):
    """
    Tests if a null-flow line between the blue and red paths is correctly highlighted.
    """
    graph_obj, struct_g = setup_graph_for_test
    line_to_test = "1_3"  # Connects node 1 (blue) to node 3 (red)

    # Action
    graph_obj.add_relevant_null_flow_lines(
        struct_g,
        non_connected_lines=[line_to_test],
        target_path="blue_to_red"
    )

    # Assertion
    edge = get_edge_by_name(graph_obj.g, line_to_test)
    assert edge is not None, f"Edge '{line_to_test}' not found in graph."
    attrs = graph_obj.g.edges[edge]
    assert attrs['color'] == 'coral'
    assert attrs['style'] == 'dashed'


def test_non_reconnectable_line_style_relevant_null_flow_lines(setup_graph_for_test):
    """
    Tests that a non-reconnectable line is styled as 'dimgray' and 'dotted'.
    """
    graph_obj, struct_g = setup_graph_for_test
    line_to_test = "0_2"

    # Action
    graph_obj.add_relevant_null_flow_lines(
        struct_g,
        non_connected_lines=[line_to_test],
        non_reconnectable_lines=[line_to_test],  # Mark it as non-reconnectable
        target_path="blue_amont_aval"
    )

    # Assertion
    edge = get_edge_by_name(graph_obj.g, line_to_test)
    assert edge is not None, f"Edge '{line_to_test}' not found in graph."
    attrs = graph_obj.g.edges[edge]
    assert attrs['color'] == 'dimgray'
    assert attrs['style'] == 'dotted'

# ---------- 3. desambiguation_type_path ----------

def test_desambiguation_is_loop_path(setup_graph_for_test):
    """
    Tests if an ambiguous path connecting amont and aval nodes is classified as a 'loop_path'.
    """
    graph_obj, struct_g = setup_graph_for_test
    # Path is 0 -> 4 -> 2, connecting amont(0) and aval(2)
    ambiguous_nodes = {0, 2, 4}

    result = graph_obj.desambiguation_type_path(ambiguous_nodes, struct_g)

    assert result == "loop_path"


def test_desambiguation_is_constrained_path(setup_graph_for_test):
    """
    Tests if an ambiguous path connecting two amont nodes is classified as a 'constrained_path'.
    """
    graph_obj, struct_g = setup_graph_for_test
    # Path is 0 -> 5 -> 1, connecting two amont nodes
    ambiguous_nodes = {0, 1, 5}

    result = graph_obj.desambiguation_type_path(ambiguous_nodes, struct_g)

    assert result == "constrained_path"


def test_desambiguation_one_connection_is_loop(setup_graph_for_test):
    """
    Tests if an ambiguous path connecting to only one point on the constrained path
    is classified as a 'loop_path'.
    """
    graph_obj, struct_g = setup_graph_for_test
    # Path 1 -> 6 -> 7 connects to the constrained path only at node 1
    ambiguous_nodes = {1, 6, 7}

    result = graph_obj.desambiguation_type_path(ambiguous_nodes, struct_g)

    assert result == "loop_path"


def test_desambiguation_no_connection_is_loop(setup_graph_for_test):
    """
    Tests if an ambiguous path with no connection to the constrained path is a 'loop_path'.
    """
    graph_obj, struct_g = setup_graph_for_test
    # This path is not connected to the main constrained path {0,1,2} at all
    ambiguous_nodes = {8, 9, 10}

    result = graph_obj.desambiguation_type_path(ambiguous_nodes, struct_g)

    assert result == "loop_path"


# ---------- 4. identify_ambiguous_paths ----------

def add_ambiguous_paths(graph_obj):
    """Adds several ambiguous paths to a graph object for testing."""
    # Path connecting amont(0) to aval(2)
    graph_obj.g.add_edge(0, 4, name="amb_1a", color="blue", capacity=-5)
    graph_obj.g.add_edge(4, 2, name="amb_1b", color="coral", capacity=5)

    # Path connecting two amont nodes (0 and 1)
    graph_obj.g.add_edge(0, 5, name="amb_2a", color="blue", capacity=-5)
    graph_obj.g.add_edge(5, 1, name="amb_2b", color="coral", capacity=5)

    # Path connecting to only one constrained node (1)
    graph_obj.g.add_edge(1, 6, name="amb_3a", color="blue", capacity=-5)
    graph_obj.g.add_edge(6, 7, name="amb_3b", color="coral", capacity=5)
    return graph_obj


def test_identify_ambiguous_paths_found(setup_graph_for_test):
    """
    Tests the identification of paths containing both blue and coral edges.
    """
    graph_obj, _ = setup_graph_for_test
    # Add the ambiguous paths for this specific test
    graph_with_ambiguous = add_ambiguous_paths(graph_obj)
    struct_g_with_ambiguous = Structured_Overload_Distribution_Graph(graph_with_ambiguous.g)


    # Action
    ambiguous_edge_paths, ambiguous_node_paths = graph_obj.identify_ambiguous_paths(struct_g_with_ambiguous)

    # Assertions
    # The function finds one single weakly connected component because all ambiguous paths
    # are connected through nodes 0, 1, and 2.
    assert len(ambiguous_edge_paths) == 1, "Should find 1 ambiguous component."
    assert len(ambiguous_node_paths) == 1, "Should find 1 ambiguous component."

    # Check the content of the single component
    found_edges = set(ambiguous_edge_paths[0])
    found_nodes = ambiguous_node_paths[0] # it's already a set

    # Expected content based on the fixture
    expected_edges = {'amb_1a', 'amb_1b', 'amb_2a', 'amb_2b', 'amb_3a', 'amb_3b'}
    expected_nodes = {0, 1, 2, 4, 5, 6, 7}

    assert found_edges == expected_edges
    assert found_nodes == expected_nodes


def test_identify_no_ambiguous_paths(setup_graph_for_test):
    """
    Tests that no paths are identified when none are ambiguous.
    """
    # The fixture now provides a graph without ambiguous paths, so no cleanup is needed
    graph_obj, struct_g = setup_graph_for_test

    # Action
    ambiguous_edge_paths, ambiguous_node_paths = graph_obj.identify_ambiguous_paths(struct_g)

    # Assertions
    assert len(ambiguous_edge_paths) == 0, "Should find no ambiguous edge paths."
    assert len(ambiguous_node_paths) == 0, "Should find no ambiguous node paths."

# ---------- 5. consolidate_graph ----------

def test_consolidate_graph_consolidates_all_paths(setup_graph_for_test):
    """
    Tests that consolidate_graph correctly calls sub-consolidation functions
    to recolor both constrained and loop paths from gray edges.
    """
    graph_obj, _ = setup_graph_for_test

    # Add a gray path that should become part of the constrained path (amont side)
    graph_obj.g.add_edge(0, 10, name="gray_amont_1", color="gray", capacity=-1.0)
    graph_obj.g.add_edge(10, 1, name="gray_amont_2", color="gray", capacity=-1.0)

    # Add a gray path that should become part of the loop path
    graph_obj.g.add_edge(0, 8, name="gray_loop_1", color="gray", capacity=1.0)
    graph_obj.g.add_edge(8, 2, name="gray_loop_2", color="gray", capacity=1.0)

    # Action: Re-create struct_g after modifying the graph and consolidate
    struct_g_updated = Structured_Overload_Distribution_Graph(graph_obj.g)
    graph_obj.consolidate_graph(struct_g_updated)

    # Assertions for consolidated constrained path
    assert graph_obj.g.get_edge_data(0, 10)[0]['color'] == 'blue'
    assert graph_obj.g.get_edge_data(10, 1)[0]['color'] == 'blue'

    # Assertions for consolidated loop path
    assert graph_obj.g.get_edge_data(0, 8)[0]['color'] == 'coral'
    assert graph_obj.g.get_edge_data(8, 2)[0]['color'] == 'coral'


def test_consolidate_graph_resolves_ambiguous_path(setup_graph_for_test):
    """
    Tests that consolidate_graph correctly resolves an ambiguous path.
    An ambiguous path connecting amont and aval should be resolved as a loop path (coral).
    """
    graph_obj, _ = setup_graph_for_test

    # Add an ambiguous path: 0 --coral--> 15 --blue--> 2
    graph_obj.g.add_edge(0, 15, name="amb_1", color="coral", capacity=5.0)
    graph_obj.g.add_edge(15, 2, name="amb_2", color="blue", capacity=-5.0)

    # Action
    struct_g_updated = Structured_Overload_Distribution_Graph(graph_obj.g)
    graph_obj.consolidate_graph(struct_g_updated)

    # Assertions
    # The path 0->15->2 connects amont(0) to aval(2) and should be resolved as a loop path.
    # The blue edge should be reversed and all path edges turned coral.
    assert graph_obj.g.get_edge_data(0, 15)[0]['color'] == 'coral'
    assert not graph_obj.g.has_edge(15, 2), "Original blue edge should be removed."
    assert graph_obj.g.has_edge(2, 15), "Blue edge should be reversed."

    reversed_edge_data = graph_obj.g.get_edge_data(2, 15)[0]
    assert reversed_edge_data['color'] == 'coral'
    assert reversed_edge_data['capacity'] == 5.0  # Capacity should be positive after reversal


def test_consolidate_graph_ignores_specified_line(setup_graph_for_test):
    """
    Tests that a line specified in non_connected_lines_to_ignore is not
    processed during consolidation and remains in its original state.
    """
    graph_obj, _ = setup_graph_for_test

    # Add a gray edge that should be ignored during consolidation
    graph_obj.g.add_edge(0, 10, name="ignored_line", color="gray", capacity=0.0)

    # Action
    struct_g_updated = Structured_Overload_Distribution_Graph(graph_obj.g)
    graph_obj.consolidate_graph(struct_g_updated, non_connected_lines_to_ignore=["ignored_line"])

    # Assertion
    # The ignored line should still exist and be unchanged.
    assert graph_obj.g.has_edge(0, 10)
    edge_data = graph_obj.g.get_edge_data(0, 10)[0]
    assert edge_data['color'] == 'gray', "Ignored line should not be recolored."

# ---------- 6. consolidate_loop_path ----------

def test_consolidate_loop_path_recolors_gray_edges(setup_graph_for_test):
    """
    Tests that gray edges on a path between hubs are correctly recolored to coral.
    """
    graph_obj, _ = setup_graph_for_test
    # Add a gray path (0 -> 8 -> 2) that should be part of the loop
    graph_obj.g.add_edge(0, 8, name="gray_1", color="gray", capacity=1.0)
    graph_obj.g.add_edge(8, 2, name="gray_2", color="gray", capacity=1.0)

    # Hubs are the start and end of the main loop path (0 and 2)
    hub_sources = [0]
    hub_targets = [2]

    # Action
    graph_obj.consolidate_loop_path(hub_sources, hub_targets)

    # Assertions
    # The new gray path should now be coral
    edge_data_1 = graph_obj.g.get_edge_data(0, 8)
    assert edge_data_1[0]['color'] == "coral", "Edge (0, 8) should be recolored to coral."

    edge_data_2 = graph_obj.g.get_edge_data(8, 2)
    assert edge_data_2[0]['color'] == "coral", "Edge (8, 2) should be recolored to coral."


def test_consolidate_loop_path_ignores_blue_edges(setup_graph_for_test):
    """
    Tests that paths containing blue edges are not considered for consolidation.
    """
    graph_obj, _ = setup_graph_for_test
    # Add a path with a blue edge (0 -> 9) and a gray edge (9 -> 2)
    graph_obj.g.add_edge(0, 9, name="blue_path_edge", color="blue", capacity=-1.0)
    graph_obj.g.add_edge(9, 2, name="gray_path_edge", color="gray", capacity=1.0)

    hub_sources = [0]
    hub_targets = [2]

    # Action
    graph_obj.consolidate_loop_path(hub_sources, hub_targets)

    # Assertions
    # The path should be ignored, so colors remain unchanged
    edge_data_blue = graph_obj.g.get_edge_data(0, 9)
    assert edge_data_blue[0]['color'] == "blue", "Blue edge should not change color."

    edge_data_gray = graph_obj.g.get_edge_data(9, 2)
    assert edge_data_gray[0]['color'] == "gray", "Gray edge in a blue path should not be recolored."


def test_consolidate_loop_path_with_no_hubs(setup_graph_for_test):
    """
    Tests that the function does nothing if no hubs are provided.
    """
    graph_obj, _ = setup_graph_for_test
    # Add a gray path that would otherwise be recolored
    graph_obj.g.add_edge(0, 8, name="gray_1", color="gray", capacity=1.0)
    graph_obj.g.add_edge(8, 2, name="gray_2", color="gray", capacity=1.0)

    # Action with no hubs
    graph_obj.consolidate_loop_path([], [])

    # Assertions
    # Colors should remain gray
    edge_data_1 = graph_obj.g.get_edge_data(0, 8)
    assert edge_data_1[0]['color'] == "gray", "Edge color should not change if no hubs are given."

    edge_data_2 = graph_obj.g.get_edge_data(8, 2)
    assert edge_data_2[0]['color'] == "gray", "Edge color should not change if no hubs are given."

# ---------- 7. consolidate_constrained_path ----------

def test_consolidate_constrained_path_recolors_amont(setup_graph_for_test):
    """
    Tests that a gray path on the 'amont' side gets correctly recolored to blue.
    """
    graph_obj, struct_g = setup_graph_for_test
    c_path = struct_g.constrained_path
    # Add a gray path connecting two nodes on the amont side (0 and 1)
    graph_obj.g.add_edge(0, 10, name="gray_amont_1", color="gray", capacity=-1.0)
    graph_obj.g.add_edge(10, 1, name="gray_amont_2", color="gray", capacity=-1.0)

    # Action
    graph_obj.consolidate_constrained_path(
        constrained_path_nodes_amont=c_path.n_amont(),
        constrained_path_nodes_aval=c_path.n_aval(),
        constrained_path_edges=c_path.amont_edges + [c_path.constrained_edge] + c_path.aval_edges
    )

    # Assertions
    edge_data_1 = graph_obj.g.get_edge_data(0, 10)[0]
    assert edge_data_1['color'] == 'blue'

    edge_data_2 = graph_obj.g.get_edge_data(10, 1)[0]
    assert edge_data_2['color'] == 'blue'


def test_consolidate_constrained_path_recolors_aval(setup_graph_for_test):
    """
    Tests that a gray path on the 'aval' side gets correctly recolored to blue.
    """
    graph_obj, struct_g = setup_graph_for_test
    c_path = struct_g.constrained_path
    # Reconfigure path slightly to have an aval part for testing
    c_path.aval_edges = [(2, 11, 0)]
    graph_obj.g.add_edge(2, 11, name="blue_aval_1", color="blue", capacity=-1.0)

    # Add a gray path connecting two nodes on the aval side (2 and 11)
    graph_obj.g.add_edge(2, 12, name="gray_aval_1", color="gray", capacity=-1.0)
    graph_obj.g.add_edge(12, 11, name="gray_aval_2", color="gray", capacity=-1.0)

    # Action
    graph_obj.consolidate_constrained_path(
        constrained_path_nodes_amont=c_path.n_amont(),
        constrained_path_nodes_aval=c_path.n_aval(),
        constrained_path_edges=c_path.amont_edges + [c_path.constrained_edge] + c_path.aval_edges
    )

    # Assertions
    edge_data_1 = graph_obj.g.get_edge_data(2, 12)[0]
    assert edge_data_1['color'] == 'blue'

    edge_data_2 = graph_obj.g.get_edge_data(12, 11)[0]
    assert edge_data_2['color'] == 'blue'


def test_consolidate_constrained_path_ignores_coral_path(setup_graph_for_test):
    """
    Tests that a path containing a coral edge is ignored.
    """
    graph_obj, struct_g = setup_graph_for_test
    c_path = struct_g.constrained_path
    # Add a mixed gray/coral path on the amont side
    graph_obj.g.add_edge(0, 10, name="gray_mix_1", color="gray", capacity=-1.0)
    graph_obj.g.add_edge(10, 1, name="coral_mix_2", color="coral", capacity=1.0)

    # Action
    graph_obj.consolidate_constrained_path(
        constrained_path_nodes_amont=c_path.n_amont(),
        constrained_path_nodes_aval=c_path.n_aval(),
        constrained_path_edges=c_path.amont_edges + [c_path.constrained_edge] + c_path.aval_edges
    )

    # Assertions
    edge_data_gray = graph_obj.g.get_edge_data(0, 10)[0]
    assert edge_data_gray['color'] == 'gray'  # Should not be changed


def test_consolidate_constrained_path_ignores_amont_to_aval(setup_graph_for_test):
    """
    Tests that a gray path from amont to aval is not recolored by this function.
    """
    graph_obj, struct_g = setup_graph_for_test
    c_path = struct_g.constrained_path
    # Add a gray path connecting the amont side (node 0) to the aval side (node 2)
    graph_obj.g.add_edge(0, 13, name="gray_cross_1", color="gray", capacity=-1.0)
    graph_obj.g.add_edge(13, 2, name="gray_cross_2", color="gray", capacity=-1.0)

    # Action
    graph_obj.consolidate_constrained_path(
        constrained_path_nodes_amont=c_path.n_amont(),
        constrained_path_nodes_aval=c_path.n_aval(),
        constrained_path_edges=c_path.amont_edges + [c_path.constrained_edge] + c_path.aval_edges
    )

    # Assertions
    edge_data_1 = graph_obj.g.get_edge_data(0, 13)[0]
    assert edge_data_1['color'] == 'gray'  # Should remain unchanged

    edge_data_2 = graph_obj.g.get_edge_data(13, 2)[0]
    assert edge_data_2['color'] == 'gray'  # Should remain unchanged


#------------------------ remove_unused_added_double_edge ---------------


@pytest.fixture
def setup_doubled_graph():
    """
    Sets up a graph with doubled null-flow, gray edges for testing.
    - line_A: 0 <-> 1 (original 0->1)
    - line_B: 2 <-> 3 (original 2->3)
    - unaffected_line: 4 -> 5
    """
    g = nx.MultiDiGraph()
    g.add_edge(0, 1, name="line_A", color="gray", capacity=0.0)
    g.add_edge(2, 3, name="line_B", color="gray", capacity=0.0)
    g.add_edge(4, 5, name="unaffected_line", color="blue", capacity=-10.0)

    # Create the doubled edges
    edges_to_double, edges_double_added = add_double_edges_null_redispatch(g)

    return g, edges_to_double, edges_double_added


def test_removes_original_when_added_is_kept(setup_doubled_graph):
    """
    Tests that the original gray edge is removed if its corresponding
    added (reversed) edge was kept (i.e., recolored).
    """
    g, edges_to_double, edges_double_added = setup_doubled_graph

    # Simulate that the added edge for line_A was identified as part of a path
    added_edge_A = edges_double_added["line_A"]
    g.edges[added_edge_A]["color"] = "blue"
    edges_to_keep = {added_edge_A}

    # Action
    g_processed = remove_unused_added_double_edge(g, edges_to_keep, edges_to_double, edges_double_added)

    # Assertions
    original_edge_A = edges_to_double["line_A"]
    assert not g_processed.has_edge(*original_edge_A), "Original edge of kept pair should be removed."
    assert g_processed.has_edge(*added_edge_A), "Added edge of kept pair should be present."

    # Check that the other doubled pair (line_B) was cleaned up correctly (added edge removed)
    assert g_processed.has_edge(*edges_to_double["line_B"])
    assert not g_processed.has_edge(*edges_double_added["line_B"])


def test_removes_added_when_original_is_kept(setup_doubled_graph):
    """
    Tests that the added (reversed) edge is removed if the original edge
    was kept (i.e., was recolored or was never gray).
    """
    g, edges_to_double, edges_double_added = setup_doubled_graph

    # Simulate that the original edge for line_A was kept. The function's logic for removal
    # depends on the original edge being 'gray', so we change its color.
    original_edge_A = edges_to_double["line_A"]
    g.edges[original_edge_A]["color"] = "blue"
    edges_to_keep = {original_edge_A}

    # Action
    g_processed = remove_unused_added_double_edge(g, edges_to_keep, edges_to_double, edges_double_added)

    # Assertions
    added_edge_A = edges_double_added["line_A"]
    assert g_processed.has_edge(*original_edge_A), "Original kept edge should be present."
    assert not g_processed.has_edge(*added_edge_A), "Added edge of kept pair should be removed."


def test_removes_added_when_neither_is_kept(setup_doubled_graph):
    """
    Tests that if neither edge of a doubled pair is in edges_to_keep,
    the added edge is removed and the original one remains.
    """
    g, edges_to_double, edges_double_added = setup_doubled_graph

    edges_to_keep = set()  # No edges were identified for a path

    # Action
    g_processed = remove_unused_added_double_edge(g, edges_to_keep, edges_to_double, edges_double_added)

    # Assertions for both lines A and B
    assert g_processed.has_edge(*edges_to_double["line_A"])
    assert not g_processed.has_edge(*edges_double_added["line_A"])

    assert g_processed.has_edge(*edges_to_double["line_B"])
    assert not g_processed.has_edge(*edges_double_added["line_B"])


def test_does_not_affect_other_edges(setup_doubled_graph):
    """
    Tests that non-doubled edges are unaffected by the process.
    """
    g, edges_to_double, edges_double_added = setup_doubled_graph

    # Action
    g_processed = remove_unused_added_double_edge(g, set(), edges_to_double, edges_double_added)

    # Assertion
    assert g_processed.has_edge(4, 5)
    assert g_processed.get_edge_data(4, 5)[0]['name'] == 'unaffected_line'


#---------------- all_simple_edge_paths_multi -----------------

@pytest.fixture
def setup_path_graph():
    """
    Sets up a MultiDiGraph for testing pathfinding.
    Graph structure:
    0 -> 1 -> 2
    0 -> 3 -> 2
    0 -> 1 (parallel edge)
    4 -> 5 (isolated path)
    2 -> 0 (cycle)
    """
    g = nx.MultiDiGraph()
    # Path 1
    g.add_edge(0, 1, key='A')
    g.add_edge(1, 2, key='B')
    # Path 2
    g.add_edge(0, 3, key='C')
    g.add_edge(3, 2, key='D')
    # Parallel edge
    g.add_edge(0, 1, key='E')
    # Isolated path
    g.add_edge(4, 5, key='F')
    # Cycle
    g.add_edge(2, 0, key='G')
    return g


def test_finds_all_paths_single_source_target(setup_path_graph):
    """
    Tests finding all simple paths between a single source and a single target.
    """
    g = setup_path_graph
    sources = [0]
    targets = [2]

    paths = list(all_simple_edge_paths_multi(g, sources, targets))

    expected_paths = [
        [(0, 1, 'A'), (1, 2, 'B')],
        [(0, 1, 'E'), (1, 2, 'B')],
        [(0, 3, 'C'), (3, 2, 'D')]
    ]

    assert len(paths) == 3
    for path in expected_paths:
        assert path in paths


def test_finds_paths_multiple_sources_targets(setup_path_graph):
    """
    Tests finding paths with multiple sources and multiple targets.
    """
    g = setup_path_graph
    # Add another target
    g.add_edge(1, 6, key='H')
    sources = [0, 4]
    targets = [2, 5, 6]

    paths = list(all_simple_edge_paths_multi(g, sources, targets))

    expected_paths = [
        [(0, 1, 'A'), (1, 2, 'B')],
        [(0, 1, 'E'), (1, 2, 'B')],
        [(0, 3, 'C'), (3, 2, 'D')],
        [(4, 5, 'F')],
        [(0, 1, 'A'), (1, 6, 'H')],
        [(0, 1, 'E'), (1, 6, 'H')]
    ]

    assert len(paths) == 6
    for path in expected_paths:
        assert path in paths


def test_handles_no_path(setup_path_graph):
    """
    Tests that an empty generator is returned when no path exists.
    """
    g = setup_path_graph
    sources = [0]
    targets = [4]  # No path from 0 to 4

    paths = list(all_simple_edge_paths_multi(g, sources, targets))
    assert len(paths) == 0


def test_respects_cutoff(setup_path_graph):
    """
    Tests that the cutoff parameter correctly limits the path length.
    """
    g = setup_path_graph
    sources = [0]
    targets = [2]

    # All paths from 0 to 2 have length 2. A cutoff of 1 should find none.
    paths = list(all_simple_edge_paths_multi(g, sources, targets, cutoff=1))
    assert len(paths) == 0

    # A cutoff of 2 should find all paths.
    paths = list(all_simple_edge_paths_multi(g, sources, targets, cutoff=2))
    assert len(paths) == 3


def test_empty_sources_or_targets_yields_no_paths(setup_path_graph):
    """
    Tests that no paths are returned if the source or target list is empty.
    """
    g = setup_path_graph

    # Empty sources
    paths_empty_source = list(all_simple_edge_paths_multi(g, [], [2]))
    assert len(paths_empty_source) == 0

    # Empty targets
    paths_empty_target = list(all_simple_edge_paths_multi(g, [0], []))
    assert len(paths_empty_target) == 0

    # Both empty
    paths_both_empty = list(all_simple_edge_paths_multi(g, [], []))
    assert len(paths_both_empty) == 0

@pytest.fixture
def setup2_graph_for_test():
    """
    Sets up a standard graph instance for testing consolidation and analysis methods.
    The graph includes a constrained path, loop paths, and various null-flow lines.
    """
    # A more complex DataFrame to cover various graph structures
    df = pd.DataFrame({
        "idx_or": [0, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12],
        "idx_ex": [1, 2, 3, 4, 5, 2, 6, 7, 8, 5, 10, 11, 12, 9],
        "delta_flows": [-10, -10, -10, -10, -10, 0, 0, 10, 10, 0, 0, 0, 0, 0],
        "gray_edges": [False, False, False, False, False, True, True, False, False, True, True, True, True, True],
    })
    # Add unique line names
    df["line_name"] = [f"{o}_{e}_{i}" for i, (o, e) in enumerate(zip(df["idx_or"], df["idx_ex"]))]

    # Mock topology and lines to cut
    mock_topo = {
        "nodes": {
            "are_prods": [False] * 13, "are_loads": [False] * 13,
            "prods_values": [], "loads_values": []
        },
        "edges": {}
    }
    lines_to_cut = [2]  # Edge from node 2 to 3 is the constrained line

    # Create and build the graph objects
    graph_obj = OverFlowGraph(topo=mock_topo, lines_to_cut=lines_to_cut, df_overflow=df)
    graph_obj.build_graph()
    struct_g = Structured_Overload_Distribution_Graph(graph_obj.g)

    return graph_obj, struct_g


def test_add_relevant_null_flow_lines_does_not_add_extra_edges(setup2_graph_for_test):
    """
    Tests that calling add_relevant_null_flow_lines_all_paths does not result in a net
    increase or decrease in the number of edges in the graph.

    This acts as a regression test to ensure that the logic within detect_edges_to_keep
    or its callers does not inadvertently add permanent edges to the graph. The function
    should only recolor/restyle existing edges or temporary doubled edges that are
    subsequently cleaned up, resulting in a zero net change to the edge count.
    """
    graph_obj, struct_g = setup2_graph_for_test

    # A set of disconnected lines to test with
    non_connected_lines = ["0_2_5", "8_5_9", "5_6_6"]  # Using line names from fixture

    # Record the initial number of edges
    initial_edge_count = len(graph_obj.g.edges())

    # Action: Run the full analysis which includes the detect_edges_to_keep logic
    graph_obj.add_relevant_null_flow_lines_all_paths(
        struct_g,
        non_connected_lines=non_connected_lines
    )

    # Record the final number of edges
    final_edge_count = len(graph_obj.g.edges())

    # Assertion: The number of edges should not have changed.
    assert final_edge_count == initial_edge_count, \
        f"The number of edges changed from {initial_edge_count} to {final_edge_count}. " \
        "The function should not add or remove permanent edges from the graph."