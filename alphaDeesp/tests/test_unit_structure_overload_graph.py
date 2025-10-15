import pytest
import networkx as nx
import pandas as pd
from alphaDeesp.core.graphsAndPaths import Structured_Overload_Distribution_Graph, ConstrainedPath


@pytest.fixture
def setup_structured_graph():
    """
    Sets up a graph and a Structured_Overload_Distribution_Graph instance for testing.
    The graph contains:
    - A constrained path (blue/black): 0 -> 1 -> 2 (amont), 2 -> 3 (constrained), 3 -> 4 -> 5 (aval)
    - A loop path (coral): 1 -> 6 -> 4
    - A disconnected gray path: 7 -> 8
    - Hubs are expected at nodes 1 and 4.
    """
    g = nx.MultiDiGraph()
    # Constrained Path (amont)
    g.add_edge(0, 1, key='A', name="line_0_1", color="blue", capacity=-5.0)
    g.add_edge(1, 2, key='B', name="line_1_2", color="blue", capacity=-10.0)
    # Constrained Edge
    g.add_edge(2, 3, key='C', name="line_2_3", color="black", capacity=-15.0)
    # Constrained Path (aval)
    g.add_edge(3, 4, key='D', name="line_3_4", color="blue", capacity=-10.0)
    g.add_edge(4, 5, key='E', name="line_4_5", color="blue", capacity=-5.0)

    # Loop Path (coral)
    g.add_edge(1, 6, key='F', name="line_1_6", color="coral", capacity=8.0)
    g.add_edge(6, 4, key='G', name="line_6_4", color="coral", capacity=8.0)

    # Isolated Gray edge
    g.add_edge(7, 8, key='H', name="line_7_8", color="gray", capacity=0.0)

    # Add some nodes that might otherwise be isolated
    g.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8])

    return Structured_Overload_Distribution_Graph(g)


@pytest.fixture
def setup_graph_with_extra_blue_path(setup_structured_graph):
    """
    Adds an extra, disconnected blue path to the standard graph.
    """
    g = setup_structured_graph.g_init.copy()
    g.add_edge(9, 10, key='I', name="line_9_10", color="blue", capacity=-2.0)
    return Structured_Overload_Distribution_Graph(g)


@pytest.fixture
def setup_graph_no_loop(setup_structured_graph):
    """
    Removes the loop path from the standard graph.
    """
    g = setup_structured_graph.g_init.copy()
    g.remove_edge(1, 6, key='F')
    g.remove_edge(6, 4, key='G')
    return Structured_Overload_Distribution_Graph(g)


@pytest.fixture
def setup_graph_multiple_loops(setup_structured_graph):
    """
    Adds a second loop path to the standard graph.
    """
    g = setup_structured_graph.g_init.copy()
    g.add_edge(2, 7, key='J', name="line_2_7", color="coral", capacity=3.0)
    g.add_edge(7, 5, key='K', name="line_7_5", color="coral", capacity=3.0)
    return Structured_Overload_Distribution_Graph(g)


def test_find_constrained_path(setup_structured_graph):
    """
    Tests that the constrained path is correctly identified.
    """
    structured_graph = setup_structured_graph
    cp = structured_graph.constrained_path

    assert isinstance(cp, ConstrainedPath)
    assert cp.constrained_edge == (2, 3, 'C')
    assert set(cp.amont_edges) == set([(0, 1, 'A'), (1, 2, 'B')])
    assert set(cp.aval_edges) == set([(3, 4, 'D'), (4, 5, 'E')])


def test_find_hubs(setup_structured_graph):
    """
    Tests that hubs (nodes connecting constrained and loop paths) are found.
    """
    structured_graph = setup_structured_graph
    hubs = structured_graph.hubs

    # Hubs should be node 1 (amont side) and node 4 (aval side)
    assert sorted(hubs) == [1, 4]


def test_find_loops(setup_structured_graph):
    """
    Tests that loop paths (coral paths between hubs) are correctly identified.
    """
    structured_graph = setup_structured_graph
    loops_df = structured_graph.red_loops

    assert not loops_df.empty
    assert len(loops_df) == 1

    loop = loops_df.iloc[0]
    assert loop['Source'] == 1
    assert loop['Target'] == 4
    assert loop['Path'] == [1, 6, 4]


def test_get_constrained_edges_and_nodes(setup_structured_graph):
    """
    Tests the retrieval of all edges and nodes belonging to the constrained path.
    """
    structured_graph = setup_structured_graph
    edges, nodes, other_blue_edges, other_blue_nodes = structured_graph.get_constrained_edges_nodes()

    expected_edges = ["line_0_1", "line_1_2", "line_2_3", "line_3_4", "line_4_5"]
    expected_nodes = [0, 1, 2, 3, 4, 5]

    assert sorted(edges) == sorted(expected_edges)
    assert sorted(nodes) == sorted(expected_nodes)

    # Now that the source function is fixed, we expect no other blue edges
    assert not other_blue_edges
    assert not other_blue_nodes


def test_get_constrained_edges_with_other_blue_path(setup_graph_with_extra_blue_path):
    """
    Tests that other_blue_edges correctly identifies blue edges not on the main constrained path.
    """
    structured_graph = setup_graph_with_extra_blue_path
    _, _, other_blue_edges, other_blue_nodes = structured_graph.get_constrained_edges_nodes()

    # The function should now find the extra blue path
    assert (9, 10) in other_blue_edges
    assert set(other_blue_nodes) == {9, 10}


def test_graph_with_no_loop_path(setup_graph_no_loop):
    """
    Tests behavior when the graph has no coral/red loop paths.
    """
    structured_graph = setup_graph_no_loop

    # find_loops should return an empty DataFrame
    assert structured_graph.red_loops.empty

    # find_hubs should return an empty list
    assert not structured_graph.hubs

    # get_dispatch_edges_nodes should return empty lists
    lines, nodes = structured_graph.get_dispatch_edges_nodes()
    assert not lines
    assert not nodes


def test_find_multiple_loops(setup_graph_multiple_loops):
    """
    Tests that multiple distinct loop paths are identified correctly.
    """
    structured_graph = setup_graph_multiple_loops
    loops_df = structured_graph.red_loops

    assert len(loops_df) == 2
    paths = loops_df['Path'].tolist()
    assert [1, 6, 4] in paths
    assert [2, 7, 5] in paths


def test_get_dispatch_edges_and_nodes(setup_structured_graph):
    """
    Tests the retrieval of all edges and nodes belonging to the dispatch (loop) path.
    """
    structured_graph = setup_structured_graph
    lines, nodes = structured_graph.get_dispatch_edges_nodes()

    expected_lines = ["line_1_6", "line_6_4"]
    expected_nodes = [1, 4, 6]

    assert sorted(lines) == sorted(expected_lines)
    assert sorted(nodes) == sorted(expected_nodes)


def test_subgraph_creation(setup_structured_graph):
    """
    Tests that the internal subgraphs are created correctly.
    """
    sg = setup_structured_graph

    # g_only_blue_components should contain the constrained path
    assert set(sg.g_only_blue_components.nodes()) == {0, 1, 2, 3, 4, 5}
    assert len(sg.g_only_blue_components.edges()) == 5

    # g_only_red_components should contain the loop path
    assert set(sg.g_only_red_components.nodes()) == {1, 4, 6}
    assert len(sg.g_only_red_components.edges()) == 2

    # g_without_gray_and_c_edge should contain both blue and red paths
    assert set(sg.g_without_gray_and_c_edge.nodes()) == {0, 1, 2, 3, 4, 5, 6}
    # This subgraph has black (constrained) and gray edges removed.
    assert len(sg.g_without_gray_and_c_edge.edges()) == 6  # 4 blue + 2 red

