# import sys
# import os

# sys.path.append(os.path.abspath("../../alphaDeesp.core"))
import networkx as nx
from alphaDeesp.core.graphsAndPaths import ConstrainedPath,Structured_Overload_Distribution_Graph,OverFlowGraph, delete_color_edges
from alphaDeesp.core.alphadeesp import *
import configparser
import numpy as np
import os
import json
from alphaDeesp.core.grid2op.Grid2opObservationLoader import Grid2opObservationLoader
from alphaDeesp.core.grid2op.Grid2opSimulation import Grid2opSimulation

# from ..core.constrainedPath import ConstrainedPath

"""Here the file with all the tests
The ideas should be listed here:
- when creating loops paths, we might want to test loop paths, that at every possible split, we gather all the red paths
- each split is a new path

# TEST PART TO MOVE SOMEWHERE
# c_path = ConstrainedPath(e_amont, constrained_edge, e_aval)
# print("c_path = ", c_path)
# print("n_amont = ", c_path.n_amont())
# print("n_aval = ", c_path.n_aval())
# print("e_amont = ", c_path.e_amont())
# print("e_aval = ", c_path.e_aval())
# print("full_e_constrained_path = ", c_path.full_e_constrained_path())
# print("full_n_constrained_path = ", c_path.full_n_constrained_path())


check if nx.min_cut function can cut 2 edges for min cut.
## Create new graph and let it cut
"""

# Three different types of graph for tests
# ============ Graph g1 ============
#        5
#         \>
# 1->-2->-3->-4
#      \>
#      6
g1 = nx.DiGraph()
g1.add_edges_from([(1,2), (2, 3), (2, 6), (3, 4), (5, 3)])

# ============ Graph g2 ============
#           4->-5
#          />   \>
# 1->-2->-3      6->-7->-8
#          \>   />
#          9->-10
g2 = nx.MultiDiGraph()
list_edges_g2=[(1,2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
                   (3, 9), (9, 10), (10, 6)]
g2.add_edges_from(list_edges_g2)

g3 = nx.DiGraph()
g3.add_edges_from([(1,2), (2, 3), (2, 6), (3, 4), (5, 3)])


def test_constrained_path():
    """This function tests the Class ConstrainedPath"""

    c_path = ConstrainedPath([], (5, 6), [(6, 13)])
    assert c_path.n_amont() == [5]
    assert c_path.n_aval() == [6, 13]
    assert c_path.full_n_constrained_path() == [5, 6, 13]

def test_structured_overload_distribution_graph():
    "Testing Structured overload Graph that dispalys a constrained path and a loop path"

    #expected structure
    nodes_c_path=[1,2,3,4,5,6,7,8]
    loop=[3,9,10,6]
    hubs=[3,6]

    #coloring the graph to be processed after to identify paths structure
    for u, v, idx,color in g2.edges(data="color", keys=True):
        if u==4 and v==5:
            g2[u][v][idx]["color"] ="black" #contrained edge
        elif u in nodes_c_path and v in nodes_c_path:
            g2[u][v][idx]["color"]="blue"
        else:
            g2[u][v][idx]["color"] = "coral"

    Overload_graph=Structured_Overload_Distribution_Graph(g2)

    assert set(Overload_graph.get_hubs())==set(hubs)
    assert set(Overload_graph.get_constrained_path().full_n_constrained_path())==set(nodes_c_path)


    loops_df=Overload_graph.get_loops()
    assert loops_df.loc[0]["Path"]==loop
    assert loops_df.loc[0]["Source"]==3
    assert loops_df.loc[0]["Target"] == 6

def test_lines_swapped():
    config = configparser.ConfigParser()
    config.read("./alphaDeesp/tests/resources_for_tests_grid2op/config_for_tests.ini")

    data_folder="./alphaDeesp/tests/ressources_for_tests/data_graph_consolidation/defaut_P.SAOL31RONCI"

    timestep = 1  # 1#36
    line_defaut = "P.SAOL31RONCI"
    ltc = [9]

    expected_lines_swapped=['C.FOUL31MERVA','LOUHAL31SSUSU','MERVAL31SSUSU','C.FOUL31NAVIL']

    with open(os.path.join(data_folder,'sim_topo_zone_dijon_defaut_P.SAOL31RONCI_t1.json')) as json_file:
        sim_topo_reduced = json.load(json_file)

    df_of_g = pd.read_csv(os.path.join(data_folder,"df_of_g_defaut_P.SAOL31RONCI_t1.csv"))

    lines_swapped=list(df_of_g[df_of_g.new_flows_swapped].line_name)
    assert(set(expected_lines_swapped).intersection(set(lines_swapped))==set(expected_lines_swapped))

def test_consolidate_constrained_path():
    config = configparser.ConfigParser()
    config.read("./alphaDeesp/tests/resources_for_tests_grid2op/config_for_tests.ini")

    timestep = 1  # 1#36
    line_defaut = "P.SAOL31RONCI"
    float_precision_graph = "%.0f"

    g_over, situation_info = basic_test_configuration(line_defaut, timestep, float_precision_graph)

    #consolidate
    g_distribution_graph = Structured_Overload_Distribution_Graph(g_over.g)

    #compute initial blue edges
    g_without_pos_edges = delete_color_edges(g_over.g, "coral")
    g_with_only_blue_edges_init=delete_color_edges(g_without_pos_edges, "gray")
    n_blue_edges_init=len(g_with_only_blue_edges_init.edges)

    # consolider le chemin en contrainte avec la connaissance des hubs, en itérant une fois de plus
    n_hubs_init = 0
    hubs_paths = g_distribution_graph.find_loops()[["Source", "Target"]].drop_duplicates()
    n_hub_paths = hubs_paths.shape[0]

    #while n_hubs_init != n_hub_paths:
    n_hubs_init = n_hub_paths

    constrained_path=g_distribution_graph.constrained_path
    nodes_amont=constrained_path.n_amont()
    nodes_aval=constrained_path.n_aval()
    constrained_path_edges=constrained_path.aval_edges+[constrained_path.constrained_edge]+constrained_path.amont_edges
    g_over.consolidate_constrained_path(nodes_amont,nodes_aval,constrained_path_edges,ignore_null_edges=False)#(constrained_path_nodes)#hubs_paths.Source, hubs_paths.Target)
    #g_distribution_graph = Structured_Overload_Distribution_Graph(g_over.g)
#
    #hubs_paths = g_distribution_graph.find_loops()[["Source", "Target"]].drop_duplicates()
    #n_hub_paths = hubs_paths.shape[0]

    #le nombre de loop path est passe de 1 a 3
    #assert(n_hub_paths==3)
    #count number of changes
    g_without_pos_edges = delete_color_edges(g_over.g, "coral")
    g_with_only_blue_edges_final=delete_color_edges(g_without_pos_edges, "gray")
    n_blue_edges_final=len(g_with_only_blue_edges_final.edges)

    assert(n_blue_edges_final>n_blue_edges_init)

    #7 edge on change de couleur
    edges_blue_name_initial = list(nx.get_edge_attributes(g_with_only_blue_edges_init, "name").values())
    edges_blue_name_final = list(nx.get_edge_attributes(g_with_only_blue_edges_final, "name").values())

    changes_seen = set(edges_blue_name_final) - set(edges_blue_name_initial)

    edge_names_changed_null_flow = set(['CPVANY632', 'GROSNL71VIELM'])
    assert (edge_names_changed_null_flow.intersection(changes_seen) == edge_names_changed_null_flow)

    #edge_name_changed_positive_capacity = set(['CPVANL31RIBAU', 'H.PAUL71VIELM', 'C.SAUL31MAGNY', 'H.PAUL71SSV.O'])
    #assert (edge_name_changed_positive_capacity.intersection(changes_seen) == edge_name_changed_positive_capacity)
#
    #negative_non_significant_capacity = set(['C.REGY631', 'C.REGY633'])
    #assert (negative_non_significant_capacity.intersection(changes_seen) == negative_non_significant_capacity)


def test_reverse_blue_edges_in_looppaths():
    config = configparser.ConfigParser()
    config.read("./alphaDeesp/tests/resources_for_tests_grid2op/config_for_tests.ini")

    timestep = 36  # 1#36
    line_defaut = "FRON5L31LOUHA"
    float_precision_graph = "%.0f"

    g_over, situation_info = basic_test_configuration(line_defaut, timestep, float_precision_graph)

    #compute initial blue edges
    g_without_pos_edges = delete_color_edges(g_over.g, "coral")
    g_with_only_blue_edges=delete_color_edges(g_without_pos_edges, "gray")
    n_blue_edges_init=len(g_with_only_blue_edges.edges)

    #consolidate
    g_distribution_graph = Structured_Overload_Distribution_Graph(g_over.g)
    constrained_path = g_distribution_graph.constrained_path.full_n_constrained_path()
    g_over.reverse_blue_edges_in_looppaths(constrained_path)

    #count number of changes
    g_without_pos_edges = delete_color_edges(g_over.g, "coral")
    g_with_only_blue_edges=delete_color_edges(g_without_pos_edges, "gray")
    n_blue_edges_final=len(g_with_only_blue_edges.edges)

    #7 edge on change de couleur
    assert(n_blue_edges_init-n_blue_edges_final)

def test_consolidate_loop_path():
    config = configparser.ConfigParser()
    config.read("./alphaDeesp/tests/resources_for_tests_grid2op/config_for_tests.ini")

    timestep = 36  # 1#36
    line_defaut = "FRON5L31LOUHA"
    float_precision_graph = "%.0f"

    g_over, situation_info = basic_test_configuration(line_defaut, timestep, float_precision_graph)

    #reverse blue edges
    g_distribution_graph = Structured_Overload_Distribution_Graph(g_over.g)
    constrained_path = g_distribution_graph.constrained_path.full_n_constrained_path()
    g_over.reverse_blue_edges_in_looppaths(constrained_path)

    #compute initial red edges
    g_without_blue_edges = delete_color_edges(g_over.g, "blue")
    g_with_only_red_edges_init=delete_color_edges(g_without_blue_edges, "gray")
    n_red_edges_init=len(g_with_only_red_edges_init.edges)

    #consolidate loop paths
    hubs_paths = g_distribution_graph.find_loops()[["Source", "Target"]].drop_duplicates()
    g_over.consolidate_loop_path(hubs_paths.Source, hubs_paths.Target,ignore_null_edges=False)

    #compute final red edges
    g_without_blue_edges = delete_color_edges(g_over.g, "blue")
    g_with_only_red_edges_final=delete_color_edges(g_without_blue_edges, "gray")
    n_red_edges_final=len(g_with_only_red_edges_final.edges)

    #7 edge on change de couleur
    edges_red_name_initial = list(nx.get_edge_attributes(g_with_only_red_edges_init, "name").values())
    edges_red_name_final = list(nx.get_edge_attributes(g_with_only_red_edges_final, "name").values())

    changes_seen = set(edges_red_name_final) - set(edges_red_name_initial)

    #if edges are ignored when launching loop paths, those ones should not appear
    edge_names_expected_change_null_flow = set(['CHALOL31LOUHA', 'CHALOY631', 'CHALOY632', 'CHALOY633','GROSNL71VIELM','CPVANY632'])
    assert (changes_seen.intersection(edge_names_expected_change_null_flow)==edge_names_expected_change_null_flow)

    #those should always appear
    other_edge_names_expected_to_change=set(['CPVANL61ZMAGN','GROSNL61ZCUR5','H.PAUL61ZCUR5','GEN.PL73VIELM'])#,
    assert (changes_seen.intersection(other_edge_names_expected_to_change)==other_edge_names_expected_to_change)

def test_consolidate_graph():
    config = configparser.ConfigParser()
    config.read("./alphaDeesp/tests/resources_for_tests_grid2op/config_for_tests.ini")

    timestep = 36  # 1#36
    line_defaut = "FRON5L31LOUHA"
    float_precision_graph = "%.0f"

    g_over, situation_info = basic_test_configuration(line_defaut, timestep, float_precision_graph)
    non_connected_reconnectable_lines = situation_info["non_connected_reconnectable_lines"]
    lines_non_reconnectable = situation_info["lines_non_reconnectable"]

    #compute initial red edges
    g_without_blue_edges = delete_color_edges(g_over.g, "blue")
    g_with_only_red_edges_init=delete_color_edges(g_without_blue_edges, "gray")
    n_red_edges_init=len(g_with_only_red_edges_init.edges)

    g_distribution_graph = Structured_Overload_Distribution_Graph(g_over.g)
    g_over.consolidate_graph(g_distribution_graph,non_connected_lines_to_ignore=non_connected_reconnectable_lines+non_connected_reconnectable_lines)
    #g_over.consolidate_graph(g_distribution_graph)

    #compute final red edges
    g_without_blue_edges = delete_color_edges(g_over.g, "blue")
    g_with_only_red_edges_final=delete_color_edges(g_without_blue_edges, "gray")
    n_red_edges_final=len(g_with_only_red_edges_final.edges)

    edges_red_name_initial = list(nx.get_edge_attributes(g_with_only_red_edges_init, "name").values())
    edges_red_name_final = list(nx.get_edge_attributes(g_with_only_red_edges_final, "name").values())

    #7 edge on change de couleur
    edge_names_expected_change=['GEN.PL61IZERN','CIZE L61IZERN','CIZE L61FLEYR','CREYSL71SSV.O','CREYSL72SSV.O','CREYSL72GEN.P','CREYSL71GEN.P']
    n_targeted_change = len(edge_names_expected_change)
    changes_seen=set(edges_red_name_final)-set(edges_red_name_initial)
    targeted_changes_seen=changes_seen.intersection(set(edge_names_expected_change))
    n_targeted_change_seen = len(targeted_changes_seen)

    edge_names_expected_change_not_change=set(['CHALOL31LOUHA','CHALOY631','CHALOY632','CHALOY633'])
    assert(len(changes_seen.intersection(edge_names_expected_change_not_change))==0)

    assert(n_targeted_change==n_targeted_change_seen)#explicit test to comply with
    #assert(n_red_edges_final-n_red_edges_init==28)#to check if anything changed, but not very explicit test and value can change if good reason

def test_identify_ambiguous_paths_and_type():
    config = configparser.ConfigParser()
    config.read("./alphaDeesp/tests/resources_for_tests_grid2op/config_for_tests.ini")

    timestep = 1  # 1#36
    line_defaut = "P.SAOL31RONCI"
    float_precision_graph = "%.0f"

    g_over, situation_info = basic_test_configuration(line_defaut, timestep, float_precision_graph)

    g_distribution_graph = Structured_Overload_Distribution_Graph(g_over.g)

    ambiguous_edge_paths, ambiguous_node_paths = g_over.identify_ambiguous_paths(g_distribution_graph)

    expected_ambiguous_edge_path_1=['MAGNYY633','C.SAUL31MAGNY','GENLIL31MAGNY','MAGNYL61ZMAGN','AUXONL31RIBAU',
                                  'CPVANL31RIBAU','C.REGL31ZCRIM','COLLOL31GENLI','C.SAUL31ZCRIM','AUXONL31COLLO']
    #{('COLLOP3', 'AUXONP3', 0): 'blue',
    #('ZCRIMP3', 'C.SAUP3', 0): 'blue',
    #('MAGNYP3', 'C.SAUP3', 0): 'coral',
    #('MAGNYP3', 'GENLIP3', 0): 'blue',
    #('CPVANP3', 'RIBAUP3', 0): 'coral',
    #('GENLIP3', 'COLLOP3', 0): 'blue',
    #('ZMAGNP6', 'MAGNYP6', 0): 'blue',
    #('MAGNYP6', 'MAGNYP3', 0): 'blue',
    #('C.REGP3', 'ZCRIMP3', 0): 'blue',
    #('AUXONP3', 'RIBAUP3', 0): 'blue'}
    assert(set(ambiguous_edge_paths[1])==set(expected_ambiguous_edge_path_1))

    path_type=g_over.desambiguation_type_path(ambiguous_node_paths[1], g_distribution_graph)
    assert(path_type=="constrained_path")

    expected_ambiguous_edge_path_2=['COMMUL61VIELM', 'COMMUL61H.PAU', 'H.PAUY762', 'H.PAUY772']
    #{('2H.PAP7', 'H.PAUP6', 0): 'blue',
    #('H.PAUP7', '2H.PAP7', 0): 'blue',
    #('VIELMP6', 'COMMUP6', 0): 'coral',
    #('COMMUP6', 'H.PAUP6', 0): 'coral'}
    assert (set(ambiguous_edge_paths[0]) == set(expected_ambiguous_edge_path_2))
    #as we only have coral values, so no need to change
    non_expected_path=['BOCTOL71M.SEI','BOCTOL72M.SEI','BOCTOL71N.SE5','M.SEIL71VIELM','M.SEIL72VIELM','N.SE5Y711']
    for edge_path in ambiguous_edge_paths:
        assert(set(non_expected_path)!=set(edge_path))
    #{('BOCTOP7', 'M.SEIP7', 0): 'coral',
    # ('BOCTOP7', 'M.SEIP7', 1): 'coral',
    # ('N.SE1P7', 'BOCTOP7', 0): 'coral',
    # ('M.SEIP7', 'VIELMP7', 0): 'coral',
    # ('M.SEIP7', 'VIELMP7', 1): 'coral',
    # ('N.SE1P1', 'N.SE1P7', 0): 'coral'}

def test_identify_ambiguous_paths_and_type_2():
    config = configparser.ConfigParser()
    config.read("./alphaDeesp/tests/resources_for_tests_grid2op/config_for_tests.ini")

    timestep = 36  # 1#36
    line_defaut = "FRON5L31LOUHA"
    float_precision_graph = "%.0f"

    g_over, situation_info = basic_test_configuration(line_defaut, timestep, float_precision_graph)

    g_distribution_graph = Structured_Overload_Distribution_Graph(g_over.g)

    expected_ambiguous_node_path=['CIZE P6', 'FLEYRP6', 'GEN.PP6', 'IZERNP6', 'VOUGLP6']
    ambiguous_edge_paths, ambiguous_node_paths = g_over.identify_ambiguous_paths(g_distribution_graph)

    #
    #{('IZERNP6', 'CIZE P6', 0): 'blue',
    # ('VOUGLP6', 'FLEYRP6', 0): 'coral',
    # ('GEN.PP6', 'IZERNP6', 0): 'blue',
    # ('CIZE P6', 'FLEYRP6', 0): 'blue'}
    is_expected_path_there=False
    path_type=None

    for path in ambiguous_node_paths:
        if set(path)==set(expected_ambiguous_node_path):
            is_expected_path_there=True
            path_type = g_over.desambiguation_type_path(expected_ambiguous_node_path, g_distribution_graph)
    assert(is_expected_path_there)

    assert(path_type=="loop_path")

def test_add_relevant_null_flow_lines():
    #vérifier le path GROSNP6, ZJOUXP6, BOISSP6, GEN.PP6 avec BOISSP6 seeulement en pointillé
    #Path CHALOP3 -> LOUHAP3
    #Path CPVANP6 PYMONP6 VOUGLP6 PYMONP3

    config = configparser.ConfigParser()
    config.read("./alphaDeesp/tests/resources_for_tests_grid2op/config_for_tests.ini")

    data_folder = "./alphaDeesp/tests/ressources_for_tests/data_graph_consolidation/defaut_PSAOL31RONCI"

    timestep = 1  # 1#36
    line_defaut = "P.SAOL31RONCI"
    float_precision_graph = "%.0f"

    g_over, g_distribution_graph, non_connected_reconnectable_lines, lines_non_reconnectable = null_flow_test_configuration(
        line_defaut, timestep, float_precision_graph)

    g_over.add_relevant_null_flow_lines_all_paths(g_distribution_graph,
                                                  non_connected_lines=non_connected_reconnectable_lines,
                                                  non_reconnectable_lines=lines_non_reconnectable)
    #g_over.add_relevant_null_flow_lines_all_paths(g_distribution_graph, non_connected_lines)

    color_edges = list(nx.get_edge_attributes(g_over.g, 'color').values())
    line_names = list(nx.get_edge_attributes(g_over.g, 'name').values())

    line_tests=['BOISSL61GEN.P', 'CHALOL31LOUHA','GEN.PY762' ]#'PYMONL61VOUGL' and 'PYMONY632' 'CPVANY632' 'GEN.PY762', are  dimgray

    significant_colors=set(["blue","coral"])#have we highlighted those lines on significant paths ?
    for line in line_tests:#
        assert(len(set([color for color, line_name in zip(color_edges, line_names) if line_name == line]).intersection(significant_colors))>=1)

    line_tests=['PYMONL61VOUGL','PYMONY632','CPVANY632']#, are  dimgray
    for line in line_tests:
        #print(line)
        #color = set([color for color, line_name in zip(color_edges, line_names) if line_name == line])
        #print(color)
        assert ("dimgray" in set([color for color, line_name in zip(color_edges, line_names) if line_name == line]))

    #g_distribution_graph = Structured_Overload_Distribution_Graph(g_over.g)
    hubs=g_distribution_graph.get_hubs()

    new_hubs_to_test=["CPVANP6","CHALOP6","GROSNP6"]
    n_new_hubs=len(new_hubs_to_test)

    assert(n_new_hubs==len(set(new_hubs_to_test)-set(hubs)))

def test_add_relevant_null_flow_lines_non_reconnectables_lines():
    #vérifier le path GROSNP6, ZJOUXP6, BOISSP6, GEN.PP6 avec BOISSP6 seeulement en pointillé
    #Path CHALOP3 -> LOUHAP3
    #Path CPVANP6 PYMONP6 VOUGLP6 PYMONP3

    config = configparser.ConfigParser()
    config.read("./alphaDeesp/tests/resources_for_tests_grid2op/config_for_tests.ini")

    data_folder = "./alphaDeesp/tests/ressources_for_tests/data_graph_consolidation/defaut_PSAOL31RONCI"

    timestep = 1  # 1#36
    line_defaut = "P.SAOL31RONCI"
    float_precision_graph = "%.0f"

    g_over, g_distribution_graph, non_connected_reconnectable_lines, lines_non_reconnectable = null_flow_test_configuration(
        line_defaut, timestep, float_precision_graph)

    g_over.add_relevant_null_flow_lines_all_paths(g_distribution_graph, non_connected_lines=non_connected_reconnectable_lines,
                                        non_reconnectable_lines=lines_non_reconnectable)

    #for i in range(2):#need two iterations to identify CHALOL31LOUHA reconnectable path under contingency "BEON L31CPVAN" at timestep 1 on chronic 28th august
    #    g_over.add_relevant_null_flow_lines_all_paths(g_distribution_graph, non_connected_lines,non_reconnectable_lines)
    #    g_distribution_graph = Structured_Overload_Distribution_Graph(g_over.g)
    #g_over.add_relevant_null_flow_lines_all_paths(g_distribution_graph, non_connected_lines)

    color_edges = nx.get_edge_attributes(g_over.g, 'color')
    style_edges = nx.get_edge_attributes(g_over.g, 'style')
    dir_edges = nx.get_edge_attributes(g_over.g, 'dir')
    line_names = nx.get_edge_attributes(g_over.g, 'name')

    line_tests=['GEN.PL73VIELM', 'PYMONL61VOUGL', 'CPVANY632', 'PYMONY632']#'CRENEL71VIELM',

    #color dimgray and style dotted
    for line in line_tests:
        line_edge=[edge for edge, line_name in line_names.items() if line_name == line][0]
        assert("gray" in color_edges[line_edge])
        assert (style_edges[line_edge] == "dotted")
        assert (dir_edges[line_edge] == "none")

    line_on_non_reconnectable_path='CPVANL61PYMON'
    line_edge = [edge for edge, line_name in line_names.items() if line_name == line_on_non_reconnectable_path][0]
    assert(color_edges[line_edge] == "dimgray")
    assert ((line_edge not in style_edges ) or (style_edges[line_edge] == "solid"))


def test_add_relevant_null_flow_lines_blue_path():
    config = configparser.ConfigParser()
    config.read("./alphaDeesp/tests/resources_for_tests_grid2op/config_for_tests.ini")

    data_folder="./alphaDeesp/tests/ressources_for_tests/data_graph_consolidation/defaut_PSAOL31RONCI"

    timestep = 1  # 1#36
    line_defaut = "P.SAOL31RONCI"
    float_precision_graph = "%.0f"

    g_over, g_distribution_graph, non_connected_reconnectable_lines, lines_non_reconnectable = null_flow_test_configuration(
        line_defaut, timestep, float_precision_graph)


    #check that this edge has been coloured blue
    #find edge non reconnectable "CPVANY632"
    edge_names_dict = nx.get_edge_attributes(g_over.g, 'name')
    line_name_target="CPVANY632"
    edge_target=[edge for edge,name in edge_names_dict.items() if name==line_name_target][0]
    assert(g_over.g.edges[edge_target]["color"]=="gray")
    g_over.add_relevant_null_flow_lines(g_distribution_graph, non_connected_lines=non_connected_reconnectable_lines,non_reconnectable_lines=lines_non_reconnectable, target_path="blue_only")
    assert ("gray" in g_over.g.edges[edge_target]["color"])#not coloured because non reconnectable, but dispalyed as dimgray
    # check that this double edge has been removed (in the other edge direction)
    assert(not g_over.g.has_edge('CPVANP3', 'CPVANP6'))

    #making CPVANY632 connectable now
    g_over, g_distribution_graph, non_connected_reconnectable_lines, lines_non_reconnectable = null_flow_test_configuration(
        line_defaut, timestep, float_precision_graph,line_to_reconnect="CPVANY632")

    g_over.add_relevant_null_flow_lines(g_distribution_graph, non_connected_lines=non_connected_reconnectable_lines,
                                        non_reconnectable_lines=lines_non_reconnectable, target_path="blue_only")

    edge_names_dict = nx.get_edge_attributes(g_over.g, 'name')
    edge_target = [edge for edge, name in edge_names_dict.items() if name == line_name_target][0]
    assert (g_over.g.edges[edge_target]["color"] == "blue")

def basic_test_configuration(line_defaut,timestep,float_precision_graph):

    data_folder = "./alphaDeesp/tests/ressources_for_tests/data_graph_consolidation/defaut_" + line_defaut  # CPVANL61ZMAGN"

    suffix_file_names = "defaut_" + line_defaut + "_t" + str(timestep)

    sim_topo_file_name = "sim_topo_zone_dijon_" + suffix_file_names + ".json"
    with open(os.path.join(data_folder, sim_topo_file_name)) as json_file:
        sim_topo_reduced = json.load(json_file)

    with open(os.path.join(data_folder, 'situation_info.json')) as json_file:
        situation_info = json.load(json_file)

    ltc = situation_info["ltc"]

    mapping = situation_info["node_name_mapping"]
    number_nodal_dict = situation_info["number_nodal_dict"]

    ##############
    # rebuild overflow graphs and attributes
    df_of_g = pd.read_csv(os.path.join(data_folder, "df_of_g_" + suffix_file_names + ".csv"))

    g_over = OverFlowGraph(sim_topo_reduced, ltc, df_of_g, float_precision=float_precision_graph)

    mapping = {int(key): value for key, value in mapping.items()}
    g_over.rename_nodes(mapping)  # g = nx.relabel_nodes(g_over.g, mapping, copy=True)

    voltage_levels_dict = situation_info["voltage_levels"]
    g_over.set_voltage_level_color(voltage_levels_dict)

    g_over.set_electrical_node_number(number_nodal_dict)

    return g_over,situation_info

def null_flow_test_configuration(line_defaut,timestep,float_precision_graph,line_to_reconnect=None):

    g_over, situation_info = basic_test_configuration(line_defaut,timestep,float_precision_graph)
    # consolidate
    g_distribution_graph = Structured_Overload_Distribution_Graph(g_over.g)

    non_connected_reconnectable_lines = situation_info["non_connected_reconnectable_lines"]
    lines_non_reconnectable = situation_info["lines_non_reconnectable"]

    if line_to_reconnect:
        non_connected_reconnectable_lines.append(line_to_reconnect)
        lines_non_reconnectable=[line for line in lines_non_reconnectable if line != line_to_reconnect]

    non_connected_lines = non_connected_reconnectable_lines + lines_non_reconnectable

    return g_over,g_distribution_graph,non_connected_reconnectable_lines,lines_non_reconnectable

def test_add_relevant_null_flow_lines_red_path_1():
    config = configparser.ConfigParser()
    config.read("./alphaDeesp/tests/resources_for_tests_grid2op/config_for_tests.ini")
    line_defaut = "CHALOL61CPVAN"
    timestep = 9  # 1#36
    float_precision_graph="%.0f"

    g_over,g_distribution_graph,non_connected_reconnectable_lines,lines_non_reconnectable=null_flow_test_configuration(line_defaut, timestep, float_precision_graph)

    ##############
    # function to test on this case
    g_over.add_relevant_null_flow_lines(g_distribution_graph, non_connected_lines=non_connected_reconnectable_lines,non_reconnectable_lines=lines_non_reconnectable, target_path="red_only")

    ##############
    # tests
    all_edges=g_over.g.edges

    edge_1=('GEN.PP6','BOISSP6', 0)
    if edge_1 not in all_edges:
        edge_1=(edge_1[1],edge_1[0],edge_1[2])
    edge_1_attributes=all_edges[edge_1]
    assert(edge_1_attributes["color"]=="coral" and edge_1_attributes["style"]=="dashed" and edge_1_attributes["dir"]=="none")
    #path
    edges_path=[('BOISSP6', 'ZJOUXP6', 0),('MACONP6', 'ZJOUXP6', 0),('GROSNP6', 'MACONP6', 0),('GROSNP6', 'CHALOP6', 0),('CPVANP6', 'CHALOP6', 0)]
    for edge in edges_path:
        if edge not in all_edges:
            edge = (edge[1], edge[0], edge[2])
        edge_attributes = all_edges[edge]
        assert (edge_attributes["color"] == "coral")

    edge2=('GENLIP3', 'MAGNYP3', 0)#this one remains gray, because there is a non reconnectable line on the path
    if edge2 not in all_edges:
        edge2=(edge2[1],edge2[0],edge2[2])
    edge2_attributes=all_edges[edge2]
    assert ("gray" in edge2_attributes["color"]  and edge2_attributes["style"] == "dashed")

    edge2_prime=('GENLIP3', 'COLLOP3', 0)#this one remains gray, because this is the non reconnectable line on the path
    if edge2_prime not in all_edges:
        edge2_prime=(edge2_prime[1],edge2_prime[0],edge2_prime[2])
    edge2_attributes_prime=all_edges[edge2_prime]
    assert ("gray" in edge2_attributes_prime["color"]  and edge2_attributes_prime["style"] == "dotted")

    edge3 = ('CHALOP3', 'LOUHAP3', 0)  # this one remains gray, because there the paths it connects too is not sensitive enough to be ighlighted on the constrained path
    if edge3 not in all_edges:
        edge3 = (edge3[1], edge3[0], edge3[2])
    edge3_attributes = all_edges[edge3]
    assert ("gray" in edge3_attributes["color"] and edge3_attributes["style"] == "dashed")

def test_add_relevant_null_flow_lines_red_path_2():
    config = configparser.ConfigParser()
    config.read("./alphaDeesp/tests/resources_for_tests_grid2op/config_for_tests.ini")
    line_defaut = "CPVANL61ZMAGN"
    timestep = 1  # 1#36
    float_precision_graph="%.0f"

    g_over,g_distribution_graph,non_connected_reconnectable_lines,lines_non_reconnectable=null_flow_test_configuration(line_defaut, timestep, float_precision_graph)

    ##############
    # function to test on this case
    g_over.add_relevant_null_flow_lines(g_distribution_graph, non_connected_lines=non_connected_reconnectable_lines,non_reconnectable_lines=lines_non_reconnectable, target_path="red_only")

    ##############
    # tests
    all_edges=g_over.g.edges

    edge_1=('GEN.PP6','BOISSP6', 0)
    if edge_1 not in all_edges:
        edge_1=(edge_1[1],edge_1[0],edge_1[2])
    edge_1_attributes=all_edges[edge_1]
    assert(edge_1_attributes["color"]=="coral" and edge_1_attributes["style"]=="dashed" and edge_1_attributes["dir"]=="none")
    #path
    edges_path=[('BOISSP6', 'ZJOUXP6', 0),('MACONP6', 'ZJOUXP6', 0),('GROSNP6', 'MACONP6', 0)]
    for edge in edges_path:
        if edge not in all_edges:
            edge = (edge[1], edge[0], edge[2])
        edge_attributes = all_edges[edge]
        assert (edge_attributes["color"] == "coral")

    edge2=('CHALOP3', 'LOUHAP3', 0)
    if edge2 not in all_edges:
        edge2=(edge2[1],edge2[0],edge2[2])
    edge2_attributes=all_edges[edge2]
    assert (edge2_attributes["color"] == "coral" and edge2_attributes["style"] == "dashed")

    edges_path = [('CHALOP6', 'CHALOP3', 0),('CHALOP6', 'CHALOP3', 1)]
    for edge in edges_path:
        if edge not in all_edges:
            edge = (edge[1], edge[0], edge[2])
        edge_attributes = all_edges[edge]
        assert (edge_attributes["color"] == "coral")


def test_highlight_significant_line_loading():
    config = configparser.ConfigParser()
    config.read("./alphaDeesp/tests/resources_for_tests_grid2op/config_for_tests.ini")

    data_folder = "./alphaDeesp/tests/ressources_for_tests/data_graph_consolidation/defaut_P.SAOL31RONCI"

    timestep = 1  # 1#36
    line_defaut = "P.SAOL31RONCI"
    float_precision_graph = "%.0f"

    g_over, situation_info = basic_test_configuration(line_defaut, timestep, float_precision_graph)

    with open(os.path.join(data_folder, 'significant_line_loading_change.json')) as json_file:
        dict_line_loading = json.load(json_file)
    g_over.highlight_significant_line_loading(dict_line_loading)

    test_edge=('CPVANP3', 'BEON P3', 0)
    assert(g_over.g.edges[test_edge]["color"]=='"black:yellow:black"')
    assert(g_over.g.edges[test_edge]["label"]=='< -30 <BR/>  <B>122%</B>  → 0%>')
    assert(g_over.g.edges[test_edge]["fontcolor"]=='darkred')
    #g_over.plot(layout=None, save_folder="./", fontsize=10, without_gray_edges=True)
