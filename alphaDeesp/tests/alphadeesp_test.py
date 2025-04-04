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

    data_folder="./alphaDeesp/tests/ressources_for_tests/data_graph_consolidation/defaut_PSAOL31RONCI"

    timestep = 1  # 1#36
    line_defaut = "P.SAOL31RONCI"
    ltc = [9]

    expected_lines_swapped=['C.FOUL31MERVA','LOUHAL31SSUSU','MERVAL31SSUSU','TAVA5Y611']

    with open(os.path.join(data_folder,'sim_topo_zone_dijon_defaut_PSAOL31RONCI_t1.json')) as json_file:
        sim_topo_reduced = json.load(json_file)

    df_of_g = pd.read_csv(os.path.join(data_folder,"df_of_g_defaut_PSAOL31RONCI_t1.csv"))

    lines_swapped=list(df_of_g[df_of_g.new_flows_swapped].line_name)
    assert(set(expected_lines_swapped)==set(lines_swapped))

def test_consolidate_constrained_path():
    config = configparser.ConfigParser()
    config.read("./alphaDeesp/tests/resources_for_tests_grid2op/config_for_tests.ini")

    data_folder="./alphaDeesp/tests/ressources_for_tests/data_graph_consolidation/defaut_PSAOL31RONCI"

    timestep = 1  # 1#36
    line_defaut = "P.SAOL31RONCI"
    ltc = [9]

    with open(os.path.join(data_folder,'sim_topo_zone_dijon_defaut_PSAOL31RONCI_t1.json')) as json_file:
        sim_topo_reduced = json.load(json_file)

    df_of_g = pd.read_csv(os.path.join(data_folder,"df_of_g_defaut_PSAOL31RONCI_t1.csv"))

    #inject modification to test for reversing properly a null flow redispatch edge on the constrained path
    #df_of_g.loc[["idx_or","idx_ex"]]CPVANY632
    idx_or,idx_ex=df_of_g[df_of_g.line_name == "CPVANY632"][["idx_ex", "idx_or"]].values[0]
    df_of_g.loc[df_of_g.line_name == "CPVANY632",["idx_or", "idx_ex"]] = [idx_or, idx_ex]

    g_over = OverFlowGraph(sim_topo_reduced, ltc, df_of_g)

    with open(os.path.join(data_folder,'node_name_mapping_defaut_PSAOL31RONCI_t1.json')) as json_file:
        mapping = json.load(json_file)
    mapping = {int(key): value for key, value in mapping.items()}
    g_over.g = nx.relabel_nodes(g_over.g, mapping, copy=True)

    #consolidate
    g_distribution_graph = Structured_Overload_Distribution_Graph(g_over.g)

    # consolider le chemin en contrainte avec la connaissance des hubs, en itérant une fois de plus
    n_hubs_init = 0
    hubs_paths = g_distribution_graph.find_loops()[["Source", "Target"]].drop_duplicates()
    n_hub_paths = hubs_paths.shape[0]

    while n_hubs_init != n_hub_paths:
        n_hubs_init = n_hub_paths

        g_over.consolidate_constrained_path(hubs_paths.Source, hubs_paths.Target)
        g_distribution_graph = Structured_Overload_Distribution_Graph(g_over.g)

        hubs_paths = g_distribution_graph.find_loops()[["Source", "Target"]].drop_duplicates()
        n_hub_paths = hubs_paths.shape[0]

    #le nombre de loop path est passe de 1 a 3
    assert(n_hub_paths==3)

    #test that some blue edges have been corrected regarding their capacity label and direction
    edge=('COMMUP6', 'VIELMP6', 0)#: {'capacity': -3.01, 'label': '-3.01'}
    current_capacities = nx.get_edge_attributes(g_over.g, 'capacity')
    assert(current_capacities[edge]==-3.01)

    current_colors = nx.get_edge_attributes(g_over.g, 'color')
    assert (current_colors[edge] == "blue")

def test_reverse_blue_edges_in_looppaths():
    config = configparser.ConfigParser()
    config.read("./alphaDeesp/tests/resources_for_tests_grid2op/config_for_tests.ini")

    data_folder="./alphaDeesp/tests/ressources_for_tests/data_graph_consolidation/defaut_FRON5L31LOUHA"

    timestep = 36  # 1#36
    line_defaut = "FRON5L31LOUHA"
    ltc = [108]

    with open(os.path.join(data_folder,'sim_topo_zone_dijon_defaut_FRON5L31LOUHA_t36.json')) as json_file:
        sim_topo_reduced = json.load(json_file)

    df_of_g = pd.read_csv(os.path.join(data_folder,"df_of_g_defaut_FRON5L31LOUHA_t36.csv"))

    g_over = OverFlowGraph(sim_topo_reduced, ltc, df_of_g)

    with open(os.path.join(data_folder,'node_name_mapping_defaut_FRON5L31LOUHA_t36.json')) as json_file:
        mapping = json.load(json_file)
    mapping = {int(key): value for key, value in mapping.items()}
    g_over.g = nx.relabel_nodes(g_over.g, mapping, copy=True)

    #compute initial blue edges
    g_without_pos_edges = delete_color_edges(g_over.g, "coral")
    g_with_only_blue_edges=delete_color_edges(g_without_pos_edges, "coral")
    n_blue_edges_init=len(g_with_only_blue_edges.edges)

    #consolidate
    g_distribution_graph = Structured_Overload_Distribution_Graph(g_over.g)
    constrained_path = g_distribution_graph.constrained_path.full_n_constrained_path()
    g_over.reverse_blue_edges_in_looppaths(constrained_path)

    #count number of changes
    g_without_pos_edges = delete_color_edges(g_over.g, "coral")
    g_with_only_blue_edges=delete_color_edges(g_without_pos_edges, "coral")
    n_blue_edges_final=len(g_with_only_blue_edges.edges)

    #7 edge on change de couleur
    assert(n_blue_edges_init-n_blue_edges_final)

def test_consolidate_loop_path():
    config = configparser.ConfigParser()
    config.read("./alphaDeesp/tests/resources_for_tests_grid2op/config_for_tests.ini")

    data_folder="./alphaDeesp/tests/ressources_for_tests/data_graph_consolidation/defaut_FRON5L31LOUHA"

    timestep = 36  # 1#36
    line_defaut = "FRON5L31LOUHA"
    ltc = [108]

    with open(os.path.join(data_folder,'sim_topo_zone_dijon_defaut_FRON5L31LOUHA_t36.json')) as json_file:
        sim_topo_reduced = json.load(json_file)

    df_of_g = pd.read_csv(os.path.join(data_folder,"df_of_g_defaut_FRON5L31LOUHA_t36.csv"))

    g_over = OverFlowGraph(sim_topo_reduced, ltc, df_of_g)

    with open(os.path.join(data_folder,'node_name_mapping_defaut_FRON5L31LOUHA_t36.json')) as json_file:
        mapping = json.load(json_file)
    mapping = {int(key): value for key, value in mapping.items()}
    g_over.g = nx.relabel_nodes(g_over.g, mapping, copy=True)

    #reverse blue edges
    g_distribution_graph = Structured_Overload_Distribution_Graph(g_over.g)
    constrained_path = g_distribution_graph.constrained_path.full_n_constrained_path()
    g_over.reverse_blue_edges_in_looppaths(constrained_path)

    #compute initial red edges
    g_without_blue_edges = delete_color_edges(g_over.g, "blue")
    g_with_only_red_edges=delete_color_edges(g_without_blue_edges, "gray")
    n_red_edges_init=len(g_with_only_red_edges.edges)

    #consolidate loop paths
    hubs_paths = g_distribution_graph.find_loops()[["Source", "Target"]].drop_duplicates()
    g_over.consolidate_loop_path(hubs_paths.Source, hubs_paths.Target)

    #compute final red edges
    g_without_blue_edges = delete_color_edges(g_over.g, "blue")
    g_with_only_red_edges=delete_color_edges(g_without_blue_edges, "gray")
    n_red_edges_final=len(g_with_only_red_edges.edges)

    #7 edge on change de couleur
    assert(n_red_edges_final-n_red_edges_init==22)

def test_consolidate_graph():
    config = configparser.ConfigParser()
    config.read("./alphaDeesp/tests/resources_for_tests_grid2op/config_for_tests.ini")

    data_folder="./alphaDeesp/tests/ressources_for_tests/data_graph_consolidation/defaut_FRON5L31LOUHA"

    timestep = 36  # 1#36
    line_defaut = "FRON5L31LOUHA"
    ltc = [108]

    with open(os.path.join(data_folder,'sim_topo_zone_dijon_defaut_FRON5L31LOUHA_t36.json')) as json_file:
        sim_topo_reduced = json.load(json_file)

    df_of_g = pd.read_csv(os.path.join(data_folder,"df_of_g_defaut_FRON5L31LOUHA_t36.csv"))

    g_over = OverFlowGraph(sim_topo_reduced, ltc, df_of_g)

    with open(os.path.join(data_folder,'node_name_mapping_defaut_FRON5L31LOUHA_t36.json')) as json_file:
        mapping = json.load(json_file)
    mapping = {int(key): value for key, value in mapping.items()}
    g_over.g = nx.relabel_nodes(g_over.g, mapping, copy=True)

    #compute initial red edges
    g_without_blue_edges = delete_color_edges(g_over.g, "blue")
    g_with_only_red_edges_init=delete_color_edges(g_without_blue_edges, "gray")
    n_red_edges_init=len(g_with_only_red_edges_init.edges)

    g_distribution_graph = Structured_Overload_Distribution_Graph(g_over.g)
    g_over.consolidate_graph(g_distribution_graph)


    #compute final red edges
    g_without_blue_edges = delete_color_edges(g_over.g, "blue")
    g_with_only_red_edges_final=delete_color_edges(g_without_blue_edges, "gray")
    n_red_edges_final=len(g_with_only_red_edges_final.edges)

    edges_red_name_initial = list(nx.get_edge_attributes(g_with_only_red_edges_init, "name").values())
    edges_red_name_final = list(nx.get_edge_attributes(g_with_only_red_edges_final, "name").values())

    #7 edge on change de couleur
    edge_names_expected_change=['GEN.PL61IZERN','CIZE L61IZERN','CIZE L61FLEYR','CREYSL71SSV.O','CREYSL72SSV.O','CREYSL72GEN.P','CREYSL71GEN.P']
    n_targeted_change = len(edge_names_expected_change)
    targeted_changes_seen=(set(edges_red_name_final)-set(edges_red_name_initial)).intersection(set(edge_names_expected_change))
    n_targeted_change_seen = len(targeted_changes_seen)

    assert(n_targeted_change==n_targeted_change_seen)#explicit test to comply with
    assert(n_red_edges_final-n_red_edges_init==28)#to check if anything changed, but not very explicit test and value can change if good reason

def test_identify_ambiguous_paths_and_type():
    config = configparser.ConfigParser()
    config.read("./alphaDeesp/tests/resources_for_tests_grid2op/config_for_tests.ini")

    data_folder = "./alphaDeesp/tests/ressources_for_tests/data_graph_consolidation/defaut_PSAOL31RONCI"

    timestep = 1  # 1#36
    line_defaut = "P.SAOL31RONCI"
    ltc = [9]

    with open(os.path.join(data_folder, 'sim_topo_zone_dijon_defaut_PSAOL31RONCI_t1.json')) as json_file:
        sim_topo_reduced = json.load(json_file)

    df_of_g = pd.read_csv(os.path.join(data_folder, "df_of_g_defaut_PSAOL31RONCI_t1.csv"))
    #make significant delta flows thereshold lower, for more interesting ambiguous path detection
    new_threshold=2#2MW
    df_of_g.loc[df_of_g.delta_flows.abs() >= new_threshold, "gray_edges"] = False

    g_over = OverFlowGraph(sim_topo_reduced, ltc, df_of_g)

    with open(os.path.join(data_folder, 'node_name_mapping_defaut_PSAOL31RONCI_t1.json')) as json_file:
        mapping = json.load(json_file)

    mapping = {int(key): value for key, value in mapping.items()}
    g_over.rename_nodes(mapping)  # g = nx.relabel_nodes(g_over.g, mapping, copy=True)

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

    data_folder = "./alphaDeesp/tests/ressources_for_tests/data_graph_consolidation/defaut_FRON5L31LOUHA"

    timestep = 36  # 1#36
    line_defaut = "FRON5L31LOUHA"
    ltc = [108]

    with open(os.path.join(data_folder,'sim_topo_zone_dijon_defaut_FRON5L31LOUHA_t36.json')) as json_file:
        sim_topo_reduced = json.load(json_file)

    df_of_g = pd.read_csv(os.path.join(data_folder,"df_of_g_defaut_FRON5L31LOUHA_t36.csv"))

    g_over = OverFlowGraph(sim_topo_reduced, ltc, df_of_g)

    with open(os.path.join(data_folder, 'node_name_mapping_defaut_FRON5L31LOUHA_t36.json')) as json_file:
        mapping = json.load(json_file)

    mapping = {int(key): value for key, value in mapping.items()}
    g_over.rename_nodes(mapping)  # g = nx.relabel_nodes(g_over.g, mapping, copy=True)

    g_distribution_graph = Structured_Overload_Distribution_Graph(g_over.g)

    expected_ambiguous_node_path=['CIZE P6', 'FLEYRP6', 'GEN.PP6', 'IZERNP6', 'VOUGLP6']
    ambiguous_edge_paths, ambiguous_node_paths = g_over.identify_ambiguous_paths(g_distribution_graph)

    #
    #{('IZERNP6', 'CIZE P6', 0): 'blue',
    # ('VOUGLP6', 'FLEYRP6', 0): 'coral',
    # ('GEN.PP6', 'IZERNP6', 0): 'blue',
    # ('CIZE P6', 'FLEYRP6', 0): 'blue'}
    assert(set(ambiguous_node_paths[0])==set(expected_ambiguous_node_path))

    path_type=g_over.desambiguation_type_path(expected_ambiguous_node_path, g_distribution_graph)
    assert(path_type=="loop_path")

def test_add_relevant_null_flow_lines(non_reconnectable_lines=[]):
    #vérifier le path GROSNP6, ZJOUXP6, BOISSP6, GEN.PP6 avec BOISSP6 seeulement en pointillé
    #Path CHALOP3 -> LOUHAP3
    #Path CPVANP6 PYMONP6 VOUGLP6 PYMONP3

    config = configparser.ConfigParser()
    config.read("./alphaDeesp/tests/resources_for_tests_grid2op/config_for_tests.ini")

    data_folder = "./alphaDeesp/tests/ressources_for_tests/data_graph_consolidation/defaut_PSAOL31RONCI"

    timestep = 1  # 1#36
    line_defaut = "P.SAOL31RONCI"
    ltc = [9]

    non_connected_lines = ['BOISSL61GEN.P', 'CHALOL31LOUHA', 'CRENEL71VIELM', 'CURTIL61ZCUR5', 'GEN.PL73VIELM',
                           'P.SAOL31RONCI',
                           'PYMONL61VOUGL', 'BUGEYY715', 'CPVANY632', 'GEN.PY762', 'PYMONY632']

    with open(os.path.join(data_folder, 'sim_topo_zone_dijon_defaut_PSAOL31RONCI_t1.json')) as json_file:
        sim_topo_reduced = json.load(json_file)

    df_of_g = pd.read_csv(os.path.join(data_folder, "df_of_g_defaut_PSAOL31RONCI_t1.csv"))

    g_over = OverFlowGraph(sim_topo_reduced, ltc, df_of_g,float_precision="%.0f")

    with open(os.path.join(data_folder, 'node_name_mapping_defaut_PSAOL31RONCI_t1.json')) as json_file:
        mapping = json.load(json_file)

    mapping = {int(key): value for key, value in mapping.items()}
    g_over.rename_nodes(mapping)  # g = nx.relabel_nodes(g_over.g, mapping, copy=True)

    # consolidate
    g_distribution_graph = Structured_Overload_Distribution_Graph(g_over.g)

    g_over.consolidate_graph(g_distribution_graph)

    # g_over.add_double_edges_null_redispatch()
    g_distribution_graph = Structured_Overload_Distribution_Graph(g_over.g)

    for i in range(2):#need two iterations to identify CHALOL31LOUHA reconnectable path under contingency "BEON L31CPVAN" at timestep 1 on chronic 28th august
        g_over.add_relevant_null_flow_lines_all_paths(g_distribution_graph, non_connected_lines,non_reconnectable_lines)
        g_distribution_graph = Structured_Overload_Distribution_Graph(g_over.g)
    #g_over.add_relevant_null_flow_lines_all_paths(g_distribution_graph, non_connected_lines)

    color_edges = list(nx.get_edge_attributes(g_over.g, 'color').values())
    line_names = list(nx.get_edge_attributes(g_over.g, 'name').values())

    line_tests=['BOISSL61GEN.P', 'CHALOL31LOUHA','PYMONL61VOUGL', 'CPVANY632', 'GEN.PY762', 'PYMONY632']

    significant_colors=set(["blue","coral"])#have we highlighted those lines on significant paths ?
    for line in line_tests:
        assert(len(set([color for color, line_name in zip(color_edges, line_names) if line_name == line]).intersection(significant_colors))>=1)

    #g_distribution_graph = Structured_Overload_Distribution_Graph(g_over.g)
    hubs=g_distribution_graph.get_hubs()

    new_hubs_to_test=["CPVANP6","CHALOP6","GROSNP6"]
    n_new_hubs=len(new_hubs_to_test)

    assert(n_new_hubs==len(set(new_hubs_to_test).intersection(set(hubs))))

def test_add_relevant_null_flow_lines_non_reconnectables_lines():
    #vérifier le path GROSNP6, ZJOUXP6, BOISSP6, GEN.PP6 avec BOISSP6 seeulement en pointillé
    #Path CHALOP3 -> LOUHAP3
    #Path CPVANP6 PYMONP6 VOUGLP6 PYMONP3

    config = configparser.ConfigParser()
    config.read("./alphaDeesp/tests/resources_for_tests_grid2op/config_for_tests.ini")

    data_folder = "./alphaDeesp/tests/ressources_for_tests/data_graph_consolidation/defaut_PSAOL31RONCI"

    timestep = 1  # 1#36
    line_defaut = "P.SAOL31RONCI"
    ltc = [9]

    non_connected_lines = ['BOISSL61GEN.P', 'CHALOL31LOUHA', 'CRENEL71VIELM', 'CURTIL61ZCUR5', 'GEN.PL73VIELM',
                           'P.SAOL31RONCI',
                           'PYMONL61VOUGL', 'BUGEYY715', 'CPVANY632', 'GEN.PY762', 'PYMONY632']
    non_reconnectable_lines =  ['CRENEL71VIELM', 'GEN.PL73VIELM', 'PYMONL61VOUGL', 'CPVANY632', 'PYMONY632']

    with open(os.path.join(data_folder, 'sim_topo_zone_dijon_defaut_PSAOL31RONCI_t1.json')) as json_file:
        sim_topo_reduced = json.load(json_file)

    df_of_g = pd.read_csv(os.path.join(data_folder, "df_of_g_defaut_PSAOL31RONCI_t1.csv"))

    g_over = OverFlowGraph(sim_topo_reduced, ltc, df_of_g,float_precision="%.0f")

    with open(os.path.join(data_folder, 'node_name_mapping_defaut_PSAOL31RONCI_t1.json')) as json_file:
        mapping = json.load(json_file)

    mapping = {int(key): value for key, value in mapping.items()}
    g_over.rename_nodes(mapping)  # g = nx.relabel_nodes(g_over.g, mapping, copy=True)

    # consolidate
    g_distribution_graph = Structured_Overload_Distribution_Graph(g_over.g)

    g_over.consolidate_graph(g_distribution_graph)

    # g_over.add_double_edges_null_redispatch()
    g_distribution_graph = Structured_Overload_Distribution_Graph(g_over.g)

    for i in range(2):#need two iterations to identify CHALOL31LOUHA reconnectable path under contingency "BEON L31CPVAN" at timestep 1 on chronic 28th august
        g_over.add_relevant_null_flow_lines_all_paths(g_distribution_graph, non_connected_lines,non_reconnectable_lines)
        g_distribution_graph = Structured_Overload_Distribution_Graph(g_over.g)
    #g_over.add_relevant_null_flow_lines_all_paths(g_distribution_graph, non_connected_lines)

    color_edges = nx.get_edge_attributes(g_over.g, 'color')
    style_edges = nx.get_edge_attributes(g_over.g, 'style')
    dir_edges = nx.get_edge_attributes(g_over.g, 'dir')
    line_names = nx.get_edge_attributes(g_over.g, 'name')

    line_tests=['GEN.PL73VIELM', 'PYMONL61VOUGL', 'CPVANY632', 'PYMONY632']#'CRENEL71VIELM',

    #color dimgray and style dotted
    for line in line_tests:
        line_edge=[edge for edge, line_name in line_names.items() if line_name == line][0]
        assert(color_edges[line_edge]=="dimgray")
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
    ltc = [9]

    non_connected_lines=['BOISSL61GEN.P','CHALOL31LOUHA','CRENEL71VIELM','CURTIL61ZCUR5','GEN.PL73VIELM','P.SAOL31RONCI',
     'PYMONL61VOUGL','BUGEYY715','CPVANY632','GEN.PY762','PYMONY632']

    with open(os.path.join(data_folder,'sim_topo_zone_dijon_defaut_PSAOL31RONCI_t1.json')) as json_file:
        sim_topo_reduced = json.load(json_file)

    df_of_g = pd.read_csv(os.path.join(data_folder,"df_of_g_defaut_PSAOL31RONCI_t1.csv"))

    g_over = OverFlowGraph(sim_topo_reduced, ltc, df_of_g)

    with open(os.path.join(data_folder,'node_name_mapping_defaut_PSAOL31RONCI_t1.json')) as json_file:
        mapping = json.load(json_file)

    mapping = {int(key): value for key, value in mapping.items()}
    g_over.rename_nodes(mapping)#g = nx.relabel_nodes(g_over.g, mapping, copy=True)

    with open(os.path.join(data_folder,'voltage_levels.json')) as json_file:
        voltage_levels_dict = json.load(json_file)
    g_over.set_voltage_level_color(voltage_levels_dict)

    with open(os.path.join(data_folder,'number_nodal_dict.json')) as json_file:
        number_nodal_dict = json.load(json_file)
    g_over.set_electrical_node_number(number_nodal_dict)


    #consolidate
    g_distribution_graph = Structured_Overload_Distribution_Graph(g_over.g)

    #check that this edge has been coloured blue
    assert(g_over.g.edges[('CPVANP6', 'CPVANP3', 1)]["color"]=="gray")
    g_over.add_relevant_null_flow_lines(g_distribution_graph, non_connected_lines, target_path="blue_only")
    assert (g_over.g.edges[('CPVANP6', 'CPVANP3', 1)]["color"] == "blue")
    # check that this double edge has been removed (in the other edge direction)
    assert(not g_over.g.has_edge('CPVANP3', 'CPVANP6'))


def test_highlight_significant_line_loading():
    config = configparser.ConfigParser()
    config.read("./alphaDeesp/tests/resources_for_tests_grid2op/config_for_tests.ini")

    data_folder = "./alphaDeesp/tests/ressources_for_tests/data_graph_consolidation/defaut_PSAOL31RONCI"

    timestep = 1  # 1#36
    line_defaut = "P.SAOL31RONCI"
    ltc = [9]

    with open(os.path.join(data_folder, 'sim_topo_zone_dijon_defaut_PSAOL31RONCI_t1.json')) as json_file:
        sim_topo_reduced = json.load(json_file)

    df_of_g = pd.read_csv(os.path.join(data_folder, "df_of_g_defaut_PSAOL31RONCI_t1.csv"))

    g_over = OverFlowGraph(sim_topo_reduced, ltc, df_of_g, float_precision="%.0f")

    with open(os.path.join(data_folder, 'node_name_mapping_defaut_PSAOL31RONCI_t1.json')) as json_file:
        mapping = json.load(json_file)

    mapping = {int(key): value for key, value in mapping.items()}
    g_over.rename_nodes(mapping)  # g = nx.relabel_nodes(g_over.g, mapping, copy=True)

    with open(os.path.join(data_folder, 'significant_line_loading_change.json')) as json_file:
        dict_line_loading = json.load(json_file)
    g_over.highlight_significant_line_loading(dict_line_loading)

    test_edge=('CPVANP3', 'BEON P3', 0)
    assert(g_over.g.edges[test_edge]["color"]=='"black:yellow:black"')
    assert(g_over.g.edges[test_edge]["label"]=='< -30 <BR/>  <B>122%</B>  → 0%>')
    assert(g_over.g.edges[test_edge]["fontcolor"]=='darkred')
    #g_over.plot(layout=None, save_folder="./", fontsize=10, without_gray_edges=True)
