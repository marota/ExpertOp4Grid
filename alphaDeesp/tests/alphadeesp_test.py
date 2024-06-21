# import sys
# import os

# sys.path.append(os.path.abspath("../../alphaDeesp.core"))
import networkx as nx
from alphaDeesp.core.graphsAndPaths import ConstrainedPath,Structured_Overload_Distribution_Graph
from alphaDeesp.core.alphadeesp import *
from alphaDeesp.core.printer import Printer

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
            g2[u][v][idx]["color"] = "red"

    Overload_graph=Structured_Overload_Distribution_Graph(g2)

    assert set(Overload_graph.get_hubs())==set(hubs)
    assert set(Overload_graph.get_constrained_path().full_n_constrained_path())==set(nodes_c_path)


    loops_df=Overload_graph.get_loops()
    assert loops_df.loc[0]["Path"]==loop
    assert loops_df.loc[0]["Source"]==3
    assert loops_df.loc[0]["Target"] == 6
