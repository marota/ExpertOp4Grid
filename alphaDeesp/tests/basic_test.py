# import sys
# import os

# sys.path.append(os.path.abspath("../../alphaDeesp.core"))
import networkx as nx
from alphaDeesp.core.constrainedPath import ConstrainedPath
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
g2 = nx.DiGraph()
g2.add_edges_from([(1,2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
                   (3, 9), (9, 10), (10, 6)])

g3 = nx.DiGraph()
g3.add_edges_from([(1,2), (2, 3), (2, 6), (3, 4), (5, 3)])


def test_constrained_path():
    """This function tests the Class ConstrainedPath"""

    c_path = ConstrainedPath([], (5, 6), [(6, 13)])
    assert c_path.n_amont() == [5]
    assert c_path.n_aval() == [6, 13]
    assert c_path.full_n_constrained_path() == [5, 6, 13]


def test_is_amont():
    """This function tests the function is_in_amont_of_node_x"""

    # p = Printer()
    # p.display_geo(g1, name="test_is_amont")

    # =================== test for graph g1 ===================
    #        5
    #         \>
    # 1->-2->-3->-4
    #      \>
    #      6
    assert AlphaDeesp.is_in_amont_of_node_x(g1, 1, 4) is True
    assert AlphaDeesp.is_in_amont_of_node_x(g1, 5, 4) is True
    assert AlphaDeesp.is_in_amont_of_node_x(g1, 2, 3) is True
    assert AlphaDeesp.is_in_amont_of_node_x(g1, 5, 6) is False
    assert AlphaDeesp.is_in_amont_of_node_x(g1, 6, 4) is False

    # =================== test for graph g2 ===================
    #           4->-5
    #          />   \>
    # 1->-2->-3      6->-7->-8
    #          \>   />
    #          9->-10
    assert AlphaDeesp.is_in_amont_of_node_x(g2, 1, 4) is True
    assert AlphaDeesp.is_in_amont_of_node_x(g2, 9, 6) is True
    assert AlphaDeesp.is_in_amont_of_node_x(g2, 1, 8) is True
    assert AlphaDeesp.is_in_amont_of_node_x(g2, 6, 9) is False
    assert AlphaDeesp.is_in_amont_of_node_x(g2, 5, 10) is False
    assert AlphaDeesp.is_in_amont_of_node_x(g2, 7, 8) is True


def test_is_aval():
    """This function tests the function is_in_amont_of_node_x"""

    # =================== test for graph g1 ===================
    #        5
    #         \>
    # 1->-2->-3->-4
    #      \>
    #      6
    assert AlphaDeesp.is_in_aval_of_node_x(g1, 2, 1) is True
    assert AlphaDeesp.is_in_aval_of_node_x(g1, 4, 1) is True
    assert AlphaDeesp.is_in_aval_of_node_x(g1, 4, 5) is True
    assert AlphaDeesp.is_in_aval_of_node_x(g1, 6, 1) is True
    assert AlphaDeesp.is_in_aval_of_node_x(g1, 5, 1) is False
    assert AlphaDeesp.is_in_aval_of_node_x(g1, 1, 2) is False
    assert AlphaDeesp.is_in_aval_of_node_x(g1, 6, 3) is False

    # =================== test for graph g2 ===================
    #           4->-5
    #          />   \>
    # 1->-2->-3      6->-7->-8
    #          \>   />
    #          9->-10
    assert AlphaDeesp.is_in_aval_of_node_x(g2, 4, 1) is True
    assert AlphaDeesp.is_in_aval_of_node_x(g2, 6, 9) is True
    assert AlphaDeesp.is_in_aval_of_node_x(g2, 8, 1) is True
    assert AlphaDeesp.is_in_aval_of_node_x(g2, 9, 6) is False
    assert AlphaDeesp.is_in_aval_of_node_x(g2, 10, 5) is False
    assert AlphaDeesp.is_in_aval_of_node_x(g2, 8, 7) is True


def test_get_constrained_path():
    """This function tests the function to obtain the constrained path.
    input: graph networkX g
    test if we get constrainedPath 5, 6, 13"""
    pass
    # input will be graph g, save in a .dot file.
    graph_dot_file_path = ""


# test_is_amont()
# test_is_aval()