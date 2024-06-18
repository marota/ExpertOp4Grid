"""Class representing a constrained path"""
import pandas as pd
import networkx as nx

class ConstrainedPath:
    def __init__(self, amont_edges, constrained_edge, aval_edges):
        print("Constrained path created")
        self.amont_edges = amont_edges
        self.constrained_edge = constrained_edge
        self.aval_edges = aval_edges

    def n_amont(self) -> list:
        """Returns a list of nodes that are in "amont" """
        return from_edges_get_nodes(self.amont_edges, "amont", self.constrained_edge)

    def n_aval(self):
        """Returns a list of nodes that are in "aval" """
        return from_edges_get_nodes(self.aval_edges, "aval", self.constrained_edge)

    def e_amont(self):
        """Returns a list of edges that are in "amont" """
        return self.amont_edges

    def e_aval(self):
        """Returns a list of edges that are in "aval" """
        return self.aval_edges

    def full_e_constrained_path(self):
        return filter_constrained_path_for_edges([self.amont_edges, self.constrained_edge, self.aval_edges])

    def full_n_constrained_path(self):
        return filter_constrained_path_for_nodes([self.amont_edges, self.constrained_edge, self.aval_edges])

    # def __repr__(self):
    #     return "ConstrainedPath(amont: %s, constrained_edge: %s, aval: %s) ## Recap: Constrained Path = %s" % (
    #     self.amont_edges, self.constrained_edge, self.aval_edges, self.full_n_constrained_path())

    def __repr__(self):
        return "################################################################\n" \
               "ConstrainedPath = %s \nDetails: (amont: %s, constrained_edge: %s, aval: %s)\n" \
               "################################################################" % (
                   self.full_n_constrained_path(), self.amont_edges, self.constrained_edge, self.aval_edges)

class Structured_Overload_Distribution_Graph:
    """
    Staring from a raw overload distibution graph with color edges, this class identifies the underlying path structure in terms of constrained path, loop paths and hub nodes
    """
    def __init__(self,g):
        self.g_init=g
        self.g_without_pos_edges = self.delete_color_edges(self.g_init, "red") #graph without loop path that have positive/red-coloured weight edges
        self.g_only_blue_components = self.delete_color_edges(self.g_without_pos_edges, "gray") #graph with only negative/blue-coloured weight edges
        self.g_without_constrained_edge = self.delete_color_edges(self.g_init, "black")
        self.g_without_gray_and_c_edge = self.delete_color_edges(self.g_without_constrained_edge, "gray")
        self.g_only_red_components = self.delete_color_edges(self.g_without_gray_and_c_edge, "blue")#graph with only loop path that have positive/red-coloured weight edges
        self.constrained_path= self.find_constrained_path() #constrained path that contains the constrained edges and their connected component of blue edges
        self.type=""#
        self.red_loops = self.find_loops() #parallel path to the constrained path on which flow can be rerouted
        self.hubs = self.find_hubs() #specific nodes at substations connecting loop paths to constrained path. This is where flow can be most easily rerouted


    def get_amont_blue_edges(self, g, node):
        res = []
        for e in nx.edge_dfs(g, node, orientation="reverse"):
            if g.edges[(e[0], e[1],e[2])]["color"] == "blue":
                res.append((e[0], e[1],e[2]))
        return res

    def get_aval_blue_edges(self, g, node):
        res = []
        # print("debug AlphaDeesp get aval blue edges")
        # print(list(nx.edge_dfs(g, node, orientation="original")))
        for e in nx.edge_dfs(g, node, orientation="original"):
            if g.edges[(e[0], e[1],e[2])]["color"] == "blue":
                res.append((e[0], e[1],e[2]))
        return res

    def delete_color_edges(self, _g, edge_color):
        """Returns a copy of g without gray edges"""
        g = _g.copy()

        gray_edges = []
        i = 1
        for u, v,idx, color in g.edges(data="color",keys=True):
            if color == edge_color:
                gray_edges.append((i, (u, v,idx)))
            i += 1

        # delete from graph gray edges
        # this extracts the (u,v) from pos_edges
        if gray_edges:
            g.remove_edges_from(list(zip(*gray_edges))[1])
        return g


    def find_hubs(self):
        """A hub (carrefour_electrique) has a constrained_path and positiv reports"""
        g = self.g_without_constrained_edge
        hubs = []

        if self.constrained_path is not None:
            print("In get_hubs(): c = ")
            print(self.constrained_path)
        else:
            e_amont, constrained_edge, e_aval = self.get_constrained_path()
            self.constrained_path = ConstrainedPath(e_amont, constrained_edge, e_aval)

        # for nodes in aval, if node has RED inputs (ie incoming flows) then it is a hub
        for node in self.constrained_path.n_aval():
            in_edges = list(g.in_edges(node,keys=True))
            for e in in_edges:
                if g.edges[e]["color"] == "red":
                    hubs.append(node)
                    break

        # for nodes in amont, if node has RED outputs (ie outgoing flows) then it is a hub
        for node in self.constrained_path.n_amont():
            out_edges = list(g.out_edges(node,keys=True))
            for e in out_edges:
                if g.edges[e]["color"] == "red":
                    hubs.append(node)
                    break

        # print("get_hubs = ", hubs)
        return hubs

    def get_hubs(self):
        return self.hubs

    def find_loops(self):
        """This function returns all parallel paths. After discussing with Antoine, start with the most "en Aval" node,
        and walk in reverse for loops and parallel path returns a dict with all data """

        # print("==================== In function get_loops ====================")
        g = self.g_only_red_components
        c_path_n = self.constrained_path.full_n_constrained_path()
        all_loop_paths = {}
        ii = 0

        for i in range(len(c_path_n)):
            for j in reversed(range(len(c_path_n))):
                if i < j:
                    # # print(i, j)
                    # # print("we compare paths from source: {} to target: {}".format(c_path_n[i], c_path_n[j]))
                    try:
                        res = nx.all_shortest_paths(g, c_path_n[i], c_path_n[j])
                        for p in res:
                            # print("path = ", p)
                            all_loop_paths[ii] = p
                            ii += 1
                    except nx.NetworkXNoPath:
                        print("shortest path between {0} and {1} failed".format(c_path_n[i], c_path_n[j]))

        # print("### Print in get_loops ###, all_loop_paths")
        # pprint.pprint(all_loop_paths)

        data_for_df = {"Source": [], "Target": [], "Path": []}
        for path in list(all_loop_paths.keys()):
            data_for_df["Source"].append(all_loop_paths[path][0])
            data_for_df["Target"].append(all_loop_paths[path][-1])
            data_for_df["Path"].append(all_loop_paths[path])

        # pprint.pprint(data_for_df)

        return pd.DataFrame.from_dict(data_for_df)

    def get_loops(self):
        return self.red_loops

    def find_constrained_path(self):
        """Return the constrained path"""
        constrained_edge = None
        tmp_constrained_path = []
        edge_list = nx.get_edge_attributes(self.g_only_blue_components, "color")
        for edge, color in edge_list.items():
            if color == "black":
                constrained_edge = edge
        amont_edges = self.get_amont_blue_edges(self.g_only_blue_components, constrained_edge[0])
        aval_edges = self.get_aval_blue_edges(self.g_only_blue_components, constrained_edge[1])

        return ConstrainedPath(amont_edges,constrained_edge,aval_edges)

    def get_constrained_path(self):
        return self.constrained_path


def from_edges_get_nodes(edges, amont_or_aval: str, constrained_edge):
    """edges is a list of tuples"""
    if edges:
        nodes = []
        for e in edges:
            for node in e[:2]:
                if node not in nodes:
                    nodes.append(node)
        return nodes
    elif amont_or_aval == "amont":
        return [constrained_edge[0]]
    elif amont_or_aval == "aval":
        return [constrained_edge[1]]
    else:
        raise ValueError("Error in function from_edges_get_nodes")


def filter_constrained_path_for_nodes(constrained_path):
    # this filters the constrained_path_lists and creates a uniq ordered list that represents the constrained_path
    set_constrained_path = []
    for path in constrained_path:
        if isinstance(path, tuple):
            edge=path
            for n in edge[0:2]:
                if n not in set_constrained_path:
                    set_constrained_path.append(n)
        else:
            for edge in path:
                if isinstance(edge, tuple):
                    for n in edge[0:2]:
                        if n not in set_constrained_path:
                            set_constrained_path.append(n)

    return set_constrained_path

def filter_constrained_path_for_edges(constrained_path):
    # this filters the constrained_path_lists and creates a uniq ordered list that represents the constrained_path
    set_constrained_path = []
    for edge in constrained_path:
        if isinstance(edge, tuple):
            set_constrained_path.append(edge)
        elif isinstance(edge, list):
            for e in edge:
                if isinstance(e, tuple):
                    set_constrained_path.append(e)

    return set_constrained_path
