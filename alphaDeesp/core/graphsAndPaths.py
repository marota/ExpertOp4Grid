
import pandas as pd
import networkx as nx
from math import fabs
from alphaDeesp.core.printer import Printer

class PowerFlowGraph:
    """
    A coloured graph of current grid state with productions, consumptions and topology
    """

    def __init__(self, topo,lines_cut,layout=None):
        """
        Parameters
        ----------

        topo: :class:`dict`
            dictionnary of two dictionnaries edges and nodes, to represent the grid topologie. edges have attributes "init_flows" representing the power flowing, as well as "idx_or","idx_ex"
             for substation extremities
             Nodes have attributes "are_prods","are_loads" if nodes have any productions or any load, as well as "prods_values","load_values" array enumerating the prod and load values at this node.

        lines_cut: ``array``
            ids of lines disconnected

        """
        self.topo=topo
        self.lines_cut=lines_cut
        self.layout=layout
        self.build_graph()
        #self.g=self.build_powerflow_graph()

    def build_graph(self):
        """This method creates the NetworkX Graph of the grid state"""
        g = nx.MultiDiGraph()

        # Get the id of lines that are disconnected from network
        # lines_cut = np.argwhere(obs.line_status == False)[:, 0]
        topo=self.topo

        # Get the whole topology information
        idx_or = topo["edges"]['idx_or']
        idx_ex = topo["edges"]['idx_ex']
        are_prods = topo["nodes"]['are_prods']
        are_loads = topo["nodes"]['are_loads']
        prods_values = topo["nodes"]['prods_values']
        loads_values = topo["nodes"]['loads_values']
        current_flows = topo["edges"]['init_flows']

        # =========================================== NODE PART ===========================================
        self.build_nodes(g, are_prods, are_loads, prods_values, loads_values)
        # =========================================== EDGE PART ===========================================
        self.build_edges(g, idx_or, idx_ex, edge_weights=current_flows)
        #return g
        self.g=g

    def build_nodes(self,g, are_prods, are_loads, prods_values, loads_values,debug=False):
        """
        Create nodes in graph for current grid state

        Parameters
        ----------

        g: :class:`nx:MultiDiGraph`
            a networkx graph to which to add edges

        are_prods: ``array`` boolean
            if there are productions at each node

        are_loads: ``array`` boolean
            if there are cosnumptions at each node

        prods_values: ``array`` float
            the production values at each node

        loads_values: ``array`` float
            the consumption values at each node

        """
        # =========================================== NODE PART ===========================================
        # print(f"There are {len(are_loads)} nodes")
        prods_iter, loads_iter = iter(prods_values), iter(loads_values)
        i = 0
        # We color the nodes depending if they are production or consumption
        for is_prod, is_load in zip(are_prods, are_loads):
            prod = next(prods_iter) if is_prod else 0.
            load = next(loads_iter) if is_load else 0.
            prod_minus_load = prod - load
            if debug:
                print(f"Node n°[{i}] : Production value: [{prod}] - Load value: [{load}] ")
            if prod_minus_load > 0:  # PROD
                g.add_node(i, pin=True, prod_or_load="prod", value=str(prod_minus_load), style="filled",
                           fillcolor="#f30000")  # red color
            elif prod_minus_load < 0:  # LOAD
                g.add_node(i, pin=True, prod_or_load="load", value=str(prod_minus_load), style="filled",
                           fillcolor="#478fd0")  # blue color
            else:  # WHITE COLOR
                g.add_node(i, pin=True, prod_or_load="load", value=str(prod_minus_load), style="filled",
                           fillcolor="#ffffed")  # white color
            i += 1

    def build_edges(self,g, idx_or, idx_ex, edge_weights):

        """
        Create edges in graph for current grid state

        Parameters
        ----------

        g: :class:`nx:MultiDiGraph`
            a networkx graph to which to add edges

        idx_or: ``array`` int
            first extremity of edge for each edge

        idx_ex: ``array`` int
            second extremity of edge for each edge

        edge_weights: ``array`` float
            the flow value for each edge

        gtype: ``str``
            if we want a powerflow graph or

        """

        #if gtype is "powerflow":
        for origin, extremity, weight_value in zip(idx_or, idx_ex, edge_weights):
            # origin += 1
            # extremity += 1
            pen_width = fabs(weight_value) / 10
            if pen_width == 0.0:
                pen_width = 0.1

            if weight_value >= 0:
                g.add_edge(origin, extremity, xlabel="%.2f" % weight_value, color="gray", fontsize=10,
                           penwidth="%.2f" % pen_width)
            else:
                g.add_edge(extremity, origin, xlabel="%.2f" % fabs(weight_value), color="gray", fontsize=10,
                           penwidth="%.2f" % pen_width)


    def get_graph(self):
        return self.g

    def plot(self,save_folder,name,state="before",sim=None):
        printer = Printer(save_folder)

        #in case the simulator also provides a plot function, use it
        if sim is not None and hasattr(sim, 'plot'):
            output_name = printer.create_namefile("geo", name=name, type="base")
            if state == "before":
                obs = sim.obs
            else:
                obs = sim.obs_linecut
            sim.plot(obs,save_file_path=output_name[1])
        else:
            if self.layout:
                printer.display_geo(self.g, self.layout, name=name)


class OverFlowGraph(PowerFlowGraph):
    """
    A coloured graph of grid overflow redispatch, displaying the delta flows before and after disconnecting the overloaded lines
    """

    def __init__(self, topo,lines_to_cut,df_overflow,layout=None):
        """
        Parameters
        ----------

        topo: :class:`dict`
            dictionnary of two dictionnaries edges and nodes, to represent the grid topologie. edges have attributes "init_flows" representing the power flowing, as well as "idx_or","idx_ex"
             for substation extremities
             Nodes have attributes "are_prods","are_loads" if nodes have any productions or any load, as well as "prods_values","load_values" array enumerating the prod and load values at this node.

        lines_to_cut: ``array``
            ids of lines disconnected

        df_overflow: :class:``pd.Dataframe``
            pandas dataframe of deltaflows after disconnecting the overloaded lines. see create_df in simulation.py. One row per powerline
            columns: idx_or, idx_ex, init_flows, new_flows, delta_flows, gray_edges (for unsignificant delta_flows below a threshold)

        """
        self.df = df_overflow
        super().__init__(topo, lines_to_cut,layout)

    def build_graph(self):
        """This method creates the NetworkX Graph of the overflow redispatch """
        g = nx.MultiDiGraph()
        self.build_nodes(g, self.topo["nodes"]["are_prods"], self.topo["nodes"]["are_loads"],
                    self.topo["nodes"]["prods_values"], self.topo["nodes"]["loads_values"])

        self.build_edges_from_df(g, self.lines_cut)

        # print("WE ARE IN BUILD GRAPH FROM DATA FRAME ===========")
        # all_edges_xlabel_attributes = nx.get_edge_attributes(g, "xlabel")  # dict[edge]
        # print("all_edges_xlabel_attributes = ", all_edges_xlabel_attributes)

        self.g=g

    def build_edges_from_df(self, g, lines_to_cut):
        """
        Create edges in graph for overflow redispatch

        Parameters
        ----------

        g: :class:`nx:MultiDiGraph`
            a networkx graph to which to add edges

        lines_to_cut: ``array`` int
            list of lines in overflow that are getting disconnected

        """

        i = 0
        for origin, extremity, reported_flow, gray_edge in zip(self.df["idx_or"], self.df["idx_ex"],
                                                               self.df["delta_flows"], self.df["gray_edges"]):
            penwidth = fabs(reported_flow) / 10
            if penwidth == 0.0:
                penwidth = 0.1
            if i in lines_to_cut:
                g.add_edge(origin, extremity, capacity=float("%.2f" % reported_flow), xlabel="%.2f" % reported_flow,
                           color="black", style="dotted, setlinewidth(2)", fontsize=10, penwidth="%.2f" % penwidth,
                           constrained=True)
            elif gray_edge:  # Gray
                g.add_edge(origin, extremity, capacity=float("%.2f" % reported_flow), xlabel="%.2f" % reported_flow,
                           color="gray", fontsize=10, penwidth="%.2f" % penwidth)
            elif reported_flow < 0:  # Blue
                g.add_edge(origin, extremity, capacity=float("%.2f" % reported_flow), xlabel="%.2f" % reported_flow,
                           color="blue", fontsize=10, penwidth="%.2f" % penwidth)
            else:  # > 0  # Red
                g.add_edge(origin, extremity, capacity=float("%.2f" % reported_flow), xlabel="%.2f" % reported_flow,
                           color="red", fontsize=10, penwidth="%.2f" % penwidth)
            i += 1
    def plot(self,layout,save_folder="",without_gray_edges=False):
        printer=Printer(save_folder)
        g=self.g
        if without_gray_edges:
            g=delete_color_edges(g, "gray")
        if save_folder=="":
            output_graphviz_svg=printer.plot_graphviz(g, layout, name="g_overflow_print")
            return output_graphviz_svg
        else:
            printer.display_geo(g, layout, name="g_overflow_print")
            return None

class ConstrainedPath:

    """
    A connected path of lines that includes the overloaded lines and for which the overflow redipsatch is negative.
    This can be regarded as the main path through which the flows get in and out of the overloaded lines.
    Hence flows should be pushed on other path to relieve the overloads.
    """

    def __init__(self, amont_edges, constrained_edge, aval_edges):
        print("Constrained path created")
        self.amont_edges = amont_edges #lines which flow goes into the overloaded lines
        self.constrained_edge = constrained_edge #overloaded lines
        self.aval_edges = aval_edges #lines which flow comes from the overloaded lines

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

    def filter_constrained_path_for_nodes(self):
        """
        This filters the constrained_path_lists and creates a uniq ordered list that represents the constrained_path

        Returns
        ----------

        res: ``array`` int
            ordered list of nodes on the constrained path

        """
        set_constrained_path = []
        for path in [self.amont_edges, self.constrained_edge, self.aval_edges]:
            if isinstance(path, tuple):
                edge = path
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

    def full_n_constrained_path(self):
        return self.filter_constrained_path_for_nodes()

    def __repr__(self):
        return "################################################################\n" \
               "ConstrainedPath = %s \nDetails: (amont: %s, constrained_edge: %s, aval: %s)\n" \
               "################################################################" % (
                   self.full_n_constrained_path(), self.amont_edges, self.constrained_edge, self.aval_edges)

class Structured_Overload_Distribution_Graph:
    """
    Staring from a raw overload distribution graph with color edges, this class identifies the underlying path structure in terms of constrained path, loop paths and hub nodes
    """
    def __init__(self,g):
        """
        Parameters
        ----------

        g: :class:`nx:MultiDiGraph`
            a raw graph from OverflowGraph

        """
        self.g_init=g
        self.g_without_pos_edges = delete_color_edges(self.g_init, "red") #graph without loop path that have positive/red-coloured weight edges
        self.g_only_blue_components = delete_color_edges(self.g_without_pos_edges, "gray") #graph with only negative/blue-coloured weight edges
        self.g_without_constrained_edge = delete_color_edges(self.g_init, "black")
        self.g_without_gray_and_c_edge = delete_color_edges(self.g_without_constrained_edge, "gray")
        self.g_only_red_components = delete_color_edges(self.g_without_gray_and_c_edge, "blue")#graph with only loop path that have positive/red-coloured weight edges

        self.constrained_path= self.find_constrained_path() #constrained path that contains the constrained edges and their connected component of blue edges
        self.type=""#
        self.red_loops = self.find_loops() #parallel path to the constrained path on which flow can be rerouted
        self.hubs = self.find_hubs() #specific nodes at substations connecting loop paths to constrained path. This is where flow can be most easily rerouted

    def get_amont_blue_edges(self, g, node):
        """
        From a given node, get blue edges (with negative overflow redispatch) that are above this node

        Parameters
        ----------

        g: :class:`nx:MultiDiGraph`
            an overflow redispatch networkx graph

        node: int
            node of interest

        Returns
        ----------

        res: ``array`` int
            ordered list of edges

        """
        res = []
        for e in nx.edge_dfs(g, node, orientation="reverse"):
            if g.edges[(e[0], e[1],e[2])]["color"] == "blue":
                res.append((e[0], e[1],e[2]))
        return res

    def get_aval_blue_edges(self, g, node):
        """
        From a given node, get blue edges (with negative overflow redispatch) that are after this node

        Parameters
        ----------

        g: :class:`nx:MultiDiGraph`
            an overflow redispatch networkx graph

        node: int
            node of interest

        Returns
        ----------

        res: ``array`` int
            ordered list of edges

        """
        res = []
        # print("debug AlphaDeesp get aval blue edges")
        # print(list(nx.edge_dfs(g, node, orientation="original")))
        for e in nx.edge_dfs(g, node, orientation="original"):
            if g.edges[(e[0], e[1],e[2])]["color"] == "blue":
                res.append((e[0], e[1],e[2]))
        return res


    def find_hubs(self):
        """
        "A hub (carrefour_electrique) has a constrained_path and positiv reports"

        Returns
        ----------

        res: list int
            a list of nodes that are detected as hubs
        """
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
        and walk in reverse for loops and parallel path returns a dict with all data

        Returns
        ----------

        res: pd.DataFrame
            a dataframe with rows representing each detected path, with column attibutes "Source, Target, Path" with Path representing a list of nodes
        """

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
                    node_source=c_path_n[i]
                    node_target = c_path_n[j]
                    if (node_source in g.nodes) and  (node_target in g.nodes):
                        try:
                            res = nx.all_simple_paths(g, node_source, node_target)#nx.all_shortest_paths(g, node_source, node_target)
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
        """Find and return the constrained path

         Returns
        ----------

        res: :class:`ConstrainedPath`
            a constrained path object
        """
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

def delete_color_edges(_g, edge_color):
    """
    Returns a copy of a graph without edges of a given color. Gray for instance, with values below a threshold of significance

    From a given node, get blue edges (with negative overflow redispatch) that are above this node

    Parameters
    ----------

    _g: :class:`nx:MultiDiGraph`
        an overflow redispatch networkx graph

    edge_color: ``str``
        color of edges to delete from graoh

    Returns
    ----------

    res: :class:`nx:MultiDiGraph`
        the graph without edges for the targeted color

    """
    g = _g.copy()

    TargetColor_edges = []
    i = 1
    for u, v,idx, color in g.edges(data="color",keys=True):
        if color == edge_color:
            TargetColor_edges.append((i, (u, v,idx)))
        i += 1

    # delete from graph gray edges
    # this extracts the (u,v) from pos_edges
    if TargetColor_edges:
        g.remove_edges_from(list(zip(*TargetColor_edges))[1])
        g.remove_nodes_from(list(nx.isolates(g)))
    return g