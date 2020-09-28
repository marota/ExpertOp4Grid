"""Class representing a constrained path"""


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


def from_edges_get_nodes(edges, amont_or_aval: str, constrained_edge):
    """edges is a list of tuples"""
    if edges:
        nodes = []
        for e in edges:
            for node in e:
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
