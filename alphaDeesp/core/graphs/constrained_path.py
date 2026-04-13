"""ConstrainedPath: main path carrying overloaded lines."""

import logging
from typing import Any, List

from alphaDeesp.core.graphs.graph_utils import from_edges_get_nodes

logger = logging.getLogger(__name__)


class ConstrainedPath:

    """
    A connected path of lines that includes the overloaded lines and for which the overflow redipsatch is negative.
    This can be regarded as the main path through which the flows get in and out of the overloaded lines.
    Hence flows should be pushed on other path to relieve the overloads.
    """

    def __init__(self, amont_edges: List[Any], constrained_edge: Any, aval_edges: List[Any]) -> None:
        logger.debug("Constrained path created")
        self.amont_edges = amont_edges #lines which flow goes into the overloaded lines
        self.constrained_edge = constrained_edge #overloaded lines
        self.aval_edges = aval_edges #lines which flow comes from the overloaded lines

    def n_amont(self) -> list:
        """Returns a list of nodes that are in "amont" """
        return from_edges_get_nodes(self.amont_edges, "amont", self.constrained_edge)

    def n_aval(self) -> List[Any]:
        """Returns a list of nodes that are in "aval" """
        return from_edges_get_nodes(self.aval_edges, "aval", self.constrained_edge)

    def e_amont(self) -> List[Any]:
        """Returns a list of edges that are in "amont" """
        return self.amont_edges

    def e_aval(self) -> List[Any]:
        """Returns a list of edges that are in "aval" """
        return self.aval_edges

    def filter_constrained_path_for_nodes(self) -> List[Any]:
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

    def full_n_constrained_path(self) -> List[Any]:
        return self.filter_constrained_path_for_nodes()

    def __repr__(self) -> str:
        return "################################################################\n" \
               "ConstrainedPath = %s \nDetails: (amont: %s, constrained_edge: %s, aval: %s)\n" \
               "################################################################" % (
                   self.full_n_constrained_path(), self.amont_edges, self.constrained_edge, self.aval_edges)
