import pytest
from alphaDeesp.core.graphsAndPaths import ConstrainedPath


@pytest.fixture
def setup_constrained_path():
    """
    Provides a standard ConstrainedPath object for testing.
    Path: (0 -> 1 -> 2) -> (2 -> 3, key='C') -> (3 -> 4 -> 5)
    """
    amont_edges = [(0, 1, 'A'), (1, 2, 'B')]
    constrained_edge = (2, 3, 'C')
    aval_edges = [(3, 4, 'D'), (4, 5, 'E')]

    return ConstrainedPath(amont_edges, constrained_edge, aval_edges)


def test_n_amont_returns_correct_nodes(setup_constrained_path):
    """
    Tests that n_amont() returns the unique nodes from the amont_edges.
    """
    constrained_path = setup_constrained_path
    amont_nodes = constrained_path.n_amont()

    # Expected nodes are 0, 1, and 2 (from the constrained edge start)
    expected = [0, 1, 2]
    assert sorted(amont_nodes) == sorted(expected)


def test_n_aval_returns_correct_nodes(setup_constrained_path):
    """
    Tests that n_aval() returns the unique nodes from the aval_edges.
    """
    constrained_path = setup_constrained_path
    aval_nodes = constrained_path.n_aval()

    # Expected nodes are 3 (from the constrained edge end), 4, and 5
    expected = [3, 4, 5]
    assert sorted(aval_nodes) == sorted(expected)


def test_e_amont_returns_original_edges(setup_constrained_path):
    """
    Tests that e_amont() returns the original list of amont edges.
    """
    constrained_path = setup_constrained_path
    assert constrained_path.e_amont() == [(0, 1, 'A'), (1, 2, 'B')]


def test_e_aval_returns_original_edges(setup_constrained_path):
    """
    Tests that e_aval() returns the original list of aval edges.
    """
    constrained_path = setup_constrained_path
    assert constrained_path.e_aval() == [(3, 4, 'D'), (4, 5, 'E')]


def test_full_n_constrained_path_returns_ordered_unique_nodes(setup_constrained_path):
    """
    Tests that full_n_constrained_path() returns a list of all unique nodes
    in the entire path.
    """
    constrained_path = setup_constrained_path
    full_path_nodes = constrained_path.full_n_constrained_path()

    # The function builds the list in order, without duplicates.
    expected = [0, 1, 2, 3, 4, 5]
    assert full_path_nodes == expected


def test_n_amont_with_empty_amont_edges():
    """
    Tests n_amont() when amont_edges is empty. It should only return the
    starting node of the constrained edge.
    """
    path = ConstrainedPath([], (1, 2, 'A'), [(2, 3, 'B')])
    assert path.n_amont() == [1]


def test_n_aval_with_empty_aval_edges():
    """
    Tests n_aval() when aval_edges is empty. It should only return the
    ending node of the constrained edge.
    """
    path = ConstrainedPath([(0, 1, 'A')], (1, 2, 'B'), [])
    assert path.n_aval() == [2]


def test_full_path_with_empty_sections():
    """
    Tests full_n_constrained_path() when amont or aval sections are empty.
    """
    # Empty amont
    path1 = ConstrainedPath([], (1, 2, 'A'), [(2, 3, 'B')])
    assert path1.full_n_constrained_path() == [1, 2, 3]

    # Empty aval
    path2 = ConstrainedPath([(0, 1, 'A')], (1, 2, 'B'), [])
    assert path2.full_n_constrained_path() == [0, 1, 2]

    # Both empty
    path3 = ConstrainedPath([], (1, 2, 'A'), [])
    assert path3.full_n_constrained_path() == [1, 2]


def test_repr_returns_formatted_string(setup_constrained_path):
    """
    Tests the __repr__ method for a correct string representation of the object.
    """
    constrained_path = setup_constrained_path
    repr_string = repr(constrained_path)

    assert "ConstrainedPath" in repr_string
    assert "amont: [(0, 1, 'A'), (1, 2, 'B')]" in repr_string
    assert "constrained_edge: (2, 3, 'C')" in repr_string
    assert "aval: [(3, 4, 'D'), (4, 5, 'E')]" in repr_string
    assert "full_n_constrained_path(), self.amont_edges, self.constrained_edge, self.aval_edges" not in repr_string
