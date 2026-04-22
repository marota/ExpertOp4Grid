"""Unit tests for :class:`ConstrainedPath`."""

from alphaDeesp.core.graphsAndPaths import ConstrainedPath


class TestConstrainedPath:

    def test_e_amont_returns_amont_edges(self):
        amont = [("A", "B", 0), ("B", "C", 0)]
        cp = ConstrainedPath(amont, ("C", "D", 0), [("D", "E", 0)])
        assert cp.e_amont() == amont

    def test_e_aval_returns_aval_edges(self):
        aval = [("D", "E", 0), ("E", "F", 0)]
        cp = ConstrainedPath([("A", "B", 0)], ("C", "D", 0), aval)
        assert cp.e_aval() == aval

    def test_n_amont_with_edges(self):
        amont = [("A", "B", 0), ("B", "C", 0)]
        cp = ConstrainedPath(amont, ("C", "D", 0), [])
        assert cp.n_amont() == ["A", "B", "C"]

    def test_n_aval_with_edges(self):
        aval = [("D", "E", 0), ("E", "F", 0)]
        cp = ConstrainedPath([], ("C", "D", 0), aval)
        assert cp.n_aval() == ["D", "E", "F"]

    def test_n_amont_empty_returns_constrained_source(self):
        cp = ConstrainedPath([], ("X", "Y", 0), [])
        assert cp.n_amont() == ["X"]

    def test_n_aval_empty_returns_constrained_target(self):
        cp = ConstrainedPath([], ("X", "Y", 0), [])
        assert cp.n_aval() == ["Y"]

    def test_full_n_constrained_path(self):
        cp = ConstrainedPath([("A", "B", 0)], ("B", "C", 0), [("C", "D", 0)])
        assert cp.full_n_constrained_path() == ["A", "B", "C", "D"]

    def test_full_n_constrained_path_deduplicates(self):
        # C appears in both amont chain and aval start, should not duplicate
        cp = ConstrainedPath(
            [("A", "B", 0), ("B", "C", 0)],
            ("C", "D", 0),
            [("D", "C", 0)])
        assert cp.full_n_constrained_path().count("C") == 1

    def test_repr(self):
        cp = ConstrainedPath([("A", "B", 0)], ("B", "C", 0), [("C", "D", 0)])
        repr_str = repr(cp)
        assert "ConstrainedPath" in repr_str
        assert "A" in repr_str
