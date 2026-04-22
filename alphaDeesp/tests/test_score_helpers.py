"""Unit tests for the module-level scoring helpers in Grid2opSimulation.

Tests cover all four helpers extracted from score_changes_between_two_observations:
  _compute_overload_counts, _compute_line_flags,
  _compute_cut_load_percent, _resolve_score.

No grid2op installation is required — the helpers only use plain Python /
numpy; observation objects are replaced by simple namespaces.
"""

import math
import numpy as np
import pytest
from types import SimpleNamespace

from alphaDeesp.core.grid2op.scoring import (
    _compute_overload_counts,
    _compute_line_flags,
    _compute_cut_load_percent,
    _resolve_score,
)


# ──────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────

def _zeros(n):
    return np.zeros(n)

def _ones(n):
    return np.ones(n)

def _arr(*vals):
    return np.array(vals, dtype=float)


# ──────────────────────────────────────────────────────────────────────
# _compute_overload_counts
# ──────────────────────────────────────────────────────────────────────

class TestComputeOverloadCounts:

    def test_no_overloads(self):
        old, new = _arr(0.5, 0.8, 0.99), _arr(0.4, 0.7, 0.9)
        assert _compute_overload_counts(old, new) == (0, 0)

    def test_all_overloaded(self):
        old, new = _arr(1.1, 1.2, 1.5), _arr(1.05, 1.3, 1.6)
        assert _compute_overload_counts(old, new) == (3, 3)

    def test_overload_cleared(self):
        old, new = _arr(1.2, 0.8), _arr(0.9, 0.8)
        assert _compute_overload_counts(old, new) == (1, 0)

    def test_new_overload_created(self):
        old, new = _arr(0.9, 0.5), _arr(0.9, 1.1)
        assert _compute_overload_counts(old, new) == (0, 1)

    def test_exactly_at_threshold_not_counted(self):
        # rho == 1.0 is NOT an overload (condition is > 1.0)
        old, new = _arr(1.0, 1.0), _arr(1.0, 1.0)
        assert _compute_overload_counts(old, new) == (0, 0)

    def test_mixed(self):
        old = _arr(1.5, 0.5, 1.1, 0.9)
        new = _arr(0.9, 1.2, 1.2, 0.8)
        old_n, new_n = _compute_overload_counts(old, new)
        assert old_n == 2
        assert new_n == 2

    def test_single_element(self):
        assert _compute_overload_counts(_arr(1.01), _arr(0.5)) == (1, 0)

    def test_empty_arrays(self):
        assert _compute_overload_counts([], []) == (0, 0)


# ──────────────────────────────────────────────────────────────────────
# _compute_line_flags
# ──────────────────────────────────────────────────────────────────────

class TestComputeLineFlags:

    def _flags(self, old, new, ltc):
        return _compute_line_flags(np.array(old), np.array(new), ltc)

    # worsened ---------------------------------------------------------

    def test_worsened_when_new_exceeds_105pct_of_old_and_above_1(self):
        worsened, _, _, _ = self._flags([0.8], [1.1], ltc=[])
        # new(1.1) > 1.05*old(0.8=0.84) and new > 1.0 → worsened
        assert worsened[0] == 1

    def test_not_worsened_when_new_below_1(self):
        worsened, _, _, _ = self._flags([0.8], [0.9], ltc=[])
        assert worsened[0] == 0

    def test_not_worsened_when_increase_within_5pct(self):
        worsened, _, _, _ = self._flags([1.1], [1.15], ltc=[])
        # new(1.15) <= 1.05*old(1.155) → not worsened
        assert worsened[0] == 0

    # relieved_30pct ---------------------------------------------------

    def test_30pct_relieved_on_ltc_line(self):
        # old=1.5 → surcharge=0.5; relieve by 0.2 → 40% > 30%
        _, rp, _, _ = self._flags([1.5], [1.3], ltc=[0])
        assert rp[0] == 1

    def test_30pct_not_reached_on_ltc_line(self):
        # old=1.5 → surcharge=0.5; relieve by 0.1 → 20% < 30%
        _, rp, _, _ = self._flags([1.5], [1.4], ltc=[0])
        assert rp[0] == 0

    def test_30pct_ignored_for_non_ltc_line(self):
        _, rp, _, _ = self._flags([1.5], [1.0], ltc=[99])
        assert rp[0] == 0

    def test_30pct_zero_when_old_not_overloaded(self):
        _, rp, _, _ = self._flags([0.9], [0.5], ltc=[0])
        assert rp[0] == 0

    # relieved ---------------------------------------------------------

    def test_relieved_when_old_overload_cleared_on_ltc(self):
        _, _, relieved, _ = self._flags([1.2], [0.9], ltc=[0])
        assert relieved[0] == 1

    def test_not_relieved_when_new_still_overloaded(self):
        _, _, relieved, _ = self._flags([1.3], [1.05], ltc=[0])
        assert relieved[0] == 0

    def test_not_relieved_when_not_in_ltc(self):
        _, _, relieved, _ = self._flags([1.2], [0.9], ltc=[5])
        assert relieved[0] == 0

    # created ----------------------------------------------------------

    def test_created_when_new_over_1_old_under_1(self):
        _, _, _, created = self._flags([0.8], [1.1], ltc=[])
        assert created[0] == 1

    def test_not_created_when_both_over_1(self):
        _, _, _, created = self._flags([1.1], [1.2], ltc=[])
        assert created[0] == 0

    def test_not_created_when_both_under_1(self):
        _, _, _, created = self._flags([0.5], [0.8], ltc=[])
        assert created[0] == 0

    # multi-line scenario ----------------------------------------------

    def test_multi_line_independence(self):
        old = [1.5, 0.8, 1.1]
        new = [0.9, 1.2, 1.05]
        ltc = [0]
        w, rp, r, c = self._flags(old, new, ltc)
        assert r[0] == 1     # line 0: relieved (on ltc)
        assert c[1] == 1     # line 1: new overload created
        assert w[1] == 1     # line 1: worsened (1.2 > 1.05*0.8=0.84, 1.2>1)
        assert rp[2] == 0    # line 2: not in ltc

    def test_returns_numpy_arrays(self):
        w, rp, r, c = self._flags([1.2], [0.9], ltc=[0])
        for arr in (w, rp, r, c):
            assert isinstance(arr, np.ndarray)


# ──────────────────────────────────────────────────────────────────────
# _compute_cut_load_percent
# ──────────────────────────────────────────────────────────────────────

def _obs(prod_p, p_or, p_ex, load_p):
    """Build a minimal observation namespace."""
    return SimpleNamespace(
        prod_p=np.array(prod_p, dtype=float),
        p_or=np.array(p_or, dtype=float),
        p_ex=np.array(p_ex, dtype=float),
        load_p=np.array(load_p, dtype=float),
    )


class TestComputeCutLoadPercent:

    def test_no_cut_when_balanced(self):
        # prod=100, losses=|10-10|=0 wait: losses=|p_or+p_ex|
        # prod=100, p_or=5, p_ex=-5 → |0| = 0 → expected=100; old_load=100
        old = _obs([], [], [], [100.0])
        new = _obs([100.0], [5.0], [-5.0], [100.0])
        assert _compute_cut_load_percent(old, new) == pytest.approx(0.0)

    def test_positive_cut_percent(self):
        # prod=80, losses=|3+(-3)|=0 → expected=80; old_load=100 → cut=20%
        old = _obs([], [], [], [100.0])
        new = _obs([80.0], [3.0], [-3.0], [])
        result = _compute_cut_load_percent(old, new)
        assert result == pytest.approx(0.20)

    def test_negative_cut_percent_when_more_load_served(self):
        # prod=120, losses=0 → expected=120; old_load=100 → cut=-20%
        old = _obs([], [], [], [100.0])
        new = _obs([120.0], [0.0], [0.0], [])
        result = _compute_cut_load_percent(old, new)
        assert result == pytest.approx(-0.20)

    def test_nan_prod_ignored(self):
        old = _obs([], [], [], [100.0])
        new = _obs([float('nan'), 80.0], [0.0], [0.0], [])
        result = _compute_cut_load_percent(old, new)
        assert result == pytest.approx(0.20)

    def test_losses_reduce_expected_load(self):
        # prod=100, p_or=10, p_ex=0 → losses=|10+0|=10 → expected=90
        # old_load=100 → cut=10%
        old = _obs([], [], [], [100.0])
        new = _obs([100.0], [10.0], [0.0], [])
        result = _compute_cut_load_percent(old, new)
        assert result == pytest.approx(0.10)

    def test_multiple_loads_summed(self):
        old = _obs([], [], [], [40.0, 60.0])
        new = _obs([100.0], [0.0], [0.0], [])
        assert _compute_cut_load_percent(old, new) == pytest.approx(0.0)


# ──────────────────────────────────────────────────────────────────────
# _resolve_score — all 8 branches
# ──────────────────────────────────────────────────────────────────────

def _flags_all_zero(n=2):
    z = np.zeros(n, dtype=int)
    return z, z.copy(), z.copy(), z.copy()


def _cascading_false(n=2):
    return np.zeros(n, dtype=bool)


class TestResolveScore:

    def test_nan_when_no_initial_overload(self):
        w, rp, r, c = _flags_all_zero()
        result = _resolve_score(0, 0, w, rp, r, c, _cascading_false(), 0.0)
        assert math.isnan(result)

    def test_score_0_when_load_shed(self):
        w, rp, r, c = _flags_all_zero()
        result = _resolve_score(2, 1, w, rp, r, c, _cascading_false(), 0.05)
        assert result == 0

    def test_score_1_when_relieved_but_new_created(self):
        relieved = np.array([1, 0])
        created  = np.array([0, 1])
        w        = np.zeros(2, dtype=int)
        rp       = np.zeros(2, dtype=int)
        result = _resolve_score(2, 2, w, rp, relieved, created, _cascading_false(), 0.0)
        assert result == 1

    def test_score_1_when_relieved_but_worsened(self):
        relieved = np.array([1, 0])
        worsened = np.array([0, 1])
        rp       = np.zeros(2, dtype=int)
        created  = np.zeros(2, dtype=int)
        result = _resolve_score(2, 1, worsened, rp, relieved, created, _cascading_false(), 0.0)
        assert result == 1

    def test_score_1_when_relieved_but_cascading(self):
        relieved  = np.array([1, 0])
        cascading = np.array([False, True])
        w, rp, _, c = _flags_all_zero()
        result = _resolve_score(2, 1, w, rp, relieved, c, cascading, 0.0)
        assert result == 1

    def test_score_4_when_all_overloads_cleared(self):
        w, rp, r, c = _flags_all_zero()
        result = _resolve_score(3, 0, w, rp, r, c, _cascading_false(), 0.0)
        assert result == 4

    def test_score_3_when_relieved_and_nothing_worsened(self):
        relieved = np.array([1, 0])
        w        = np.zeros(2, dtype=int)
        rp       = np.zeros(2, dtype=int)
        c        = np.zeros(2, dtype=int)
        result = _resolve_score(2, 1, w, rp, relieved, c, _cascading_false(), 0.0)
        assert result == 3

    def test_score_2_when_30pct_relieved_no_worsening(self):
        rp = np.array([0, 1])
        w  = np.zeros(2, dtype=int)
        r  = np.zeros(2, dtype=int)
        c  = np.zeros(2, dtype=int)
        result = _resolve_score(2, 1, w, rp, r, c, _cascading_false(), 0.0)
        assert result == 2

    def test_score_0_when_no_30pct_relief(self):
        w, rp, r, c = _flags_all_zero()
        result = _resolve_score(2, 1, w, rp, r, c, _cascading_false(), 0.0)
        assert result == 0

    def test_score_0_when_worsened_and_no_relief(self):
        worsened = np.array([0, 1])
        rp = np.zeros(2, dtype=int)
        r  = np.zeros(2, dtype=int)
        c  = np.zeros(2, dtype=int)
        result = _resolve_score(2, 1, worsened, rp, r, c, _cascading_false(), 0.0)
        assert result == 0

    def test_load_shed_threshold_is_exclusive(self):
        # cut_load_percent == 0.01 should NOT trigger score 0
        w, rp, r, c = _flags_all_zero()
        result = _resolve_score(0, 0, w, rp, r, c, _cascading_false(), 0.01)
        # old_count=0 → NaN branch fires first
        assert math.isnan(result)

    def test_score_4_takes_priority_over_score_1_no_side_effects(self):
        # new_count==0 with no created/worsened → score 4, not 1
        w        = np.zeros(2, dtype=int)
        rp       = np.zeros(2, dtype=int)
        relieved = np.array([1, 0])
        c        = np.zeros(2, dtype=int)
        result = _resolve_score(2, 0, w, rp, relieved, c, _cascading_false(), 0.0)
        assert result == 4

    def test_load_shed_beats_score_4(self):
        # Even if new_count==0, load shedding → score 0
        w, rp, r, c = _flags_all_zero()
        result = _resolve_score(2, 0, w, rp, r, c, _cascading_false(), 0.05)
        assert result == 0
