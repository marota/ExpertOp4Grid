# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0
"""Pure-numpy scoring helpers for score_changes_between_two_observations.

Kept in a separate module so they can be unit-tested without a grid2op
installation (Grid2opSimulation.py pulls in grid2op at import time).
"""

from typing import Any, List, Tuple, Union

import numpy as np


def _compute_overload_counts(
    old_rho: Any,
    new_rho: Any,
) -> Tuple[int, int]:
    old_count = sum(1 for r in old_rho if r > 1.0)
    new_count = sum(1 for r in new_rho if r > 1.0)
    return old_count, new_count


def _compute_line_flags(
    old_rho: Any,
    new_rho: Any,
    ltc: List[int],
) -> Tuple[Any, Any, Any, Any]:
    worsened, relieved_30pct, relieved, created = [], [], [], []
    for line_id, (old, new) in enumerate(zip(old_rho, new_rho)):
        worsened.append(1 if (new > 1.05 * old) and (new > 1.0) else 0)

        if (old > 1.0) and (line_id in ltc):
            pct = (old - new) * 100 / (old - 1.0)
            relieved_30pct.append(1 if pct > 30.0 else 0)
        else:
            relieved_30pct.append(0)

        relieved.append(1 if (old > 1.0 > new) and (line_id in ltc) else 0)
        created.append(1 if old < 1.0 < new else 0)

    return (
        np.array(worsened),
        np.array(relieved_30pct),
        np.array(relieved),
        np.array(created),
    )


def _compute_cut_load_percent(old_obs: Any, new_obs: Any) -> float:
    total_prod = np.nansum(new_obs.prod_p)
    losses = np.nansum(np.abs(new_obs.p_or + new_obs.p_ex))
    expected_load = total_prod - losses
    return (np.sum(old_obs.load_p) - expected_load) / np.sum(old_obs.load_p)


def _resolve_score(
    old_count: int,
    new_count: int,
    worsened: Any,
    relieved_30pct: Any,
    relieved: Any,
    created: Any,
    cascading: Any,
    cut_load_percent: float,
) -> Union[int, float]:
    if old_count == 0:
        return float('nan')
    if cut_load_percent > 0.01:
        return 0
    if (relieved == 1).any() and (
        (created == 1).any() or (worsened == 1).any() or cascading.any()
    ):
        return 1
    if old_count > 0 and new_count == 0:
        return 4
    if (relieved == 1).any() and (worsened == 0).all():
        return 3
    if (relieved_30pct == 1).any() and (worsened == 0).all():
        return 2
    if (relieved_30pct == 0).all() or (worsened == 1).any():
        return 0
    raise ValueError("Probleme with Scoring")
