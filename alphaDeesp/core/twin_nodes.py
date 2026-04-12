# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of ExpertOp4Grid, an expert system approach to solve flow congestions in power grids

"""
Twin-node id scheme.

When a substation is split into two busbars, AlphaDeesp represents the second
busbar as a "twin node" of the original substation. Historically this was
encoded as ``int("666" + str(substation_id))``, which breaks as soon as a
substation id is >= 1000 (the decoding via ``str(...)[3:]`` strips three
characters unconditionally) and silently collides with any real substation id
whose decimal representation starts with ``666``.

This module replaces that string-prefixing hack with a simple additive offset
scheme. Twin node ids sit above ``TWIN_NODE_OFFSET`` so they are disjoint from
any realistic substation id on the grids AlphaDeesp targets (l2rpn_2019,
rte_case14, l2rpn_wcci_2022, etc.; the largest published grids today have a
few thousand substations).
"""

# Chosen large enough to exceed any substation id on grids AlphaDeesp targets,
# while still fitting comfortably inside a 32-bit integer for downstream tools
# that round-trip node ids through numpy / graphviz.
TWIN_NODE_OFFSET = 10_000_000


def twin_node_id(substation_id) -> int:
    """
    Return the twin-node id associated with ``substation_id``.

    ``substation_id`` must be a non-negative integer (or a value coercible to
    one) strictly below :data:`TWIN_NODE_OFFSET`; otherwise a ``ValueError``
    is raised to prevent silent collisions with the twin-node id space.
    """
    sub_id = int(substation_id)
    if sub_id < 0:
        raise ValueError(
            "twin_node_id expects a non-negative substation id, got %r" % (substation_id,)
        )
    if sub_id >= TWIN_NODE_OFFSET:
        raise ValueError(
            "substation id %r collides with the twin-node id space "
            "(>= TWIN_NODE_OFFSET=%d); bump TWIN_NODE_OFFSET if the target "
            "grid has that many substations" % (substation_id, TWIN_NODE_OFFSET)
        )
    return TWIN_NODE_OFFSET + sub_id


def is_twin_node_id(node_id) -> bool:
    """Return ``True`` iff ``node_id`` is a twin-node id (as built above)."""
    try:
        n = int(node_id)
    except (TypeError, ValueError):
        return False
    return n >= TWIN_NODE_OFFSET


def original_substation_id(twin_id) -> int:
    """
    Return the substation id that a twin node id was built from.

    Raises ``ValueError`` if ``twin_id`` is not in the twin-node id space.
    """
    n = int(twin_id)
    if n < TWIN_NODE_OFFSET:
        raise ValueError(
            "%r is not a twin-node id (expected >= %d)" % (twin_id, TWIN_NODE_OFFSET)
        )
    return n - TWIN_NODE_OFFSET
