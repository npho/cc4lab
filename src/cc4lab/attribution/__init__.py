"""Attribution methods over the 5 blue agents.

Coalition indexing convention (shared by ``shapley``, ``loo`` and, in
Milestone 2, ``rollout``): coalition ``c`` in ``[0, 2**n)`` contains agent ``i``
iff bit ``i`` of ``c`` is set. So index 0 is the empty coalition (every agent
follows the reference policy) and index ``2**n - 1`` is the full coalition
(every agent follows its actual policy). Value arrays have the coalition on
axis 0 and may carry any trailing batch axes (seeds, branch points, ...).
"""

from __future__ import annotations

import numpy as np


def coalition_masks(n: int = 5) -> np.ndarray:
    """``(2**n, n)`` bool array; ``masks[c, i]`` is True iff agent ``i`` is in coalition ``c``."""
    return ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(bool)


def n_players(values) -> int:
    """Number of players implied by a coalition-value array's leading axis (must be a power of two)."""
    size = values.shape[0]
    n = size.bit_length() - 1
    if size < 2 or 2**n != size:
        raise ValueError(f"leading axis must have length 2**n with n >= 1, got {size}")
    return n
