"""Leave-one-out baseline: all-actual vs agent i swapped to the reference policy.

    loo_i = v(full) - v(full without i)

Only the full coalition and its ``n`` single-agent removals are needed (6 of the
32 coalitions for n = 5). Unlike Shapley it cannot see redundancy: if any two of
three agents suffice, removing one changes nothing, so every agent scores 0.
Coalition indexing is documented in ``cc4lab.attribution``.
"""

from __future__ import annotations

import jax.numpy as jnp

from cc4lab.attribution import n_players


def loo_values(values) -> jnp.ndarray:
    """``values`` shape ``(2**n, *batch)`` -> leave-one-out values shape ``(n, *batch)``."""
    values = jnp.asarray(values)
    n = n_players(values)
    full = 2**n - 1
    return jnp.stack([values[full] - values[full ^ (1 << i)] for i in range(n)])
