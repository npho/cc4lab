"""Exact Shapley values from the ``2**n`` coalition values.

    phi_i = sum over S not containing i of  |S|! (n-|S|-1)! / n!  *  (v(S + i) - v(S))

Coalition indexing is documented in ``cc4lab.attribution``. The result has the
player on axis 0; trailing batch axes of ``values`` are carried through, and the
function is jit/vmap friendly (all index arrays are static numpy constants).

Efficiency holds by construction: ``phi.sum(0) == v[full] - v[empty]``.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from cc4lab.attribution import n_players


def shapley_values(values) -> jnp.ndarray:
    """``values`` shape ``(2**n, *batch)`` -> Shapley values shape ``(n, *batch)``."""
    values = jnp.asarray(values)
    n = n_players(values)
    coalitions = np.arange(2**n)
    size = np.array([bin(c).count("1") for c in coalitions])
    weight = np.array([math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n) for s in range(n)])
    phi = []
    for i in range(n):
        without = coalitions[(coalitions >> i) & 1 == 0]  # coalitions S not containing i
        marginal = values[without | (1 << i)] - values[without]
        phi.append(jnp.tensordot(jnp.asarray(weight[size[without]], dtype=values.dtype), marginal, axes=(0, 0)))
    return jnp.stack(phi)
