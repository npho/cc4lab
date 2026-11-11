"""Trivial blue policies for the feasibility spike.

Policy signature used across cc4lab:

    policy(keys (N_BLUE, 2), obs (N_BLUE, OBS_DIM), masks (N_BLUE, ACTION_DIM)) -> actions (N_BLUE,) int32

``keys`` carries one PRNG key per agent so swapping one agent's policy leaves
the other agents' random streams untouched (needed for common random numbers).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxborg.actions.encoding import BLUE_MONITOR, BLUE_SLEEP


def random_valid(keys: jax.Array, obs: jax.Array, masks: jax.Array) -> jax.Array:
    """Uniform over each agent's valid actions."""
    del obs
    logits = jnp.where(masks, 0.0, -jnp.inf)
    return jax.vmap(jax.random.categorical)(keys, logits).astype(jnp.int32)


def always_sleep(keys: jax.Array, obs: jax.Array, masks: jax.Array) -> jax.Array:
    del keys, obs
    return jnp.full(masks.shape[0], BLUE_SLEEP, dtype=jnp.int32)


def always_monitor(keys: jax.Array, obs: jax.Array, masks: jax.Array) -> jax.Array:
    del keys, obs
    return jnp.full(masks.shape[0], BLUE_MONITOR, dtype=jnp.int32)
