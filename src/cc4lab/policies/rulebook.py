"""JAX-traceable rulebook reference agent (obs -> action).

First rule only: **the firewall matches the mission phase's communication
policy** for every subnet the agent observes. Everything else is a no-op
(``default_action``). Further rules are the team's decision (see HANDOFF).

How the rule reads the observation (``jaxborg.observations``): after the mission
phase there is one block per observed subnet, each holding a one-hot of the
subnet, a ``blocked`` row (firewall: which source subnets are blocked from
reaching it) and a ``comms`` row (the policy: which source subnets *should* be
blocked), both in CybORG subnet order. A mismatch is fixed with
``BlockTrafficZone`` when the policy says blocked but the firewall is open, and
``AllowTrafficZone`` in the opposite case. The diagonal (a subnet toward itself)
is excluded: the policy marks it "not connected" everywhere.

Action indices follow ``jaxborg.actions.action_defs.encode_blue_action``. The
function is a pure function of ``obs`` and ``masks`` (no PRNG use, no simulator
state), so it works under jit/vmap/scan. It fixes one mismatch per step, the
first in (observed slot, CybORG column) order, and only if the action is valid
under ``masks``. It never emits Restore, so the busy-agent action-cost caveat in
NOTES.md does not apply to it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxborg.actions.action_defs import BLUE_ALLOW_TRAFFIC_START, BLUE_BLOCK_TRAFFIC_START, BLUE_SLEEP
from jaxborg.constants import BLUE_MAX_OBSERVED_SUBNETS, NUM_SUBNETS
from jaxborg.observations import CYBORG_POS_TO_JAX_ID, SUBNET_BLOCK_SIZE

_SLOTS = BLUE_MAX_OBSERVED_SUBNETS


def firewall_candidates(obs: jax.Array, masks: jax.Array) -> tuple[jax.Array, jax.Array]:
    """For one agent: the fixing action for every (observed slot, CybORG column), and where a fix is needed.

    ``obs`` is ``(OBS_DIM,)``, ``masks`` is ``(ACTION_DIM,)``. Returns
    ``(action, needs_fix)``, both ``(slots, NUM_SUBNETS)``; ``needs_fix`` is True
    only where the firewall disagrees with the policy, the slot is real, the
    column is not the subnet itself, and the action is valid under ``masks``.
    """
    blocks = obs[1 : 1 + _SLOTS * SUBNET_BLOCK_SIZE].reshape(_SLOTS, SUBNET_BLOCK_SIZE)
    n = NUM_SUBNETS
    onehot, blocked, comms = blocks[:, :n], blocks[:, n : 2 * n] > 0.5, blocks[:, 2 * n : 3 * n] > 0.5
    active = onehot.sum(-1) > 0.5
    to_jax = jnp.asarray(CYBORG_POS_TO_JAX_ID)
    dst = to_jax[jnp.argmax(onehot, axis=-1)][:, None]  # (slots, 1) absolute subnet ids
    src = to_jax[None, :]  # (1, n): column j <-> absolute subnet id
    src_offset = jnp.where(src < dst, src, src - 1)
    index = src_offset * _SLOTS + jnp.arange(_SLOTS)[:, None]
    action = jnp.where(comms, BLUE_BLOCK_TRAFFIC_START + index, BLUE_ALLOW_TRAFFIC_START + index)
    needs_fix = active[:, None] & (src != dst) & (blocked != comms) & (masks[action] > 0)
    return action, needs_fix


def _agent_action(obs: jax.Array, masks: jax.Array, default_action: int) -> jax.Array:
    action, needs_fix = firewall_candidates(obs, masks)
    flat = needs_fix.reshape(-1)
    return jnp.where(flat.any(), action.reshape(-1)[jnp.argmax(flat)], default_action).astype(jnp.int32)


def make_firewall_rulebook(default_action: int = BLUE_SLEEP):
    """Rulebook policy with a chosen no-op action (Sleep by default; Monitor is the other natural choice)."""

    def policy(keys: jax.Array, obs: jax.Array, masks: jax.Array) -> jax.Array:
        del keys  # deterministic
        return jax.vmap(lambda o, m: _agent_action(o, m, default_action))(obs, masks)

    return policy


firewall_rulebook = make_firewall_rulebook()
