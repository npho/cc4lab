"""Per-step rule-violation flags computed from observations and chosen actions.

One rule so far, matching ``policies/rulebook.py``: the firewall matches the
mission phase's communication policy for every subnet the agent observes.

Two flags per agent per step:

* ``firewall_mismatch``: the observation shows at least one observed-subnet
  firewall entry that disagrees with the policy (a *state*, not blame).
* ``firewall_unaddressed``: there is a mismatch and the chosen action is not
  one of the actions that fix some mismatch. Agents that are busy executing a
  multi-tick action cannot act, so pass ``busy`` (``blue_pending_ticks > 0``
  from the simulator state) to exempt them; observations alone do not carry it.

Fixing any one mismatch counts as addressing it: the agent can only do one
thing per step, so this does not demand the rulebook's particular order.
``traffic_action_kind`` classifies a single Block/Allow action against the
same policy (compliant, violating or redundant), for diagnosing what a trained
agent's traffic actions do; it is not yet a rulebook rule.

All functions are jit/vmap friendly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxborg.actions.action_defs import (
    BLUE_ALLOW_TRAFFIC_END,
    BLUE_ALLOW_TRAFFIC_START,
    BLUE_BLOCK_TRAFFIC_END,
    BLUE_BLOCK_TRAFFIC_START,
)

from cc4lab.env import ACTION_DIM
from cc4lab.policies.rulebook import firewall_candidates, firewall_cells

TRAFFIC_KINDS = ("compliant", "violating", "redundant")
"""Kinds returned by ``traffic_action_kind`` (by index)."""


def firewall_violations(obs: jax.Array, actions: jax.Array, masks: jax.Array | None = None, busy: jax.Array | None = None):
    """``obs (N_BLUE, OBS_DIM)``, ``actions (N_BLUE,)`` -> dict of ``(N_BLUE,)`` bool arrays.

    ``masks`` (``(N_BLUE, ACTION_DIM)``) restricts "fixing actions" to ones the
    agent could legally take; by default every action is considered legal, which
    only matters if an agent has a mismatch it has no valid action for.
    """
    if masks is None:
        masks = jnp.ones((obs.shape[0], ACTION_DIM), dtype=bool)

    def per_agent(o, m, a):
        action, needs_fix = firewall_candidates(o, m)
        return needs_fix.any(), (needs_fix & (action == a)).any()

    mismatch, addresses = jax.vmap(per_agent)(obs, masks, actions)
    unaddressed = mismatch & ~addresses
    if busy is not None:
        unaddressed = unaddressed & ~busy
    return {"firewall_mismatch": mismatch, "firewall_unaddressed": unaddressed}


def traffic_action_kind(obs: jax.Array, action: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """One agent's ``obs (OBS_DIM,)`` and ``action`` -> ``(is_block, is_allow, kind)``.

    ``kind`` indexes ``TRAFFIC_KINDS``, or is -1 if the action is not a traffic
    action on a real observed link. A Block is *compliant* if the phase's policy
    says the link should be blocked and it is open, *violating* if the policy
    says it should stay open and it is open, *redundant* if it is already
    blocked. An Allow is compliant if it opens a link the policy wants open,
    violating if it opens one the policy wants blocked, redundant if the link
    is already open.
    """
    index, real, blocked, comms = firewall_cells(obs)
    is_block = (action >= BLUE_BLOCK_TRAFFIC_START) & (action < BLUE_BLOCK_TRAFFIC_END)
    is_allow = (action >= BLUE_ALLOW_TRAFFIC_START) & (action < BLUE_ALLOW_TRAFFIC_END)
    idx = jnp.where(is_block, action - BLUE_BLOCK_TRAFFIC_START, action - BLUE_ALLOW_TRAFFIC_START)
    cell = real & (index == idx)
    hit = cell.any()
    should_block = (cell & comms).any()
    now_blocked = (cell & blocked).any()
    block_kind = jnp.where(now_blocked, 2, jnp.where(should_block, 0, 1))
    allow_kind = jnp.where(~now_blocked, 2, jnp.where(should_block, 1, 0))
    kind = jnp.where(hit & is_block, block_kind, jnp.where(hit & is_allow, allow_kind, -1))
    return is_block, is_allow, kind
