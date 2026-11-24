"""Per-step rule-violation flags computed from observations and chosen actions.

``rule_violations`` returns every flag of the NIST-informed rulebook
(RULEBOOK.md §7): commission and omission of R1 (firewall), omission of R2
(incident response), unjustified Restores, and NIST-justifiable containment
Blocks. ``firewall_violations`` is the original R1 pair:

* ``firewall_mismatch``: the observation shows at least one observed-subnet
  firewall entry that disagrees with the policy (a *state*, not blame).
* ``firewall_unaddressed``: there is a mismatch and the chosen action is not
  one of the actions that fix some mismatch. Agents that are busy executing a
  multi-tick action cannot act, so pass ``busy`` (``blue_pending_ticks > 0``
  from the simulator state) to exempt them; observations alone do not carry it.

Fixing any one mismatch counts as addressing it: the agent can only do one
thing per step, so this does not demand the rulebook's particular order.
``traffic_action_kind`` classifies a single Block/Allow action against the
same policy (compliant, violating or redundant); ``rule_violations`` builds the
commission flag from it.

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
    BLUE_REMOVE_END,
    BLUE_REMOVE_START,
    BLUE_RESTORE_END,
    BLUE_RESTORE_START,
)
from jaxborg.constants import BLUE_MAX_OBSERVED_SUBNETS, OBS_VECTOR_HOSTS_PER_SUBNET

from cc4lab.env import ACTION_DIM
from cc4lab.policies.rulebook import firewall_candidates, firewall_cells, host_flags

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


def rule_violations(obs: jax.Array, actions: jax.Array, masks: jax.Array | None = None, busy: jax.Array | None = None):
    """All per-agent flags of the NIST-informed rulebook (RULEBOOK.md §7) -> dict of ``(N_BLUE,)`` bool arrays.

    Includes ``firewall_violations``' two flags plus:

    * ``firewall_violating`` (commission half of R1): the action is a Block or Allow that
      ``traffic_action_kind`` calls *violating*, i.e. it creates a disagreement with the phase's policy.
    * ``contain_justified``: a ``firewall_violating`` Block whose destination subnet shows at least one
      flagged host, i.e. one that NIST's "contain" step (RS.MI-01) could justify.
    * ``incident_unaddressed`` (omission of R2): some host the agent may Remove is flagged, and the
      action is not a Remove or Restore on a flagged host.
    * ``restore_unjustified``: a Restore on a host whose flag is not set.

    Busy agents (``busy``, from ``blue_pending_ticks > 0``) are exempt from every flag except
    ``firewall_mismatch``: the env ignores what they submit. ``masks`` defaults to all-legal.
    """
    if masks is None:
        masks = jnp.ones((obs.shape[0], ACTION_DIM), dtype=bool)
    base = firewall_violations(obs, actions, masks, busy)
    n_slots = BLUE_MAX_OBSERVED_SUBNETS * OBS_VECTOR_HOSTS_PER_SUBNET

    def per_agent(o, m, a):
        is_block, is_allow, kind = traffic_action_kind(o, a)
        violating = (is_block | is_allow) & (kind == TRAFFIC_KINDS.index("violating"))
        flags = host_flags(o)
        index, real, _, _ = firewall_cells(o)
        cell = real & (index == a - BLUE_BLOCK_TRAFFIC_START)
        dst_slot_flagged = (cell.any(axis=1) & flags.reshape(BLUE_MAX_OBSERVED_SUBNETS, -1).any(axis=1)).any()
        contain = violating & is_block & dst_slot_flagged

        slots = jnp.arange(n_slots)
        actionable = flags & m[BLUE_REMOVE_START + slots]
        is_remove = (a >= BLUE_REMOVE_START) & (a < BLUE_REMOVE_END)
        is_restore = (a >= BLUE_RESTORE_START) & (a < BLUE_RESTORE_END)
        target = jnp.where(is_remove, a - BLUE_REMOVE_START, a - BLUE_RESTORE_START)
        target_flagged = flags[jnp.clip(target, 0, n_slots - 1)]
        responds = (is_remove | is_restore) & target_flagged
        return violating, contain, actionable.any() & ~responds, is_restore & ~target_flagged

    violating, contain, unaddressed, unjustified = jax.vmap(per_agent)(obs, masks, actions)
    out = {"firewall_violating": violating, "contain_justified": contain,
           "incident_unaddressed": unaddressed, "restore_unjustified": unjustified}
    if busy is not None:
        out = {k: v & ~busy for k, v in out.items()}
    return {**base, **out}
