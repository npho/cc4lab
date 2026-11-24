"""JAX-traceable rulebook reference agents (obs -> action).

``make_rulebook(RulebookConfig)`` is the NIST-informed reference specified in
RULEBOOK.md: R1 Firewall, R2 Remove, optional R4 Restore, an idle action, and the
ladder of configurations in ``RUNGS``. ``firewall_rulebook`` /
``make_firewall_rulebook`` are the original R1-only agent, kept for compatibility
(``make_rulebook(RulebookConfig(idle=...))`` returns identical actions).

R1, **the firewall matches the mission phase's communication policy** for every
subnet the agent observes, is described below.

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

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxborg.actions.action_defs import (
    BLUE_ALLOW_TRAFFIC_START,
    BLUE_BLOCK_TRAFFIC_START,
    BLUE_MONITOR,
    BLUE_REMOVE_START,
    BLUE_RESTORE_START,
    BLUE_SLEEP,
)
from jaxborg.constants import BLUE_MAX_OBSERVED_SUBNETS, NUM_SUBNETS, OBS_VECTOR_HOSTS_PER_SUBNET
from jaxborg.observations import CYBORG_POS_TO_JAX_ID, SUBNET_BLOCK_SIZE

_SLOTS = BLUE_MAX_OBSERVED_SUBNETS


def firewall_cells(obs: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Decode one agent's ``obs`` into its firewall cells, each ``(slots, NUM_SUBNETS)``.

    Returns ``(index, real, blocked, comms)``: the traffic-action offset for the
    cell (Block = ``BLUE_BLOCK_TRAFFIC_START + index``, Allow likewise), whether
    it is a real link (observed slot, not the subnet itself), the firewall bit
    and the policy bit ("should be blocked"). Column j is CybORG position j.
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
    real = active[:, None] & (src != dst)
    return index, real, blocked, comms


def firewall_candidates(obs: jax.Array, masks: jax.Array) -> tuple[jax.Array, jax.Array]:
    """For one agent: the fixing action for every (observed slot, CybORG column), and where a fix is needed.

    ``obs`` is ``(OBS_DIM,)``, ``masks`` is ``(ACTION_DIM,)``. Returns
    ``(action, needs_fix)``, both ``(slots, NUM_SUBNETS)``; ``needs_fix`` is True
    only where the firewall disagrees with the policy, the slot is real, the
    column is not the subnet itself, and the action is valid under ``masks``.
    """
    index, real, blocked, comms = firewall_cells(obs)
    action = jnp.where(comms, BLUE_BLOCK_TRAFFIC_START + index, BLUE_ALLOW_TRAFFIC_START + index)
    needs_fix = real & (blocked != comms) & (masks[action] > 0)
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


# --- NIST-informed rulebook (RULEBOOK.md §4-§5) ------------------------------------------------------

_H = OBS_VECTOR_HOSTS_PER_SUBNET
_IDLE = {"sleep": BLUE_SLEEP, "monitor": BLUE_MONITOR}


@dataclass(frozen=True)
class RulebookConfig:
    """Which rules the reference agent uses (RULEBOOK.md §5.1). R3 Contain is not implemented (§12).

    ``firewall`` R1: the firewall matches the phase's communication policy (PR.IR-01).
    ``remove`` R2: Remove one flagged host, chosen uniformly at random (RS.MI-02).
    ``restore_prob`` R4: with this probability, Restore that host instead (RC.RP-02); 0 disables it.
      Use ``cost_mode="busy_corrected"`` whenever it is > 0 (see ``needs_busy_corrected``).
    ``idle`` action when no rule fires: ``"sleep"`` or ``"monitor"``.
    ``priority`` ``"protect_first"`` (R1 before R2/R4) or ``"respond_first"`` (ablation only).
    """

    firewall: bool = True
    remove: bool = False
    restore_prob: float = 0.0
    idle: str = "sleep"
    priority: str = "protect_first"

    @property
    def needs_busy_corrected(self) -> bool:
        """True if the rulebook can emit Restore, whose repeat charges while busy distort ``cost_mode="raw"``."""
        return self.restore_prob > 0


RUNGS: dict[str, RulebookConfig] = {
    "L1": RulebookConfig(),
    "L2": RulebookConfig(remove=True),
    "L3": RulebookConfig(remove=True, idle="monitor"),
    "L5": RulebookConfig(remove=True, restore_prob=0.25),
    "L2-respond-first": RulebookConfig(remove=True, priority="respond_first"),
}
"""The reference ladder (RULEBOOK.md §6.1). L0 is ``trivial.always_sleep``; L4 (Contain) was dropped."""


def host_flags(obs: jax.Array) -> jax.Array:
    """``(slots * H,)`` bool: the observation's ``malicious_processes`` flag per host slot ``r * H + k``.

    Slot ``r * H + k`` is the same host as action ``BLUE_{REMOVE,RESTORE}_START + r * H + k``: both index
    ``const.obs_host_map[subnet, k]`` (``jaxborg.observations``, ``actions.encoding._slot_to_global_host``).
    """
    blocks = obs[1 : 1 + _SLOTS * SUBNET_BLOCK_SIZE].reshape(_SLOTS, SUBNET_BLOCK_SIZE)
    return (blocks[:, 3 * NUM_SUBNETS : 3 * NUM_SUBNETS + _H] > 0.5).reshape(-1)


def _respond(key: jax.Array, obs: jax.Array, masks: jax.Array, restore_prob: float):
    """R2/R4 for one agent: ``(fires, action)``."""
    flags = host_flags(obs)
    slots = jnp.arange(_SLOTS * _H)
    can_remove = flags & masks[BLUE_REMOVE_START + slots]
    can_restore = flags & masks[BLUE_RESTORE_START + slots]
    k_pick, k_escalate = jax.random.split(key)
    remove_slot = jax.random.categorical(k_pick, jnp.where(can_remove, 0.0, -jnp.inf))
    action = BLUE_REMOVE_START + remove_slot
    fires = can_remove.any()
    if restore_prob > 0:
        restore_slot = jax.random.categorical(k_pick, jnp.where(can_restore, 0.0, -jnp.inf))
        escalate = (jax.random.uniform(k_escalate) < restore_prob) & can_restore.any()
        action = jnp.where(escalate, BLUE_RESTORE_START + restore_slot, action)
        fires = fires | escalate
    return fires, action


def make_rulebook(cfg: RulebookConfig = RulebookConfig()):
    """Reference policy ``(keys (N_BLUE, 2), obs, masks) -> actions`` for ``cfg`` (RULEBOOK.md §5).

    A per-agent priority list evaluated from that agent's own observation; a pure function of
    ``(key, obs, mask)``. Agent keys are used only by R2/R4 (random choice among flagged hosts and
    the Restore escalation), so a firewall-only config ignores them.
    """
    if cfg.idle not in _IDLE:
        raise ValueError(f"idle must be one of {sorted(_IDLE)}, got {cfg.idle!r}")
    if cfg.priority not in ("protect_first", "respond_first"):
        raise ValueError(f"priority must be 'protect_first' or 'respond_first', got {cfg.priority!r}")
    if not 0.0 <= cfg.restore_prob <= 1.0:
        raise ValueError(f"restore_prob must be in [0, 1], got {cfg.restore_prob}")
    if cfg.restore_prob > 0 and not cfg.remove:
        raise ValueError("restore_prob > 0 escalates R2, so it requires remove=True")
    idle = _IDLE[cfg.idle]

    def agent(key, obs, masks):
        action = jnp.int32(idle)
        rules = []
        if cfg.firewall:
            fw_action, needs_fix = firewall_candidates(obs, masks)
            flat = needs_fix.reshape(-1)
            rules.append((flat.any(), fw_action.reshape(-1)[jnp.argmax(flat)]))
        if cfg.remove:
            rules.insert(0 if cfg.priority == "respond_first" else len(rules), _respond(key, obs, masks, cfg.restore_prob))
        for fires, act in reversed(rules):  # highest priority applied last
            action = jnp.where(fires, act, action)
        return action.astype(jnp.int32)

    def policy(keys: jax.Array, obs: jax.Array, masks: jax.Array) -> jax.Array:
        if keys is None:  # deterministic configs may be called without keys
            keys = jnp.zeros((obs.shape[0], 2), dtype=jnp.uint32)
        return jax.vmap(agent)(keys, obs, masks)

    return policy
