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
All functions are jit/vmap friendly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from cc4lab.env import ACTION_DIM
from cc4lab.policies.rulebook import firewall_candidates


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
