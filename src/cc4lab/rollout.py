"""Rollouts under ``jax.lax.scan``.

Milestone 1 provides the single-policy primitive (``rollout``). Milestone 2
adds coalition-masked rollouts on top of it.

Key schedule: the key for absolute step t is ``fold_in(episode_key, t)``,
split into an env key and a policy key; agent i's policy key is
``fold_in(policy_key, i)``. Because the schedule depends only on
(episode_key, t), a rollout branched at step t with key k replays exactly the
randomness the unbranched run with key k used from t onward.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jaxborg.actions.encoding import BLUE_RESTORE_END, BLUE_RESTORE_START

from cc4lab import env as E

Policy = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]


class Trajectory(NamedTuple):
    actions: jax.Array  # (T, N_BLUE) int32, as submitted by the policy
    rewards: jax.Array  # (T,) float32 team reward
    pending_ticks: jax.Array  # (T, N_BLUE) int32 blue_pending_ticks after each step
    busy: jax.Array  # (T, N_BLUE) bool, agent was mid-action BEFORE the step (its submitted action is ignored)


def step_keys(episode_key: jax.Array, t: jax.Array) -> tuple[jax.Array, jax.Array]:
    """(env_key, per-agent policy keys (N_BLUE, 2)) for absolute step t."""
    k = jax.random.fold_in(episode_key, t)
    env_key, pol_key = jax.random.split(k)
    agent_keys = jax.vmap(lambda i: jax.random.fold_in(pol_key, i))(jnp.arange(E.N_BLUE))
    return env_key, agent_keys


def rollout(env, policy: Policy, state, episode_key: jax.Array, n_steps: int):
    """Run ``n_steps`` from ``state``. Returns (final_state, Trajectory).

    ``n_steps`` must be static (it sets the scan length). Wrap in ``jax.jit``
    with ``static_argnums`` for env, policy and n_steps.
    """
    masks = E.action_masks(env, state)  # static per topology

    def body(carry, _):
        st = carry
        obs = env.get_obs(st)
        obs = jnp.stack([obs[a] for a in E.BLUE_AGENTS])
        env_key, agent_keys = step_keys(episode_key, st.state.time)
        actions = policy(agent_keys, obs, masks)
        busy = st.state.blue_pending_ticks > 0
        _, st, reward, _, _ = E.step(env, env_key, st, actions)
        return st, Trajectory(actions, reward, st.state.blue_pending_ticks, busy)

    return jax.lax.scan(body, state, None, length=n_steps)


def make_rollout_fn(env, policy: Policy, n_steps: int):
    """Jitted ``(state, key) -> (final_state, Trajectory)``."""
    return jax.jit(lambda state, key: rollout(env, policy, state, key, n_steps))


def ignored_restore_count(traj: Trajectory) -> jax.Array:
    """``(T,)`` Restores submitted by agents that were busy: ignored by the env but still charged -1.

    jaxborg charges -1 per *submitted* Restore regardless of busy state
    (``rewards.py``, intentional CybORG parity), so coalition masking can change
    the cost without changing the simulation (see NOTES.md).
    """
    is_restore = (traj.actions >= BLUE_RESTORE_START) & (traj.actions < BLUE_RESTORE_END)
    return (is_restore & traj.busy).sum(axis=-1)


def make_coalition_returns_fn(
    env, actual: Policy, reference: Policy, n_steps: int, *, batch_mode: str = "map", cost_mode: str = "raw"
):
    """Jitted ``(state, keys (K, 2), masks (C, N_BLUE) bool) -> returns (C, K)``.

    In coalition ``masks[c]``, agent ``i`` follows ``actual`` if ``masks[c, i]``
    else ``reference``. The return is the team reward summed over ``n_steps``
    from ``state``. Every coalition uses the same ``K`` keys (common random
    numbers), and both policies are evaluated for every agent with that agent's
    own key, so swapping agent ``i`` leaves the others' random draws unchanged.

    ``batch_mode``: ``"map"`` runs sequentially (fast on CPU: under ``vmap``
    ``lax.cond`` becomes ``select`` and every branch runs, ~8x slower, see
    NOTES.md); ``"vmap"`` batches over coalitions and keys (intended for GPU).

    ``cost_mode``: ``"raw"`` counts the reward exactly as jaxborg reports it;
    ``"busy_corrected"`` refunds the -1 charged for each Restore an agent
    submitted while busy (the env ignored it). Which one defines v(S) is an open
    decision (HANDOFF); ``raw`` is the default.
    """
    if batch_mode not in ("map", "vmap"):
        raise ValueError(f"batch_mode must be 'map' or 'vmap', got {batch_mode!r}")
    if cost_mode not in ("raw", "busy_corrected"):
        raise ValueError(f"cost_mode must be 'raw' or 'busy_corrected', got {cost_mode!r}")

    def value(state, key, member):
        def policy(k, o, m):
            return jnp.where(member, actual(k, o, m), reference(k, o, m))

        _, traj = rollout(env, policy, state, key, n_steps)
        total = traj.rewards.sum()
        if cost_mode == "busy_corrected":
            total = total + ignored_restore_count(traj).sum()
        return total

    if batch_mode == "vmap":
        fn = jax.vmap(jax.vmap(value, in_axes=(None, 0, None)), in_axes=(None, None, 0))
    else:

        def fn(state, keys, masks):
            n_c, n_k = masks.shape[0], keys.shape[0]
            pairs = (jnp.repeat(jnp.arange(n_c), n_k), jnp.tile(jnp.arange(n_k), n_c))
            flat = jax.lax.map(lambda ck: value(state, keys[ck[1]], masks[ck[0]]), pairs)
            return flat.reshape(n_c, n_k)

    return jax.jit(fn)
