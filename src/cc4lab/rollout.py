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

from cc4lab import env as E

Policy = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]


class Trajectory(NamedTuple):
    actions: jax.Array  # (T, N_BLUE) int32, as submitted by the policy
    rewards: jax.Array  # (T,) float32 team reward
    pending_ticks: jax.Array  # (T, N_BLUE) int32 blue_pending_ticks after each step


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
        _, st, reward, _, _ = E.step(env, env_key, st, actions)
        return st, Trajectory(actions, reward, st.state.blue_pending_ticks)

    return jax.lax.scan(body, state, None, length=n_steps)


def make_rollout_fn(env, policy: Policy, n_steps: int):
    """Jitted ``(state, key) -> (final_state, Trajectory)``."""
    return jax.jit(lambda state, key: rollout(env, policy, state, key, n_steps))
