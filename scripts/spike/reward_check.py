"""Spike 4: reward structure.

Questions:
1. Does jaxborg return one shared team reward or per-agent rewards?
2. Are per-zone penalty components recoverable?

Method: run one 500-step random-valid episode, collecting every blue agent's
reward, jaxborg's ``info`` breakdown, and the post-step per-host flags. Then
rebuild the reward per subnet from state fields with
``const.phase_rewards`` (the same formula as
``jaxborg.rewards.compute_reward_breakdown``) and check that
sum(per-subnet) + action_cost == reward at every step.
"""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np
from _common import write_json
from jaxborg.actions.encoding import BLUE_RESTORE_END, BLUE_RESTORE_START
from jaxborg.constants import NUM_SUBNETS, SUBNET_NAMES
from jaxborg.rewards import ASF, LWF, RIA

from cc4lab import env as E
from cc4lab import rollout as R
from cc4lab.policies.trivial import random_valid

TOL = 1e-4


def per_subnet_reward(state, const):
    """(NUM_SUBNETS, 3) reward split by subnet and term [LWF, ASF, RIA]."""
    w = const.phase_rewards[state.mission_phase]  # (NUM_SUBNETS, 3)
    active = const.host_active.astype(jnp.float32)
    flags = jnp.stack(
        [state.green_lwf_this_step, state.green_asf_this_step, state.red_impact_attempted], axis=-1
    ).astype(jnp.float32)  # (hosts, 3), column order LWF, ASF, RIA
    per_host = flags * w[const.host_subnet][:, jnp.array([LWF, ASF, RIA])] * active[:, None]
    onehot = jax.nn.one_hot(const.host_subnet, NUM_SUBNETS) * active[:, None]
    return onehot.T @ per_host


def run_episode(env, s0, ep_key):
    masks = E.action_masks(env, s0)

    def body(st, _):
        obs = jnp.stack([env.get_obs(st)[a] for a in E.BLUE_AGENTS])
        env_key, agent_keys = R.step_keys(ep_key, st.state.time)
        actions = random_valid(agent_keys, obs, masks)
        act = {a: actions[i] for i, a in enumerate(E.BLUE_AGENTS)}
        _, st, rewards, _, info = env.step_env(env_key, st, act)
        out = {
            "rewards": jnp.stack([rewards[a] for a in E.BLUE_AGENTS]),
            "actions": actions,
            "per_subnet": per_subnet_reward(st.state, st.const),
            **{k: info[k] for k in ("reward_ria", "reward_lwf", "reward_asf", "action_cost")},
        }
        return st, out

    return jax.lax.scan(body, s0, None, length=E.EPISODE_LENGTH)


def main() -> int:
    backend = E.describe_backend()
    env = E.make_env()
    _, s0 = E.reset(env, jax.random.PRNGKey(0))
    _, out = jax.jit(lambda s, k: run_episode(env, s, k))(s0, jax.random.PRNGKey(1))
    out = jax.tree.map(np.asarray, out)

    rew = out["rewards"]  # (T, 5)
    shared = bool(np.all(rew == rew[:, :1]))
    team = rew[:, 0]
    per_subnet = out["per_subnet"]  # (T, 9, 3)
    restore = (out["actions"] >= BLUE_RESTORE_START) & (out["actions"] < BLUE_RESTORE_END)
    per_agent_cost = -restore.astype(np.float32)  # (T, 5)

    recon = per_subnet.sum(axis=(1, 2)) + per_agent_cost.sum(axis=1)
    info_total = out["reward_ria"] + out["reward_lwf"] + out["reward_asf"] + out["action_cost"]
    terms_match = {
        "lwf": float(np.max(np.abs(per_subnet[:, :, 0].sum(1) - out["reward_lwf"]))),
        "asf": float(np.max(np.abs(per_subnet[:, :, 1].sum(1) - out["reward_asf"]))),
        "ria": float(np.max(np.abs(per_subnet[:, :, 2].sum(1) - out["reward_ria"]))),
        "action_cost": float(np.max(np.abs(per_agent_cost.sum(1) - out["action_cost"]))),
    }
    owner = np.asarray(s0.const.blue_agent_subnets)  # (5, 9) bool
    ep_by_subnet = per_subnet.sum(axis=0)  # (9, 3)

    result = {
        "backend": backend,
        "shared_team_reward": shared,
        "max_abs_reward_spread_across_agents": float(np.max(rew.max(1) - rew.min(1))),
        "episode_return": float(team.sum()),
        "info_terms_sum_to_reward_max_err": float(np.max(np.abs(info_total - team))),
        "per_subnet_reconstruction_max_err": float(np.max(np.abs(recon - team))),
        "per_term_reconstruction_max_err": terms_match,
        "episode_by_subnet": {
            SUBNET_NAMES[z]: dict(zip(("lwf", "asf", "ria"), ep_by_subnet[z].tolist())) for z in range(NUM_SUBNETS)
        },
        "episode_action_cost_by_agent": per_agent_cost.sum(0).tolist(),
        "blue_agent_subnets": {
            f"blue_{i}": [SUBNET_NAMES[z] for z in np.flatnonzero(owner[i])] for i in range(E.N_BLUE)
        },
    }
    ok = shared and result["per_subnet_reconstruction_max_err"] < TOL and max(terms_match.values()) < TOL
    result["pass"] = bool(ok)
    write_json("reward_check.json", result)
    for k in ("shared_team_reward", "max_abs_reward_spread_across_agents", "episode_return",
              "info_terms_sum_to_reward_max_err", "per_subnet_reconstruction_max_err", "per_term_reconstruction_max_err",
              "episode_action_cost_by_agent", "blue_agent_subnets"):
        print(f"{k}: {result[k]}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
