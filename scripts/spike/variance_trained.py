"""Seed count K, re-measured with a trained policy (follow-up to spike 3, ``variance.py``).

The spike used random-valid blue agents. Here the "actual" policy is a trained
jaxborg checkpoint, and for each agent i we measure

    D_i = V(all actual) - V(agent i -> reference)       from the branch step t = 250,

where V is team return from t to the end of the episode and the reference is
always-Sleep (comparable with the spike; the firewall-only rulebook scored the
same as Sleep, see NOTES.md). All coalitions for one key share it (common random
numbers, CRN); the independent-keys comparison is run for agent 0 only, to
re-check the variance-reduction factor.

Recommended K per agent to resolve a ``--effect``-point marginal contribution at
2 standard errors: ``K >= (2 * sigma_D / effect) ** 2``. The overall
recommendation is the maximum over agents (pooled across branch bases, and the
worst single base).

    uv run python scripts/spike/variance_trained.py MODEL.safetensors
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from _common import write_json

from cc4lab import env as E
from cc4lab import rollout as R
from cc4lab.policies.loaders import checkpoint_info, load_policy
from cc4lab.policies.trivial import always_sleep

BRANCH_T = 250
N_SE = 2.0
DEFAULT_MODEL = Path.home() / "cc4lab-exp/ippo_jax/default_seed42/model_default_seed42.safetensors"


def recommended_k(sigma: float, effect: float) -> int:
    return max(1, math.ceil((N_SE * sigma / effect) ** 2))


def make_returns_fn(env, actual, n_steps):
    """Jitted ``(state, keys, member) -> returns``: agents with ``member[i]`` follow ``actual``, the rest Sleep.

    ``actual`` is evaluated for every agent with its own key, so swapping agent i
    leaves every other agent's draws unchanged. ``member`` is a traced argument,
    so all coalitions share one compilation. ``lax.map``, not ``vmap``: on CPU,
    vmap turns lax.cond into select (all branches run), ~8x slower per episode.
    """

    def one(state, key, member):
        def policy(k, o, m):
            return jnp.where(member, actual(k, o, m), always_sleep(k, o, m))

        return R.rollout(env, policy, state, key, n_steps)[1].rewards.sum()

    return jax.jit(lambda state, keys, member: jax.lax.map(lambda k: one(state, k, member), keys))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model", nargs="?", default=str(DEFAULT_MODEL))
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--bases", type=int, default=4, help="branch states (distinct topologies)")
    p.add_argument("--keys", type=int, default=128, help="rollouts per coalition per base")
    p.add_argument("--effect", type=float, default=50.0, help="marginal contribution to resolve")
    p.add_argument("--out", default="variance_trained.json")
    args = p.parse_args()

    backend = E.describe_backend()
    info = checkpoint_info(args.model)
    print(f"[checkpoint] {args.model}: {info}", flush=True)
    env = E.make_env()
    actual = load_policy(args.model)
    n = args.episodes

    # --- A. per-episode sigma of total return ------------------------------
    full = lambda s, k: R.rollout(env, actual, s, k, E.EPISODE_LENGTH)[1].rewards.sum()  # noqa: E731
    reset_v = jax.jit(jax.vmap(lambda k: E.reset(env, k)[1]))
    ep_keys = jax.random.split(jax.random.PRNGKey(7), n)
    states = reset_v(jax.random.split(jax.random.PRNGKey(3), n))
    ret_indep = np.asarray(jax.jit(lambda ss, ks: jax.lax.map(lambda x: full(*x), (ss, ks)))(states, ep_keys))
    _, s_fixed = E.reset(env, jax.random.PRNGKey(3))
    ret_fixed = np.asarray(jax.jit(lambda s, ks: jax.lax.map(lambda k: full(s, k), ks))(s_fixed, ep_keys))
    print(f"A. return sigma: independent topologies {ret_indep.std(ddof=1):.1f} (mean {ret_indep.mean():.1f}); "
          f"fixed topology {ret_fixed.std(ddof=1):.1f} (mean {ret_fixed.mean():.1f})", flush=True)

    # --- B. sigma of D_i for each agent ------------------------------------
    n_rest = E.EPISODE_LENGTH - BRANCH_T
    returns = make_returns_fn(env, actual, n_rest)
    prefix = jax.jit(lambda s, k: R.rollout(env, actual, s, k, BRANCH_T)[0])
    everyone = jnp.ones(E.N_BLUE, dtype=bool)
    swapped = [everyone.at[i].set(False) for i in range(E.N_BLUE)]

    per_base = []
    for b in range(args.bases):
        _, s0 = E.reset(env, jax.random.PRNGKey(100 + b))
        s_t = prefix(s0, jax.random.PRNGKey(200 + b))
        keys = jax.random.split(jax.random.PRNGKey(300 + b), args.keys)
        alt_keys = jax.random.split(jax.random.PRNGKey(400 + b), args.keys)
        v_all = np.asarray(returns(s_t, keys, everyone))
        row = {"base": b, "v_all_mean": float(v_all.mean()), "v_all_sigma": float(v_all.std(ddof=1)), "agents": []}
        for i in range(E.N_BLUE):
            d_crn = v_all - np.asarray(returns(s_t, keys, swapped[i]))
            agent = {"agent": i, "diff_mean": float(d_crn.mean()), "diff_sigma_crn": float(d_crn.std(ddof=1))}
            if i == 0:
                d_ind = v_all - np.asarray(returns(s_t, alt_keys, swapped[0]))
                agent["diff_sigma_indep"] = float(d_ind.std(ddof=1))
            row["agents"].append(agent)
        per_base.append(row)
        print(f"B. base {b}: v_all {row['v_all_mean']:.0f}; per agent (mean D, sigma_D CRN) "
              + ", ".join(f"{a['diff_mean']:.0f}/{a['diff_sigma_crn']:.0f}" for a in row["agents"])
              + f"; agent 0 indep sigma {row['agents'][0]['diff_sigma_indep']:.0f}", flush=True)

    pooled = [float(np.sqrt(np.mean([r["agents"][i]["diff_sigma_crn"] ** 2 for r in per_base]))) for i in range(E.N_BLUE)]
    worst = [max(r["agents"][i]["diff_sigma_crn"] for r in per_base) for i in range(E.N_BLUE)]
    sig_ind0 = float(np.sqrt(np.mean([r["agents"][0]["diff_sigma_indep"] ** 2 for r in per_base])))
    summary = {
        "sigma_return_independent_topologies": float(ret_indep.std(ddof=1)),
        "sigma_return_fixed_topology": float(ret_fixed.std(ddof=1)),
        "mean_return_independent_topologies": float(ret_indep.mean()),
        "sigma_diff_crn_pooled_by_agent": pooled,
        "recommended_K_crn_by_agent": [recommended_k(s, args.effect) for s in pooled],
        "recommended_K_crn": max(recommended_k(s, args.effect) for s in pooled),
        "recommended_K_crn_worst_base": max(recommended_k(s, args.effect) for s in worst),
        "agent0_sigma_diff_indep_pooled": sig_ind0,
        "agent0_variance_reduction_factor": (sig_ind0 / pooled[0]) ** 2 if pooled[0] > 0 else None,
        "criterion": f"{N_SE:g} * sigma_D / sqrt(K) <= {args.effect:g}",
    }
    print(summary, flush=True)
    write_json(args.out, {
        "backend": backend, "checkpoint": args.model, "checkpoint_info": info,
        "reference_policy": "always_sleep", "branch_t": BRANCH_T, "episodes": n, "keys_per_base": args.keys,
        "summary": summary, "per_base": per_base,
        "returns_independent_topologies": ret_indep, "returns_fixed_topology": ret_fixed,
    })


if __name__ == "__main__":
    main()
