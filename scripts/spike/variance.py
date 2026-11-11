"""Spike 3: seed count K.

A. Per-episode sigma of total return over ~100 full episodes:
   (i) each episode on its own topology (independent reset keys), and
   (ii) all episodes on one topology (fixed reset key, different episode
   keys), the relevant case for counterfactuals of one episode.

B. From branch states at t = 250, sigma of the difference
       D = V(all actual) - V(agent 0 -> reference)
   where "actual" is random-valid and the reference is always-Sleep, under
   (a) CRN: both coalitions use the same episode key k, and
   (b) independent: the second coalition uses a different key k'.
   Recommended K to resolve a 50-point marginal contribution at 2 standard
   errors: 2 * sigma_D / sqrt(K) <= 50  =>  K >= (2 * sigma_D / 50)^2.
"""

from __future__ import annotations

import argparse
import math

import jax
import jax.numpy as jnp
import numpy as np
from _common import write_json

from cc4lab import env as E
from cc4lab import rollout as R
from cc4lab.policies.trivial import always_sleep, random_valid

BRANCH_T = 250
EFFECT = 50.0
N_SE = 2.0


def coalition_policy(member: jax.Array):
    """Agents with member[i] follow random_valid, the rest always_sleep.

    random_valid is evaluated for every agent with its own key, so swapping
    agent i leaves every other agent's random draws unchanged.
    """

    def policy(keys, obs, masks):
        return jnp.where(member, random_valid(keys, obs, masks), always_sleep(keys, obs, masks))

    return policy


def returns_fn(env, policy, n_steps):
    def one(state, key):
        _, traj = R.rollout(env, policy, state, key, n_steps)
        return traj.rewards.sum()

    # lax.map, not vmap: on CPU vmap turns lax.cond into select (all branches
    # run), ~8x slower per episode (see results/spike/benchmark_novmap.json).
    return jax.jit(lambda state, keys: jax.lax.map(lambda k: one(state, k), keys))


def recommended_k(sigma: float) -> int:
    return max(1, math.ceil((N_SE * sigma / EFFECT) ** 2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--bases", type=int, default=4, help="branch states (distinct topologies)")
    p.add_argument("--keys", type=int, default=256, help="rollouts per coalition per base")
    args = p.parse_args()

    backend = E.describe_backend()
    env = E.make_env()
    n = args.episodes

    # --- A. per-episode sigma ---------------------------------------------
    full = lambda s, k: R.rollout(env, random_valid, s, k, E.EPISODE_LENGTH)[1].rewards.sum()  # noqa: E731
    reset_v = jax.jit(jax.vmap(lambda k: E.reset(env, k)[1]))
    ep_keys = jax.random.split(jax.random.PRNGKey(7), n)

    states = reset_v(jax.random.split(jax.random.PRNGKey(3), n))
    ret_indep_topo = np.asarray(jax.jit(lambda ss, ks: jax.lax.map(lambda x: full(*x), (ss, ks)))(states, ep_keys))
    _, s_fixed = E.reset(env, jax.random.PRNGKey(3))
    ret_fixed_topo = np.asarray(jax.jit(lambda s, ks: jax.lax.map(lambda k: full(s, k), ks))(s_fixed, ep_keys))
    print(f"A. return sigma: independent topologies {ret_indep_topo.std(ddof=1):.1f} "
          f"(mean {ret_indep_topo.mean():.1f}); fixed topology {ret_fixed_topo.std(ddof=1):.1f} "
          f"(mean {ret_fixed_topo.mean():.1f})", flush=True)

    # --- B. sigma of a coalition difference from t = 250 -------------------
    n_rest = E.EPISODE_LENGTH - BRANCH_T
    all_actual = jnp.ones(E.N_BLUE, dtype=bool)
    swap0 = all_actual.at[0].set(False)
    v_all = returns_fn(env, coalition_policy(all_actual), n_rest)
    v_swap = returns_fn(env, coalition_policy(swap0), n_rest)
    prefix = jax.jit(lambda s, k: R.rollout(env, random_valid, s, k, BRANCH_T)[0])

    per_base = []
    for b in range(args.bases):
        _, s0 = E.reset(env, jax.random.PRNGKey(100 + b))
        s_t = prefix(s0, jax.random.PRNGKey(200 + b))
        keys = jax.random.split(jax.random.PRNGKey(300 + b), args.keys)
        alt_keys = jax.random.split(jax.random.PRNGKey(400 + b), args.keys)
        a = np.asarray(v_all(s_t, keys))
        s_crn = np.asarray(v_swap(s_t, keys))
        s_ind = np.asarray(v_swap(s_t, alt_keys))
        d_crn, d_ind = a - s_crn, a - s_ind
        row = {
            "base": b,
            "v_all_mean": a.mean(), "v_all_sigma": a.std(ddof=1),
            "v_swap0_mean": s_crn.mean(), "v_swap0_sigma": s_crn.std(ddof=1),
            "diff_mean_crn": d_crn.mean(), "diff_sigma_crn": d_crn.std(ddof=1),
            "diff_mean_indep": d_ind.mean(), "diff_sigma_indep": d_ind.std(ddof=1),
            "corr_all_swap0_crn": float(np.corrcoef(a, s_crn)[0, 1]),
        }
        per_base.append({k: float(v) if not isinstance(v, int) else v for k, v in row.items()})
        print(f"B. base {b}: sigma_D CRN {row['diff_sigma_crn']:.1f}, indep {row['diff_sigma_indep']:.1f}, "
              f"mean D {row['diff_mean_crn']:.1f}, corr {row['corr_all_swap0_crn']:.3f}", flush=True)

    # pooled within-base sigma (bases differ in mean, so pool variances)
    pool = lambda key: float(np.sqrt(np.mean([r[key] ** 2 for r in per_base])))  # noqa: E731
    sig_crn, sig_ind = pool("diff_sigma_crn"), pool("diff_sigma_indep")
    worst_crn = max(r["diff_sigma_crn"] for r in per_base)
    worst_ind = max(r["diff_sigma_indep"] for r in per_base)
    summary = {
        "sigma_return_independent_topologies": float(ret_indep_topo.std(ddof=1)),
        "sigma_return_fixed_topology": float(ret_fixed_topo.std(ddof=1)),
        "mean_return_independent_topologies": float(ret_indep_topo.mean()),
        "sigma_diff_crn_pooled": sig_crn,
        "sigma_diff_indep_pooled": sig_ind,
        "variance_reduction_factor": (sig_ind / sig_crn) ** 2 if sig_crn > 0 else None,
        "recommended_K_crn": recommended_k(sig_crn),
        "recommended_K_indep": recommended_k(sig_ind),
        "recommended_K_crn_worst_base": recommended_k(worst_crn),
        "recommended_K_indep_worst_base": recommended_k(worst_ind),
        "criterion": f"{N_SE:g} * sigma_D / sqrt(K) <= {EFFECT:g}",
    }
    print({k: round(v, 2) if isinstance(v, float) else v for k, v in summary.items()})
    write_json("variance.json", {
        "backend": backend,
        "actual_policy": "random_valid",
        "reference_policy": "always_sleep",
        "branch_t": BRANCH_T,
        "episodes": n,
        "keys_per_base": args.keys,
        "summary": summary,
        "per_base": per_base,
        "returns_independent_topologies": ret_indep_topo,
        "returns_fixed_topology": ret_fixed_topo,
    })


if __name__ == "__main__":
    main()
