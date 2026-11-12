"""End-to-end attribution for one branched episode (Milestone 2 pipeline).

    # branch at a fixed step:
    uv run python scripts/attribute_branch.py MODEL.safetensors --branch-t 250 --keys 8
    # or branch around the worst failure window of the observed episode:
    uv run python scripts/attribute_branch.py MODEL.safetensors --worst-window 20 --term ria --keys 8

1. Reset one episode (the seed fixes the topology) and run it with every agent
   following the trained policy. This is the *observed* episode.
2. Choose the branch point: a fixed ``--branch-t``, or the ``--rank``-th worst
   window of ``--worst-window`` steps (optionally for one ``--term``) of the
   observed episode, branching ``--lookback`` steps before it and evaluating
   through it plus ``--tail`` steps (``cc4lab.failures``).
3. From the branch state, evaluate all 32 coalitions (agents in the coalition
   follow the trained policy, the rest follow ``--reference``) over K shared
   keys (common random numbers). Key 0 is the observed episode's own key, so
   the all-actual coalition replays the observed failure exactly; the other
   K-1 keys are fresh randomness from the same branch state.
4. Exact Shapley values and leave-one-out per key. Key 0 gives the *realized*
   attribution of this specific failure (ex-post); the mean over keys is the
   *expected* attribution from the branch state (ex-ante). Plus an efficiency check.

The open design choices are switches: ``--cost-mode``, ``--reference``, and the
window parameters. Which failures to analyse is the team's call (HANDOFF).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jaxborg.constants import SUBNET_NAMES

from cc4lab import env as E
from cc4lab import rollout as R
from cc4lab.attribution import coalition_masks
from cc4lab.attribution.loo import loo_values
from cc4lab.attribution.shapley import shapley_values
from cc4lab.failures import TERMS, find_windows, observed_profile, plan
from cc4lab.policies.loaders import checkpoint_info, load_policy
from cc4lab.policies.rulebook import firewall_rulebook
from cc4lab.policies.trivial import always_sleep

DEFAULT_MODEL = Path.home() / "cc4lab-exp/ippo_jax/default_seed42/model_default_seed42.safetensors"
RESULTS = Path(__file__).resolve().parents[1] / "results" / "attribution"
REFERENCES = {"sleep": always_sleep, "rulebook": firewall_rulebook}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model", nargs="?", default=str(DEFAULT_MODEL))
    p.add_argument("--episode-seed", type=int, default=0)
    where = p.add_argument_group("branch point (fixed step, or worst failure window)")
    where.add_argument("--branch-t", type=int, default=250, help="fixed branch step (ignored with --worst-window)")
    where.add_argument("--horizon", type=int, default=None, help="steps to evaluate (fixed mode; default: to episode end)")
    where.add_argument("--worst-window", type=int, default=None, metavar="W", help="window length W; enables automatic selection")
    where.add_argument("--rank", type=int, default=0, help="0 = worst window, 1 = second worst, ...")
    where.add_argument("--term", choices=("total", *TERMS), default="total", help="penalty term to rank windows by")
    where.add_argument("--lookback", type=int, default=20, help="branch this many steps before the window")
    where.add_argument("--tail", type=int, default=50, help="evaluate this many steps after the window")
    p.add_argument("--keys", type=int, default=4, help="K, including the observed key")
    p.add_argument("--reference", choices=REFERENCES, default="sleep")
    p.add_argument("--cost-mode", choices=("raw", "busy_corrected"), default="raw")
    p.add_argument("--batch-mode", choices=("map", "vmap"), default="map")
    p.add_argument("--out", default="attribution.json")
    args = p.parse_args()

    backend = E.describe_backend()
    info = checkpoint_info(args.model)
    env = E.make_env()
    actual, reference = load_policy(args.model), REFERENCES[args.reference]
    length = E.EPISODE_LENGTH

    _, s0 = E.reset(env, jax.random.PRNGKey(args.episode_seed))
    obs_key = jax.random.PRNGKey(10_000 + args.episode_seed)  # the observed episode's key

    window = None
    if args.worst_window is not None:
        profile = np.asarray(jax.jit(lambda s, k: observed_profile(env, actual, s, k, length))(s0, obs_key))
        found = find_windows(profile, args.worst_window, n=args.rank + 1, term=args.term)
        if len(found) <= args.rank:
            raise SystemExit(f"only {len(found)} non-overlapping windows of {args.worst_window} steps fit; --rank {args.rank} is too high")
        window = found[args.rank]
        branch_t, horizon = plan(window, args.lookback, args.tail, length)
        print(f"window rank {window.rank}: steps {window.start}-{window.end - 1}, {args.term} reward {window.reward:.0f}, "
              f"mostly {SUBNET_NAMES[window.subnet]} / {window.term}; branching at {branch_t}, evaluating {horizon} steps")
    else:
        branch_t = args.branch_t
        horizon = args.horizon if args.horizon is not None else length - branch_t
        horizon = min(horizon, length - branch_t)

    observed_rewards = np.asarray(jax.jit(lambda s, k: R.rollout(env, actual, s, k, length)[1].rewards)(s0, obs_key))
    s_t, _ = jax.jit(lambda s, k: R.rollout(env, actual, s, k, branch_t))(s0, obs_key)
    fresh = jax.random.split(jax.random.PRNGKey(20_000 + args.episode_seed), max(args.keys - 1, 0))
    keys = jnp.concatenate([obs_key[None], fresh])[: args.keys]

    returns = R.make_coalition_returns_fn(env, actual, reference, horizon, batch_mode=args.batch_mode, cost_mode=args.cost_mode)
    v = returns(s_t, keys, jnp.asarray(coalition_masks(E.N_BLUE)))  # (32, K)
    phi, loo = np.asarray(shapley_values(v)), np.asarray(loo_values(v))
    v = np.asarray(v)

    k = keys.shape[0]
    se = lambda x: x.std(axis=1, ddof=1) / np.sqrt(k) if k > 1 else np.full(x.shape[0], np.nan)  # noqa: E731
    efficiency_err = float(np.abs(phi.sum(0) - (v[-1] - v[0])).max())
    observed_tail = float(observed_rewards[branch_t : branch_t + horizon].sum())
    print(f"\nbranch t={branch_t}, horizon {horizon}, episode seed {args.episode_seed}, K={k}, reference={args.reference}, cost={args.cost_mode}")
    print(f"v(all actual): key 0 (realized) {v[-1, 0]:.1f}, mean over keys {v[-1].mean():.1f};  v(none) mean {v[0].mean():.1f};  efficiency max error {efficiency_err:.2e}")
    if args.cost_mode == "raw":
        print(f"replay check: observed return over the horizon {observed_tail:.1f} vs key-0 all-actual {v[-1, 0]:.1f} "
              f"({'match' if observed_tail == float(v[-1, 0]) else 'MISMATCH'})")
    print("agent   Shapley realized (key 0)   Shapley expected (mean +/- SE)   leave-one-out expected")
    for i in range(E.N_BLUE):
        print(f"  {i}        {phi[i, 0]:9.1f}               {phi[i].mean():9.1f} +/- {se(phi)[i]:6.1f}            {loo[i].mean():9.1f} +/- {se(loo)[i]:6.1f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / args.out
    out.write_text(json.dumps({
        "backend": backend, "checkpoint": args.model, "checkpoint_info": info, "args": vars(args),
        "window": None if window is None else {**window._asdict(), "subnet_name": SUBNET_NAMES[window.subnet]},
        "branch_t": branch_t, "horizon": horizon, "observed_return_over_horizon": observed_tail,
        "values": v.tolist(), "shapley": phi.tolist(), "loo": loo.tolist(), "efficiency_max_error": efficiency_err,
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
