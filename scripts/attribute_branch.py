"""End-to-end attribution for one branched episode (Milestone 2 pipeline).

    uv run python scripts/attribute_branch.py MODEL.safetensors --keys 4

1. Reset one episode (the seed fixes the topology) and run it to ``--branch-t``
   with every agent following the trained policy.
2. From that state, evaluate all 32 coalitions (agents in the coalition follow
   the trained policy, the rest follow ``--reference``) over K shared keys
   (common random numbers); v(S) is the team return to the end of the episode.
3. Exact Shapley values and the leave-one-out baseline per key, then mean and
   standard error over keys, plus a numerical check of efficiency
   (Shapley values sum to v(all) - v(none)).

Which failure to branch at, and how, is an open decision (HANDOFF): the branch
step here is just a parameter. ``--cost-mode`` and ``--reference`` are likewise
the open "busy-tick cost" and "reference policy" choices, exposed as switches.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cc4lab import env as E
from cc4lab import rollout as R
from cc4lab.attribution import coalition_masks
from cc4lab.attribution.loo import loo_values
from cc4lab.attribution.shapley import shapley_values
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
    p.add_argument("--branch-t", type=int, default=250)
    p.add_argument("--keys", type=int, default=4)
    p.add_argument("--reference", choices=REFERENCES, default="sleep")
    p.add_argument("--cost-mode", choices=("raw", "busy_corrected"), default="raw")
    p.add_argument("--batch-mode", choices=("map", "vmap"), default="map")
    p.add_argument("--out", default="attribution.json")
    args = p.parse_args()

    backend = E.describe_backend()
    info = checkpoint_info(args.model)
    env = E.make_env()
    actual, reference = load_policy(args.model), REFERENCES[args.reference]

    _, s0 = E.reset(env, jax.random.PRNGKey(args.episode_seed))
    prefix_key = jax.random.PRNGKey(10_000 + args.episode_seed)
    s_t, _ = jax.jit(lambda s, k: R.rollout(env, actual, s, k, args.branch_t))(s0, prefix_key)
    keys = jax.random.split(jax.random.PRNGKey(20_000 + args.episode_seed), args.keys)

    returns = R.make_coalition_returns_fn(
        env, actual, reference, E.EPISODE_LENGTH - args.branch_t, batch_mode=args.batch_mode, cost_mode=args.cost_mode
    )
    v = returns(s_t, keys, jnp.asarray(coalition_masks(E.N_BLUE)))  # (32, K)
    phi, loo = np.asarray(shapley_values(v)), np.asarray(loo_values(v))
    v = np.asarray(v)

    k = args.keys
    se = lambda x: x.std(axis=1, ddof=1) / np.sqrt(k) if k > 1 else np.full(x.shape[0], np.nan)  # noqa: E731
    efficiency_err = float(np.abs(phi.sum(0) - (v[-1] - v[0])).max())
    print(f"\nbranch t={args.branch_t}, episode seed {args.episode_seed}, K={k}, reference={args.reference}, "
          f"cost={args.cost_mode}")
    print(f"v(all actual) = {v[-1].mean():.1f}   v(none) = {v[0].mean():.1f}   efficiency max error = {efficiency_err:.2e}")
    print("agent   Shapley (mean +/- SE)    leave-one-out (mean +/- SE)")
    for i in range(E.N_BLUE):
        print(f"  {i}     {phi[i].mean():9.1f} +/- {se(phi)[i]:6.1f}      {loo[i].mean():9.1f} +/- {se(loo)[i]:6.1f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / args.out
    out.write_text(json.dumps({
        "backend": backend, "checkpoint": args.model, "checkpoint_info": info, "args": vars(args),
        "values": v.tolist(), "shapley": phi.tolist(), "loo": loo.tolist(), "efficiency_max_error": efficiency_err,
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
