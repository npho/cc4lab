"""Play a trained jaxborg checkpoint in cc4lab against trivial baselines.

    uv run python scripts/eval_checkpoint.py PATH/TO/model_<tag>.safetensors --episodes 8

Loads each checkpoint through ``cc4lab.policies.loaders`` (so it exercises the
same path attribution will use), runs full 500-step episodes, and compares mean
team return (with standard error) against ``firewall_rulebook``, ``random_valid``
and ``always_sleep`` on the same episode seeds. Also prints each checkpoint's
provenance from its recipe sidecar.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np

from cc4lab import env as E
from cc4lab.policies import trivial
from cc4lab.policies.loaders import checkpoint_info, load_policy
from cc4lab.policies.rulebook import firewall_rulebook
from cc4lab.rollout import make_rollout_fn


def mean_return(env, policy, episodes: int) -> tuple[float, float]:
    fn = make_rollout_fn(env, policy, E.EPISODE_LENGTH)
    totals = []
    for ep in range(episodes):
        key = jax.random.PRNGKey(ep)
        _, state = E.reset(env, key)
        _, traj = fn(state, key)
        totals.append(float(traj.rewards.sum()))
    return float(np.mean(totals)), float(np.std(totals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "models", nargs="+", help="paths to model_<tag>.safetensors (sidecar recipe_<tag>.yaml must sit beside each)"
    )
    ap.add_argument("--episodes", type=int, default=8)
    args = ap.parse_args()

    E.describe_backend()
    env = E.make_env()
    policies = {}
    for path in args.models:
        print(f"[checkpoint] {path}: {checkpoint_info(path)}")
        policies[Path(path).stem] = load_policy(path)
    policies |= {
        "firewall_rulebook": firewall_rulebook,
        "random_valid": trivial.random_valid,
        "always_sleep": trivial.always_sleep,
    }
    for name, policy in policies.items():
        mean, std = mean_return(env, policy, args.episodes)
        se = std / np.sqrt(args.episodes)
        print(f"{name:28s} mean return {mean:9.1f} +/- {se:6.1f} (std {std:7.1f}, n={args.episodes})")


if __name__ == "__main__":
    main()
