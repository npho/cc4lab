"""Share of steps each blue agent spends on each action type, per checkpoint.

    uv run python scripts/action_mix.py MODEL.safetensors [MODEL2 ...] --episodes 8

Full 500-step episodes (seeds 0..N-1), policy sampled as in attribution. Shows
how far a trained policy is from uniform-over-valid-actions (for reference,
uniform gives about 20% each for analyse, remove, restore and decoy, and about
10% each for block and allow) and which agents lean on which actions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np
from jaxborg.actions import encoding as A

from cc4lab import env as E
from cc4lab import rollout as R
from cc4lab.policies.loaders import load_policy

TYPES = {
    "sleep": (A.BLUE_SLEEP, A.BLUE_SLEEP + 1),
    "monitor": (A.BLUE_MONITOR, A.BLUE_MONITOR + 1),
    "analyse": (A.BLUE_ANALYSE_START, A.BLUE_ANALYSE_END),
    "remove": (A.BLUE_REMOVE_START, A.BLUE_REMOVE_END),
    "restore": (A.BLUE_RESTORE_START, A.BLUE_RESTORE_END),
    "decoy": (A.BLUE_DECOY_START, A.BLUE_DECOY_END),
    "block": (A.BLUE_BLOCK_TRAFFIC_START, A.BLUE_BLOCK_TRAFFIC_END),
    "allow": (A.BLUE_ALLOW_TRAFFIC_START, A.BLUE_ALLOW_TRAFFIC_END),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+")
    p.add_argument("--episodes", type=int, default=8)
    args = p.parse_args()

    E.describe_backend()
    env = E.make_env()
    for path in args.models:
        fn = R.make_rollout_fn(env, load_policy(path), E.EPISODE_LENGTH)
        acts = []
        for ep in range(args.episodes):
            key = jax.random.PRNGKey(ep)
            _, state = E.reset(env, key)
            acts.append(np.asarray(fn(state, key)[1].actions))
        acts = np.concatenate(acts)  # (episodes * 500, N_BLUE)
        print(f"{Path(path).stem}: share of steps by action type (%), agents 0-4")
        for name, (lo, hi) in TYPES.items():
            share = ((acts >= lo) & (acts < hi)).mean(axis=0) * 100
            print(f"  {name:8s}" + "".join(f"{x:7.1f}" for x in share))


if __name__ == "__main__":
    main()
