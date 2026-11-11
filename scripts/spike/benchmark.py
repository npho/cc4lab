"""Spike 1: throughput.

jaxborg: rollout-only (no training), 500 steps, random-valid blue policy, at
batch sizes 1, 32, 1024 (vmap over independent episodes). Compile time is
measured separately via ``lower().compile()``; steps/s excludes compile and
reset and is the best of ``--reps`` timed runs.

CybORG: one process, one 500-step episode through ``EnterpriseMAE`` (the
wrapper jaxborg's own CybORG runner uses), random-valid actions from the
wrapper's action mask.
"""

from __future__ import annotations

import argparse
import csv
import time

import jax
import jax.numpy as jnp
import numpy as np
from _common import RESULTS, write_json

from cc4lab import env as E
from cc4lab import rollout as R
from cc4lab.policies.trivial import random_valid


def bench_jaxborg(env, batch: int, n_steps: int, reps: int) -> dict:
    keys = jax.random.split(jax.random.PRNGKey(batch), batch)
    states = jax.jit(jax.vmap(lambda k: E.reset(env, k)[1]))(keys)
    ep_keys = jax.random.split(jax.random.PRNGKey(10_000 + batch), batch)

    def run(states, ep_keys):
        _, traj = jax.vmap(lambda s, k: R.rollout(env, random_valid, s, k, n_steps))(states, ep_keys)
        return traj.rewards.sum(axis=1)  # only returns leave the device

    t0 = time.perf_counter()
    compiled = jax.jit(run).lower(states, ep_keys).compile()
    compile_s = time.perf_counter() - t0

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(compiled(states, ep_keys))
        times.append(time.perf_counter() - t0)
    best = min(times)
    return {
        "engine": "jaxborg",
        "batch": batch,
        "steps": n_steps,
        "compile_s": compile_s,
        "run_s": best,
        "run_s_all": times,
        "steps_per_s": batch * n_steps / best,
    }


def bench_unbatched(env, n_steps: int, reps: int) -> dict:
    """Single episode without vmap: lax.cond stays a real branch."""
    _, state = E.reset(env, jax.random.PRNGKey(1))
    key = jax.random.PRNGKey(10_001)
    run = lambda s, k: R.rollout(env, random_valid, s, k, n_steps)[1].rewards.sum()  # noqa: E731
    t0 = time.perf_counter()
    compiled = jax.jit(run).lower(state, key).compile()
    compile_s = time.perf_counter() - t0
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(compiled(state, key))
        times.append(time.perf_counter() - t0)
    best = min(times)
    return {"engine": "jaxborg-novmap", "batch": 1, "steps": n_steps, "compile_s": compile_s,
            "run_s": best, "run_s_all": times, "steps_per_s": n_steps / best}


def bench_cyborg(n_steps: int, seed: int = 0) -> dict:
    try:
        from CybORG.Agents.Wrappers import EnterpriseMAE
        from jaxborg.evaluation.cyborg_env_factory import make_cyborg_env, reset_cyborg_env
        from jaxborg.scenarios.cc4.game_variants import CC4_STOCK
    except ImportError as e:  # pragma: no cover
        return {"engine": "cyborg", "skipped": f"CybORG not importable: {e}"}

    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    env = make_cyborg_env(CC4_STOCK, seed, wrapper_class=EnterpriseMAE)
    r = reset_cyborg_env(env, CC4_STOCK, ep_seed=seed)
    setup_s = time.perf_counter() - t0
    obs, info = r.obs, r.info
    agents = sorted(a for a in obs if a.startswith("blue"))

    t0 = time.perf_counter()
    for _ in range(n_steps):
        act = {a: int(rng.choice(np.flatnonzero(info[a]["action_mask"]))) for a in agents}
        obs, _, term, trunc, info = env.step(act)
        if any(term.values()) or any(trunc.values()):
            break
    run_s = time.perf_counter() - t0
    return {
        "engine": "cyborg",
        "batch": 1,
        "steps": n_steps,
        "compile_s": None,
        "setup_s": setup_s,
        "run_s": run_s,
        "steps_per_s": n_steps / run_s,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batches", type=int, nargs="*", default=[1, 32, 1024])
    p.add_argument("--steps", type=int, default=E.EPISODE_LENGTH)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--skip-cyborg", action="store_true")
    p.add_argument("--skip-novmap", action="store_true")
    p.add_argument("--out", default="benchmark")
    args = p.parse_args()

    backend = E.describe_backend()
    env = E.make_env()
    rows = []
    if not args.skip_novmap:
        row = bench_unbatched(env, args.steps, args.reps)
        print(f"jaxborg no-vmap   compile={row['compile_s']:.1f}s run={row['run_s']:.2f}s "
              f"steps/s={row['steps_per_s']:,.0f}", flush=True)
        rows.append(row)
    for b in args.batches:
        reps = 1 if b >= 1024 else args.reps
        row = bench_jaxborg(env, b, args.steps, reps)
        print(f"jaxborg batch={b:5d} compile={row['compile_s']:.1f}s run={row['run_s']:.2f}s "
              f"steps/s={row['steps_per_s']:,.0f}", flush=True)
        rows.append(row)
    if not args.skip_cyborg:
        row = bench_cyborg(args.steps)
        print(f"cyborg  {row}", flush=True)
        rows.append(row)

    write_json(f"{args.out}.json", {"backend": backend, "policy": "random_valid", "rows": rows})
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / f"{args.out}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["engine", "batch", "steps", "steps_per_s", "compile_s", "run_s"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
