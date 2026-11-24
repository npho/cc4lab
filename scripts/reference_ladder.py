"""Evaluate the reference-policy ladder (RULEBOOK.md §6) on one seed set.

    uv run python scripts/reference_ladder.py --seed-start 1000 --episodes 32   # selection seeds
    uv run python scripts/reference_ladder.py --seed-start 0 --episodes 32      # evaluation seeds

Runs full 500-step episodes for L0 (always Sleep) and every rung in
``cc4lab.policies.rulebook.RUNGS`` on the same episode seeds (same topology and
episode key per seed, so differences are paired). Reports, per rung, the mean
return +/- SE under both cost modes, and paired differences against L0, L1 and L2,
plus the two data-driven decisions of §6.3:

1. idle: Sleep unless L3 (Monitor) beats L2 by more than 2 paired SE;
2. cost_mode: rungs that can emit Restore are reported under busy_corrected.

Per-episode returns go to ``--out`` (JSON) so the decisions can be re-derived.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from cc4lab import env as E
from cc4lab import rollout as R
from cc4lab.policies.rulebook import RUNGS, make_rulebook
from cc4lab.policies.trivial import always_sleep


def returns(env, policy, seeds):
    """``(raw, busy_corrected)`` per-episode returns over ``seeds``."""
    fn = R.make_rollout_fn(env, policy, E.EPISODE_LENGTH)
    raw, corrected = [], []
    for s in seeds:
        key = jax.random.PRNGKey(s)
        _, state = E.reset(env, key)
        _, traj = fn(state, key)
        r = float(traj.rewards.sum())
        raw.append(r)
        corrected.append(r + float(R.ignored_restore_count(traj).sum()))
    return np.array(raw), np.array(corrected)


def mse(x):
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed-start", type=int, default=1000)
    p.add_argument("--episodes", type=int, default=32)
    p.add_argument("--rungs", nargs="+", default=list(RUNGS))
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = Path(args.out or Path.home() / f"cc4lab-exp/results/reference_ladder_seeds{args.seed_start}.json")

    E.describe_backend()
    env = E.make_env()
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    policies = {"L0": always_sleep} | {name: make_rulebook(RUNGS[name]) for name in args.rungs}
    res = {}
    for name, pol in policies.items():
        raw, cor = returns(env, pol, seeds)
        res[name] = {"raw": raw, "busy_corrected": cor}
        print(f"  done {name}", flush=True)

    def headline(name):
        cfg = RUNGS.get(name)
        return "busy_corrected" if cfg is not None and cfg.needs_busy_corrected else "raw"

    print(f"\nseeds {seeds[0]}-{seeds[-1]} (n={len(seeds)}): mean return +/- SE")
    print(f"  {'rung':18s}{'raw':>20s}{'busy_corrected':>22s}   headline   {'- L0':>16s}{'- L1':>16s}{'- L2':>16s}")
    for name in policies:
        cm = headline(name)
        line = f"  {name:18s}"
        for mode in ("raw", "busy_corrected"):
            m, se = mse(res[name][mode])
            line += f"{m:12.1f} +/- {se:5.1f}" + ("  " if mode == "raw" else "")
        line += f"   {cm:9s}"
        for base in ("L0", "L1", "L2"):
            if base in res and base != name:
                m, se = mse(res[name][cm] - res[base][cm])
                line += f"{m:9.1f} +/-{se:5.1f}"
            else:
                line += f"{'':16s}"
        print(line)

    decisions = {}
    if "L2" in res and "L3" in res:
        m, se = mse(res["L3"]["raw"] - res["L2"]["raw"])
        decisions["idle"] = "monitor" if m > 2 * se else "sleep"
        print(f"\nidle decision (rule 1): L3 - L2 = {m:.1f} +/- {se:.1f} -> {decisions['idle']}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "seeds": seeds, "backend": E.describe_backend(print_it=False), "decisions": decisions,
        "configs": {k: vars(v) for k, v in RUNGS.items() if k in policies},
        "returns": {k: {m: v.tolist() for m, v in d.items()} for k, d in res.items()},
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
