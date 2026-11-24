"""How often does each agent break each rulebook rule? (RULEBOOK.md §7)

    uv run python scripts/violation_rates.py MODEL.safetensors [MODEL2 ...] --episodes 8

Plays full episodes (seeds 0..N-1, as ``eval_checkpoint.py``) and counts, per agent,
every flag of ``cc4lab.rules.rule_violations`` per 100 steps, with busy agents
exempt (their submitted actions are ignored by the env). Also reports the share of
each agent's policy-violating Blocks that NIST containment (RS.MI-01) could
justify (``contain_justified``: the blocked link's destination subnet shows a
flagged host). Pass ``--rungs L1 L2`` to include reference agents for comparison.
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
from cc4lab.policies.loaders import load_policy
from cc4lab.policies.rulebook import RUNGS, make_rulebook
from cc4lab.rules import rule_violations

FLAGS = ("firewall_unaddressed", "firewall_violating", "contain_justified", "incident_unaddressed", "restore_unjustified")


def count_fn(env, policy):
    """Jitted ``(state, key) -> (N_BLUE, len(FLAGS))`` flag counts over one full episode."""

    def run(state, key):
        masks = E.action_masks(env, state)

        def body(st, _):
            obs = jnp.stack([env.get_obs(st)[a] for a in E.BLUE_AGENTS])
            env_key, agent_keys = R.step_keys(key, st.state.time)
            acts = policy(agent_keys, obs, masks)
            f = rule_violations(obs, acts, masks, busy=st.state.blue_pending_ticks > 0)
            _, st, _, _, _ = E.step(env, env_key, st, acts)
            return st, jnp.stack([f[k] for k in FLAGS], axis=-1)

        return jax.lax.scan(body, state, None, length=E.EPISODE_LENGTH)[1].sum(axis=0)

    return jax.jit(run)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="*")
    p.add_argument("--rungs", nargs="*", default=[])
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--out", default=str(Path.home() / "cc4lab-exp/results/violation_rates.json"))
    args = p.parse_args()

    E.describe_backend()
    env = E.make_env()
    policies = {Path(m).stem: load_policy(m) for m in args.models} | {f"rulebook_{r}": make_rulebook(RUNGS[r]) for r in args.rungs}
    report = {"flags": FLAGS, "episodes": args.episodes, "policies": {}}
    for name, pol in policies.items():
        fn = count_fn(env, pol)
        total = 0
        for ep in range(args.episodes):
            key = jax.random.PRNGKey(ep)
            _, s0 = E.reset(env, key)
            total = total + np.asarray(fn(s0, key))
        per100 = total / (args.episodes * E.EPISODE_LENGTH) * 100
        viol, just = total[:, FLAGS.index("firewall_violating")], total[:, FLAGS.index("contain_justified")]
        share = np.where(viol > 0, just / np.maximum(viol, 1), np.nan)
        print(f"\n{name}: violations per 100 steps (busy exempt), {args.episodes} episodes")
        print(f"  {'agent':6s}" + "".join(f"{f:>22s}" for f in FLAGS) + f"{'justified share':>17s}")
        for i in range(E.N_BLUE):
            print(f"  {i:<6d}" + "".join(f"{x:22.2f}" for x in per100[i]) + f"{share[i]:17.2f}")
        report["policies"][name] = {"per100": per100.tolist(), "counts": total.tolist(), "justified_share": share.tolist()}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
