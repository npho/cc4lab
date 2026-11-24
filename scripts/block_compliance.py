"""Do an agent's traffic actions follow the mission's communication policy, and which ones cost the team?

    uv run python scripts/block_compliance.py MODEL.safetensors [MODEL2 ...] --agent 4

Each BlockTrafficZone / AllowTrafficZone action is classified by
``cc4lab.rules.traffic_action_kind`` against the phase's communication policy. A Block is

* ``compliant``  the policy says the link should be blocked and it is open,
* ``violating``  the policy says the link should stay open and it is open,
* ``redundant``  the link is already blocked;

and an Allow is ``compliant`` (opens a link the policy wants open), ``violating``
(opens a link the policy wants blocked) or ``redundant`` (already open).

Part 1 counts effective (not busy) traffic actions per agent over full episodes.
Part 2 is a counterfactual for ``--agent`` on the same branch states and shared
keys as ``action_type_swap.py``: replace only its violating Blocks (or only its
compliant Blocks, or all Blocks) with Sleep, and report D = V(actual) - V(variant)
split into LWF, ASF, RIA and Restore cost.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jaxborg.actions.action_defs import BLUE_SLEEP

from cc4lab import env as E
from cc4lab import rollout as R
from cc4lab.failures import per_subnet_terms
from cc4lab.policies.loaders import load_policy
from cc4lab.rules import TRAFFIC_KINDS as KINDS
from cc4lab.rules import traffic_action_kind

COLS = ["lwf", "asf", "ria", "restore_cost", "total"]


def count_fn(env, actual, n_steps: int):
    """Jitted ``(state, key) -> (N_BLUE, 2, 3)`` effective Block/Allow counts by kind."""

    def run(state, key):
        masks = E.action_masks(env, state)

        def body(st, _):
            obs = jnp.stack([env.get_obs(st)[a] for a in E.BLUE_AGENTS])
            env_key, agent_keys = R.step_keys(key, st.state.time)
            acts = actual(agent_keys, obs, masks)
            free = st.state.blue_pending_ticks == 0
            is_b, is_a, kind = jax.vmap(traffic_action_kind)(obs, acts)
            oh = jax.nn.one_hot(kind, 3) * free[:, None]
            out = jnp.stack([oh * is_b[:, None], oh * is_a[:, None]], axis=1)
            _, st, _, _, _ = E.step(env, env_key, st, acts)
            return st, out

        return jax.lax.scan(body, state, None, length=n_steps)[1].sum(axis=0)

    return jax.jit(run)


def make_eval(env, actual, agent: int, n_steps: int):
    """Jitted ``(state, keys, drop (3,) bool) -> (K, 5)``: agent's Blocks of the dropped kinds become Sleep."""
    is_agent = jnp.arange(E.N_BLUE) == agent

    def run(state, key, drop):
        masks = E.action_masks(env, state)

        def body(st, _):
            obs = jnp.stack([env.get_obs(st)[a] for a in E.BLUE_AGENTS])
            env_key, agent_keys = R.step_keys(key, st.state.time)
            acts = actual(agent_keys, obs, masks)
            is_b, _, kind = jax.vmap(traffic_action_kind)(obs, acts)
            swap = is_agent & is_b & (kind >= 0) & drop[jnp.maximum(kind, 0)]
            acts = jnp.where(swap, BLUE_SLEEP, acts).astype(jnp.int32)
            _, st, reward, _, _ = E.step(env, env_key, st, acts)
            return st, (per_subnet_terms(st.state, st.const).sum(axis=0), reward)

        _, (terms, rewards) = jax.lax.scan(body, state, None, length=n_steps)
        t = terms.sum(axis=0)
        total = rewards.sum()
        return jnp.concatenate([t, (total - t.sum())[None], total[None]])

    return jax.jit(lambda state, keys, drop: jax.lax.map(lambda k: run(state, k, drop), keys))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+")
    p.add_argument("--agent", type=int, default=4)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--branch-t", type=int, default=250)
    p.add_argument("--bases", type=int, default=3)
    p.add_argument("--keys", type=int, default=16)
    p.add_argument("--out", default=str(Path.home() / "cc4lab-exp/results/block_compliance.json"))
    args = p.parse_args()

    E.describe_backend()
    env = E.make_env()
    rest = E.EPISODE_LENGTH - args.branch_t
    variants = {
        "violating->sleep": [False, True, False],
        "compliant->sleep": [True, False, False],
        "redundant->sleep": [False, False, True],
        "all blocks->sleep": [True, True, True],
    }
    report = {"args": vars(args), "backend": E.describe_backend(print_it=False), "models": {}}
    for path in args.models:
        stem = Path(path).stem
        actual = load_policy(path)

        counter = count_fn(env, actual, E.EPISODE_LENGTH)
        counts = 0
        for ep in range(args.episodes):
            key = jax.random.PRNGKey(ep)
            _, s0 = E.reset(env, key)
            counts = counts + np.asarray(counter(s0, key))
        per100 = counts / (args.episodes * E.EPISODE_LENGTH) * 100
        print(f"\n{stem}: effective traffic actions per 100 steps ({args.episodes} episodes), by kind")
        print(f"  {'agent':6s}" + "".join(f"{'block ' + k[:4]:>12s}" for k in KINDS)
              + "".join(f"{'allow ' + k[:4]:>12s}" for k in KINDS))
        for i in range(E.N_BLUE):
            print(f"  {i:<6d}" + "".join(f"{x:12.1f}" for x in per100[i, 0]) + "".join(f"{x:12.1f}" for x in per100[i, 1]))

        fn = make_eval(env, actual, args.agent, rest)
        prefix = jax.jit(lambda s, k, actual=actual: R.rollout(env, actual, s, k, args.branch_t)[0])
        diffs = {v: [] for v in variants}
        for b in range(args.bases):
            _, s0 = E.reset(env, jax.random.PRNGKey(100 + b))
            state = prefix(s0, jax.random.PRNGKey(200 + b))
            keys = jax.random.split(jax.random.PRNGKey(300 + b), args.keys)
            base = np.asarray(fn(state, keys, jnp.zeros(3, dtype=bool)))
            for v, drop in variants.items():
                diffs[v].append(base - np.asarray(fn(state, keys, jnp.asarray(drop))))
        print(f"  agent {args.agent}: D = V(actual) - V(variant), mean over {args.bases} bases x {args.keys} keys (SE of total)")
        print(f"  {'variant':18s}" + "".join(f"{c:>13s}" for c in COLS) + "    SE")
        rows = {}
        for v, d in diffs.items():
            d = np.stack(d)  # (bases, K, 5)
            m = d.mean(axis=(0, 1))
            se = float(d[..., -1].std(axis=1, ddof=1).mean() / np.sqrt(d.shape[0] * d.shape[1]))
            rows[v] = {"mean": dict(zip(COLS, m.tolist())), "se_total": se}
            print(f"  {v:18s}" + "".join(f"{x:13.0f}" for x in m) + f"  {se:5.0f}")
        report["models"][stem] = {"traffic_per_100": per100.tolist(), "kinds": KINDS, "rows": rows}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
