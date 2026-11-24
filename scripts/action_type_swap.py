"""Which of an agent's action types cause its marginal contribution? Swap one type at a time.

    uv run python scripts/action_type_swap.py MODEL.safetensors [MODEL2 ...] --agent 4

Same branch states and shared keys (CRN) as ``swap_decomposition.py``: base b
resets with key 100+b, runs the policy to ``--branch-t`` with key 200+b, and
evaluates the rest of the episode over keys split from 300+b. For agent ``i``
and each action type X it reports, split into LWF, ASF, RIA and Restore cost:

    removal    D_rm(X)  = V(all actual) - V(agent i's X actions -> Sleep)
    isolation  D_iso(X) = V(agent i does only its X actions) - V(agent i always Sleep)

plus ``everything -> Sleep``, which equals ``swap_decomposition``'s D. If the
types acted additively, D_rm(X) = D_iso(X) and both would sum to the full D.

Caveat: Sleep lasts 1 tick and the replaced actions last 1-5, so a swapped agent
is free more often and takes more of its other actions. The table of effective
actions per variant (submitted while not busy, per 100 steps) shows that shift.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jaxborg.actions import encoding as A
from jaxborg.constants import SUBNET_NAMES

from cc4lab import env as E
from cc4lab import rollout as R
from cc4lab.failures import per_subnet_terms
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
SWAPPED = ["monitor", "analyse", "remove", "restore", "decoy", "block", "allow"]
COLS = ["lwf", "asf", "ria", "restore_cost", "total"]


def type_vector(names) -> np.ndarray:
    """``(ACTION_DIM,)`` bool: action index belongs to one of ``names``."""
    v = np.zeros(E.ACTION_DIM, dtype=bool)
    for n in names:
        lo, hi = TYPES[n]
        v[lo:hi] = True
    return v


def type_of(actions: jax.Array) -> jax.Array:
    """Type index (order of ``TYPES``) of each action."""
    out = jnp.full(actions.shape, -1, dtype=jnp.int32)
    for j, (lo, hi) in enumerate(TYPES.values()):
        out = jnp.where((actions >= lo) & (actions < hi), j, out)
    return out


def make_eval(env, actual, agent: int, n_steps: int):
    """Jitted ``(state, keys (K, 2), replace (ACTION_DIM,) bool) -> dict`` of per-key results.

    Agent ``agent`` follows ``actual`` except that any action in ``replace`` becomes Sleep.
    """
    is_agent = jnp.arange(E.N_BLUE) == agent

    def run(state, key, replace):
        def pol(k, o, m):
            a = actual(k, o, m)
            return jnp.where(is_agent & replace[a], A.BLUE_SLEEP, a).astype(jnp.int32)

        masks = E.action_masks(env, state)

        def body(st, _):
            obs = jnp.stack([env.get_obs(st)[a] for a in E.BLUE_AGENTS])
            env_key, agent_keys = R.step_keys(key, st.state.time)
            acts = pol(agent_keys, obs, masks)
            busy = st.state.blue_pending_ticks[agent] > 0
            _, st, reward, _, _ = E.step(env, env_key, st, acts)
            return st, (per_subnet_terms(st.state, st.const), reward, acts[agent], busy)

        _, (terms, rewards, acts, busy) = jax.lax.scan(body, state, None, length=n_steps)
        effective = jax.nn.one_hot(jnp.where(busy, -1, type_of(acts)), len(TYPES)).sum(axis=0)
        return {"terms": terms.sum(axis=0), "total": rewards.sum(), "effective": effective}

    return jax.jit(lambda state, keys, replace: jax.lax.map(lambda k: run(state, k, replace), keys))


def split(r) -> np.ndarray:
    """``(K, 5)``: LWF, ASF, RIA, Restore cost (remainder), total."""
    t = np.asarray(r["terms"]).sum(axis=1)  # (K, 3)
    total = np.asarray(r["total"])
    return np.column_stack([t, total - t.sum(axis=1), total])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+")
    p.add_argument("--agent", type=int, default=4)
    p.add_argument("--branch-t", type=int, default=250)
    p.add_argument("--bases", type=int, default=3)
    p.add_argument("--keys", type=int, default=16)
    p.add_argument("--out", default=str(Path.home() / "cc4lab-exp/results/action_type_swap.json"))
    args = p.parse_args()

    E.describe_backend()
    env = E.make_env()
    rest = E.EPISODE_LENGTH - args.branch_t
    variants = {"actual": type_vector([]), "all->sleep": type_vector(TYPES)}
    for x in SWAPPED:
        variants[f"rm {x}"] = type_vector([x])
        variants[f"only {x}"] = type_vector([n for n in TYPES if n != x])

    report = {"args": vars(args), "backend": E.describe_backend(print_it=False), "models": {}}
    for path in args.models:
        actual = load_policy(path)
        fn = make_eval(env, actual, args.agent, rest)
        prefix = jax.jit(lambda s, k, actual=actual: R.rollout(env, actual, s, k, args.branch_t)[0])
        per_base = {v: [] for v in variants}  # each: list over bases of (K, 5)
        subnet_asf = {v: [] for v in variants}  # each: list over bases of (K, subnets)
        eff = {v: [] for v in variants}
        for b in range(args.bases):
            _, s0 = E.reset(env, jax.random.PRNGKey(100 + b))
            state = prefix(s0, jax.random.PRNGKey(200 + b))
            keys = jax.random.split(jax.random.PRNGKey(300 + b), args.keys)
            for v, vec in variants.items():
                r = fn(state, keys, jnp.asarray(vec))
                per_base[v].append(split(r))
                subnet_asf[v].append(np.asarray(r["terms"])[:, :, 1])
                eff[v].append(np.asarray(r["effective"]).mean(axis=0) / rest * 100)

        def diff(a, b, table=per_base):
            """Mean over keys then bases of a - b, and SE from the per-key paired differences."""
            d = np.stack([x - y for x, y in zip(table[a], table[b])])  # (bases, K, cols)
            se = d.std(axis=1, ddof=1).mean(axis=0) / np.sqrt(d.shape[1] * d.shape[0])
            return d.mean(axis=(0, 1)), se

        stem = Path(path).stem
        rows = {"all->sleep": diff("actual", "all->sleep")}
        for x in SWAPPED:
            rows[f"rm {x}"] = diff("actual", f"rm {x}")
            rows[f"only {x}"] = diff(f"only {x}", "all->sleep")
        print(f"\n{stem}, agent {args.agent}: D by term (mean over {args.bases} bases x {args.keys} keys; SE of total)")
        print(f"  {'variant':14s}" + "".join(f"{c:>13s}" for c in COLS) + "    SE")
        for name, (m, se) in rows.items():
            print(f"  {name:14s}" + "".join(f"{x:13.0f}" for x in m) + f"  {se[-1]:5.0f}")
        rm_sum = sum(rows[f"rm {x}"][0] for x in SWAPPED)
        iso_sum = sum(rows[f"only {x}"][0] for x in SWAPPED)
        print(f"  {'sum rm':14s}" + "".join(f"{x:13.0f}" for x in rm_sum))
        print(f"  {'sum only':14s}" + "".join(f"{x:13.0f}" for x in iso_sum))

        asf_full, _ = diff("actual", "all->sleep", table=subnet_asf)
        names = SUBNET_NAMES
        print("  ASF part of D (actual - all->sleep) by subnet: "
              + ", ".join(f"{n}: {x:.0f}" for n, x in zip(names, asf_full) if abs(x) >= 1))

        print(f"  effective actions per 100 steps (not busy), agent {args.agent}:")
        print(f"  {'variant':14s}" + "".join(f"{n:>9s}" for n in TYPES))
        for v in variants:
            print(f"  {v:14s}" + "".join(f"{x:9.1f}" for x in np.mean(eff[v], axis=0)))

        report["models"][stem] = {
            "rows": {k: {"mean": dict(zip(COLS, m.tolist())), "se_total": float(se[-1])} for k, (m, se) in rows.items()},
            "asf_by_subnet_full_swap": asf_full.tolist(),
            "effective_per_100": {v: dict(zip(TYPES, np.mean(eff[v], axis=0).tolist())) for v in variants},
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
