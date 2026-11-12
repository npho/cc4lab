"""Where does an agent's marginal contribution come from? Split D by reward term.

    uv run python scripts/swap_decomposition.py MODEL.safetensors [MODEL2 ...] --agents 0 4

For each checkpoint, branch an observed episode at ``--branch-t`` (``--bases``
distinct topologies) and, over ``--keys`` shared keys (CRN), compare everyone
following the policy with agent ``i`` swapped to Sleep:

    D_i = V(all actual) - V(agent i -> Sleep),  split into LWF, ASF, RIA and Restore cost.

LWF/ASF/RIA are the green local-work failure, green access-service failure and
red impact terms (``cc4lab.failures``); the Restore cost is the remainder
(total D minus the three terms). A negative term means the agent's behavior
costs the team that much relative to sleeping. Means over keys, then bases.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cc4lab import env as E
from cc4lab import rollout as R
from cc4lab.failures import observed_profile
from cc4lab.policies.loaders import load_policy
from cc4lab.policies.trivial import always_sleep


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+")
    p.add_argument("--agents", type=int, nargs="+", default=[0, 4])
    p.add_argument("--branch-t", type=int, default=250)
    p.add_argument("--bases", type=int, default=3)
    p.add_argument("--keys", type=int, default=16)
    args = p.parse_args()

    E.describe_backend()
    env = E.make_env()
    rest = E.EPISODE_LENGTH - args.branch_t
    everyone = jnp.ones(E.N_BLUE, dtype=bool)

    for path in args.models:
        actual = load_policy(path)

        def parts(state, keys, member, actual=actual):
            def pol(k, o, m):
                return jnp.where(member, actual(k, o, m), always_sleep(k, o, m))

            def one(key):
                terms = observed_profile(env, pol, state, key, rest).sum(axis=(0, 1))  # (3,) LWF, ASF, RIA
                total = R.rollout(env, pol, state, key, rest)[1].rewards.sum()
                return jnp.concatenate([terms, total[None]])

            return jax.lax.map(one, keys)

        fn = jax.jit(parts)
        prefix = jax.jit(lambda s, k, actual=actual: R.rollout(env, actual, s, k, args.branch_t)[0])
        rows = {i: [] for i in args.agents}
        for b in range(args.bases):
            _, s0 = E.reset(env, jax.random.PRNGKey(100 + b))
            state = prefix(s0, jax.random.PRNGKey(200 + b))
            keys = jax.random.split(jax.random.PRNGKey(300 + b), args.keys)
            full = np.asarray(fn(state, keys, everyone))
            for i in args.agents:
                lwf, asf, ria, total = (full - np.asarray(fn(state, keys, everyone.at[i].set(False)))).mean(axis=0)
                rows[i].append([lwf, asf, ria, total - (lwf + asf + ria), total])
        for i in args.agents:
            a = np.mean(rows[i], axis=0)
            print(f"{Path(path).stem} agent {i}: D by term  LWF {a[0]:7.0f}  ASF {a[1]:7.0f}  RIA {a[2]:7.0f}  "
                  f"restore-cost {a[3]:6.0f}  | total D {a[4]:7.0f}")


if __name__ == "__main__":
    main()
