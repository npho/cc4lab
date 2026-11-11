"""Spike 2: branch determinism.

For t in (100, 250): run to t, copy the state, roll out to 500 twice with the
same key and assert bitwise-identical trajectories and final states. Also:

* replay: the branched continuation must equal the unbranched run from t
  (the key schedule depends only on (episode_key, t), so any mismatch means
  the copied state is missing something);
* divergence: a different key must give a different trajectory;
* timers: in-progress blue actions survive the copy. At a branch state with a
  busy agent, changing only that agent's submitted action must not change
  the next state (it is executing its queued action), while changing an idle
  agent's action must.
"""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np
from jaxborg.actions.encoding import BLUE_BLOCK_TRAFFIC_START, BLUE_SLEEP
from _common import bitwise_equal, differing_leaves, write_json

from cc4lab import env as E
from cc4lab import rollout as R
from cc4lab.policies.trivial import random_valid

BRANCH_TS = (100, 250)
RESET_SEED, EP_SEED, ALT_SEED = 0, 1, 2


def timer_probe(env, state, ep_key):
    """Show the busy-agent semantics are carried by the copied state."""
    ticks = np.asarray(state.state.blue_pending_ticks)
    busy = np.flatnonzero(ticks > 0)
    idle = np.flatnonzero(ticks == 0)
    if len(busy) == 0 or len(idle) == 0:
        return {"skipped": f"need >=1 busy and >=1 idle agent, ticks={ticks.tolist()}"}
    env_key, agent_keys = R.step_keys(ep_key, state.state.time)
    base = random_valid(agent_keys, None, E.action_masks(env, state))
    # Sleep vs BlockTraffic(slot 0): both 1-tick and neither is Restore (so
    # action cost is unchanged); Block has a real env effect on blocked_zones.
    alt = BLUE_BLOCK_TRAFFIC_START

    def with_action(agent, a):
        return base.at[agent].set(a)

    step = jax.jit(lambda s, a: E.step(env, env_key, s, a)[1])
    b, i = int(busy[0]), int(idle[0])
    busy_same = bitwise_equal(step(state, with_action(b, BLUE_SLEEP)), step(state, with_action(b, alt)))
    idle_diff = differing_leaves(step(state, with_action(i, BLUE_SLEEP)), step(state, with_action(i, alt)))
    return {
        "blue_pending_ticks": ticks.tolist(),
        "blue_pending_action": np.asarray(state.state.blue_pending_action).tolist(),
        "red_pending_ticks": np.asarray(state.state.red_pending_ticks).tolist(),
        "busy_agent": b,
        "busy_agent_input_ignored": busy_same,
        "idle_agent": i,
        "idle_agent_input_changes_state": bool(idle_diff),
        "idle_agent_changed_fields": idle_diff,
    }


def main() -> int:
    backend = E.describe_backend()
    env = E.make_env()
    ep_key, alt_key = jax.random.PRNGKey(EP_SEED), jax.random.PRNGKey(ALT_SEED)
    _, s0 = E.reset(env, jax.random.PRNGKey(RESET_SEED))

    full_final, full = R.make_rollout_fn(env, random_valid, E.EPISODE_LENGTH)(s0, ep_key)
    results = {"backend": backend, "policy": "random_valid", "branches": {}}
    ok = True

    for t in BRANCH_TS:
        n = E.EPISODE_LENGTH - t
        s_t, prefix = R.make_rollout_fn(env, random_valid, t)(s0, ep_key)
        branch = E.copy_state(s_t)
        cont = R.make_rollout_fn(env, random_valid, n)
        fa, a = cont(branch, ep_key)
        fb, b = cont(branch, ep_key)
        fc, c = cont(branch, alt_key)

        tail = jax.tree.map(lambda x: x[t:], full)
        r = {
            "branch_time": int(s_t.state.time),
            "prefix_matches_full_run": bitwise_equal(prefix, jax.tree.map(lambda x: x[:t], full)),
            "same_key_trajectory_identical": bitwise_equal(a, b),
            "same_key_final_state_identical": bitwise_equal(fa, fb),
            "branch_replays_unbranched_run": bitwise_equal(a, tail),
            "branch_final_equals_unbranched_final": bitwise_equal(fa, full_final),
            "diff_key_actions_differ": not np.array_equal(np.asarray(a.actions), np.asarray(c.actions)),
            "diff_key_rewards_differ": not np.array_equal(np.asarray(a.rewards), np.asarray(c.rewards)),
            "return_same_key": float(jnp.sum(a.rewards)),
            "return_diff_key": float(jnp.sum(c.rewards)),
            "first_action_divergence_step": int(
                t + np.argmax(np.any(np.asarray(a.actions) != np.asarray(c.actions), axis=1))
            ),
            "timer_probe": timer_probe(env, branch, ep_key),
        }
        if not r["branch_final_equals_unbranched_final"]:
            r["final_state_diff_fields"] = differing_leaves(fa, full_final)
        results["branches"][str(t)] = r
        checks = [k for k, v in r.items() if isinstance(v, bool)]
        ok &= all(r[k] for k in checks) and r["timer_probe"].get("busy_agent_input_ignored", False)
        print(f"t={t}: " + ", ".join(f"{k}={r[k]}" for k in checks))
        print(f"  timer_probe: {r['timer_probe']}")

    results["pass"] = bool(ok)
    write_json("branching.json", results)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
