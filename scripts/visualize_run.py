"""Replay one episode of a trained run and write an animated, self-contained HTML player.

    uv run python scripts/visualize_run.py ~/cc4lab-exp/ippo_jax/default_seed42 --seed 0
    uv run python scripts/visualize_run.py ~/cc4lab-exp/s43 --seed 0 --seed 1 --seed 2
    uv run python scripts/visualize_run.py ~/cc4lab-exp/s44 --policy rulebook   # baseline on the same topology

``RUN`` is a run directory (anything containing exactly one ``model_<tag>.safetensors``,
searched recursively, so ``~/cc4lab-exp/s43`` works) or the ``.safetensors`` file itself.
The output goes to ``<dir of the model>/viz/episode_<policy>_seed<N>.html`` so it stays
with the run it visualizes. The HTML has no external dependencies; open it in any browser.

Training runs store weights and training curves, not episodes, so this script re-plays
one. It uses the same reset key (``PRNGKey(seed)``) and per-step key schedule as
``cc4lab.rollout``, so ``--seed N`` is the same episode that ``eval_checkpoint.py`` and
``action_mix.py`` score as episode N. ``--policy`` swaps in a baseline for agents
(not the topology or red's randomness), for side-by-side comparison.

Recorded per step: blue actions (decoded to type/target, busy flag, and for Block/Allow
whether the move complies with the mission phase's comms policy, per ``cc4lab.rules``),
team reward, host compromise level, red sessions per host, red scan/exploit activity,
decoys, blocked links, and mission phase.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jaxborg.actions.encoding import decode_blue_action
from jaxborg.constants import SUBNET_NAMES

from cc4lab import env as E
from cc4lab import rollout as R
from cc4lab.policies import trivial
from cc4lab.policies.loaders import checkpoint_info, load_policy
from cc4lab.policies.rulebook import firewall_rulebook
from cc4lab.rules import TRAFFIC_KINDS, traffic_action_kind

TEMPLATE = Path(__file__).with_name("visualize_run_template.html")
BLUE_TYPES = ("sleep", "monitor", "analyse", "remove", "restore", "decoy", "block", "allow")  # BLUE_ACTION_TYPE_* order
BASELINES = {
    "rulebook": firewall_rulebook,
    "random": trivial.random_valid,
    "sleep": trivial.always_sleep,
}


def resolve_model(run: Path) -> Path:
    run = run.expanduser().resolve()
    if run.is_file():
        return run
    found = sorted(run.rglob("model_*.safetensors"))
    if len(found) != 1:
        listing = "\n  ".join(map(str, found)) or "(none)"
        raise SystemExit(f"expected exactly one model_*.safetensors under {run}, found:\n  {listing}")
    return found[0]


def record_episode(env, policy, seed: int):
    """Full episode from ``PRNGKey(seed)``. Returns (initial env state, dict of per-step numpy arrays)."""
    key = jax.random.PRNGKey(seed)
    _, state0 = E.reset(env, key)
    const = state0.const
    masks = E.action_masks(env, state0)
    agent_ids = jnp.arange(E.N_BLUE)

    def snapshot(st):
        s = st.state
        bits = (1 << jnp.arange(s.red_sessions.shape[0]))[:, None]
        return {
            "compromised": s.host_compromised.astype(jnp.int8),
            "red_mask": (s.red_sessions * bits).sum(0).astype(jnp.int8),  # bit r = red agent r has a session
            "red_activity": s.red_activity_this_step.astype(jnp.int8),  # 0 none, 1 scan, 2 exploit
            "detected": s.host_activity_detected | s.host_exploit_detected,
            "decoys": s.host_decoys.any(-1),
            "impact": s.red_impact_attempted,
            "blocked": s.blocked_zones,
            "phase": s.mission_phase,
            "red_active": s.red_agent_active,
            "pending_ticks": s.blue_pending_ticks,
            "pending_action": s.blue_pending_action,
        }

    def body(st, _):
        obs = jnp.stack([o for o in (env.get_obs(st)[a] for a in E.BLUE_AGENTS)])
        env_key, agent_keys = R.step_keys(key, st.state.time)
        actions = policy(agent_keys, obs, masks)
        busy = st.state.blue_pending_ticks > 0
        # What actually runs this step: busy agents keep executing their pending action.
        effective = jnp.where(busy, st.state.blue_pending_action, actions)
        a_type, target, _, src, dst = jax.vmap(decode_blue_action, in_axes=(0, 0, None))(effective, agent_ids, const)
        _, _, kind = jax.vmap(traffic_action_kind)(obs, effective)
        _, st, reward, _, _ = E.step(env, env_key, st, actions)
        out = {
            "action": actions,
            "effective": effective,
            "busy": busy,
            "a_type": a_type,
            "target": target,
            "src": src,
            "dst": dst,
            "kind": kind,
            "reward": reward,
        } | snapshot(st)
        return st, out

    run = jax.jit(lambda s: jax.lax.scan(body, s, None, length=E.EPISODE_LENGTH))
    _, steps = run(state0)
    first = jax.tree.map(lambda x: np.asarray(x)[None], snapshot(state0))
    return state0, {k: np.asarray(v) for k, v in steps.items()}, first


def host_names(const, active_ids):
    sub = np.asarray(const.host_subnet)
    router, server = np.asarray(const.host_is_router), np.asarray(const.host_is_server)
    names, roles, counters = [], [], {}
    for h in active_ids:
        role = "router" if router[h] else "server" if server[h] else "user"
        n = counters.setdefault((sub[h], role), 0)
        counters[(sub[h], role)] += 1
        base = SUBNET_NAMES[sub[h]].lower()
        names.append(f"{base}_router" if role == "router" else f"{base}_{role}_host_{n}")
        roles.append(role)
    return names, roles


def build_payload(state0, steps, first, *, title: str, meta: dict) -> dict:
    const = state0.const
    active = np.nonzero(np.asarray(const.host_active))[0]
    remap = -np.ones(np.asarray(const.host_active).shape[0] + 1, dtype=int)
    remap[active] = np.arange(len(active))
    names, roles = host_names(const, active)

    def hosts(arr):  # (T, 137) -> list of strings, one char per active host
        return ["".join(map(str, row)) for row in np.asarray(arr)[:, active].astype(int)]

    def cat(name):
        return np.concatenate([first[name], steps[name]])

    T = steps["reward"].shape[0]
    blue_target = np.where(steps["target"] >= 0, remap[steps["target"]], -1)
    return {
        "title": title,
        "meta": meta,
        "subnets": list(SUBNET_NAMES),
        "blue_types": list(BLUE_TYPES),
        "traffic_kinds": list(TRAFFIC_KINDS),
        "hosts": {
            "name": names,
            "role": roles,
            "subnet": np.asarray(const.host_subnet)[active].tolist(),
        },
        "adjacency": np.asarray(const.subnet_adjacency).astype(int).tolist(),
        "blue_subnets": [np.nonzero(r)[0].tolist() for r in np.asarray(const.blue_agent_subnets)],
        "red_start": [int(remap[h]) if h >= 0 else -1 for h in np.asarray(const.red_start_hosts)],
        "phase_boundaries": np.asarray(const.phase_boundaries).tolist(),
        "comms_policy": np.asarray(const.comms_policy).astype(int).tolist(),
        "T": int(T),
        # frames 0..T: frame 0 is the reset state, frame t is the state after step t
        "frames": {
            "compromised": hosts(cat("compromised")),
            "red_mask": [",".join(map(str, r)) for r in cat("red_mask")[:, active].astype(int)],
            "red_activity": hosts(cat("red_activity")),
            "detected": hosts(cat("detected")),
            "decoys": hosts(cat("decoys")),
            "impact": hosts(cat("impact")),
            "blocked": [[[int(i), int(j)] for i, j in zip(*np.nonzero(b))] for b in cat("blocked")],
            "phase": cat("phase").astype(int).tolist(),
            "red_active": [int(sum(1 << i for i, a in enumerate(r) if a)) for r in cat("red_active")],
        },
        # steps 1..T (index t-1): what blue did during step t and the reward it earned
        "steps": {
            "reward": np.round(steps["reward"].astype(float), 3).tolist(),
            "a_type": steps["a_type"].astype(int).tolist(),
            "target": blue_target.astype(int).tolist(),
            "src": steps["src"].astype(int).tolist(),
            "dst": steps["dst"].astype(int).tolist(),
            "kind": steps["kind"].astype(int).tolist(),
            "busy": steps["busy"].astype(int).tolist(),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="run directory or model_<tag>.safetensors")
    ap.add_argument("--seed", type=int, action="append", help="episode seed (repeatable; default 0)")
    ap.add_argument(
        "--policy", default="trained", choices=["trained", *BASELINES], help="who plays blue (default: the run's model)"
    )
    ap.add_argument("--out-dir", type=Path, help="override output directory (default: <model dir>/viz)")
    args = ap.parse_args()
    seeds = args.seed or [0]

    model = resolve_model(args.run)
    E.describe_backend()
    info = checkpoint_info(model)
    policy = load_policy(model) if args.policy == "trained" else BASELINES[args.policy]
    env = E.make_env()
    out_dir = (args.out_dir or model.parent / "viz").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    template = TEMPLATE.read_text()

    for seed in seeds:
        state0, steps, first = record_episode(env, policy, seed)
        total = float(steps["reward"].sum())
        meta = {
            "model": str(model),
            "policy": args.policy,
            "seed": seed,
            "return": round(total, 2),
            "recipe": info.get("recipe"),
            "jaxborg_commit": E.jaxborg_commit(),
        }
        title = f"{model.parent.name} | {args.policy} | seed {seed}"
        payload = build_payload(state0, steps, first, title=title, meta=meta)
        data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")  # never close the <script> early
        html = template.replace("/*__DATA__*/null", data)
        out = out_dir / f"episode_{args.policy}_seed{seed}.html"
        out.write_text(html)
        print(f"[viz] seed {seed}: return {total:.1f} -> {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
