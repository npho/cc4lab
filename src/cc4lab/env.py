"""Thin adapter over jaxborg's blue-only CC4 env (``FsmRedCC4Env``).

Conventions used throughout cc4lab:

* Blue observations, masks and actions are stacked arrays with a leading
  agent axis of size ``N_BLUE`` (5), not jaxborg's per-agent dicts.
* The env state is the jaxborg ``FsmRedEnvState`` pytree. It is immutable, so
  "copying" a state for branching is just holding a reference; ``copy_state``
  exists to make the intent explicit and to force fresh buffers if a caller
  ever donates arrays.
* ``step`` calls ``step_env`` (no auto-reset). Episodes are run for exactly
  ``max_steps - state.time`` steps by the caller.
"""

from __future__ import annotations

import functools
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxborg
from jaxborg.actions.encoding import BLUE_ALLOW_TRAFFIC_END
from jaxborg.constants import BLUE_OBS_SIZE, MAX_STEPS, NUM_BLUE_AGENTS
from jaxborg.evaluation.jax_env_factory import make_jax_env
from jaxborg.parity.fsm_red_env import FsmRedCC4Env, FsmRedEnvState
from jaxborg.scenarios.cc4.game_variants import CC4_STOCK

N_BLUE = NUM_BLUE_AGENTS
OBS_DIM = BLUE_OBS_SIZE
ACTION_DIM = BLUE_ALLOW_TRAFFIC_END
EPISODE_LENGTH = MAX_STEPS
BLUE_AGENTS = tuple(f"blue_{i}" for i in range(N_BLUE))

JAXBORG_DIR = Path(jaxborg.__file__).resolve().parents[2]


@functools.cache
def jaxborg_commit() -> str:
    """HEAD of the jaxborg checkout cc4lab is running against (``+dirty`` if modified)."""
    try:
        sha = subprocess.check_output(["git", "-C", str(JAXBORG_DIR), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(JAXBORG_DIR), "status", "--porcelain"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("+dirty" if dirty else "")


def describe_backend(print_it: bool = True) -> dict:
    """Report JAX devices/version and the jaxborg commit. Call at the top of every script."""
    info = {
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "jaxborg_commit": jaxborg_commit(),
        "jaxborg_dir": str(JAXBORG_DIR),
    }
    if print_it:
        print(
            f"[cc4lab] jax {info['jax_version']} backend={info['backend']} devices={info['devices']}\n"
            f"[cc4lab] jaxborg {info['jaxborg_commit']} ({info['jaxborg_dir']})"
        )
    return info


def make_env() -> FsmRedCC4Env:
    """Stock CC4: vanilla FSM red, generative topology, 500-step episodes."""
    return make_jax_env(CC4_STOCK)


def reset(env: FsmRedCC4Env, key: jax.Array) -> tuple[jax.Array, FsmRedEnvState]:
    """Reset from ``key``. The key also determines the topology, so all
    counterfactuals for one episode must share it."""
    obs, state = env.reset(key)
    return _stack(obs), state


def step(
    env: FsmRedCC4Env, key: jax.Array, state: FsmRedEnvState, actions: jax.Array
) -> tuple[jax.Array, FsmRedEnvState, jax.Array, jax.Array, dict]:
    """One env step. ``actions`` is (N_BLUE,) int32.

    Returns (obs (N_BLUE, OBS_DIM), next_state, team_reward scalar, done, info).
    jaxborg gives every blue agent the same reward, so one scalar is returned.
    """
    act = {a: actions[i] for i, a in enumerate(BLUE_AGENTS)}
    obs, state, rewards, dones, info = env.step_env(key, state, act)
    return _stack(obs), state, rewards[BLUE_AGENTS[0]], dones["__all__"], info


def action_masks(env: FsmRedCC4Env, state: FsmRedEnvState) -> jax.Array:
    """(N_BLUE, ACTION_DIM) bool. Static per topology; does not reflect busy state."""
    return _stack(env.get_avail_actions(state))


def copy_state(state: FsmRedEnvState) -> FsmRedEnvState:
    """Branch point snapshot with fresh buffers (pytree of jnp arrays)."""
    return jax.tree.map(jnp.copy, state)


def _stack(d: dict) -> jax.Array:
    return jnp.stack([d[a] for a in BLUE_AGENTS])
