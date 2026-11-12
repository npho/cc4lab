"""Failure profiles of an observed episode, and branch-point selection.

Why windows and not events: with a trained policy every episode is a continuous
stream of penalties, not a few discrete failures (in 16 episodes red impact hit
on 46-137 of 500 steps, first hit at step 3-66, and green work failure was the
largest cost; see NOTES.md). "Branch before the failure" has no single trigger,
so a *failure* here is a window of consecutive steps with a large penalty.

``observed_profile`` replays an episode and records the reward split by subnet
and term (LWF = green local-work failure, ASF = green access-service failure,
RIA = red impact; each <= 0). ``find_windows`` ranks non-overlapping worst
windows, optionally for one term. ``plan`` turns a window into a branch step
(some ``lookback`` before it, so the decisions that led there can change) and a
horizon (through the window plus a ``tail`` for after-effects).

Column order of the profile's last axis is ``TERMS``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxborg.constants import NUM_SUBNETS
from jaxborg.rewards import ASF, LWF, RIA

from cc4lab import env as E
from cc4lab import rollout as R

TERMS = ("lwf", "asf", "ria")


def per_subnet_terms(state, const) -> jax.Array:
    """``(NUM_SUBNETS, 3)`` reward of the step that produced ``state``, split by subnet and ``TERMS``.

    Rebuilt from the post-step per-host flags and the phase weights; matches
    jaxborg's ``compute_reward_breakdown`` exactly (spike 4, ``reward_check.py``).
    It excludes the per-agent Restore action cost.
    """
    w = const.phase_rewards[state.mission_phase]  # (NUM_SUBNETS, 3)
    active = const.host_active.astype(jnp.float32)
    flags = jnp.stack(
        [state.green_lwf_this_step, state.green_asf_this_step, state.red_impact_attempted], axis=-1
    ).astype(jnp.float32)
    per_host = flags * w[const.host_subnet][:, jnp.array([LWF, ASF, RIA])] * active[:, None]
    onehot = jax.nn.one_hot(const.host_subnet, NUM_SUBNETS) * active[:, None]
    return onehot.T @ per_host


def observed_profile(env, policy, state, episode_key: jax.Array, n_steps: int) -> jax.Array:
    """``(n_steps, NUM_SUBNETS, 3)`` penalties per step along a rollout of ``policy`` from ``state``.

    Uses the same key schedule as ``rollout.rollout``, so with the same
    ``episode_key`` it follows exactly the trajectory ``rollout`` would.
    """
    masks = E.action_masks(env, state)

    def body(st, _):
        obs = jnp.stack([env.get_obs(st)[a] for a in E.BLUE_AGENTS])
        env_key, agent_keys = R.step_keys(episode_key, st.state.time)
        actions = policy(agent_keys, obs, masks)
        _, st, _, _, _ = E.step(env, env_key, st, actions)
        return st, per_subnet_terms(st.state, st.const)

    return jax.lax.scan(body, state, None, length=n_steps)[1]


class Window(NamedTuple):
    rank: int  # 0 = worst
    start: int  # first step of the window (absolute episode step)
    end: int  # one past the last step
    reward: float  # summed (negative) reward of the selected term(s) over the window
    subnet: int  # subnet contributing the most penalty in the window (index into SUBNET_NAMES)
    term: str  # term contributing the most penalty in the window


def find_windows(profile, window: int, n: int = 1, term: str = "total") -> list[Window]:
    """Up to ``n`` non-overlapping windows of ``window`` consecutive steps, worst first.

    ``profile`` is ``(T, NUM_SUBNETS, 3)``. ``term`` is ``"total"`` or one of
    ``TERMS``; windows are scored by the summed reward of that term (most
    negative is worst). Greedy: take the worst window, exclude every window that
    overlaps it, repeat. Ties go to the earliest start.
    """
    profile = np.asarray(profile, dtype=np.float64)
    steps = profile.shape[0]
    if not 1 <= window <= steps:
        raise ValueError(f"window must be in [1, {steps}], got {window}")
    if term != "total" and term not in TERMS:
        raise ValueError(f"term must be 'total' or one of {TERMS}, got {term!r}")

    cols = slice(None) if term == "total" else slice(TERMS.index(term), TERMS.index(term) + 1)
    series = profile[:, :, cols].sum(axis=(1, 2))
    csum = np.concatenate([[0.0], np.cumsum(series)])
    sums = csum[window:] - csum[:-window]  # sums[t0] = reward over [t0, t0 + window)

    out: list[Window] = []
    available = np.ones_like(sums, dtype=bool)
    for rank in range(n):
        if not available.any():
            break
        t0 = int(np.argmin(np.where(available, sums, np.inf)))
        seg = profile[t0 : t0 + window, :, cols]
        per_subnet, per_term = seg.sum(axis=(0, 2)), seg.sum(axis=(0, 1))
        dominant_term = term if term != "total" else TERMS[int(np.argmin(per_term))]
        out.append(Window(rank, t0, t0 + window, float(sums[t0]), int(np.argmin(per_subnet)), dominant_term))
        available[max(0, t0 - window + 1) : t0 + window] = False
    return out


def plan(win: Window, lookback: int, tail: int, episode_length: int) -> tuple[int, int]:
    """``(branch_t, horizon)`` for a window: branch ``lookback`` steps before it, evaluate through it plus ``tail``."""
    branch_t = max(0, win.start - lookback)
    horizon = min(episode_length - branch_t, win.end - branch_t + tail)
    return branch_t, horizon
