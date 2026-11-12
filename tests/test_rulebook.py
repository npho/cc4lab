"""Firewall rulebook and its violation flags, checked against simulator ground truth."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxborg.actions.encoding import BLUE_ALLOW_TRAFFIC_END, BLUE_BLOCK_TRAFFIC_START, BLUE_SLEEP, encode_blue_action
from jaxborg.observations import CYBORG_POS_TO_JAX_ID

from cc4lab import env as E
from cc4lab.policies.rulebook import firewall_rulebook
from cc4lab.rollout import make_rollout_fn, step_keys
from cc4lab.rules import firewall_violations

CYBORG_TO_JAX = np.asarray(CYBORG_POS_TO_JAX_ID)


@pytest.fixture(scope="module")
def world():
    env = E.make_env()
    key = jax.random.PRNGKey(0)
    _, state = E.reset(env, key)
    return env, key, state


def mismatches(state):
    """Ground truth: (N_BLUE,) count of observed-subnet firewall entries that disagree with the comms policy."""
    const, phase = state.const, int(state.state.mission_phase)
    policy, blocked = np.asarray(const.comms_policy[phase]), np.asarray(state.state.blocked_zones)
    out = []
    for dsts in np.asarray(const.blue_obs_subnets):
        out.append(sum(int(policy[d, s] != blocked[d, s]) for d in dsts if d >= 0 for s in range(9) if s != d))
    return np.array(out)


def obs_of(env, state):
    o = env.get_obs(state)
    return jnp.stack([o[a] for a in E.BLUE_AGENTS])


def test_rulebook_drives_firewall_to_policy_then_idles(world):
    env, key, state = world
    assert mismatches(state).sum() > 0  # non-vacuous: the episode starts out of compliance
    final, traj = make_rollout_fn(env, firewall_rulebook, 25)(state, key)
    assert mismatches(final).sum() == 0
    masks = E.action_masks(env, state)
    acts = np.asarray(traj.actions)
    assert np.asarray(masks)[np.arange(E.N_BLUE)[None, :], acts].all()  # every submitted action is valid
    assert ((acts == BLUE_SLEEP) | ((acts >= BLUE_BLOCK_TRAFFIC_START) & (acts < BLUE_ALLOW_TRAFFIC_END))).all()
    idle = firewall_rulebook(step_keys(key, final.state.time)[1], obs_of(env, final), masks)
    assert (np.asarray(idle) == BLUE_SLEEP).all()  # in compliance -> default action


def test_rulebook_readapts_when_mission_phase_changes(world):
    env, key, state = world
    settled, _ = make_rollout_fn(env, firewall_rulebook, 25)(state, key)
    assert mismatches(settled).sum() == 0
    new_phase = settled.replace(state=settled.state.replace(mission_phase=jnp.asarray(1, settled.state.mission_phase.dtype)))
    assert mismatches(new_phase).sum() > 0
    final, _ = make_rollout_fn(env, firewall_rulebook, 25)(new_phase, key)
    assert mismatches(final).sum() == 0


def test_first_action_matches_jaxborg_encoder(world):
    """Independent oracle: jaxborg's own encoder, driven from state tables, for the first mismatch of each agent."""
    env, key, state = world
    masks = E.action_masks(env, state)
    acts = np.asarray(firewall_rulebook(step_keys(key, state.state.time)[1], obs_of(env, state), masks))
    const, phase = state.const, int(state.state.mission_phase)
    policy, blocked = np.asarray(const.comms_policy[phase]), np.asarray(state.state.blocked_zones)
    for a in range(E.N_BLUE):
        expected = None
        for d in np.asarray(const.blue_obs_subnets[a]):
            if d < 0:
                continue
            for j in range(9):  # observation column order = cyborg position
                s = int(CYBORG_TO_JAX[j])
                if s == d or policy[d, s] == blocked[d, s]:
                    continue
                name = "BlockTrafficZone" if policy[d, s] else "AllowTrafficZone"
                cand = encode_blue_action(name, -1, a, const=const, src_subnet=s, dst_subnet=int(d))
                if np.asarray(masks)[a, cand]:
                    expected = cand
                    break
            if expected is not None:
                break
        assert expected is not None and acts[a] == expected, (a, acts[a], expected)


def test_violation_flags(world):
    env, key, state = world
    obs, masks = obs_of(env, state), E.action_masks(env, state)
    keys = step_keys(key, state.state.time)[1]
    truth = mismatches(state) > 0

    good = firewall_violations(obs, firewall_rulebook(keys, obs, masks))
    sleep = firewall_violations(obs, jnp.full(E.N_BLUE, BLUE_SLEEP, jnp.int32))
    busy_sleep = firewall_violations(obs, jnp.full(E.N_BLUE, BLUE_SLEEP, jnp.int32), busy=jnp.ones(E.N_BLUE, bool))

    assert (np.asarray(good["firewall_mismatch"]) == truth).all()  # flag agrees with ground truth
    assert not np.asarray(good["firewall_unaddressed"]).any()  # rulebook always addresses it
    assert (np.asarray(sleep["firewall_unaddressed"]) == truth).all()  # sleeping through a mismatch is a violation
    assert not np.asarray(busy_sleep["firewall_unaddressed"]).any()  # busy agents are exempt

    final, _ = make_rollout_fn(env, firewall_rulebook, 25)(state, key)
    quiet = firewall_violations(obs_of(env, final), jnp.full(E.N_BLUE, BLUE_SLEEP, jnp.int32))
    assert not np.asarray(quiet["firewall_mismatch"]).any() and not np.asarray(quiet["firewall_unaddressed"]).any()
