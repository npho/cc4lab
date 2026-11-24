"""NIST-informed rulebook (RULEBOOK.md §5, §8): R2 Remove, R4 Restore, idle switch, priority, regression.

R2/R4 read only the observation's per-host ``malicious_processes`` flags, so most
tests set those bits on a real observation by hand. The independent oracle for
"which action targets this host" is jaxborg's own ``encode_blue_action``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxborg.actions.encoding import (
    BLUE_MONITOR,
    BLUE_REMOVE_END,
    BLUE_REMOVE_START,
    BLUE_RESTORE_END,
    BLUE_RESTORE_START,
    BLUE_SLEEP,
    encode_blue_action,
)
from jaxborg.constants import BLUE_MAX_OBSERVED_SUBNETS, NUM_SUBNETS, OBS_VECTOR_HOSTS_PER_SUBNET
from jaxborg.observations import SUBNET_BLOCK_SIZE

from cc4lab import env as E
from cc4lab import rollout as R
from cc4lab.attribution import coalition_masks
from cc4lab.attribution.shapley import shapley_values
from cc4lab.policies.rulebook import RUNGS, RulebookConfig, firewall_rulebook, make_firewall_rulebook, make_rulebook
from cc4lab.policies.trivial import random_valid

H = OBS_VECTOR_HOSTS_PER_SUBNET
FLAG_OFFSET = 3 * NUM_SUBNETS  # malicious_processes come after one-hot, blocked, comms


@pytest.fixture(scope="module")
def world():
    env = E.make_env()
    key = jax.random.PRNGKey(0)
    _, state = E.reset(env, key)
    state, _ = R.make_rollout_fn(env, random_valid, 30)(state, key)
    obs = jnp.stack([env.get_obs(state)[a] for a in E.BLUE_AGENTS])
    return env, state, obs, E.action_masks(env, state)


def with_flags(obs_a, slots):
    """One agent's obs with malicious_processes cleared, then set for flat host slots ``slots`` (r * H + k)."""
    o = np.array(obs_a)
    for r in range(BLUE_MAX_OBSERVED_SUBNETS):
        start = 1 + r * SUBNET_BLOCK_SIZE + FLAG_OFFSET
        o[start : start + H] = 0.0
    for s in slots:
        r, k = divmod(s, H)
        o[1 + r * SUBNET_BLOCK_SIZE + FLAG_OFFSET + k] = 1.0
    return jnp.asarray(o)


def valid_host_slots(state, masks, agent):
    """Flat slots (r * H + k) of real hosts the agent may Remove, with the global host each maps to."""
    const = state.const
    out = []
    for r, subnet in enumerate(np.asarray(const.blue_obs_subnets[agent])):
        if subnet < 0:
            continue
        for k in range(H):
            host = int(const.obs_host_map[subnet, k])
            if host < const.host_active.shape[0] and bool(masks[agent, BLUE_REMOVE_START + r * H + k]):
                out.append((r * H + k, host))
    return out


def run_one(policy, obs, masks, agent, obs_a, key=0):
    """Policy action for ``agent`` when its obs row is replaced by ``obs_a``."""
    o = obs.at[agent].set(obs_a)
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(key), i))(jnp.arange(E.N_BLUE))
    return int(policy(keys, o, masks)[agent])


REMOVE_ONLY = RulebookConfig(firewall=False, remove=True)


@pytest.mark.parametrize("agent", [0, 4])
def test_remove_targets_the_flagged_host(world, agent):
    env, state, obs, masks = world
    policy = make_rulebook(REMOVE_ONLY)
    slots = valid_host_slots(state, masks, agent)
    assert len(slots) >= 2
    for slot, host in slots[:: max(1, len(slots) // 4)]:
        got = run_one(policy, obs, masks, agent, with_flags(obs[agent], [slot]))
        assert got == encode_blue_action("Remove", host, agent, const=state.const), (agent, slot, host, got)


def test_no_flag_or_invalid_mask_falls_through_to_idle(world):
    env, state, obs, masks = world
    policy = make_rulebook(REMOVE_ONLY)
    agent = 0
    assert run_one(policy, obs, masks, agent, with_flags(obs[agent], [])) == BLUE_SLEEP
    slot, _ = valid_host_slots(state, masks, agent)[0]
    blocked = masks.at[agent, BLUE_REMOVE_START + slot].set(False).at[agent, BLUE_RESTORE_START + slot].set(False)
    assert run_one(policy, obs, blocked, agent, with_flags(obs[agent], [slot])) == BLUE_SLEEP


def test_remove_choice_is_random_and_covers_every_flagged_host(world):
    env, state, obs, masks = world
    policy = make_rulebook(REMOVE_ONLY)
    agent = 4
    chosen = [s for s, _ in valid_host_slots(state, masks, agent)[:3]]
    o = with_flags(obs[agent], chosen)
    picks = {run_one(policy, obs, masks, agent, o, key=k) - BLUE_REMOVE_START for k in range(200)}
    assert picks == set(chosen)


def test_restore_probability(world):
    env, state, obs, masks = world
    agent = 1
    slot, host = valid_host_slots(state, masks, agent)[0]
    flagged = with_flags(obs[agent], [slot])
    never = make_rulebook(RulebookConfig(firewall=False, remove=True, restore_prob=0.0))
    always = make_rulebook(RulebookConfig(firewall=False, remove=True, restore_prob=1.0))
    half = make_rulebook(RulebookConfig(firewall=False, remove=True, restore_prob=0.5))
    restore = encode_blue_action("Restore", host, agent, const=state.const)
    for k in range(20):
        assert not BLUE_RESTORE_START <= run_one(never, obs, masks, agent, flagged, k) < BLUE_RESTORE_END
        assert run_one(always, obs, masks, agent, flagged, k) == restore
        assert run_one(always, obs, masks, agent, with_flags(obs[agent], []), k) == BLUE_SLEEP
    kinds = {run_one(half, obs, masks, agent, flagged, k) for k in range(60)}
    assert kinds == {restore, BLUE_REMOVE_START + slot}


def test_idle_switch(world):
    env, state, obs, masks = world
    o = with_flags(obs[0], [])
    assert run_one(make_rulebook(RulebookConfig(firewall=False, remove=True, idle="monitor")), obs, masks, 0, o) == BLUE_MONITOR
    assert run_one(make_rulebook(RulebookConfig(firewall=False, remove=True, idle="sleep")), obs, masks, 0, o) == BLUE_SLEEP


def test_priority_order(world):
    """With both a firewall mismatch and a flag: protect_first fixes the firewall, respond_first removes."""
    env, state, obs, masks = world
    fw_actions = np.asarray(firewall_rulebook(None, obs, masks))
    agent = int(np.argmax(fw_actions != BLUE_SLEEP))
    assert fw_actions[agent] != BLUE_SLEEP  # some agent is out of compliance after 30 random steps
    slot, host = valid_host_slots(state, masks, agent)[0]
    o = with_flags(obs[agent], [slot])
    first = make_rulebook(RulebookConfig(remove=True))
    second = make_rulebook(RulebookConfig(remove=True, priority="respond_first"))
    assert run_one(first, obs, masks, agent, o) == fw_actions[agent]
    assert run_one(second, obs, masks, agent, o) == encode_blue_action("Remove", host, agent, const=state.const)


def test_firewall_only_config_equals_the_old_rulebook(world):
    env, state, obs, masks = world
    key = jax.random.PRNGKey(3)
    final, traj = R.make_rollout_fn(env, random_valid, 40)(state, key)
    for st in (state, final):
        o = jnp.stack([env.get_obs(st)[a] for a in E.BLUE_AGENTS])
        keys = R.step_keys(key, st.state.time)[1]
        for idle, default in (("sleep", BLUE_SLEEP), ("monitor", BLUE_MONITOR)):
            new = make_rulebook(RulebookConfig(idle=idle))(keys, o, masks)
            old = make_firewall_rulebook(default)(keys, o, masks)
            assert (np.asarray(new) == np.asarray(old)).all()


def test_rungs_and_config_validation():
    assert set(RUNGS) >= {"L1", "L2", "L3", "L5"}
    assert RUNGS["L5"].restore_prob > 0 and RUNGS["L2"].remove and not RUNGS["L1"].remove
    assert RUNGS["L3"].idle == "monitor"
    with pytest.raises(ValueError):
        make_rulebook(RulebookConfig(idle="nap"))
    with pytest.raises(ValueError):
        make_rulebook(RulebookConfig(restore_prob=1.5))


def test_rung_inside_coalition_rollouts_keeps_efficiency_and_endpoints(world):
    env, state, _, _ = world
    keys = jax.random.split(jax.random.PRNGKey(7), 2)
    ref = make_rulebook(RUNGS["L2"])
    n = 20
    v = np.asarray(R.make_coalition_returns_fn(env, random_valid, ref, n)(state, keys, jnp.asarray(coalition_masks())))
    for j in range(2):
        assert v[-1, j] == float(R.rollout(env, random_valid, state, keys[j], n)[1].rewards.sum())
        assert v[0, j] == float(R.rollout(env, ref, state, keys[j], n)[1].rewards.sum())
    phi = np.asarray(shapley_values(jnp.asarray(v)))
    np.testing.assert_allclose(phi.sum(axis=0), v[-1] - v[0], atol=1e-3)
