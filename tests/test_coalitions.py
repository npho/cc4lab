"""Coalition-masked rollouts: composition, CRN, cost modes, and Shapley efficiency on real rollouts."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxborg.actions.encoding import BLUE_RESTORE_END, BLUE_RESTORE_START

from cc4lab import env as E
from cc4lab import rollout as R
from cc4lab.attribution import coalition_masks
from cc4lab.attribution.loo import loo_values
from cc4lab.attribution.shapley import shapley_values
from cc4lab.policies.trivial import always_sleep, random_valid

N_STEPS = 30
FULL = 2**E.N_BLUE - 1


@pytest.fixture(scope="module")
def world():
    env = E.make_env()
    key = jax.random.PRNGKey(0)
    _, state = E.reset(env, key)
    keys = jax.random.split(jax.random.PRNGKey(1), 3)
    return env, state, keys


def always_restore(keys, obs, masks):
    """Submit some valid Restore every step (busy or not)."""
    del keys, obs
    return (jnp.argmax(masks[:, BLUE_RESTORE_START:BLUE_RESTORE_END], axis=1) + BLUE_RESTORE_START).astype(jnp.int32)


def plain_return(env, policy, state, key):
    return float(R.rollout(env, policy, state, key, N_STEPS)[1].rewards.sum())


def test_full_and_empty_coalitions_reduce_to_single_policy_rollouts(world):
    env, state, keys = world
    fn = R.make_coalition_returns_fn(env, random_valid, always_sleep, N_STEPS)
    v = np.asarray(fn(state, keys, jnp.asarray(coalition_masks())))
    assert v.shape == (32, 3)
    for j in range(3):
        assert v[FULL, j] == plain_return(env, random_valid, state, keys[j])
        assert v[0, j] == plain_return(env, always_sleep, state, keys[j])


def test_mixed_coalition_matches_manual_composition(world):
    env, state, keys = world
    member = np.array([False, False, True, False, True])
    c = int(sum(1 << i for i in range(5) if member[i]))
    fn = R.make_coalition_returns_fn(env, random_valid, always_sleep, N_STEPS)
    v = np.asarray(fn(state, keys, jnp.asarray(coalition_masks())))

    def composed(k, o, m):
        return jnp.where(jnp.asarray(member), random_valid(k, o, m), always_sleep(k, o, m))

    for j in range(3):
        assert v[c, j] == plain_return(env, composed, state, keys[j])


def test_busy_corrected_removes_exactly_the_ignored_restore_costs(world):
    env, state, keys = world
    raw = R.make_coalition_returns_fn(env, always_restore, always_sleep, N_STEPS)
    fixed = R.make_coalition_returns_fn(env, always_restore, always_sleep, N_STEPS, cost_mode="busy_corrected")
    masks = jnp.asarray(coalition_masks()[[0, 1, FULL]])
    v_raw, v_fix = np.asarray(raw(state, keys, masks)), np.asarray(fixed(state, keys, masks))
    assert np.array_equal(v_raw[0], v_fix[0])  # nobody submits Restore in the empty coalition
    assert (v_fix[2] > v_raw[2]).all()  # five agents repeatedly restoring while busy

    traj = R.rollout(env, always_restore, state, keys[0], N_STEPS)[1]
    ignored = np.asarray(R.ignored_restore_count(traj)).sum()
    assert ignored > 0
    assert v_fix[2, 0] - v_raw[2, 0] == ignored  # each ignored Restore was charged exactly -1


def test_shapley_efficiency_holds_on_real_rollouts(world):
    env, state, keys = world
    fn = R.make_coalition_returns_fn(env, random_valid, always_sleep, N_STEPS)
    v = fn(state, keys, jnp.asarray(coalition_masks()))
    phi = shapley_values(v)
    assert phi.shape == (5, 3)
    np.testing.assert_allclose(np.asarray(phi.sum(0)), np.asarray(v[FULL] - v[0]), atol=1e-2)
    assert loo_values(v).shape == (5, 3)


@pytest.mark.skipif(not os.environ.get("CC4LAB_SLOW"), reason="vmap compiles for minutes on CPU; set CC4LAB_SLOW=1")
def test_vmap_and_map_agree(world):
    env, state, keys = world
    masks = jnp.asarray(coalition_masks()[[0, 5, FULL]])
    a =R.make_coalition_returns_fn(env, random_valid, always_sleep, 10, batch_mode="map")(state, keys[:2], masks)
    b = R.make_coalition_returns_fn(env, random_valid, always_sleep, 10, batch_mode="vmap")(state, keys[:2], masks)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-3)


def test_rejects_unknown_modes(world):
    env, *_ = world
    with pytest.raises(ValueError):
        R.make_coalition_returns_fn(env, random_valid, always_sleep, 5, batch_mode="pmap")
    with pytest.raises(ValueError):
        R.make_coalition_returns_fn(env, random_valid, always_sleep, 5, cost_mode="free")
