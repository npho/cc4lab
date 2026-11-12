"""Failure profiles and branch-point selection."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cc4lab import env as E
from cc4lab import rollout as R
from cc4lab.attribution import coalition_masks
from cc4lab.failures import TERMS, find_windows, observed_profile, plan
from cc4lab.policies.trivial import always_sleep, random_valid

T = 500
LWF, ASF, RIA = (TERMS.index(t) for t in ("lwf", "asf", "ria"))


def synthetic():
    p = np.zeros((T, 9, 3))
    p[100:110, 3, RIA] = -5.0  # big impact burst on subnet 3
    p[300:305, 6, LWF] = -4.0  # smaller work-failure burst on subnet 6
    p[50, 1, LWF] = -30.0  # a single huge LWF spike elsewhere
    return p


def test_worst_windows_are_ranked_and_do_not_overlap():
    wins = find_windows(synthetic(), window=10, n=3)
    assert [w.rank for w in wins] == [0, 1, 2]
    assert 91 <= wins[0].start <= 100 and wins[0].subnet == 3 and wins[0].term == "ria"  # contains the burst
    rewards = [w.reward for w in wins]
    assert rewards == sorted(rewards)  # worst (most negative) first
    spans = sorted((w.start, w.end) for w in wins)
    assert all(a_end <= b_start for (_, a_end), (b_start, _) in zip(spans, spans[1:]))


def test_term_filter_ignores_other_terms():
    ria = find_windows(synthetic(), window=10, n=1, term="ria")[0]
    lwf = find_windows(synthetic(), window=10, n=1, term="lwf")[0]
    assert ria.term == "ria" and 91 <= ria.start <= 100
    assert lwf.term == "lwf" and lwf.start <= 50 < lwf.end  # the -30 spike beats the -4 burst (-20 total)


def test_total_term_sums_everything():
    w = find_windows(synthetic(), window=10, n=1, term="total")[0]
    assert w.reward == pytest.approx(-50.0)  # the 10-step RIA burst: 10 x -5


def test_bad_arguments():
    with pytest.raises(ValueError):
        find_windows(synthetic(), window=T + 1, n=1)
    with pytest.raises(ValueError):
        find_windows(synthetic(), window=10, n=1, term="nope")


def test_plan_clamps_to_episode():
    w = find_windows(synthetic(), window=10, n=1, term="lwf")[0]  # starts near step 41..50
    branch_t, horizon = plan(w, lookback=100, tail=50, episode_length=T)
    assert branch_t == 0 and horizon == w.end + 50
    late = np.zeros((T, 9, 3))
    late[495:, 0, LWF] = -1.0
    last = find_windows(late, window=10, n=1)[0]
    assert last.end == T
    branch_t, horizon = plan(last, lookback=20, tail=50, episode_length=T)
    assert (branch_t, horizon) == (T - 30, 30)  # tail cannot run past the end of the episode


@pytest.fixture(scope="module")
def world():
    env = E.make_env()
    key = jax.random.PRNGKey(0)
    _, state = E.reset(env, key)
    return env, key, state


def test_profile_equals_team_reward_when_no_action_cost(world):
    env, key, state = world
    n = 40
    prof = np.asarray(jax.jit(lambda s, k: observed_profile(env, always_sleep, s, k, n))(state, key))
    assert prof.shape == (n, 9, 3)
    traj = R.rollout(env, always_sleep, state, key, n)[1]  # nobody restores: reward is exactly the 3 terms
    np.testing.assert_allclose(prof.sum(axis=(1, 2)), np.asarray(traj.rewards), atol=1e-4)


def test_observed_key_replays_exactly_from_a_branch(world):
    """The ex-post claim: branching at t with the observed key reproduces the observed return from t."""
    env, key, state = world
    n, t = 40, 15
    traj = R.rollout(env, random_valid, state, key, n)[1]
    observed_tail = float(np.asarray(traj.rewards)[t:].sum())
    s_t = R.rollout(env, random_valid, state, key, t)[0]
    fn = R.make_coalition_returns_fn(env, random_valid, always_sleep, n - t)
    v = np.asarray(fn(s_t, key[None], jnp.asarray(coalition_masks()[[31]])))
    assert v[0, 0] == observed_tail
