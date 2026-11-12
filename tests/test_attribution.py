import itertools
import math

import jax
import numpy as np
import pytest

from cc4lab.attribution import coalition_masks
from cc4lab.attribution.loo import loo_values
from cc4lab.attribution.shapley import shapley_values

N = 5
FULL = 2**N - 1


def _game(fn, n=N):
    """Coalition-value array from fn(set_of_players)."""
    return np.array([fn({i for i in range(n) if (c >> i) & 1}) for c in range(2**n)], dtype=np.float64)


def test_coalition_masks_layout():
    m = coalition_masks(N)
    assert m.shape == (32, 5) and m.dtype == bool
    assert not m[0].any() and m[FULL].all()
    assert m[5].tolist() == [True, False, True, False, False]  # bit i <-> agent i


def test_two_player_asymmetric_game():
    # v({0})=1, v({1})=2, v({0,1})=4: each gets own value + half the synergy.
    v = np.array([0.0, 1.0, 2.0, 4.0])
    np.testing.assert_allclose(shapley_values(v), [1.5, 2.5], atol=1e-6)


def test_additive_game_shapley_is_own_value():
    a = np.array([3.0, -1.0, 0.5, 7.0, 2.0])
    np.testing.assert_allclose(shapley_values(_game(lambda s: sum(a[i] for i in s))), a, atol=1e-5)


def test_symmetric_game_splits_evenly():
    v = _game(lambda s: len(s) ** 2 + 3.0)  # depends only on |S|
    np.testing.assert_allclose(shapley_values(v), np.full(N, (v[FULL] - v[0]) / N), atol=1e-5)


def test_three_player_majority_game():
    v = _game(lambda s: float(len(s) >= 2), n=3)
    np.testing.assert_allclose(shapley_values(v), [1 / 3] * 3, atol=1e-6)


def test_null_player_gets_zero():
    v = _game(lambda s: float(len(s - {2}) ** 2))  # agent 2 never matters
    assert abs(float(shapley_values(v)[2])) < 1e-5


def test_efficiency_on_random_game():
    v = np.random.default_rng(0).normal(size=32)
    phi = shapley_values(v)
    np.testing.assert_allclose(float(phi.sum()), v[FULL] - v[0], atol=1e-4)


def test_matches_brute_force_over_orderings():
    v = np.random.default_rng(1).normal(size=32)
    expected = np.zeros(N)
    for order in itertools.permutations(range(N)):
        coalition = 0
        for i in order:
            expected[i] += v[coalition | (1 << i)] - v[coalition]
            coalition |= 1 << i
    expected /= math.factorial(N)
    np.testing.assert_allclose(shapley_values(v), expected, atol=1e-5)


def test_trailing_batch_axes_and_jit():
    v = np.random.default_rng(2).normal(size=(32, 7, 3))
    phi = jax.jit(shapley_values)(v)
    assert phi.shape == (N, 7, 3)
    np.testing.assert_allclose(phi[:, 4, 1], shapley_values(v[:, 4, 1]), atol=1e-5)


def test_rejects_non_power_of_two_length():
    with pytest.raises(ValueError):
        shapley_values(np.zeros(30))


def test_loo_additive_game_is_own_value():
    a = np.array([3.0, -1.0, 0.5, 7.0, 2.0])
    np.testing.assert_allclose(loo_values(_game(lambda s: sum(a[i] for i in s))), a, atol=1e-5)


def test_loo_is_all_actual_minus_agent_swapped():
    v = np.random.default_rng(3).normal(size=32)
    expected = [v[FULL] - v[FULL ^ (1 << i)] for i in range(N)]
    np.testing.assert_allclose(loo_values(v), expected, atol=1e-6)


def test_loo_misses_redundancy_that_shapley_credits():
    # Any 2 of 3 agents suffice: removing one changes nothing (LOO = 0), yet
    # each agent is pivotal in some orderings (Shapley = 1/3).
    v = _game(lambda s: float(len(s) >= 2), n=3)
    np.testing.assert_allclose(loo_values(v), [0, 0, 0], atol=1e-6)
    assert np.all(np.asarray(shapley_values(v)) > 0.3)


def test_loo_trailing_batch_axes():
    v = np.random.default_rng(4).normal(size=(32, 6))
    assert loo_values(v).shape == (N, 6)
