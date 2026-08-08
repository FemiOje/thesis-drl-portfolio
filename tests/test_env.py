"""Non-negotiable unit tests for the portfolio environment.

The four mandated checks:
  1. weights always sum to 1;
  2. with c = 0, final wealth == analytic product Π_t (y_t · w_{t-1})  (to 1e-8);
  3. μ_t iteration converges and lies in (0, 1];
  4. an all-cash policy yields r_t ≈ 0 every step.

Plus a few guardrail tests (spaces, determinism, softmax validity).

Run:  python -m pytest tests/test_env.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("env", "data"):
    p = os.path.join(_ROOT, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import data_loader as dl  # noqa: E402
import portfolio_env as pe  # noqa: E402


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def dataset():
    """Cached dataset — reads data/processed (or raw CSVs); never re-downloads."""
    return dl.build_dataset()


def _cash_biased_action(n: int) -> np.ndarray:
    """Action whose softmax is (essentially exactly) all-cash [1, 0, ..., 0]."""
    a = np.full(n, -50.0)
    a[0] = 50.0
    return a


# ---------------------------------------------------------------- test 1
def test_weights_always_sum_to_one(dataset):
    """(1) Applied weights sum to 1 for every step, under random actions."""
    rng = np.random.default_rng(0)
    env = pe.make_env("train", commission=0.0025, dataset=dataset,
                      episode_length=200, random_start=True)
    for ep in range(3):
        obs, _ = env.reset(seed=ep)
        # observed previous-weights are valid too
        assert np.isclose(obs["w_prev"].sum(), 1.0, atol=1e-6)
        term = trunc = False
        while not (term or trunc):
            a = rng.uniform(-pe.ACTION_BOUND, pe.ACTION_BOUND, size=env.action_space.shape)
            obs, r, term, trunc, info = env.step(a)
            w = info["weights"]
            assert np.all(w >= 0.0)
            assert np.isclose(w.sum(), 1.0, atol=1e-9), f"weights sum {w.sum()}"
            assert np.isclose(obs["w_prev"].sum(), 1.0, atol=1e-6)


# ---------------------------------------------------------------- test 2
def test_zero_cost_wealth_matches_analytic_product(dataset):
    """(2) With c = 0, env wealth == Π_t (y_{t+1} · w_t), computed independently."""
    env = pe.make_env("val", commission=0.0, dataset=dataset,
                      episode_length=None, random_start=False)
    rng = np.random.default_rng(42)
    env.reset(seed=0)

    # Build the cash-augmented price-relative table independently from the data.
    y_stocks = dataset.y
    cash = np.ones((y_stocks.shape[0], 1))
    y_ext = np.concatenate([cash, y_stocks], axis=1)  # (T, m+1)

    analytic = 1.0
    term = trunc = False
    while not (term or trunc):
        a = rng.uniform(-pe.ACTION_BOUND, pe.ACTION_BOUND, size=env.action_space.shape)
        _, r, term, trunc, info = env.step(a)
        t = info["t"]
        w = info["weights"]
        analytic *= float(np.dot(y_ext[t + 1], w))
        # reward must equal ln(gross) exactly when cost-free (μ == 1)
        assert np.isclose(info["mu"], 1.0, atol=1e-12)
        assert np.isclose(r, np.log(np.dot(y_ext[t + 1], w)), atol=1e-10)

    assert np.isclose(env.p, analytic, rtol=0, atol=1e-8), (
        f"env wealth {env.p!r} != analytic product {analytic!r}"
    )


# ---------------------------------------------------------------- test 3
def test_mu_converges_and_in_unit_interval():
    """(3) μ_t iteration converges into (0, 1] for many random rebalances."""
    rng = np.random.default_rng(7)
    m1 = 9  # m + cash
    for _ in range(2000):
        w_prime = rng.dirichlet(np.ones(m1))
        w = rng.dirichlet(np.ones(m1))
        for c in (0.001, 0.0025, 0.01):
            mu = pe.transaction_remainder(w_prime, w, c, c)
            assert 0.0 < mu <= 1.0, f"mu={mu} out of (0,1] at c={c}"

    # No-trade -> no cost -> mu == 1 exactly.
    w = rng.dirichlet(np.ones(m1))
    assert np.isclose(pe.transaction_remainder(w, w, 0.0025, 0.0025), 1.0, atol=1e-9)
    # Zero commission -> mu == 1 exactly for any rebalance.
    assert pe.transaction_remainder(rng.dirichlet(np.ones(m1)),
                                    rng.dirichlet(np.ones(m1)), 0.0, 0.0) == 1.0


# ---------------------------------------------------------------- test 4
def test_all_cash_policy_zero_reward(dataset):
    """(4) A hold-cash-only policy yields r_t ≈ 0 at every step."""
    for split in ("train", "val", "test"):
        env = pe.make_env(split, commission=0.0025, dataset=dataset,
                          episode_length=None, random_start=False)
        env.reset(seed=0)
        a = _cash_biased_action(env.action_space.shape[0])
        term = trunc = False
        while not (term or trunc):
            _, r, term, trunc, info = env.step(a)
            assert abs(r) < 1e-6, f"cash reward {r} on split {split}"
            assert np.isclose(info["mu"], 1.0, atol=1e-9)
        # Wealth is preserved (no gains, no losses, no costs).
        assert np.isclose(env.p, 1.0, atol=1e-6)


# ---------------------------------------------------------------- guardrails
def test_softmax_valid():
    rng = np.random.default_rng(1)
    for _ in range(1000):
        w = pe.softmax(rng.uniform(-50, 50, size=9))
        assert np.all(w >= 0.0)
        assert np.isclose(w.sum(), 1.0, atol=1e-12)


def test_observation_in_space(dataset):
    env = pe.make_env("test", dataset=dataset, random_start=False)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    a = env.action_space.sample()
    obs, _, _, _, _ = env.step(a)
    assert env.observation_space.contains(obs)


def test_drift_identity_on_flat_returns():
    """Eq. 7: with all y == 1, weights do not drift."""
    w = np.array([0.2, 0.1, 0.3, 0.4])
    y = np.ones(4)
    assert np.allclose(pe.drifted_weights(w, y), w)
