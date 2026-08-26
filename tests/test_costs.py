"""Phase 2 gate: mu_t. Eq. 14-16."""

import numpy as np
import pytest
import torch

from src.costs import drift, transaction_remainder as mu_fn

RNG = np.random.default_rng(0)


def simplex(n, m):
    x = RNG.exponential(size=(n, m))
    return x / x.sum(-1, keepdims=True)


def test_zero_commission_is_exactly_one():
    w1, w2 = simplex(200, 9), simplex(200, 9)
    assert np.allclose(mu_fn(w1, w2, c=0.0), 1.0, atol=0, rtol=0)


def test_no_trade_is_one():
    w = simplex(200, 9)
    assert np.allclose(mu_fn(w, w, c=0.01), 1.0, atol=1e-12)


def test_bounded_on_random_simplex_pairs():
    w1, w2 = simplex(10_000, 9), simplex(10_000, 9)
    mu = mu_fn(w1, w2, c=0.001)
    assert (mu > 0).all() and (mu <= 1.0).all()


def test_monotone_decreasing_in_turnover():
    """Interpolating the target away from the drift raises turnover monotonically;
    mu must fall monotonically."""
    w_drift = np.array([0.1, 0.4, 0.3, 0.2])
    far = np.array([0.7, 0.1, 0.1, 0.1])
    mus, turns = [], []
    for a in np.linspace(0.0, 1.0, 25):
        w_t = (1 - a) * w_drift + a * far
        mus.append(float(mu_fn(w_drift, w_t, c=0.005)))
        turns.append(np.abs(w_t - w_drift).sum())
    assert np.all(np.diff(turns) > 0)
    assert np.all(np.diff(mus) < 0)


def test_numpy_and_torch_agree():
    w1, w2 = simplex(5000, 9), simplex(5000, 9)
    a = mu_fn(w1, w2, c=0.001)
    b = mu_fn(torch.tensor(w1), torch.tensor(w2), c=0.001, backend="torch").numpy()
    assert np.abs(a - b).max() < 1e-6


def test_worked_two_asset_purchase():
    """Pure purchase, cash 0.5 -> risky 1.0 at c = 0.01. The sales term (w'_i - mu w_i)^+
    is zero because the holding grows, so Eq. 14 reduces to 1 - c_p w'_0 = 0.995."""
    mu = float(mu_fn(np.array([0.5, 0.5]), np.array([0.0, 1.0]), c=0.01, k=60))
    assert abs(mu - 0.995) < 1e-12


def test_worked_two_asset_sale():
    """Pure sale, risky 1.0 -> cash 0.5 at c = 0.01. Exercises both terms:
        mu = (1 - k(1 - 0.5 mu)) / (1 - 0.01*0.5),   k = 2c - c^2
    so mu (0.995 - k/2) = 1 - k."""
    c = 0.01
    k = 2 * c - c * c
    mu = float(mu_fn(np.array([0.0, 1.0]), np.array([0.5, 0.5]), c=c, k=60))
    assert abs(mu - (1 - k) / (1 - c * 0.5 - k * 0.5)) < 1e-12
    assert 0.994 < mu < 0.995


def test_iteration_has_converged_at_k5():
    w1, w2 = simplex(2000, 9), simplex(2000, 9)
    a = mu_fn(w1, w2, c=0.001, k=5)
    b = mu_fn(w1, w2, c=0.001, k=50)
    assert np.abs(a - b).max() < 1e-9


def test_gradient_flows_through_mu():
    w_drift = torch.tensor(simplex(16, 9))
    w_target = torch.tensor(simplex(16, 9), requires_grad=True)
    mu = mu_fn(w_drift, w_target, c=0.001, backend="torch")
    assert mu.requires_grad
    mu.sum().backward()
    assert w_target.grad is not None and torch.isfinite(w_target.grad).all()
    assert w_target.grad.abs().sum() > 0


def test_larger_commission_costs_more():
    w1, w2 = simplex(500, 9), simplex(500, 9)
    assert (mu_fn(w1, w2, c=0.01) <= mu_fn(w1, w2, c=0.001) + 1e-12).all()


def test_drift_is_a_simplex_and_matches_eq7():
    w = simplex(100, 9)
    y = 1 + RNG.normal(0, 0.02, (100, 9))
    y[:, 0] = 1.0
    d = drift(w, y)
    assert np.allclose(d.sum(-1), 1.0)
    assert np.allclose(d, (y * w) / (y * w).sum(-1, keepdims=True))


def test_no_price_move_leaves_weights_unchanged():
    w = simplex(50, 9)
    y = np.ones((50, 9))
    assert np.allclose(drift(w, y), w)
