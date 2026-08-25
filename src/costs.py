"""Transaction remainder factor mu_t. Jiang Eq. 14-16, Theorem 1.

Two backends that must agree: numpy for the env, torch for PG's loss (the gradient
flows through mu, so the k iterations are unrolled explicitly and stay differentiable).
"""

import numpy as np
import torch

K_DEFAULT = 5


def _step_np(mu, w_drift, w_target, c_s, c_p):
    excess = np.clip(w_drift[..., 1:] - mu[..., None] * w_target[..., 1:], 0.0, None).sum(-1)
    return (1.0 - c_p * w_drift[..., 0] - (c_s + c_p - c_s * c_p) * excess) \
        / (1.0 - c_p * w_target[..., 0])


def _step_th(mu, w_drift, w_target, c_s, c_p):
    excess = torch.clamp(w_drift[..., 1:] - mu.unsqueeze(-1) * w_target[..., 1:], min=0.0).sum(-1)
    return (1.0 - c_p * w_drift[..., 0] - (c_s + c_p - c_s * c_p) * excess) \
        / (1.0 - c_p * w_target[..., 0])


def transaction_remainder(w_drift, w_target, c, k=K_DEFAULT, backend="numpy", c_p=None):
    """Solve Eq. 14 by the Eq. 15 iteration, warm-started at Eq. 16.

    w_drift  : w'_t, post-drift weights (Eq. 7).  (..., m+1), index 0 = cash
    w_target : w_t,  agent's target weights.      (..., m+1)
    returns  : mu in (0, 1], shape (...,)

    Worked two-asset example (cash + one risky, c = 0.01):
        w_drift = [0.5, 0.5], w_target = [0.0, 1.0]
        The agent buys 0.5 of the risky asset with cash, paying c on the purchase.
        Eq. 14 with c_s = c_p = c has denominator 1 - c*0 = 1, so
            mu = 1 - c*0.5 - (2c - c^2)*(0.5 - mu*1.0)^+
        The fixed point is mu = 0.99498..., i.e. just under the naive 1 - c*0.5 = 0.995,
        the difference being the second-order term. See test_costs.py.
    """
    c_s = c
    c_p = c if c_p is None else c_p
    if backend == "numpy":
        w_drift = np.asarray(w_drift, dtype=np.float64)
        w_target = np.asarray(w_target, dtype=np.float64)
        # Eq. 16 warm start: first-order cost estimate c * sum|w' - w|.
        mu = 1.0 - c_s * np.abs(w_drift[..., 1:] - w_target[..., 1:]).sum(-1)
        for _ in range(k):
            mu = _step_np(mu, w_drift, w_target, c_s, c_p)
        return np.clip(mu, 0.0, 1.0)
    if backend == "torch":
        mu = 1.0 - c_s * (w_drift[..., 1:] - w_target[..., 1:]).abs().sum(-1)
        for _ in range(k):
            mu = _step_th(mu, w_drift, w_target, c_s, c_p)
        return mu.clamp(0.0, 1.0)
    raise ValueError(f"backend must be 'numpy' or 'torch', got {backend!r}")


def drift(w_prev, y):
    """w'_t = (y_t * w_{t-1}) / (y_t . w_{t-1}) — Eq. 7. numpy or torch."""
    num = y * w_prev
    return num / num.sum(-1)[..., None]
