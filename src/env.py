"""PortfolioEnv. State (X_t, w_{t-1}) Eq. 20, action = raw scores, reward Eq. 10."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .costs import drift, transaction_remainder


def project_to_simplex(action, tau):
    """w = softmax(tau * action). THE single projection — env, PG actor and the
    reachability test all import this one. tau is config, never a literal."""
    a = np.asarray(action, dtype=np.float64) * tau
    e = np.exp(a - a.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def hhi(w):
    return float((w ** 2).sum())


def entropy(w):
    p = np.clip(w, 1e-12, None)
    return float(-(p * np.log(p)).sum())


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, X, y, cfg, indices=None):
        """X (T, f, m, n) observation tensors, y (T, m+1) price relatives with cash at
        index 0. indices selects the episode's decision steps (a split)."""
        super().__init__()
        self.X, self.y, self.cfg = X, y, cfg
        self.idx = np.arange(len(X)) if indices is None else np.asarray(indices)
        f, m, n = X.shape[1:]
        self.m = m
        self.tau = cfg.env.tau
        self.c = cfg.env.commission
        self.k = cfg.env.mu_iterations
        self.observation_space = spaces.Dict({
            "tensor": spaces.Box(0.0, np.inf, (f, m, n), np.float32),
            "weights": spaces.Box(0.0, 1.0, (m + 1,), np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, (m + 1,), np.float32)

    def _obs(self):
        return {"tensor": self.X[self.idx[self.t]].astype(np.float32),
                "weights": self.w_prev.astype(np.float32)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.w_prev = np.zeros(self.m + 1)
        self.w_prev[0] = 1.0                 # all capital in cash, Eq. 5
        self.value = self.cfg.env.initial_capital
        return self._obs(), {}

    def step(self, action):
        y = self.y[self.idx[self.t]].astype(np.float64)
        w_target = project_to_simplex(action, self.tau)
        gross = float(y @ self.w_prev)                    # period return on w_{t-1}
        w_drift = drift(self.w_prev, y)                   # Eq. 7
        mu = float(transaction_remainder(w_drift, w_target, self.c, self.k))  # Eq. 14
        reward = float(np.log(mu * gross))                # Eq. 10
        self.value *= mu * gross
        turnover = float(np.abs(w_target - w_drift).sum())

        info = {"weights": w_target, "mu": mu, "gross": gross, "value": self.value,
                "turnover": turnover, "hhi": hhi(w_target), "entropy": entropy(w_target),
                "max_weight": float(w_target.max())}
        self.w_prev = w_target
        self.t += 1
        done = self.t >= len(self.idx)
        obs = self._obs() if not done else {
            "tensor": self.X[self.idx[-1]].astype(np.float32),
            "weights": self.w_prev.astype(np.float32)}
        return obs, reward, done, False, info
