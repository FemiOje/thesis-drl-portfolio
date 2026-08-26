"""UBAH, UCRP, Best Stock, rolling Markowitz — all as policy_fn(obs) -> action."""

import numpy as np
from scipy.optimize import minimize

from .costs import drift


def weights_to_action(w, tau):
    """Invert w = softmax(tau * a) subject to a in [-1, 1].

    The action space is bounded because DDPG squashes through tanh, so only weight
    vectors with max/min ratio <= exp(2*tau) are exactly representable. Weights are
    floored at max/exp(2*tau) first, which perturbs a one-hot target by ~1/exp(2*tau)
    (4.5e-5 at tau=5) and leaves equal weights untouched. backtest() records the
    realised weights, so the residual is measured rather than assumed.
    """
    w = np.asarray(w, float)
    w = np.clip(w, w.max() * np.exp(-2.0 * tau), None)
    w = w / w.sum()
    a = np.log(w) / tau
    a -= 0.5 * (a.max() + a.min())          # centre the range inside [-1, 1]
    return np.clip(a, -1.0, 1.0)


def y_from_obs(obs):
    """Price relatives for this step, recovered from the observation alone.

    Channel 0 is close/latest close, so column -1 is 1 and column -2 is
    close[t-1]/close[t]. Hence y_t = 1 / X[0, :, -2]. Cash prepended as 1.0.

    The observation is float32 (SB3 requires it), so y is recovered to ~1e-7. Policies
    that hold by targeting the drift therefore trade a residual ~1e-7 per step. Measured
    end-to-end that is a wealth leak of 8e-9 (UBAH) to 3e-7 (BestStock) over a 249-day
    split — far below any reported digit, and the agents read the same float32 tensor.
    """
    prev = obs["tensor"][0, :, -2]
    return np.concatenate([[1.0], 1.0 / prev.astype(np.float64)])


class _Base:
    def __init__(self, n_assets, tau):
        self.m, self.tau = n_assets, tau

    def reset(self):
        self.t = 0

    def _act(self, w):
        self.t += 1
        return weights_to_action(w, self.tau)


class UCRP(_Base):
    """Rebalance to equal weights every step."""

    def __call__(self, obs):
        return self._act(np.full(self.m + 1, 1.0 / (self.m + 1)))


class UBAH(_Base):
    """Equal weights at t=0, never rebalance: target the drifted weights so mu = 1."""

    def __call__(self, obs):
        if self.t == 0:
            return self._act(np.full(self.m + 1, 1.0 / (self.m + 1)))
        return self._act(drift(obs["weights"].astype(np.float64), y_from_obs(obs)))


class BestStock(_Base):
    """Single best asset over the evaluation window, chosen in hindsight, then held.

    An upper reference, not a strategy: it is not implementable. Reported as such.
    """

    def __init__(self, n_assets, tau, close_split):
        super().__init__(n_assets, tau)
        total = close_split[-1] / close_split[0]
        self.best = int(np.argmax(total))

    def __call__(self, obs):
        if self.t == 0:
            w = np.zeros(self.m + 1)
            w[self.best + 1] = 1.0
            return self._act(w)
        return self._act(drift(obs["weights"].astype(np.float64), y_from_obs(obs)))


class Markowitz(_Base):
    """Rolling 252-day covariance, long-only max-Sharpe, monthly rebalance.

    Reads the close panel by absolute index and never past the current decision bar.
    Between rebalances it holds (targets the drift), so it pays no cost.
    """

    def __init__(self, n_assets, tau, close, indices, lookback=252, every=21):
        super().__init__(n_assets, tau)
        self.close = np.asarray(close, float)
        self.indices = np.asarray(indices)
        self.lookback, self.every = lookback, every

    def reset(self):
        super().reset()
        self.w = None

    def _solve(self, end):
        lo = max(0, end - self.lookback)
        r = self.close[lo + 1:end + 1] / self.close[lo:end] - 1.0
        if len(r) < 20:
            return np.full(self.m + 1, 1.0 / (self.m + 1))
        mean, cov = r.mean(0), np.cov(r, rowvar=False) + 1e-8 * np.eye(self.m)

        def neg_sharpe(x):
            v = np.sqrt(x @ cov @ x)
            return -(mean @ x) / v if v > 0 else 0.0

        res = minimize(neg_sharpe, np.full(self.m, 1.0 / self.m), method="SLSQP",
                       bounds=[(0.0, 1.0)] * self.m,
                       constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1.0}],
                       options={"maxiter": 200, "ftol": 1e-9})
        x = res.x if res.success else np.full(self.m, 1.0 / self.m)
        x = np.clip(x, 0, None)
        return np.concatenate([[0.0], x / x.sum()])

    def __call__(self, obs):
        if self.t % self.every == 0:
            self.w = self._solve(int(self.indices[self.t]))   # bars 0..t only
            return self._act(self.w)
        return self._act(drift(obs["weights"].astype(np.float64), y_from_obs(obs)))


def build(cfg, ds, split):
    m, tau = cfg.universe.n_assets, cfg.env.tau
    idx = ds["splits"][split]
    close = ds["panel"]["close"].values
    return {
        "UBAH": UBAH(m, tau),
        "UCRP": UCRP(m, tau),
        "BestStock": BestStock(m, tau, close[idx]),
        "Markowitz": Markowitz(m, tau, close, idx),
    }
