"""Gymnasium portfolio-management environment.

Implements the environment / formalism of Jiang, Xu & Liang (2017)
("A Deep Reinforcement Learning Framework for the Financial Portfolio
Management Problem", arXiv:1706.10059). Equation numbers below refer to the aforementioned
paper.

Per-step formalism
------------------
* Observation (Dict -> forces SB3 ``MultiInputPolicy``):
    - ``X`` : normalized price tensor X_t of shape (F=3, n=50, m)   (Eq. 18)
    - ``w_prev`` : previous portfolio weights, length m+1 (index 0 = cash).
* Action: a raw real vector in R^{m+1}; the env applies a **softmax** so any
  vector maps to a valid weight vector (>= 0, sums to 1).
* Price relative: y_t (Eq. 1), cash entry == 1 (prepended here).
* Weight drift within a period: w'_t = (y ⊙ w) / (y · w)               (Eq. 7).
* Transaction remainder factor μ_t ∈ (0,1]: fixed-point iteration       (Eq. 14-16).
* Reward: r_t = ln(μ_t · y_t · w_{t-1})                                 (Eq. 10).

Timing convention (documented for the write-up)
-----------------------------------------------
At decision index ``t`` the agent chooses target weights ``w_t = softmax(a)``.
The transaction cost μ_t is charged for moving from the drifted actual
holdings into ``w_t``. The reward is then
``ln(μ_t · (y·w_t))``, where ``y = y[t+1]`` is the price relative realized while
``w_t`` is held. This "forward" indexing matches Jiang Eq. 10 in form while
giving the RL agent immediate credit for the allocation it just chose. With
zero commission μ_t ≡ 1, so episode wealth is exactly Π_t (y·w_t) — the analytic
product checked by ``tests/test_env.py``.

Assumptions (documented, not solved): zero slippage, zero market
impact; orders fill at the close.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

try:  # gymnasium is the SB3 API; import lazily-friendly for tooling
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError("gymnasium is required for the portfolio environment") from exc

# Make the sibling data/ package importable regardless of CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if os.path.join(_ROOT, "data") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "data"))
import data_loader as dl  # noqa: E402

DEFAULT_COMMISSION: float = 0.0025  # 0.25% both sides (Jiang default; also test 0.1%)
# Softmax logit bound. Because softmax is shift-invariant, this constrains the
# *spread* between the highest and lowest score (2*bound), not the absolute
# score. Chosen by measurement, not taste: a working PPO policy uses a spread of
# 6.90, so a bound of 5 (spread 10) leaves ~45% headroom and still permits
# 99.96% concentration in a single asset.
#
# It was originally 10. That let DDPG's deterministic actor rail its tanh against
# the bound (100% of outputs saturated -> zero gradient -> the policy froze), and
# turned a saturated action into a 485,000,000:1 allocation ratio. At 5 the
# saturated-case ratio is 22,000:1. Pass ``action_bound=10.0`` to reproduce the
# original *training* environment. (Evaluating an old checkpoint needs no such
# care: SB3 clips predictions to the model's own stored action space, which
# travels inside the .zip, and this env never clips what it is handed.)
ACTION_BOUND: float = 5.0


# =============================================================================
# Core math (module-level + independently unit-testable)
# =============================================================================
def softmax(a: np.ndarray) -> np.ndarray:
    """Map a raw real action vector to valid portfolio weights (>=0, sum 1).

    Numerically stable (subtract the max). This is the "softmax inside the env"
    that keeps SB3's unbounded Gaussian / deterministic outputs valid.
    """
    a = np.asarray(a, dtype=np.float64)
    z = a - np.max(a)
    e = np.exp(z)
    return e / np.sum(e)


def drifted_weights(w: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Weight drift over a period — Jiang Eq. 7.

    Given weights ``w`` held at the start of a period and the period's price
    relative ``y`` (cash entry == 1), returns the drifted weights
    ``w' = (y ⊙ w) / (y · w)``.
    """
    num = y * w
    return num / np.sum(num)


def transaction_remainder(
    w_prime: np.ndarray,
    w: np.ndarray,
    c_s: float = DEFAULT_COMMISSION,
    c_p: float = DEFAULT_COMMISSION,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> float:
    """Transaction remainder factor μ_t ∈ (0,1] — Jiang Eq. 14-16.

    Fraction of portfolio wealth surviving the rebalance from the drifted
    weights ``w_prime`` (w'_t, Eq. 7) to the new target weights ``w`` (w_t).
    Index 0 is cash. ``c_s`` / ``c_p`` are the selling / purchasing commission
    rates.

    Solves Jiang's implicit Eq. 14 by fixed-point iteration
    ``μ ← f(μ)`` with
        f(μ) = [1 - c_p·w'_0 - (c_s+c_p-c_s·c_p)·Σ_i (w'_i - μ·w_i)^+ ]
               / (1 - c_p·w_0)
    initialized at μ⁰ = c·Σ_i |w'_i - w_i| (Eq. 16), iterated to |Δμ| < ``tol``.
    (x)^+ = max(x, 0). With c_s = c_p = 0 this returns exactly 1.
    """
    w_prime = np.asarray(w_prime, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    w0_prime = w_prime[0]
    w0 = w[0]
    risky_prime = w_prime[1:]
    risky = w[1:]

    denom = 1.0 - c_p * w0
    k = c_s + c_p - c_s * c_p

    # Initial guess (Eq. 16), using c = c_s as the representative rate.
    mu = c_s * float(np.sum(np.abs(risky_prime - risky)))
    for _ in range(max_iter):
        nxt = (1.0 - c_p * w0_prime - k * float(np.sum(np.maximum(risky_prime - mu * risky, 0.0)))) / denom
        if abs(nxt - mu) < tol:
            mu = nxt
            break
        mu = nxt
    return float(mu)


# =============================================================================
# Environment
# =============================================================================
@dataclass
class StepRecord:
    """One step of episode history (for evaluation and plotting)."""

    t: int
    date: np.datetime64
    reward: float
    wealth: float
    mu: float
    gross_return: float
    turnover: float
    weights: np.ndarray


class PortfolioEnv(gym.Env):
    """Continuous multi-asset portfolio-allocation env (Jiang formalism).

    Parameters
    ----------
    dataset : dl.Dataset
        Output of ``data_loader.build_dataset`` (cached; never re-downloads).
    start, end : int
        Half-open panel-index range [start, end) for this env's split.
    commission : float
        Symmetric commission rate c_s = c_p (default 0.25%). Pass 0.001 to
        replicate Jiang's 0.1% run, or 0.0 for the analytic (costless) test.
    episode_length : int or None
        If given, each episode is a sub-window of this many decision steps
        (variety during training). If None, one full pass over the split.
    random_start : bool
        If True (and ``episode_length`` set), sample the episode start
        uniformly at random within the split. Order within an episode is always
        chronological — time is never shuffled.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataset: "dl.Dataset",
        start: int,
        end: int,
        commission: float = DEFAULT_COMMISSION,
        episode_length: int | None = None,
        random_start: bool = False,
        action_bound: float = ACTION_BOUND,
    ) -> None:
        super().__init__()
        self.ds = dataset
        self.window = dataset.window
        self.m = dataset.n_assets
        self.c_s = float(commission)
        self.c_p = float(commission)
        self.episode_length = episode_length
        self.random_start = random_start
        self.action_bound = float(action_bound)

        # Price relatives with a cash column (== 1) prepended -> (T, m+1).
        y_stocks = dataset.y                      # (T, m)
        cash = np.ones((y_stocks.shape[0], 1), dtype=np.float64)
        self.y_ext = np.concatenate([cash, y_stocks], axis=1)  # (T, m+1)

        # Valid decision indices for this split.
        #   need `window` days of history for X_t  -> t >= window-1
        #   need next-period return y[t+1] (< end)  -> t <= end-2
        # (Windows may reach back across a split boundary — observable history,
        #  not future leakage; see the data-pipeline note in docs/WORKLOG.md.)
        self._t_first = max(start, self.window - 1)
        self._t_last = end - 2
        if self._t_last < self._t_first:
            raise ValueError(
                f"Split [{start},{end}) too short for window={self.window}."
            )

        n_assets_cash = self.m + 1
        self.observation_space = spaces.Dict(
            {
                "X": spaces.Box(
                    low=0.0, high=np.inf,
                    shape=(len(dl.FEATURES), self.window, self.m),
                    dtype=np.float32,
                ),
                "w_prev": spaces.Box(
                    low=0.0, high=1.0, shape=(n_assets_cash,), dtype=np.float32
                ),
            }
        )
        self.action_space = spaces.Box(
            low=-self.action_bound, high=self.action_bound,
            shape=(n_assets_cash,), dtype=np.float32,
        )

        # Episode state (initialized in reset()).
        self.t: int = self._t_first
        self._t_end: int = self._t_last
        self.p: float = 1.0
        self.w_prev_action = self._all_cash()
        self.w_hold = self._all_cash()
        self.history: list[StepRecord] = []

    # ------------------------------------------------------------------ utils
    def _all_cash(self) -> np.ndarray:
        w = np.zeros(self.m + 1, dtype=np.float64)
        w[0] = 1.0
        return w

    def _obs(self) -> dict[str, np.ndarray]:
        return {
            "X": self.ds.tensor(self.t).astype(np.float32),
            "w_prev": self.w_prev_action.astype(np.float32),
        }

    # ------------------------------------------------------------------ API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        if self.episode_length is not None:
            span = self.episode_length
            hi = self._t_last - span + 1
            if hi < self._t_first:  # split shorter than requested episode
                start = self._t_first
                self._t_end = self._t_last
            elif self.random_start:
                start = int(self.np_random.integers(self._t_first, hi + 1))
                self._t_end = start + span - 1
            else:
                start = self._t_first
                self._t_end = min(self._t_first + span - 1, self._t_last)
        else:
            start = self._t_first
            self._t_end = self._t_last

        self.t = start
        self.p = 1.0
        self.w_prev_action = self._all_cash()
        self.w_hold = self._all_cash()
        self.history = []
        return self._obs(), {}

    def step(self, action: np.ndarray):
        t = self.t
        w_new = softmax(action)                       # target weights (Eq.: softmax-in-env)

        # Transaction cost: move from drifted actual holdings -> target (Eq. 14-16).
        w_hold_before = self.w_hold
        mu = transaction_remainder(w_hold_before, w_new, self.c_s, self.c_p)
        turnover = float(np.sum(np.abs(w_new - w_hold_before)))

        # Realize the coming period's return and update wealth (Eq. 10 / Eq. 11).
        y = self.y_ext[t + 1]                          # cash entry == 1 (Eq. 1)
        gross = float(np.dot(y, w_new))                # y_t · w_t
        port_mult = mu * gross                         # μ_t · (y_t · w_t)
        reward = float(np.log(port_mult))
        self.p *= port_mult

        # Drift the held weights over the period for the next step's cost (Eq. 7).
        self.w_hold = drifted_weights(w_new, y)
        self.w_prev_action = w_new

        rec = StepRecord(
            t=t,
            date=self.ds.dates.values[t + 1],
            reward=reward,
            wealth=self.p,
            mu=mu,
            gross_return=gross,
            turnover=turnover,
            weights=w_new.copy(),
        )
        self.history.append(rec)

        truncated = t >= self._t_end
        terminated = False
        self.t = t + 1

        info = {
            "t": t,
            "wealth": self.p,
            "mu": mu,
            "gross_return": gross,
            "turnover": turnover,
            "weights": w_new,
        }
        return self._obs(), reward, terminated, truncated, info

def make_env(
    split: str = "train",
    commission: float = DEFAULT_COMMISSION,
    episode_length: int | None = None,
    random_start: bool | None = None,
    dataset: "dl.Dataset | None" = None,
    action_bound: float = ACTION_BOUND,
) -> PortfolioEnv:
    """Build a PortfolioEnv for a named split ("train"/"val"/"test").

    Loads the cached dataset if none is supplied (never re-downloads). Defaults
    ``random_start`` to True for training only. ``action_bound`` must match the
    bound a checkpoint was trained under (see the ACTION_BOUND note above).
    """
    if dataset is None:
        dataset = dl.build_dataset()
    start, end = getattr(dataset.splits, split)
    if random_start is None:
        random_start = split == "train"
    return PortfolioEnv(
        dataset,
        start,
        end,
        commission=commission,
        episode_length=episode_length,
        random_start=random_start,
        action_bound=action_bound,
    )


if __name__ == "__main__":
    # Quick self-check: run one full train-split episode with random actions.
    env = make_env("train", episode_length=None, random_start=False)
    obs, _ = env.reset(seed=0)
    total, steps = 0.0, 0
    term = trunc = False
    while not (term or trunc):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        total += r
        steps += 1
    print("=== PortfolioEnv self-check ===")
    print(f"obs keys        : {list(obs.keys())}")
    print(f"X shape         : {obs['X'].shape}  (F, n, m)")
    print(f"w_prev shape    : {obs['w_prev'].shape}  (m+1)")
    print(f"steps in episode: {steps}")
    print(f"final wealth    : {env.p:.4f}   (random policy)")
    print(f"sum(reward)     : {total:.4f}")
    print(f"weights sum     : {info['weights'].sum():.6f}")
    print("ENV OK.")
