"""Policy-degeneracy diagnostic.

Asks one question of every trained checkpoint: "does the learned policy respond
to the market observation at all?"

For each (algo, seed) the deterministic policy is run over the test window and
the portfolio weights it emits are recorded. If the agent were trading, those
weights would move as the price tensor X_t changes. The diagnostic reports, per
model:

* ``max weight range``  — the largest peak-to-trough movement of any single
  weight over the window. A trading policy shows O(0.1); a constant policy shows
  numerical noise (< 1e-4).
* ``mean |w_t - w̄|``    — average absolute deviation from the time-mean weight.
* ``verdict``           — CONSTANT if the policy never meaningfully moves.

It also runs a harder control: the same policy is queried on observations drawn
from *widely separated* market conditions (the first, middle and last valid
index of each split), which are far less correlated than consecutive days.

Interpretation is in the README and docs/METHODOLOGY.md §9.2. In short: every
agent trained here collapsed to a fixed allocation and ignores X_t, so the
reported fAPV differences are differences between *constant portfolios*, not
between trading strategies.

Run:  python experiments/policy_diagnostic.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("env", "data"):
    _p = os.path.join(_ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import data_loader as dl  # noqa: E402
import portfolio_env as pe  # noqa: E402

MODELS_DIR = os.path.join(_ROOT, "results", "models")
ALGOS = ["ppo", "a2c", "ddpg"]
ALGO_LABEL = {"ppo": "PPO", "a2c": "A2C (PG)", "ddpg": "DDPG"}

# A policy whose weights move less than this over a whole window is not
# responding to the market in any economically meaningful way. Set far above
# float32 noise (~1e-6) and far below any real reallocation (~1e-1).
CONSTANT_TOL = 1e-4


def load(algo: str, seed: int, models_dir: str, checkpoint: str = "best"):
    from stable_baselines3 import A2C, DDPG, PPO

    cls = {"ppo": PPO, "a2c": A2C, "ddpg": DDPG}[algo]
    path = os.path.join(models_dir, algo, f"seed{seed}", f"{checkpoint}_model.zip")
    if not os.path.exists(path):
        return None
    return cls.load(path, device="cpu")


def weights_over_window(model, env) -> np.ndarray:
    """Deterministic weights emitted over one full pass. Returns (T, m+1)."""
    obs, _ = env.reset(seed=0)
    out, done = [], False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        out.append(pe.softmax(action))
        obs, _r, term, trunc, _i = env.step(action)
        done = term or trunc
    return np.stack(out)


def decorrelated_probe(model, dataset) -> float:
    """Max weight spread over observations from very different market regimes.

    Consecutive daily observations overlap in 49 of 50 days, so a small spread
    across them proves little. This samples the first/middle/last valid index of
    every split instead — windows that share no data at all.
    """
    idx = []
    for split in ("train", "val", "test"):
        start, end = getattr(dataset.splits, split)
        lo, hi = max(start, dataset.window - 1), end - 2
        idx += [lo, (lo + hi) // 2, hi]

    weights = []
    for t in idx:
        obs = {
            "X": dataset.tensor(t).astype(np.float32),
            # Probe the policy's price-tensor response with the portfolio state
            # held fixed, so any movement is attributable to X_t alone.
            "w_prev": np.eye(dataset.n_assets + 1, dtype=np.float32)[0],
        }
        action, _ = model.predict(obs, deterministic=True)
        weights.append(pe.softmax(action))
    w = np.stack(weights)
    return float((w.max(axis=0) - w.min(axis=0)).max())


def main() -> int:
    ap = argparse.ArgumentParser(description="Check whether policies respond to X_t.")
    ap.add_argument("--algos", nargs="+", default=ALGOS, choices=ALGOS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--checkpoint", choices=["best", "final"], default="best")
    ap.add_argument("--models-dir", default=MODELS_DIR)
    args = ap.parse_args()

    dataset = dl.build_dataset()
    print(f"Policy-degeneracy diagnostic — {args.split} window, "
          f"{args.checkpoint} checkpoints")
    print(f"A trading policy moves its weights by O(0.1); anything below "
          f"{CONSTANT_TOL:g} is constant.\n")
    print(f"{'algo':<10}{'seed':>5}{'max weight range':>20}"
          f"{'mean |w-wbar|':>16}{'decorrelated probe':>21}   verdict")

    n_constant = n_total = 0
    for algo in args.algos:
        for seed in args.seeds:
            model = load(algo, seed, args.models_dir, args.checkpoint)
            if model is None:
                print(f"{algo:<10}{seed:>5}   (checkpoint missing — skipped)")
                continue
            env = pe.make_env(args.split, episode_length=None,
                              random_start=False, dataset=dataset)
            w = weights_over_window(model, env)
            rng = float((w.max(axis=0) - w.min(axis=0)).max())
            dev = float(np.abs(w - w.mean(axis=0)).mean())
            probe = decorrelated_probe(model, dataset)

            constant = max(rng, probe) < CONSTANT_TOL
            n_constant += int(constant)
            n_total += 1
            print(f"{algo:<10}{seed:>5}{rng:>20.2e}{dev:>16.2e}{probe:>21.2e}"
                  f"   {'CONSTANT' if constant else 'responds to X_t'}")

    print(f"\n{n_constant}/{n_total} policies emit a constant allocation that "
          f"ignores the price tensor.")
    if n_constant == n_total and n_total:
        print("Every trained agent degenerated to a fixed portfolio. The reported")
        print("fAPV differences are differences between CONSTANT portfolios, not")
        print("between trading strategies. See README / METHODOLOGY.md §9.2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
