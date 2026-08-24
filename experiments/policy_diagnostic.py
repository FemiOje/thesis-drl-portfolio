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

# Three outcomes, not two. The original version asked a single question ("do the
# weights move?") because at the time every failing policy failed both measures
# at once. The convolutional models produced a case that did not exist before:
# weights that move while the price tensor contributes nothing, because w_prev
# feeds back into the observation and the portfolio drifts on its own. Judging
# that on weight movement alone reports it as healthy. The decorrelated probe is
# therefore authoritative for market response, and window movement only
# distinguishes "frozen" from "drifting".
# Two thresholds, because "not numerically zero" and "economically meaningful"
# are different questions. CONSTANT_TOL is the noise floor. MEANINGFUL_RESPONSE
# is the bar for calling a policy a trader: a probe below 1% means the
# allocation shifts by under one percentage point across market regimes as
# different as 2021 and 2026, which is not a trading decision. For scale, the
# genuinely responsive policies measured here score 0.5-0.76.
MEANINGFUL_RESPONSE = 1e-2

FROZEN = "FROZEN"                 # weights never move at all
MARKET_BLIND = "MARKET-BLIND"     # weights move, but not because of X_t
NEGLIGIBLE = "negligible"         # responds to X_t, but immaterially
RESPONDS = "responds to X_t"      # genuine trading policy


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
    ap.add_argument("--action-bound", type=float, default=pe.ACTION_BOUND,
                    help="must match the bound the checkpoints were trained under")
    ap.add_argument("--models-dir", default=MODELS_DIR)
    args = ap.parse_args()

    dataset = dl.build_dataset()
    print(f"Policy-degeneracy diagnostic — {args.split} window, "
          f"{args.checkpoint} checkpoints")
    print(f"A trading policy moves its weights by O(0.1); anything below "
          f"{CONSTANT_TOL:g} is constant.")
    print("The decorrelated probe decides market response: weights can move "
          "purely from\nw_prev feedback while the price tensor contributes "
          "nothing.\n")
    print(f"{'algo':<10}{'seed':>5}{'max weight range':>20}"
          f"{'mean |w-wbar|':>16}{'decorrelated probe':>21}   verdict")

    counts = {FROZEN: 0, MARKET_BLIND: 0, NEGLIGIBLE: 0, RESPONDS: 0}
    n_total = 0
    for algo in args.algos:
        for seed in args.seeds:
            model = load(algo, seed, args.models_dir, args.checkpoint)
            if model is None:
                print(f"{algo:<10}{seed:>5}   (checkpoint missing — skipped)")
                continue
            env = pe.make_env(args.split, episode_length=None,
                              random_start=False, dataset=dataset,
                              action_bound=args.action_bound)
            w = weights_over_window(model, env)
            rng = float((w.max(axis=0) - w.min(axis=0)).max())
            dev = float(np.abs(w - w.mean(axis=0)).mean())
            probe = decorrelated_probe(model, dataset)

            # Probe first: a policy that ignores X_t is degenerate however much
            # its weights happen to drift. Only then ask whether it moved.
            if probe < CONSTANT_TOL:
                verdict = FROZEN if rng < CONSTANT_TOL else MARKET_BLIND
            elif probe < MEANINGFUL_RESPONSE:
                verdict = NEGLIGIBLE
            else:
                verdict = RESPONDS
            counts[verdict] += 1
            n_total += 1
            print(f"{algo:<10}{seed:>5}{rng:>20.2e}{dev:>16.2e}{probe:>21.2e}"
                  f"   {verdict}")

    if not n_total:
        print("\nNo checkpoints found.")
        return 1

    degenerate = counts[FROZEN] + counts[MARKET_BLIND] + counts[NEGLIGIBLE]
    print(f"\n{counts[RESPONDS]}/{n_total} policies respond to the price tensor "
          f"meaningfully (probe >= {MEANINGFUL_RESPONSE:g}).")
    print(f"  {counts[FROZEN]:>2} frozen        (weights never move)")
    print(f"  {counts[MARKET_BLIND]:>2} market-blind  (weights move, but not from X_t)")
    print(f"  {counts[NEGLIGIBLE]:>2} negligible    (responds, but under "
          f"{MEANINGFUL_RESPONSE:g} across regimes)")
    if degenerate == n_total:
        print("\nEvery trained agent is degenerate. The reported fAPV differences")
        print("are differences between portfolios that ignore the market, not")
        print("between trading strategies. See README / METHODOLOGY.md §9.2.")
    elif degenerate:
        print(f"\n{degenerate}/{n_total} agents are degenerate — exclude them from any")
        print("claim about trading behaviour, and report the split explicitly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
