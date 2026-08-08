"""Training-progress figure and convergence check.

Reads the per-seed validation curves that SB3's ``EvalCallback`` wrote during
training (``results/models/{algo}/seed{n}/evaluations.npz``) and plots validation
fAPV against training timesteps, mean ± std across seeds.

This is the figure that answers "had each algorithm converged by 300k steps?".
It also prints a numeric convergence check: the slope of each algorithm's
validation curve over its final third, expressed as fAPV gained per 100k steps.
A curve that is still climbing at the step budget is undertrained, and the
comparison at that budget understates it.

The stored eval metric is mean episode reward = Σ log-returns over one full
validation pass = log(fAPV_val), so fAPV = exp(reward).

Run:  python experiments/plot_training.py
"""

from __future__ import annotations

import argparse
import os

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(_ROOT, "results", "models")
OUT_DIR = os.path.join(_ROOT, "results", "evaluation")

ALGOS = ["ppo", "a2c", "ddpg"]
ALGO_LABEL = {"ppo": "PPO", "a2c": "A2C (PG)", "ddpg": "DDPG"}
ALGO_COLOR = {"ppo": "#0072B2", "a2c": "#009E73", "ddpg": "#D55E00"}


def load_curves(algo: str, seeds: list[int], models_dir: str):
    """Return (timesteps, fapv[n_seeds, n_evals]) or None if nothing on disk."""
    steps, curves = None, []
    for seed in seeds:
        path = os.path.join(models_dir, algo, f"seed{seed}", "evaluations.npz")
        if not os.path.exists(path):
            continue
        d = np.load(path)
        # results is (n_evals, n_episodes); one episode per eval here.
        curves.append(np.exp(d["results"].mean(axis=1)))
        steps = d["timesteps"]
    if not curves:
        return None
    n = min(len(c) for c in curves)  # guard against a short/interrupted run
    return steps[:n], np.stack([c[:n] for c in curves])


def tail_slope(steps: np.ndarray, mean_curve: np.ndarray, frac: float = 1 / 3):
    """fAPV gained per 100k steps over the final ``frac`` of training.

    A least-squares fit rather than an endpoint difference, so one noisy final
    evaluation cannot flip the sign.
    """
    k = max(3, int(len(steps) * frac))
    x, y = steps[-k:].astype(float), mean_curve[-k:]
    slope = np.polyfit(x, y, 1)[0]
    return float(slope * 100_000), k


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Plot validation curves from training and check convergence."
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--models-dir", default=MODELS_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    print("=== Convergence check: validation fAPV slope over the final third ===")
    print(f"{'algo':<10}{'final fAPV':>14}{'slope /100k':>14}   verdict")

    verdicts = {}
    for algo in ALGOS:
        data = load_curves(algo, args.seeds, args.models_dir)
        if data is None:
            print(f"{algo:<10}  (no evaluations.npz found — skipped)")
            continue
        steps, curves = data
        mean, std = curves.mean(axis=0), curves.std(axis=0, ddof=1)
        color = ALGO_COLOR[algo]

        ax.plot(steps, mean, color=color, lw=2.0, label=ALGO_LABEL[algo], zorder=3)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15,
                        lw=0, zorder=2)

        slope, k = tail_slope(steps, mean)
        # Threshold is deliberately loose: "still gaining >1% fAPV per 100k
        # steps at the budget" is the claim, not a formal convergence test.
        still_rising = slope > 0.01
        verdicts[algo] = (mean[-1], slope, still_rising)
        print(f"{algo:<10}{mean[-1]:>14.4f}{slope:>+14.4f}   "
              f"{'STILL RISING (undertrained)' if still_rising else 'plateaued'}"
              f"  [fit over last {k} evals]")

    ax.set_xlabel("training timesteps")
    ax.set_ylabel("validation fAPV  (mean ± std over seeds)")
    ax.set_title("Validation performance during training — 3 algorithms × "
                 f"{len(args.seeds)} seeds")
    ax.axhline(1.0, color="#c8c8c8", lw=1, zorder=1)
    ax.grid(True, axis="y", color="#ececec", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = os.path.join(args.out_dir, "training_curves.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)

    rising = [ALGO_LABEL[a] for a, (_, _, r) in verdicts.items() if r]
    print()
    if rising:
        print(f"Not yet plateaued at the step budget: {', '.join(rising)}.")
    else:
        print("All algorithms had plateaued by the step budget.")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
