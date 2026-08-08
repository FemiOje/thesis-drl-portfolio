"""One-command end-to-end reproduction.

Runs the whole pipeline in order, from cached raw data to the final results
table and figures:

    1. Data      — build/verify the dataset (data/data_loader.py)
    2. Tests     — the environment unit tests (tests/test_env.py)
    3. Training  — 3 algorithms x 5 seeds @ 300k timesteps (experiments/train.py)
    4. Evaluation— held-out test window + benchmarks (experiments/evaluate.py)
    5. Diagnostics— train/val windows, convergence curves, policy degeneracy

Examples
--------
    # full reproduction (retrains everything; ~6-7 h on CPU, DDPG is ~80% of it)
    python experiments/run_all.py

    # reproduce only the reported results from the existing checkpoints (~1 min)
    python experiments/run_all.py --skip-train

    # fast end-to-end check that every stage wires together (~2 min)
    python experiments/run_all.py --steps 2000

Stage 3 is the only expensive one. ``--skip-train`` reuses whatever is already
in results/models/ and is the right choice when you only want to regenerate the
tables and figures.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PY = sys.executable

STEP_ESTIMATE_HOURS = 6.5  # measured: PPO ~10 min, A2C ~5 min, DDPG ~65 min per seed


def _banner(n: int, total: int, title: str) -> None:
    print(f"\n{'=' * 74}\n[{n}/{total}] {title}\n{'=' * 74}", flush=True)


def _run(cmd: list[str], title: str) -> float:
    """Run a stage, streaming its output. Raises on failure."""
    t0 = time.time()
    print(f"$ {' '.join(cmd[1:])}\n", flush=True)
    result = subprocess.run(cmd, cwd=_ROOT)
    elapsed = time.time() - t0
    if result.returncode != 0:
        raise SystemExit(f"\n!! stage failed ({title}), exit code "
                         f"{result.returncode}. Pipeline stopped.")
    print(f"\n-- {title}: OK ({elapsed / 60:.1f} min)", flush=True)
    return elapsed


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Reproduce the whole thesis pipeline end to end."
    )
    ap.add_argument("--skip-train", action="store_true",
                    help="reuse existing checkpoints in results/models/ "
                         "(regenerates only tables + figures)")
    ap.add_argument("--skip-tests", action="store_true",
                    help="skip the environment unit tests")
    ap.add_argument("--steps", type=int, default=300_000,
                    help="training timesteps per seed (default 300k, the "
                         "equal-budget protocol)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--commission", type=float, default=0.0025)
    args = ap.parse_args()

    total = 5 - int(args.skip_train) - int(args.skip_tests)
    stage = 0
    t_start = time.time()

    print("=" * 74)
    print("DRL PORTFOLIO MANAGEMENT - FULL REPRODUCTION")
    print("=" * 74)
    if args.skip_train:
        print("Training SKIPPED - evaluating the existing checkpoints in "
              "results/models/.")
    else:
        n_runs = 3 * len(args.seeds)
        est = STEP_ESTIMATE_HOURS * (args.steps / 300_000) * (len(args.seeds) / 5)
        print(f"Training {n_runs} runs x {args.steps:,} timesteps. "
              f"Rough estimate: {est:.1f} h on CPU (DDPG is ~80% of it).")
        print("Use --skip-train to reproduce only the reported results.")

    # --- 1. Data ----------------------------------------------------------
    stage += 1
    _banner(stage, total, "Data pipeline - download (cached) / clean / split")
    _run([_PY, os.path.join("data", "data_loader.py")], "data pipeline")

    # --- 2. Tests ---------------------------------------------------------
    if not args.skip_tests:
        stage += 1
        _banner(stage, total, "Environment unit tests")
        _run([_PY, "-m", "pytest", "tests/test_env.py", "-q"], "unit tests")

    # --- 3. Training ------------------------------------------------------
    if not args.skip_train:
        stage += 1
        _banner(stage, total, f"Training - 3 algos x {len(args.seeds)} seeds "
                              f"@ {args.steps:,} steps")
        _run([_PY, os.path.join("experiments", "train.py"),
              "--algo", "all",
              "--seeds", *[str(s) for s in args.seeds],
              "--steps", str(args.steps),
              "--commission", str(args.commission)], "training sweep")

    # --- 4. Evaluation ----------------------------------------------------
    # Test first: it is the headline result and the one that must not silently
    # depend on anything the diagnostics do.
    stage += 1
    _banner(stage, total, "Evaluation - held-out test window + benchmarks")
    seeds = [str(s) for s in args.seeds]
    _run([_PY, os.path.join("experiments", "evaluate.py"),
          "--split", "test",
          "--seeds", *seeds,
          "--commission", str(args.commission)], "evaluation (test)")

    # --- 5. In-sample diagnostics ----------------------------------------
    stage += 1
    _banner(stage, total, "Diagnostics - train/val windows, convergence, "
                          "policy degeneracy")
    for split in ("val", "train"):
        _run([_PY, os.path.join("experiments", "evaluate.py"),
              "--split", split,
              "--seeds", *seeds,
              "--commission", str(args.commission)],
             f"evaluation ({split})")
    _run([_PY, os.path.join("experiments", "plot_training.py"),
          "--seeds", *seeds], "training curves + convergence check")
    _run([_PY, os.path.join("experiments", "policy_diagnostic.py"),
          "--seeds", *seeds], "policy-degeneracy diagnostic")

    mins = (time.time() - t_start) / 60
    print(f"\n{'=' * 74}")
    print(f"REPRODUCTION COMPLETE in {mins:.1f} min")
    print(f"  Results       : results/evaluation/test/RESULTS.md  (+ val/, train/)")
    print(f"  Figures       : results/evaluation/*/*.png")
    print(f"  Summary       : README.md")
    print(f"  Methodology   : docs/METHODOLOGY.md")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
