"""Training CLI for the DRL portfolio agents.

Trains Stable-Baselines3 agents on the portfolio environment (env/portfolio_env.py)
with the **MultiInputPolicy** (required: the observation is a Dict of the price
tensor X_t + previous weights; MlpPolicy errors on Dict spaces).

Algorithms (per the plan; Liang et al. 2018 comparison):
  * ppo  — PPO, the stochastic policy-gradient representative.
  * ddpg — DDPG, the deterministic actor-critic (Gaussian action noise added).
  * a2c  — A2C, used as the practical vanilla policy-gradient ("PG") baseline
           (this PG->A2C mapping is stated in the thesis / docs).

Protocol:
  * Train on the TRAIN split (random ~250-step sub-windows for variety).
  * 300k timesteps/seed by default; 5 seeds per algorithm (report mean +/- std).
  * Checkpoint the **best-on-validation** model per seed (SB3 EvalCallback,
    deterministic full pass over the VAL split). The test split is never touched.
  * TensorBoard logs under results/tensorboard/.

Examples
--------
    # one run
    python experiments/train.py --algo ppo --seed 0 --steps 300000
    # full sweep (3 algos x 5 seeds) in one command
    python experiments/train.py --algo all --seeds 0 1 2 3 4 --steps 300000
    # quick smoke test that the plumbing works
    python experiments/train.py --algo all --seed 0 --steps 2000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from stable_baselines3 import A2C, DDPG, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import set_random_seed

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("env", "data"):
    p = os.path.join(_ROOT, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import data_loader as dl  # noqa: E402
import portfolio_env as pe  # noqa: E402

ALGOS = {"ppo": PPO, "a2c": A2C, "ddpg": DDPG}

RESULTS_DIR = os.path.join(_ROOT, "results")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
TB_DIR = os.path.join(RESULTS_DIR, "tensorboard")


# =============================================================================
# Per-algorithm default hyperparameters (CLI flags override; None => default).
# Tune only 2-3 (learning rate, rollout/batch size) on VALIDATION.
# =============================================================================
def hparams(algo: str, args: argparse.Namespace) -> dict:
    lr = args.lr
    n_steps = args.n_steps
    batch = args.batch_size
    if algo == "ppo":
        return dict(
            learning_rate=3e-4 if lr is None else lr,
            n_steps=2048 if n_steps is None else n_steps,
            batch_size=64 if batch is None else batch,
            n_epochs=10,
            gamma=args.gamma,
            gae_lambda=0.95,
            ent_coef=0.0,
        )
    if algo == "a2c":
        return dict(
            learning_rate=7e-4 if lr is None else lr,
            n_steps=16 if n_steps is None else n_steps,
            gamma=args.gamma,
            gae_lambda=1.0,
            ent_coef=0.0,
        )
    if algo == "ddpg":
        return dict(
            learning_rate=1e-3 if lr is None else lr,
            buffer_size=args.buffer_size,
            batch_size=256 if batch is None else batch,
            learning_starts=1000,
            gamma=args.gamma,
            train_freq=(1, "step"),
        )
    raise ValueError(algo)


def build_model(algo: str, train_env, seed: int, tb_dir: str, args: argparse.Namespace):
    hp = hparams(algo, args)
    common = dict(
        policy="MultiInputPolicy",
        env=train_env,
        seed=seed,
        verbose=0,
        device=args.device,
        tensorboard_log=tb_dir,
    )
    if algo == "ddpg":
        n = train_env.action_space.shape[0]
        common["action_noise"] = NormalActionNoise(
            mean=np.zeros(n), sigma=args.ddpg_sigma * np.ones(n)
        )
    return ALGOS[algo](**common, **hp)


# =============================================================================
# Single (algo, seed) training run
# =============================================================================
def train_one(algo: str, seed: int, args: argparse.Namespace, dataset) -> dict:
    set_random_seed(seed)
    t0 = time.time()

    train_env = Monitor(
        pe.make_env("train", commission=args.commission,
                    episode_length=args.episode_length, random_start=True,
                    dataset=dataset)
    )
    # Deterministic full pass over VAL for "best-on-validation" checkpointing.
    val_env = Monitor(
        pe.make_env("val", commission=args.commission,
                    episode_length=None, random_start=False, dataset=dataset)
    )

    model_dir = os.path.join(MODELS_DIR, algo, f"seed{seed}")
    os.makedirs(model_dir, exist_ok=True)

    eval_freq = args.eval_freq or max(args.steps // 20, 2000)
    eval_cb = EvalCallback(
        val_env,
        best_model_save_path=model_dir,   # writes best_model.zip on val improvement
        log_path=model_dir,               # writes evaluations.npz (val reward curve)
        eval_freq=eval_freq,
        n_eval_episodes=1,                # val is one fixed chronological window
        deterministic=True,
        render=False,
        verbose=0,
    )

    model = build_model(algo, train_env, seed, TB_DIR, args)
    model.learn(
        total_timesteps=args.steps,
        callback=eval_cb,
        tb_log_name=f"{algo}_seed{seed}",
        progress_bar=False,
    )
    model.save(os.path.join(model_dir, "final_model"))

    elapsed = time.time() - t0
    # EvalCallback tracks best mean val reward = sum of log-returns = log(fAPV_val).
    best_logret = float(eval_cb.best_mean_reward)
    fapv_val = float(np.exp(best_logret)) if np.isfinite(best_logret) else float("nan")

    summary = {
        "algo": algo,
        "seed": seed,
        "steps": args.steps,
        "commission": args.commission,
        "episode_length": args.episode_length,
        "hparams": hparams(algo, args),
        "best_val_logreturn": best_logret,
        "best_val_fapv": fapv_val,
        "elapsed_sec": round(elapsed, 1),
        "best_model": os.path.join(model_dir, "best_model.zip"),
        "final_model": os.path.join(model_dir, "final_model.zip"),
    }
    with open(os.path.join(model_dir, "config.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print(f"[{algo} seed{seed}] {args.steps} steps in {elapsed:.0f}s | "
          f"best val fAPV={fapv_val:.4f} (logret={best_logret:.4f}) -> {model_dir}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Train DRL portfolio agents (SB3).")
    ap.add_argument("--algo", choices=list(ALGOS) + ["all"], default="ppo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="override --seed with a list, e.g. --seeds 0 1 2 3 4")
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--episode-length", type=int, default=250,
                    help="training sub-window length (steps); random start on train")
    ap.add_argument("--commission", type=float, default=pe.DEFAULT_COMMISSION)
    # Tunable hyperparameters (defaults per-algo when omitted).
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--n-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--buffer-size", type=int, default=50_000,
                    help="DDPG replay buffer (kept modest for 8GB RAM)")
    ap.add_argument("--ddpg-sigma", type=float, default=0.3,
                    help="DDPG Gaussian action-noise std (action space is +/-10)")
    ap.add_argument("--eval-freq", type=int, default=None,
                    help="env steps between val evaluations (default steps//20)")
    ap.add_argument("--device", default="cpu",
                    help="torch device: cpu (default) | cuda | auto (e.g. Colab GPU)")
    args = ap.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(TB_DIR, exist_ok=True)

    algos = list(ALGOS) if args.algo == "all" else [args.algo]
    seeds = args.seeds if args.seeds is not None else [args.seed]

    dataset = dl.build_dataset()  # cached; never re-downloads
    print(f"Training {algos} x seeds {seeds} @ {args.steps} steps "
          f"(commission {args.commission:.4%}, val-checkpointed)")

    summaries = []
    for algo in algos:
        for seed in seeds:
            summaries.append(train_one(algo, seed, args, dataset))

    # Aggregate mean +/- std of best-on-val fAPV per algorithm.
    print("\n=== best-on-validation fAPV (mean +/- std over seeds) ===")
    for algo in algos:
        vals = [s["best_val_fapv"] for s in summaries if s["algo"] == algo]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            print(f"  {algo:5s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}  (n={len(vals)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
