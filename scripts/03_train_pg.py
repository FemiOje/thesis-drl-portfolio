"""Phase 4: train PG, select on validation, backtest the selected checkpoint.

  python scripts/03_train_pg.py [--seeds N] [--steps N] [--run-id pg]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data as D
from src import metrics as Mx
from src import universe as U
from src.agents.pg import PGPolicy, train
from src.backtest import backtest
from src.config import PROJECT_ROOT, gradient_steps, load_config
from src.env import PortfolioEnv

SPLITS = ("train", "validate", "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=None, help="use the first N run seeds")
    ap.add_argument("--steps", type=int, default=None, help="override gradient_steps")
    ap.add_argument("--lr", type=float, default=None, help="override learning_rate")
    ap.add_argument("--run-id", default="pg")
    a = ap.parse_args()

    cfg = load_config()
    if a.steps:
        cfg.agent.pg["gradient_steps"] = a.steps
    if a.lr:
        cfg.agent.pg["learning_rate"] = a.lr
    seeds = list(cfg.run_seeds)[:a.seeds] if a.seeds else list(cfg.run_seeds)

    ds = D.build_dataset(U.HEADLINE, cfg)
    rf_all, rf_note = D.risk_free_daily(cfg, ds["dates"])
    per_year = cfg.evaluation.trading_days_per_year
    envs = {s: PortfolioEnv(ds["X"], ds["y"], cfg, ds["splits"][s]) for s in SPLITS}
    obs_space = envs["train"].observation_space

    def run(actor, split):
        return backtest(envs[split], PGPolicy(actor))

    def evaluate(actor):
        return {s: float(run(actor, s)["value"][-1]) for s in ("train", "validate")}

    out = PROJECT_ROOT / "results" / a.run_id / "PG"
    out.mkdir(parents=True, exist_ok=True)
    curves = {s: [] for s in SPLITS}
    hists, rows = [], []

    for seed in seeds:
        t0 = time.time()

        def log(step, loss, e):
            print(f"  seed {seed}  step {step:6d}  logret {loss:+.6f}  "
                  f"train {e['train']:.4f}  val {e['validate']:.4f}", flush=True)

        actor, h = train(ds, cfg, obs_space, seed=seed, evaluate=evaluate, log=log)
        hists.append(h)
        for s in SPLITS:
            rec = run(actor, s)
            curves[s].append(rec["value"])
            m = Mx.summarise(rec, rf=rf_all[ds["splits"][s]], periods=per_year)
            m.update(strategy="PG", split=s, seed=seed)
            rows.append(m)
        torch.save(actor.state_dict(), out / f"actor_seed{seed}.pt")
        print(f"seed {seed}: best step {h['best_step']}  "
              f"train {rows[-3]['final_value']:.4f}  val {rows[-2]['final_value']:.4f}  "
              f"test {rows[-1]['final_value']:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    for s in SPLITS:
        np.save(out / f"{s}.npy", np.asarray(curves[s]))
    np.savez_compressed(
        out / "history.npz",
        eval_step=hists[0]["eval_step"],
        reward=np.stack([h["reward"] for h in hists]),
        train=np.stack([h["train"] for h in hists]),
        validate=np.stack([h["validate"] for h in hists]),
        best_step=np.array([h["best_step"] for h in hists]),
    )

    per_seed = pd.DataFrame(rows)
    cols = ["split", "strategy", "final_value", "CR", "AR", "sharpe", "sortino", "MDD",
            "turnover", "win_rate", "HHI", "entropy", "max_weight", "mu_min", "n_days"]
    per_seed[["seed"] + cols].to_csv(out.parent / "metrics_per_seed.csv", index=False,
                                     float_format="%.6f")
    # F8 indexes by strategy, so one row per split: the median seed, not the best.
    med = per_seed.groupby(["split", "strategy"], as_index=False)[
        [c for c in cols if c not in ("split", "strategy")]].median()
    med[cols].to_csv(out.parent / "metrics.csv", index=False, float_format="%.6f")

    (out.parent / "meta.json").write_text(json.dumps(
        {"algo": "PG", "seeds": seeds, "universe": U.HEADLINE, "risk_free": rf_note,
         "hyperparams": dict(cfg.agent.pg), "tau": cfg.env.tau,
         "commission": cfg.env.commission, "gamma": cfg.agent.gamma,
         "gradient_steps_all": gradient_steps(cfg.agent),
         "n_params": sum(q.numel() for q in actor.parameters()),
         "best_step": [int(h["best_step"]) for h in hists]}, indent=2), encoding="utf-8")

    print(f"\nrisk-free: {rf_note}")
    print(per_seed.groupby("split")[["final_value", "sharpe", "MDD", "turnover",
                                     "max_weight", "HHI"]]
          .median().to_string(float_format=lambda x: f"{x:8.4f}"))
    print(f"-> {out.parent}")


if __name__ == "__main__":
    main()
