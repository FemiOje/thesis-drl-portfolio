"""Phase 5: train PPO or DDPG, select on validation, backtest the selected checkpoint.

  python scripts/05_train_sb3.py --algo ppo [--seeds N] [--steps N] [--run-id ppo]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data as D
from src import metrics as Mx
from src import universe as U
from src.agents.pg import PGActor
from src.agents.sb3 import SB3Policy, train
from src.backtest import backtest
from src.config import PROJECT_ROOT, gradient_steps, load_config
from src.env import PortfolioEnv

SPLITS = ("train", "validate", "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=["ppo", "ddpg"])
    ap.add_argument("--seeds", type=int, default=None, help="use the first N run seeds")
    ap.add_argument("--steps", type=int, default=None, help="override total_timesteps")
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--fresh", action="store_true", help="ignore completed seeds")
    a = ap.parse_args()

    ALGO = a.algo.upper()
    cfg = load_config()
    hp = getattr(cfg.agent, a.algo)
    seeds = list(cfg.run_seeds)[:a.seeds] if a.seeds else list(cfg.run_seeds)
    run_id = a.run_id or a.algo

    ds = D.build_dataset(U.HEADLINE, cfg)
    rf_all, rf_note = D.risk_free_daily(cfg, ds["dates"])
    per_year = cfg.evaluation.trading_days_per_year
    mk = lambda s: PortfolioEnv(ds["X"], ds["y"], cfg, ds["splits"][s])
    # Evaluation envs are SEPARATE instances from the one learn() steps: backtest()
    # calls reset(), which would rewind an in-flight rollout. See agents/sb3.py.
    envs = {s: mk(s) for s in SPLITS}

    # PG's extractor is the reference; every arm must match it exactly.
    n_params = sum(p.numel() for p in
                   PGActor(envs["train"].observation_space, cfg).net.parameters())
    expect = None if (a.steps or a.eval_every) else gradient_steps(cfg.agent)[ALGO]

    def evaluate(model):
        p = SB3Policy(model)
        return {s: float(backtest(envs[s], p)["value"][-1]) for s in ("train", "validate")}

    out = PROJECT_ROOT / "results" / run_id / ALGO
    out.mkdir(parents=True, exist_ok=True)

    # Per-seed artefacts are written the moment a seed finishes, and a rerun skips any
    # seed already on disk. A 10-seed run is ~4 h; without this an interruption at seed
    # 9 discards every completed seed, which is exactly what happened the first time.
    def seed_file(seed):
        return out / f"seed{seed}.npz"

    def save_seed(seed, h, cs, rws):
        np.savez_compressed(seed_file(seed), rows=json.dumps(rws),
                            optimizer_steps=json.dumps(h["optimizer_steps"]),
                            best_step=h["best_step"], best_validate=h["best_validate"],
                            updates=h["updates"], eval_step=h["eval_step"],
                            reward=h["reward"], train=h["train"], validate=h["validate"],
                            **{f"curve_{k}": v for k, v in cs.items()})

    def load_seed(seed):
        z = np.load(seed_file(seed), allow_pickle=False)
        h = {"best_step": int(z["best_step"]), "best_validate": float(z["best_validate"]),
             "updates": int(z["updates"]), "eval_step": z["eval_step"],
             "reward": z["reward"], "train": z["train"], "validate": z["validate"],
             "optimizer_steps": json.loads(str(z["optimizer_steps"]))}
        return h, {k: z[f"curve_{k}"] for k in SPLITS}, json.loads(str(z["rows"]))

    curves = {s: [] for s in SPLITS}
    hists, rows = [], []

    for seed in seeds:
        t0 = time.time()
        if seed_file(seed).exists() and not a.fresh:
            h, cs, rws = load_seed(seed)
            hists.append(h)
            for s in SPLITS:
                curves[s].append(cs[s])
            rows.extend(rws)
            print(f"seed {seed}: reusing {seed_file(seed).name}  "
                  f"best step {h['best_step']}  test {rws[-1]['final_value']:.4f}",
                  flush=True)
            continue

        def log(step, reward, e):
            print(f"  seed {seed}  step {step:6d}  reward {reward:+.6f}  "
                  f"train {e['train']:.4f}  val {e['validate']:.4f}", flush=True)

        model, h = train(ALGO, mk("train"), cfg, seed, evaluate, log=log,
                         n_params=n_params, total=a.steps,
                         eval_every=a.eval_every, expect=expect)
        hists.append(h)
        pol = SB3Policy(model)
        cs, rws = {}, []
        for s in SPLITS:
            rec = backtest(envs[s], pol)
            cs[s] = rec["value"]
            curves[s].append(rec["value"])
            m = Mx.summarise(rec, rf=rf_all[ds["splits"][s]], periods=per_year)
            m.update(strategy=ALGO, split=s, seed=seed)
            rows.append(m)
            rws.append(m)
        model.save(out / f"model_seed{seed}")
        save_seed(seed, h, cs, rws)
        print(f"seed {seed}: best step {h['best_step']}  updates {h['updates']}  "
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
    med = per_seed.groupby(["split", "strategy"], as_index=False)[
        [c for c in cols if c not in ("split", "strategy")]].median()
    med[cols].to_csv(out.parent / "metrics.csv", index=False, float_format="%.6f")

    (out.parent / "meta.json").write_text(json.dumps(
        {"algo": ALGO, "seeds": seeds, "universe": U.HEADLINE, "risk_free": rf_note,
         "hyperparams": dict(hp), "tau": cfg.env.tau, "commission": cfg.env.commission,
         "gamma": cfg.agent.gamma, "gradient_steps_all": gradient_steps(cfg.agent),
         "realised_updates": [int(h["updates"]) for h in hists],
         "optimizer_steps": hists[0]["optimizer_steps"],
         "resumed": [int(s) for s in seeds if seed_file(s).exists()],
         "extractor_params": int(n_params),
         "share_features_extractor": bool(hp["share_features_extractor"]),
         "best_step": [int(h["best_step"]) for h in hists]}, indent=2), encoding="utf-8")

    print(f"\nrisk-free: {rf_note}")
    print(per_seed.groupby("split")[["final_value", "sharpe", "MDD", "turnover",
                                     "max_weight", "HHI"]]
          .median().to_string(float_format=lambda x: f"{x:8.4f}"))
    print(f"-> {out.parent}")


if __name__ == "__main__":
    main()
