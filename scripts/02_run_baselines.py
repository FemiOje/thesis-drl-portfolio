"""Phase 3: all four baselines through backtest(), full metrics, all three splits."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import baselines as B
from src import data as D
from src import metrics as Mx
from src import universe as U
from src.backtest import backtest
from src.config import PROJECT_ROOT, load_config
from src.env import PortfolioEnv

OUT = PROJECT_ROOT / "results" / "phase3"


def main():
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    ds = D.build_dataset(U.HEADLINE, cfg)
    rf_all, rf_note = D.risk_free_daily(cfg, ds["dates"])
    per_year = cfg.evaluation.trading_days_per_year

    rows, curves = [], {}
    for split in ("train", "validate", "test"):
        idx = ds["splits"][split]
        rf = rf_all[idx]
        for name, policy in B.build(cfg, ds, split).items():
            env = PortfolioEnv(ds["X"], ds["y"], cfg, idx)
            rec = backtest(env, policy)
            s = Mx.summarise(rec, rf=rf, periods=per_year)
            s.update(strategy=name, split=split)
            rows.append(s)
            curves[f"{split}/{name}"] = rec["value"]
            pd.DataFrame(rec["weights"], index=pd.DatetimeIndex(ds["dates"])[idx],
                         columns=["CASH"] + U.HEADLINE).to_csv(
                OUT / f"weights_{split}_{name}.csv", float_format="%.8f")

    df = pd.DataFrame(rows)[["split", "strategy", "final_value", "CR", "AR", "sharpe",
                             "sortino", "MDD", "turnover", "win_rate", "HHI", "entropy",
                             "max_weight", "mu_min", "n_days"]]
    df.to_csv(OUT / "baselines.csv", index=False, float_format="%.6f")
    np.savez_compressed(OUT / "curves.npz",
                        dates=np.array([str(d.date()) for d in ds["dates"]]),
                        **{k.replace("/", "_"): v for k, v in curves.items()},
                        **{f"idx_{s}": ds["splits"][s] for s in ds["splits"]})
    (OUT / "meta.json").write_text(json.dumps(
        {"universe": U.HEADLINE, "risk_free": rf_note, "commission": cfg.env.commission,
         "tau": cfg.env.tau, "splits": {k: int(len(v)) for k, v in ds["splits"].items()}},
        indent=2), encoding="utf-8")

    pd.set_option("display.width", 200, "display.max_columns", 40)
    for split in ("train", "validate", "test"):
        print(f"\n=== {split} ===")
        print(df[df.split == split].drop(columns=["split", "sortino", "entropy", "mu_min"])
              .to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
    print(f"\nrisk-free: {rf_note}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
