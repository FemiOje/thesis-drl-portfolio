"""Significance table for the agent runs: paired t-tests, Bonferroni, bootstrap CIs.

  python scripts/07_stats.py [--run-id phase5] [--split test] [--block 1]

Writes results/<run_id>/stats_<split>.csv -- the artefact Chapter 4 cites. The test
split is scored once; running this does not re-touch it, it reads committed curves.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import results, stats as S
from src.config import PROJECT_ROOT, load_config

BASELINES = ("UCRP", "UBAH", "Markowitz", "BestStock")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="phase5")
    ap.add_argument("--split", default="test")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--block", type=int, default=1, help="moving-block length; 1 = iid")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = load_config()
    n_boot = cfg.evaluation.bootstrap_samples

    base = results.load_baselines(PROJECT_ROOT / "results" / "phase3")
    agents = results.load_agents(PROJECT_ROOT / "results" / a.run_id)
    ac = agents["curves"][a.split]
    bc = {k: v[0] for k, v in base["curves"][a.split].items() if k in BASELINES}
    if not ac:
        raise SystemExit(f"no agent curves for split={a.split} in run {a.run_id}")

    m = len(ac) * len(bc)
    df = S.compare(ac, bc, alpha=a.alpha, n_boot=n_boot, seed=a.seed, block=a.block,
                   adjusted_alpha=a.alpha / m)
    df = S.bonferroni(df, alpha=a.alpha)

    out = PROJECT_ROOT / "results" / a.run_id / f"stats_{a.split}.csv"
    df.to_csv(out, index=False, float_format="%.8g")

    sl = df[df.level == "seeds"]
    print(f"split={a.split}  n_boot={n_boot}  block={a.block}  "
          f"m={sl['m'].iloc[0]}  alpha_adj={sl['alpha_adj'].iloc[0]:.5f}\n")
    show = sl[["algo", "baseline", "n_better", "mean_diff", "p", "p_adj", "reject",
               "ret_diff", "ret_lo", "ret_hi"]].copy()
    show["ret_diff"] = show["ret_diff"] * 1e4
    show["ret_lo"] = show["ret_lo"] * 1e4
    show["ret_hi"] = show["ret_hi"] * 1e4
    show = show.rename(columns={"ret_diff": "bp/day", "ret_lo": "bp_lo", "ret_hi": "bp_hi",
                                "mean_diff": "d_wealth"})
    print("seed level. d_wealth/p/reject test FINAL WEALTH against the baseline's "
          "realised\nvalue over this fixed window (seed variation only). bp/day and its "
          "interval\nalso carry day variation, so they are wider by construction -- a "
          "rejection beside\nan interval spanning zero is the expected pattern, not a "
          "contradiction.\n")
    print(show.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))

    day = df[df.level == "median"]
    print("\nday-level, median seed (paired t on daily returns):")
    print(day[["algo", "baseline", "seed_used", "mean_diff", "p", "p_adj", "reject"]]
          .to_string(index=False, float_format=lambda x: f"{x:10.6f}"))

    per = df[df.level.str.startswith("seed") & (df.level != "seeds")]
    surv = (per.assign(win=lambda d: (d.mean_diff > 0) & (d.p < d.alpha_adj))
              .groupby(["algo", "baseline"]).win.sum())
    print("\nseeds beating the baseline at the Bonferroni-adjusted alpha (day-level):")
    print(surv.to_string())
    print(f"\n-> {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
