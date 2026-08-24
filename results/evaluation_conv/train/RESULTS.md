# Training-window results (in-sample diagnostic)

Train window: **2021-10-04 → 2025-01-21** (828 decision steps). Commission **0.25%** both sides. Agents: best-on-validation checkpoints at 500,000 timesteps, 5 seeds, acting deterministically.

**In-sample diagnostic, not a headline result.** The agents were trained on this window, so these figures show fit, not generalization. Metrics follow Jiang §6.2; turnover is Σ|Δw| per step (0-2).

## Summary (agents: mean ± std over seeds)

| Strategy | fAPV | Annualized return | Sharpe | Max drawdown | Turnover |
|---|---|---|---|---|---|
| PPO | 31.8469 ± 28.9066 | 152.46 ± 107.27% | 2.551 ± 1.266 | 24.27 ± 14.68% | 0.0940 ± 0.0511 |
| A2C (PG) | 5.4485 ± 1.1135 | 66.84 ± 11.39% | 1.340 ± 0.010 | 57.71 ± 8.74% | 0.0093 ± 0.0041 |
| DDPG | 4.3957 ± 3.3744 | 51.37 ± 39.17% | 1.237 ± 0.127 | 49.33 ± 24.03% | 0.0084 ± 0.0084 |
| Buy & Hold | 2.1672 | 26.54% | 1.258 | 22.51% | 0.0024 |
| UCRP | 1.9315 | 22.18% | 1.163 | 23.78% | 0.0134 |
| Best stock (hindsight) | 6.7847 | 79.09% | 1.327 | 66.34% | 0.0024 |
| All-cash | 1.0000 | 0.00% | n/a | 0.00% | 0.0000 |

## Per-seed detail

| Algorithm | Seed | fAPV | Sharpe | Max drawdown | Turnover |
|---|---|---|---|---|---|
| PPO | 0 | 67.7189 | 3.573 | 19.20% | 0.1542 |
| PPO | 1 | 52.1321 | 3.889 | 12.02% | 0.1173 |
| PPO | 2 | 32.9369 | 2.820 | 18.49% | 0.0927 |
| PPO | 3 | 4.6869 | 1.381 | 49.72% | 0.0149 |
| PPO | 4 | 1.7599 | 1.090 | 21.90% | 0.0910 |
| A2C (PG) | 0 | 5.6470 | 1.328 | 60.60% | 0.0083 |
| A2C (PG) | 1 | 5.4529 | 1.340 | 58.27% | 0.0095 |
| A2C (PG) | 2 | 3.5762 | 1.357 | 42.61% | 0.0161 |
| A2C (PG) | 3 | 6.3849 | 1.338 | 64.03% | 0.0053 |
| A2C (PG) | 4 | 6.1818 | 1.340 | 63.07% | 0.0074 |
| DDPG | 0 | 2.0096 | 1.148 | 32.34% | 0.0143 |
| DDPG | 1 | 6.7817 | 1.327 | 66.32% | 0.0024 |

## Figures

* `01_wealth_curves.png` — wealth over the window (log scale), agent means with per-seed spread, against the four benchmarks.
* `02_allocation_heatmaps.png` — portfolio weights over time for the median-fAPV seed of each algorithm.
* `03_seed_distributions.png` — per-seed fAPV and Sharpe spread.

Regenerate with `python experiments/evaluate.py`.
