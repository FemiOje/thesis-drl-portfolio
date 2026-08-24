# Validation-window results (in-sample diagnostic)

Val window: **2025-01-23 → 2025-10-20** (187 decision steps). Commission **0.25%** both sides. Agents: best-on-validation checkpoints at 500,000 timesteps, 5 seeds, acting deterministically.

**In-sample diagnostic, not a headline result.** Checkpoints were selected on this window, so these figures are optimistically biased. Reported to show the train/val/test progression. Metrics follow Jiang §6.2; turnover is Σ|Δw| per step (0-2).

## Summary (agents: mean ± std over seeds)

| Strategy | fAPV | Annualized return | Sharpe | Max drawdown | Turnover |
|---|---|---|---|---|---|
| PPO | 1.3288 ± 0.1499 | 47.03 ± 22.12% | 1.255 ± 0.324 | 24.16 ± 4.81% | 0.0769 ± 0.0309 |
| A2C (PG) | 1.2580 ± 0.0224 | 36.26 ± 3.28% | 0.961 ± 0.195 | 30.38 ± 5.16% | 0.0177 ± 0.0039 |
| DDPG | 1.1868 ± 0.0739 | 26.01 ± 10.56% | 0.886 ± 0.102 | 26.82 ± 13.02% | 0.0163 ± 0.0079 |
| Buy & Hold | 1.1266 | 17.43% | 0.951 | 17.37% | 0.0107 |
| UCRP | 1.1376 | 18.98% | 0.976 | 17.67% | 0.0212 |
| Best stock (hindsight) | 1.3623 | 51.68% | 2.165 | 12.73% | 0.0107 |
| All-cash | 1.0000 | 0.00% | n/a | 0.00% | 0.0000 |

## Per-seed detail

| Algorithm | Seed | fAPV | Sharpe | Max drawdown | Turnover |
|---|---|---|---|---|---|
| PPO | 0 | 1.5066 | 1.649 | 23.22% | 0.1017 |
| PPO | 1 | 1.4076 | 1.516 | 26.36% | 0.0963 |
| PPO | 2 | 1.3473 | 1.177 | 28.90% | 0.0705 |
| PPO | 3 | 1.2742 | 1.074 | 25.96% | 0.0259 |
| PPO | 4 | 1.1081 | 0.857 | 16.33% | 0.0902 |
| A2C (PG) | 0 | 1.2638 | 0.921 | 31.52% | 0.0160 |
| A2C (PG) | 1 | 1.2504 | 0.905 | 30.64% | 0.0168 |
| A2C (PG) | 2 | 1.2940 | 1.302 | 21.61% | 0.0235 |
| A2C (PG) | 3 | 1.2368 | 0.824 | 34.77% | 0.0130 |
| A2C (PG) | 4 | 1.2451 | 0.853 | 33.35% | 0.0193 |
| DDPG | 0 | 1.1345 | 0.958 | 17.61% | 0.0219 |
| DDPG | 1 | 1.2390 | 0.814 | 36.02% | 0.0107 |

## Figures

* `01_wealth_curves.png` — wealth over the window (log scale), agent means with per-seed spread, against the four benchmarks.
* `02_allocation_heatmaps.png` — portfolio weights over time for the median-fAPV seed of each algorithm.
* `03_seed_distributions.png` — per-seed fAPV and Sharpe spread.

Regenerate with `python experiments/evaluate.py`.
