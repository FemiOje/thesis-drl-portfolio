# Validation-window results (in-sample diagnostic)

Val window: **2025-01-23 → 2025-10-20** (187 decision steps). Commission **0.25%** both sides. Agents: best-on-validation checkpoints at 300,000 timesteps, 5 seeds, acting deterministically.

**In-sample diagnostic, not a headline result.** Checkpoints were selected on this window, so these figures are optimistically biased. Reported to show the train/val/test progression. Metrics follow Jiang §6.2; turnover is Σ|Δw| per step (0-2).

## Summary (agents: mean ± std over seeds)

| Strategy | fAPV | Annualized return | Sharpe | Max drawdown | Turnover |
|---|---|---|---|---|---|
| PPO | 1.1852 ± 0.0091 | 25.73 ± 1.30% | 0.874 ± 0.019 | 24.25 ± 1.35% | 0.0209 ± 0.0005 |
| A2C (PG) | 1.2280 ± 0.0258 | 31.91 ± 3.72% | 0.849 ± 0.062 | 32.83 ± 5.75% | 0.0127 ± 0.0019 |
| DDPG | 1.0998 ± 0.0365 | 13.69 ± 5.06% | 0.833 ± 0.439 | 17.20 ± 5.57% | 0.0193 ± 0.0034 |
| Buy & Hold | 1.1266 | 17.43% | 0.951 | 17.37% | 0.0107 |
| UCRP | 1.1376 | 18.98% | 0.976 | 17.67% | 0.0212 |
| Best stock (hindsight) | 1.3623 | 51.68% | 2.165 | 12.73% | 0.0107 |
| All-cash | 1.0000 | 0.00% | n/a | 0.00% | 0.0000 |

## Per-seed detail

| Algorithm | Seed | fAPV | Sharpe | Max drawdown | Turnover |
|---|---|---|---|---|---|
| PPO | 0 | 1.1710 | 0.884 | 22.06% | 0.0214 |
| PPO | 1 | 1.1880 | 0.897 | 23.83% | 0.0206 |
| PPO | 2 | 1.1960 | 0.882 | 25.33% | 0.0214 |
| PPO | 3 | 1.1841 | 0.852 | 25.01% | 0.0206 |
| PPO | 4 | 1.1869 | 0.857 | 25.02% | 0.0204 |
| A2C (PG) | 0 | 1.2404 | 0.826 | 35.12% | 0.0122 |
| A2C (PG) | 1 | 1.1818 | 0.960 | 22.55% | 0.0159 |
| A2C (PG) | 2 | 1.2396 | 0.824 | 35.13% | 0.0124 |
| A2C (PG) | 3 | 1.2391 | 0.821 | 35.45% | 0.0118 |
| A2C (PG) | 4 | 1.2393 | 0.816 | 35.88% | 0.0110 |
| DDPG | 0 | 1.1349 | 1.534 | 8.82% | 0.0136 |
| DDPG | 1 | 1.1220 | 0.707 | 22.50% | 0.0212 |
| DDPG | 2 | 1.1163 | 0.916 | 16.42% | 0.0198 |
| DDPG | 3 | 1.0457 | 0.359 | 22.09% | 0.0226 |
| DDPG | 4 | 1.0798 | 0.651 | 16.18% | 0.0191 |

## Figures

* `01_wealth_curves.png` — wealth over the window (log scale), agent means with per-seed spread, against the four benchmarks.
* `02_allocation_heatmaps.png` — portfolio weights over time for the median-fAPV seed of each algorithm.
* `03_seed_distributions.png` — per-seed fAPV and Sharpe spread.

Regenerate with `python experiments/evaluate.py`.
