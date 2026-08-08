# Held-out test-window results

Test window: **2025-10-22 → 2026-07-22** (187 decision steps). Commission **0.25%** both sides. Agents: best-on-validation checkpoints at 300,000 timesteps, 5 seeds, acting deterministically.

The test split was touched exactly once, here. Metrics follow Jiang §6.2; turnover is Σ|Δw| per step (0-2).

## Summary (agents: mean ± std over seeds)

| Strategy | fAPV | Annualized return | Sharpe | Max drawdown | Turnover |
|---|---|---|---|---|---|
| PPO | 1.1507 ± 0.0108 | 20.83 ± 1.53% | 1.003 ± 0.092 | 11.64 ± 1.11% | 0.0208 ± 0.0006 |
| A2C (PG) | 1.1033 ± 0.1532 | 14.60 ± 20.62% | 0.451 ± 0.697 | 21.93 ± 4.83% | 0.0128 ± 0.0022 |
| DDPG | 1.1450 ± 0.0603 | 20.08 ± 8.58% | 1.464 ± 0.702 | 9.93 ± 3.61% | 0.0200 ± 0.0034 |
| Buy & Hold | 1.1506 | 20.81% | 1.852 | 5.64% | 0.0107 |
| UCRP | 1.1532 | 21.18% | 1.793 | 6.04% | 0.0226 |
| Best stock (hindsight) | 1.3975 | 56.99% | 1.811 | 20.11% | 0.0107 |
| All-cash | 1.0000 | 0.00% | n/a | 0.00% | 0.0000 |

## Per-seed detail

| Algorithm | Seed | fAPV | Sharpe | Max drawdown | Turnover |
|---|---|---|---|---|---|
| PPO | 0 | 1.1518 | 1.131 | 9.72% | 0.0215 |
| PPO | 1 | 1.1449 | 0.973 | 11.80% | 0.0204 |
| PPO | 2 | 1.1686 | 1.062 | 11.92% | 0.0214 |
| PPO | 3 | 1.1407 | 0.911 | 12.38% | 0.0207 |
| PPO | 4 | 1.1475 | 0.937 | 12.38% | 0.0202 |
| A2C (PG) | 0 | 1.1732 | 0.771 | 19.54% | 0.0122 |
| A2C (PG) | 1 | 0.8293 | -0.795 | 30.56% | 0.0166 |
| A2C (PG) | 2 | 1.1743 | 0.776 | 19.67% | 0.0124 |
| A2C (PG) | 3 | 1.1698 | 0.756 | 19.78% | 0.0118 |
| A2C (PG) | 4 | 1.1699 | 0.749 | 20.10% | 0.0110 |
| DDPG | 0 | 1.2439 | 2.581 | 8.25% | 0.0145 |
| DDPG | 1 | 1.1318 | 1.120 | 13.44% | 0.0231 |
| DDPG | 2 | 1.1532 | 1.697 | 7.29% | 0.0193 |
| DDPG | 3 | 1.1063 | 0.810 | 14.18% | 0.0225 |
| DDPG | 4 | 1.0899 | 1.112 | 6.46% | 0.0207 |

## Figures

* `01_wealth_curves.png` — wealth over the window (log scale), agent means with per-seed spread, against the four benchmarks.
* `02_allocation_heatmaps.png` — portfolio weights over time for the median-fAPV seed of each algorithm.
* `03_seed_distributions.png` — per-seed fAPV and Sharpe spread.

Regenerate with `python experiments/evaluate.py`.
