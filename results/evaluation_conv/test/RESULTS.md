# Held-out test-window results

Test window: **2025-10-22 → 2026-07-22** (187 decision steps). Commission **0.25%** both sides. Agents: best-on-validation checkpoints at 500,000 timesteps, 5 seeds, acting deterministically.

The test split was touched exactly once, here. Metrics follow Jiang §6.2; turnover is Σ|Δw| per step (0-2).

## Summary (agents: mean ± std over seeds)

| Strategy | fAPV | Annualized return | Sharpe | Max drawdown | Turnover |
|---|---|---|---|---|---|
| PPO | 1.2074 ± 0.0893 | 29.04 ± 12.91% | 1.325 ± 0.312 | 12.30 ± 3.91% | 0.0936 ± 0.0476 |
| A2C (PG) | 1.1982 ± 0.0341 | 27.61 ± 4.92% | 0.988 ± 0.319 | 15.26 ± 3.95% | 0.0174 ± 0.0037 |
| DDPG | 1.1720 ± 0.0040 | 23.84 ± 0.57% | 1.207 ± 0.654 | 14.20 ± 8.49% | 0.0165 ± 0.0082 |
| Buy & Hold | 1.1506 | 20.81% | 1.852 | 5.64% | 0.0107 |
| UCRP | 1.1532 | 21.18% | 1.793 | 6.04% | 0.0226 |
| Best stock (hindsight) | 1.3975 | 56.99% | 1.811 | 20.11% | 0.0107 |
| All-cash | 1.0000 | 0.00% | n/a | 0.00% | 0.0000 |

## Per-seed detail

| Algorithm | Seed | fAPV | Sharpe | Max drawdown | Turnover |
|---|---|---|---|---|---|
| PPO | 0 | 1.2293 | 1.092 | 13.33% | 0.1558 |
| PPO | 1 | 1.1970 | 1.547 | 11.52% | 0.1157 |
| PPO | 2 | 1.3417 | 1.711 | 16.73% | 0.0876 |
| PPO | 3 | 1.1707 | 0.955 | 13.76% | 0.0261 |
| PPO | 4 | 1.0981 | 1.319 | 6.16% | 0.0827 |
| A2C (PG) | 0 | 1.1947 | 0.894 | 15.12% | 0.0156 |
| A2C (PG) | 1 | 1.1986 | 0.937 | 14.94% | 0.0166 |
| A2C (PG) | 2 | 1.2547 | 1.545 | 8.97% | 0.0228 |
| A2C (PG) | 3 | 1.1759 | 0.790 | 18.77% | 0.0132 |
| A2C (PG) | 4 | 1.1671 | 0.774 | 18.50% | 0.0191 |
| DDPG | 0 | 1.1748 | 1.670 | 8.20% | 0.0223 |
| DDPG | 1 | 1.1691 | 0.744 | 20.21% | 0.0107 |

## Figures

* `01_wealth_curves.png` — wealth over the window (log scale), agent means with per-seed spread, against the four benchmarks.
* `02_allocation_heatmaps.png` — portfolio weights over time for the median-fAPV seed of each algorithm.
* `03_seed_distributions.png` — per-seed fAPV and Sharpe spread.

Regenerate with `python experiments/evaluate.py`.
