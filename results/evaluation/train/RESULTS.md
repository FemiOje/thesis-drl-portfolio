# Training-window results (in-sample diagnostic)

Train window: **2021-10-04 → 2025-01-21** (828 decision steps). Commission **0.25%** both sides. Agents: best-on-validation checkpoints at 300,000 timesteps, 5 seeds, acting deterministically.

**In-sample diagnostic, not a headline result.** The agents were trained on this window, so these figures show fit, not generalization. Metrics follow Jiang §6.2; turnover is Σ|Δw| per step (0-2).

## Summary (agents: mean ± std over seeds)

| Strategy | fAPV | Annualized return | Sharpe | Max drawdown | Turnover |
|---|---|---|---|---|---|
| PPO | 3.5585 ± 0.2512 | 47.09 ± 3.23% | 1.326 ± 0.004 | 45.20 ± 3.08% | 0.0148 ± 0.0003 |
| A2C (PG) | 5.6436 ± 2.0202 | 66.54 ± 23.72% | 1.239 ± 0.194 | 60.39 ± 11.05% | 0.0046 ± 0.0022 |
| DDPG | 1.9846 ± 0.4795 | 22.54 ± 9.47% | 1.069 ± 0.167 | 25.83 ± 12.23% | 0.0126 ± 0.0026 |
| Buy & Hold | 2.1672 | 26.54% | 1.258 | 22.51% | 0.0024 |
| UCRP | 1.9315 | 22.18% | 1.163 | 23.78% | 0.0134 |
| Best stock (hindsight) | 6.7847 | 79.09% | 1.327 | 66.34% | 0.0024 |
| All-cash | 1.0000 | 0.00% | n/a | 0.00% | 0.0000 |

## Per-seed detail

| Algorithm | Seed | fAPV | Sharpe | Max drawdown | Turnover |
|---|---|---|---|---|---|
| PPO | 0 | 3.1579 | 1.330 | 39.79% | 0.0153 |
| PPO | 1 | 3.4658 | 1.325 | 45.67% | 0.0149 |
| PPO | 2 | 3.7136 | 1.321 | 46.65% | 0.0147 |
| PPO | 3 | 3.6965 | 1.324 | 46.63% | 0.0146 |
| PPO | 4 | 3.7590 | 1.332 | 47.25% | 0.0145 |
| A2C (PG) | 0 | 6.4606 | 1.325 | 64.93% | 0.0041 |
| A2C (PG) | 1 | 2.0358 | 0.892 | 40.64% | 0.0083 |
| A2C (PG) | 2 | 6.4340 | 1.323 | 64.82% | 0.0042 |
| A2C (PG) | 3 | 6.5575 | 1.328 | 65.44% | 0.0038 |
| A2C (PG) | 4 | 6.7302 | 1.327 | 66.10% | 0.0027 |
| DDPG | 0 | 1.3075 | 0.793 | 8.39% | 0.0083 |
| DDPG | 1 | 2.3482 | 1.133 | 28.69% | 0.0136 |
| DDPG | 2 | 1.7761 | 1.059 | 26.33% | 0.0130 |
| DDPG | 3 | 2.5188 | 1.127 | 42.54% | 0.0150 |
| DDPG | 4 | 1.9723 | 1.234 | 23.20% | 0.0133 |

## Figures

* `01_wealth_curves.png` — wealth over the window (log scale), agent means with per-seed spread, against the four benchmarks.
* `02_allocation_heatmaps.png` — portfolio weights over time for the median-fAPV seed of each algorithm.
* `03_seed_distributions.png` — per-seed fAPV and Sharpe spread.

Regenerate with `python experiments/evaluate.py`.
