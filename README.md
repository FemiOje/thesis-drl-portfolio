# Reinforcement Learning for Stock Market Portfolio Management

B.Sc. Computer Science final year project, University of Lagos.
Ojetokun Oluwafemi Akinwale

A controlled comparison of three continuous-action deep reinforcement learning
algorithms — **Policy Gradient (PG)**, **Proximal Policy Optimization (PPO)** and
**Deep Deterministic Policy Gradient (DDPG)** — on an eight-asset S&P 500 portfolio
under realistic transaction costs. All three agents share the same state
representation, feature extractor, action projection, cost model, discount factor
and number of optimiser updates; the only thing allowed to differ is the learning
rule itself.

## Method

**Universe.** Eight assets, one per GICS sector, chosen mechanically as the largest
S&P 500 constituent by market capitalisation as of 2021-08-25 (`config/universe.yaml`).
Sectors are pre-specified on economic grounds (growth, rate-sensitive positive,
rate-sensitive negative, commodity, defensive x2, cyclical x2) to avoid picking
correlated proxies of the same factor; a pairwise-correlation gate (ρ ≤ 0.70,
measured on the train split only) would substitute the next-largest constituent in
any sector that breached it — no substitution was needed for the headline universe.
4-, 8- and 16-asset variants and three alternative "baskets" (rank-1/2/3 constituent
per sector) are pre-registered and built (`data/processed/`), for later robustness
sweeps.

**Data & splits.** Daily OHLC from yfinance, 2021-08-25 to 2026-08-24 (5 years),
split chronologically 60/20/20 into train (735 trading days) / validate (249) /
test (251). The test split is scored exactly once. Raw CSVs are committed so results
don't depend on a live API.

**State, action, reward.** Following the EIIE formulation (Jiang et al.'s deep RL
portfolio framework): the state is a `(3, 8, 20)` tensor of close/high/low prices
over a 20-day window, each normalised by the latest close, plus the previous
portfolio weight vector. The action is 9 raw scores (8 assets + cash) in `[-1, 1]`,
projected to the simplex by `w = softmax(τ · a)` with `τ = 5.0` — high enough that
no single asset is structurally capped (`src/config.py` refuses to load a config
where the reachable max weight falls below 0.95). The reward is the log growth rate
`log(μ_t · y_t · w_{t-1})`, where `μ_t` is a transaction-cost remainder solved by a
5-iteration fixed point (`src/costs.py`) under 10 bps commission on both purchases
and sales.

**Architecture.** All three agents use the identical EIIE convolutional extractor
(`src/extractors.py`): two 1×3 / 1×(n-2) convolutions across the time axis (never
across the asset axis, so assets stay independent until the final softmax), a third
1×1 convolution that folds in the previous weight vector, and a learned cash bias.
This is the controlled variable of the whole study — a features-extractor identity
and parameter-count check (`assert_eiie`) runs at the start of every PPO/DDPG
training call to catch SB3 silently substituting a different extractor.

**Agents.**
| | Rule | Critic | Notes |
|---|---|---|---|
| PG | Jiang's deterministic policy gradient | none | gradient flows through the cost model itself (torch-differentiable `μ_t`); trained with a Portfolio-Vector Memory so mini-batches of consecutive days can be sampled without replaying whole episodes |
| PPO | clipped surrogate (Stable-Baselines3) | value head, separate extractor instance | |
| DDPG | deterministic actor-critic + OU noise (Stable-Baselines3) | Q head, `[64, 64]`, separate extractor instance | actor kept as a linear head on the extractor (`pi=[]`) to match PG's and PPO's architecture; the critic needs real capacity to fuse state and action, an asymmetry inherent to the algorithm and reported rather than hidden |

`γ = 1.0` for all three (PG's objective is undiscounted; SB3's `γ = 0.99` default
would otherwise confound the learning rule with the planning horizon), and
`share_features_extractor = False` for PPO/DDPG (PG's extractor only ever sees the
portfolio-return gradient, so PPO's default of sharing it with the value head would
optimise the controlled variable differently in that one arm).

**Training budget.** Equalised across all three algorithms at 60,000 optimiser
updates of batch size 50 — a deliberate methodological choice (neither Jiang's nor
Liang's papers equalise gradient steps or use validation-based checkpoint selection;
this project does both so "which algorithm" is never confounded with "which one got
more updates"). PG replays the fixed 735-day train split directly; PPO and DDPG each
consume 300,000 environment steps to reach the same update count
(`src/config.gradient_steps()`), and the realised count is asserted against actual
`optimizer.step()` calls rather than SB3's internal counters, because PPO's
`_n_updates` counts epochs, not minibatches, and would silently under-count. Every
agent is trained for 10 seeds (0-9); the checkpoint reported is the one with the
best validation-split performance, evaluated every 1,500 gradient steps.

**Baselines.** Uniform Buy-And-Hold (UBAH), Uniform Constant Rebalanced Portfolio
(UCRP), a rolling 252-day long-only max-Sharpe Markowitz portfolio (rebalanced every
21 days), and Best-Stock-in-hindsight (an upper reference, not an implementable
strategy). All four run through the same `PortfolioEnv` and cost model as the
agents, so no strategy is compared against a differently-costed version of itself.

**Evaluation.** Cumulative/annualised return, annualised Sharpe and Sortino, max
drawdown, turnover, win rate, and portfolio-concentration diagnostics (HHI, entropy,
max weight) computed identically for every strategy (`src/metrics.py`).
Significance is assessed with paired t-tests against each baseline (per-seed and
aggregated across all 10 seeds), Bonferroni-corrected across the resulting
comparisons, plus bootstrap confidence intervals on the return differential
(`scripts/07_stats.py`).

## Results

Headline universe (M8, basket 1), test split (251 days, scored once):

| Strategy | Final value | CR | Sharpe | Sortino | MDD | Turnover |
|---|---|---|---|---|---|---|
| PG | 1.277 | +27.7% | 2.22 | 3.30 | 5.1% | 1.6% |
| PPO | 1.248 | +24.8% | 1.20 | 1.67 | 9.3% | 25.8% |
| DDPG | 1.240 | +24.0% | 1.31 | 1.87 | 9.3% | 2.0% |
| UBAH | 1.260 | +25.9% | 2.09 | 3.13 | 5.1% | 0.7% |
| UCRP | 1.275 | +27.5% | 2.23 | 3.30 | 5.1% | 1.7% |
| Markowitz | 1.371 | +37.1% | 2.24 | 3.63 | 8.9% | 2.6% |
| BestStock* | 1.567 | +56.7% | 2.31 | 3.77 | 11.0% | 0.8% |

\* hindsight upper reference, not an implementable strategy.

On the test split, none of the three agents beats UBAH or UCRP by a statistically
significant margin after Bonferroni correction across all 10 seeds — all three land
close to the naive baselines' risk-adjusted return, with PG tracking them most
closely (highest Sharpe/Sortino among the agents, lowest drawdown and turnover of
the three). All three agents lose significantly to Markowitz and to BestStock (PPO's
loss to BestStock does not reach significance). PPO reaches its return with far
higher turnover (25.8% vs. ~1-2% for PG and DDPG) and a much more concentrated
portfolio, consistent with a stochastic policy exploiting fewer, larger positions.
On the validation split, by contrast, DDPG and PPO *do* beat UBAH/UCRP
significantly — a validate/test divergence that is itself a finding: with only ~250
trading days per split, checkpoint selection and significance both have limited
power, and results should be read as indicative rather than conclusive (see
Limitations below). Full per-seed curves, significance tables and figures are in
`results/` (`results/phase5/metrics.csv`, `stats_test.csv`, `stats_validate.csv`,
`results/figures/`).

**Limitations.** With a 735-day training window, PG's training and validation
returns diverge sharply past its selected checkpoint — at the full 60,000-step
budget the policy fits the training window closely enough that the
validation-argmax used for checkpoint selection itself becomes unstable across
seeds. The equalised 60,000-update budget, the τ=5.0 softmax temperature, and the
validation-based checkpoint-selection protocol are all choices made for this study;
they are not reproduced from Jiang's or Liang's original papers, which use neither
convention.

## Setup

Requires **Python 3.10** (see `requirements.txt` for why).

```bash
py -3.10 -m venv .venv
.venv/Scripts/activate      # Windows;  source .venv/bin/activate on POSIX
pip install -r requirements.txt
pytest
```

`reproduce_colab.ipynb` runs the full pipeline end-to-end on Google Colab.

## Layout

```
config/     base.yaml (locked experiment parameters) and universe.yaml (the selection rule)
data/raw/   per-ticker CSVs + MANIFEST.json (committed, so results don't depend on a live API)
data/processed/  built observation tensors per universe (M4, M8_basket1-3, M16)
src/        library code (config, data, universe, env, costs, extractors, agents/, baselines,
            backtest, metrics, stats, plots, results)
scripts/    numbered entry points:
              01_build_data.py    resolve universes, download, gate, build tensors
              02_run_baselines.py UBAH/UCRP/BestStock/Markowitz through backtest()
              03_train_pg.py      train PG, select on validation, backtest
              04_pg_lr_sweep.py   PG learning-rate diagnostic (3 seeds x 3 LRs)
              05_train_sb3.py     train PPO or DDPG, select on validation, backtest
              06_figures.py       regenerate the figure suite from committed run directories
              07_stats.py         paired t-tests, Bonferroni, bootstrap CIs
tests/      the gates (config, data, env, costs, extractors, agents, stats, plots)
results/    one directory per run — meta.json, per-seed CSVs, aggregated metrics.csv,
            stats_<split>.csv, and results/figures/ for the full figure suite
```

Every parameter comes from `config/base.yaml` via `src/config.py`, which validates
it at load time — an unequal training budget, a mismatched discount factor, or a
`τ` too low to reach a concentrated portfolio all fail loudly there rather than
producing a silently-wrong result.
