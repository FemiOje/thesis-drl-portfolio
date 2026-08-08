# Methodology

Reference document for the thesis chapter: what was built, which equations it
implements, how it was trained and evaluated, and where it deliberately departs
from the source paper.

**Every equation number below was verified against the PDF in the repo root:
`1706.10059.pdf`, which is arXiv:1706.10059**v1** (30 Jun 2017).** Later arXiv
versions of that paper renumber some equations — cite v1 explicitly in the
thesis, or re-check the numbers against whichever version you cite.

---

## 1. Problem formalism (Jiang, Xu & Liang 2017)

A portfolio of `m` risky assets plus cash. At the end of each period `t` the
agent chooses a portfolio vector `w_t` (weights summing to 1, cash at index 0);
market movement then drives wealth from `p_{t-1}` to `p_t`.

| Concept | Jiang (v1) | Meaning |
|---|---|---|
| Price relative vector | **Eq. 1** | `y_t := v_t ⊘ v_{t-1}`, cash entry ≡ 1 |
| Costless wealth update | Eq. 2 | `p_t = p_{t-1} · y_t · w_{t-1}` |
| Rate of return | Eq. 3 | `ρ_t := p_t/p_{t-1} − 1` |
| Costless log return | Eq. 4 | `r_t := ln(y_t · w_{t-1})` |
| Initial portfolio vector | **Eq. 5** | `w_0 = (1, 0, …, 0)ᵀ` — **all capital starts in cash** |
| Costless final value | **Eq. 6** | `p_f = p_0 ∏ y_t · w_{t-1}` |
| Weight drift over a period | **Eq. 7** | `w'_t = (y_t ⊙ w_{t-1}) / (y_t · w_{t-1})` |
| Cost applied to wealth | Eq. 8 | `p_t = μ_t · p'_t` |
| Rate of return with cost | **Eq. 9** | `ρ_t = μ_t · y_t · w_{t-1} − 1` |
| **Log return with cost** | **Eq. 10** | `r_t = ln(μ_t · y_t · w_{t-1})` ← the reward |
| Final value with cost | **Eq. 11** | `p_f = p_0 ∏ μ_t · y_t · w_{t-1}` |
| μ implicit equation | **Eq. 14** | `μ_t = 1/(1 − c_p w_{t,0}) · [1 − c_p w'_{t,0} − (c_s + c_p − c_s c_p) Σ_i (w'_{t,i} − μ_t w_{t,i})⁺]` |
| μ fixed-point sequence | **Eq. 15** | `μ^(k) = f(μ^(k−1))`, converges for any `μ^(0) ∈ [0,1]` (Theorem 1) |
| μ initial guess | **Eq. 16** | `μ^(0) = c · Σ_i \|w'_{t,i} − w_{t,i}\|` (Moody et al. 1998) |
| μ is a function of | Eq. 17 | `μ_t(w_{t-1}, w_t, y_t)` |
| **Price tensor** | **Eq. 18** | `X_t` = stack of normalized price matrices, shape `(f=3, n, m)` |
| fAPV | **Eq. 27** | `p_f = p_f/p_0`, with `p_0 = 1` |
| Sharpe ratio | **Eq. 28** | `S = E[ρ_t − ρ_F] / √var(ρ_t − ρ_F)`, `ρ_F = 0` |
| Maximum drawdown | **Eq. 29** | `D = max_{τ>t} (p_t − p_τ)/p_t` |

Sections cited: **§2.4** (the two hypotheses — zero slippage, zero market
impact), **§3.1** (survival-bias avoidance in asset preselection), **§3.2**
(price tensor construction), **§6.2** (performance measures).

> Note on Eq. 14 vs Theorem 1: the statement of Theorem 1 in v1 writes `f(μ)`
> with `(w'_{t,i} − w_{t,i})⁺` inside the sum, omitting the `μ` that Eq. 14
> carries. The iteration is only a fixed point with `μ` present, so the
> implementation follows **Eq. 14**: `(w'_{t,i} − μ·w_{t,i})⁺`.

---

## 2. Code ↔ equation map

| Code | Implements |
|---|---|
| [`data/data_loader.py`](../data/data_loader.py) → `price_relatives` | Eq. 1 (stocks; the cash column ≡ 1 is prepended by the env) |
| `data_loader.price_tensor` | Eq. 18 / §3.2 — 50-day window ÷ latest close |
| `TICKERS` fixed at window start | §3.1 — survival-bias avoidance |
| [`env/portfolio_env.py`](../env/portfolio_env.py) → `softmax` | maps SB3's unbounded action to a valid `w_t` (see §5, deviation 7) |
| `portfolio_env.drifted_weights` | Eq. 7 |
| `portfolio_env.transaction_remainder` | Eq. 14–16 (fixed-point iteration, tol 1e-10) |
| `PortfolioEnv._all_cash` / `reset` | Eq. 5 — every episode starts fully in cash |
| `PortfolioEnv.step` → `reward` | Eq. 10 |
| `PortfolioEnv.step` → `self.p` | Eq. 11 |
| [`tests/test_env.py`](../tests/test_env.py) test 2 | Eq. 6 — costless wealth equals the analytic product |
| [`experiments/evaluate.py`](../experiments/evaluate.py) → `compute_metrics` | Eq. 27 (fAPV), Eq. 28 (Sharpe, annualized — deviation 2), Eq. 29 (MDD) |

---

## 3. Data

* **Source.** `data_loader._download_raw` is the **only** function that touches
  an external provider (currently yfinance, `auto_adjust=True` → split- and
  dividend-adjusted OHLC). Swapping providers — e.g. to the Microsoft source the
  supervisor suggested — touches this one function and nothing else.
  *This remains an open item with the supervisor.*
* **Universe (fixed, approved).** 8 large-cap S&P 500 constituents across
  sectors: AAPL, MSFT, JPM, JNJ, XOM, PG, AMZN, NVDA, plus cash → `m + 1 = 9`.
  Liquidity is the justification: these are among the most heavily traded US
  equities, which is the condition Jiang's §2.4 hypotheses require.
* **Survivorship.** The list is fixed **as of the start of the data window** and
  held constant throughout, per §3.1. No forward-looking ranking is used.
* **Window.** 2021-07-23 → 2026-07-22, **1254 trading days**.
* **Cleaning.** Tickers aligned on a common calendar; isolated gaps
  forward-filled; leading rows that are still incomplete are trimmed. Zero NaNs
  remain in the panel.
* **Chronological split** (never shuffled — time order is preserved):

  | Split | Index range | Dates | Days |
  |---|---|---|---|
  | Train | `[0, 878)` | 2021-07-23 → 2025-01-21 | 878 (≈3.5y) |
  | Validation | `[878, 1066)` | 2025-01-22 → 2025-10-20 | 188 (≈0.75y) |
  | **Test** | `[1066, 1254)` | 2025-10-21 → 2026-07-22 | 188 (≈0.75y) |

* **Split-boundary observation windows.** For its first ~50 decision steps, a
  val/test observation `X_t` reaches back across the split boundary into earlier
  prices. This is recent *observable* history at decision time, not future
  leakage, and it preserves the full ~50 trading days per window. Documented
  here because a reader will otherwise ask.

---

## 4. Environment

`env/portfolio_env.py`, a Gymnasium environment.

* **Observation** — a `Dict` (which is why SB3 requires `MultiInputPolicy`;
  `MlpPolicy` raises on `Dict` spaces):
  * `X`: the price tensor, `Box(0, ∞)`, shape `(3, 50, 8)` (Eq. 18)
  * `w_prev`: the previous portfolio vector, `Box(0, 1)`, shape `(9,)`
* **Action** — `Box(−10, +10)`, shape `(9,)`, passed through a softmax inside
  the env. The bounds are generous enough for softmax to reach concentrated
  allocations while staying a bounded space DDPG can respect.
* **Reward** — `r_t = ln(μ_t · y_t · w_t)` (Eq. 10).
* **Commission** — `c_s = c_p = 0.25%`, which is Jiang's own default (v1 §2.3
  states 0.25% as the maximum rate at Poloniex). `commission=0.001` reproduces a
  0.1% run; `commission=0.0` gives the analytic costless case.
* **Episodes** — training uses random ~250-step sub-windows for variety
  (`random_start=True`, chronological *within* an episode); validation and test
  use one full deterministic pass.

### Timing convention

At decision index `t` the agent picks `w_t = softmax(a)`. The cost `μ_t` is
charged for moving from the **drifted actual holdings** (Eq. 7) into `w_t`, and
the reward uses the next realized relative `y[t+1]` held over that period:
`r_t = ln(μ_t · (y[t+1] · w_t))`.

This "forward" indexing matches Eq. 10 in form while giving the agent immediate
credit for the allocation it just chose. The alternative (rewarding the previous
period's weights) is equally Jiang-faithful and passes the same tests, but gives
weaker credit assignment. With `c = 0`, `μ_t ≡ 1` and episode wealth is exactly
`∏_t (y_{t+1} · w_t)` — Eq. 6, which is what test 2 asserts.

Decision indices per split: `t_first = max(start, window−1)`, `t_last = end−2`
(needs 50 days of history for `X_t`, and the next day's return for the reward).

### Unit tests (`tests/test_env.py`, 7 passing)

The four mandated tests plus three guardrails:

1. Weights are ≥ 0 and sum to 1 at every step, under random actions, across
   multiple episodes.
2. With `c = 0`, episode wealth equals an independently computed
   `∏_t (y_{t+1}·w_t)` to 1e-8 (Eq. 6), `μ ≡ 1`, and reward equals the log gross.
3. `μ_t ∈ (0, 1]` and the iteration converges, over 2000 random rebalances × 3
   commission rates; `μ = 1` exactly for a no-trade or zero-cost rebalance.
4. An all-cash policy yields `|r_t| < 1e-6` at every step on all three splits.

---

## 5. Deviations from Jiang et al.

1. **Daily bars, not 30-minute.** Jiang trades a 30-minute period on
   cryptocurrency; this work uses daily equity bars. Fewer decisions, less
   compounding of transaction costs, and a much smaller sample.
2. **Sharpe is annualized (×√252); Jiang's Eq. 28 is not.** Jiang reports
   per-period Sharpe on 30-minute data, which is why their table shows values
   like 0.087. Annualizing is the standard convention for daily equity data and
   makes the numbers comparable to the finance literature — but it means **this
   thesis's Sharpe values are not directly comparable to Jiang's table**.
3. **Asset universe.** 8 US large-cap equities + USD cash (`m+1 = 9`) vs Jiang's
   11 cryptocurrencies + Bitcoin as cash (`m+1 = 12`).
4. **Cash is genuinely risk-free here.** USD cash has `y ≡ 1` exactly. Jiang's
   "cash" is Bitcoin, which is itself volatile — so their risk-free rate of zero
   is an approximation in a way that this setup's is not.
5. **Preselection rule.** Jiang preselects by trading volume measured just
   before the back-test. This work uses a fixed, pre-declared list chosen at the
   window start — the same survival-bias fix (§3.1), applied more bluntly.
6. **Feature channel order.** Eq. 18 stacks `(V^(lo), V^(hi), V)`; this
   implementation stacks `(High, Low, Close)`. A permutation of input channels
   is functionally irrelevant to the networks, but the order differs from the
   equation as literally written.
7. **Softmax lives in the environment, not the network.** Jiang's network emits
   the portfolio vector directly. Here SB3 owns the policy, so the env applies
   the softmax — which keeps PPO's unbounded Gaussian samples and DDPG's
   deterministic outputs valid without modifying the algorithms.
8. **No EIIE topology, no Portfolio-Vector Memory, no online stochastic batch
   learning.** These are Jiang's architectural contributions. This work uses
   stock SB3 agents with `MultiInputPolicy`; the contribution here is the
   environment and an honest three-algorithm comparison, not a network
   architecture. (EIIE was a labelled stretch goal in the plan; it was not
   reached.)
9. **μ solved to tolerance, always.** Jiang uses a fixed iteration count `k`
   during training and a tolerance during back-testing; this implementation
   iterates to `|Δμ| < 1e-10` everywhere. Strictly more accurate, marginally
   slower.

---

## 6. Algorithms

Three Stable-Baselines3 agents, all on `MultiInputPolicy` with default network
sizes, per the plan's "working end-to-end with defaults first" guard.

| Plan name | Implementation | Role in the comparison |
|---|---|---|
| PPO | `stable_baselines3.PPO` | stochastic policy-gradient representative |
| DDPG | `stable_baselines3.DDPG` | deterministic actor-critic; Gaussian `NormalActionNoise`, σ = 0.3 on the ±10 action space |
| **PG** | `stable_baselines3.A2C` | vanilla policy-gradient baseline — **see mapping note** |

### The PG → A2C mapping

The plan calls for "PG", a vanilla policy-gradient method. SB3 ships no bare
REINFORCE implementation, so **A2C is used as the practical vanilla
policy-gradient baseline** and is labelled "A2C (PG)" in every figure and table.
A2C is a direct policy-gradient method with a learned value baseline; Liang et
al.'s "PG" is likewise a direct policy-gradient method. **This substitution must
be stated explicitly in the thesis** — it is a defensible engineering choice,
not an equivalence.

It matters more than a typical implementation detail here, because PG is exactly
the algorithm Liang et al.'s headline finding is about ("PG is more desirable in
financial market than DDPG and PPO"). This work reaches the opposite ranking, so
the A2C ≠ PG gap is load-bearing in that comparison — see §9.1.

**What Liang et al.'s "PG" actually is** (verified in their §network structure):
not a generic REINFORCE. They state *"Motivated by Jiang et al., we use so
called Identical Independent Evaluators (IIE)"* — independent network flows for
the m+1 assets with **shared parameters**, each emitting a scalar preference for
one asset, the m+1 scalars then **softmax-normalized into the weight vector**.
They swap Jiang's CNN for a **deep residual network**, and for PG specifically
they say *"we adapt similar settings with Jiang's and we would not go specific
about them here"* — their hyperparameter table lists only DDPG and PPO, with no
PG row. So their PG is Jiang's EIIE-style architecture trained by direct policy
gradient. That is a **domain-specialized architecture**, which §5 deviation 8
records this work as deliberately not implementing.

### Hyperparameters as run

Exactly as recorded in each `results/models/{algo}/seed{n}/config.json`.

| | PPO | A2C (PG) | DDPG |
|---|---|---|---|
| learning rate | 3e-4 | 7e-4 | 1e-3 |
| `n_steps` (rollout) | 2048 | 16 | — |
| batch size | 64 | — | 256 |
| `n_epochs` | 10 | — | — |
| `gamma` | 0.99 | 0.99 | 0.99 |
| `gae_lambda` | 0.95 | 1.0 | — |
| `ent_coef` | 0.0 | 0.0 | — |
| replay buffer | — | — | 50,000 |
| `learning_starts` | — | — | 1000 |
| `train_freq` | — | — | 1 step |
| action noise | — | — | Normal, σ = 0.3 |

The DDPG buffer is capped at 50k (≈0.5 GB) to fit the 8 GB development laptop.
Only the plan's sanctioned tunables (learning rate, rollout/batch size) are
exposed as CLI flags, and they were tuned **on validation only**.

---

## 7. Training protocol

* **300,000 timesteps per seed, 5 seeds (0–4) per algorithm**, all three
  algorithms on an **equal step budget**. Results are reported as mean ± std
  across seeds; single-seed numbers are not reported as findings.
* Training runs on the **train** split only, as random ~250-step sub-windows.
* **Best-on-validation checkpointing.** An SB3 `EvalCallback` runs one
  deterministic full pass over the **validation** split every `steps/20`
  env-steps and saves `best_model.zip` whenever mean episode reward improves.
  Because reward is the log return, episode reward = `Σ r_t = log(fAPV_val)` —
  so the checkpoint criterion is literally validation wealth.
* **The test split is never touched during training or tuning.**
* TensorBoard logs to `results/tensorboard/`; a full run record
  (`config.json`) and the validation curve (`evaluations.npz`) are saved per seed.

### Step-budget decision (2026-07-25, reaffirmed 2026-07-30)

PPO's validation curve was still rising at 300k, so a supplementary 500k run was
done on Colab: **PPO 500k reached 1.2154 ± 0.0074 on validation**, a
non-overlapping improvement over PPO 300k (1.185 ± 0.008). The head-to-head
comparison nonetheless uses **300k for all three algorithms**, because an equal
step budget is the defensible protocol. The 500k result is reported as a
supplementary *"more steps help PPO"* finding, not as part of the comparison.
A2C/DDPG were not run at 500k — deliberately out of scope, not unfinished.

Validation results at 300k (best-on-validation fAPV, mean ± std over 5 seeds):
**A2C 1.228 ± 0.023 · PPO 1.185 ± 0.008 · DDPG 1.100 ± 0.033.**

---

## 8. Evaluation protocol

`experiments/evaluate.py`. Run **once** on the held-out test window.

* Each algorithm's **best-on-validation** checkpoint per seed, acting
  **deterministically**, over one full chronological pass of the test split
  (187 decision steps, 2025-10-22 → 2026-07-22).
* **Agents and benchmarks are stepped through the same `PortfolioEnv`.**
  Benchmarks express target weights as log-weight actions, which the env's
  softmax inverts exactly (round-trip asserted at 1e-9). There is no separate
  benchmark simulator that could drift out of sync with the agents' cost
  accounting.
* **Every strategy starts fully in cash** (Eq. 5), so all of them — buy-and-hold
  included — pay commission on their opening purchase.

### Benchmarks

| Benchmark | Definition | Jiang's name |
|---|---|---|
| Buy & Hold | equal weight across the 8 stocks at the first step, never trades again | UBAH |
| UCRP | rebalances to equal weight **every** step, paying the cost | UCRP |
| Best stock (hindsight) | highest cumulative price relative over the window, bought at the start and held — **not achievable ex ante**, an upper reference | Best Stock |
| All-cash | never enters the market; `p_t ≡ 1` | — |

### Metrics

fAPV (Eq. 27) · annualized Sharpe (Eq. 28 + deviation 2) · MDD (Eq. 29) · mean
turnover `Σ_i |w_{t,i} − w'_{t,i}|` per step, in [0, 2], where 2 is a full
rotation. Turnover is not a Jiang metric; it is reported to make cost behaviour
legible.

Sharpe is reported as **undefined** for any path whose returns are numerically
zero (all-cash): the env's softmax cannot emit an exact zero weight, so ~1e-12
leaks into the stocks, and dividing that noise by its own near-zero standard
deviation produces a meaningless O(1) number.

### Verification of the benchmark accounting

At `c = 0`, the env-driven benchmarks match closed-form values:

* equal-weight buy-and-hold = 1.1534554318 vs analytic mean of per-stock
  cumulative relatives = 1.1534554318 (difference 1.7e-13);
* best-stock = analytic max of cumulative relatives (difference 2.3e-12);
* post-opening turnover ≤ 7e-13; `μ ≡ 1` throughout.

At `c = 0.25%`, buy-and-hold's fAPV equals its costless value × (1 − 0.0025) —
exactly one opening commission, as expected for a strategy that never trades
again. Both hold-style benchmarks report mean turnover 0.0107 = 2/187, the
single opening trade amortized over the window.

---

## 9. Results

Full tables in [`results/evaluation/test/RESULTS.md`](../results/evaluation/test/RESULTS.md).
Test window, 187 steps, `c = 0.25%`, mean ± std over 5 seeds:

| Strategy | fAPV | Sharpe | MDD | Turnover |
|---|---|---|---|---|
| PPO | 1.1507 ± 0.0108 | 1.003 ± 0.092 | 11.64 ± 1.11% | 0.0208 |
| A2C (PG) | 1.1033 ± 0.1532 | 0.451 ± 0.697 | 21.93 ± 4.83% | 0.0128 |
| DDPG | 1.1450 ± 0.0603 | 1.464 ± 0.702 | 9.93 ± 3.61% | 0.0200 |
| Buy & Hold | 1.1506 | 1.852 | 5.64% | 0.0107 |
| UCRP | 1.1532 | 1.793 | 6.04% | 0.0226 |
| Best stock (hindsight, XOM) | 1.3975 | 1.811 | 20.11% | 0.0107 |
| All-cash | 1.0000 | n/a | 0.00% | 0.0000 |

Four findings the write-up should state plainly:

1. **No agent beats the passive benchmarks on risk-adjusted return.** All three
   land at or below UCRP and Buy-and-Hold on fAPV while carrying roughly 2–4×
   the drawdown, so every agent Sharpe sits well below the benchmarks' ~1.8. The
   agents reach a comparable return by taking materially more risk.
2. **The validation ranking did not survive the test window.** Validation said
   A2C > PPO > DDPG; test says PPO ≈ DDPG > A2C. A2C's advantage was specific to
   the validation window — a clean demonstration of why the split was held out.
3. **Seed variance dominates the gaps between algorithms.** A2C seed 1 collapses
   to fAPV 0.8293 (Sharpe −0.795) while its other four seeds sit tightly near
   1.17; DDPG spans 1.0899–1.2439. PPO is by far the most stable
   (1.1407–1.1686), consistent with its tight validation spread. Any single-seed
   claim here would have been noise.
4. **The learned policies are not merely near-static — they are constant.**
   The allocation heatmaps show flat weights across all 187 steps, and direct
   measurement (**§9.2**) confirms all 15 policies emit the same weight vector
   every day and ignore the price tensor entirely. The agents converge to a
   fixed allocation and rebalance to it daily (turnover ~0.01–0.02/step) rather
   than timing the market. A2C's median seed sits at ~100% NVDA; PPO's holds
   ~60% NVDA plus a spread. With this observation space and reward, the agents
   learn *an allocation*, not a *strategy* — the central finding of the chapter,
   and the one that reframes findings 1–3.

### 9.1 Comparison with Liang et al. (2018)

Liang et al. run the same three-algorithm comparison (DDPG, PPO, PG) on **China
A-share** data, with a risk-adjusted APV objective and their own "Adversarial
Training" modification. Their findings and this work's line up on three points
and disagree on one — all four are worth a paragraph in the discussion chapter.

**Where the results agree** (independent corroboration on a different market):

* **DRL underperforms expectations in this domain.** Their conclusion states
  that "reinforcement learning does not gain such remarkable performance in
  portfolio management so far as those in game playing or robot control." This
  work's agents likewise fail to beat passive benchmarks on risk-adjusted return
  (finding 1).
* **Instability across runs.** They report that "deep reinforcement learning is
  highly sensitive so that its performance is unstable" — the same phenomenon as
  this work's finding 3, where seed variance dominates the gaps between
  algorithms (A2C seed 1 at fAPV 0.8293 against ~1.17 for its other four seeds).
* **Degenerate, concentrated allocations.** They report "the degeneration of our
  reinforcement learning agent, which often tends to buy only one asset at a
  time." This is precisely finding 4: the allocation heatmaps here show
  near-static weights, with A2C's median seed sitting at ~100% NVDA. Two
  independent implementations, two different markets, the same failure mode —
  this is the strongest external support for the chapter's central observation.
* **Beating UCRP requires more than the stock algorithms.** Their PG agent
  outperforms UCRP only *after* the Adversarial Training modification. This
  work implements no such modification, and correspondingly no agent beats UCRP
  (1.1532) — consistent rather than contradictory.

**Where the results disagree:**

* **Algorithm ranking.** Liang et al. conclude that "PG is more desirable in
  financial market than DDPG and PPO, although both of them are more advanced."
  This work's test window gives the opposite order: PPO (1.1507) ≈ DDPG (1.1450)
  > A2C/PG (1.1033), with PPO also the most stable across seeds.

Three caveats before treating that as a genuine contradiction — the write-up
should state all three rather than claim a refutation:

1. **The PG substitution is much wider than "A2C vs REINFORCE" (§6).** Their PG
   is not a generic policy-gradient agent: it is **Jiang's IIE architecture**
   (per-asset network flows with shared weights, softmax head), with residual
   blocks replacing Jiang's CNN, trained by direct policy gradient. This work's
   "PG" is stock SB3 A2C on `MultiInputPolicy` with default networks. So the
   comparison is *general-purpose deep RL algorithm* vs *domain-specialized
   architecture* — and their result that PG beats DDPG and PPO may well be
   reporting **the architecture's advantage rather than the algorithm
   family's**, since it is the one agent of their three carrying Jiang's design.
   A2C's single collapsed seed drives most of the remaining gap.
2. **Different markets.** China A-shares (with the market irregularities and
   government intervention their conclusion explicitly blames for non-stationary
   transitions) versus 8 US large-cap S&P 500 equities.
3. **Different budgets, objectives and horizons.** They tune across optimizers,
   learning rates, objective functions and feature combinations with a
   risk-adjusted objective; this work fixes an equal 300k-step budget across
   algorithms with a log-return reward and a light validation-only tuning of two
   hyperparameters.

### 9.2 Policy degeneracy — the central finding

Finding 4 above was verified directly rather than inferred from the heatmaps.
`experiments/policy_diagnostic.py` runs each checkpoint's deterministic policy
over the test window and measures how far its output weights move.

| Algorithm | Largest movement of any weight over 187 days |
|---|---|
| PPO | 4.1e-07 – 1.3e-06 |
| A2C (PG) | 2.7e-08 – 4.1e-07 |
| DDPG | exactly 0.0 |

**All 15 policies are constant functions.** A trading policy moves weights by
O(0.1); these move by float32 rounding error. The result survives a harder
control: querying each policy on observations from the start, middle and end of
each split — windows sharing no data at all — produces the same weights, with
`w_prev` held fixed so any movement would be attributable to `X_t` alone.

DDPG is the extreme case. Its actor output is saturated at the action bounds
(every component exactly ±10), identical between the `best` and `final`
checkpoints, and its validation score is **bit-identical across all 20
evaluations** from 15k to 300k steps. It froze before the first evaluation.

**What this means for the thesis.** The reported fAPV differences are
differences between *constant portfolios*, not between trading strategies. The
agents' turnover (~0.02/step) is drift correction — rebalancing back to a fixed
target — which is why it nearly matches UCRP's 0.0226. The train→test
progression (§9, table) is then readable as the optimizer selecting the
historically best fixed allocation: A2C's median seed holds ~100% NVDA, the best
stock of the *training* window (cumulative relative 6.80), which stopped winning
out of sample.

**Plausible causes — hypotheses, not tested here:**

1. **The price tensor's structure is destroyed before the network sees it.**
   This is the most concrete cause, and it is verifiable from the saved
   checkpoints rather than inferred. SB3's `MultiInputPolicy` builds a
   `CombinedExtractor`, which gives a `Dict` entry a CNN only if it is an *image
   space*; everything else gets `nn.Flatten()`. A `(3, 50, 8)` Box is not an
   image space, so `X_t` is flattened to 1200 numbers and concatenated with
   `w_prev` (9) into a **1209-dimensional vector**, fed to the default MLP:

   ```
   'X'      -> Flatten(start_dim=1, end_dim=-1)     # 3 x 50 x 8 = 1200
   'w_prev' -> Flatten(start_dim=1, end_dim=-1)     # 9
   features_dim: 1209
   policy net: Linear(1209, 64) -> Tanh -> Linear(64, 64) -> Tanh
   ```

   **There is no convolution anywhere in the trained models.** Every piece of
   structure Eq. 18 encodes — that the 50 entries are consecutive days, the 8
   columns different assets, the 3 channels high/low/close — is discarded at the
   flatten. The network must learn 1209 unrelated input weights from scratch.
   This is precisely what Jiang's EIIE topology exists to prevent: per-asset
   flows with *shared* parameters learn "how to evaluate an asset" once, instead
   of separately for all 8. The plan listed a CNN feature extractor as optional
   "if time"; it was never added, and the diagnostic suggests it was not
   optional.
2. **Input scale.** After Eq. 18 normalization, every input sits at
   0.988 ± 0.092: a near-constant, non-zero-centred signal. A default SB3 MLP
   (orthogonal init, gain √2) sees almost no variation to key on. No observation
   normalization (`VecNormalize`) was used. Combined with cause 1, the first
   layer receives 1209 inputs that are nearly all the same number.
3. **Reward signal-to-noise.** Daily log returns have mean 6.7e-04 against
   standard deviation 0.0191 — a per-step SNR of ~0.035. The policy gradient is
   roughly 97% noise.
4. **A constant allocation may be the rational local optimum** at that SNR. If
   next-day direction is not learnable from this observation, the best available
   policy *is* a fixed diversified portfolio. The agents may have solved the
   problem as posed.
5. **DDPG additionally saturated** its tanh actor against the ±10 bounds, which
   zeroes the gradient and freezes the policy permanently. The wide bounds were
   chosen (§4) so softmax could reach concentrated allocations; that choice
   appears to have cost DDPG its ability to learn at all. Its signature is
   visible in the allocation heatmap: five assets at *exactly* 20.0% each, which
   is what softmax returns for five saturated `+10`s and four `−10`s.

Cheapest discriminating experiments, in order: add observation normalization and
retrain one algorithm; narrow DDPG's action bounds to ±3; replace the flattening
feature extractor with a convolutional or EIIE-style one (§9.1) — cause 1 and
the Liang comparison point at the same fix from opposite directions.

---

## 10. Limitations

* **Zero slippage and zero market impact** are assumed throughout (Jiang §2.4).
  Orders fill at the close. Realistic for this universe's liquidity at small
  size; not realistic at scale.
* **Daily bars** — far fewer decision points than Jiang's 30-minute setting.
* **Small universe** — 8 assets + cash. Conclusions may not transfer to a
  broader or less liquid universe.
* **One market regime.** The test window is a single ~9-month stretch of one
  market. It is a *sample of one* regime; nothing here establishes behaviour
  across bull/bear cycles.
* **Five seeds** is enough to expose variance (and it did) but not enough for
  a statistical significance claim. No hypothesis test is reported, and none
  should be inferred from the mean ± std figures.
* **Hindsight benchmark.** "Best stock" uses future information by construction
  and is included only as an upper reference.
* **No hyperparameter search worth the name.** Only learning rate and
  rollout/batch size were adjustable, tuned lightly on validation. A stronger
  DDPG or A2C very likely exists.

---

## 11. Reproduction

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
python -m pip install -r requirements.txt

python experiments/run_all.py        # data → tests → training → evaluation
```

`run_all.py --skip-train` reuses the existing checkpoints in `results/models/`
and reproduces only the evaluation, tables and figures (~1 minute). The full
run retrains 3 algorithms × 5 seeds and takes roughly **6–7 hours on CPU**,
with DDPG accounting for about 80% of that.

Verified environment: Python 3.12.10, torch 2.13.0+cpu, stable-baselines3 2.9.0,
gymnasium 1.0.0, numpy 1.26.4, pandas 2.3.3, yfinance 1.5.2, matplotlib 3.11.1.

Raw CSVs are cached under `data/raw/` and are never re-downloaded by an
experiment; `data/processed/dataset.npz` caches the assembled panel.

---

## 12. References

* **Jiang, Z., Xu, D. & Liang, J. (2017).** *A Deep Reinforcement Learning
  Framework for the Financial Portfolio Management Problem.*
  arXiv:1706.10059**v1**. — The environment, formalism and metrics. PDF in the
  repo root; all equation numbers in this document verified against v1.
* **Liang, Z., Chen, H., Zhu, J., Jiang, K. & Li, Y. (2018).** *Adversarial Deep
  Reinforcement Learning in Portfolio Management.* arXiv:1808.09940**v3**
  [q-fin.PM], 18 Nov 2018. Likelihood Technology / Sun Yat-sen University. —
  The DDPG / PPO / PG comparison this thesis mirrors. PDF in the repo root
  (`1808.09940v3.pdf`); citation verified against it. See §9.1 for how their
  findings compare with this work's.
* **Filos, A. (2018).** *Reinforcement Learning for Portfolio Management.* MEng
  dissertation, Imperial College London; arXiv:1909.09571. — Also in the repo
  root. A separate work; it is **not** the Liang et al. reference and should not
  be cited as such.
* **Ormos, M. & Urbán, A. (2013)** — the recursive transaction-cost formulation
  Jiang extends (Eq. 12–14).
* **Moody, J. et al. (1998)** — the `μ` approximation used as Eq. 16's initial
  guess.

---

## 13. Open items

1. **Data source.** yfinance is in use; a Microsoft data source was recommended.
   The swap costs one function (`_download_raw`) but would invalidate the
   trained checkpoints and require re-running the sweep.
2. **PG → A2C substitution** — §6; flagged for explicit approval in the write-up.
   Note this is the one substitution that weakens the direct comparison with
   Liang et al., whose headline result is specifically about PG (§9.1).

*(Resolved 2026-07-30: the Liang et al. (2018) citation — the correct paper is
now in the repo and verified, §12.)*
