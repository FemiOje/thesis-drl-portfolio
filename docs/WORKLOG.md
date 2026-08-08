# Work Log

Running log of what changed, why, and any decision the supervisor/student needs to make.

---

## 2026-07-23 — Setup + smoke test

**What was built**
- Repo skeleton created: `data/{raw,processed}/`, `env/`, `experiments/`, `results/`,
  `notebooks/`, `docs/`, `tests/` (empty dirs tracked via `.gitkeep`).
- `requirements.txt` — pinned stack: torch (CPU), stable-baselines3, gymnasium, numpy (<2.0),
  pandas, yfinance, matplotlib, tensorboard, pytest.
- `.gitignore` — ignores venv, raw/processed data CSVs, training artifacts (models, runs,
  figures); keeps `.gitkeep` placeholders.
- `README.md` — project overview, setup steps, progress checklist.
- `experiments/smoke_test.py` — trains SB3 PPO on `Pendulum-v1` for 10k steps.

**Environment**
- No usable Python existed on the machine at start (only a 0-byte Microsoft Store stub).
  Installed **Python 3.12.10** (per-user, python.org silent install, PrependPath=1).
- Created venv at `.venv/`. Installed all requirements. Key resolved versions:
  torch 2.13.0+cpu · stable-baselines3 2.9.0 · gymnasium 1.0.0 · numpy 1.26.4 ·
  pandas 2.3.3 · yfinance 1.5.2.
- torch failed to load initially — the **Microsoft Visual C++ 2015–2022 x64 Redistributable**
  was missing (vcruntime140/msvcp140 DLLs absent). Installed vc_redist.x64.exe (elevated,
  silent). DLLs now present; torch imports fine.

**Verification**
- `python experiments/smoke_test.py` → SMOKE TEST PASSED. 10,000 timesteps in ~6.3s on CPU;
  eval episode ran; loss decreasing across 5 iterations. Confirms torch + gymnasium + SB3
  stack is functional before any custom thesis code.

**Decisions / talking points for the supervisor**
- **Data source (open):** plan uses yfinance for now; supervisor recommended a Microsoft data
  source. To confirm before finalizing the write-up. Download will be isolated behind one
  swappable function in the data pipeline so switching sources is cheap.
- No other design decisions pending. Stock list, split dates, commission (0.25%) are per the
  approved plan and will be applied in the data pipeline and environment.

**Next:** the data pipeline (`data/data_loader.py`): download 5y daily adjusted OHLC
for the fixed 8-stock universe, chronological train/val/test split, Jiang §3.2 price-tensor
builder. Awaiting go-ahead.

---

## 2026-07-23 — Data pipeline

**What was built**
- `data/data_loader.py` — end-to-end pipeline:
  - `_download_raw` is the SINGLE swappable data-source boundary (only function that
    imports yfinance). Downloads adjusted OHLC (`auto_adjust=True`); swap its body to
    change providers (e.g. the supervisor's Microsoft source) without touching anything else.
  - `load_ohlc` caches per-ticker CSVs to `data/raw/` and reloads from disk thereafter —
    experiments never re-download (reproducibility).
  - `build_panel` aligns tickers on a common calendar, forward-fills isolated gaps, trims
    leading NaNs → dense panel `(T, m, 3)`, features (High, Low, Close).
  - `chronological_split` — 70/15/15 by date, no shuffling.
  - `price_tensor` — Jiang Eq. 18: 50-day window ÷ latest close → `X_t` shape `(3, 50, m)`.
  - `price_relatives` — Jiang Eq. 1: `y_t = close_t / close_{t-1}` (stocks; cash added by env).
  - `build_dataset` assembles + caches `data/processed/dataset.npz`.
- `notebooks/01_data_sanity.ipynb` (+ runnable mirror `notebooks/plot_tensors.py`) — plots
  adjusted closes with split lines, the normalized-tensor heatmap, in-window normalized
  trajectories, and the price-relative histogram.

**Data as built (2026-07-23)**
- Universe (fixed, approved): AAPL, MSFT, JPM, JNJ, XOM, PG, AMZN, NVDA + cash.
- Window `2021-07-23 → 2026-07-22`, **1254 trading days**. Splits:
  train `[0,878)` (2021-07-23→2025-01-21, 878d ≈ 3.5y) ·
  val `[878,1066)` (2025-01-22→2025-10-20, 188d ≈ 0.75y) ·
  test `[1066,1254)` (2025-10-21→2026-07-22, 188d ≈ 0.75y, held out).

**Verification**
- `python data/data_loader.py` → DATA PIPELINE OK. Close-channel latest column == 1.0 for
  all 8 assets (Eq. 18 exact, max deviation 0.0). Panel has zero NaNs post-clean.
- All 6 extreme daily relatives (|move| ≳ 13%) traced to real events (AMZN Q1'22 −14%,
  NVDA AI-guidance +24% 2023-05-25, DeepSeek −17% 2025-01-27, tariff-pause rally 2025-04-09)
  → confirms adjustment is clean, not glitched. y range [0.830, 1.244].
- `python notebooks/plot_tensors.py` → 3 figures to `results/data_checks/`; Eq. 18 assertion passes.

**Decisions / notes**
- **Split-boundary observation windows (decided, documented):** for the first ~50 decision
  steps of val/test, `X_t` reaches back across the split boundary into earlier prices. This
  is recent *observable* history at decision time (not future leakage) and preserves ~50
  trading days per window. Standard practice; noted here for the write-up.
- Data source still yfinance; the Microsoft-source talking point from setup remains open but is
  cheap to switch (one function). No stock-list / split / commission changes.

**Next:** the environment — `env/portfolio_env.py` (Gymnasium env, Jiang formalism: Dict obs of
`X_t` + prev weights, softmax action, μ_t fixed-point cost, log-return reward) + the four
non-negotiable pytest unit tests. Awaiting go-ahead.

---

## 2026-07-24 — Portfolio environment + unit tests

**What was built**
- `env/portfolio_env.py` — Gymnasium env implementing Jiang's formalism, with each
  equation-bearing function citing its Jiang equation number:
  - `softmax` — maps SB3's unbounded raw action to valid weights (>=0, sum 1); softmax-in-env.
  - `drifted_weights` — Jiang Eq. 7, `w' = (y⊙w)/(y·w)`.
  - `transaction_remainder` — μ_t via Jiang's fixed-point iteration (Eq. 14–16), initialized
    at `μ⁰ = c·Σ|w'_i − w_i|` (Eq. 16), iterated to `|Δμ| < 1e-10`. Returns exactly 1 when c=0.
  - `PortfolioEnv` — Dict observation (`X`: 3×50×m Box≥0; `w_prev`: m+1 Box[0,1]) → forces
    SB3 `MultiInputPolicy`. Action = Box(−10,10, m+1) → softmax. Reward
    `r_t = ln(μ_t · y_t · w_t)` (Eq. 10). Tracks wealth `p_t`, per-step μ, turnover, weights,
    dates (for the evaluation plots). Supports full-window episodes or random ~N-step sub-windows.
  - `make_env(split, commission, episode_length, random_start)` — factory over the cached
    dataset (never re-downloads); `random_start` defaults True for train only.
- `tests/test_env.py` — the four mandated tests + 3 guardrails (pytest).

**Timing convention (decided, documented)**
- "Forward" indexing: at decision index `t` the agent picks `w_t = softmax(action)`; μ_t is
  charged for moving from the *drifted actual holdings* (Eq. 7) into `w_t`; the reward uses the
  next realized relative `y = y[t+1]` while `w_t` is held → `r_t = ln(μ_t · (y·w_t))`. This
  matches Jiang Eq. 10 in form and gives the RL agent immediate credit for the allocation it
  just chose. With c=0, μ_t≡1 so episode wealth is exactly Π_t (y·w_t) — the analytic product
  asserted by test 2. Alternative (pure PVM-backward: reward the *previous* weight's return)
  was rejected for weaker credit assignment; both are Jiang-faithful and pass all four tests.
- Decision indices per split: `t_first = max(start, window−1)`, `t_last = end−2` (needs 50d
  history for `X_t` and the next-day return `y[t+1]`). Cross-boundary observation windows for
  early val/test steps are observable history, per the data-pipeline note — no future leakage.

**Verification**
- `python -m pytest tests/test_env.py -v` → **7 passed**. Mandated: (1) weights ≥0 & sum to 1
  every step under random actions across 3 episodes; (2) c=0 wealth == independently-computed
  Π_t(y_{t+1}·w_t) to 1e-8 (and μ==1, reward==ln gross); (3) μ ∈ (0,1] and converges over 2000
  random rebalances × 3 commission rates, μ==1 exactly for no-trade / zero-cost; (4) all-cash
  policy gives |r_t|<1e-6 every step on all three splits, wealth stays 1.0.
- `python env/portfolio_env.py` → ENV OK (full 828-step train episode, obs shapes (3,50,8) &
  (9,), weights sum 1).
- SB3 `check_env` passes; `PPO("MultiInputPolicy", env).learn(500)` runs clean → training is
  de-risked (the Dict observation is required to use `MultiInputPolicy`, not `MlpPolicy`).

**Decisions / notes**
- Action bounds ±10 (generous so softmax can reach concentrated allocations; DDPG respects
  them). Commission default 0.25% both sides; 0.1% available via `commission=0.001`.
- No dependency, stock-list, split, or commission-default changes.

**Next:** training — `experiments/train.py` (CLI `--algo {ppo,ddpg,a2c} --seed --steps`), SB3
`MultiInputPolicy`, 300k steps × 5 seeds × 3 algos, TensorBoard logging, best-on-validation
checkpointing. Awaiting go-ahead.

---

## 2026-07-24 — Training pipeline (script built + verified; full sweep pending)

**What was built**
- `experiments/train.py` — SB3 training CLI, `MultiInputPolicy` (required for the Dict obs):
  - Algorithms: `ppo` (PPO), `a2c` (A2C = the "PG" baseline; mapping stated in docs), `ddpg`
    (DDPG + Gaussian `NormalActionNoise`, σ=0.3 on the ±10 action space).
  - Trains on the TRAIN split as random ~250-step sub-windows (`random_start=True`) for variety.
  - **Best-on-validation checkpointing** via SB3 `EvalCallback`: deterministic full pass over
    the VAL split every `steps//20` env-steps; saves `best_model.zip` when val mean episode
    reward improves. That reward == Σ log-returns == `log(fAPV_val)`, so the checkpoint
    metric is validation wealth. **The TEST split is never touched in training.**
  - Per-algo default hyperparameters; the 2–3 tunables (lr, rollout `--n-steps`, `--batch-size`)
    are CLI flags — tune on VAL only. TensorBoard logs to `results/tensorboard/`.
  - DDPG replay buffer capped at 50k (`--buffer-size`) ≈ 0.5 GB for the 8 GB laptop.
  - CLI: `--algo {ppo,a2c,ddpg,all} --seed N | --seeds N...` `--steps` (default 300k)
    `--commission` (default 0.25%). One command can run the whole 3×5 sweep; prints per-algo
    mean ± std of best-on-val fAPV.
  - Artifacts per run: `results/models/{algo}/seed{n}/{best_model,final_model}.zip`,
    `evaluations.npz` (val curve), `config.json` (full run record for reproducibility).

**Verification (smoke run, 2000 steps each, then deleted)**
- `python experiments/train.py --algo all --seed 0 --steps 2000` → all three algos trained,
  evaluated on val, checkpointed, and reported fAPV without error (ppo 1.123 / a2c 1.125 /
  ddpg 1.171 — meaningless at 2k steps; only proves plumbing).
- Confirmed on disk: `best_model.zip`, `final_model.zip`, `config.json`, `evaluations.npz`,
  and a TensorBoard run per (algo, seed). A saved `best_model.zip` reloads and `predict()`s a
  valid weight vector (sums to 1) on the held-out test env.
- Smoke artifacts then cleared so the real sweep starts clean.

**Measured throughput (this laptop, CPU) → full-sweep estimate**
- PPO ≈ 10 min/seed · A2C ≈ 5 min/seed · **DDPG ≈ 65 min/seed** (1 gradient step/env-step is
  the bottleneck). Full 3 algos × 5 seeds × 300k ≈ **6–7 hours**, DDPG being ~80% of it.

**Decision needed before the sweep (compute budget — see plan "Scope guards"):** run the full
300k sweep locally in the background (~6–7 h), run a reduced first pass (e.g. 150k, ~3–3.5 h),
or offload to Colab. Student chose: **full 300k local, background.**

---

## 2026-07-25 — Sweep results (300k local complete; 500k on Colab in progress)

**300k local sweep — DONE (3 algos × 5 seeds, best-on-validation fAPV):**
| algo | mean ± std | seeds |
|------|-----------|-------|
| A2C  | **1.228 ± 0.023** | 1.240, 1.182, 1.240, 1.239, 1.239 |
| PPO  | **1.185 ± 0.008** | 1.171, 1.188, 1.196, 1.184, 1.187 |
| DDPG | **1.100 ± 0.033** | 1.135, 1.122, 1.116, 1.046, 1.080 |

- Validation ranking: **A2C > PPO > DDPG.** A2C best but higher seed variance; PPO very tight;
  DDPG weakest and most variable (seed3 only 1.046) — consistent with DDPG's known instability.
- Wall-clock note: DDPG dominated (~5 h/seed on this laptop under concurrent load; one PPO seed
  also stretched to ~6 h when the machine was busy). Results unaffected; DDPG is simply the
  expensive/finicky algorithm to train here — a fair point for the write-up.

**500k refinement on Colab (parallel, `--device` added to train.py for optional GPU):**
- Motivation: PPO's eval was still rising at 300k (undertrained). Ran 500k on Colab, saving
  checkpoints to Google Drive (survives disconnects; per-seed resumable).
- **PPO 500k: 1.2154 ± 0.0074** (seeds 1.203, 1.213, 1.222, 1.223, 1.217). **Non-overlapping
  improvement** over PPO 300k (worst 500k seed 1.203 > best 300k seed 1.196) → **adopt PPO 500k.**
- A2C / DDPG at 500k: pending (A2C had plateaued by 300k so likely marginal; DDPG 500k is the
  long one, being run in seed batches).
- Colab bundle: `colab_bundle.zip` (code + cached CSVs; gitignored). Setup = mount Drive, unzip,
  pip install SB3/gymnasium/yfinance, symlink `results/` → Drive.

**Step-budget decision (student, 2026-07-25):** use **300k for all three algorithms** at evaluation
— an equal-budget comparison, which is the defensible protocol. The PPO 500k result
(1.215 ± 0.007) is reported as a **supplementary "more steps help PPO" finding**, not part of
the head-to-head. Evaluated models = best-on-validation checkpoints at 300k:
PPO 1.185 · A2C 1.228 · DDPG 1.100. **Test split still untouched.**

**Next:** evaluation — `experiments/evaluate.py`: run each algo's best-on-val 300k model ONCE on the
held-out test window; benchmarks (Buy-and-Hold, UCRP, best-stock, all-cash); metrics fAPV /
Sharpe / MaxDrawdown + turnover; wealth-curve, allocation-heatmap, seed-distribution figures.
Awaiting go-ahead. No test-window evaluation has been run.

---

## 2026-07-30 — Held-out test-window evaluation

**Scope confirmed before running:** 300k equal-budget head-to-head stands (student, 2026-07-30);
PPO-500k remains a supplementary finding, and the A2C/DDPG 500k runs listed as "pending" on
2026-07-25 are **not needed** — they are dropped from scope, not unfinished. Nothing pushed to a
remote.

**What was built**
- `experiments/evaluate.py` — test-window evaluation, benchmarks, metrics, figures.
  - Loads the best-on-validation 300k checkpoint per (algo, seed) and runs one **deterministic**
    full pass over the TEST split. **The test window was touched exactly once, here.**
  - **Single cost path:** benchmarks are driven through the *same* `PortfolioEnv` as the agents by
    expressing target weights as log-weight actions (`weights_to_action`; the env's softmax is the
    exact inverse, round-trip asserted at 1e-9). So Jiang Eq. 14-16 is literally the same code for
    every strategy — no parallel benchmark simulator to drift out of sync.
  - **Every strategy starts in cash**, so all of them (buy-and-hold included) pay commission on
    their opening purchase. No strategy gets a free entry.
  - Benchmarks: Buy-and-Hold (equal weight, opens once then targets its own drifted holdings →
    turnover 0, μ=1 thereafter) · UCRP (rebalances to equal weight daily, pays for it) ·
    Best single stock (hindsight, upper reference, labelled as such) · All-cash.
  - Metrics (Jiang §6.2): fAPV · annualized Sharpe on ρ_t = exp(r_t)−1 (√252, ddof=1) · MDD ·
    mean turnover Σ|Δw| ∈ [0,2].
  - Outputs to `results/evaluation/<split>/`: `RESULTS.md`, `summary.csv`, `per_seed_results.csv`, and
    3 figures. CLI: `--algos --seeds --commission --checkpoint {best,final} --no-figures`.

**Test-window results (2025-10-22 → 2026-07-22, 187 steps, c = 0.25%, mean ± std over 5 seeds)**

| Strategy | fAPV | Sharpe | MDD | Turnover |
|---|---|---|---|---|
| PPO | 1.1507 ± 0.0108 | 1.003 ± 0.092 | 11.64 ± 1.11% | 0.0208 |
| A2C (PG) | 1.1033 ± 0.1532 | 0.451 ± 0.697 | 21.93 ± 4.83% | 0.0128 |
| DDPG | 1.1450 ± 0.0603 | 1.464 ± 0.702 | 9.93 ± 3.61% | 0.0200 |
| Buy & Hold | 1.1506 | 1.852 | 5.64% | 0.0107 |
| UCRP | 1.1532 | 1.793 | 6.04% | 0.0226 |
| Best stock (hindsight, XOM) | 1.3975 | 1.811 | 20.11% | 0.0107 |
| All-cash | 1.0000 | n/a | 0.00% | 0.0000 |

**Findings for the write-up (state these honestly)**
- **No agent beats the passive benchmarks on risk-adjusted return.** All three land at or below
  UCRP/Buy-and-Hold on fAPV while carrying ~2-4x the drawdown, so every Sharpe is well below the
  benchmarks' ~1.8. The agents earn a similar return by taking materially more risk.
- **Validation ranking did not survive the test window.** Val said A2C (1.228) > PPO (1.185) >
  DDPG (1.100); test says PPO ≈ DDPG > A2C. A2C's advantage was validation-window-specific — a
  clean, honest illustration of why the held-out split exists.
- **Seed variance dominates the algorithm gaps.** A2C seed1 collapses to 0.8293 (Sharpe −0.795)
  while its other four seeds sit tightly at ~1.17; DDPG spans 1.0899-1.2439. PPO is by far the most
  stable (1.1407-1.1686), consistent with its tight validation spread. Single-seed numbers here
  would have been meaningless — the 5-seed protocol earned its keep.
- **The learned policies are near-static.** The allocation heatmaps show essentially constant
  weights across all 187 steps: the agents converge to a fixed allocation and rebalance to it daily
  (turnover ~0.01-0.02/step) rather than timing the market. A2C's median seed goes ~100% NVDA;
  PPO's holds ~60% NVDA plus a spread. This is the most defensible single observation in the
  chapter: with this observation space and reward, the agents learn *an allocation*, not a
  *strategy*.

**Verification**
- Benchmark accounting checked against closed form at c = 0: env equal-weight buy-and-hold
  = 1.1534554318 vs analytic mean of per-stock cumulative relatives = 1.1534554318 (diff 1.7e-13);
  best-stock = analytic max (diff 2.3e-12); post-open turnover ≤ 7e-13; μ ≡ 1 throughout.
- Cost sanity: buy-and-hold at c = 0.25% (1.1506) = costless value × (1 − 0.0025) — i.e. exactly
  the one opening commission, as expected for a strategy that never trades again.
- Turnover cross-check: buy-and-hold and best-stock both report 0.0107 = 2/187, the single opening
  trade amortized over the window. All-cash reports exactly 0.
- Fixed during the build: all-cash Sharpe initially printed **1.751**. The env's softmax cannot emit
  an exact zero weight, so ~1e-12 leaks into the stocks; dividing that noise by its own near-zero
  std produced a meaningless O(1) figure. Sharpe is now reported as undefined (n/a) for any path
  whose returns are numerically zero.

**Repo note:** `.gitignore` previously excluded `results/**/*.png` and `*.csv`, which would have
dropped the evaluation figures and results table — the actual thesis deliverables. Added negations for
`results/data_checks/` and `results/evaluation/`; model `.zip` checkpoints stay ignored (large, regenerable).

**Next:** documentation — `docs/METHODOLOGY.md` (code ↔ Jiang equation map, algorithm configs, seeds,
splits, the PG→A2C mapping, honest-limitations note) + one-command reproduction in the README.
Awaiting go-ahead.

---

## 2026-07-30 — Methodology doc + one-command reproduction

**What was built**
- `docs/METHODOLOGY.md` — the thesis-chapter reference: Jiang formalism table, code ↔ equation
  map, data/environment/algorithm/training/evaluation specs, results, deviations, limitations,
  reproduction, references, open supervisor items.
- `experiments/run_all.py` — one-command end-to-end reproduction: data → env tests → training
  (3 algos × 5 seeds @ 300k) → evaluation. `--skip-train` reproduces the reported tables and
  figures from the existing checkpoints in ~1 min; `--steps 2000` is a fast wiring check. Stages
  stream their output and the pipeline halts on the first non-zero exit code.
- `README.md` — reproduction section, results table, updated layout and status.

**Equation numbers VERIFIED against the source paper (not taken on trust)**
- The repo PDF is **arXiv:1706.10059v1** (30 Jun 2017). Text was extracted with a throwaway
  stdlib PDF reader (no new packages installed into the venv) and every equation number the
  codebase cites was checked against it. All confirmed: **Eq. 1** (price relative), **Eq. 5**
  (`w_0` all-cash start), **Eq. 6** (costless final value — what test 2 asserts), **Eq. 7**
  (drift), **Eq. 9** (ρ_t), **Eq. 10** (log reward), **Eq. 11** (wealth with cost), **Eq. 14–16**
  (μ implicit / fixed-point sequence / initial guess), **Eq. 18** (price tensor), **§2.4**
  (hypotheses), **§3.1** (survival bias), **§3.2** (price tensor), **§6.2 Eq. 27/28/29**
  (fAPV / Sharpe / MDD). The codebase had the numbering right throughout.
- **Caveat recorded in the doc:** later arXiv versions renumber. The thesis should cite **v1**
  explicitly, or re-check against whichever version it cites.
- Also confirmed from the paper: **c = 0.25% is Jiang's own default** (v1 §2.3, the maximum rate
  at Poloniex), so the commission choice is the paper's, not an arbitrary pick. And Jiang's
  benchmark set (UBAH / UCRP / Best Stock) is the one `evaluate.py` implements.

**Two substantive findings from the verification**
1. **Sharpe is annualized here; Jiang's Eq. 28 is not.** Jiang reports per-period Sharpe on
   30-minute bars, which is why their table shows values like 0.087. Annualizing (×√252) is the
   right convention for daily equity data, but it means **this thesis's Sharpe figures are not
   directly comparable to Jiang's table.** Recorded as deviation 2 — worth a sentence in the
   chapter to pre-empt the question.
2. **Feature channel order differs.** Eq. 18 stacks `(V^(lo), V^(hi), V)`; `data_loader.FEATURES`
   is `(High, Low, Close)`. A channel permutation is functionally irrelevant to the networks, but
   it differs from the equation as literally written. Recorded as deviation 6. No code change —
   changing it now would invalidate the trained checkpoints for no benefit.
- Also noted: v1's Theorem 1 states `f(μ)` with `(w'_i − w_i)⁺`, omitting the `μ` that Eq. 14
  carries. The iteration is only a fixed point with `μ` present, so the implementation follows
  Eq. 14. Documented so a reader comparing code to Theorem 1 is not confused.

**Citation problem found (needs the student's action)**
- The plan anchors the three-algorithm comparison on **Liang et al. (2018)**, but that paper is
  **not in the repo**. The PDF filed as `1909.09571 … reading.pdf` is a *different work*:
  **Filos, A. (2018), "Reinforcement Learning for Portfolio Management", MEng dissertation,
  Imperial College London** (arXiv:1909.09571). Verified by extracting its title page.
  The Liang paper (commonly cited as arXiv:1808.09940, "Adversarial Deep Reinforcement Learning
  in Portfolio Management") must be obtained and the citation checked before submission.
  Logged in METHODOLOGY.md §12 and §13.

**Verification**
- `python experiments/run_all.py --skip-train` → all 3 stages green (data OK, 7/7 tests pass,
  evaluation reproduced the reported numbers identically), 0.1 min total.
- Console output kept ASCII (`+/-`, `-` not `—`): the Windows terminal codepage mangles those
  glyphs. Markdown files keep proper Unicode (written UTF-8).

**Repo note:** `requirements.txt` went missing mid-session and was restored by the student; it was
never committed, so git could not have recovered it. Everything else in the tree is likewise
uncommitted — **still nothing pushed, per the standing instruction.** A local commit remains the
single cheapest risk reduction available.

**Status: all workstreams complete.** Definition of done: (1) env tests pass ✓ (2) 3 algos × 5
seeds trained ✓ (3) test-window table + figures from one script ✓ (4) METHODOLOGY.md complete ✓
(5) repo reproduces end-to-end from one command ✓.

**Open items for the supervisor:** data source (yfinance vs the recommended Microsoft source) ·
the Liang et al. citation · explicit sign-off on the PG→A2C substitution.

---

## 2026-07-30 — Liang et al. (2018) obtained and cross-checked (citation item CLOSED)

**Paper.** Student downloaded `1808.09940v3.pdf`. Verified by text extraction:
**Liang, Z., Chen, H., Zhu, J., Jiang, K. & Li, Y. (2018), "Adversarial Deep Reinforcement
Learning in Portfolio Management", arXiv:1808.09940v3 [q-fin.PM], 18 Nov 2018**, Likelihood
Technology / Sun Yat-sen University. It does implement DDPG, PPO and PG on portfolio management,
so the plan's anchor was substantively right all along — only the PDF in the repo was the wrong
work. **METHODOLOGY.md §12 updated with the full verified citation; open item closed.** The Filos
dissertation is now listed separately with an explicit "do not cite as Liang" note.

**Cross-check against our test-window findings — added as METHODOLOGY.md §9.1.** Their conclusions
corroborate three of our four findings, on a completely different market (China A-shares):
- *"reinforcement learning does not gain such remarkable performance in portfolio management so
  far as those in game playing or robot control"* → our finding 1 (no agent beats the passive
  benchmarks on risk-adjusted return).
- *"deep reinforcement learning is highly sensitive so that its performance is unstable"* → our
  finding 3 (seed variance dominates the algorithm gaps).
- *"the degeneration of our reinforcement learning agent, which often tends to buy only one asset
  at a time"* → our finding 4 (near-static, concentrated allocations; A2C's median seed ~100%
  NVDA). **Two independent implementations, two different markets, the same failure mode.** This
  is the strongest external support the chapter has for its central observation — lead with it.
- Their PG beats UCRP only *after* their Adversarial Training modification. We implement no such
  modification and no agent beats UCRP — consistent, not contradictory.

**The one disagreement.** Their headline is *"PG is more desirable in financial market than DDPG
and PPO"*; our test window gives the opposite order (PPO ≈ DDPG > A2C/PG). Three caveats recorded
so the write-up does not overclaim a refutation: (1) **our PG is A2C, and the gap is far wider
than "A2C vs REINFORCE"** — see below; (2) different markets; (3) different budgets/objectives
(they tune across optimizers, objectives and feature sets; we fix an equal 300k budget with a
log-return reward).

**What Liang et al.'s "PG" actually is (checked, not assumed).** Their network-structure section
states *"Motivated by Jiang et al., we use so called Identical Independent Evaluators (IIE)"* —
independent per-asset network flows with **shared parameters**, each emitting a scalar preference,
the m+1 scalars then **softmax-normalized into the weight vector**. They replace Jiang's CNN with
a **deep residual network**, and for PG specifically: *"we adapt similar settings with Jiang's and
we would not go specific about them here"* — their hyperparameter table (Table I) has rows for
DDPG and PPO only, **no PG row**. So their PG is **Jiang's EIIE-style architecture trained by
direct policy gradient**, not a generic REINFORCE.

**Why that matters for the write-up.** Our "PG" is stock SB3 A2C on `MultiInputPolicy` with
default networks; EIIE was a labelled stretch goal we deliberately did not implement (deviation 8).
So the head-to-head with Liang is *general-purpose deep RL algorithm* vs *domain-specialized
architecture*. Their "PG beats DDPG and PPO" may therefore be reporting **the architecture's
advantage rather than the algorithm family's** — PG is the one agent of their three carrying
Jiang's design. This is a stronger and more interesting reading of the disagreement than "we got
the opposite ranking", and it also converts the EIIE stretch goal from an unfinished item into a
clearly motivated *further work* item. METHODOLOGY.md §6 and §9.1 updated accordingly.

**Docs updated:** METHODOLOGY.md §6 (PG→A2C note now flags why the substitution is load-bearing),
§9.1 (new comparison section), §12 (verified references), §13 (citation item closed; two open
items remain).

---

## 2026-07-30 — Supervisor-facing README + three-period reporting + POLICY DEGENERACY FOUND

**Trigger:** rewrite README.md as a document to send the supervisor, covering all three periods.
Producing the train/validation tables required new artifacts, and building them surfaced the most
important finding in the project.

**THE FINDING: all 15 trained policies are constant functions that ignore the market.**
- Built `experiments/policy_diagnostic.py`. It runs each checkpoint's deterministic policy over a
  window and measures how far the emitted weights move. Result over the 187-day test window:
  **PPO 4.1e-07–1.3e-06 · A2C 2.7e-08–4.1e-07 · DDPG exactly 0.0.** A trading policy moves weights
  by O(0.1); these move by float32 rounding error. **15/15 CONSTANT.**
- Survives a harder control: querying each policy on observations from the start/middle/end of
  every split — windows sharing no data — gives the same weights, with `w_prev` pinned so any
  movement would be attributable to `X_t` alone.
- **DDPG is the extreme case.** Its actor is saturated at the action bounds (every component
  exactly ±10), `best` and `final` checkpoints are identical, and its validation score is
  **bit-identical across all 20 evaluations** from 15k to 300k. It froze before the first eval.
  The ±10 bounds were chosen so softmax could reach concentrated allocations; that choice appears
  to have cost DDPG the ability to learn at all.
- **This reframes the whole results chapter.** The reported fAPV differences are differences
  between *constant portfolios*, not between trading strategies. Agent turnover (~0.02/step) is
  drift correction, which is why it nearly matches UCRP's 0.0226.
- Quantified two plausible causes (hypotheses, not tested): inputs sit at **0.988 ± 0.092** after
  Eq. 18 normalization — near-constant and not zero-centred, with no `VecNormalize`; and daily
  log-return **SNR ≈ 0.035** (mean 6.7e-04 vs std 0.0191), so the gradient is ~97% noise. A fixed
  diversified allocation may simply be the rational local optimum at that SNR.
- Written up as **METHODOLOGY.md §9.2**; finding 4 in §9 upgraded from "near-static" to "constant".

**Three-period reporting (new)**
- `evaluate.py` gained `--split {train,val,test}` (default test, backward compatible). Train and
  each split writes to `results/evaluation/<split>/`. Each split's RESULTS.md carries its own framing —
  train/val are labelled **in-sample diagnostics, not headline results**, because a reader opening
  those files directly has no surrounding context.
- **Consistency check passed:** the val pass independently reproduces training's recorded
  best-on-validation numbers (PPO 1.1852 vs 1.185, A2C 1.2280 vs 1.228, DDPG 1.0998 vs 1.100).
- **Train→val→test progression (annualized return):** PPO 47.1→25.7→20.8% · A2C 66.5→31.9→14.6% ·
  DDPG 22.5→13.7→20.1% · best benchmark 26.5→19.0→**21.2%**. Agents beat the benchmarks on both
  in-sample windows and lose out of sample. A2C degrades most and concentrated hardest (~100%
  NVDA, the best stock of the *training* window, cumulative relative 6.80) — i.e. the optimizer
  selected the historically best fixed allocation, which did not persist.

**Convergence: PPO is the algorithm that had not plateaued**
- Built `experiments/plot_training.py` — reads the per-seed `evaluations.npz` curves, plots
  validation fAPV vs timesteps (mean ± std over seeds), and prints a numeric check: least-squares
  slope over the final third, in fAPV per 100k steps.
- **PPO +0.0321/100k (STILL RISING) · A2C +0.0035 (plateaued by ~45k) · DDPG −0.0000 (frozen).**
  This confirms the earlier training observation numerically and independently of the Colab 500k run.
- Figure: `results/evaluation/training_curves.png`.

**Other changes**
- `run_all.py` gained a 5th stage so the README's reproduction claim is true end to end: it now
  runs test evaluation, then val + train evaluations, training curves and the policy diagnostic.
  Verified: `--skip-train` completes all 5 stages green in 0.5 min.
- `README.md` rewritten as a supervisor-facing report: headline finding, setup, data + the
  yfinance rationale, design, all three result tables, the progression table, figures, convergence,
  conclusions, recommended next steps, a full repo map, and open decisions.
- `.gitignore`: a `dir/*.png` negation does not reach one level down, so the
  new train/ and val/ artifacts were still ignored. Added per-level negations. Verified
  all deliverables tracked, model `.zip` still ignored.
- Console output fixes in `evaluate.py`: the results header said "Test-window" regardless of split,
  and the MDD column overflowed into Sharpe on wide train values.

**Verification:** `run_all.py --skip-train` → 5/5 stages green. 7/7 unit tests. All 21 paths
referenced by the README confirmed to exist.

**Still nothing pushed** — per the standing instruction. The repo remains uncommitted.

---

## 2026-07-30 — Cleanup pass: phase terminology removed, results/ renamed

Presentation cleanup before sending the repo to the supervisor. No results, metrics or
methodology changed — only names and wording.

**Terminology.** Every "Phase N" reference removed from the codebase and documents (77 across
16 files): module docstrings, console output, CLI help, section headings in
IMPLEMENTATION_PLAN.md, and this log's own entry titles, which are now dated and descriptive.
Internal project phases meant nothing to an outside reader.

**`results/` renamed to describe contents rather than project stages:**

| Before | After |
|---|---|
| `results/phase1/` | `results/data_checks/` |
| `results/phase4/*` (test files at top level) | `results/evaluation/test/` |
| `results/phase4/{train,val}/` | `results/evaluation/{train,val}/` |
| `results/phase4/04_training_curves.png` | `results/evaluation/training_curves.png` |

The split layout is now symmetric — `test/`, `val/` and `train/` are siblings with identical
contents, instead of test sitting at the top level with the diagnostics nested beneath it.
`evaluate.py --out-dir` now defaults to `results/evaluation/<split>`, so `run_all.py` no longer
passes the path explicitly.

**Generated report titles** changed with the TITLE table: "Held-out test-window results",
"Validation-window results (in-sample diagnostic)", "Training-window results (in-sample
diagnostic)". The in-sample warnings on the train/val pages are unchanged.

**`.gitignore`** negations rewritten for the new paths, with a note that a `dir/*.png` negation
does not reach one level down — the trap that already caught the train/ and val/ artifacts once.

**Verification:** all results deleted and regenerated from scratch under the new layout —
`run_all.py --skip-train` green on all 5 stages, 7/7 unit tests pass, all 24 README paths and 5
METHODOLOGY links resolve, and no "phase" string remains in any tracked file except
AGENT_PROMPT.md.

**AGENT_PROMPT.md left untouched — flagged for the student.** It is the internal prompt used to
drive this work; its whole structure is the phase workflow, so scrubbing the word would leave it
incoherent. More to the point, it is not a deliverable and discloses how the work was produced.
Whether it is sent, scrubbed or deleted is the student's call, not a cleanup decision.

---

## 2026-07-30 — Architecture trace: the price tensor is flattened (no CNN)

**Question asked:** where does the price tensor actually get fed to the agent? Tracing it end to
end turned up a fourth, and probably primary, cause of the policy degeneracy.

**The path.** `data_loader.price_tensor` (Eq. 18) → `Dataset.tensor(t)` →
`PortfolioEnv._obs()` puts it in the observation dict under `"X"` → declared in
`observation_space` as `Box(0, ∞, (3, 50, 8))` → returned by `reset()`/`step()` →
`train.py` sets `policy="MultiInputPolicy"`.

**What SB3 does with it — verified against the saved checkpoints, not the docs.**
`MultiInputPolicy` builds a `CombinedExtractor`, which assigns a CNN **only to image spaces**
and `nn.Flatten()` to everything else. `(3, 50, 8)` is not an image space, so:

```
'X'      -> Flatten(start_dim=1, end_dim=-1)     # 3 x 50 x 8 = 1200
'w_prev' -> Flatten(start_dim=1, end_dim=-1)     # 9
features_dim: 1209
policy net: Linear(1209, 64) -> Tanh -> Linear(64, 64) -> Tanh
```

**There is no convolution in any trained model.** The temporal ordering of the 50 days, the
identity of the 8 assets and the meaning of the 3 channels are all discarded at the flatten;
the network has to learn 1209 unrelated input weights. This is exactly what Jiang's EIIE
topology prevents — per-asset flows with *shared* parameters learn "how to evaluate an asset"
once instead of eight times. The plan listed a CNN extractor as optional "if time"; the
diagnostic suggests it was not optional.

**Why it matters.** It converts the EIIE recommendation from "the literature suggests it" into
"our own architecture trace and the literature point at the same fix from opposite directions",
and it is now the **first** recommended next step, ahead of observation normalization.

**Docs updated:** METHODOLOGY.md §9.2 (new cause 1, list renumbered to five; discriminating
experiments reordered), README §7 (plausibility list and next steps reordered).

---

## 2026-07-30 — Allocation heatmap: exact weights labelled

Colour alone reads poorly for exact magnitudes, and since the policies are constant each row
collapses to a single number. Added a right-hand axis to every heatmap panel printing that
number, with `fig.colorbar(..., pad=0.07)` so the labels clear the colourbar. If a weight ever
does move, the label appends its peak-to-trough range (`±x.x`) rather than silently showing only
the mean — so the figure stays honest for a non-degenerate future run.

The test-window portfolios now read directly off the figure:
**PPO** 57.0% NVDA · 10.4% cash · 2.8-5.8% each of the rest ·
**A2C** 99.3% NVDA, everything else ≈0 ·
**DDPG** exactly 20.0% in each of AAPL, MSFT, XOM, AMZN, NVDA, 0.0% elsewhere including cash.

DDPG's "exactly 20.0% × 5" is the saturated actor made visible: it is what softmax returns for
five `+10`s and four `−10`s. All three windows regenerated.
