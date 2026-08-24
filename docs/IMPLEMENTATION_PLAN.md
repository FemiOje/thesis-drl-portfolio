# Implementation Plan

Comparative evaluation of **PG, PPO and DDPG** for S&P 500 portfolio management.

Strategy, justification and paper citations live in the roadmap artifact. This document is
the engineering contract: what gets built, in what order, with what invariants, and what
each phase must produce before the next begins.

Equation numbers refer to **arXiv:1706.10059v1** (Jiang, Xu & Liang, 30 Jun 2017).
Later arXiv versions renumber — cite v1 explicitly.

---

## 0. Locked decisions

Nothing below is open. Changing any of these invalidates results already produced.

| Parameter | Value | Source |
|---|---|---|
| Data window | 2021-09-01 → 2026-08-31 (~1,260 trading days) | decided |
| Train | 2021-09-01 → 2024-08-31 (~756 d) | 60% |
| Validate | 2024-09-01 → 2025-08-31 (~252 d) | 20% |
| Test | 2025-09-01 → 2026-08-31 (~252 d) | 20%, scored once |
| Universe | 8 risky + cash = **9-dim** weight vector | Eq. 5 reserves index 0 for cash |
| Lookback `n` | 20 | one trading month |
| Features `f` | 3 — close, high, low, each ÷ latest close | Eq. 18 |
| Observation tensor | `(3, 8, 20)` | Eq. 18 |
| State | `(X_t, w_{t-1})` | Eq. 20 |
| Action | `w_t`, simplex | Eq. 19 |
| Reward | `ln(μ_t · y_t · w_{t-1})` | Eq. 10 |
| Cost factor `μ_t` | fixed-point, k=5, warm start Eq. 16 | Eq. 14–15, Thm 1 |
| Commission `c` | 0.001 (10 bps), `c_s = c_p` | headline; swept |
| Discount `γ` | **1.0 for all three agents** | Eq. 21; prevents horizon confound |
| Softmax temperature `tau` | **5.0** | see §2 — a lower value silently caps allocations |
| Data source | yfinance primary, Qlib cross-check | see §3 |
| Seeds | 10 | primary variance control |
| Risk-free rate | `^IRX` 13-week T-bill, or 0 with stated assumption | see §7 |
| Sharpe | annualised, `× √252` | §3.5.2 currently omits this |

### Universe selection rule

> Pre-specify eight GICS sectors spanning distinct economic drivers; within each, take the
> largest S&P 500 constituent by market capitalisation **as of 2021-09-01**, excluding
> diversified holding companies, subject to complete price history over the window.

| Sector | Indicative | Driver |
|---|---|---|
| Information Technology | AAPL / MSFT | long-duration growth |
| Financials | JPM | rate-sensitive, positive |
| Utilities | NEE | rate-sensitive, negative |
| Energy | XOM | commodity / inflation |
| Health Care | JNJ / UNH | defensive |
| Consumer Staples | PG / WMT | defensive, low beta |
| Consumer Discretionary | AMZN → HD | cyclical consumer |
| Industrials | HON / UNP / UPS | cyclical capex |

**Pre-registered tie-break** (write into §3.2 *before* running anything): if any pairwise
correlation of simple returns, measured on the **training split only**, exceeds `0.70`,
substitute the next-largest constituent in the affected sector. Communication Services,
Materials and Real Estate are excluded on driver-duplication grounds.

---

## 1. Repository layout

```
thesis-drl-portfolio/
├── config/
│   ├── base.yaml              # window, splits, n, cost, gamma, seeds, tau
│   └── universe.yaml          # sector list, resolved tickers, selection provenance
├── data/
│   ├── raw/                   # per-ticker CSVs + MANIFEST.json — COMMITTED
│   └── processed/             # tensors .npz — gitignored, rebuildable
├── src/
│   ├── config.py              # frozen dataclass, loaded once, passed down
│   ├── universe.py            # selection rule + correlation gate
│   ├── data.py                # download, cache, tensor build, split by decision date
│   ├── costs.py               # mu_t — numpy AND torch, must agree
│   ├── env.py                 # PortfolioEnv(gymnasium.Env)
│   ├── extractors.py          # EIIEExtractor — shared by all three agents
│   ├── agents/
│   │   ├── pg.py              # hand-written deterministic PG + PVM
│   │   └── sb3.py             # PPO / DDPG factories
│   ├── baselines.py           # UBAH, UCRP, BestStock, rolling Markowitz
│   ├── backtest.py            # single rollout loop — agents AND baselines
│   ├── metrics.py             # CR, AR, Sharpe, MDD, turnover, win rate, HHI
│   ├── stats.py               # bootstrap CI, paired t-test, Bonferroni
│   └── plots.py               # one style module — F0–F17, PNG + PDF
├── scripts/
│   ├── 01_build_data.py   02_run_baselines.py   03_train.py
│   ├── 04_evaluate.py     05_sweeps.py          06_figures.py
├── tests/
│   ├── test_costs.py   test_env.py   test_extractor.py   test_metrics.py
├── results/<run_id>/          # config.json, per-seed CSVs, figures/
├── docs/IMPLEMENTATION_PLAN.md
├── requirements.txt
└── README.md
```

`requirements.txt` pins exactly, per §3.3: Python 3.10, `gymnasium==0.29.1`,
`stable-baselines3==2.2.1`, `torch==2.1.0`, `numpy==1.24.3`, `pandas==2.0.3`,
`matplotlib==3.7.2`, `seaborn==0.12.2`, `yfinance` (version recorded in MANIFEST).

---

## 2. Module contracts

### `costs.py` — the piece most likely to be silently wrong

```python
def transaction_remainder(w_drift, w_target, c, k=5, backend="numpy"):
    """Solve Eq. 14 by the Eq. 15 iteration, warm-started at Eq. 16.

    w_drift  : w'_t, post-drift weights (Eq. 7).  (..., m+1)
    w_target : w_t,  agent's target weights.      (..., m+1)
    returns  : mu in (0, 1], shape (...,)
    """
```

Two implementations — NumPy for the env, Torch for PG's loss — and they **must** agree.
Unroll the `k` iterations explicitly so the Torch version stays differentiable; PG's
gradient flows through `μ_t`.

Required tests (`test_costs.py`):

- `c = 0` → `mu == 1.0` exactly
- `w_target == w_drift` → `mu == 1.0` (no trade, no cost)
- `0 < mu <= 1` over 10,000 random simplex pairs
- `mu` monotonically decreasing in turnover `Σ|w' − w|`
- NumPy and Torch agree to `1e-6`
- hand-computed two-asset case matches a worked example in the docstring
- `mu.requires_grad` is `True` when `w_target.requires_grad` is `True`

### `env.py`

```python
observation_space = Dict({
    "tensor":  Box(0.0, np.inf, (3, 8, 20), float32),   # Eq. 18
    "weights": Box(0.0, 1.0,    (9,),       float32),   # w_{t-1}, Eq. 20
})
action_space = Box(-1.0, 1.0, (9,), float32)            # raw scores, NOT weights
```

The env owns the simplex projection: `w = softmax(tau * action)`.

> **Gotcha that silently caps the portfolio.** SB3's DDPG squashes its actor output through
> `tanh` and rescales to the action-space bounds. With bounds `[-1, 1]` and `tau = 1`, the
> most extreme achievable allocation is `e / (e + 8/e) ≈ 0.48` — the agent becomes
> structurally incapable of holding more than ~48% in any asset, and nothing warns you.
> Every result would be quietly wrong.

`tau` is therefore a first-class, locked configuration value, not a magic number:

- Declared in `config/base.yaml` as `tau: 5.0`; never hard-coded anywhere.
- Reachability at `tau = 5.0`: `e⁵ / (e⁵ + 8·e⁻⁵) ≈ 0.9996` — the full simplex is available.
- **Applied identically in three places** — the env's projection, PG's actor head, and the
  reachability test. Import one `project_to_simplex(action, tau)` from `env.py`; do not
  reimplement it in `pg.py`.
- PG's actor also passes its logits through `tanh` before the same `softmax(tau · ·)`, so
  all three agents share one bounded, identical projection. Without the `tanh`, PG's logits
  are unbounded and it alone could reach allocations the other two cannot.
- Tuned on **validation only**, over `tau ∈ {2, 5, 10}`, and the selected value reported in
  §3.4 alongside `n` and `γ`.

Required assertions in `test_env.py`:

```python
w_max = project_to_simplex(np.array([1., -1., ..., -1.]), tau=cfg.tau)[0]
assert w_max > 0.95, f"tau={cfg.tau} caps max allocation at {w_max:.3f}"
assert abs(project_to_simplex(a, tau).sum() - 1.0) < 1e-9
assert (project_to_simplex(a, tau) >= 0).all()
```

Log the realised maximum single-asset weight per run. If it clusters suspiciously near a
ceiling across every seed and every algorithm, `tau` is the first suspect.

Because SB3 sees a `Dict` observation, both agents must use `policy="MultiInputPolicy"`.

`step()` order, per Eq. 7 → 14 → 10:
1. `w_target = softmax(tau * action)`
2. `w_drift = (y_t ⊙ w_prev) / (y_t · w_prev)` — Eq. 7
3. `mu = transaction_remainder(w_drift, w_target, c)` — Eq. 14
4. `reward = log(mu * (y_t · w_prev))` — Eq. 10
5. `info = {"weights": w_target, "mu": mu, "turnover": ..., "hhi": ...}`

**`info["weights"]` is the only correct source of allocations.** SB3 logs the pre-softmax
action; allocation heatmaps and turnover built from raw actions look entirely plausible and
mean nothing.

`reset()` sets `w_0 = (1, 0, …, 0)` — all capital in cash (Eq. 5).

Tests: constant-weight policy reproduces UCRP computed independently in pandas to `1e-10`;
weights always sum to 1 and stay non-negative; episode length equals split length.

### `extractors.py` — EIIE, shared by all three agents

Convolution kernels of height 1 so asset rows never mix before the softmax, with parameters
shared across rows:

```
input  (B, 3, 8, 20)                    # (batch, features, assets, time)
Conv2d(3 → 2,  kernel=(1,3))  → ReLU    # (B,  2, 8, 18)
Conv2d(2 → 20, kernel=(1,18)) → ReLU    # (B, 20, 8,  1)
concat w_{t-1} as a 21st channel        # (B, 21, 8,  1)
Conv2d(21 → 1, kernel=(1,1))            # (B,  1, 8,  1)
squeeze → append cash bias scalar       # (B, 9)  — logits, NOT weights
```

Every kernel has height 1. The asset axis is never convolved over, never flattened, never
reduced — it passes through as an independent axis until the softmax. That is the whole
point of EIIE, and it is the controlled variable: **all three agents use this identical
class.** Any architectural difference between agents voids the comparison.

#### Proving it does not flatten or ignore the tensor

Four tests, two of which are decisive on their own (`test_extractor.py`):

1. **Parameter-count invariance — decisive.**
   `n_params(EIIEExtractor(n_assets=4)) == n_params(n_assets=8) == n_params(n_assets=16)`.
   Parameter count must be *exactly* independent of the asset count. If it varies, some
   layer spans the asset axis — i.e. it is flattening — and the test fails loudly.

2. **Permutation equivariance — decisive.**
   For any permutation `π` of the eight asset rows,
   `f(π(X), π(w_prev)) == π(f(X, w_prev))`, with the cash element fixed.
   This proves parameters are genuinely shared across rows and that the network cannot
   memorise asset identity by position. A flattened network fails this immediately.

3. **Time sensitivity.** Perturb `X` at each time index `t ∈ [0, 20)` in turn; the output
   must change for *every* `t`. Catches the failure where the window is accepted but
   collapsed — e.g. only the last column reaching the head.

4. **Runtime shape assertion.** Inside `forward()`, assert
   `obs["tensor"].shape[1:] == (3, n_assets, window)` before the first convolution. Cheap,
   and it fires immediately if the observation is ever pre-flattened upstream.

Log `n_params` and the resolved `features_dim` into each run's `config.json` so the numbers
appear in Chapter 4 rather than being asserted from memory.

### `agents/pg.py` — the centrepiece, hand-written

Deterministic actor, no critic, no sampling, no log-prob term. Gradient flows straight
through the reward.

```python
w      = softmax(tau * actor(X_batch, w_prev))   # same tau, same softmax as env
mu     = transaction_remainder(w_prev_drift, w, c, backend="torch")
reward = torch.log(mu * (w_prev * y).sum(-1))    # Eq. 10
loss   = -reward.mean()                          # Eq. 21, negated
loss.backward(); opt.step()
```

**Portfolio-Vector Memory** (§5.2): array `(T, 9)` initialised uniform; each step reads
`w[t-1]` and writes `w[t]`. This is what permits mini-batch training without chaining
gradients across the batch. Sample batch start indices uniformly over the training split;
Jiang's geometric OSBL sampling (Eq. 26) is for online learning and is optional here.

Starting point: batch 50, Adam `lr=3e-5` (Jiang Table B.1), L2 `1e-8`, ~10,000 gradient
steps, early stop on validation.

### `agents/sb3.py`

```python
policy_kwargs = dict(
    features_extractor_class=EIIEExtractor,
    features_extractor_kwargs=dict(n_assets=8, window=20, n_features=3),
    net_arch=[],                      # PPO: extractor does the work, heads stay thin
)
PPO("MultiInputPolicy", env, gamma=1.0, seed=s, policy_kwargs=policy_kwargs, ...)
DDPG("MultiInputPolicy", env, gamma=1.0, seed=s,
     policy_kwargs={**policy_kwargs, "net_arch": dict(pi=[], qf=[64, 64])},
     action_noise=OrnsteinUhlenbeckActionNoise(...), ...)
```

`gamma=1.0` is not optional — SB3 defaults to `0.99`, which would confound the learning
rule with the planning horizon against PG's undiscounted objective.

#### The flattening trap, and the assertion that catches it

With a `Dict` observation space, SB3's default extractor is `CombinedExtractor`, which
**flattens every Box sub-space and concatenates them**. If `features_extractor_class` is
ever dropped, misspelled, or silently overridden, the `(3, 8, 20)` tensor becomes a
480-vector, the EIIE structure vanishes, and training still runs — producing entirely
plausible, entirely meaningless results. Nothing warns you.

Assert it immediately after constructing every model, in `sb3.py`, not in a test:

```python
fe = model.policy.features_extractor
assert isinstance(fe, EIIEExtractor), f"extractor was replaced by {type(fe).__name__}"
assert not isinstance(fe, CombinedExtractor)
```

For DDPG also check the critic's extractor, since it may be a separate instance (below).

#### Two unavoidable asymmetries to document, not hide

- **Shared vs separate extractors.** PPO's policy defaults to
  `share_features_extractor=True` — one extractor feeding both actor and critic. SB3's
  TD3/DDPG policies default to `False`, giving actor and critic separate copies. Set this
  explicitly rather than inheriting it, record the choice in `config.json`, and state it in
  §3.4. It is not fatal, but an unrecorded default is indefensible under questioning.
- **Critic capacity.** DDPG's critic must fuse features *and* the action, so `net_arch`
  for `qf` needs real capacity while `pi` stays empty. PG has no critic at all. This
  asymmetry is inherent to the algorithms being compared — which is the point of the study
  — so report it in the architecture table rather than pretending the three are identical.

Starting budgets: PPO 200k–500k env steps, DDPG 100k–200k. Tune on validation.

### `baselines.py` and `backtest.py`

Every baseline is a `policy_fn(obs) -> action` consumed by **the same** `backtest()` loop as
the agents, so all strategies pay identical transaction costs through identical machinery.
A baseline evaluated outside the env is not comparable to one inside it.

- **UBAH** — buy equal weights at t=0, never rebalance
- **UCRP** — rebalance to equal weights every step
- **Best Stock** — single best asset in hindsight (upper reference, not a strategy)
- **Markowitz** — rolling 252-day covariance, long-only max-Sharpe, monthly rebalance

### `metrics.py` / `stats.py`

CR, AR, annualised Sharpe (`× √252`), MDD (Eq. 29), turnover (§3.5.2 Eq. 13), win rate,
plus **concentration** — Herfindahl index and entropy of `w`, logged every episode.
Liang et al. report agents *"degenerat[ing] … to buy only one asset at a time"*; if DDPG
collapses, that is a reportable finding, but only if it was measured.

Stats: paired t-test on daily returns vs each baseline, Bonferroni-corrected, **plus**
bootstrap confidence intervals — 252 test days makes point estimates fragile.

---

## 3. Data source: yfinance or Microsoft Qlib

A finding that changes how to use the choice: **Qlib's US dataset is not an independent
source.** It is built by `scripts/data_collector/yahoo/collector.py` and lands in
`~/.qlib/qlib_data/us_data` — the same Yahoo Finance numbers, normalised and re-stored in
Qlib's binary format. Choosing Qlib over yfinance changes the pipeline, not the data.

So do not treat them as alternatives. Use them as primary and control:

| | Role | Why |
|---|---|---|
| **yfinance** | Primary | Matches the §3.3 pin; eight tickers need no heavyweight framework; raw CSVs are committed so reproducibility does not depend on a live API |
| **Qlib** | Optional cross-check | Independent *adjustment and calendar* logic over the same raw source — catches yfinance split/dividend-adjustment bugs, which are a real and recurring failure mode |

**Recommendation:** build on yfinance. If the cross-check is cheap to run, do it once in
Phase 1 and record the outcome; if Qlib's install proves troublesome, skip it — it is a
control, not a dependency, and nothing downstream requires it.

Be precise in §3.2 about what the cross-check does and does not establish: agreement
between yfinance and Qlib confirms that *adjustment handling* is consistent. It cannot
detect an error in Yahoo's underlying data, because both trace to the same origin. Claiming
otherwise would overstate it.

Record in `data/raw/MANIFEST.json`: tickers, retrieval date, yfinance version,
`auto_adjust` setting, row counts per ticker, and the Qlib cross-check result if run.

---

## 4. Figure suite

Every figure is produced by `scripts/06_figures.py` from committed run directories, so
Chapter 4 is regenerable end-to-end. Numbering below is the artefact numbering, mapped to
where each lands in the thesis. Liang et al. and the deepcrypto fork between them report
training curves, wealth curves, backtest comparisons and a feature/APV bar chart; this
suite covers those and adds the diagnostics a single-split design needs.

### Data checks — Phase 1, thesis §3.2

- **F0a** Normalised price series, all eight assets, split boundaries marked.
- **F0b** Correlation matrix heatmap of training-split simple returns, with the 0.70 gate
  threshold annotated. This is the L2 diversification evidence.
- **F0c** Sample input tensor as a heatmap — one window, three feature channels — confirming
  the `(3, 8, 20)` structure visually.

### Training diagnostics — Phase 4/5, thesis §4.x

These answer *"does training plateau?"*, which cannot be eyeballed from a wealth curve.

- **F1 Learning curves.** Mean log return per episode vs gradient step, one panel per
  algorithm, median with IQR band across 10 seeds. Individual seeds as faint lines behind.
- **F2 Train vs validation wealth.** Both curves per algorithm on one axis, with the
  selected checkpoint marked. The gap opening is the overfitting signature, and with ~756
  training days it is the single most important diagnostic in the project.
- **F3 Loss curves.** PG: objective. PPO: policy loss, value loss, entropy, approximate KL.
  DDPG: critic loss and actor loss. Liang devote two full figures to critic loss under
  varying learning rates — worth reproducing if the sweep is run.
- **F4 Plateau diagnostic.** Rolling slope of the validation curve with a shaded
  no-further-improvement band and the early-stop step marked. Turns "it plateaued" into a
  quantitative claim with a stated threshold.

### Performance — thesis §4.x, the headline results

- **F5 Wealth curves on test**, `p_t/p_0`, log-y axis (Jiang plot log-10): three agents plus
  UBAH, UCRP, Best Stock and Markowitz. **This is the headline figure.**
- **F6 Wealth curves on train and validation**, same layout — shows in-sample fit against
  the generalisation gap.
- **F7 Drawdown curves** beneath the test wealth curves, following Filos's P&L-plus-drawdown
  pairing.
- **F8 Metric table as a heatmap** — CR, AR, annualised Sharpe, MDD, turnover, win rate ×
  every strategy, colour-scaled per column.

### Distribution and significance — the variance control

- **F9 Seed distributions.** Box or violin plot of final wealth and Sharpe across 10 seeds
  per algorithm, with baseline reference lines. Answers whether the ranking is stable or an
  artefact of initialisation — and Liang report no equivalent.
- **F10 Forest plot** of mean daily return differences, each agent vs each baseline, with
  bootstrap and Bonferroni-adjusted intervals. Communicates far more honestly than a table
  of asterisks.

### Behaviour — thesis Objective 4

- **F11 Allocation heatmap**, assets × time, colour = weight, one panel per agent, built
  from `info["weights"]`.
- **F12 Concentration over time** — HHI and entropy per agent. This is the DDPG-degeneration
  watch: Liang report agents collapsing onto a single holding.
- **F13 Turnover** — time series and distribution per agent, which explains cost sensitivity
  in F14.

### Robustness sweeps — thesis §3.5.4

- **F14 Performance vs transaction cost**, `{0, 0.05, 0.1, 0.25, 0.5}%`, line per algorithm.
  Crossover points are the interesting result.
- **F15 Performance vs portfolio scale**, M ∈ {4, 8, 16}.
- **F16 Performance across alternative baskets**, isolating "these eight names" from
  "eight names".
- **F17 Adversarial ablation** — PG with and without `N(0, 0.002)` price noise: wealth curves
  plus seed box plot, mirroring Liang's Figure 12.

### Conventions

Single style module in `src/plots.py`. Consistent colour per strategy across every figure.
Log-scale wealth axes. Seed bands as median + IQR, never mean ± SD on skewed wealth data.
Every figure saved as both PNG (thesis) and PDF (vector), with the source `run_id` stamped
in the caption metadata.

---

## 5. Phases and gates

Nothing proceeds until the gate passes. Each gate is a committed artefact.

### Phase 0 — Scaffold
Repo layout, pinned `requirements.txt`, `config/base.yaml`, venv, `pytest` running green on
empty test stubs.
**Gate:** `pip install -r requirements.txt` succeeds on Python 3.10; `pytest` exits 0.

### Phase 1 — Data
Apply the §0 selection rule as of 2021-09-01 for M = 4, 8, 16 and the alternative baskets.
Download once, write `data/raw/*.csv` + `MANIFEST.json` (tickers, retrieval date, yfinance
version, `auto_adjust` setting). Build the `(3, 8, 20)` tensor over the *continuous* series;
split by **decision date**, not by re-windowing per split.
**Gate:** **F0a–F0c**; the 0.70 correlation gate applied with its outcome recorded in
`universe.yaml`; `MANIFEST.json` written; optional Qlib cross-check result; written
confirmation that no NaNs exist and that a lookback crossing a split boundary uses only
past data.

### Phase 2 — Costs, environment, extractor
`costs.py`, `env.py` and `extractors.py` with the full test suites above. The extractor is
built here rather than in Phase 4 so the non-flattening tests gate everything downstream.
**Gate:** `test_costs.py`, `test_env.py`, `test_extractor.py` green — including the `tau`
reachability assertion, the UCRP-equivalence check to `1e-10`, and both decisive extractor
tests (parameter-count invariance across M ∈ {4, 8, 16}, and permutation equivariance).

### Phase 3 — Baselines
All four baselines through `backtest()`, full metric suite, all three splits.
**Gate:** baseline table committed. **You now know exactly what the agents must beat, before
writing one.**

### Phase 3.5 — Baseline figure pipeline
Build `src/plots.py` and `scripts/06_figures.py` now, against baseline results only. Writing
the plotting layer before any agent exists means Phase 4 onward produces publication figures
automatically instead of accumulating a backlog of "I'll plot it later".
**Gate:** **F5–F8** rendering for baselines alone, PNG + PDF, consistent strategy colours.

### Phase 4 — PG
`agents/pg.py` + PVM, on the Phase 2 extractor. Built before the SB3 agents: it is the
algorithm that must be correct.
**Gate:** **F1–F4** for PG — learning curve, train-vs-validation, loss, plateau diagnostic.
Training log return rises; concentration (HHI) logged; realised max single-asset weight
logged and not pinned at a ceiling; final wealth beats UCRP on *training* data (if it cannot
fit in-sample, it is broken).

### Phase 5 — PPO and DDPG
SB3 with the identical extractor, `gamma=1.0`, `tau=5.0`, OU noise for DDPG.
**Gate:** the `isinstance(fe, EIIEExtractor)` assertion passing for every constructed model
including DDPG's critic; **F1–F4** for all three; a written architecture table recording
identical extractor, `tau`, `γ` and cost model, plus the two documented asymmetries
(`share_features_extractor`, critic `net_arch`).

### Phase 6 — Full experiment
3 algorithms × 10 seeds = 30 runs. Validation selects checkpoints and hyperparameters.
**Test is touched exactly once, at the end.** Then the sweeps: cost
{0, 0.05, 0.1, 0.25, 0.5}%, scale M ∈ {4, 8, 16}, and 3–4 alternative baskets.
**Gate:** **F5–F16** complete; per-seed results CSV; Bonferroni-corrected paired t-tests with
bootstrap CIs; allocation heatmaps built from `info["weights"]`, never from raw actions.

### Phase 7 — Ablation and write-up
PG with and without adversarial price noise `N(0, 0.002)` (Liang Algorithm 3) — ~3 lines,
and they report p=0.0076 on daily return, p=0.0338 on Sharpe, with MDD significantly worse
(p=2.73e-8). Then Chapter 4 from the artefacts, and the Chapter 1–3 revisions.
**Gate:** **F17**; every figure in Chapter 4 traceable to a committed run directory by
`run_id`; `scripts/06_figures.py` regenerates the entire suite from scratch in one command.

---

## 6. Chapter 1–3 edits this design requires

Track these; they are easy to forget once coding starts.

| Section | Change |
|---|---|
| Abstract, §1.2, §1.3, §1.4 | DDQN + PPO → **PG, PPO, DDPG**; recast as three points on the actor–critic spectrum |
| §1.4, §3.2 | Window 2015–2024 → **2021-09 – 2026-08**; "2,516 trading days" → ~1,260 |
| §1.4, §3.2 | 30 tech stocks → the eight-sector **rule** (not a ticker list) |
| §1.4, §3.5.4 | **Delete COVID references** — March 2020 is outside the window |
| §3.4.1 | DDQN → PG (deterministic policy gradient by direct reward maximisation) |
| §3.4 (new) | Formal MDP: state Eq. 20, action Eq. 19, reward Eq. 10, cost Eq. 14–16 |
| §3.4 (new) | Softmax placement, `tau`, and the deviation from Liang's post-softmax noise |
| §3.5.1 | `γ = 1.0`; 10 seeds; frozen-policy evaluation stated as a choice |
| §3.5.2 | Sharpe annualisation `× √252`; risk-free rate treatment |
| §3.5.3 | Add bootstrap CIs alongside t-tests |
| §3.5.4 | "top 10 vs 30" → **M ∈ {4, 8, 16}**; regime analysis → cost/scale/basket axes |
| §3.2 (new) | Baselines named: UBAH, UCRP, Best Stock, Markowitz |
| Ch. 5 | Limitations: no out-of-sample stress regime; no online retraining; ~756 training days |

---

## 7. Known risks

1. **~756 training days is thin** for three deep RL agents. EIIE weight sharing gives
   ~6,000 asset-windows of signal, which is the mitigation, but expect high seed variance —
   which is exactly why 10 seeds and the seed-distribution plot are mandatory, not optional.
2. **No stress regime is out-of-sample.** Structural under a chronological split on this
   window. Declare it in Chapter 5; discharge Objective 5 via the cost, scale and
   basket/seed axes instead.
3. **The agents may not beat buy-and-hold.** This is the normal result and Liang et al. say
   so themselves. The research question is the *replication* question — whether
   PG > PPO > DDPG transfers to a developed market — and a negative answer is a complete
   thesis. Write Chapter 4's skeleton so either outcome fits.
4. **DDPG may degenerate to a single holding.** Documented by Liang. Log HHI from Phase 4.
5. **`tau` interacts with everything.** Too low silently caps allocations; too high makes the
   policy near-one-hot and turnover explosive. Tune on validation, report the value.

---

## 8. Working agreement

Phases are gated. Each phase stops at its gate for review before the next begins.
Nothing is pushed to a remote without explicit approval.
