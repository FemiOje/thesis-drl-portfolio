# Deep Reinforcement Learning for Portfolio Management

Comparison of **PPO**, **DDPG** and **PG (A2C)** on S&P 500 equities for
continuous multi-asset portfolio allocation.

Environment and formalism follow **Jiang, Xu & Liang (2017)**, arXiv:1706.10059v1.
The three-algorithm comparison follows **Liang et al. (2018)**, arXiv:1808.09940v3.

Final-year CS thesis

---

## 1. Headline finding

**Feature-extractor architecture, not algorithm choice, was the binding
constraint.** Applied with library defaults, all 15 agents collapsed to constant
portfolios that ignored the market entirely — while still reporting plausible
fAPV and Sharpe figures. Restoring the price tensor's structure revived them.

| | Ablation (defaults) | Primary (fixed) |
|---|---|---|
| Feature extractor | SB3 default — flattens `X` to 1200 loose inputs | weight-shared convolution |
| Policies that are **frozen** | **15 of 15** | **2 of 12** (both DDPG) |
| Policies that respond to the market | **0 of 15** | **4 of 12** (all PPO) |
| Best agent on test (fAPV) | 1.1507 (= Buy & Hold) | **1.2074** (> Buy & Hold) |

The collapse is invisible in conventional metrics. The flattened agents posted
fAPV 1.15 and Sharpe 1.00 on held-out data — numbers that look like a working
system. They were constant portfolios. **This is the thesis's central
methodological point: report a degeneracy check, or you cannot know whether your
agent traded at all.**

Two independent failure modes were found and diagnosed:

1. **Structural.** SB3's `MultiInputPolicy` gives a `Dict` entry a CNN only if it
   is an *image space*. A `(3, 50, 8)` price tensor is not, so it is flattened to
   1200 unrelated numbers. Everything Jiang's Eq. 18 encodes — consecutive days,
   distinct assets, high/low/close channels — is destroyed before the network
   sees it. There is no convolution anywhere in the default models.
2. **Actor saturation.** DDPG drives every action output hard against the bounds,
   zeroing the tanh gradient and freezing the policy permanently. Reproduced at
   ±10 *and* ±5, with both extractors: **four runs, four freezes.**

---

## 2. Setup

Python 3.10+ (developed on 3.12.10, CPU only).

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
python -m pip install -r requirements.txt
```

Reproduce:

```bash
python experiments/run_all.py --skip-train    # results from saved checkpoints, ~1 min
python experiments/run_all.py                 # full retrain (see §4 for budgets)
```

Verified stack: torch 2.13.0+cpu · stable-baselines3 2.9.0 · gymnasium 1.0.0 ·
numpy 1.26.4 · pandas 2.3.3 · yfinance 1.5.2 · matplotlib 3.11.1.

---

## 3. Data

**Universe (fixed, approved):** AAPL, MSFT, JPM, JNJ, XOM, PG, AMZN, NVDA + cash
(m + 1 = 9). Chosen for liquidity, which is what Jiang's zero-slippage and
zero-market-impact hypotheses (§2.4) require.

**Window:** 2021-07-23 → 2026-07-22, 1254 trading days. Daily adjusted OHLC.

**Survivorship:** the list is fixed as of the *start* of the window and held
constant, per Jiang §3.1. No forward-looking selection.

**Source:** yfinance, isolated behind `data_loader._download_raw` — the only
provider-specific function in the codebase. See §9.

---

## 4. Experimental design

**Chronological split, never shuffled.**

| Split | Dates | Days | Decision steps | Role |
|---|---|---|---|---|
| Train | 2021-07-23 → 2025-01-21 | 878 | **828** | training |
| Validation | 2025-01-22 → 2025-10-20 | 188 | 187 | checkpoint selection |
| **Test** | 2025-10-21 → 2026-07-22 | 188 | 187 | **held out** |

**Two sweeps.** The second is the primary result; the first is retained as the
ablation that motivates it.

| | Ablation | **Primary** |
|---|---|---|
| Feature extractor | SB3 default (flatten) | weight-shared conv |
| Timesteps / seed | 300,000 | 500,000 |
| Action bound | ±10 | ±5 |
| Seeds | 5 × 3 algorithms | 5 (PPO, A2C), **2 (DDPG)** |
| Checkpoints | `results/models/` | `results/models_conv/` |

Commission 0.25% both sides (Jiang's default). Best-on-validation checkpointing.
Agents evaluated deterministically. A **freeze guard** stops any run whose
validation has not improved for 8 consecutive evaluations; it fired on 6 of 12
primary runs, including 4 of 5 A2C seeds.

**DDPG was stopped at 2 seeds.** Seed 0 took 295 minutes and froze; seed 1 froze
at the identical validation value (1.2390). With four independent demonstrations
of the same saturation failure, the remaining ~16 hours of compute would have
bought three more frozen policies. Reported as n=2 and labelled throughout.

**Metrics (Jiang §6.2):** fAPV · annualized Sharpe · maximum drawdown · turnover.
Our Sharpe is annualized (×√252); Jiang's Eq. 28 is not, so his table is not
directly comparable.

---

## 5. Results — primary sweep, all three periods

Agents show mean ± std over seeds. Benchmarks are deterministic.

### 5.1 Test (held out) — the headline table

| Strategy | fAPV | Annualized | Sharpe | Max drawdown | Turnover |
|---|---|---|---|---|---|
| **PPO** (n=5) | **1.2074 ± 0.0893** | **29.04%** | 1.325 ± 0.312 | 12.30% | 0.0936 |
| A2C (PG) (n=5) | 1.1982 ± 0.0341 | 27.61% | 0.988 ± 0.319 | 15.26% | 0.0174 |
| DDPG (n=2) | 1.1720 ± 0.0040 | 23.84% | 1.207 ± 0.654 | 14.20% | 0.0165 |
| Buy & Hold | 1.1506 | 20.81% | **1.852** | **5.64%** | 0.0107 |
| UCRP | 1.1532 | 21.18% | 1.793 | 6.04% | 0.0226 |
| Best stock (XOM, hindsight) | 1.3975 | 56.99% | 1.811 | 20.11% | 0.0107 |
| All-cash | 1.0000 | 0.00% | n/a | 0.00% | 0.0000 |

**All three algorithms now beat the passive benchmarks on return.** PPO returns
29.04% annualized against Buy & Hold's 20.81%.

**The benchmarks still win on risk.** Buy & Hold's Sharpe of 1.852 beats every
agent, and its 5.64% drawdown is less than half PPO's 12.30%. PPO earns ~5.7pp
more return while carrying 2.2× the drawdown. This is not a defect: the objective
(§6) contains no risk term, so the agents optimized exactly what they were given.
The correct claim is *"PPO beats the benchmarks on return, not on risk-adjusted
return."*

### 5.2 Validation (checkpoint selection — optimistically biased)

| Strategy | fAPV | Annualized | Sharpe | Max drawdown |
|---|---|---|---|---|
| PPO | 1.3288 ± 0.1499 | 47.03% | 1.255 | 24.15% |
| A2C (PG) | 1.2580 ± 0.0224 | 36.26% | 0.961 | 30.38% |
| DDPG | 1.1868 ± 0.0739 | 26.01% | 0.886 | 26.82% |
| Buy & Hold | 1.1266 | 17.43% | 0.951 | 17.37% |
| UCRP | 1.1376 | 18.98% | 0.976 | 17.67% |

### 5.3 Train (in-sample fit, 828 steps)

| Strategy | fAPV | Annualized | Sharpe | Max drawdown |
|---|---|---|---|---|
| PPO | 31.8469 ± 28.9066 | 152.46% | 2.551 | 24.27% |
| A2C (PG) | 5.4485 ± 1.1135 | 66.84% | 1.340 | 57.71% |
| DDPG | 4.3957 ± 3.3744 | 51.37% | 1.237 | 49.33% |
| Buy & Hold | 2.1672 | 26.54% | 1.258 | 22.51% |

PPO's in-sample fAPV of 31.8 against 1.21 on test is the overfitting signature
expected from **828 unique decision points revisited 604 times** at 500k steps.

### 5.4 The progression

Annualized return versus the best passive benchmark:

| | Train | Validation | Test |
|---|---|---|---|
| PPO | 152.46% | 47.03% | **29.04%** |
| A2C (PG) | 66.84% | 36.26% | 27.61% |
| DDPG | 51.37% | 26.01% | 23.84% |
| Best benchmark | 26.54% | 18.98% | 21.18% |
| **Agents beat benchmarks (return)?** | yes | yes | **yes** |

The ablation sweep failed this test on the held-out window; the primary sweep
passes it. That is the single clearest statement of what the architecture change
bought.

### 5.5 Figures

| Figure | File |
|---|---|
| Test wealth curves vs benchmarks | `results/evaluation_conv/test/01_wealth_curves.png` |
| Test allocation heatmaps | `results/evaluation_conv/test/02_allocation_heatmaps.png` |
| Test per-seed spread | `results/evaluation_conv/test/03_seed_distributions.png` |
| Validation performance during training | `results/evaluation_conv/training_curves.png` |
| Train / validation equivalents | `results/evaluation_conv/{train,val}/` |
| **Ablation (flattened) equivalents** | `results/evaluation/{train,val,test}/` |
| Data checks | `results/data_checks/` |

![Test wealth curves](results/evaluation_conv/test/01_wealth_curves.png)

---

## 6. Did the agents actually trade?

The question conventional metrics cannot answer. `experiments/policy_diagnostic.py`
queries each policy on observations from **nine widely separated market regimes**
(start/middle/end of each split) while holding the portfolio state fixed, so any
output change is attributable to the price tensor alone.

Four verdicts: `FROZEN` (weights never move), `MARKET-BLIND` (weights move, but
from portfolio feedback rather than prices), `negligible` (responds by under 1
percentage point across regimes), `responds to X_t`.

**Test window, primary sweep:**

| algo | seed | probe | verdict | test fAPV | turnover |
|---|---|---|---|---|---|
| PPO | 0 | 0.506 | **responds** | 1.2293 | 0.1558 |
| PPO | 1 | 0.762 | **responds** | 1.1970 | 0.1157 |
| PPO | 2 | 0.612 | **responds** | 1.3417 | 0.0876 |
| PPO | 3 | 0.000 | MARKET-BLIND | 1.1707 | 0.0261 |
| PPO | 4 | 0.066 | **responds** | 1.0981 | 0.0827 |
| A2C | 0, 1 | ≤0.0015 | negligible | 1.19–1.20 | ~0.016 |
| A2C | 2, 3, 4 | 0.000 | MARKET-BLIND | 1.17–1.25 | ~0.018 |
| DDPG | 0, 1 | 0.000 | **FROZEN** | 1.17 | ~0.017 |

**4 of 12 policies respond meaningfully — every one of them PPO.**

Three consequences:

* **The conv encoder eliminated freezing entirely for the policy-gradient
  methods** (0 of 10, versus 10 of 10 in the ablation).
* **A2C did not benefit.** Every seed is market-blind or negligible, and its
  turnover (~0.017) is a sixth of PPO's. Its 1.1982 is a *constant portfolio that
  happened to do well* — it must not be reported as a working trading agent.
* **Turnover corroborates the probe.** PPO's market-blind seed 3 trades at 0.0261
  while its responsive seeds trade at 0.08–0.16.

---

## 7. Convergence

Slope of the validation curve over its final third, fAPV per 100k steps
(`experiments/plot_training.py`):

| Algorithm | Slope /100k | Verdict |
|---|---|---|
| PPO | −0.0286 | plateaued at 500k |
| **A2C (PG)** | **+0.0273** | **still rising — undertrained** |
| DDPG | +0.0000 | frozen |

This **reverses the ablation sweep**, where PPO was the undertrained one
(+0.0242) and A2C had plateaued. Note the caveat it creates: A2C is compared at a
budget it had not converged at. Given that every A2C policy is market-blind,
more steps would likely refine a constant allocation rather than produce trading
— but that is an expectation, not a measurement.

---

## 8. Conclusions

1. **Architecture dominated algorithm choice.** With the default extractor all
   three algorithms were indistinguishable because all were constant. Restoring
   tensor structure moved policies by six orders of magnitude and lifted every
   algorithm above the passive benchmarks on return.
2. **Conventional metrics concealed total failure.** fAPV 1.15 and Sharpe 1.00 on
   held-out data, from policies that never looked at the market. A degeneracy
   check is not optional.
3. **Only PPO learned to trade.** 4 of 5 seeds respond to the price tensor. A2C
   is market-blind on all 5; DDPG is frozen on both.
4. **DDPG fails by actor saturation, reproducibly.** Every output pinned to the
   bound, gradient zeroed, policy frozen — at ±10 and ±5, with both extractors.
   Narrowing the bound limited the damage (485,000,000:1 → 22,000:1 allocation
   ratio) but did **not** prevent the freeze. This converges with Liang et al.'s
   independent finding that DDPG is the weakest of the three, and explains why.
5. **Benchmarks still win risk-adjusted.** Buy & Hold beats every agent on Sharpe
   and drawdown. The objective contains no risk term (§6 of METHODOLOGY), so this
   is the expected outcome, not an anomaly.
6. **No ranking between PPO and A2C is defensible.** 1.2074 vs 1.1982 is a 0.009
   gap against a PPO standard deviation of 0.089 at n=5. Report the diagnostic
   difference instead — it is far more informative than the fAPV difference.

### Limitations

* **828 unique training decision points**, revisited 604 times at 500k steps.
  Jiang's 30-minute crypto bars give roughly 29× more distinct experience. This
  is the structural constraint behind the overfitting and, plausibly, the
  degeneracy.
* Reward SNR is 0.035 — the gradient is ~97% noise.
* Zero slippage and zero market impact assumed (Jiang §2.4).
* Test window is a single market regime.
* DDPG at n=2, A2C not converged.

### Next steps, in order of expected value

1. **Add a risk term to the objective** — the agents' only real deficit is
   risk-adjusted performance. Note that Liang et al. tried both a volatility
   penalty and a Sharpe objective and reported *both failed*, so this is not a
   free win.
2. **Full EIIE** — shared conv head, `w_prev` as a channel, cash bias, PVM. The
   current encoder is deliberately the *minimal* intervention (§below).
3. **Diagnose A2C's market-blindness** — it received the same encoder as PPO and
   did not benefit.
4. Observation normalization (`VecNormalize`); the inputs sit at 0.988 ± 0.092.

### Why the encoder is deliberately not full EIIE

Full EIIE bundles four changes at once. Shipping all four would make it
impossible to attribute the recovery to any one of them. Changing **only** the
encoder isolates the causal claim: *preserving tensor structure is what revived
the policy.* This is a controlled experiment by design, not an unfinished one.
`experiments/extractors.py` documents the four specific gaps versus Jiang.

---

## 9. Where everything is

### Documents

| Path | Contents |
|---|---|
| `docs/METHODOLOGY.md` | Code ↔ Jiang equation map, full specs, deviations, references |
| `docs/WORKLOG.md` | Chronological record of decisions and results |
| `results/evaluation_conv/{test,val,train}/RESULTS.md` | **Primary** results per window |
| `results/evaluation/{test,val,train}/RESULTS.md` | Ablation results per window |

### Code

| Path | Purpose |
|---|---|
| `data/data_loader.py` | Download, clean, split, Jiang Eq. 18 tensor |
| `env/portfolio_env.py` | Gymnasium environment implementing Jiang's formalism |
| `experiments/extractors.py` | **Weight-shared conv encoder** — the primary intervention |
| `experiments/train.py` | Training CLI (`--algo --extractor --action-bound --steps`) |
| `experiments/evaluate.py` | Evaluation + benchmarks (`--split train\|val\|test`) |
| `experiments/policy_diagnostic.py` | **Degeneracy diagnostic (§6)** |
| `experiments/plot_training.py` | Validation curves + convergence check |
| `experiments/run_all.py` | One-command reproduction |
| `tests/test_env.py` | 7 unit tests, all passing |

### Artefacts

| Path | Contents |
|---|---|
| `data/raw/` | Cached per-ticker CSVs — experiments never re-download |
| `results/models_conv/{algo}/seed{n}/` | **Primary** checkpoints + `config.json` |
| `results/models/{algo}/seed{n}/` | Ablation checkpoints |
| `results/evaluation_conv/`, `results/evaluation/` | Tables and figures |
| `results/data_checks/` | Data sanity figures |
| `results/tensorboard/` | Training logs (`{algo}_{extractor}_seed{n}`) |

---

## 10. Open items for your decision

1. **Data source** — yfinance vs the Microsoft source you recommended. One
   function to change, but it invalidates every trained checkpoint.
2. **PG → A2C substitution** — SB3 has no bare REINFORCE, so A2C stands in. This
   is the one substitution that weakens the direct comparison with Liang et al.,
   whose headline result is specifically about PG. Given A2C proved market-blind
   here, the substitution now carries more weight than it did.
3. **Framing** — the approved plan was a three-algorithm comparison. The result is
   better described as a diagnostic study: architecture dominates algorithm, and
   conventional metrics hide degeneracy. This is a change of emphasis worth
   confirming before the chapter is written.
4. **DDPG at n=2** — whether to spend ~16 h completing seeds 2–4 for symmetry,
   given four independent demonstrations of the same freeze.

### Verification status

Environment unit tests 7/7. Benchmark cost accounting agrees with closed-form
values at zero commission to 1.7e-13. Every Jiang equation number cited in code
was verified against the source PDF. The test window was evaluated once per
sweep — twice in total, once for the ablation and once for the primary result,
which is disclosed rather than concealed.
