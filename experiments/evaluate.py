"""Held-out test-window evaluation, benchmarks, metrics and figures.

Runs each trained agent once on the TEST split (the window is touched exactly
here, at the end of the project), alongside four classical
benchmarks, and reports the Jiang §6.2 metric set.

Protocol
* Models: the best-on-validation checkpoint of each (algo, seed) from the
  training sweep, at 300k timesteps — the agreed equal-budget head-to-head.
  Agents act deterministically (``predict(..., deterministic=True)``).
* Every strategy, agents and benchmarks alike, is stepped through the same
  ``PortfolioEnv``, so the transaction-cost accounting (Jiang Eq. 14-16) is
   the same code path for all of them. Benchmarks express their target
  weights as log-weight actions, which the env's softmax inverts exactly
  (round-trip asserted in ``weights_to_action``).
* Every strategy starts fully in cash, so all of them (including buy-and-hold)
  pay the commission on their initial purchase. No strategy gets a free entry.

Benchmarks
* Buy-and-Hold — equal weight across the m stocks on the first decision step,
  then never trades again (target = the drifted holdings, so turnover 0, μ = 1).
* UCRP — uniform constant rebalanced portfolio: rebalances back to equal
  weight across the m stocks every step, and pays the cost of doing so.
* Best stock (hindsight) — the single stock with the highest cumulative price
  relative over the test window, bought at the start and held. Not achievable
  ex ante; it is an upper reference, labelled as such.
* All-cash — the do-nothing floor; wealth stays 1.0 (r_t = 0 every step).

Metrics
* fAPV — final accumulated portfolio value p_f / p_0 (p_0 = 1).
* Sharpe — annualized, risk-free 0: mean(ρ)/std(ρ)·sqrt(252) on daily simple
  returns ρ_t = p_t/p_{t-1} - 1 = exp(r_t) - 1 (sample std, ddof=1).
* MDD — maximum drawdown, max_t (1 - p_t / max_{s<=t} p_s).
* Turnover — mean Σ_i |w_t,i - w'_t,i| per step, in [0, 2] (2 = full
  rotation). Not a Jiang metric; reported for interpretability of cost behaviour.

Outputs (under results/evaluation/<split>/)
* ``per_seed_results.csv``  — one row per (strategy, seed)
* ``summary.csv`` / ``RESULTS.md`` — mean ± std per algorithm + benchmark rows
* ``01_wealth_curves.png`` — log-scale wealth, agents (mean + per-seed) vs benchmarks
* ``02_allocation_heatmaps.png`` — weight allocation over time, median seed per algo
* ``03_seed_distributions.png`` — per-seed fAPV / Sharpe spread per algorithm

Run:  python experiments/evaluate.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass, field

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("env", "data"):
    _p = os.path.join(_ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import data_loader as dl  # noqa: E402
import portfolio_env as pe  # noqa: E402

RESULTS_DIR = os.path.join(_ROOT, "results")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
EVAL_DIR = os.path.join(RESULTS_DIR, "evaluation")

TRADING_DAYS = 252  # annualization factor for daily bars

ALGOS = ["ppo", "a2c", "ddpg"]
ALGO_LABEL = {"ppo": "PPO", "a2c": "A2C (PG)", "ddpg": "DDPG"}

# Categorical hues for the three agents. Benchmarks are deliberately achromatic and
# dashed so they read as a subordinate reference group, with identity carried by
# dash pattern + direct labels rather than colour alone.
ALGO_COLOR = {"ppo": "#0072B2", "a2c": "#009E73", "ddpg": "#D55E00"}

# Per-split report framing. Only `test` is a headline result; train and val are
# in-sample diagnostics and must say so on their own page, because a reader who
# opens one of those files directly will not have the surrounding context.
TITLE = {
    "test": "Held-out test-window results",
    "val": "Validation-window results (in-sample diagnostic)",
    "train": "Training-window results (in-sample diagnostic)",
}
PREAMBLE = {
    "test": "The test split was touched exactly once, here. Metrics follow "
            "Jiang §6.2; turnover is Σ|Δw| per step (0-2).",
    "val": "**In-sample diagnostic, not a headline result.** Checkpoints were "
           "selected on this window, so these figures are optimistically "
           "biased. Reported to show the train/val/test progression. Metrics "
           "follow Jiang §6.2; turnover is Σ|Δw| per step (0-2).",
    "train": "**In-sample diagnostic, not a headline result.** The agents were "
             "trained on this window, so these figures show fit, not "
             "generalization. Metrics follow Jiang §6.2; turnover is Σ|Δw| "
             "per step (0-2).",
}
BENCH_STYLE = {
    "Buy & Hold": ("#3d3d3d", (0, (6, 2))),
    "UCRP": ("#6b6b6b", (0, (2, 2))),
    "Best stock (hindsight)": ("#9a9a9a", (0, (1, 1.6))),
    "All-cash": ("#bdbdbd", "solid"),
}


# =============================================================================
# Weights <-> action (the env applies a softmax to whatever action it is given)
# =============================================================================
def weights_to_action(w: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return an action ``a`` with ``softmax(a) == w`` (to ~1e-12).

    Lets a benchmark with explicit target weights be driven through the *same*
    env as the agents, so the Eq. 14-16 cost accounting is identical for all
    strategies. ``log`` is the exact softmax inverse up to an additive constant;
    zero weights are floored at ``eps`` (a zero weight is unreachable by a
    softmax, but 1e-12 is far below any accounting tolerance).
    """
    w = np.asarray(w, dtype=np.float64)
    a = np.log(np.clip(w, eps, None))
    err = float(np.max(np.abs(pe.softmax(a) - w)))
    assert err < 1e-9, f"weights->action round-trip broke (err={err:.2e})"
    return a


# =============================================================================
# Policies — each maps (obs, env) -> raw action
# =============================================================================
def agent_policy(model):
    """Deterministic SB3 policy."""

    def _policy(obs, env):
        action, _ = model.predict(obs, deterministic=True)
        return action

    return _policy


def all_cash_policy(m: int):
    w = np.zeros(m + 1)
    w[0] = 1.0
    a = weights_to_action(w)
    return lambda obs, env: a


def ucrp_policy(m: int):
    """Uniform constant rebalanced over the m stocks — rebalances every step."""
    w = np.zeros(m + 1)
    w[1:] = 1.0 / m
    a = weights_to_action(w)
    return lambda obs, env: a


def hold_policy(initial_weights: np.ndarray):
    """Buy ``initial_weights`` on the first step, then never trade again.

    After the opening trade the target is the env's *drifted* holdings
    (Jiang Eq. 7), which means zero turnover and μ = 1 for every later step —
    i.e. a genuine hold, not a rebalance.
    """
    a0 = weights_to_action(initial_weights)
    state = {"opened": False}

    def _policy(obs, env):
        if not state["opened"]:
            state["opened"] = True
            return a0
        return weights_to_action(env.w_hold)

    return _policy


def equal_weight_stocks(m: int) -> np.ndarray:
    w = np.zeros(m + 1)
    w[1:] = 1.0 / m
    return w


def single_stock(m: int, idx: int) -> np.ndarray:
    """All wealth in stock ``idx`` (0-based among the m stocks)."""
    w = np.zeros(m + 1)
    w[1 + idx] = 1.0
    return w


# =============================================================================
# Episode runner + metrics
# =============================================================================
@dataclass
class Run:
    """One strategy's full pass over the test window."""

    name: str
    kind: str  # "agent" | "benchmark"
    dates: np.ndarray
    wealth: np.ndarray  # (T,)  p_t, starting from p_0 = 1
    weights: np.ndarray  # (T, m+1)
    turnover: np.ndarray  # (T,)
    mu: np.ndarray  # (T,)
    algo: str | None = None
    seed: int | None = None
    metrics: dict = field(default_factory=dict)


def run_episode(env: pe.PortfolioEnv, policy, name: str, kind: str,
                algo: str | None = None, seed: int | None = None) -> Run:
    """Step ``policy`` through one full deterministic pass over ``env``."""
    obs, _ = env.reset(seed=0)  # test env is deterministic; seed fixes nothing but is explicit
    done = False
    while not done:
        obs, _reward, terminated, truncated, _info = env.step(policy(obs, env))
        done = terminated or truncated

    h = env.history
    run = Run(
        name=name,
        kind=kind,
        algo=algo,
        seed=seed,
        dates=np.array([r.date for r in h]),
        wealth=np.array([r.wealth for r in h]),
        weights=np.stack([r.weights for r in h]),
        turnover=np.array([r.turnover for r in h]),
        mu=np.array([r.mu for r in h]),
    )
    run.metrics = compute_metrics(run.wealth, run.turnover)
    return run


def compute_metrics(wealth: np.ndarray, turnover: np.ndarray) -> dict:
    """Jiang §6.2 metrics (+ turnover) from a wealth path starting at p_0 = 1."""
    p = np.concatenate([[1.0], np.asarray(wealth, dtype=np.float64)])
    rho = p[1:] / p[:-1] - 1.0  # daily simple returns
    n = len(rho)

    # Sharpe is undefined for a strategy that takes no risk and earns no return
    # (all-cash). Its returns are not *exactly* zero — the env's softmax cannot
    # emit a true zero weight, so ~1e-12 leaks into the stocks — and dividing
    # that noise by its own near-zero std yields a meaningless O(1) number. Any
    # path whose returns are numerically zero is reported as undefined (NaN).
    sd = float(np.std(rho, ddof=1)) if n > 1 else 0.0
    degenerate = float(np.max(np.abs(rho))) < 1e-10
    sharpe = (float("nan") if degenerate or sd <= 0.0
              else float(np.mean(rho) / sd * np.sqrt(TRADING_DAYS)))

    peak = np.maximum.accumulate(p)
    mdd = float(np.max(1.0 - p / peak))

    fapv = float(p[-1])
    ann_return = float(fapv ** (TRADING_DAYS / n) - 1.0)

    return {
        "fAPV": fapv,
        "ann_return": ann_return,
        "sharpe": sharpe,
        "mdd": mdd,
        "turnover": float(np.mean(turnover)),
    }


# =============================================================================
# Evaluation driver
# =============================================================================
def load_model(algo: str, seed: int, models_dir: str, checkpoint: str):
    """Load a training checkpoint. Returns None (with a warning) if missing."""
    from stable_baselines3 import A2C, DDPG, PPO

    cls = {"ppo": PPO, "a2c": A2C, "ddpg": DDPG}[algo]
    path = os.path.join(models_dir, algo, f"seed{seed}", f"{checkpoint}_model.zip")
    if not os.path.exists(path):
        print(f"  !! missing checkpoint, skipping: {path}")
        return None
    return cls.load(path, device="cpu")


def evaluate_all(args) -> list[Run]:
    dataset = dl.build_dataset()  # cached; never re-downloads
    m = dataset.n_assets

    def split_env() -> pe.PortfolioEnv:
        # Fresh env per run: full chronological pass, no random start.
        # random_start=False is explicit — make_env would otherwise default it
        # to True on the train split, which would break a deterministic pass.
        return pe.make_env(args.split, commission=args.commission,
                           episode_length=None, random_start=False, dataset=dataset)

    a, b = getattr(dataset.splits, args.split)
    probe = split_env()
    print(f"{args.split.upper()} window: idx [{a}, {b})  "
          f"{dataset.dates[a].date()} -> {dataset.dates[b - 1].date()}  "
          f"({probe._t_last - probe._t_first + 1} decision steps, "
          f"commission {args.commission:.2%})")
    if args.split != "test":
        print(f"NOTE: '{args.split}' is an IN-SAMPLE diagnostic — the agents were "
              f"trained on train and checkpointed on val. Not a headline result.")

    runs: list[Run] = []

    # --- agents ------------------------------------------------------------
    for algo in args.algos:
        for seed in args.seeds:
            model = load_model(algo, seed, args.models_dir, args.checkpoint)
            if model is None:
                continue
            runs.append(run_episode(
                split_env(), agent_policy(model),
                name=f"{ALGO_LABEL[algo]} seed{seed}", kind="agent",
                algo=algo, seed=seed,
            ))

    # --- benchmarks --------------------------------------------------------
    # Best single stock is chosen with hindsight over the *decision* window
    # actually traded, i.e. the same y's the strategies experience.
    lo, hi = probe._t_first, probe._t_last
    cum = np.prod(dataset.y[lo + 1: hi + 2], axis=0)  # (m,) cumulative relative
    best_idx = int(np.argmax(cum))
    best_name = f"Best stock (hindsight): {dataset.tickers[best_idx]}"

    bench = [
        ("Buy & Hold", hold_policy(equal_weight_stocks(m))),
        ("UCRP", ucrp_policy(m)),
        ("Best stock (hindsight)", hold_policy(single_stock(m, best_idx))),
        ("All-cash", all_cash_policy(m)),
    ]
    for name, policy in bench:
        runs.append(run_episode(split_env(), policy, name=name, kind="benchmark"))
    print(f"Hindsight best stock over this window: {best_name.split(': ')[1]} "
          f"(cumulative relative {cum[best_idx]:.4f})")

    return runs


# =============================================================================
# Reporting — CSV + markdown
# =============================================================================
METRIC_COLS = ["fAPV", "ann_return", "sharpe", "mdd", "turnover"]


def aggregate(runs: list[Run]) -> list[dict]:
    """Collapse agent seeds to mean ± std; benchmarks pass through as-is."""
    rows: list[dict] = []
    for algo in ALGOS:
        seed_runs = [r for r in runs if r.algo == algo]
        if not seed_runs:
            continue
        row = {"strategy": ALGO_LABEL[algo], "kind": "agent", "n": len(seed_runs)}
        for k in METRIC_COLS:
            vals = np.array([r.metrics[k] for r in seed_runs], dtype=np.float64)
            row[f"{k}_mean"] = float(np.nanmean(vals))
            row[f"{k}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    for r in runs:
        if r.kind != "benchmark":
            continue
        row = {"strategy": r.name, "kind": "benchmark", "n": 1}
        for k in METRIC_COLS:
            row[f"{k}_mean"] = r.metrics[k]
            row[f"{k}_std"] = 0.0
        rows.append(row)
    return rows


def write_csvs(runs: list[Run], summary: list[dict], out_dir: str) -> None:
    per_seed = os.path.join(out_dir, "per_seed_results.csv")
    with open(per_seed, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["strategy", "kind", "algo", "seed"] + METRIC_COLS)
        for r in runs:
            w.writerow([r.name, r.kind, r.algo or "", "" if r.seed is None else r.seed]
                       + [f"{r.metrics[k]:.6f}" for k in METRIC_COLS])

    summ = os.path.join(out_dir, "summary.csv")
    cols = ["strategy", "kind", "n"] + [f"{k}_{s}" for k in METRIC_COLS
                                        for s in ("mean", "std")]
    with open(summ, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for row in summary:
            w.writerow({c: (f"{row[c]:.6f}" if isinstance(row[c], float) else row[c])
                        for c in cols})
    print(f"Wrote {per_seed}\nWrote {summ}")


def _pm(mean: float, std: float, fmt: str, n: int, pm: str = "±") -> str:
    """Format ``mean ± std`` (or just the mean for a single run). NaN -> n/a."""
    if not np.isfinite(mean):
        return "n/a"
    return f"{mean:{fmt}} {pm} {std:{fmt}}" if n > 1 else f"{mean:{fmt}}"


def write_markdown(runs: list[Run], summary: list[dict], out_dir: str,
                   args, meta: dict) -> None:
    lines = [
        f"# {TITLE[args.split]}",
        "",
        f"{args.split.capitalize()} window: **{meta['start']} → {meta['end']}** "
        f"({meta['steps']} decision steps). Commission **{args.commission:.2%}** "
        f"both sides. Agents: best-on-validation checkpoints at "
        f"{meta['train_steps']} timesteps, {len(args.seeds)} seeds, "
        "acting deterministically.",
        "",
        PREAMBLE[args.split],
        "",
        "## Summary (agents: mean ± std over seeds)",
        "",
        "| Strategy | fAPV | Annualized return | Sharpe | Max drawdown | Turnover |",
        "|---|---|---|---|---|---|",
    ]
    for row in summary:
        n = row["n"]
        lines.append(
            f"| {row['strategy']} "
            f"| {_pm(row['fAPV_mean'], row['fAPV_std'], '.4f', n)} "
            f"| {_pm(row['ann_return_mean'] * 100, row['ann_return_std'] * 100, '.2f', n)}% "
            f"| {_pm(row['sharpe_mean'], row['sharpe_std'], '.3f', n)} "
            f"| {_pm(row['mdd_mean'] * 100, row['mdd_std'] * 100, '.2f', n)}% "
            f"| {_pm(row['turnover_mean'], row['turnover_std'], '.4f', n)} |"
        )

    lines += ["", "## Per-seed detail", "",
              "| Algorithm | Seed | fAPV | Sharpe | Max drawdown | Turnover |",
              "|---|---|---|---|---|---|"]
    for r in runs:
        if r.kind != "agent":
            continue
        mt = r.metrics
        lines.append(
            f"| {ALGO_LABEL[r.algo]} | {r.seed} | {mt['fAPV']:.4f} | "
            f"{mt['sharpe']:.3f} | {mt['mdd'] * 100:.2f}% | {mt['turnover']:.4f} |"
        )

    lines += [
        "",
        "## Figures",
        "",
        "* `01_wealth_curves.png` — wealth over the window (log scale), "
        "agent means with per-seed spread, against the four benchmarks.",
        "* `02_allocation_heatmaps.png` — portfolio weights over time for the "
        "median-fAPV seed of each algorithm.",
        "* `03_seed_distributions.png` — per-seed fAPV and Sharpe spread.",
        "",
        "Regenerate with `python experiments/evaluate.py`.",
        "",
    ]
    path = os.path.join(out_dir, "RESULTS.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"Wrote {path}")


# =============================================================================
# Figures
# =============================================================================
def make_figures(runs: list[Run], dataset, out_dir: str, split: str = "test") -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless; write PNGs without a display
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    agents = [r for r in runs if r.kind == "agent"]
    benches = [r for r in runs if r.kind == "benchmark"]
    algos_present = [a for a in ALGOS if any(r.algo == a for r in agents)]

    # --- Fig 1: wealth curves (log scale) ----------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    dates = runs[0].dates

    for r in benches:
        color, dash = BENCH_STYLE[r.name]
        ax.plot(r.dates, r.wealth, color=color, ls=dash, lw=1.6, label=r.name, zorder=2)

    for algo in algos_present:
        seed_runs = [r for r in agents if r.algo == algo]
        stack = np.stack([r.wealth for r in seed_runs])
        color = ALGO_COLOR[algo]
        for r in seed_runs:  # per-seed spread, recessive
            ax.plot(r.dates, r.wealth, color=color, lw=0.7, alpha=0.28, zorder=3)
        ax.plot(dates, stack.mean(axis=0), color=color, lw=2.0,
                label=f"{ALGO_LABEL[algo]} (mean of {len(seed_runs)} seeds)", zorder=4)

    # Log scale (per the plan), but the wealth range here is narrow, so the
    # default decade ticks label almost nothing. Force evenly spaced ticks in
    # plain decimals across the observed range instead.
    ax.set_yscale("log")
    all_wealth = np.concatenate([r.wealth for r in runs])
    lo, hi = float(all_wealth.min()), float(all_wealth.max())
    step = 0.05 if (hi - lo) <= 0.5 else 0.1
    ticks = np.arange(np.floor(lo / step) * step, np.ceil(hi / step) * step + 1e-9, step)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.set_ylim(lo * 0.98, hi * 1.02)
    ax.set_ylabel("portfolio value  $p_t$  ($p_0 = 1$, log scale)")
    ax.set_title(f"{split.capitalize()}-window wealth — DRL agents vs. classical benchmarks "
                 f"({str(dates[0])[:10]} to {str(dates[-1])[:10]})")
    ax.grid(True, which="both", axis="y", color="#e8e8e8", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_wealth_curves.png"), dpi=130)
    plt.close(fig)

    # --- Fig 2: allocation heatmaps (median-fAPV seed per algo) ------------
    labels = ["CASH"] + list(dataset.tickers)
    n_panels = len(algos_present)
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 3.1 * n_panels), squeeze=False)
    for ax, algo in zip(axes[:, 0], algos_present):
        seed_runs = sorted([r for r in agents if r.algo == algo],
                           key=lambda r: r.metrics["fAPV"])
        med = seed_runs[len(seed_runs) // 2]  # median-fAPV seed
        im = ax.imshow(med.weights.T, aspect="auto", origin="lower",
                       cmap="viridis", vmin=0.0, vmax=1.0,
                       extent=[0, len(med.weights), -0.5, len(labels) - 0.5])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(f"{ALGO_LABEL[algo]} — seed {med.seed} (median fAPV "
                     f"{med.metrics['fAPV']:.4f}), mean turnover "
                     f"{med.metrics['turnover']:.3f}", fontsize=10)
        ax.set_xlabel(f"{split} decision step")
        # pad leaves room for the per-row weight labels added below, which sit
        # against the right spine and would otherwise run into the colorbar.
        fig.colorbar(im, ax=ax, label="weight", pad=0.07)

        # Colour alone reads poorly for exact magnitudes, and these policies are
        # constant, so the whole row collapses to one number. Print it on a right
        # axis. If a weight does move, append its peak-to-trough range rather
        # than silently showing only the mean.
        mean_w = med.weights.mean(axis=0)
        range_w = med.weights.max(axis=0) - med.weights.min(axis=0)
        right = ax.twinx()
        right.set_ylim(ax.get_ylim())
        right.set_yticks(range(len(labels)))
        right.set_yticklabels(
            [f"{m * 100:5.1f}%" + (f" ±{r * 100:.1f}" if r > 0.005 else "")
             for m, r in zip(mean_w, range_w)],
            fontsize=8, fontfamily="monospace",
        )
        right.tick_params(length=0)
        for side in ("top", "right"):
            right.spines[side].set_visible(False)
    fig.suptitle(f"Portfolio allocation over the {split} window", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(os.path.join(out_dir, "02_allocation_heatmaps.png"), dpi=130)
    plt.close(fig)

    # --- Fig 3: per-seed distributions -------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, key, title in [(axes[0], "fAPV", "Final accumulated portfolio value"),
                           (axes[1], "sharpe", "Sharpe ratio (annualized)")]:
        data = [[r.metrics[key] for r in agents if r.algo == a] for a in algos_present]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops=dict(color="#2b2b2b", lw=1.6),
                        flierprops=dict(marker="", ls="none"))
        for patch, algo in zip(bp["boxes"], algos_present):
            patch.set_facecolor(ALGO_COLOR[algo])
            patch.set_alpha(0.25)
            patch.set_edgecolor(ALGO_COLOR[algo])
            patch.set_linewidth(1.4)
        # 5 seeds is few for a box — show every seed as a point.
        for i, (vals, algo) in enumerate(zip(data, algos_present), start=1):
            jitter = np.linspace(-0.09, 0.09, len(vals))
            ax.plot(i + jitter, vals, "o", ms=5, color=ALGO_COLOR[algo],
                    mec="white", mew=0.8, zorder=3)
        # Benchmark reference lines. They cluster tightly (especially on Sharpe),
        # so identity goes in a shared legend below rather than inline labels,
        # which collide unreadably at this scale.
        for r in benches:
            if not np.isfinite(r.metrics[key]):
                continue
            color, dash = BENCH_STYLE[r.name]
            ax.axhline(r.metrics[key], color=color, ls=dash, lw=1.2, zorder=1)
        ax.set_xticks(range(1, len(algos_present) + 1))
        ax.set_xticklabels([ALGO_LABEL[a] for a in algos_present], fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", color="#ececec", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles = [Line2D([], [], color=BENCH_STYLE[r.name][0], ls=BENCH_STYLE[r.name][1],
                      lw=1.4, label=r.name)
               for r in benches if np.isfinite(r.metrics["fAPV"])]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=8, frameon=False, title="Benchmarks", title_fontsize=8)
    fig.suptitle(f"Per-seed spread on the {split} window (one point per seed)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    fig.savefig(os.path.join(out_dir, "03_seed_distributions.png"), dpi=130)
    plt.close(fig)

    print(f"Wrote 3 figures to {out_dir}")


# =============================================================================
# CLI
# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate trained agents on a chronological split."
    )
    ap.add_argument("--split", choices=["train", "val", "test"], default="test",
                    help="which chronological split to evaluate on "
                         "(default: test — the headline result)")
    ap.add_argument("--algos", nargs="+", default=ALGOS, choices=ALGOS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--commission", type=float, default=pe.DEFAULT_COMMISSION)
    ap.add_argument("--checkpoint", choices=["best", "final"], default="best",
                    help="which saved checkpoint to evaluate (default: "
                         "best-on-validation)")
    ap.add_argument("--models-dir", default=MODELS_DIR)
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: results/evaluation/<split>)")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join(EVAL_DIR, args.split)
    os.makedirs(args.out_dir, exist_ok=True)
    runs = evaluate_all(args)
    if not any(r.kind == "agent" for r in runs):
        print("No agent checkpoints found — nothing to evaluate.")
        return 1

    dataset = dl.build_dataset()
    summary = aggregate(runs)

    print(f"\n=== {args.split.upper()}-window results "
          f"(agents: mean +/- std over seeds) ===")
    print(f"{'strategy':<26}{'fAPV':>20}{'Sharpe':>20}{'MDD':>20}{'turnover':>12}")
    for row in summary:
        n = row["n"]
        mdd = _pm(row["mdd_mean"] * 100, row["mdd_std"] * 100, ".2f", n, pm="+/-")
        print(f"{row['strategy']:<26}"
              f"{_pm(row['fAPV_mean'], row['fAPV_std'], '.4f', n, pm='+/-'):>20}"
              f"{_pm(row['sharpe_mean'], row['sharpe_std'], '.3f', n, pm='+/-'):>20}"
              f"{mdd + '%' if mdd != 'n/a' else mdd:>20}"
              f"{row['turnover_mean']:>12.4f}")

    agent_run = next(r for r in runs if r.kind == "agent")
    meta = {
        "start": str(agent_run.dates[0])[:10],
        "end": str(agent_run.dates[-1])[:10],
        "steps": len(agent_run.dates),
        "train_steps": _train_steps(args),
    }
    write_csvs(runs, summary, args.out_dir)
    write_markdown(runs, summary, args.out_dir, args, meta)
    if not args.no_figures:
        make_figures(runs, dataset, args.out_dir, args.split)
    print("\nEVALUATION OK.")
    return 0


def _train_steps(args) -> str:
    """Read the training step budget back out of a run's config.json."""
    import json

    for algo in args.algos:
        for seed in args.seeds:
            p = os.path.join(args.models_dir, algo, f"seed{seed}", "config.json")
            if os.path.exists(p):
                with open(p) as fh:
                    return f"{json.load(fh)['steps']:,}"
    return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
