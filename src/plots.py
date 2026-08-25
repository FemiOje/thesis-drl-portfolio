"""Figure style + F0a-F0c. Extended in Phase 3.5 with F5-F8."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({"figure.dpi": 140, "savefig.bbox": "tight", "font.size": 9})

STRATEGY_COLORS = {
    "PG": "#d62728", "PPO": "#1f77b4", "DDPG": "#2ca02c",
    "UBAH": "#7f7f7f", "UCRP": "#9467bd", "BestStock": "#8c564b", "Markowitz": "#e377c2",
}


LOWER_IS_BETTER = {"MDD", "turnover"}


def save(fig, out_dir, name, run_id=""):
    """PNG for the thesis, PDF for vector. run_id stamped in both files' metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        key = "Title" if ext == "pdf" else "Software"
        fig.savefig(out_dir / f"{name}.{ext}", metadata={key: f"{name} run_id={run_id}"})
    plt.close(fig)
    return out_dir / f"{name}.png"


def style_for(name):
    """BestStock is dashed: it is a hindsight upper reference, not a strategy."""
    return dict(color=STRATEGY_COLORS.get(name, "#333333"),
                ls="--" if name == "BestStock" else "-",
                lw=1.4 if name in ("PG", "PPO", "DDPG") else 1.1)


def log_y_ticks(ax, nbins=8):
    """Wealth spans far less than a decade, where LogLocator gives either one tick or a
    crush of decade subdivisions — and sometimes none at all. Pick round ticks over the
    realised range instead. Every log wealth axis in the suite goes through here."""
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    ticks = matplotlib.ticker.MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
    keep = [v for v in ticks if lo <= v <= hi]
    if len(keep) < 2:                       # degenerate range: fall back to the ends
        keep = [lo, (lo * hi) ** 0.5, hi]
    ax.set_yticks(keep)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(lo, hi)


def wealth_axis(ax, ylabel="portfolio value  $p_t/p_0$"):
    log_y_ticks(ax)
    ax.axhline(1.0, color="k", lw=0.6, alpha=0.4)
    ax.set_ylabel(ylabel)


def _band(ax, x, curves, name):
    """Median with IQR across seeds. One seed (baselines) draws a bare line."""
    c = np.atleast_2d(curves)
    st = style_for(name)
    ax.plot(x, np.median(c, 0), label=name, **st)
    if len(c) > 1:
        ax.fill_between(x, np.percentile(c, 25, 0), np.percentile(c, 75, 0),
                        color=st["color"], alpha=0.18, lw=0)


def f5_test_wealth(dates, curves, out_dir, run_id=""):
    """THE headline figure: wealth on the test split, log-y."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for name, c in curves.items():
        _band(ax, dates, c, name)
    wealth_axis(ax)
    ax.set_title("F5  Test-split wealth, net of transaction costs")
    ax.legend(ncol=4, fontsize=8)
    return save(fig, out_dir, "F5_test_wealth", run_id)


def f6_train_val_wealth(dates, curves, out_dir, run_id=""):
    """In-sample fit against the generalisation gap."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, split in zip(axes, ("train", "validate")):
        for name, c in curves[split].items():
            _band(ax, dates[split], c, name)
        wealth_axis(ax)
        ax.set_title(split)
        ax.tick_params(axis="x", rotation=30)
    axes[1].legend(ncol=2, fontsize=8)
    fig.suptitle("F6  Train and validation wealth", y=1.02)
    return save(fig, out_dir, "F6_train_val_wealth", run_id)


def f7_wealth_and_drawdown(dates, curves, out_dir, run_id=""):
    """Wealth with drawdown beneath, sharing the x-axis (Filos's pairing)."""
    from .metrics import drawdown
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    for name, c in curves.items():
        _band(a1, dates, c, name)
        dd = np.atleast_2d(np.array([drawdown(np.r_[1.0, s])[1:] for s in np.atleast_2d(c)]))
        _band(a2, dates, dd * 100, name)
    wealth_axis(a1)
    a1.set_title("F7  Test wealth and drawdown")
    a1.legend(ncol=4, fontsize=8)
    a2.set_ylabel("drawdown (%)")
    a2.axhline(0, color="k", lw=0.6, alpha=0.4)
    return save(fig, out_dir, "F7_drawdown", run_id)


def normalise_columns(df):
    """Per-column 0-1 scaling, inverted where lower is better, so darker is always
    better regardless of the metric's direction."""
    z = df.copy().astype(float)
    for col in z.columns:
        v = z[col].values
        span = v.max() - v.min()
        s = np.full_like(v, 0.5) if span == 0 else (v - v.min()) / span
        z[col] = 1.0 - s if col in LOWER_IS_BETTER else s
    return z


def f8_metric_heatmap(metrics, out_dir, run_id="", split="test"):
    """Metric table as a heatmap, colour-scaled per column."""
    cols = ["CR", "AR", "sharpe", "MDD", "turnover", "win_rate"]
    df = metrics[metrics.split == split].set_index("strategy")[cols]
    fig, ax = plt.subplots(figsize=(7.4, 0.55 * len(df) + 2.0))
    labels = df.applymap(lambda v: f"{v:.3f}")
    sns.heatmap(normalise_columns(df), annot=labels, fmt="", cmap="YlGnBu",
                cbar=False, linewidths=0.5, ax=ax, annot_kws={"size": 8})
    ax.set_title(f"F8  {split}-split metrics (darker = better; MDD and turnover "
                 "inverted)" + "\n" +
                 "turnover is the mean per-step value; Sharpe annualised x sqrt(252)",
                 fontsize=9)
    ax.set_ylabel("")
    ax.tick_params(axis="y", rotation=0)
    return save(fig, out_dir, f"F8_metrics_{split}", run_id)


def f0a_prices(close, splits, out_dir, run_id=""):
    """Normalised price series with split boundaries."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    norm = close / close.iloc[0]
    for t in norm.columns:
        ax.plot(norm.index, norm[t], lw=1.1, label=t)
    for s in splits[1:]:
        ax.axvline(s.start, color="k", ls="--", lw=0.9, alpha=0.7)
    lo = norm.min().min()
    for s in splits:
        mid = s.start + (s.end - s.start) / 2
        ax.text(mid, lo, s.name, ha="center", va="bottom", fontsize=8, alpha=0.6)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylabel("price / price at start (log)")
    ax.set_title("F0a  Normalised adjusted close, split boundaries marked")
    ax.legend(ncol=4, fontsize=8)
    return save(fig, out_dir, "F0a_prices", run_id)


def f0b_correlation(train_returns, threshold, out_dir, run_id=""):
    """Train-split return correlation with the gate threshold annotated."""
    c = train_returns.corr()
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sns.heatmap(c, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
                square=True, cbar_kws={"label": "correlation"}, ax=ax,
                annot_kws={"size": 7})
    off = c.where(~np.eye(len(c), dtype=bool)).stack()
    ax.set_title(f"F0b  Train-split daily return correlation\n"
                 f"gate = {threshold:.2f}   max off-diagonal = {off.max():.3f} "
                 f"({off.idxmax()[0]}-{off.idxmax()[1]})")
    return save(fig, out_dir, "F0b_correlation", run_id)


def f0c_tensor(X, tickers, dates, idx, out_dir, run_id=""):
    """One input window, three feature channels."""
    names = ["close / latest close", "high / latest close", "low / latest close"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for k, ax in enumerate(axes):
        sns.heatmap(X[idx, k], cmap="viridis", ax=ax, cbar=k == 2,
                    yticklabels=list(tickers) if k == 0 else False)
        if k == 0:
            ax.set_yticklabels(list(tickers), rotation=0, fontsize=8)
        ax.set_title(names[k], fontsize=9)
        ax.set_xlabel("lookback step")
    fig.suptitle(f"F0c  Input tensor {tuple(X.shape[1:])} at decision date "
                 f"{str(dates[idx].date())}", y=1.04)
    return save(fig, out_dir, "F0c_tensor", run_id)


# ---- F1-F4: training diagnostics (Phase 4 onward) ----

def _seed_band(ax, x, c, color, label=None, seeds=True):
    """Median + IQR with the individual seeds faint behind."""
    c = np.atleast_2d(c)
    if seeds and len(c) > 1:
        for s in c:
            ax.plot(x, s, color=color, lw=0.5, alpha=0.25)
        ax.fill_between(x, np.percentile(c, 25, 0), np.percentile(c, 75, 0),
                        color=color, alpha=0.18, lw=0)
    ax.plot(x, np.median(c, 0), color=color, lw=1.5, label=label)


def _mean_log_return(wealth, n_days):
    """Episode wealth -> mean log return per day, the quantity PG maximises."""
    return np.log(np.asarray(wealth, float)) / n_days


def f1_learning_curves(hist, n_days, out_dir, run_id=""):
    """Mean log return per episode vs gradient step, one panel per algorithm."""
    fig, axes = plt.subplots(1, len(hist), figsize=(5.2 * len(hist), 4.0), squeeze=False)
    for ax, (algo, h) in zip(axes[0], hist.items()):
        for split, ls in (("train", "-"), ("validate", "--")):
            c = _mean_log_return(h[split], n_days[split])
            col = STRATEGY_COLORS.get(algo, "#333") if split == "train" else "#555"
            _seed_band(ax, h["eval_step"], c, col, label=split)
            ax.lines[-1].set_linestyle(ls)
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
        ax.set_title(algo)
        ax.set_xlabel("gradient step")
        ax.set_ylabel("mean log return per day")
        ax.legend(fontsize=8)
    fig.suptitle("F1  Learning curves (median, IQR, individual seeds)", y=1.02)
    return save(fig, out_dir, "F1_learning_curves", run_id)


def f2_train_val_wealth_vs_step(hist, out_dir, run_id="", ucrp=None):
    """Final wealth on train and validation vs gradient step. The gap opening is the
    overfitting signature; the selected checkpoint is marked."""
    fig, axes = plt.subplots(1, len(hist), figsize=(5.2 * len(hist), 4.0), squeeze=False)
    for ax, (algo, h) in zip(axes[0], hist.items()):
        _seed_band(ax, h["eval_step"], h["train"], "#1f77b4", label="train")
        _seed_band(ax, h["eval_step"], h["validate"], "#d62728", label="validate")
        best = np.atleast_1d(h["best_step"])
        ax.axvline(np.median(best), color="k", ls=":", lw=1.2,
                   label=f"selected (median step {int(np.median(best))})")
        if ucrp:
            for split, v in ucrp.items():
                ax.axhline(v, color="#9467bd", lw=0.8, alpha=0.7)
                ax.text(ax.get_xlim()[1], v, f" UCRP {split}", fontsize=7,
                        va="center", color="#9467bd")
        log_y_ticks(ax, nbins=10)
        ax.set_title(algo)
        ax.set_xlabel("gradient step")
        ax.set_ylabel("final wealth $p_T/p_0$")
        ax.legend(fontsize=8)
    fig.suptitle("F2  Train vs validation wealth, checkpoint selected on validation",
                 y=1.02)
    return save(fig, out_dir, "F2_train_val_vs_step", run_id)


def f3_loss(hist, out_dir, run_id="", smooth=500):
    """PG has one objective and no critic, so the loss curve is the negated Eq. 21
    batch objective. PPO and DDPG add their own panels in Phase 5."""
    fig, axes = plt.subplots(1, len(hist), figsize=(5.2 * len(hist), 3.8), squeeze=False)
    for ax, (algo, h) in zip(axes[0], hist.items()):
        r = np.atleast_2d(h["reward"])
        k = np.ones(smooth) / smooth
        sm = np.array([np.convolve(s, k, mode="valid") for s in r])
        x = np.arange(len(sm[0])) + smooth
        # seeds omitted: batch-to-batch variance is set by which 50 days the batch
        # landed on, and the spaghetti hides the trend this figure exists to show
        _seed_band(ax, x, sm, STRATEGY_COLORS.get(algo, "#333"), seeds=False)
        ax.fill_between(x, sm.min(0), sm.max(0),
                        color=STRATEGY_COLORS.get(algo, "#333"), alpha=0.15, lw=0)
        ax.set_title(f"{algo} — batch objective (Eq. 21), {smooth}-step mean")
        ax.set_xlabel("gradient step")
        ax.set_ylabel("mean log return per step")
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    fig.suptitle("F3  Loss curves", y=1.03)
    return save(fig, out_dir, "F3_loss", run_id)


def f4_plateau(hist, out_dir, run_id="", win=5, frac=0.10):
    """Rolling slope of the validation curve. 'It plateaued' becomes quantitative:
    the band is |slope| < frac x the largest slope seen, and the marked step is where
    the selected checkpoint was taken."""
    fig, axes = plt.subplots(1, len(hist), figsize=(5.2 * len(hist), 3.8), squeeze=False)
    for ax, (algo, h) in zip(axes[0], hist.items()):
        x, v = np.asarray(h["eval_step"], float), np.atleast_2d(h["validate"])
        if len(x) <= win:
            ax.text(0.5, 0.5, "too few evaluations", ha="center", transform=ax.transAxes)
            continue
        sl = np.array([(s[win:] - s[:-win]) / (x[win:] - x[:-win]) for s in v])
        xs = x[win:]
        med = np.median(sl, 0)
        thr = frac * np.abs(med).max()
        _seed_band(ax, xs, sl * 1e6, STRATEGY_COLORS.get(algo, "#333"))
        ax.axhspan(-thr * 1e6, thr * 1e6, color="k", alpha=0.10,
                   label=f"|slope| < {frac:.0%} of max")
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
        best = int(np.median(np.atleast_1d(h["best_step"])))
        ax.axvline(best, color="k", ls=":", lw=1.2, label=f"selected step {best}")
        flat = xs[np.abs(med) < thr]
        if len(flat):
            ax.axvline(flat[0], color="#d62728", ls="--", lw=1.0,
                       label=f"plateau from {int(flat[0])}")
        ax.set_title(algo)
        ax.set_xlabel("gradient step")
        ax.set_ylabel(r"d(val wealth)/d(step)  $\times 10^{-6}$")
        ax.legend(fontsize=7)
    fig.suptitle("F4  Plateau diagnostic on the validation curve", y=1.03)
    return save(fig, out_dir, "F4_plateau", run_id)
