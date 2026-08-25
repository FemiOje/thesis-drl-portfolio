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


def save(fig, out_dir, name, run_id=""):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", metadata={"Title": f"{name} run_id={run_id}"}
                    if ext == "pdf" else None)
    plt.close(fig)
    return out_dir / f"{name}.png"


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
