"""Sanity plots for the data pipeline (runnable mirror of
notebooks/01_data_sanity.ipynb).

Generates three figures under results/data_checks/:
  1. adjusted close prices (raw scale) for the 8-stock universe, with split lines;
  2. a heatmap of one normalized price tensor X_t (close channel), verifying the
     latest column is ~1.0 (Jiang Eq. 18);
  3. normalized close trajectories inside a single 50-day window for all assets.

Run:  python notebooks/plot_tensors.py
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless; write PNGs without a display
import matplotlib.pyplot as plt

# Make the sibling data/ package importable when run as a script.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "data"))
import data_loader as dl  # noqa: E402

OUT_DIR = os.path.join(_ROOT, "results", "data_checks")


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    ds = dl.build_dataset()
    close = ds.panel[:, :, dl.FEATURES.index("Close")]  # (T, m)

    # --- Fig 1: adjusted closes with chronological split boundaries ---------
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, t in enumerate(ds.tickers):
        ax.plot(ds.dates, close[:, i], label=t, lw=1)
    for b, lbl in [(ds.splits.train[1], "train|val"), (ds.splits.val[1], "val|test")]:
        ax.axvline(ds.dates[b], color="k", ls="--", lw=1)
        ax.text(ds.dates[b], ax.get_ylim()[1], lbl, fontsize=8, ha="center", va="bottom")
    ax.set_title("Adjusted close — 8-stock universe (chronological splits)")
    ax.set_ylabel("adjusted close (USD)")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "01_prices.png"), dpi=110)
    plt.close(fig)

    # --- Fig 2: normalized tensor heatmap (close channel) at last valid t ---
    t = len(ds.dates) - 1
    xt = ds.tensor(t)  # (F, n, m)
    close_ch = xt[dl.FEATURES.index("Close")]  # (n, m)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(close_ch.T, aspect="auto", cmap="viridis", origin="lower")
    ax.set_title(f"Normalized close channel of X_t  (t={ds.dates[t].date()}, Eq. 18)")
    ax.set_xlabel("day within 50-step window (49 = latest)")
    ax.set_ylabel("asset")
    ax.set_yticks(range(ds.n_assets))
    ax.set_yticklabels(ds.tickers)
    fig.colorbar(im, ax=ax, label="price / latest close")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "02_tensor_heatmap.png"), dpi=110)
    plt.close(fig)

    # --- Fig 3: normalized close trajectories within the window -------------
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, tkr in enumerate(ds.tickers):
        ax.plot(range(ds.window), close_ch[:, i], label=tkr, lw=1.2)
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_title("Normalized close within the latest 50-day window (all ~1.0 at day 49)")
    ax.set_xlabel("day within window")
    ax.set_ylabel("price / latest close")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "03_window_normalized.png"), dpi=110)
    plt.close(fig)

    # Numeric sanity assertion: latest close column must be 1.0 for every asset.
    latest = close_ch[-1]
    assert abs(latest - 1.0).max() < 1e-9, "Eq. 18 normalization broken!"
    print(f"Saved 3 figures to {OUT_DIR}")
    print(f"Eq. 18 check OK: latest-close column max deviation from 1.0 = "
          f"{abs(latest - 1.0).max():.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
