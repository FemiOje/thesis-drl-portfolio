"""Data pipeline for the DRL portfolio-management thesis.

Downloads ~5 years of daily adjusted OHLC for a fixed large-cap S&P 500 universe,
caches the raw data to disk, and
builds the normalized price tensor X_t of Jiang, Xu & Liang (2017)
(arXiv:1706.10059), Eq. 18.

Design notes
* The only function that talks to the external data source is ``_download_raw``.
  Everything downstream is source-agnostic, so switching yfinance to a Microsoft
  data source touches one place.
* Universe is fixed as of the start of the window (Jiang §3.1 survivorship-bias
  fix): the 8 tickers were large-cap S&P 500 constituents at the window start and
  are held constant throughout.
* Chronological split only (never shuffle time): 70% train / 15% val / 15% test,
  i.e. ~3.5y / ~0.75y / ~0.75y. The test window is touched exactly once, at the end.
* Val/test observation windows may reach back across a split boundary for their
  first ~50 steps. This is recent observable history at decision time (not future
  leakage), and preserves ~50 trading days per window.

Run standalone to (re)build the cached dataset and print a summary:
    python data/data_loader.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

# --- Fixed configuration
TICKERS: list[str] = ["AAPL", "MSFT", "JPM", "JNJ", "XOM", "PG", "AMZN", "NVDA"]
FEATURES: list[str] = ["High", "Low", "Close"]  # 3 channels, order matches Jiang Eq. 18
WINDOW: int = 50                                  # n = 50 trading days
START_DATE: str = "2021-07-23"                    # ~5y before 2026-07-23
END_DATE: str = "2026-07-23"                      # yfinance end is exclusive
TRAIN_FRAC: float = 0.70                          # 3.5y / 5y
VAL_FRAC: float = 0.15                            # 0.75y / 5y  (test = remaining 0.15)

# Resolve paths relative to this file so the module works from any CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(_HERE, "raw")
PROCESSED_DIR = os.path.join(_HERE, "processed")


# Data source (swappable)
def _download_raw(
    tickers: list[str], start: str, end: str
) -> dict[str, pd.DataFrame]:
    """Download daily adjusted OHLCV for each ticker. the only function that hits
    the external source. swap the body to change data providers.

    Returns a dict {ticker -> DataFrame[Open, High, Low, Close, Volume]} indexed by
    date. ``auto_adjust=True`` returns split/dividend-adjusted prices (the "adjusted
    OHLC" the plan requires).
    """
    import yfinance as yf

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
    )

    out: dict[str, pd.DataFrame] = {}
    for tkr in tickers:
        # With multiple tickers + group_by="ticker", columns are a MultiIndex
        # (ticker, field). Single-ticker downloads come back flat; handle both.
        if isinstance(raw.columns, pd.MultiIndex):
            df = raw[tkr].copy()
        else:
            df = raw.copy()
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.index.name = "Date"
        out[tkr] = df
    return out


def load_ohlc(
    tickers: list[str] = TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
    raw_dir: str = RAW_DIR,
    force_download: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load per-ticker OHLCV from cached CSVs in ``raw_dir``; download and cache if
    missing (or if ``force_download``). Experiments should read the cache and never
    trigger a network call, guaranteeing reproducibility.
    """
    os.makedirs(raw_dir, exist_ok=True)
    cached = {t: os.path.join(raw_dir, f"{t}.csv") for t in tickers}
    have_all = all(os.path.exists(p) for p in cached.values())

    if have_all and not force_download:
        return {
            t: pd.read_csv(p, index_col="Date", parse_dates=True)
            for t, p in cached.items()
        }

    data = _download_raw(tickers, start, end)
    for t, df in data.items():
        df.to_csv(cached[t])
    return data


# 2. Cleaning + alignment (source-agnostic)
def build_panel(
    ohlc: dict[str, pd.DataFrame], features: list[str] = FEATURES
) -> tuple[np.ndarray, pd.DatetimeIndex, list[str]]:
    """Align all tickers on their common trading calendar, forward-fill isolated
    gaps, and stack into a dense panel of shape ``(T, m, len(features))``.

    Returns ``(panel, dates, tickers)``. Feature order matches ``features``
    ([High, Low, Close]); ``panel[:, :, 2]`` is therefore the close matrix.
    """
    tickers = list(ohlc.keys())

    # Union of all dates, then forward-fill isolated gaps (e.g. a ticker missing a
    # single bar). Leading rows that are still NaN after ffill are dropped so every
    # asset has a real observation from t=0 onward.
    all_dates = sorted(set().union(*[df.index for df in ohlc.values()]))
    dates = pd.DatetimeIndex(all_dates)

    frames = []
    for t in tickers:
        df = ohlc[t].reindex(dates)[features].ffill()
        frames.append(df.to_numpy(dtype=np.float64))  # (T, len(features))

    panel = np.stack(frames, axis=1)  # (T, m, F)

    valid = ~np.isnan(panel).any(axis=(1, 2))
    first_valid = int(np.argmax(valid))  # first fully-observed row
    panel = panel[first_valid:]
    dates = dates[first_valid:]

    if np.isnan(panel).any():
        raise ValueError("NaNs remain after ffill/trim — check the raw data source.")
    return panel, dates, tickers


# 3. Chronological split
@dataclass(frozen=True)
class Splits:
    """Half-open [start, end) panel-index ranges for each chronological split."""

    train: tuple[int, int]
    val: tuple[int, int]
    test: tuple[int, int]


def chronological_split(
    n_days: int, train_frac: float = TRAIN_FRAC, val_frac: float = VAL_FRAC
) -> Splits:
    """Split ``n_days`` trading days chronologically into train/val/test index
    ranges. No shuffling — order is preserved so the test window is strictly the
    most recent period.
    """
    train_end = int(round(n_days * train_frac))
    val_end = int(round(n_days * (train_frac + val_frac)))
    return Splits(
        train=(0, train_end),
        val=(train_end, val_end),
        test=(val_end, n_days),
    )


# =============================================================================
# 4. Jiang price tensor ( Eq. 18)
# =============================================================================
def price_tensor(panel: np.ndarray, t: int, window: int = WINDOW) -> np.ndarray:
    """Normalized price tensor X_t for decision time index ``t`` — Jiang Eq. 18.

    Takes the ``window`` bars ending at day ``t`` (inclusive) and divides every
    price by that asset's *latest* close (the close on day ``t``). Returns shape
    ``(F, window, m)`` = (feature, time, asset), matching Jiang's (3, n, m) tensor.
    The latest-close entry of the close channel is therefore exactly 1 by
    construction.
    """
    if t < window - 1:
        raise IndexError(f"Need {window} days of history; t={t} too small.")
    win = panel[t - window + 1 : t + 1]          # (window, m, F)
    latest_close = panel[t, :, FEATURES.index("Close")]  # (m,)
    normed = win / latest_close[None, :, None]   # broadcast over time & feature
    return np.transpose(normed, (2, 0, 1))       # (F, window, m)


def price_relatives(panel: np.ndarray) -> np.ndarray:
    """Price-relative vector y_t = v_t ⊘ v_{t-1} of closes (Jiang Eq. 1), for stocks
    only (cash's ≡ 1 is added by the environment).

    Returns shape ``(T, m)`` with ``y[0]`` set to 1 (no prior day). ``y[t]`` is the
    gross return realized *over* the period from day ``t-1`` to day ``t``.
    """
    close = panel[:, :, FEATURES.index("Close")]  # (T, m)
    y = np.ones_like(close)
    y[1:] = close[1:] / close[:-1]
    return y


# =============================================================================
# 5. Top-level dataset assembly + cache
# =============================================================================
@dataclass
class Dataset:
    panel: np.ndarray          # (T, m, F) cleaned adjusted OHLC (High, Low, Close)
    dates: pd.DatetimeIndex    # (T,)
    tickers: list[str]         # length m
    y: np.ndarray              # (T, m) close price relatives
    splits: Splits
    window: int

    @property
    def n_assets(self) -> int:
        return len(self.tickers)

    def tensor(self, t: int) -> np.ndarray:
        """X_t for panel index ``t`` (Jiang Eq. 18)."""
        return price_tensor(self.panel, t, self.window)


def build_dataset(
    tickers: list[str] = TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
    window: int = WINDOW,
    force_download: bool = False,
) -> Dataset:
    """End-to-end: load (cached) raw OHLC -> clean/align panel -> price relatives ->
    chronological split. Caches the assembled panel to
    ``data/processed/dataset.npz`` for fast, network-free reloads.
    """
    ohlc = load_ohlc(tickers, start, end, force_download=force_download)
    panel, dates, tickers = build_panel(ohlc)
    y = price_relatives(panel)
    splits = chronological_split(len(dates))

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.savez(
        os.path.join(PROCESSED_DIR, "dataset.npz"),
        panel=panel,
        dates=dates.values.astype("datetime64[D]"),
        tickers=np.array(tickers),
        y=y,
        window=window,
        split_train=np.array(splits.train),
        split_val=np.array(splits.val),
        split_test=np.array(splits.test),
    )
    return Dataset(panel, dates, tickers, y, splits, window)


def _summary(ds: Dataset) -> None:
    print("=== Dataset summary ===")
    print(f"tickers ({ds.n_assets}): {', '.join(ds.tickers)}")
    print(f"features        : {FEATURES}")
    print(f"trading days    : {len(ds.dates)}  "
          f"({ds.dates[0].date()} -> {ds.dates[-1].date()})")
    print(f"panel shape     : {ds.panel.shape}  (T, m, F)")
    print(f"window (n)      : {ds.window}")
    for name, (a, b) in [
        ("train", ds.splits.train), ("val", ds.splits.val), ("test", ds.splits.test)
    ]:
        print(f"  {name:5s}: idx [{a:4d}, {b:4d})  "
              f"{ds.dates[a].date()} -> {ds.dates[b - 1].date()}  ({b - a} days)")
    xt = ds.tensor(len(ds.dates) - 1)
    print(f"sample X_t shape: {xt.shape}  (F, n, m)")
    print(f"  close-channel latest col (should be ~1.0): "
          f"{xt[FEATURES.index('Close'), -1, :].round(4).tolist()}")
    print(f"y range         : [{ds.y[1:].min():.4f}, {ds.y[1:].max():.4f}]  "
          f"(daily close relatives)")
    print("DATA PIPELINE OK.")


if __name__ == "__main__":
    _summary(build_dataset())
