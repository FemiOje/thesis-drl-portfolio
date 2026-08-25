"""Download, cache, tensor build, split by decision date."""

import json
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def _patch_yfinance():
    """yfinance 0.2.54's cookie/crumb bootstrap (fc.yahoo.com, /v1/test/getcrumb) is
    blocked here and it misreports the failure as a rate limit. The v8 chart endpoint
    needs no crumb, so skip the bootstrap. curl_cffi (yfinance >=0.2.55) is blocked by
    Windows Application Control on this machine, hence the 0.2.54 pin."""
    from yfinance import data as ydata
    ydata.YfData._get_cookie_and_crumb = lambda self, proxy=None, timeout=30: (None, None, "basic")
    # Yahoo also 429s yfinance's stock Edge user agent; a current Chrome UA is served.
    ydata.YfData().user_agent_headers = {"User-Agent": UA}


def download(tickers, start, end, raw_dir, force=False):
    """One CSV per ticker in raw_dir, auto_adjust=True. Cached; idempotent."""
    import yfinance as yf
    _patch_yfinance()
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = {}
    for t in tickers:
        path = raw_dir / f"{t}.csv"
        if path.exists() and not force:
            rows[t] = len(pd.read_csv(path))
            continue
        for attempt in range(6):   # Yahoo rate-limits bursts; back off and retry
            df = yf.download(t, start=str(start), end=str(end + timedelta(days=1)),
                             auto_adjust=True, progress=False, actions=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[COLUMNS].dropna() if len(df) else df
            if len(df):
                break
            time.sleep(15 * (attempt + 1))
        else:
            raise RuntimeError(f"{t}: no data returned after 6 attempts")
        df.index.name = "Date"
        df.to_csv(path, float_format="%.10f")
        rows[t] = len(df)
        time.sleep(1.5)          # be polite; bursts trigger 429
    return rows


def load_panel(tickers, raw_dir, start, end):
    """Aligned close/high/low frames on the intersection of trading calendars."""
    frames = {}
    for t in tickers:
        df = pd.read_csv(raw_dir / f"{t}.csv", index_col="Date", parse_dates=True)
        frames[t] = df.loc[str(start):str(end)]
    idx = frames[tickers[0]].index
    for t in tickers[1:]:
        idx = idx.intersection(frames[t].index)
    out = {}
    for field in ("Close", "High", "Low"):
        out[field.lower()] = pd.DataFrame(
            {t: frames[t].loc[idx, field] for t in tickers}, index=idx)[tickers]
    return out


def build_tensor(panel, window):
    """X[t] = (3, M, window), each channel divided by the latest close in the window
    (Eq. 18). Built over the continuous series; t indexes the decision bar.

    Returns X (T-window+1, 3, M, window) and the decision dates.
    """
    close, high, low = panel["close"].values, panel["high"].values, panel["low"].values
    T, M = close.shape
    n = T - window + 1
    X = np.empty((n, 3, M, window), dtype=np.float32)
    for i in range(n):
        t = i + window - 1                      # decision bar, inclusive
        sl = slice(t - window + 1, t + 1)
        denom = close[t]                        # (M,) latest close
        X[i, 0] = (close[sl] / denom).T
        X[i, 1] = (high[sl] / denom).T
        X[i, 2] = (low[sl] / denom).T
    dates = panel["close"].index[window - 1:]
    return X, dates


def price_relatives(panel):
    """y[t] = close[t] / close[t-1], cash (1.0) prepended at index 0 (Eq. 5).

    y[t] is the relative realised over t-1 -> t, so the reward for the decision at
    t-1 uses y[t].
    """
    close = panel["close"]
    y = (close / close.shift(1)).iloc[1:]
    cash = pd.DataFrame(1.0, index=y.index, columns=["CASH"])
    return pd.concat([cash, y], axis=1)


def split_indices(dates, splits):
    """Map each split's date range onto decision indices. Splits by DECISION DATE,
    not by re-windowing: X[i]'s lookback legitimately reaches into the prior split
    and never past its own decision bar."""
    d = pd.DatetimeIndex(dates)
    out = {}
    for s in splits:
        m = (d >= pd.Timestamp(s.start)) & (d <= pd.Timestamp(s.end))
        out[s.name] = np.flatnonzero(m)
    return out


def build_dataset(tickers, cfg, raw_dir=None):
    raw_dir = raw_dir or cfg.data.raw_dir
    panel = load_panel(tickers, raw_dir, cfg.data.start, cfg.data.end)
    X, dates = build_tensor(panel, cfg.env.window)
    y = price_relatives(panel)
    # y is indexed one bar ahead of the decision; align to decision dates.
    y_dec = y.reindex(dates).values.astype(np.float32)
    return {"X": X, "dates": dates, "y": y_dec, "tickers": list(tickers),
            "panel": panel, "splits": split_indices(dates, cfg.data.splits)}


def save_dataset(ds, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, X=ds["X"], y=ds["y"],
        dates=np.array([str(d.date()) for d in ds["dates"]]),
        tickers=np.array(ds["tickers"]),
        **{f"split_{k}": v for k, v in ds["splits"].items()})


def write_manifest(path, tickers, rows, gate, extra):
    import yfinance as yf
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "retrieved": datetime.now().isoformat(timespec="seconds"),
        "source": "yfinance",
        "yfinance_version": yf.__version__,
        "auto_adjust": True,
        "tickers": list(tickers),
        "rows_per_ticker": rows,
        "correlation_gate": gate,
        **extra,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload
