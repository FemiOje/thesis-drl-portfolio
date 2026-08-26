"""Performance metrics. Sharpe is annualised (x sqrt(252))."""

import numpy as np


def drawdown(value):
    v = np.asarray(value, float)
    return v / np.maximum.accumulate(v) - 1.0


def max_drawdown(value):
    """MDD, Eq. 29. Returned positive."""
    return float(-drawdown(value).min())


def cumulative_return(value):
    return float(value[-1] / value[0] - 1.0)


def annualised_return(value, periods=252):
    v = np.asarray(value, float)
    return float((v[-1] / v[0]) ** (periods / len(v)) - 1.0)


def sharpe(returns, rf=0.0, periods=252):
    r = np.asarray(returns, float) - np.asarray(rf, float)
    s = r.std(ddof=1)
    return float(r.mean() / s * np.sqrt(periods)) if s > 0 else 0.0


def sortino(returns, rf=0.0, periods=252):
    r = np.asarray(returns, float) - np.asarray(rf, float)
    d = r[r < 0].std(ddof=1) if (r < 0).sum() > 1 else 0.0
    return float(r.mean() / d * np.sqrt(periods)) if d > 0 else 0.0


def win_rate(returns):
    r = np.asarray(returns, float)
    return float((r > 0).mean())


def summarise(record, rf=0.0, periods=252):
    """record: the dict returned by backtest()."""
    v = np.concatenate([[1.0], record["value"]])
    ret = np.asarray(record["value"], float) / np.concatenate([[1.0], record["value"][:-1]]) - 1.0
    return {
        "final_value": float(record["value"][-1]),
        "CR": cumulative_return(v),
        "AR": annualised_return(v, periods),
        "sharpe": sharpe(ret, rf, periods),
        "sortino": sortino(ret, rf, periods),
        "MDD": max_drawdown(v),
        "turnover": float(np.mean(record["turnover"])),
        "win_rate": win_rate(ret),
        "HHI": float(np.mean(record["hhi"])),
        "entropy": float(np.mean(record["entropy"])),
        "max_weight": float(np.max(record["max_weight"])),
        "mu_min": float(np.min(record["mu"])),
        "n_days": len(ret),
    }
