"""Universe selection rule + 0.70 correlation gate. See docs/IMPLEMENTATION_PLAN.md §0."""

from pathlib import Path
import pandas as pd
import yaml

from .config import PROJECT_ROOT

UNIVERSE_YAML = PROJECT_ROOT / "config" / "universe.yaml"


def load_spec(path=None):
    with open(Path(path) if path else UNIVERSE_YAML, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def sector_order(spec):
    return [s["name"] for s in spec["sectors"]]


def ranked(spec, sector):
    for s in spec["sectors"]:
        if s["name"] == sector:
            return [c["ticker"] for c in s["candidates"]]
    raise KeyError(sector)


def all_tickers(spec):
    return [c["ticker"] for s in spec["sectors"] for c in s["candidates"]]


def sector_of(spec, ticker):
    for s in spec["sectors"]:
        if ticker in [c["ticker"] for c in s["candidates"]]:
            return s["name"]
    raise KeyError(ticker)


# Fixed universes, resolved once by the §0 market-cap rule as of the as_of date and
# frozen here. Order is the declared sector order and must never change: it is the
# EIIE extractor's asset axis.
UNIVERSES = {
    "M8_basket1": ["AAPL", "JPM", "NEE", "XOM", "JNJ", "WMT", "AMZN", "UPS"],
    "M8_basket2": ["MSFT", "BAC", "DUK", "CVX", "UNH", "PG", "TSLA", "HON"],
    "M8_basket3": ["NVDA", "WFC", "SO", "COP", "PFE", "KO", "HD", "UNP"],
    "M4":         ["AAPL", "JPM", "NEE", "XOM"],
    "M16":        ["AAPL", "MSFT", "JPM", "BAC", "NEE", "DUK", "XOM", "CVX",
                   "JNJ", "UNH", "WMT", "PG", "AMZN", "TSLA", "UPS", "HON"],
}
HEADLINE = UNIVERSES["M8_basket1"]


def resolve(spec, n_assets, basket_rank=1):
    if n_assets == 16:
        assert basket_rank == 1, "basket_rank does not apply at M=16"
        return list(UNIVERSES["M16"])
    if n_assets == 4:
        return list(UNIVERSES["M4"])
    if n_assets == 8:
        return list(UNIVERSES[f"M8_basket{basket_rank}"])
    raise ValueError(f"n_assets={n_assets} not in [4, 8, 16]")


def pair_corr(returns):
    """Off-diagonal pairwise correlations, descending."""
    c = returns.corr()
    cols = list(c.columns)
    out = {(a, b): float(c.loc[a, b]) for i, a in enumerate(cols) for b in cols[i + 1:]}
    return pd.Series(out).sort_values(ascending=False)


def apply_gate(spec, tickers, train_returns):
    """Substitute the next-largest name in the affected sector while any pair > 0.70.

    train_returns must cover every candidate so substitution needs no new download.
    Not applied when the universe draws >1 name per sector (M=16): 'next-largest in
    the affected sector' is undefined there, so we measure and report only.
    """
    gate = spec["correlation_gate"]
    thr, order = gate["threshold"], sector_order(spec)
    secs = [sector_of(spec, t) for t in tickers]
    applies = len(set(secs)) == len(secs) and "one_per_sector" in gate["applies_to"]

    cur, subs, rounds = list(tickers), [], []
    for rnd in range(1, gate["max_rounds"] + 1):
        corr = pair_corr(train_returns[cur])
        (a, b), top = corr.index[0], float(corr.iloc[0])
        breaches = [{"pair": list(p), "corr": round(float(v), 6)}
                    for p, v in corr.items() if v > thr]
        rounds.append({"round": rnd, "tickers": list(cur), "max_pair": [a, b],
                       "max_corr": round(top, 6), "breaches": breaches})
        if not breaches or not applies:
            break
        # tie-break: replace the constituent of the later-declared sector
        victim = a if order.index(sector_of(spec, a)) > order.index(sector_of(spec, b)) else b
        sec = sector_of(spec, victim)
        bench = ranked(spec, sec)
        nxt = bench.index(victim) + 1
        if nxt >= len(bench):
            raise ValueError(f"no rank-{nxt + 1} candidate in {sec} to substitute for {victim}")
        rep = bench[nxt]
        subs.append({"round": rnd, "pair": [a, b], "corr": round(top, 6),
                     "sector": sec, "removed": victim, "added": rep})
        cur = [rep if t == victim else t for t in cur]
    else:
        raise ValueError(f"gate did not converge in {gate['max_rounds']} rounds: {cur}")

    last = rounds[-1]
    return {"threshold": thr, "measured_on": "train", "applied": applies,
            "initial_tickers": list(tickers), "final_tickers": cur,
            "max_corr": last["max_corr"], "max_pair": last["max_pair"],
            "passed": not last["breaches"], "substitutions": subs, "rounds": rounds}
