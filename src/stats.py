"""Significance layer: paired t-tests vs each baseline, Bonferroni, bootstrap CIs.

251 test days and 10 seeds are two nested sources of variation, and they answer
different questions. DAY level: given this policy, did it beat the baseline over this
window? SEED level: does this ALGORITHM beat the baseline, given that a practitioner
draws one seed? Both are computed; only the seed level tests the claim that seed
variance dominates the algorithm effect.

Everything is deterministic given ``rng``
"""

import numpy as np
import pandas as pd
from scipy import stats as sps

from .metrics import sharpe


def returns(wealth):
    """Daily simple returns from a wealth curve, matching metrics.summarise exactly.

    metrics.py:45 prepends v_0 = 1.0, so day 0's return is wealth[0] - 1, NOT dropped.
    Diverging here would silently desynchronise the stats table from the metrics table
    by one observation. Accepts (T,) or (n_seeds, T); acts on the last axis.
    """
    v = np.asarray(wealth, float)
    prev = np.concatenate([np.ones(v.shape[:-1] + (1,)), v[..., :-1]], axis=-1)
    return v / prev - 1.0


def paired_t(a, b):
    """Paired t-test on daily returns. Both arms see the same day, so pairing differences
    out the market and is far more powerful than an unpaired comparison.

    A zero-variance difference (identical strategies) makes ttest_rel return nan; that is
    'no evidence of a difference', so it is reported as p = 1.0, not propagated as nan.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    if d.std(ddof=1) == 0:
        return {"t": 0.0, "p": 1.0, "mean_diff": float(d.mean()), "n": int(d.size)}
    t, p = sps.ttest_rel(np.asarray(a, float), np.asarray(b, float))
    return {"t": float(t), "p": float(p), "mean_diff": float(d.mean()), "n": int(d.size)}


def _day_idx(T, n_boot, rng, block=1):
    """(n_boot, T) resampled day indices. block > 1 gives a moving-block bootstrap.

    iid over days is the standard choice for a mean difference. Returns are close to
    serially uncorrelated but volatility is clustered, so a Sharpe CI is better served by
    block > 1; the default stays 1 and the choice is the caller's, stated, not assumed.
    """
    if block <= 1:
        return rng.integers(0, T, size=(n_boot, T))
    n_blocks = int(np.ceil(T / block))
    starts = rng.integers(0, T - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)
    return idx[:, :T]


def bootstrap_diff_ci(a, b, n_boot=10000, alpha=0.05, rng=None, block=1):
    """Percentile CI on mean(a - b) over days.

    THE trap: one index vector is drawn and applied to BOTH series. Resampling a and b
    independently destroys the pairing, turns a paired comparison back into an unpaired
    one and inflates the interval enormously -- quietly undoing the whole point of
    pairing. Same rule applies to any per-day covariate (e.g. rf): resample once, apply
    everywhere.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    d = np.asarray(a, float) - np.asarray(b, float)
    idx = _day_idx(d.size, n_boot, rng, block)
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(d.mean()), "lo": float(lo), "hi": float(hi)}


def bootstrap_sharpe_ci(wealth, rf=0.0, periods=252, n_boot=10000, alpha=0.05,
                        rng=None, block=1):
    """CI on annualised Sharpe. rf is resampled with the SAME day indices as the returns
    -- metrics.sharpe subtracts it elementwise, so an unaligned rf is a silent bias."""
    rng = rng if rng is not None else np.random.default_rng(0)
    r = returns(wealth)
    rf_arr = np.broadcast_to(np.asarray(rf, float), r.shape)
    idx = _day_idx(r.size, n_boot, rng, block)
    vals = np.array([sharpe(r[i], rf_arr[i], periods) for i in idx])
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": sharpe(r, rf, periods), "lo": float(lo), "hi": float(hi)}


def seed_level(agent_scalars, baseline_scalar, n_boot=10000, alpha=0.05, rng=None):
    """Is the ALGORITHM better than the baseline, treating each seed as one observation?

    Scope, which must be quoted with the p-value: this conditions on the realised test
    window. The baseline's final wealth is a FIXED known constant here -- over one fixed
    window it is not a random draw -- so the only variation is the seed. The question is
    'does this algorithm's seed distribution sit below the baseline's realised outcome',
    and nothing wider. It is therefore much more powerful than the per-day interval from
    ``nested_ci``, which additionally carries 251 days of ~1% daily volatility, and the
    two WILL disagree: a rejection here beside a per-day CI spanning zero is the expected
    pattern, not a contradiction. Report them as the two different questions they are.

    n = 10, so this is low-powered and the wealth distribution is skewed: the bootstrap
    percentile CI on the seed mean is the summary, the t-test the secondary. Resampling
    10 distinct values gives a visibly lumpy bootstrap distribution -- that is honest
    about how little a 10-seed run pins down, not a defect to smooth away.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    x = np.asarray(agent_scalars, float) - float(baseline_scalar)
    if x.std(ddof=1) == 0:
        t, p = 0.0, 1.0
    else:
        tt = sps.ttest_1samp(x, 0.0)
        t, p = float(tt.statistic), float(tt.pvalue)
    draws = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"t": t, "p": p, "mean_diff": float(x.mean()), "median_diff": float(np.median(x)),
            "lo": float(lo), "hi": float(hi), "n": int(x.size),
            "n_better": int((x > 0).sum())}


def nested_ci(agent_curves, baseline_curve, n_boot=10000, alpha=0.05, rng=None, block=1):
    """CI on the mean daily return difference that carries BOTH sources of variation:
    resample seeds with replacement, then days with replacement within the draw.

    A day-only CI answers 'was this seed lucky in this window'. When seeds disagree as
    much as they do here, that interval is the wrong one to put on an algorithm.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    ra = returns(np.atleast_2d(agent_curves))
    rb = returns(np.asarray(baseline_curve, float))
    n_seeds, T = ra.shape
    s_idx = rng.integers(0, n_seeds, size=(n_boot, n_seeds))
    d_idx = _day_idx(T, n_boot, rng, block)
    d = ra - rb[None, :]
    means = np.array([d[s][:, i].mean() for s, i in zip(s_idx, d_idx)])
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(d.mean()), "lo": float(lo), "hi": float(hi)}


def compare(agent_curves, base_curves, alpha=0.05, n_boot=10000, seed=0, block=1,
            adjusted_alpha=None):
    """Tidy table, one row per (algo, baseline, level).

    levels:
      seed<i>  -- day-level paired test for one seed's policy
      median   -- day-level paired test for the median-final-wealth seed, matching the
                  median convention metrics.csv already uses
      seeds    -- seed-level test: n = n_seeds, the algorithm-vs-baseline question

    ``adjusted_alpha`` additionally records a Bonferroni-width bootstrap interval
    (lo_adj/hi_adj) so a forest plot can draw the corrected whisker as geometry instead
    of a table of asterisks.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for algo, curves in agent_curves.items():
        c = np.atleast_2d(np.asarray(curves, float))
        med_seed = int(np.argsort(c[:, -1])[len(c) // 2])
        for base, bc in base_curves.items():
            b = np.asarray(bc, float).ravel()
            rb = returns(b)
            for i in range(len(c)):
                r = paired_t(returns(c[i]), rb)
                rows.append({"algo": algo, "baseline": base, "level": f"seed{i}", **r,
                             "final_value": float(c[i, -1])})
            r = paired_t(returns(c[med_seed]), rb)
            ci = bootstrap_diff_ci(returns(c[med_seed]), rb, n_boot, alpha, rng, block)
            rows.append({"algo": algo, "baseline": base, "level": "median",
                         "seed_used": med_seed, **r,
                         "lo": ci["lo"], "hi": ci["hi"],
                         "final_value": float(c[med_seed, -1])})
            sl = seed_level(c[:, -1], b[-1], n_boot, alpha, rng)
            nc = nested_ci(c, b, n_boot, alpha, rng, block)
            row = {"algo": algo, "baseline": base, "level": "seeds", **sl,
                   # p/t/mean_diff above are FINAL WEALTH over the fixed window (seed
                   # variation only); ret_* below are per-day (seed + day variation).
                   # Different questions, different power -- see seed_level.__doc__.
                   "test": "final_wealth_vs_realised_baseline",
                   "ret_diff": nc["point"], "ret_lo": nc["lo"], "ret_hi": nc["hi"]}
            if adjusted_alpha is not None:
                adj = nested_ci(c, b, n_boot, adjusted_alpha, rng, block)
                row["ret_lo_adj"], row["ret_hi_adj"] = adj["lo"], adj["hi"]
            rows.append(row)
    return pd.DataFrame(rows)


def bonferroni(df, alpha=0.05, level="seeds"):
    """Correct within the family actually tested: one comparison per (algo, baseline).

    m is counted from the table, never hardcoded, so adding an algorithm or a baseline
    cannot leave a stale divisor behind. Both raw and adjusted p are kept -- the reader
    should see the correction, not only its verdict.

    Note for the interval: a Bonferroni-adjusted CI is the (1 - alpha/m) interval, so at
    alpha=0.05, m=12 the bootstrap tails sit at the 0.21st and 99.79th percentiles. That
    needs n_boot >= ~2000 to be resolvable at all; 10,000 is comfortable.
    """
    out = df.copy()
    fam = out[out["level"] == level]
    m = int(len(fam.groupby(["algo", "baseline"]).size()))
    out["m"] = m
    out["alpha_adj"] = alpha / m
    out["p_adj"] = np.minimum(1.0, out["p"] * m)
    out["reject"] = out["p"] < (alpha / m)
    return out
