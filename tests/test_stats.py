"""Each test pins a specific way the significance layer can be silently wrong.

A null result is only evidence if the machinery can detect a signal that IS there, so
the known-signal cases below gate every null reported from real runs.
"""

import numpy as np
import pytest

from src import metrics as Mx
from src import stats as S


def wealth(rets):
    return np.cumprod(1.0 + np.asarray(rets, float))


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


# ---- returns() convention ----

def test_returns_matches_the_metrics_day_one_convention():
    """metrics.py:45 prepends v_0 = 1.0. Dropping day 0 instead would desynchronise the
    stats table from the metrics table by one observation."""
    v = np.array([1.02, 1.01, 1.05, 1.04])
    rec = {"value": v}
    expected = np.asarray(rec["value"]) / np.concatenate([[1.0], rec["value"][:-1]]) - 1.0
    assert np.allclose(S.returns(v), expected)
    assert len(S.returns(v)) == Mx.summarise(
        {"value": v, "turnover": [0], "hhi": [0], "entropy": [0],
         "max_weight": [0], "mu": [1]})["n_days"]


def test_returns_is_row_wise_on_a_seed_matrix():
    """Broadcasting trap: (n_seeds, T) must act on the last axis, not flatten."""
    c = np.array([[1.01, 1.02, 1.03], [0.99, 0.98, 1.05]])
    got = S.returns(c)
    for i in range(len(c)):
        assert np.allclose(got[i], S.returns(c[i]))


# ---- the pairing trap ----

def test_bootstrap_resamples_the_pair_not_each_series(rng):
    """THE trap. Two series differing by a constant have a constant difference, so the
    paired CI must be degenerate. Resampling a and b independently would give a wide
    interval and quietly convert a paired test into an unpaired one."""
    a = rng.normal(0.001, 0.01, 400)
    b = a - 0.0005
    ci = S.bootstrap_diff_ci(a, b, n_boot=2000, rng=rng)
    assert ci["hi"] - ci["lo"] < 1e-12
    assert abs(ci["point"] - 0.0005) < 1e-12


def test_sharpe_ci_keeps_rf_aligned_with_its_days(rng):
    """rf is subtracted elementwise, so it must ride the same index vector. A constant rf
    is the case where misalignment hides, so use a varying one."""
    w = wealth(rng.normal(0.001, 0.01, 300))
    rf = np.linspace(0.0, 0.0004, 300)
    ci = S.bootstrap_sharpe_ci(w, rf=rf, n_boot=500, rng=rng)
    assert ci["lo"] < ci["point"] < ci["hi"]


# ---- known signal / known null ----

def test_a_real_edge_is_detected(rng):
    """If a 10bp/day edge cannot be found, no null from this module means anything.

    The edge carries its own noise: a CONSTANT offset gives an exactly-constant
    difference, which the t-test only 'detects' via catastrophic cancellation. That
    would pass while testing nothing.
    """
    base = rng.normal(0.0, 0.01, 251)
    agent = base + 0.001 + rng.normal(0.0, 0.002, 251)
    r = S.paired_t(agent, base)
    ci = S.bootstrap_diff_ci(agent, base, n_boot=2000, rng=rng)
    assert r["p"] < 1e-6
    assert ci["lo"] > 0


def test_identical_strategies_give_p_one_not_nan():
    """Zero-variance difference makes ttest_rel return nan; nan is not 'no difference'."""
    a = np.array([0.01, -0.02, 0.03, 0.0])
    r = S.paired_t(a, a)
    assert r["p"] == 1.0 and not np.isnan(r["t"])


def test_no_edge_is_not_rejected(rng):
    base = rng.normal(0.0, 0.01, 251)
    agent = base + rng.normal(0.0, 1e-6, 251)
    assert S.paired_t(agent, base)["p"] > 0.05


# ---- seed level ----

def test_seed_level_counts_and_direction(rng):
    x = np.array([1.10, 1.20, 0.95, 1.30, 1.05, 1.15, 1.25, 0.90, 1.02, 1.18])
    r = S.seed_level(x, 1.0, n_boot=2000, rng=rng)
    assert r["n"] == 10 and r["n_better"] == 8
    assert abs(r["mean_diff"] - (x.mean() - 1.0)) < 1e-12


def test_seed_level_widens_when_seeds_disagree(rng):
    """The point of the seed level: identical means, different spreads, wider interval."""
    tight = np.full(10, 1.10) + rng.normal(0, 0.001, 10)
    loose = np.full(10, 1.10) + rng.normal(0, 0.150, 10)
    a = S.seed_level(tight, 1.0, n_boot=4000, rng=np.random.default_rng(1))
    b = S.seed_level(loose, 1.0, n_boot=4000, rng=np.random.default_rng(1))
    assert (b["hi"] - b["lo"]) > (a["hi"] - a["lo"])


def test_nested_ci_is_wider_than_the_day_only_ci(rng):
    """Seed disagreement must show up as uncertainty about the ALGORITHM."""
    b = wealth(rng.normal(0.0005, 0.01, 251))
    rb = S.returns(b)
    seeds = np.array([wealth(rb + rng.normal(off, 0.002, 251))
                      for off in rng.normal(0.0003, 0.0008, 10)])
    day = S.bootstrap_diff_ci(S.returns(seeds[0]), rb, n_boot=4000,
                              rng=np.random.default_rng(2))
    nest = S.nested_ci(seeds, b, n_boot=4000, rng=np.random.default_rng(2))
    assert (nest["hi"] - nest["lo"]) > (day["hi"] - day["lo"])


# ---- correction and reproducibility ----

def test_bonferroni_m_is_counted_not_hardcoded(rng):
    b = wealth(rng.normal(0.0005, 0.01, 120))
    agents = {"A": np.array([wealth(rng.normal(0.0006, 0.01, 120)) for _ in range(3)]),
              "B": np.array([wealth(rng.normal(0.0004, 0.01, 120)) for _ in range(3)])}
    bases = {"X": b, "Y": wealth(rng.normal(0.0003, 0.01, 120))}
    df = S.bonferroni(S.compare(agents, bases, n_boot=200, seed=0), alpha=0.05)
    assert df["m"].iloc[0] == 4                       # 2 algos x 2 baselines
    assert np.all(df["p_adj"] >= df["p"] - 1e-15)
    assert np.all(df["p_adj"] <= 1.0)


def test_compare_is_reproducible(rng):
    b = wealth(rng.normal(0.0005, 0.01, 120))
    agents = {"A": np.array([wealth(rng.normal(0.0006, 0.01, 120)) for _ in range(4)])}
    bases = {"X": b}
    d1 = S.compare(agents, bases, n_boot=300, seed=7)
    d2 = S.compare(agents, bases, n_boot=300, seed=7)
    assert d1.equals(d2)


def test_compare_covers_every_seed_and_both_levels(rng):
    b = wealth(rng.normal(0.0005, 0.01, 120))
    agents = {"A": np.array([wealth(rng.normal(0.0006, 0.01, 120)) for _ in range(5)])}
    df = S.compare(agents, {"X": b}, n_boot=200, seed=0)
    lv = set(df["level"])
    assert {f"seed{i}" for i in range(5)} <= lv
    assert {"median", "seeds"} <= lv


def test_block_bootstrap_shape_is_exactly_T(rng):
    """A moving-block resample builds ceil(T/block) blocks and must be trimmed back."""
    idx = S._day_idx(251, 50, rng, block=10)
    assert idx.shape == (50, 251)
    assert idx.min() >= 0 and idx.max() < 251
