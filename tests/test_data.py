"""Phase 1 gate: tensor construction, causality, splits, selection rule."""

import numpy as np
import pandas as pd
import pytest

from src import data as D
from src import universe as U
from src.config import load_config

CFG = load_config()
SPEC = U.load_spec()


@pytest.fixture(scope="module")
def ds():
    tk = U.resolve(SPEC, 8, 1)
    return D.build_dataset(tk, CFG)


def _synthetic(T=60, M=3):
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2021-09-01", periods=T)
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, .01, (T, M)), 0),
                         index=idx, columns=list("ABC")[:M])
    return {"close": close, "high": close * 1.01, "low": close * 0.99}


# ---- tensor ----

def test_tensor_shape(ds):
    assert ds["X"].shape[1:] == (CFG.env.n_features, CFG.universe.n_assets, CFG.env.window)
    assert len(ds["dates"]) == len(ds["X"])


def test_last_close_column_is_one(ds):
    """Eq. 18 divides by the LATEST close in the window, so channel 0's last column
    is exactly 1. Catches an off-by-one in the normalisation."""
    assert np.allclose(ds["X"][:, 0, :, -1], 1.0, atol=1e-6)


def test_tensor_finite_and_positive(ds):
    assert np.isfinite(ds["X"]).all()
    assert (ds["X"] > 0).all()


def test_high_ge_close_ge_low(ds):
    X = ds["X"]
    assert (X[:, 1] >= X[:, 0] - 1e-6).all()
    assert (X[:, 2] <= X[:, 0] + 1e-6).all()


def test_tensor_matches_direct_computation():
    p = _synthetic()
    w = 20
    X, dates = D.build_tensor(p, w)
    c = p["close"].values
    for i in (0, 5, len(X) - 1):
        t = i + w - 1
        assert np.allclose(X[i, 0], (c[t - w + 1:t + 1] / c[t]).T)
        assert dates[i] == p["close"].index[t]


def test_lookback_is_causal():
    """X[i] must be invariant to every bar strictly after its decision bar."""
    p = _synthetic()
    w = 20
    X, _ = D.build_tensor(p, w)
    i = 10
    t = i + w - 1
    p2 = {k: v.copy() for k, v in p.items()}
    for k in p2:
        p2[k].iloc[t + 1:] *= 3.0            # corrupt the entire future
    X2, _ = D.build_tensor(p2, w)
    assert np.allclose(X[i], X2[i])


# ---- price relatives ----

def test_price_relatives_cash_and_alignment():
    p = _synthetic()
    y = D.price_relatives(p)
    assert y.columns[0] == "CASH"
    assert (y["CASH"] == 1.0).all()
    c = p["close"]
    assert np.allclose(y["A"].values, (c["A"] / c["A"].shift(1)).iloc[1:].values)


# ---- splits ----

def test_splits_by_decision_date(ds):
    d = pd.DatetimeIndex(ds["dates"])
    for s in CFG.data.splits:
        idx = ds["splits"][s.name]
        assert len(idx) > 0
        assert (d[idx] >= pd.Timestamp(s.start)).all()
        assert (d[idx] <= pd.Timestamp(s.end)).all()


def test_splits_disjoint_and_ordered(ds):
    tr, va, te = (ds["splits"][k] for k in ("train", "validate", "test"))
    assert tr[-1] < va[0] < va[-1] < te[0]
    assert len(set(tr) | set(va) | set(te)) == len(tr) + len(va) + len(te)


def test_splits_are_contiguous(ds):
    """Not re-windowed per split: decision indices run without gaps across the joins,
    so a validation lookback legitimately reaches back into training bars."""
    tr, va, te = (ds["splits"][k] for k in ("train", "validate", "test"))
    assert va[0] == tr[-1] + 1
    assert te[0] == va[-1] + 1


# ---- universe rule ----

def test_resolve_one_per_sector():
    tk = U.resolve(SPEC, 8, 1)
    assert len(tk) == 8
    assert len({U.sector_of(SPEC, t) for t in tk}) == 8


def test_baskets_are_disjoint():
    b = [set(U.resolve(SPEC, 8, r)) for r in (1, 2, 3)]
    assert not (b[0] & b[1]) and not (b[1] & b[2]) and not (b[0] & b[2])


def test_scale_universes():
    assert len(U.resolve(SPEC, 4, 1)) == 4
    assert set(U.resolve(SPEC, 4, 1)) <= set(U.resolve(SPEC, 8, 1))
    m16 = U.resolve(SPEC, 16, 1)
    assert len(m16) == 16 and len(set(m16)) == 16
    assert set(U.resolve(SPEC, 8, 1)) <= set(m16)


def test_gate_substitutes_when_breached():
    """Force a breach with two perfectly correlated series; the later-declared
    sector's constituent must be the one replaced."""
    tk = U.resolve(SPEC, 8, 1)
    order = U.sector_order(SPEC)
    a, b = tk[0], tk[1]
    later = a if order.index(U.sector_of(SPEC, a)) > order.index(U.sector_of(SPEC, b)) else b
    rng = np.random.default_rng(0)
    cols = U.all_tickers(SPEC)
    r = pd.DataFrame(rng.normal(0, .01, (500, len(cols))), columns=cols)
    r[b] = r[a]                                   # correlation 1.0
    g = U.apply_gate(SPEC, tk, r)
    assert g["substitutions"], "gate failed to fire on a perfectly correlated pair"
    assert g["substitutions"][0]["removed"] == later
    assert later not in g["final_tickers"]


def test_gate_reports_without_substituting_at_m16():
    tk = U.resolve(SPEC, 16, 1)
    cols = U.all_tickers(SPEC)
    rng = np.random.default_rng(1)
    r = pd.DataFrame(rng.normal(0, .01, (500, len(cols))), columns=cols)
    r[tk[1]] = r[tk[0]]
    g = U.apply_gate(SPEC, tk, r)
    assert g["applied"] is False
    assert g["substitutions"] == []
    assert g["final_tickers"] == tk
