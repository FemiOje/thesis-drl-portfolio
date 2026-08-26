"""Phase 3 gate: baselines run through the same backtest loop and pay real costs."""

import numpy as np
import pytest

from src import baselines as B
from src import data as D
from src import metrics as Mx
from src import universe as U
from src.backtest import backtest
from src.config import load_config
from src.env import PortfolioEnv, project_to_simplex

CFG = load_config()
M, TAU = CFG.universe.n_assets, CFG.env.tau


@pytest.fixture(scope="module")
def ds():
    return D.build_dataset(U.HEADLINE, CFG)


def run(ds, policy, split="validate"):
    idx = ds["splits"][split]
    return backtest(PortfolioEnv(ds["X"], ds["y"], CFG, idx), policy)


# ---- action inversion ----

def test_equal_weights_invert_exactly():
    w = np.full(M + 1, 1.0 / (M + 1))
    assert np.allclose(project_to_simplex(B.weights_to_action(w, TAU), TAU), w, atol=1e-12)


def test_inversion_round_trips_representable_weights():
    rng = np.random.default_rng(0)
    for _ in range(200):
        w = rng.dirichlet(np.ones(M + 1) * 2.0)          # ratios well inside exp(2 tau)
        got = project_to_simplex(B.weights_to_action(w, TAU), TAU)
        assert np.abs(got - w).max() < 1e-9


def test_one_hot_is_reachable_to_within_the_floor():
    w = np.zeros(M + 1)
    w[3] = 1.0
    got = project_to_simplex(B.weights_to_action(w, TAU), TAU)
    assert got[3] > 0.999
    assert np.abs(got - w).max() < 1e-3


def test_action_stays_inside_the_bounds():
    rng = np.random.default_rng(1)
    for _ in range(200):
        a = B.weights_to_action(rng.dirichlet(np.ones(M + 1) * 0.3), TAU)
        assert (a >= -1.0).all() and (a <= 1.0).all()


# ---- observation-only price relatives ----

def test_y_from_obs_matches_the_dataset(ds):
    idx = ds["splits"]["validate"]
    env = PortfolioEnv(ds["X"], ds["y"], CFG, idx)
    obs, _ = env.reset()
    for t in range(50):
        y_true = ds["y"][idx[t]].astype(np.float64)
        assert np.abs(B.y_from_obs(obs) - y_true).max() < 1e-4
        obs, _, done, _, _ = env.step(np.zeros(M + 1))
        if done:
            break


# ---- individual baselines ----

def test_ubah_trades_once_and_then_never(ds):
    rec = run(ds, B.UBAH(M, TAU))
    assert rec["turnover"][0] > 1.5                       # cash -> equal weights
    # Holding means targeting the drift, recovered from a float32 observation, so the
    # residual is ~1e-7 rather than 0. See y_from_obs.
    assert rec["turnover"][1:].max() < 1e-6
    assert np.allclose(rec["mu"][1:], 1.0, atol=1e-9)
    assert 1.0 - np.prod(rec["mu"][1:]) < 1e-7            # total leak from holding


def test_ucrp_holds_equal_weights_every_step(ds):
    rec = run(ds, B.UCRP(M, TAU))
    assert np.abs(rec["weights"] - 1.0 / (M + 1)).max() < 1e-9
    assert np.allclose(rec["hhi"], 1.0 / (M + 1))


def test_ucrp_matches_the_env_constant_action(ds):
    """The two routes into the same policy must agree exactly."""
    a = run(ds, B.UCRP(M, TAU))
    b = run(ds, lambda obs: np.zeros(M + 1))
    assert abs(a["value"][-1] - b["value"][-1]) < 1e-12


def test_best_stock_picks_the_hindsight_winner(ds):
    idx = ds["splits"]["test"]
    close = ds["panel"]["close"].values[idx]
    best = int(np.argmax(close[-1] / close[0]))
    pol = B.BestStock(M, TAU, close)
    assert pol.best == best
    rec = run(ds, pol, "test")
    assert rec["weights"][:, best + 1].min() > 0.99
    assert rec["turnover"][1:].max() < 1e-3               # buy and hold
    assert 1.0 - np.prod(rec["mu"][1:]) < 1e-6


def test_best_stock_beats_ucrp_in_hindsight(ds):
    """It is an upper reference, not a strategy. If it loses, something is wrong."""
    a = run(ds, B.BestStock(M, TAU, ds["panel"]["close"].values[ds["splits"]["test"]]), "test")
    b = run(ds, B.UCRP(M, TAU), "test")
    assert a["value"][-1] > b["value"][-1]


def test_markowitz_is_long_only_and_rebalances_monthly(ds):
    idx = ds["splits"]["validate"]
    rec = run(ds, B.Markowitz(M, TAU, ds["panel"]["close"].values, idx))
    assert (rec["weights"] >= 0).all()
    traded = np.flatnonzero(rec["turnover"] > 1e-3)      # above the float32 residual
    assert len(traded) == int(np.ceil(len(idx) / 21))     # monthly, not daily


def test_markowitz_uses_only_past_data(ds):
    """Corrupting every bar after the decision bar must not change the solution."""
    idx = ds["splits"]["validate"]
    close = ds["panel"]["close"].values
    end = int(idx[100])
    a = B.Markowitz(M, TAU, close, idx)._solve(end)
    poisoned = close.copy()
    poisoned[end + 1:] *= 5.0
    b = B.Markowitz(M, TAU, poisoned, idx)._solve(end)
    assert np.abs(a - b).max() < 1e-12


# ---- shared machinery ----

@pytest.mark.parametrize("name", ["UBAH", "UCRP", "BestStock", "Markowitz"])
def test_every_baseline_pays_costs_and_stays_on_the_simplex(ds, name):
    rec = run(ds, B.build(CFG, ds, "validate")[name])
    w = rec["weights"]
    assert np.abs(w.sum(1) - 1.0).max() < 1e-9 and (w >= 0).all()
    assert (rec["mu"] > 0).all() and (rec["mu"] <= 1.0 + 1e-12).all()
    assert rec["mu"].min() < 1.0                          # something was actually traded
    assert len(rec["value"]) == len(ds["splits"]["validate"])


@pytest.mark.parametrize("name", ["UBAH", "UCRP", "BestStock", "Markowitz"])
def test_zero_commission_dominates_ten_bps(ds, name):
    """Same policy, no costs, must end at least as wealthy."""
    import copy
    cfg0 = copy.deepcopy(CFG)
    object.__setattr__(cfg0.env, "commission", 0.0)
    idx = ds["splits"]["validate"]
    paid = backtest(PortfolioEnv(ds["X"], ds["y"], CFG, idx),
                    B.build(CFG, ds, "validate")[name])
    free = backtest(PortfolioEnv(ds["X"], ds["y"], cfg0, idx),
                    B.build(cfg0, ds, "validate")[name])
    assert free["value"][-1] >= paid["value"][-1] - 1e-12


def test_metrics_on_a_known_series():
    rec = {"value": np.array([1.10, 1.21, 1.089]), "turnover": np.zeros(3),
           "hhi": np.ones(3), "entropy": np.zeros(3), "max_weight": np.ones(3),
           "mu": np.ones(3)}
    s = Mx.summarise(rec)
    assert abs(s["CR"] - 0.089) < 1e-12
    assert abs(s["MDD"] - 0.1) < 1e-12                    # 1.21 -> 1.089
    assert abs(s["win_rate"] - 2 / 3) < 1e-12


def test_sharpe_is_annualised():
    r = np.full(252, 0.001)
    r[::2] += 0.0005
    daily = Mx.sharpe(r, periods=1)
    assert abs(Mx.sharpe(r) - daily * np.sqrt(252)) < 1e-9
