"""Phase 2 gate: simplex projection, tau reachability, UCRP equivalence."""

import numpy as np
import pandas as pd
import pytest

from src import data as D
from src import universe as U
from src.config import load_config
from src.costs import drift, transaction_remainder
from src.env import PortfolioEnv, project_to_simplex

CFG = load_config()
M = CFG.universe.n_assets


@pytest.fixture(scope="module")
def ds():
    return D.build_dataset(U.HEADLINE, CFG)


@pytest.fixture(scope="module")
def env(ds):
    return PortfolioEnv(ds["X"], ds["y"], CFG, ds["splits"]["validate"])


# ---- projection ----

def test_tau_reaches_the_full_simplex():
    """tau too low silently caps the maximum allocation. At tau=1 with 8 risky assets
    the ceiling is ~0.48 and nothing warns you."""
    a = np.full(M + 1, -1.0)
    a[0] = 1.0
    w_max = project_to_simplex(a, CFG.env.tau)[0]
    assert w_max > 0.95, f"tau={CFG.env.tau} caps max allocation at {w_max:.3f}"


def test_tau_one_would_fail_the_reachability_bound():
    a = np.full(M + 1, -1.0)
    a[0] = 1.0
    assert project_to_simplex(a, 1.0)[0] < 0.5


def test_projection_is_a_simplex():
    rng = np.random.default_rng(0)
    for a in rng.uniform(-1, 1, (500, M + 1)):
        w = project_to_simplex(a, CFG.env.tau)
        assert abs(w.sum() - 1.0) < 1e-9
        assert (w >= 0).all()


def test_projection_is_shift_invariant_and_ordered():
    rng = np.random.default_rng(1)
    a = rng.uniform(-1, 1, M + 1)
    w = project_to_simplex(a, CFG.env.tau)
    assert np.allclose(w, project_to_simplex(a + 0.3, CFG.env.tau))
    assert np.argmax(w) == np.argmax(a)


# ---- spaces and reset ----

def test_spaces(env):
    assert env.observation_space["tensor"].shape == \
        (CFG.env.n_features, M, CFG.env.window)
    assert env.observation_space["weights"].shape == (M + 1,)
    assert env.action_space.shape == (M + 1,)
    assert (env.action_space.low == -1).all() and (env.action_space.high == 1).all()


def test_reset_starts_in_cash(env):
    obs, _ = env.reset()
    assert obs["weights"][0] == 1.0 and obs["weights"][1:].sum() == 0.0


def test_episode_length_equals_split_length(ds):
    for name, idx in ds["splits"].items():
        e = PortfolioEnv(ds["X"], ds["y"], CFG, idx)
        e.reset()
        n, done = 0, False
        while not done:
            _, _, done, _, _ = e.step(np.zeros(M + 1))
            n += 1
        assert n == len(idx), name


def test_weights_stay_on_the_simplex(env):
    rng = np.random.default_rng(2)
    env.reset()
    done = False
    while not done:
        _, _, done, _, info = env.step(rng.uniform(-1, 1, M + 1))
        w = info["weights"]
        assert abs(w.sum() - 1.0) < 1e-9 and (w >= 0).all()


# ---- UCRP equivalence, computed independently in pandas ----

def _ucrp_pandas(ds, idx, c, k):
    """Equal-weight rebalance every step, replicating the env recursion in pandas:
        p_t = p_{t-1} * mu_t * (y_t . w_{t-1}),  w'_t = drift(w_{t-1}, y_t)."""
    y = pd.DataFrame(ds["y"][idx]).astype(float).values
    w_eq = np.full(M + 1, 1.0 / (M + 1))
    w_prev = np.zeros(M + 1)
    w_prev[0] = 1.0
    p, rewards = 1.0, []
    for t in range(len(idx)):
        gross = float(y[t] @ w_prev)
        w_d = drift(w_prev, y[t])
        mu = float(transaction_remainder(w_d, w_eq, c, k))
        p *= mu * gross
        rewards.append(np.log(mu * gross))
        w_prev = w_eq.copy()
    return p, np.array(rewards)


def test_constant_weight_policy_reproduces_ucrp(ds):
    idx = ds["splits"]["validate"]
    e = PortfolioEnv(ds["X"], ds["y"], CFG, idx)
    e.reset()
    action = np.zeros(M + 1)          # softmax of a constant -> exactly equal weights
    rewards, done = [], False
    while not done:
        _, r, done, _, info = e.step(action)
        rewards.append(r)
    p_ref, r_ref = _ucrp_pandas(ds, idx, CFG.env.commission, CFG.env.mu_iterations)
    assert abs(e.value - p_ref) < 1e-10
    assert np.abs(np.array(rewards) - r_ref).max() < 1e-10


def test_zero_cost_wealth_equals_product_of_gross_returns(ds):
    """With c = 0 the env must reduce to plain compounding, no cost machinery."""
    import copy
    cfg0 = copy.deepcopy(CFG)
    object.__setattr__(cfg0.env, "commission", 0.0)
    idx = ds["splits"]["test"]
    e = PortfolioEnv(ds["X"], ds["y"], cfg0, idx)
    e.reset()
    action = np.zeros(M + 1)
    w_eq = np.full(M + 1, 1.0 / (M + 1))
    y = ds["y"][idx].astype(float)
    p, w_prev = 1.0, np.eye(M + 1)[0]
    done = False
    t = 0
    while not done:
        _, _, done, _, info = e.step(action)
        p *= float(y[t] @ w_prev)
        w_prev = w_eq
        t += 1
    assert abs(e.value - p) < 1e-10


def test_mu_is_one_on_the_first_step_only_if_no_trade(ds):
    idx = ds["splits"]["validate"]
    e = PortfolioEnv(ds["X"], ds["y"], CFG, idx)
    e.reset()
    a = np.full(M + 1, -1.0)
    a[0] = 1.0                              # stay in cash
    _, _, _, _, info = e.step(a)
    assert info["mu"] > 1 - 1e-3
    assert info["turnover"] < 1e-3


def test_info_reports_concentration(env):
    env.reset()
    _, _, _, _, info = env.step(np.zeros(M + 1))
    assert abs(info["hhi"] - 1.0 / (M + 1)) < 1e-9
    assert abs(info["entropy"] - np.log(M + 1)) < 1e-9
    assert abs(info["max_weight"] - 1.0 / (M + 1)) < 1e-9


# ---- SB3 integration guards ----

def test_env_passes_sb3_env_checker(ds):
    from stable_baselines3.common.env_checker import check_env
    check_env(PortfolioEnv(ds["X"], ds["y"], CFG, ds["splits"]["validate"][:50]), warn=False)


def test_tensor_subspace_is_not_treated_as_an_image(env):
    """A (3, m, n) float Box trips SB3's image heuristic. If it were ever classed as an
    image, SB3 would auto-wrap in VecTransposeImage and silently permute the axes the
    EIIE extractor depends on."""
    from stable_baselines3.common.preprocessing import is_image_space
    assert not is_image_space(env.observation_space["tensor"])


def test_vecenv_preserves_tensor_layout(ds):
    from stable_baselines3.common.vec_env import DummyVecEnv
    v = DummyVecEnv([lambda: PortfolioEnv(ds["X"], ds["y"], CFG, ds["splits"]["validate"])])
    obs = v.reset()
    assert obs["tensor"].shape == (1, CFG.env.n_features, M, CFG.env.window)
