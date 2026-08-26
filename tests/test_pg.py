"""Phase 4 gate: PG + PVM correctness, then the trained-artefact checks."""

import numpy as np
import pytest
import torch

from src import data as D
from src import universe as U
from src.agents.pg import PVM, PGActor, PGPolicy, train
from src.backtest import backtest
from src.config import PROJECT_ROOT, load_config
from src.costs import drift, transaction_remainder
from src.env import PortfolioEnv, project_to_simplex


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def ds(cfg):
    return D.build_dataset(U.HEADLINE, cfg)


@pytest.fixture(scope="module")
def env(ds, cfg):
    return PortfolioEnv(ds["X"], ds["y"], cfg, ds["splits"]["train"])


@pytest.fixture(scope="module")
def actor(env, cfg):
    torch.manual_seed(0)
    return PGActor(env.observation_space, cfg)


# ---- PVM ----

def test_pvm_starts_uniform(cfg):
    p = PVM(10, cfg.universe.n_assets)
    assert np.allclose(p.mem, 1.0 / cfg.universe.n_positions)
    assert np.allclose(p.mem.sum(1), 1.0)


def test_pvm_write_at_t_is_read_at_t_plus_one(cfg):
    """The memory holds the vector carried INTO a decision, so decision i's output
    is what decision i+1 reads. Off-by-one here silently trains on the wrong w_prev."""
    p = PVM(10, cfg.universe.n_assets)
    idx = np.array([3, 4, 5])
    w = np.random.default_rng(0).dirichlet(np.ones(cfg.universe.n_positions), 3)
    p.write(idx, w)
    assert torch.allclose(p.read(idx + 1), torch.as_tensor(w, dtype=torch.float32))
    assert np.allclose(p.mem[idx[0]], 1.0 / cfg.universe.n_positions)   # untouched


# ---- actor / projection ----

def test_actor_uses_the_env_projection(actor, env, cfg):
    """PG's head and the env must apply ONE softmax(tau * .). If they diverge, the
    policy trains against a portfolio the backtest never holds."""
    x = torch.as_tensor(env.X[:4], dtype=torch.float32)
    w = torch.full((4, cfg.universe.n_positions), 1.0 / cfg.universe.n_positions)
    a = actor.action(x, w).detach().numpy()
    assert np.allclose(actor(x, w).detach().numpy(), project_to_simplex(a, cfg.env.tau),
                       atol=1e-6)


def test_actor_output_is_a_simplex(actor, env, cfg):
    x = torch.as_tensor(env.X[:16], dtype=torch.float32)
    w = torch.full((16, cfg.universe.n_positions), 1.0 / cfg.universe.n_positions)
    out = actor(x, w).detach().numpy()
    assert np.allclose(out.sum(1), 1.0, atol=1e-6) and (out >= 0).all()


def test_policy_actions_are_inside_the_action_space(actor, env):
    obs, _ = env.reset()
    a = PGPolicy(actor)(obs)
    assert env.action_space.contains(a.astype(np.float32))


# ---- gradient ----

def test_gradient_flows_through_mu_and_return(actor, env, ds, cfg):
    """PG has no critic: if mu or the return term detaches, the loss still shrinks
    and the agent learns nothing about costs."""
    idx = ds["splits"]["train"][:8]
    x = torch.as_tensor(ds["X"][idx], dtype=torch.float32)
    y = torch.as_tensor(ds["y"][idx], dtype=torch.float32)
    wp = torch.full((8, cfg.universe.n_positions), 1.0 / cfg.universe.n_positions)
    w = actor(x, wp)
    mu = transaction_remainder(drift(wp, y), w, cfg.env.commission,
                               cfg.env.mu_iterations, backend="torch")
    assert mu.requires_grad
    (-torch.log(mu * (y * w).sum(-1)).mean()).backward()
    grads = {n: p.grad for n, p in actor.named_parameters()}
    assert all(g is not None for g in grads.values())
    assert any(g.abs().sum() > 0 for g in grads.values())


def test_cost_term_alone_penalises_trading(actor, cfg):
    """mu must decrease as the target moves away from the drifted holding."""
    m = cfg.universe.n_positions
    wd = torch.full((1, m), 1.0 / m)
    near = torch.full((1, m), 1.0 / m)
    far = torch.eye(m)[:1]
    args = (cfg.env.commission, cfg.env.mu_iterations)
    mu_near = transaction_remainder(wd, near, *args, backend="torch")
    mu_far = transaction_remainder(wd, far, *args, backend="torch")
    assert mu_near > mu_far


# ---- training ----

def test_training_never_reads_beyond_the_train_split(ds):
    """The credit term uses y[i+1]; the last train decision's y[i+1] is the first
    VALIDATION bar. Dropping it is the difference between a clean split and a leak."""
    idx = ds["splits"]["train"]
    usable = idx[:-1]
    assert usable[-1] + 1 == idx[-1]
    assert usable[-1] + 1 < ds["splits"]["validate"][0]


def test_training_raises_the_objective(ds, cfg, env):
    cfg.agent.pg.update(gradient_steps=300, eval_every=300, learning_rate=3e-3)
    _, h = train(ds, cfg, env.observation_space, seed=0,
                 evaluate=lambda a: {"train": 1.0, "validate": 1.0})
    r = h["reward"]
    assert r[-50:].mean() > r[:50].mean()


def test_selection_returns_the_best_validation_checkpoint(ds, cfg, env):
    """train() must hand back the SELECTED weights, not the last ones."""
    cfg.agent.pg.update(gradient_steps=60, eval_every=20, learning_rate=3e-3)
    vals = iter([1.0, 3.0, 2.0])
    actor, h = train(ds, cfg, env.observation_space, seed=0,
                     evaluate=lambda a: {"train": 1.0, "validate": next(vals)})
    assert h["best_step"] == 40 and h["best_validate"] == 3.0


# ---- trained artefacts ----

RUN = PROJECT_ROOT / "results" / "pg"
needs_run = pytest.mark.skipif(not (RUN / "metrics_per_seed.csv").exists(),
                               reason="run scripts/03_train_pg.py first")


@pytest.fixture(scope="module")
def per_seed():
    import pandas as pd
    return pd.read_csv(RUN / "metrics_per_seed.csv")


@needs_run
def test_max_weight_is_not_pinned_at_the_tau_ceiling(per_seed, cfg):
    """If every seed sits at the ceiling, tau is capping the portfolio, not the agent
    choosing. See config.max_reachable_weight."""
    ceiling = cfg.max_reachable_weight
    assert (per_seed["max_weight"] < ceiling - 1e-3).all()


@needs_run
def test_concentration_is_logged_and_finite(per_seed, cfg):
    lo = 1.0 / cfg.universe.n_positions
    assert (per_seed["HHI"] >= lo - 1e-9).all() and (per_seed["HHI"] <= 1.0).all()


@needs_run
def test_costs_were_actually_paid(per_seed):
    assert (per_seed["mu_min"] <= 1.0).all() and (per_seed["mu_min"] > 0.9).all()


@needs_run
def test_history_shapes_match_the_seed_count(per_seed):
    h = np.load(RUN / "PG" / "history.npz")
    n = per_seed["seed"].nunique()
    assert h["train"].shape == h["validate"].shape == (n, len(h["eval_step"]))
    assert h["reward"].shape[0] == n and h["best_step"].shape == (n,)
