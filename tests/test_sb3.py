"""Phase 5 guards: the three arms must differ ONLY in the learning rule."""

import math

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, CombinedExtractor

from src import data as D
from src import universe as U
from src.agents.pg import PGActor
from src.agents.sb3 import (BUILD, SB3Policy, assert_eiie, count_optimizer_steps,
                            realised_updates, train)
from src.backtest import backtest
from src.config import gradient_steps, load_config
from src.env import PortfolioEnv, project_to_simplex
from src.extractors import EIIEExtractor

ALGOS = ("PPO", "DDPG")


@pytest.fixture(scope="module")
def fx():
    cfg = load_config()
    ds = D.build_dataset(U.HEADLINE, cfg)
    mk = lambda s: PortfolioEnv(ds["X"], ds["y"], cfg, ds["splits"][s])
    n = sum(p.numel() for p in PGActor(mk("train").observation_space, cfg).net.parameters())
    return cfg, ds, mk, n


@pytest.mark.parametrize("algo", ALGOS)
def test_every_extractor_is_eiie_including_the_critic(fx, algo):
    cfg, _, mk, n = fx
    model = BUILD[algo](mk("train"), cfg, 0)
    found = [m for m in model.policy.modules() if isinstance(m, BaseFeaturesExtractor)]
    assert len(found) >= 2, "expected separate actor/critic extractors with share=False"
    assert all(isinstance(f, EIIEExtractor) for f in found)
    assert assert_eiie(model, n) == len(found)


@pytest.mark.parametrize("algo", ALGOS)
def test_extractor_matches_pg_parameter_for_parameter(fx, algo):
    """isinstance alone passes with a wrong n_assets/window kwarg; the count does not."""
    cfg, _, mk, n = fx
    model = BUILD[algo](mk("train"), cfg, 0)
    for f in (m for m in model.policy.modules() if isinstance(m, EIIEExtractor)):
        assert sum(p.numel() for p in f.parameters()) == n


def test_the_flattening_trap_is_actually_caught(fx):
    """SB3 silently substitutes CombinedExtractor, which turns (3,8,20) into a
    480-vector and trains happily on nothing. This is the failure Phase 5 exists for."""
    cfg, _, mk, n = fx
    bad = PPO("MultiInputPolicy", mk("train"), seed=0, verbose=0)
    assert isinstance(bad.policy.features_extractor, CombinedExtractor)
    with pytest.raises(AssertionError, match="CombinedExtractor"):
        assert_eiie(bad, n)


@pytest.mark.parametrize("algo", ALGOS)
def test_extractors_are_not_shared_between_actor_and_critic(fx, algo):
    cfg, _, mk, _ = fx
    model = BUILD[algo](mk("train"), cfg, 0)
    assert model.policy.share_features_extractor is False
    ids = {id(m) for m in model.policy.modules() if isinstance(m, EIIEExtractor)}
    assert len(ids) >= 2, "actor and critic must hold distinct extractor instances"


@pytest.mark.parametrize("algo", ALGOS)
def test_gamma_is_undiscounted(fx, algo):
    cfg, _, mk, _ = fx
    assert BUILD[algo](mk("train"), cfg, 0).gamma == 1.0 == cfg.agent.gamma


@pytest.mark.parametrize("algo", ALGOS)
def test_policy_returns_a_raw_action_the_env_projects(fx, algo):
    """Same contract as PGPolicy: the ENV owns the softmax, so training and evaluation
    cannot drift apart."""
    cfg, _, mk, _ = fx
    env = mk("validate")
    model = BUILD[algo](mk("train"), cfg, 0)
    obs, _ = env.reset()
    a = SB3Policy(model)(obs)
    assert env.action_space.contains(np.asarray(a, np.float32))
    w = project_to_simplex(a, cfg.env.tau)
    assert math.isclose(float(w.sum()), 1.0, rel_tol=1e-9)
    _, _, _, _, info = env.step(a)
    assert np.allclose(info["weights"], w)


@pytest.mark.parametrize("algo", ALGOS)
def test_realised_update_count_matches_the_config_prediction(fx, algo):
    """config.gradient_steps() is a prediction; only optimizer.step() calls are evidence.
    _n_updates would UNDER-count PPO by n_steps/batch_size (ppo.py:284 counts epochs)."""
    cfg, _, mk, n = fx
    hp = getattr(cfg.agent, algo.lower())
    total = 2000
    if algo == "PPO":
        pred = (total // hp["n_steps"]) * hp["n_epochs"] * math.ceil(
            hp["n_steps"] / hp["batch_size"])
    else:
        pred = (total // hp["train_freq"]) * hp["gradient_steps"]
    _, h = train(algo, mk("train"), cfg, 0, lambda m: {"train": 1.0, "validate": 1.0},
                 n_params=n, total=total, eval_every=10 ** 9, expect=pred)
    assert h["updates"] == pred


def test_ppo_n_updates_is_not_the_minibatch_count(fx):
    """Pins the trap itself, so a future refactor cannot quietly revert to _n_updates."""
    cfg, _, mk, n = fx
    hp = cfg.agent.ppo
    model, h = train("PPO", mk("train"), cfg, 0,
                     lambda m: {"train": 1.0, "validate": 1.0},
                     n_params=n, total=2000, eval_every=10 ** 9)
    assert h["updates"] == model._n_updates * math.ceil(hp["n_steps"] / hp["batch_size"])
    assert h["updates"] != model._n_updates


@pytest.mark.parametrize("algo", ALGOS)
def test_evaluation_uses_a_separate_env_from_training(fx, algo):
    """backtest() resets the env; evaluating on the training env rewinds an in-flight
    rollout and the next step indexes past the split."""
    cfg, ds, mk, n = fx
    shared = mk("train")
    def evaluate(m):
        return {"train": float(backtest(shared, SB3Policy(m))["value"][-1]),
                "validate": 1.0}
    with pytest.raises(IndexError):
        train(algo, shared, cfg, 0, evaluate, n_params=n, total=1200, eval_every=600)
