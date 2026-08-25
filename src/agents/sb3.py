"""PPO and DDPG on the SAME EIIEExtractor, tau, gamma and cost model as PG.

The only thing that differs between the three arms is the learning rule. Everything
that could confound that is pinned here and asserted at construction, not in a test:
the extractor class, its parameter count, gamma, and the optimiser budget.

share_features_extractor=False for both. PG's extractor receives only the Eq. 21
gradient; PPO's default (True) would also push 0.5 * value-MSE into the same weights
that emit the portfolio. False is SB3's DDPG default and matches Liang's code.
"""

import copy

import numpy as np
import torch
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from ..extractors import EIIEExtractor


def assert_eiie(model, n_params):
    """Every extractor in the policy must be an EIIEExtractor of the expected size.

    Walks the module tree instead of naming attributes: with share=False the actor,
    critic, both targets and PPO's vf side each hold their own instance, and the names
    differ between ActorCriticPolicy and TD3Policy. A missed one is the failure this
    exists to catch -- SB3 silently substitutes CombinedExtractor, which flattens the
    (3, 8, 20) tensor to a 480-vector and trains happily on nothing.

    The parameter count is the stronger half: isinstance alone still passes with a wrong
    n_assets/window/n_features kwarg.
    """
    found = [m for m in model.policy.modules() if isinstance(m, BaseFeaturesExtractor)]
    assert found, "no features extractor in the policy at all"
    for fe in found:
        assert isinstance(fe, EIIEExtractor), f"extractor replaced by {type(fe).__name__}"
        got = sum(p.numel() for p in fe.parameters())
        assert got == n_params, f"extractor has {got} params, PG's has {n_params}"
    return len(found)


def count_optimizer_steps(model):
    """Count real optimizer.step() calls. Returns a dict of live counters.

    SB3's ``model._n_updates`` is NOT the minibatch count for PPO: ppo.py:284 increments
    it once per EPOCH, outside the ``for rollout_data`` loop whose body calls
    optimizer.step() at :282. Asserting against it would have under-counted PPO by
    exactly n_steps/batch_size and made the equal-budget guard silently false.

    The controlled quantity is minibatches of ``batch_size`` drawn and backpropagated.
    For DDPG that is the critic optimiser's step count -- policy_delay=1 (ddpg.py:101)
    gives the actor an equal number, so both are recorded and either can be quoted.
    """
    opts = {}
    pol = model.policy
    if hasattr(pol, "optimizer"):
        opts["policy"] = pol.optimizer
    for name in ("actor", "critic"):
        sub = getattr(pol, name, None)
        if sub is not None and hasattr(sub, "optimizer"):
            opts[name] = sub.optimizer
    counts = {k: 0 for k in opts}

    def wrap(key, opt):
        original = opt.step

        def step(*a, **kw):
            counts[key] += 1
            return original(*a, **kw)

        opt.step = step

    for k, o in opts.items():
        wrap(k, o)
    return counts


def realised_updates(counts):
    """The minibatch count to compare against config.gradient_steps()."""
    return counts["critic"] if "critic" in counts else counts["policy"]


def _kwargs(cfg, hp):
    return dict(
        features_extractor_class=EIIEExtractor,
        features_extractor_kwargs=dict(n_assets=cfg.universe.n_assets,
                                       window=cfg.env.window,
                                       n_features=cfg.env.n_features),
        share_features_extractor=bool(hp["share_features_extractor"]),
    )


def build_ppo(env, cfg, seed):
    hp = cfg.agent.ppo
    return PPO("MultiInputPolicy", env, seed=seed, gamma=cfg.agent.gamma,
               learning_rate=float(hp["learning_rate"]), n_steps=hp["n_steps"],
               batch_size=hp["batch_size"], n_epochs=hp["n_epochs"],
               policy_kwargs={**_kwargs(cfg, hp), "net_arch": []}, verbose=0)


def build_ddpg(env, cfg, seed):
    hp = cfg.agent.ddpg
    m1 = cfg.universe.n_assets + 1
    noise = OrnsteinUhlenbeckActionNoise(np.zeros(m1),
                                         float(hp["ou_sigma"]) * np.ones(m1),
                                         theta=float(hp["ou_theta"]))
    # pi=[] keeps the actor a linear head on the extractor, as PG's is. The critic must
    # fuse features AND the action, so qf needs real capacity -- an asymmetry inherent
    # to the algorithms, reported in the architecture table rather than hidden.
    return DDPG("MultiInputPolicy", env, seed=seed, gamma=cfg.agent.gamma,
                learning_rate=float(hp["learning_rate"]), batch_size=hp["batch_size"],
                train_freq=hp["train_freq"], gradient_steps=hp["gradient_steps"],
                learning_starts=hp["learning_starts"], action_noise=noise,
                policy_kwargs={**_kwargs(cfg, hp),
                               "net_arch": dict(pi=[], qf=[64, 64])}, verbose=0)


BUILD = {"PPO": build_ppo, "DDPG": build_ddpg}


class SB3Policy:
    """backtest() adapter. Returns the RAW action; the ENV applies softmax(tau * a),
    so evaluation and training share one projection -- same contract as PGPolicy."""

    def __init__(self, model):
        self.model = model

    def __call__(self, obs):
        return self.model.predict(obs, deterministic=True)[0]


class EvalCallback(BaseCallback):
    """Evaluate every eval_every env steps; keep the best-validation checkpoint.

    x-axis is the realised minibatch count, NOT num_timesteps and NOT _n_updates (which
    counts epochs for PPO), so F1/F2/F4 put all three algorithms on one gradient-step
    axis. reward is the mean env log-return over the interval --
    the same units as PG's per-batch mean, though sampled differently.
    """

    def __init__(self, evaluate, eval_every, counts, log=None):
        super().__init__()
        self.evaluate, self.eval_every, self.log = evaluate, eval_every, log
        self.counts = counts
        self.hist = {"reward": [], "eval_step": [], "train": [], "validate": []}
        self.best = (-np.inf, 0, None)
        self._r, self._next = [], eval_every

    def _on_step(self):
        r = self.locals.get("rewards")
        if r is not None:
            self._r.append(float(np.mean(r)))
        if self.num_timesteps >= self._next:
            self._next += self.eval_every
            step = realised_updates(self.counts)   # minibatches, NOT _n_updates
            e = self.evaluate(self.model)
            self.hist["eval_step"].append(step)
            self.hist["reward"].append(float(np.mean(self._r)) if self._r else 0.0)
            self._r = []
            for s in ("train", "validate"):
                self.hist[s].append(e[s])
            if e["validate"] > self.best[0]:
                self.best = (e["validate"], step,
                             copy.deepcopy(self.model.policy.state_dict()))
            if self.log:
                self.log(step, self.hist["reward"][-1], e)
        return True

    def finish(self):
        if self.best[2] is not None:
            self.model.policy.load_state_dict(self.best[2])
        h = {k: np.asarray(v) for k, v in self.hist.items()}
        h["best_step"], h["best_validate"] = self.best[1], self.best[0]
        return h


def train(algo, env, cfg, seed, evaluate, log=None, n_params=None, total=None,
          eval_every=None, expect=None):
    """Returns (model with best-validation weights, history dict in PG's schema).

    ``env`` MUST be a separate PortfolioEnv instance from any env ``evaluate`` touches.
    backtest() calls env.reset(), so evaluating on the training env rewinds self.t
    underneath an in-flight rollout and the next step indexes past the split. PG never
    hit this because it trains on tensors and never steps an env.
    """
    hp = getattr(cfg.agent, algo.lower())
    model = BUILD[algo](env, cfg, seed)
    if n_params is not None:
        assert_eiie(model, n_params)
    assert model.gamma == cfg.agent.gamma, f"gamma is {model.gamma}"
    counts = count_optimizer_steps(model)
    cb = EvalCallback(evaluate, eval_every or hp["eval_every_steps"], counts, log)
    model.learn(total_timesteps=total or hp["total_timesteps"], callback=cb,
                progress_bar=False)
    h = cb.finish()
    got = realised_updates(counts)
    h["updates"] = got
    h["optimizer_steps"] = counts
    if expect is not None:
        assert got == expect, f"{algo} did {got} minibatch updates, config predicts {expect}"
    return model, h
