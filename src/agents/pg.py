"""Deterministic policy gradient (Jiang §5) + Portfolio-Vector Memory.

No critic, no sampling, no log-prob term: the gradient flows straight through the
reward, so mu_t must be differentiable (costs.py torch backend, k iterations unrolled).

Timing. The env pays reward_t = log(mu_t * y_t . w_{t-1}) — Eq. 10. Attributing each
term to the action that caused it, the action w_i is charged mu_i (cost of moving into
it) and credited y_{i+1} . w_i (what holding it earns). Both depend on w_i, which is
what makes the gradient informative; charging w_i only through mu_i would reward doing
nothing. Summed over an episode the two attributions are the same product, so this is
the env's objective, not an approximation of it.
"""

import copy

import numpy as np
import torch
import torch.nn as nn

from ..costs import drift, transaction_remainder
from ..extractors import EIIEExtractor


class PVM:
    """Portfolio-Vector Memory, Jiang §5.2.

    mem[i] is the vector held entering decision i, i.e. the output of decision i-1.
    Reading it instead of chaining lets a mini-batch of consecutive steps be trained
    without back-propagating through the whole episode.
    """

    def __init__(self, T, m):
        self.mem = np.full((T + 1, m + 1), 1.0 / (m + 1))

    def read(self, idx):
        return torch.as_tensor(self.mem[idx], dtype=torch.float32)

    def write(self, idx, w):
        self.mem[np.asarray(idx) + 1] = w


class PGActor(nn.Module):
    """EIIE logits -> tanh -> softmax(tau * .). The tanh keeps PG inside the same
    bounded action space SB3's DDPG squashes into, so all three agents can reach
    exactly the same set of portfolios."""

    def __init__(self, obs_space, cfg):
        super().__init__()
        self.net = EIIEExtractor(obs_space, cfg.universe.n_assets, cfg.env.window,
                                 cfg.env.n_features)
        self.tau = cfg.env.tau

    def action(self, x, w_prev):
        return torch.tanh(self.net({"tensor": x, "weights": w_prev}))

    def forward(self, x, w_prev):
        return torch.softmax(self.tau * self.action(x, w_prev), dim=-1)


class PGPolicy:
    """backtest() adapter. Returns the raw action; the ENV applies softmax(tau * a),
    so evaluation and training share one projection."""

    def __init__(self, actor):
        self.actor = actor

    def reset(self):
        self.actor.eval()

    def __call__(self, obs):
        with torch.no_grad():
            x = torch.as_tensor(obs["tensor"])[None]
            w = torch.as_tensor(obs["weights"])[None]
            return self.actor.action(x, w)[0].numpy()


def train(ds, cfg, obs_space, seed=0, evaluate=None, log=None):
    """Returns (actor with the best-validation weights, history dict).

    evaluate: callable(actor) -> dict of scalars logged every eval_every steps and used
    for selection via key "validate".
    """
    hp = cfg.agent.pg
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    idx = ds["splits"]["train"]
    # y_{i+1} is needed for the credit term, and idx[-1] + 1 is the first VALIDATION
    # bar. Dropping the last decision keeps training strictly inside the split.
    usable = idx[:-1]
    B, T = hp["batch_size"], len(usable)
    if T <= B:
        raise ValueError(f"train split has {T} usable steps, batch is {B}")

    X = torch.as_tensor(ds["X"], dtype=torch.float32)
    Y = torch.as_tensor(ds["y"], dtype=torch.float32)
    actor = PGActor(obs_space, cfg)
    opt = torch.optim.Adam(actor.parameters(), lr=hp["learning_rate"],
                           weight_decay=hp["l2"])
    pvm = PVM(len(ds["X"]), cfg.universe.n_assets)

    c, k = cfg.env.commission, cfg.env.mu_iterations
    hist = {"reward": [], "eval_step": [], "train": [], "validate": []}
    best = (-np.inf, 0, copy.deepcopy(actor.state_dict()))

    for step in range(1, hp["gradient_steps"] + 1):
        b = rng.integers(0, T - B + 1)
        batch = usable[b:b + B]                       # consecutive, Jiang §5.3
        actor.train()
        w_prev = pvm.read(batch)
        w = actor(X[batch], w_prev)
        mu = transaction_remainder(drift(w_prev, Y[batch]), w, c, k, backend="torch")
        reward = torch.log(mu * (Y[batch + 1] * w).sum(-1))     # Eq. 10, re-attributed
        loss = -reward.mean()                                   # Eq. 21, negated
        opt.zero_grad()
        loss.backward()
        opt.step()
        pvm.write(batch, w.detach().numpy())
        hist["reward"].append(float(-loss))

        if evaluate is not None and step % hp["eval_every"] == 0:
            actor.eval()
            e = evaluate(actor)
            hist["eval_step"].append(step)
            for s in ("train", "validate"):
                hist[s].append(e[s])
            if e["validate"] > best[0]:
                best = (e["validate"], step, copy.deepcopy(actor.state_dict()))
            if log:
                log(step, float(-loss), e)

    actor.load_state_dict(best[2])
    hist = {kk: np.asarray(v) for kk, v in hist.items()}
    hist["best_step"] = best[1]
    hist["best_validate"] = best[0]
    return actor, hist
