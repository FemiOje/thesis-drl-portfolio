"""One rollout loop, used by agents AND baselines so costs are paid identically."""

import numpy as np


def backtest(env, policy):
    """policy: callable(obs) -> action in [-1, 1]^(m+1). Objects with .reset() are reset
    at episode start. Allocations come from info['weights'], never from the raw action."""
    if hasattr(policy, "reset"):
        policy.reset()
    obs, _ = env.reset()
    rec = {k: [] for k in ("reward", "value", "mu", "turnover", "hhi", "entropy",
                           "max_weight", "gross")}
    weights, actions = [], []
    done = False
    while not done:
        a = np.asarray(policy(obs), float)
        actions.append(a)
        obs, r, done, _, info = env.step(a)
        rec["reward"].append(r)
        for k in ("value", "mu", "turnover", "hhi", "entropy", "max_weight", "gross"):
            rec[k].append(info[k])
        weights.append(info["weights"])
    out = {k: np.asarray(v) for k, v in rec.items()}
    out["weights"] = np.asarray(weights)
    out["actions"] = np.asarray(actions)
    return out
