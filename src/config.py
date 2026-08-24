"""Frozen configuration, loaded once and passed down.
Every experiment parameter lives in ``config/base.yaml`` and arrives here, validated, as an immutable dataclass.

The validation is not decorative. Two failure modes in this project are silent 
(they produce plausible results from a broken setup) and both are caught at load
time rather than in a test:

* ``tau`` too low structurally caps the maximum single-asset allocation.
* ``gamma`` != 1.0 confounds the learning rule with the planning horizon, because
  PG optimises an undiscounted objective (Jiang Eq. 21) while Stable-Baselines3
  defaults to 0.99.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "base.yaml"

# Minimum single-asset weight the action space must be able to reach. Below this
# the agent cannot express a concentrated portfolio and every result is suspect.
MIN_REACHABLE_WEIGHT = 0.95


class ConfigError(ValueError):
    """Raised when a configuration is internally inconsistent."""


def max_reachable_weight(tau: float, n_assets: int) -> float:
    """Largest single-asset weight reachable through the simplex projection.

    Actions are bounded to ``[-1, 1]`` (SB3 squashes its actor output through
    ``tanh`` and rescales to the action-space bounds). The most concentrated
    action is therefore ``+1`` in one position and ``-1`` in the other
    ``n_assets``, giving::

        w_max = e^tau / (e^tau + n_assets * e^-tau)

    At ``tau=1`` with 8 risky assets this is ~0.48 — the agent becomes
    structurally incapable of holding more than half in any one asset, silently.
    At ``tau=5`` it is ~0.9996 and the full simplex is available.
    """
    hi = math.exp(tau)
    lo = math.exp(-tau)
    return hi / (hi + n_assets * lo)


@dataclass(frozen=True)
class Split:
    name: str
    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start >= self.end:
            raise ConfigError(f"split {self.name!r}: start {self.start} is not before end {self.end}")


@dataclass(frozen=True)
class DataConfig:
    source: str
    start: date
    end: date
    raw_dir: Path
    processed_dir: Path
    splits: tuple[Split, ...]

    def split(self, name: str) -> Split:
        for s in self.splits:
            if s.name == name:
                return s
        raise KeyError(f"no split named {name!r}; have {[s.name for s in self.splits]}")


@dataclass(frozen=True)
class UniverseConfig:
    n_assets: int
    as_of: date
    correlation_gate: float

    @property
    def n_positions(self) -> int:
        """Length of the weight vector: risky assets plus cash (Jiang Eq. 5)."""
        return self.n_assets + 1


@dataclass(frozen=True)
class EnvConfig:
    window: int
    n_features: int
    tau: float
    commission: float
    mu_iterations: int
    initial_capital: float

    def tensor_shape(self, n_assets: int) -> tuple[int, int, int]:
        """Observation tensor shape ``(features, assets, time)`` — Jiang Eq. 18."""
        return (self.n_features, n_assets, self.window)


@dataclass(frozen=True)
class AgentConfig:
    gamma: float
    pg: dict[str, Any]
    ppo: dict[str, Any]
    ddpg: dict[str, Any]


@dataclass(frozen=True)
class EvaluationConfig:
    trading_days_per_year: int
    risk_free: str
    bootstrap_samples: int


@dataclass(frozen=True)
class Config:
    run_seeds: tuple[int, ...]
    results_dir: Path
    data: DataConfig
    universe: UniverseConfig
    env: EnvConfig
    agent: AgentConfig
    evaluation: EvaluationConfig
    sweeps: dict[str, list[Any]]

    @property
    def max_reachable_weight(self) -> float:
        return max_reachable_weight(self.env.tau, self.universe.n_assets)


def _as_date(value: Any, field: str) -> date:
    if isinstance(value, date):
        return value
    raise ConfigError(f"{field}: expected a YYYY-MM-DD date, got {value!r}")


def _validate(cfg: Config) -> None:
    """Fail loudly and early on any internally inconsistent configuration."""
    env, uni, data = cfg.env, cfg.universe, cfg.data

    if env.window < 1:
        raise ConfigError(f"env.window must be >= 1, got {env.window}")
    if env.n_features < 1:
        raise ConfigError(f"env.n_features must be >= 1, got {env.n_features}")
    if env.mu_iterations < 1:
        raise ConfigError(f"env.mu_iterations must be >= 1, got {env.mu_iterations}")
    if not 0.0 <= env.commission < 1.0:
        raise ConfigError(f"env.commission must be in [0, 1), got {env.commission}")
    if env.tau <= 0:
        raise ConfigError(f"env.tau must be positive, got {env.tau}")
    if uni.n_assets < 2:
        raise ConfigError(f"universe.n_assets must be >= 2, got {uni.n_assets}")
    if not 0.0 < uni.correlation_gate <= 1.0:
        raise ConfigError(f"universe.correlation_gate must be in (0, 1], got {uni.correlation_gate}")

    # The silent-cap guard. See max_reachable_weight().
    w_max = cfg.max_reachable_weight
    if w_max < MIN_REACHABLE_WEIGHT:
        raise ConfigError(
            f"env.tau={env.tau} with {uni.n_assets} risky assets caps the maximum "
            f"single-asset weight at {w_max:.4f}, below the required "
            f"{MIN_REACHABLE_WEIGHT}. The agent could not express a concentrated "
            f"portfolio and every result would be silently wrong. "
            f"Raise env.tau (5.0 gives {max_reachable_weight(5.0, uni.n_assets):.4f})."
        )

    # PG maximises an undiscounted objective (Jiang Eq. 21). SB3 defaults to 0.99.
    if cfg.agent.gamma != 1.0:
        raise ConfigError(
            f"agent.gamma must be 1.0 for all three agents, got {cfg.agent.gamma}. "
            "A discounted PPO/DDPG against an undiscounted PG confounds the learning "
            "rule with the planning horizon."
        )

    if not cfg.run_seeds:
        raise ConfigError("run.seeds must not be empty")
    if len(set(cfg.run_seeds)) != len(cfg.run_seeds):
        raise ConfigError(f"run.seeds contains duplicates: {cfg.run_seeds}")

    # Splits: chronological, non-overlapping, and spanning the declared window exactly.
    if not data.splits:
        raise ConfigError("data.splits must not be empty")
    for prev, nxt in zip(data.splits, data.splits[1:]):
        if nxt.start <= prev.end:
            raise ConfigError(
                f"splits {prev.name!r} and {nxt.name!r} overlap: "
                f"{prev.name} ends {prev.end}, {nxt.name} starts {nxt.start}"
            )
    if data.splits[0].start != data.start:
        raise ConfigError(
            f"first split starts {data.splits[0].start} but data.start is {data.start}"
        )
    if data.splits[-1].end != data.end:
        raise ConfigError(
            f"last split ends {data.splits[-1].end} but data.end is {data.end}"
        )
    if uni.as_of != data.start:
        raise ConfigError(
            f"universe.as_of ({uni.as_of}) must equal data.start ({data.start}); "
            "measuring market caps at any later date reintroduces look-ahead bias."
        )


def load_config(path: str | Path | None = None) -> Config:
    """Load, parse and validate the experiment configuration."""
    path = Path(path) if path is not None else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    d, u, e = raw["data"], raw["universe"], raw["env"]

    data = DataConfig(
        source=d["source"],
        start=_as_date(d["start"], "data.start"),
        end=_as_date(d["end"], "data.end"),
        raw_dir=PROJECT_ROOT / d["raw_dir"],
        processed_dir=PROJECT_ROOT / d["processed_dir"],
        splits=tuple(
            Split(
                name=s["name"],
                start=_as_date(s["start"], f"data.splits.{s['name']}.start"),
                end=_as_date(s["end"], f"data.splits.{s['name']}.end"),
            )
            for s in d["splits"]
        ),
    )

    cfg = Config(
        run_seeds=tuple(raw["run"]["seeds"]),
        results_dir=PROJECT_ROOT / raw["run"]["results_dir"],
        data=data,
        universe=UniverseConfig(
            n_assets=u["n_assets"],
            as_of=_as_date(u["as_of"], "universe.as_of"),
            correlation_gate=float(u["correlation_gate"]),
        ),
        env=EnvConfig(
            window=e["window"],
            n_features=e["n_features"],
            tau=float(e["tau"]),
            commission=float(e["commission"]),
            mu_iterations=e["mu_iterations"],
            initial_capital=float(e["initial_capital"]),
        ),
        agent=AgentConfig(
            gamma=float(raw["agent"]["gamma"]),
            pg=dict(raw["agent"]["pg"]),
            ppo=dict(raw["agent"]["ppo"]),
            ddpg=dict(raw["agent"]["ddpg"]),
        ),
        evaluation=EvaluationConfig(**raw["evaluation"]),
        sweeps=dict(raw["sweeps"]),
    )

    _validate(cfg)
    return cfg


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    c = load_config()
    print(f"window={c.env.window}  tau={c.env.tau}  gamma={c.agent.gamma}")
    print(f"tensor shape = {c.env.tensor_shape(c.universe.n_assets)}")
    print(f"weight vector length = {c.universe.n_positions}")
    print(f"max reachable single-asset weight = {c.max_reachable_weight:.4f}")
    for s in c.data.splits:
        print(f"  {s.name:9s} {s.start} -> {s.end}")
