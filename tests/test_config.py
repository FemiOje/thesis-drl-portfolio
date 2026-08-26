"""The configuration loads, and the two silent failure modes are caught.

The `tau` guard in particular encodes the failure where a too-low temperature caps the maximum
allocation and produces plausible but meaningless results.
"""

from __future__ import annotations

import math
from datetime import date

import pytest
import yaml

from src.config import (
    MIN_REACHABLE_WEIGHT,
    ConfigError,
    load_config,
    max_reachable_weight,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _write(tmp_path, raw):
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return p


@pytest.fixture
def raw():
    with open("config/base.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# --------------------------------------------------------------------------- #
# The committed configuration is internally consistent
# --------------------------------------------------------------------------- #

def test_config_loads(cfg):
    assert cfg.env.window == 20
    assert cfg.env.n_features == 3
    assert cfg.universe.n_assets == 8
    assert cfg.universe.n_positions == 9, "8 risky assets + cash (Jiang Eq. 5)"


def test_tensor_shape_matches_plan(cfg):
    assert cfg.env.tensor_shape(cfg.universe.n_assets) == (3, 8, 20)


def test_splits_are_chronological_and_non_overlapping(cfg):
    splits = cfg.data.splits
    assert [s.name for s in splits] == ["train", "validate", "test"]
    for prev, nxt in zip(splits, splits[1:]):
        assert nxt.start > prev.end, f"{prev.name} and {nxt.name} overlap"


def test_splits_span_the_declared_window(cfg):
    assert cfg.data.splits[0].start == cfg.data.start
    assert cfg.data.splits[-1].end == cfg.data.end


def test_universe_as_of_equals_data_start(cfg):
    """Measuring market caps later than the window start reintroduces look-ahead bias."""
    assert cfg.universe.as_of == cfg.data.start == date(2021, 8, 25)


def test_gamma_is_undiscounted(cfg):
    """PG maximises an undiscounted objective (Jiang Eq. 21); SB3 defaults to 0.99."""
    assert cfg.agent.gamma == 1.0


def test_ten_seeds(cfg):
    assert len(cfg.run_seeds) == 10
    assert len(set(cfg.run_seeds)) == 10


# --------------------------------------------------------------------------- #
# The tau guard — IMPLEMENTATION_PLAN.md §2
# --------------------------------------------------------------------------- #

def test_max_reachable_weight_closed_form():
    """w_max = e^tau / (e^tau + n * e^-tau)."""
    got = max_reachable_weight(tau=5.0, n_assets=8)
    want = math.exp(5.0) / (math.exp(5.0) + 8 * math.exp(-5.0))
    assert got == pytest.approx(want)


def test_tau_one_would_cap_allocations_near_half():
    """The documented failure: tau=1 makes >48% in any asset unreachable."""
    assert max_reachable_weight(tau=1.0, n_assets=8) == pytest.approx(0.4802, abs=1e-3)


def test_committed_tau_reaches_the_full_simplex(cfg):
    assert cfg.max_reachable_weight > MIN_REACHABLE_WEIGHT
    assert cfg.max_reachable_weight == pytest.approx(0.9996, abs=1e-3)


def test_low_tau_is_rejected_at_load_time(tmp_path, raw):
    """A capped tau must fail loudly at startup, not silently at training time."""
    raw["env"]["tau"] = 1.0
    with pytest.raises(ConfigError, match="caps the maximum"):
        load_config(_write(tmp_path, raw))


# --------------------------------------------------------------------------- #
# Inconsistent configurations are rejected
# --------------------------------------------------------------------------- #

def test_discounted_gamma_is_rejected(tmp_path, raw):
    raw["agent"]["gamma"] = 0.99
    with pytest.raises(ConfigError, match="gamma must be 1.0"):
        load_config(_write(tmp_path, raw))


def test_overlapping_splits_are_rejected(tmp_path, raw):
    raw["data"]["splits"][1]["start"] = raw["data"]["splits"][0]["end"]
    with pytest.raises(ConfigError, match="overlap"):
        load_config(_write(tmp_path, raw))


def test_as_of_after_data_start_is_rejected(tmp_path, raw):
    raw["universe"]["as_of"] = date(2023, 1, 1)
    with pytest.raises(ConfigError, match="look-ahead"):
        load_config(_write(tmp_path, raw))


def test_negative_commission_is_rejected(tmp_path, raw):
    raw["env"]["commission"] = -0.001
    with pytest.raises(ConfigError, match="commission"):
        load_config(_write(tmp_path, raw))


# ---- training budget is a controlled variable ----

def test_all_three_algorithms_get_the_same_number_of_updates():
    """Architecture, tau, gamma and cost model are already identical. If the budget
    differs, "which algorithm won" is confounded with "which trained longest"."""
    from src.config import batch_sizes, gradient_steps
    cfg = load_config()
    assert set(gradient_steps(cfg.agent).values()) == {60000}
    assert set(batch_sizes(cfg.agent).values()) == {50}


def test_unequal_budgets_are_rejected_at_load_time():
    from src.config import ConfigError, _validate
    cfg = load_config()
    cfg.agent.ppo["total_timesteps"] *= 3
    try:
        with pytest.raises(ConfigError, match="not equal across algorithms"):
            _validate(cfg)
    finally:
        cfg.agent.ppo["total_timesteps"] //= 3


def test_evaluation_cadence_matches_across_algorithms():
    """F1/F2/F4 put gradient steps on the x-axis, so the three must be sampled at
    the same points or the curves are not comparable."""
    cfg = load_config()
    pg_evals = cfg.agent.pg["gradient_steps"] // cfg.agent.pg["eval_every"]
    for name in ("ppo", "ddpg"):
        a = getattr(cfg.agent, name)
        assert a["total_timesteps"] // a["eval_every_steps"] == pg_evals


def test_ddpg_learning_starts_does_not_swallow_a_trigger():
    """SB3 trains only when num_timesteps > learning_starts, so learning_starts must
    sit strictly inside the first train_freq window or one trigger is lost silently."""
    d = load_config().agent.ddpg
    assert d["learning_starts"] < d["train_freq"]
