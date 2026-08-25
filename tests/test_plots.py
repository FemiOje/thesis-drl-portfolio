"""Phase 3.5 gate: F5-F8 render for baselines, PNG + PDF, consistent colours."""

import numpy as np
import pandas as pd
import pytest

from src import plots, results
from src.config import PROJECT_ROOT

PHASE3 = PROJECT_ROOT / "results" / "phase3"
pytestmark = pytest.mark.skipif(not (PHASE3 / "curves.npz").exists(),
                                reason="run scripts/02_run_baselines.py first")
STRATEGIES = ["UBAH", "UCRP", "BestStock", "Markowitz"]


@pytest.fixture(scope="module")
def r():
    return results.load_baselines(PHASE3)


def both_formats(path):
    return path.exists() and path.with_suffix(".pdf").exists()


# ---- loader ----

def test_loader_shapes(r):
    for split in results.SPLITS:
        assert set(r["curves"][split]) == set(STRATEGIES)
        for name, c in r["curves"][split].items():
            assert c.ndim == 2 and c.shape[0] == 1          # (n_seeds, T)
            assert c.shape[1] == len(r["dates"][split])


def test_loader_matches_the_metrics_table(r):
    for _, row in r["metrics"].iterrows():
        c = r["curves"][row["split"]][row["strategy"]][0]
        assert abs(c[-1] - row["final_value"]) < 1e-6


def test_merge_without_agents_is_a_passthrough(r):
    m = results.merge(r, None)
    assert set(m["curves"]["test"]) == set(STRATEGIES)


def test_merge_adds_agent_curves(r):
    fake = {"curves": {s: {"PG": np.ones((3, len(r["dates"][s])))} for s in results.SPLITS},
            "metrics": pd.DataFrame()}
    m = results.merge(r, fake)
    assert "PG" in m["curves"]["test"] and m["curves"]["test"]["PG"].shape[0] == 3
    assert set(STRATEGIES) <= set(m["curves"]["test"])


# ---- figures ----

def test_f5_writes_png_and_pdf(r, tmp_path):
    assert both_formats(plots.f5_test_wealth(r["dates"]["test"], r["curves"]["test"],
                                             tmp_path, "t"))


def test_f6_writes_png_and_pdf(r, tmp_path):
    assert both_formats(plots.f6_train_val_wealth(r["dates"], r["curves"], tmp_path, "t"))


def test_f7_writes_png_and_pdf(r, tmp_path):
    assert both_formats(plots.f7_wealth_and_drawdown(r["dates"]["test"],
                                                     r["curves"]["test"], tmp_path, "t"))


@pytest.mark.parametrize("split", ["train", "validate", "test"])
def test_f8_writes_png_and_pdf(r, tmp_path, split):
    assert both_formats(plots.f8_metric_heatmap(r["metrics"], tmp_path, "t", split))


def test_figures_handle_multiple_seeds(r, tmp_path):
    """The IQR band path must work before any agent exists."""
    curves = {k: np.repeat(v, 5, axis=0) * np.linspace(0.9, 1.1, 5)[:, None]
              for k, v in r["curves"]["test"].items()}
    assert both_formats(plots.f5_test_wealth(r["dates"]["test"], curves, tmp_path, "t"))


# ---- conventions ----

def test_every_strategy_has_a_distinct_colour():
    names = ["PG", "PPO", "DDPG"] + STRATEGIES
    cols = [plots.STRATEGY_COLORS[n] for n in names]
    assert len(set(cols)) == len(names)


def test_best_stock_is_visually_distinguished():
    """A hindsight upper reference must not read as a competing strategy."""
    assert plots.style_for("BestStock")["ls"] == "--"
    assert all(plots.style_for(n)["ls"] == "-" for n in ["PG", "PPO", "DDPG", "UCRP"])


def test_column_normalisation_inverts_lower_is_better():
    df = pd.DataFrame({"CR": [0.1, 0.5], "MDD": [0.1, 0.5], "turnover": [0.1, 0.5]})
    z = plots.normalise_columns(df)
    assert z["CR"].tolist() == [0.0, 1.0]        # higher CR is better
    assert z["MDD"].tolist() == [1.0, 0.0]       # lower MDD is better
    assert z["turnover"].tolist() == [1.0, 0.0]


def test_column_normalisation_is_per_column_and_survives_ties():
    df = pd.DataFrame({"CR": [0.1, 0.9], "sharpe": [7.0, 7.0]})
    z = plots.normalise_columns(df)
    assert z["CR"].max() == 1.0 and z["CR"].min() == 0.0
    assert z["sharpe"].tolist() == [0.5, 0.5]    # no division by zero


# ---- F9 / F10 (significance layer) ----

def _fake_per_seed(n=6):
    rows = []
    for split in ("test", "validate"):
        for algo in ("PG", "PPO", "DDPG"):
            for s in range(n):
                rows.append({"seed": s, "split": split, "strategy": algo,
                             "final_value": 1.0 + 0.03 * s, "sharpe": 0.5 + 0.1 * s})
    return pd.DataFrame(rows)


def test_f9_writes_png_and_pdf(r, tmp_path):
    assert both_formats(plots.f9_seed_distributions(_fake_per_seed(), r["metrics"],
                                                    tmp_path, "t", "test"))


def test_f9_draws_every_seed_not_just_the_box(tmp_path, r, monkeypatch):
    """n = 10 makes the individual seeds the finding; a box alone hides bimodality.

    save() closes the figure, so intercept it and count the marker artists actually
    drawn -- asserting on the input dataframe would test nothing about the figure.
    """
    grabbed = {}
    real_save = plots.save

    def spy(fig, out_dir, name, run_id=""):
        grabbed["axes"] = fig.axes
        grabbed["counts"] = [sum(len(ln.get_xdata()) for ln in ax.lines
                                 if ln.get_marker() == "o") for ax in fig.axes]
        return real_save(fig, out_dir, name, run_id)

    monkeypatch.setattr(plots, "save", spy)
    plots.f9_seed_distributions(_fake_per_seed(6), r["metrics"], tmp_path, "t", "test")
    assert len(grabbed["axes"]) == 2                      # final wealth + Sharpe
    assert all(c == 18 for c in grabbed["counts"])        # 6 seeds x 3 algos, each panel


def test_f9_handles_a_missing_algorithm(r, tmp_path):
    """Runs where only some arms finished must still render."""
    ps = _fake_per_seed()
    ps = ps[ps.strategy != "PPO"]
    assert both_formats(plots.f9_seed_distributions(ps, r["metrics"], tmp_path, "t", "test"))


def _fake_stats():
    rows = []
    for algo in ("PG", "PPO", "DDPG"):
        for base in ("UCRP", "UBAH", "Markowitz", "BestStock"):
            rows.append({"algo": algo, "baseline": base, "level": "seeds", "m": 12,
                         "n": 10, "n_better": 4, "p": 0.2, "p_adj": 1.0,
                         "ret_diff": -1e-4, "ret_lo": -5e-4, "ret_hi": 3e-4,
                         "ret_lo_adj": -8e-4, "ret_hi_adj": 6e-4})
    return pd.DataFrame(rows)


def test_f10_writes_png_and_pdf(tmp_path):
    assert both_formats(plots.f10_forest(_fake_stats(), tmp_path, "t"))


def test_f10_renders_without_adjusted_columns(tmp_path):
    """stats_*.csv produced without an adjusted_alpha must still plot."""
    df = _fake_stats().drop(columns=["ret_lo_adj", "ret_hi_adj"])
    assert both_formats(plots.f10_forest(df, tmp_path, "t"))


def test_f10_ignores_non_seed_level_rows(tmp_path):
    """The table also holds per-seed and median rows; the forest is the seed level."""
    df = pd.concat([_fake_stats(),
                    _fake_stats().assign(level="median"),
                    _fake_stats().assign(level="seed0")], ignore_index=True)
    assert both_formats(plots.f10_forest(df, tmp_path, "t"))
