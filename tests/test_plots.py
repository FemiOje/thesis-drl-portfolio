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
