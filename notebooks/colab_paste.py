# =============================================================================
# DRL Portfolio Management - results, as two pasteable Colab cells.
#
# Equivalent to notebooks/colab_results.ipynb, minus the markdown commentary.
# Prefer the notebook (File -> Upload notebook) when showing this to a reader;
# this file exists for the case where pasting is easier than uploading.
#
# CELL 1 - paste and run:
#
#   !pip install -q "stable-baselines3==2.9.0" "gymnasium==1.0.0" yfinance
#
# Do NOT pin numpy below 2. Colab's pandas and torch are compiled against
# numpy 2.x, and downgrading numpy breaks them with
#   ValueError: numpy.dtype size changed, may indicate binary incompatibility
# stable-baselines3 2.9 declares numpy<3.0,>=1.20, so numpy 2.x is supported.
# The numpy<2 pin in requirements.txt records the local environment that
# produced these results; it is not a requirement of the code.
#
# CELL 2 - paste everything below this banner and run it.
# =============================================================================

# --- fetch -------------------------------------------------------------------
%cd /content
!rm -rf thesis-drl-portfolio
!git clone -q https://github.com/FemiOje/thesis-drl-portfolio.git
%cd thesis-drl-portfolio

import glob
import subprocess

import numpy
import pandas as pd
from IPython.display import Image, Markdown, display

print("numpy:", numpy.__version__, " pandas:", pd.__version__)
print("commit:", subprocess.run(["git", "log", "--oneline", "-1"],
                                capture_output=True, text=True).stdout.strip())
print("price CSVs:  ", len(glob.glob("data/raw/*.csv")), "(expect 8)")
print("checkpoints: ", len(glob.glob("results/models_conv/*/*/best_model.zip")), "(expect 12)")
# The CSV count alone is not enough: yfinance re-downloads 8 files if the repo
# copies are absent. The commit must be the one that tracks them.

# --- environment tests -------------------------------------------------------
!python -m pytest tests/ -q

# --- evaluate all three algorithms on the held-out test window ---------------
# Primary sweep: weight-shared conv encoder, 500k steps, action bound +/-5.
# PPO and A2C have 5 seeds; DDPG has 2 (it froze identically on both), so its
# missing seeds are skipped with a warning - that is expected, not an error.
!python experiments/evaluate.py --split test \
    --algos ppo a2c ddpg --seeds 0 1 2 3 4 \
    --models-dir results/models_conv \
    --out-dir results/evaluation_conv/test \
    --action-bound 5

display(Markdown(open("results/evaluation_conv/test/RESULTS.md").read()))

pd.set_option("display.width", 200)
display(pd.read_csv("results/evaluation_conv/test/per_seed_results.csv"))

for f in ["01_wealth_curves.png", "02_allocation_heatmaps.png", "03_seed_distributions.png"]:
    display(Image(filename="results/evaluation_conv/test/" + f))

# --- before / after: what the feature extractor changed ----------------------
# SB3's MultiInputPolicy gives a Dict entry a CNN only if it is an image space.
# The (3, 50, 8) price tensor is not one, so it took the nn.Flatten() branch:
# 1200 unrelated inputs, no convolution anywhere in the trained models.
comparison = pd.DataFrame(
    {
        "Ablation (flatten, 300k, +/-10)": ["1.1507 +/- 0.0108", "1.1033 +/- 0.1532",
                                            "1.1450 +/- 0.0603", "1.1506", "10 of 10"],
        "Primary (conv, 500k, +/-5)":      ["1.2074 +/- 0.0893", "1.1982 +/- 0.0341",
                                            "1.1720 (n=2)",      "1.1506", "0 of 10"],
    },
    index=["PPO test fAPV", "A2C test fAPV", "DDPG test fAPV",
           "Buy & Hold", "Frozen policies (PG methods)"],
)
display(comparison)
display(Markdown("**Ablation results in full, for reference:**"))
display(Markdown(open("results/evaluation/test/RESULTS.md").read()))

# --- policy diagnostic -------------------------------------------------------
# fAPV cannot distinguish a trading policy from a constant one that happened to
# do well. This probes every checkpoint on nine widely separated market regimes
# with w_prev held fixed, so any change is attributable to the price tensor.
!python experiments/policy_diagnostic.py \
    --algos ppo a2c ddpg --seeds 0 1 2 3 4 \
    --models-dir results/models_conv \
    --split test --action-bound 5

# --- training curves ---------------------------------------------------------
!python experiments/plot_training.py --models-dir results/models_conv \
    --out-dir results/evaluation_conv

display(Image(filename="results/evaluation_conv/training_curves.png"))
