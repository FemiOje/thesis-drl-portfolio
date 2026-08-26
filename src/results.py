"""Load run directories into the shape the figure functions expect.

Curves are always (n_seeds, T). Baselines have n_seeds = 1, so F5-F7 need no change
when per-seed agent runs arrive in Phase 6 — the IQR band simply collapses.
"""

import numpy as np
import pandas as pd

SPLITS = ("train", "validate", "test")


def load_baselines(run_dir):
    z = np.load(run_dir / "curves.npz", allow_pickle=False)
    dates = pd.DatetimeIndex([d.decode() if isinstance(d, bytes) else str(d)
                              for d in z["dates"]])
    curves = {s: {} for s in SPLITS}
    for key in z.files:
        for s in SPLITS:
            if key.startswith(f"{s}_"):
                curves[s][key[len(s) + 1:]] = z[key][None, :]     # (1, T)
    split_dates = {s: dates[z[f"idx_{s}"]] for s in SPLITS}
    metrics = pd.read_csv(run_dir / "baselines.csv")
    return {"curves": curves, "dates": split_dates, "metrics": metrics}


def load_agents(run_dir):
    """Phase 6: per-seed curves as results/<run_id>/<algo>/<split>.npy of shape
    (n_seeds, T), plus metrics.csv with the same columns as baselines.csv."""
    curves = {s: {} for s in SPLITS}
    for algo_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        for s in SPLITS:
            f = algo_dir / f"{s}.npy"
            if f.exists():
                curves[s][algo_dir.name] = np.atleast_2d(np.load(f))
    csv = run_dir / "metrics.csv"
    return {"curves": curves,
            "metrics": pd.read_csv(csv) if csv.exists() else pd.DataFrame()}


def merge(base, agents):
    """Agents drawn on the same axes as the baselines they must beat."""
    out = {"curves": {s: dict(base["curves"][s]) for s in SPLITS},
           "dates": base["dates"],
           "metrics": base["metrics"]}
    if not agents:
        return out
    for s in SPLITS:
        out["curves"][s].update(agents["curves"].get(s, {}))
    if len(agents["metrics"]):
        out["metrics"] = pd.concat([base["metrics"], agents["metrics"]], ignore_index=True)
    return out


def load_history(run_dir):
    """Training diagnostics per algorithm: results/<run_id>/<algo>/history.npz."""
    out = {}
    if not run_dir.is_dir():
        return out
    for d in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        f = d / "history.npz"
        if f.exists():
            out[d.name] = {k: v for k, v in np.load(f).items()}
    return out
