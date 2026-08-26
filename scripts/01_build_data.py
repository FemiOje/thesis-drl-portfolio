"""Phase 1: resolve universes, download, gate, build tensors, render F0a-F0c."""

import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data as D
from src import plots, universe as U
from src.config import PROJECT_ROOT, load_config

FIG_DIR = PROJECT_ROOT / "results" / "phase1" / "figures"
MARKER = "# Pre-registered before any run."


def _block(name, d):
    body = yaml.safe_dump(d, sort_keys=False, width=100, default_flow_style=False)
    return name + ":\n" + textwrap.indent(body, "  ")


def write_back(spec, gates, universes):
    """Record the gate outcome and the resolution in universe.yaml. Idempotent: the
    pre-registered head above MARKER is never touched."""
    txt = U.UNIVERSE_YAML.read_text(encoding="utf-8")
    gate = dict(spec["correlation_gate"])
    gate["result"] = gates
    resolved = {"tickers": universes,
                "market_caps_as_of": str(spec["rule"]["as_of"]),
                "resolved_on": str(np.datetime64("today"))}
    U.UNIVERSE_YAML.write_text(
        txt.split(MARKER)[0]
        + MARKER + " Recorded here so the commit history proves it was\n"
        + "# fixed in advance rather than chosen after seeing results.\n"
        + "# `result` and `resolved` below are written by scripts/01_build_data.py.\n"
        + _block("correlation_gate", gate) + "\n" + _block("resolved", resolved),
        encoding="utf-8")


def main():
    cfg = load_config()
    spec = U.load_spec()
    cands = U.all_tickers(spec)

    print(f"downloading {len(cands)} candidates {cfg.data.start} -> {cfg.data.end}")
    rows = D.download(cands, cfg.data.start, cfg.data.end, cfg.data.raw_dir)

    panel_all = D.load_panel(cands, cfg.data.raw_dir, cfg.data.start, cfg.data.end)
    dates_all = panel_all["close"].index
    train = cfg.data.split("train")
    tr = (dates_all >= str(train.start)) & (dates_all <= str(train.end))
    train_ret = panel_all["close"].pct_change().iloc[1:][tr[1:]]

    # ---- resolve every universe and gate it ----
    universes, gates = {}, {}
    for name, (m, rank) in {"M8_basket1": (8, 1), "M8_basket2": (8, 2),
                            "M8_basket3": (8, 3), "M4": (4, 1),
                            "M16": (16, 1)}.items():
        tk = U.resolve(spec, m, rank)
        g = U.apply_gate(spec, tk, train_ret)
        universes[name], gates[name] = g["final_tickers"], g
        flag = "PASS" if g["passed"] else ("SUBSTITUTED" if g["substitutions"] else "REPORT-ONLY")
        print(f"  {name:12s} {flag:12s} max_corr={g['max_corr']:.3f} "
              f"{g['max_pair']}  {g['final_tickers']}")

    headline = universes["M8_basket1"]

    # ---- build tensors ----
    built = {}
    for name, tk in universes.items():
        ds = D.build_dataset(tk, cfg)
        D.save_dataset(ds, cfg.data.processed_dir / f"{name}.npz")
        built[name] = ds
    ds = built["M8_basket1"]
    X, dates = ds["X"], ds["dates"]

    # ---- invariants ----
    checks = {}
    checks["no_nan_raw"] = not any(panel_all[f].isna().any().any() for f in panel_all)
    checks["no_nan_tensor"] = bool(np.isfinite(X).all())
    checks["no_nan_y"] = bool(np.isfinite(ds["y"][1:]).all())
    checks["tensor_shape"] = list(X.shape[1:])
    checks["last_close_column_is_one"] = bool(np.allclose(X[:, 0, :, -1], 1.0, atol=1e-6))
    checks["tensor_positive"] = bool((X > 0).all())
    # causality: X[i] is built from close[t-w+1 .. t] with t = i+w-1, so the highest
    # source index equals the decision index. Verified against a direct recompute.
    close = ds["panel"]["close"].values
    w = cfg.env.window
    i = len(X) - 1
    t = i + w - 1
    checks["causal_lookback"] = bool(
        np.allclose(X[i, 0], (close[t - w + 1:t + 1] / close[t]).T, atol=1e-6))
    checks["split_sizes"] = {k: int(len(v)) for k, v in ds["splits"].items()}
    checks["split_disjoint"] = len(set().union(*[set(v.tolist()) for v in ds["splits"].values()])) \
        == sum(len(v) for v in ds["splits"].values())
    checks["first_date"] = str(dates[0].date())
    checks["last_date"] = str(dates[-1].date())
    checks["config_end"] = str(cfg.data.end)
    checks["n_decision_dates"] = int(len(dates))

    print("\ninvariants:")
    for k, v in checks.items():
        print(f"  {k:26s} {v}")
    bad = [k for k, v in checks.items() if isinstance(v, bool) and not v]
    if bad:
        raise SystemExit(f"FAILED: {bad}")

    # ---- figures ----
    hp = ds["panel"]
    plots.f0a_prices(hp["close"], cfg.data.splits, FIG_DIR, "phase1")
    plots.f0b_correlation(train_ret[headline], spec["correlation_gate"]["threshold"],
                          FIG_DIR, "phase1")
    plots.f0c_tensor(X, headline, dates, int(ds["splits"]["train"][-1]), FIG_DIR, "phase1")
    print(f"\nfigures -> {FIG_DIR}")

    # ---- write back universe.yaml + MANIFEST ----
    write_back(spec, gates, universes)

    D.write_manifest(
        cfg.data.raw_dir / "MANIFEST.json", cands, rows, gates,
        {"window": str(cfg.data.start) + ".." + str(cfg.data.end),
         "universes": universes, "checks": checks,
         "access_notes": [
             "yfinance pinned 0.2.54 (last requests-only release): curl_cffi, required "
             "by >=0.2.55, is blocked by Windows Application Control on this machine.",
             "yfinance's cookie/crumb bootstrap (fc.yahoo.com 404, /v1/test/getcrumb 401) "
             "is unreachable here and is misreported as a rate limit; bypassed in "
             "src/data.py:_patch_yfinance. The v8 chart endpoint needs no crumb.",
             "Qlib cross-check: not run (optional control, plan §3).",
         ],
         "market_cap_provenance": "pre-registered constants in config/universe.yaml; "
                                  "AAPL/MSFT and JNJ/UNH flagged close_call"})
    print(f"manifest -> {cfg.data.raw_dir / 'MANIFEST.json'}")
    print(json.dumps({"headline": headline}, indent=2))


if __name__ == "__main__":
    main()
