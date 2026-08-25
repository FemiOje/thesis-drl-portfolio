"""Regenerate the figure suite from committed run directories.

Draws whatever the run directories support and reports what it skipped, so the same
command works in Phase 3.5 (baselines only) and in Phase 7 (everything).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import plots, results
from src.config import PROJECT_ROOT

PHASE3 = PROJECT_ROOT / "results" / "phase3"
FIGS = PROJECT_ROOT / "results" / "figures"


def main(run_id="baselines"):
    if not (PHASE3 / "curves.npz").exists():
        raise SystemExit("no baseline results; run scripts/02_run_baselines.py first")
    base = results.load_baselines(PHASE3)

    agent_dir = PROJECT_ROOT / "results" / run_id
    agents = results.load_agents(agent_dir) if agent_dir.is_dir() and run_id != "baselines" else None
    r = results.merge(base, agents)

    done = []
    done.append(plots.f5_test_wealth(r["dates"]["test"], r["curves"]["test"], FIGS, run_id))
    done.append(plots.f6_train_val_wealth(r["dates"], r["curves"], FIGS, run_id))
    done.append(plots.f7_wealth_and_drawdown(r["dates"]["test"], r["curves"]["test"],
                                             FIGS, run_id))
    for split in ("test", "validate", "train"):
        done.append(plots.f8_metric_heatmap(r["metrics"], FIGS, run_id, split))

    for p in done:
        print(f"  {p.relative_to(PROJECT_ROOT)}  (+ .pdf)")
    n_strat = len(r["curves"]["test"])
    print(f"\n{len(done)} figures, {n_strat} strategies, run_id={run_id}")
    if agents is None:
        print("skipped F1-F4, F9-F17: no agent runs yet (Phase 4 onward)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "baselines")
