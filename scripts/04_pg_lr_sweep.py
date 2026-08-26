"""Phase 4 diagnostic: PG learning rate vs the generalisation gap.

With 735 training days the gap, not the optimiser, is the binding constraint. Recorded
here because it is what Phase 6's hyperparameter selection has to act on.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LRS = ["3e-5", "3e-4", "3e-3"]

for lr in LRS:
    print(f"\n===== lr = {lr} =====", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "03_train_pg.py"),
                    "--seeds", "3", "--lr", lr, "--run-id", f"pg_lr{lr}"], check=True)
