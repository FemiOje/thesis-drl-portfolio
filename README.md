# Reinforcement Learning for Stock Market Portfolio Management

B.Sc. Computer Science final year project, University of Lagos.
Ojetokun Oluwafemi Akinwale (190805019)

A comparative evaluation of three continuous-action deep reinforcement learning
algorithms — **Policy Gradient (PG)**, **Proximal Policy Optimization (PPO)** and
**Deep Deterministic Policy Gradient (DDPG)** — on an eight-asset S&P 500 portfolio
under realistic transaction costs.


## Setup

Requires **Python 3.10** (see `requirements.txt` for why).

```bash
py -3.10 -m venv .venv
.venv/Scripts/activate      # Windows;  source .venv/bin/activate on POSIX
pip install -r requirements.txt
pytest
```

## Layout

```
config/     base.yaml (locked parameters) and universe.yaml (the selection rule)
data/raw/   per-ticker CSVs + MANIFEST.json
src/        library code
scripts/    numbered entry points, 01 -> 06
tests/      the gates
results/    one directory per run: config.json, per-seed CSVs, figures
```

every parameter comes from `config/base.yaml` via
`src/config.py`, which validates it at load time.
