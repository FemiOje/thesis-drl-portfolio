# Reinforcement Learning for Stock Market Portfolio Management

B.Sc. Computer Science final year project — University of Lagos.
Ojetokun Oluwafemi Akinwale (190805019). Supervisor: Dr B. A. Sawyerr.

A comparative evaluation of three continuous-action deep reinforcement learning
algorithms — **Policy Gradient (PG)**, **Proximal Policy Optimization (PPO)** and
**Deep Deterministic Policy Gradient (DDPG)** — on an eight-asset S&P 500 portfolio
under realistic transaction costs.

The research question is a replication one: Liang et al. (2018) found PG outperforms
PPO and DDPG on China A-shares, arguing that portfolio value has an explicit
differentiable form so a learned critic contributes only approximation error. **Does
that ordering hold on a developed, more information-efficient market?**

## Documents

| | |
|---|---|
| `docs/IMPLEMENTATION_PLAN.md` | The engineering contract — locked parameters, module contracts, phase gates |
| `1706.10059.pdf` | Jiang, Xu & Liang (2017) — the mathematical formalism. **Cite v1**; later versions renumber |
| `1808.09940v3.pdf` | Liang et al. (2018) — the template being replicated |
| `1909.09571 ....pdf` | Filos (2019) — an MEng dissertation on the same problem |

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
data/raw/   per-ticker CSVs + MANIFEST.json — COMMITTED, so results reproduce
src/        library code
scripts/    numbered entry points, 01 -> 06
tests/      the gates
results/    one directory per run: config.json, per-seed CSVs, figures
```

Nothing reads a magic number: every parameter comes from `config/base.yaml` via
`src/config.py`, which validates it at load time.

## Status

| Phase | | |
|---|---|---|
| 0 | Scaffold | complete, pending the Python version decision |
| 1 | Data pipeline and frozen universe | not started |
| 2 | Costs, environment, extractor | not started |
| 3 | Baselines | not started |
| 3.5 | Figure pipeline | not started |
| 4 | PG agent | not started |
| 5 | PPO and DDPG | not started |
| 6 | Full experiment and sweeps | not started |
| 7 | Adversarial ablation and write-up | not started |
