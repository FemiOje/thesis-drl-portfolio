# Architecture table — Phase 5 gate

The comparison claims one free axis: the learning rule. This records what was actually
held identical, what differs, and which of the differences were chosen rather than
inherited from the algorithms. Every row is traceable to code or to a committed run
directory.

Runs covered: `results/pg`, `results/ppo`, `results/ddpg` (assembled as
`results/phase5`), 10 seeds each, universe `M8_basket1`.

---

## 1. Controlled variables

| Quantity | PG | PPO | DDPG | Source |
|---|---|---|---|---|
| Feature extractor | `EIIEExtractor` | `EIIEExtractor` | `EIIEExtractor` | `extractors.py:13`; asserted at construction, `sb3.py:24` |
| Extractor parameters | 783 | 783 | 783 | asserted per instance, incl. both DDPG targets |
| Extractor instances | 1 | 2 (pi, vf) | 4 (actor, critic, 2 targets) | `share_features_extractor=False` |
| Observation | `Dict(tensor (3,8,20), weights (9,))` | same | same | `env.py`, Eq. 20 |
| Asset axis | never convolved, flattened or reduced | same | same | all kernels height 1, `extractors.py:18-20` |
| Action space | `Box(-1, 1, (9,))` | same | same | `env.py` |
| Projection to simplex | `softmax(tau * a)` | same | same | `project_to_simplex`, `env.py:10` — one function, imported by env, PG actor and the reachability test |
| `tau` | 5.0 | 5.0 | 5.0 | `config/base.yaml` `env.tau` |
| `gamma` | 1.0 | 1.0 | 1.0 | `agent.gamma`; asserted `sb3.py:193` (SB3 default 0.99 would confound the learning rule with the horizon) |
| Commission | 0.001 both sides | same | same | `env.commission` |
| Cost fixed-point iterations | 5 | 5 | 5 | `env.mu_iterations`, Eq. 15 |
| Reward | `log(mu_t * y_t . w_{t-1})` | same | same | `env.py:65`, Eq. 10 |
| Minibatch size | 50 | 50 | 50 | `batch_sizes()`, `config.py:80` |
| Optimiser updates | 60,000 | 60,000 | 60,000 | predicted by `gradient_steps()`; realised count asserted, §4 |
| Seeds | 0-9 | 0-9 | 0-9 | `run.seeds` |
| Splits | 735 / 249 / 251 | same | same | chronological, by decision date |
| Evaluations per run | 40 | 40 | 40 | every 1,500 gradient steps — shared F1/F2/F4 x-axis |
| Checkpoint rule | argmax validation final wealth | same | same | `pg.py`, `sb3.py:167` |
| Evaluation | frozen policy, separate env instance | same | same | `sb3.py:181` |

### Extractor parameter budget

The 783 is the controlled variable. `assert_eiie` checks the count, not just the class,
because `isinstance` alone still passes with a wrong `n_assets`/`window`/`n_features`.

| Layer | Shape | Params |
|---|---|---|
| `conv1` | `Conv2d(3, 2, (1,3))` | 18 + 2 |
| `conv2` | `Conv2d(2, 20, (1,18))` | 720 + 20 |
| `conv3` | `Conv2d(21, 1, (1,1))` | 21 + 1 |
| `cash_bias` | `(1,)` | 1 |
| **Total** | | **783** |

`conv3`'s 21st input channel is `w_{t-1}` over the risky assets. All biases are
zero-initialised: Eq. 18 inputs sit near 1.0, and with 2 filters a random bias kills whole
ReLU channels in 14/50 seeds (`extractors.py:22`).

---

## 2. Asymmetries

### 2.1 `share_features_extractor = False` — chosen

Set explicitly for PPO and DDPG. PG's extractor receives only the Eq. 21 gradient. PPO's
SB3 default (`True`) would additionally push `0.5 * value-MSE` into the same 783 weights
that emit the portfolio, so the controlled variable would be optimised differently in one
arm. `False` is already SB3's DDPG default and matches Liang, whose actor and critic
convolutions live in separate variable scopes. Cost: PPO holds 2 extractor copies and DDPG
4, all asserted identical in class and size.

### 2.2 Critic `net_arch` — inherent

| Arm | `pi` | `qf` | Actor params | Critic params |
|---|---|---|---|---|
| PG | — (extractor output is the action) | no critic | 783 | — |
| PPO | `[]` | `[]` (linear value head) | 882 | 10 |
| DDPG | `[]` | `[64, 64]` | 873 | 6,224 |

DDPG's critic must fuse features *and* the action — input is 9 features + 9 action dims —
so `qf` needs real capacity. PPO's value head is linear on the 9-dim extractor output.
This asymmetry is inherent to the algorithms being compared, which is the point of the
study: reported, not hidden.

### 2.3 Policy head — inherent, **not previously documented**

PG's actor *is* the extractor: `tanh(extractor(obs))`, 783 parameters total, no head
(`pg.py:52`). SB3 always interposes an `action_net`, so both SB3 arms carry a `Linear(9,9)`
= 90 parameters that PG does not have, and PPO carries a further 9-parameter
state-independent `log_std`:

| Arm | Extractor | Head | log_std | Actor total |
|---|---|---|---|---|
| PG | 783 | 0 | — | **783** |
| PPO | 783 | 90 | 9 | **882** |
| DDPG | 783 | 90 | — | **873** |

`sb3.py:104` describes `pi=[]` as keeping the actor "a linear head on the extractor, as
PG's is". That is not exact: PG has no head at all. The extractor is identical and
asserted; the map from its 9 outputs to the action is not. SB3 cannot be configured to
drop `action_net`, so this is structural rather than a choice — but it means "identical
architecture" is true of the *extractor* only, and Chapter 3 should say so in those terms.

### 2.4 Data consumption and exploration — inherent

| | PG | PPO | DDPG |
|---|---|---|---|
| Env steps collected | 0 | 300,000 | 300,000 |
| Training data | replays the fixed 735-day train split via PVM | on-policy rollouts | replay buffer |
| Exploration | none (deterministic) | stochastic policy, learned `log_std` | OU noise, `theta=0.15`, `sigma=0.20` |
| Critic | none | value head | single Q, `policy_delay=1` |

Equalising gradient steps does not equalise environment interaction, and cannot: PG
collects none by construction. The budget is equalised on the axis that was in our gift
(optimiser updates of identical batch size); the axis that is not is recorded here.

---

## 3. Per-algorithm hyperparameters

| | PG | PPO | DDPG |
|---|---|---|---|
| Optimiser | Adam, `weight_decay=1e-8` | Adam (SB3 default) | Adam (SB3 default) |
| Learning rate | 3e-5 (Jiang Table B.1) | 3e-4 | 1e-3 |
| `total_timesteps` | — | 300,000 | 300,000 |
| `n_steps` / `train_freq` | — | 500 | 500 |
| `n_epochs` / `gradient_steps` | — | 10 | 100 |
| `learning_starts` | — | — | 250 |
| Eval interval | 1,500 grad steps | 7,500 env steps | 7,500 env steps |

PG's learning rate is Jiang's published value. Liang's are not citable: Table I gives actor
1e-3 / critic 1e-1 and `ddpg.py` gives the reverse, so PPO and DDPG use SB3 defaults, set
explicitly and tuned on validation only.

---

## 4. Budget arithmetic and realised evidence

Predicted, in minibatches of 50 drawn and backpropagated (`config.gradient_steps()`):

- **PG** — 60,000 updates, one per sampled batch.
- **PPO** — `(300,000 / 500) rollouts x 10 epochs x ceil(500/50) minibatches` = 600 x 10 x 10 = **60,000**.
- **DDPG** — `(300,000 / 500) triggers x 100 updates` = 600 x 100 = **60,000**.

Prediction is not evidence, so the realised count is asserted against it at the end of
every run (`sb3.py:203`). The counter wraps `optimizer.step` directly and deliberately does
**not** use `model._n_updates`: for PPO that increments once per epoch (`ppo.py:284`,
outside the minibatch loop) and would under-count by `n_steps/batch_size` = 10x. For DDPG
the critic optimiser is quoted; `policy_delay=1` gives the actor an equal number.

Realised: `meta.json` `realised_updates` records **60,000 for all 10 seeds of both PPO and
DDPG**.

---

## 5. Provenance and known gaps

- Facts above were read from `config/base.yaml`, `src/extractors.py`,
  `src/agents/{pg,sb3}.py`, `src/env.py`, `src/config.py` and the three `meta.json` files.
  Parameter counts were computed by constructing each policy against the real observation
  space, not derived by hand.
- Gate assertions live in `tests/test_sb3.py`: extractor identity including DDPG's critic,
  parameter-for-parameter match with PG, the flattening trap, non-sharing, `gamma`, the
  realised update count, the `_n_updates` trap, and the separate eval env. Full suite: 136
  passed.
- **Gap:** `meta.json` `optimizer_steps` — the per-optimiser breakdown — is `{}` in all
  three run directories and in the per-seed `.npz` files. All ten seeds are listed under
  `resumed`, so these artefacts were written by an earlier revision of `05_train_sb3.py`
  than the one now on disk; the mechanism itself is green in tests. The substantive
  evidence (`realised_updates` = 60,000, asserted at run time) is intact, but if the
  breakdown is wanted as a committed artefact, one `--fresh` seed per algorithm
  repopulates it.
