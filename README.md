# mPES — multiple Pandemic Experiment Scenario

Multi-package Python workspace for **reinforcement-learning** experiments on a
resource-allocation task (the *Pandemic Scenario*).

An agent must distribute **39 resources** across ~360 trials (8 blocks × 8
sequences × 3–10 trials) to minimise disease severity. Six algorithmic variants
share the same experiment framework, making side-by-side comparison
straightforward.

## Packages

| Package | Algorithm | Key files |
|---------|-----------|-----------|
| `pes` | Tabular Q-Learning (baseline) | `ext/pandemic.py`, `ext/train_rl.py` |
| `pes_bline` | Q-Learning + Bayesian optimisation (Optuna) | `ext/optimize_rl.py` |
| `pes_qlv2` | Double Q-Learning, ε-decay warm-up, PBRS | `ext/pandemic.py`, `ext/optimize_rl.py` |
| `pes_dqn` | Deep Q-Network (experience replay + target net) | `ext/dqn_model.py`, `ext/train_dqn.py`, `ext/optimize_dqn.py` |
| `pes_ac` | Advantage Actor-Critic (A2C) | `ext/ac_model.py`, `ext/train_ac.py`, `ext/optimize_ac.py` |
| `pes_trf` | Causal Transformer encoder + RL | `ext/transformer_model.py`, `ext/train_transformer.py`, `ext/optimize_tr.py` |
| `utils` | Shared helpers (notifications, shell scripts) | `notify.py`, `run_bayesian_opt.sh`, `run_bayesian_opt.ps1` |

## Package layout

```
<pkg>/
├── __init__.py          # Config re-exports, ANSI codes, numpy/TF setup
├── __main__.py          # Experiment entry point (blocks/sequences/trials)
├── config/CONFIG.py     # All tuneable constants
├── doc/                 # Markdown & HTML documentation
├── ext/                 # Core algorithms (Gym env, training, optimisation)
├── inputs/              # Generated data (date-stamped subdirs)
├── outputs/             # Logs and results (date-stamped subdirs)
└── src/                 # Support modules
    ├── exp_utils.py       # Severity calculations, sequence helpers
    ├── log_utils.py       # Dual-stream logging (console + file)
    ├── pygameMediator.py  # Pygame UI bridge
    ├── result_formatter.py# Matplotlib result plots
    └── terminal_utils.py  # Rich console output (header, section, info…)
```

## Setup

### Requirements

| Dependency | Version |
|------------|---------|
| Python | 3.10 (Windows) / 3.12 (Linux) |
| TensorFlow | 2.16.2 |
| Keras | 3.3.3 |
| NumPy | 1.26.4 |
| matplotlib | 3.8.2 |
| scipy | 1.11.4 |
| Optuna | 4.7.0 |
| Gym | 0.26.1 |
| Pygame | 2.5.2 |

### Virtual environment

```bash
# Linux
python3 -m venv linux_mpes_env
source linux_mpes_env/bin/activate

# Windows (PowerShell)
python -m venv win_mpes_env
win_mpes_env\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Environment variables

Set these **before** running training or optimisation:

| Variable | Value | Purpose |
|----------|-------|---------|
| `VIRTUAL_ENV` | Path to active venv | Prevents `__init__.py` interactive prompt |
| `PYTHONIOENCODING` | `utf-8` | Avoids `UnicodeEncodeError` on Windows |
| `TF_ENABLE_ONEDNN_OPTS` | `0` | Suppresses oneDNN info messages |

## Usage

### Run an experiment

```bash
python -m pes          # Tabular Q-Learning
python -m pes_bline    # Q-Learning (Bayesian-tuned)
python -m pes_qlv2     # Double Q-Learning
python -m pes_dqn      # Deep Q-Network
python -m pes_ac       # Actor-Critic
python -m pes_trf      # Transformer
```

### Train an agent

```bash
# Tabular Q-Learning (1M episodes)
python -m pes.ext.train_rl 1000000

# Deep Q-Network
python -m pes_dqn.ext.train_dqn 500000

# Actor-Critic
python -m pes_ac.ext.train_ac 500000

# Transformer
python -m pes_trf.ext.train_transformer 500000
```

### Bayesian hyperparameter optimisation

```bash
# Linux
./utils/run_bayesian_opt.sh bayesian 100    # pes_bline, 100 trials
./utils/run_bayesian_opt.sh qlv2 100        # pes_qlv2
./utils/run_bayesian_opt.sh dqn 30          # pes_dqn
./utils/run_bayesian_opt.sh transformer 30  # pes_trf

# Windows (PowerShell)
.\utils\run_bayesian_opt.ps1 bayesian 100
.\utils\run_bayesian_opt.ps1 dqn 30
```

## The Pandemic Scenario

- **State space**: `[resources_left (0–30), trial_number (0–10), severity (0–10)]` → 3,751 states
- **Action space**: allocate 0–10 resources (11 discrete actions)
- **Dynamics**: `new_severity = 1.4 × initial_severity − 0.4 × resources_allocated`
- **Reward**: negative cumulative severity (the agent minimises total damage)
- **Infeasible actions**: allocations exceeding remaining resources are masked

## Experiment structure

```
Experimento (1)
├── Bloque (8)
│   ├── Secuencia / Mapa (8)
│   │   ├── Trial / Ciudad (3–10)
│   │   │   └── Decisión de Recursos (0–10)
```

- **1** experiment → **8** blocks → **8** sequences per block → **3–10** trials per sequence
- ~360 total trials per experiment (~45 per block)

## License

Private repository — all rights reserved.
