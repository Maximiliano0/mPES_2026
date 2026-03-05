# GitHub Copilot Instructions

> Last updated: 2026-03-04

## Project Overview

**mPES** (multiple Pandemic Experiment Scenario) is a multi-package Python
workspace for reinforcement-learning experiments on a resource-allocation task
(the "Pandemic Scenario"). Each package implements a different algorithmic
variant sharing the same experiment framework.

| Package | Algorithm | Key files |
|---------|-----------|-----------|
| `pes` | Tabular Q-Learning (baseline) | `ext/pandemic.py`, `ext/train_rl.py` |
| `pes_base_line` | Q-Learning + Bayesian hyperparam optimisation (Optuna) | `ext/optimize_rl.py` |
| `pes_qlv2` | Double Q-Learning, ε-decay warm-up, PBRS | `ext/pandemic.py`, `ext/optimize_rl.py` |
| `pes_dqn` | Deep Q-Network (experience replay + target net) | `ext/dqn_model.py`, `ext/train_dqn.py`, `ext/optimize_dqn.py` |
| `pes_actor_critic` | Advantage Actor-Critic (A2C, separate actor + critic nets) | `ext/ac_model.py`, `ext/train_ac.py`, `ext/optimize_ac.py` |
| `pes_transformer` | Causal Transformer encoder + RL | `ext/transformer_model.py`, `ext/train_transformer.py`, `ext/optimize_tr.py` |
| `utils` | Shared helpers (notifications, shell scripts) | `notify.py` |

### Common package layout

```
<PKG>/
├── __init__.py          # Config re-exports, ANSI codes, numpy/TF setup
├── __main__.py          # Experiment entry point (blocks/sequences/trials)
├── config/CONFIG.py     # All tuneable constants
├── doc/                 # Markdown documentation
├── ext/                 # Core algorithms (Gym env, training, optimisation)
├── inputs/              # Generated data (date-stamped subdirs)
├── outputs/             # Logs and results (date-stamped subdirs)
└── src/                 # Support modules
    ├── exp_utils.py       # Severity calculations, sequence helpers
    ├── log_utils.py       # Dual-stream logging (console + file)
    ├── pygameMediator.py  # Pygame UI bridge
    ├── result_formatter.py # Matplotlib result plots
    └── terminal_utils.py  # Rich console output (header, section, info…)
```
## Virtual Environment

Always activate `linux_mpes_env` before running any tool:
```bash
source linux_mpes_env/bin/activate
```
Python 3.12 · dependencies listed in `requirements.txt`.

## Code Style

| Rule | Standard |
|------|----------|
| Max line length | 120 characters |
| Indentation | 4 spaces (PEP 8) |
| Variable naming | `snake_case` **and** `PascalCase` accepted (scientific convention) |
| NumPy alias | `numpy` (never `np`) |
| Type hints | Use where practical; pyright must pass with 0 errors |
| Docstrings | NumPy-style, required on every public function and class |
| Unused vars | Prefix with `_` (e.g., `_fig, ax = plt.subplots()`) |

### Import conventions

- **`__init__.py`** may use `from .config.CONFIG import *` (wildcard re-export).
- **All other modules** must use explicit imports:
  ```python
  from .. import ANSI, INPUTS_PATH, VERBOSE   # ✅
  from .. import *                              # ❌ (except __init__.py)
  ```
- Section comments above import blocks:
  ```python
  ##########################
  ##  Imports externos    ##
  ##########################

  ##########################
  ##  Imports internos    ##
  ##########################
  ```

## Functionality

Implement features and fixes that align with the project's goals and
requirements. Avoid adding unrelated functionality. Each package is
self-contained — do **not** cross-reference between packages.

## Testing

Ensure that any new code is properly tested and does not break existing
functionality. Write unit tests where appropriate.

## Documentation

- Every public function and class must have a docstring (NumPy-style).
- Maintain clear and concise documentation for any new features or changes.
- Update existing documentation if necessary to reflect changes.

## Version Control

Do not commit or push changes directly. Instead, provide code suggestions
and improvements through pull requests for review by the project maintainers.