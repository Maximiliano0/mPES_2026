# GitHub Copilot Instructions

> Last updated: 2026-02-26

## Project Overview

**mPES** (multiple Pandemic Experiment Scenario) is a multi-package Python
workspace for reinforcement-learning experiments on a resource-allocation task
(the "Pandemic Scenario"). Each package implements a different algorithmic
variant sharing the same experiment framework.

| Package | Algorithm | Key files |
|---------|-----------|-----------|
| `PES` | Tabular Q-Learning (baseline) | `ext/pandemic.py`, `ext/train_rl.py` |
| `PES_Bayesian` | Q-Learning + Bayesian hyperparam optimisation (Optuna) | `ext/optimize_rl.py` |
| `PES_QLv2` | Double Q-Learning, ε-decay warm-up, PBRS | `ext/pandemic.py`, `ext/optimize_rl.py` |
| `PES_Transformer` | Causal Transformer encoder + RL | `ext/transformer_model.py`, `ext/train_transformer.py` |
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

## Code Quality — Mandatory Workflow

After **every** code modification, run the following checks **in order**:

1. **pyright** — static type checking
   ```bash
   source linux_mpes_env/bin/activate && pyright <PACKAGE_DIR>/
   ```
   Fix every error, warning, and information-level issue before proceeding.

2. **pylint** — linting with project standard
   ```bash
   source linux_mpes_env/bin/activate && pylint --rcfile=.pylintrc <PACKAGE_DIR>/
   ```
   Fix every reported issue. The `.pylintrc` at the project root defines the
   correction standard for all packages. Do **not** suppress enforced rules —
   fix them in source.

3. **Coherence audit** — verify the changed module still makes sense:
   - Imports resolve and are used.
   - Public functions and classes have NumPy-style docstrings.
   - No trailing whitespace or missing final newlines.
   - No unused variables/arguments (prefix with `_` if intentionally unused).
   - Changed code is consistent with the rest of the package.

**Target:** `10.00/10` pylint score, `0 errors` in pyright.

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