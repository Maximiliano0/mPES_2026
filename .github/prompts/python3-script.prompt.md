# Python 3 Script Template

> Last updated: 2026-02-26

Generate a new Python 3 script following the mPES project conventions.

## Packages

| Package | Description |
|---------|-------------|
| `PES` | Tabular Q-Learning baseline |
| `PES_Bayesian` | Q-Learning + Bayesian optimisation (Optuna) |
| `PES_QLv2` | Double Q-Learning, ε-decay warm-up, PBRS |
| `PES_Transformer` | Causal Transformer encoder + RL |
| `utils` | Shared helpers |

## Structure

The generated script MUST follow this structure, in order:

1. **Module docstring** — Triple-quoted description of what the script does.
2. **External imports** — Under a `## Imports externos ##` section header comment.
3. **Internal imports** — Under a `## Imports internos ##` section header comment.
   Use **explicit** relative imports (`from .. import X, Y`, `from .module import Z`).
   Never use `from .. import *` outside `__init__.py`.
4. **Constants / configuration** — If needed, under a `## Configuración ##` section header.
5. **Functions / Classes** — Core logic, each with a NumPy-style docstring.
6. **`main()` function** — Entry point with a one-line docstring and the primary workflow.
7. **`if __name__ == '__main__': main()`** — Guard at the bottom.

## Conventions

- Use `numpy` instead of `np` for the numpy alias.
- Use `os.path.join()` for paths, never f-strings with `/`.
- Use `datetime.now().strftime("%Y-%m-%d")` for date stamps.
- Use terminal utilities from `..src.terminal_utils`: `header`, `section`, `success`, `info`, `list_item`.
- Save outputs to a date-stamped subdirectory under the package's `inputs/` or `outputs/` folder.
- Prefix unused variables with `_` (e.g., `_fig, ax = plt.subplots()`).
- PascalCase variables are acceptable (project convention for scientific code).
- 120-character max line length, 4-space indentation.
- Suppress non-critical warnings at the top:
  ```python
  warnings.filterwarnings('ignore', category=UserWarning, message='.*Box bound precision.*')
  ```
- Force CPU for TensorFlow if imported:
  ```python
  os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
  ```

## Quality gates

After generating the script, run:

1. `source linux_mpes_env/bin/activate && pyright <PACKAGE_DIR>/` → 0 errors
2. `source linux_mpes_env/bin/activate && pylint --rcfile=.pylintrc <PACKAGE_DIR>/` → 10.00/10

## Example skeleton

```python
'''
<PKG_NAME> — <Short description>

<Longer explanation of what this script does.>
'''

##########################
##  Imports externos    ##
##########################
import os
import sys
import numpy
import warnings
from datetime import datetime

##########################
##  Imports internos    ##
##########################
from .. import INPUTS_PATH
from ..src.terminal_utils import header, section, success, info, list_item

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*Box bound precision.*')

###################################
##             Main             ###
###################################
def main():
    """Run the <description> pipeline."""

    header("<SCRIPT TITLE>", width=80)

    run_date = datetime.now().strftime("%Y-%m-%d")
    out_dir = os.path.join(INPUTS_PATH, f'{run_date}_<SUFFIX>')
    os.makedirs(out_dir, exist_ok=True)
    info(f"Output directory: {out_dir}")

    # ... core logic ...

    section("Complete", width=80)
    success("Pipeline finished successfully!")
    info(f"Output directory: {out_dir}")


if __name__ == '__main__':
    main()
```

## Rules

- Adapt the package name (`PES`, `PES_Bayesian`, `PES_QLv2`, `PES_Transformer`)
  based on where the user wants to place the script.
- Do NOT cross-reference between packages — each package is self-contained.
- Ask the user for the script's purpose if not specified.
