# Python 3 Script Template

Generate a new Python 3 script following the project conventions of mPES.

## Structure

The generated script MUST follow this structure, in order:

1. **Module docstring** — Triple-quoted description of what the script does.
2. **External imports** — Under a `## Imports externos ##` section header comment.
3. **Internal imports** — Under a `## Imports internos ##` section header comment. Use relative imports (`from .. import`, `from .module import`).
4. **Constants / configuration** — If needed, under a `## Configuración ##` section header.
5. **Functions / Classes** — Core logic, each with a docstring.
6. **`main()` function** — Entry point with the primary workflow.
7. **`if __name__ == '__main__': main()`** — Guard at the bottom.

## Conventions

- Use `numpy` instead of `np` for the numpy alias.
- Use `os.path.join()` for paths, never f-strings with `/`.
- Use `datetime.now().strftime("%Y-%m-%d")` for date stamps.
- Use terminal utilities from `..src.terminal_utils`: `header`, `section`, `success`, `info`, `list_item`.
- Save outputs to a date-stamped subdirectory under the package's `inputs/` or `outputs/` folder.
- Suppress non-critical warnings at the top:
  ```python
  warnings.filterwarnings('ignore', category=UserWarning, message='.*Box bound precision.*')
  ```
- Force CPU for TensorFlow if imported:
  ```python
  os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
  ```

## Example skeleton

```python
'''
PES_QLv2 - <Short description>

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

- Do NOT reference or modify files inside `PES/` or `PES_Bayesian/`.
- Adapt the package name (`PES_QLv2`, etc.) based on where the user wants to place the script.
- Ask the user for the script's purpose if not specified.
