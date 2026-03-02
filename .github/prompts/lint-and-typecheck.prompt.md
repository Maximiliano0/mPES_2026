# Lint & Type-Check (Fix Loop)

> Last updated: 2026-02-26

Run the mandatory quality gates on a package and **iteratively fix every
issue in source code** until both tools report zero problems.

## Inputs

- `$PACKAGE_DIR` — the package directory to check (e.g. `pes`, `pes_base_line`,
  `pes_qlv2`, `pes_transformer`, `utils`).
  If omitted, infer the package from the file that was just edited.

## Workflow

Repeat the loop below until **both** targets are met in the same iteration.

```
while issues remain:
    1. Run pyright  → read output → fix every issue in source
    2. Run pylint   → read output → fix every issue in source
```

### Step 1 — Pyright (static type checking)

```bash
source linux_mpes_env/bin/activate && pyright $PACKAGE_DIR/
```

- Read the full output. For **each** error, warning, or information:
  1. Open the reported file and line.
  2. Fix the root cause in source (add/correct type hints, fix imports, etc.).
- Do **not** use `# type: ignore` — fix the code instead.
- Re-run pyright after fixes. Repeat until output is
  `0 errors, 0 warnings, 0 informations`.

### Step 2 — Pylint (linting with project standard)

```bash
source linux_mpes_env/bin/activate && pylint --rcfile=.pylintrc $PACKAGE_DIR/
```

- Read the full output. For **each** reported message:
  1. Open the reported file and line.
  2. Apply the appropriate fix (see common fixes below).
- Re-run pylint after fixes. Repeat until the score is `10.00/10`.

#### Common fixes

| Pylint message | Fix |
|----------------|-----|
| Unused import (`W0611`) | Remove the import line. |
| Unused variable (`W0612`) | Remove the variable, or prefix with `_`. |
| Unused argument (`W0613`) | Prefix with `_` (e.g. `_state`). |
| Bare `except:` (`W0702`) | Specify the exception type (e.g. `except OSError:`). |
| Missing docstring (`C0114/C0115/C0116`) | Add a NumPy-style docstring. |
| Trailing whitespace (`C0303`) | Remove trailing spaces. |
| Missing final newline (`C0304`) | Add a newline at EOF. |
| Unnecessary pass (`W0107`) | Remove the `pass` after a docstring or code. |
| `raise` missing `from` (`W0707`) | Use `raise ... from e`. |
| Bad unary operand (`E1130`) | Fix the operand type. |
| Wildcard import unused (`W0611` on `*`) | Replace `from .. import *` with explicit imports. |

### Step 3 — Cross-check

If fixes from step 2 introduced new pyright issues (or vice-versa), go back
to step 1 and repeat the full loop.

## Targets

| Tool    | Target                                |
|---------|---------------------------------------|
| pyright | `0 errors, 0 warnings, 0 informations` |
| pylint  | `10.00/10`                            |

Both targets must be met **simultaneously** before the task is considered done.

## Rules

- Always activate the virtual environment first (`source linux_mpes_env/bin/activate`).
- Run checks from the **workspace root** (`mPES/`), never from inside a package.
- Fix issues **in source code** — do not suppress, silence, or work around them.
- Do **not** add `# pylint: disable=` for rules that are **enforced** in `.pylintrc`.
- Do **not** add `# type: ignore` comments.
- Do **not** modify `.pylintrc` or `pyproject.toml` to work around failures.
- When replacing `from .. import *` with explicit imports, check which names
  the module actually uses (grep/search the file) and import only those.
- Preserve existing functionality — fixes must be behaviour-neutral.
