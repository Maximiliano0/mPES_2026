# Update Package Documentation

> Last updated: 2026-03-04

Read a target package's source code, then update and export its Markdown
documentation so that it faithfully explains the current implementation.

## Usage

When invoking this prompt, specify the **target package** name. Examples:

```
@update-pkg-docs pes
@update-pkg-docs pes_base_line
@update-pkg-docs pes_qlv2
@update-pkg-docs pes_dqn
@update-pkg-docs pes_actor_critic
@update-pkg-docs pes_transformer
```

Throughout this prompt, `<PKG>` refers to the package name provided by the
user (e.g. `pes`, `pes_base_line`, `pes_qlv2`, `pes_dqn`, `pes_actor_critic`,
`pes_transformer`).

## Scope

Discover the package layout dynamically:

| Item | Path |
|------|------|
| Package root | `<PKG>/` |
| Config | `<PKG>/config/CONFIG.py` |
| Init / exports | `<PKG>/__init__.py` |
| Experiment entry | `<PKG>/__main__.py` |
| Core algorithms | `<PKG>/ext/*.py` |
| Support modules | `<PKG>/src/*.py` |
| Documentation | `<PKG>/doc/*.md` → `<PKG>/doc/*.html` |

## Step 1 — Read the project

1. List the contents of `<PKG>/`, `<PKG>/ext/`, `<PKG>/src/`, and `<PKG>/doc/`
   to discover all files.
2. Read **every** `.py` file inside the package.
3. Build a mental model of:
   - The Gym environment (`Pandemic` class in `ext/pandemic.py`): state space,
     action space, reward, transition dynamics.
   - Training / optimisation scripts in `ext/` (e.g. `train_rl.py`,
     `optimize_rl.py`): hyperparameters, algorithm variant, output artefacts.
   - The experiment runner (`__main__.py`): block/sequence/trial hierarchy,
     agent response flow, performance metrics.
   - Support modules in `src/`: severity calculations, logging, Pygame bridge,
     result plots, terminal output.
   - All tuneable constants from `config/CONFIG.py`.
   - Any package-specific features (Double Q-Learning, Bayesian optimisation,
     Transformer model, PBRS, etc.) — these vary by package.

## Step 2 — Read the existing `.md` files

1. List `<PKG>/doc/*.md` to discover all Markdown documentation files.
2. Read each one in full.
3. Note the purpose and structure of each file (some packages may have
   different `.md` files than others).

## Step 3 — Update the `.md` files

For **each** `.md` file, compare its content against the actual code read in
Step 1 and fix every discrepancy. The goal is that a reader can trust the
documentation as a faithful mirror of the code.

### What to check and update

- **Hyperparameter values**: learning rate, discount factor, epsilon, episodes,
  decay parameters. Must match what `train_rl.py` actually uses.
- **Code snippets**: every code block must correspond to the real source. Update
  variable names, function signatures, and logic if the code has changed.
  Use the variable name exactly as it appears in the source (e.g.
  `resolved_decay_rate` vs `decay_rate`).
- **Formulas**: verify that mathematical formulas (LaTeX) match the code
  implementation. Check reward formulas, severity update formulas, Q-update
  rules, and epsilon decay formulas.
- **State/action spaces**: check dimensions, ranges, and descriptions against
  `Pandemic.__init__()` and `CONFIG.py`.
- **Module descriptions**: verify that each module/function mentioned still
  exists, has the described purpose, and uses the described signature.
- **Constant values**: `MAX_SEVERITY`, `SEED`, `AVAILABLE_RESOURCES_PER_SEQUENCE`,
  `NUM_BLOCKS`, `NUM_SEQUENCES`, multipliers, etc.
- **Pipeline flow**: verify the described workflow (train → experiment → results)
  matches the actual scripts and their outputs.
- **New content**: if the code contains significant functionality not covered
  by any existing `.md`, add a new section (or a new `.md` if the topic is
  large enough to warrant one). Follow the same structure and tone.

### Style rules for `.md` content

- Write in **Spanish** (matching existing docs).
- Use **KaTeX-compatible LaTeX**: `$...$` for inline, `$$...$$` for display.
- Use fenced code blocks with `python` language tag.
- Keep tables aligned with `|---|` separators.
- Section numbering: `## 1.`, `### 1.1`, `#### subtitle`.
- ASCII diagrams for architecture/flow (match existing style).
- Include academic references where relevant (author, year, title, venue).

## Step 4 — Update docstrings

Using the mental model built in Step 1, review **every** public function and
class across all `.py` files in the package and ensure their docstrings are
present, accurate, and follow the **NumPy-style** convention.

### What to check and update

- **Missing docstrings**: every public function, method, and class must have a
  docstring. Add one if missing.
- **Stale descriptions**: if the implementation has changed (parameters added or
  removed, return type changed, logic altered), update the docstring to match.
- **Parameter lists**: verify that `Parameters`, `Returns`, `Raises`, and
  `Attributes` sections list exactly the current parameters/returns, with
  correct types and descriptions.
- **Examples section**: if present, confirm any example code still runs
  correctly against the current implementation.
- **Internal helpers**: private functions (prefixed with `_`) do not require
  docstrings but may have them — if present, ensure they are correct.

### NumPy-style template

```python
def function_name(param_a: int, param_b: str = "default") -> bool:
    """One-line summary (imperative mood, no period).

    Extended description explaining behaviour, edge cases, or
    algorithmic notes when useful.

    Parameters
    ----------
    param_a : int
        Description of param_a.
    param_b : str, optional
        Description of param_b (default "default").

    Returns
    -------
    bool
        Description of the return value.

    Raises
    ------
    ValueError
        When ``param_a`` is negative.
    """
```

### Style rules for docstrings

- Write docstrings in **English**.
- Use triple double-quotes (`"""`).
- First line: imperative summary ≤ 79 characters.
- Blank line between summary and body.
- Wrap body text at **120 characters** (project standard).
- Reference `CONFIG.py` constants by name when describing default values
  sourced from configuration.

## Step 5 — Export each `.md` to `.html`

Convert every `.md` inside `<PKG>/doc/` to a matching `.html` file in the same
directory. Use the Python `markdown` library with this pipeline:

```python
import markdown

body = markdown.markdown(
    md_text,
    extensions=["tables", "fenced_code", "codehilite", "toc"],
    extension_configs={"codehilite": {"guess_lang": False, "css_class": "highlight"}},
)
```

Wrap the body in the standard project HTML template:

```html
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DOCUMENT_TITLE</title>
  <link rel="stylesheet"
        href="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.css">
  <script defer
        src="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.js"></script>
  <script defer
        src="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/contrib/auto-render.min.js"
        onload="renderMathInElement(document.body, {delimiters:[
          {left:'$$',right:'$$',display:true},
          {left:'$',right:'$',display:false}
        ]})"></script>
  <style>
    :root { --bg:#ffffff; --fg:#1a1a2e; --accent:#0f3460; --code-bg:#f4f4f8;
             --border:#ddd; --link:#1565c0; --table-stripe:#f9f9fc; }
    @media (prefers-color-scheme:dark) {
      :root { --bg:#1a1a2e; --fg:#e0e0e0; --accent:#64b5f6; --code-bg:#16213e;
               --border:#333; --link:#90caf9; --table-stripe:#1e2a45; }
    }
    *,*::before,*::after { box-sizing:border-box; }
    html { scroll-behavior:smooth; }
    body { font-family:"Segoe UI",system-ui,-apple-system,sans-serif;
           line-height:1.7; color:var(--fg); background:var(--bg);
           max-width:52em; margin:2em auto; padding:0 1.5em; }
    h1 { border-bottom:3px solid var(--accent); padding-bottom:.3em; }
    h2 { border-bottom:1px solid var(--border); padding-bottom:.2em; margin-top:2em; }
    h3,h4 { margin-top:1.6em; }
    a { color:var(--link); text-decoration:none; }
    a:hover { text-decoration:underline; }
    code { font-family:"JetBrains Mono","Fira Code",Consolas,monospace;
           font-size:.92em; background:var(--code-bg); padding:.15em .35em;
           border-radius:4px; }
    pre { background:var(--code-bg); padding:1em 1.2em; border-radius:8px;
          overflow-x:auto; border:1px solid var(--border); }
    pre code { background:none; padding:0; font-size:.88em; }
    table { border-collapse:collapse; width:100%; margin:1em 0; }
    th,td { border:1px solid var(--border); padding:.55em .9em; text-align:left; }
    th { background:var(--accent); color:#fff; }
    tr:nth-child(even) { background:var(--table-stripe); }
    blockquote { border-left:4px solid var(--accent); margin-left:0;
                 padding:.4em 1em; color:#666; background:var(--code-bg);
                 border-radius:0 6px 6px 0; }
    .katex-display { overflow-x:auto; overflow-y:hidden; padding:.5em 0; }
  </style>
</head>
<body>
  BODY_HTML
</body>
</html>
```

Key requirements:
- One `.html` per `.md` (same base name, same directory).
- KaTeX auto-render for `$...$` and `$$...$$` math.
- Dark-mode support via `prefers-color-scheme`.
- `<title>` must match the `# H1` of the Markdown.

## Checklist

Before finishing, verify:

- [ ] Every `.md` in `<PKG>/doc/` has been read and compared against the source code.
- [ ] Hyperparameters, formulas, code snippets, and descriptions match the code.
- [ ] No dead references to functions/classes/variables that no longer exist.
- [ ] Every `.md` has a corresponding up-to-date `.html` in `<PKG>/doc/`.
- [ ] HTML files render math correctly (KaTeX delimiters, no broken `$`).
- [ ] No English leaking into Spanish-language documentation (except code/names).
- [ ] Every public function and class in `<PKG>/` has a NumPy-style docstring.
- [ ] Docstring parameter lists, return types, and descriptions match the code.
- [ ] No stale references in docstrings to removed parameters or changed behaviour.
