# Run Bayesian Optimisation

> Last updated: 2026-03-04

Launch the Bayesian optimisation for a given package **and** start the
watcher that commits+pushes results when the process finishes.

## Inputs

- `$PACKAGE` — the target package.
  Valid values: `pes_bline`, `pes_qlv2`, `pes_dqn`, `pes_ac`, `pes_trf`.
- `$N_TRIALS` — number of optimisation trials (default: **30**).
- `$RESUME_DATE` *(optional)* — `YYYY-MM-DD` date string to resume a previous
  run stored under that date.

## Package → Module Map

Each package has its own optimisation module:

| Package | Module | Alias(es) |
|---------|--------|-----------|
| `pes_bline` | `pes_bline.ext.optimize_rl` | `bayesian`, `bay`, `1` |
| `pes_qlv2` | `pes_qlv2.ext.optimize_rl` | `qlv2`, `ql`, `2` |
| `pes_dqn` | `pes_dqn.ext.optimize_dqn` | `dqn`, `3` |
| `pes_ac` | `pes_ac.ext.optimize_ac` | `ac`, `a2c`, `actor-critic`, `4` |
| `pes_trf` | `pes_trf.ext.optimize_tr` | `transformer`, `tr`, `5` |

If the user provides an alias instead of the full package name, resolve it
using the table above.

## Workflow

All commands run from the **workspace root** (`mPES/`) using **relative
paths** only.  Never use absolute paths.

### Step 0 — Validate environment

```bash
source linux_mpes_env/bin/activate
```

Confirm the virtual environment activated successfully before proceeding.

### Step 1 — Resolve the optimisation module

Using the table above, derive two variables:

- `PKG` — full package name (e.g. `pes_dqn`).
- `OPT_MODULE` — Python module path (e.g. `pes_dqn.ext.optimize_dqn`).

### Step 2 — Prepare the log directory

```bash
LOG_DIR="${PKG}/inputs"
mkdir -p "$LOG_DIR"
```

### Step 3 — Prevent system suspension and shutdown

Disable GNOME lid-close suspension, screen blanking, automatic suspend,
and idle-triggered shutdown so the laptop stays awake and running:

```bash
# Lid close → do nothing (AC and battery)
gsettings set org.gnome.settings-daemon.plugins.power lid-close-ac-action 'nothing'
gsettings set org.gnome.settings-daemon.plugins.power lid-close-battery-action 'nothing'

# Disable automatic suspend on AC and battery
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type 'nothing'

# Disable screen blanking / screen lock
gsettings set org.gnome.desktop.session idle-delay 0
gsettings set org.gnome.desktop.screensaver lock-enabled false
```

> **Note:** these settings persist across sessions.  The prompt does **not**
> restore them automatically — the user should revert them manually after
> the optimisation finishes if desired.

### Step 4 — Launch the optimisation

Build the command arguments:

```
ARGS="$N_TRIALS"                          # fresh run
ARGS="$N_TRIALS --resume $RESUME_DATE"    # resumed run (if date provided)
```

Then launch in background with `nohup`:

```bash
nohup python3 -m "$OPT_MODULE" $ARGS > "$LOG_DIR/bayesian_opt.log" 2>&1 &
OPT_PID=$!
```

Report the PID and log path to the user.

### Step 5 — Launch the suspend and shutdown inhibitor

Keep the system awake and block shutdown while the optimisation is running:

```bash
nohup systemd-inhibit \
    --what=idle:sleep:shutdown:handle-lid-switch \
    --who="mPES Bayesian Optimization ($PKG)" \
    --why="Running $N_TRIALS-trial Bayesian optimization" \
    --mode=block \
    tail --pid=$OPT_PID -f /dev/null > /dev/null 2>&1 &
```

This inhibit lock is automatically released when `$OPT_PID` exits.

### Step 6 — Launch the watcher

Start the watch-and-push script so it monitors the optimisation PID and
auto-commits results when the process finishes:

```bash
nohup utils/watch_and_push.sh "$PKG" $OPT_PID > "$LOG_DIR/watcher.log" 2>&1 &
WATCHER_PID=$!
```

### Step 7 — Report to the user

Print a summary:

```
Optimisation launched
  Package:     $PKG
  Module:      $OPT_MODULE
  Trials:      $N_TRIALS
  PID:         $OPT_PID
  Watcher PID: $WATCHER_PID
  Log:         $LOG_DIR/bayesian_opt.log
  Watcher log: $LOG_DIR/watcher.log
```

And useful monitoring commands:

```
  Progress:    grep 'Trial' $LOG_DIR/bayesian_opt.log | tail -10
  Live:        tail -f $LOG_DIR/bayesian_opt.log
  Alive?:      kill -0 $OPT_PID && echo "running" || echo "done"
```

## Rules

- **Relative paths only** — all paths must be relative to the workspace root.
  Never use absolute paths (no `/home/…`).
- Always activate `linux_mpes_env` before running any command.
- Run all commands from the workspace root directory.
- If `$PACKAGE` is not in the module map, stop and ask the user for
  clarification.
- If `$N_TRIALS` is not specified, default to **30**.
- If `$RESUME_DATE` is not specified, start a **fresh** run.
- Do **not** modify any Python source files — this prompt only *launches*
  existing optimisation scripts.
