# How to Train and Test the Double Q-Learning Agent

> Package: **pes_dql** — Double Q-Learning with exponential ε-decay warm-up
> and Potential-Based Reward Shaping (PBRS)

---

## Prerequisites

| Requirement | Detail |
|-------------|--------|
| Python | 3.10 (Windows) / 3.12 (Linux) |
| Virtual environment | `win_mpes_env` (Windows) or `linux_mpes_env` (Linux) |
| Optuna | 4.7.0 (included in `requirements.txt`) |
| Input data | `inputs/initial_severity.csv` and `inputs/sequence_lengths.csv` |

Activate the environment once per terminal session:

**Linux / macOS:**

```bash
source linux_mpes_env/bin/activate
```

**Windows (PowerShell):**

```powershell
win_mpes_env\Scripts\Activate.ps1
```

---

## 1. Training the Double Q-Learning Agent

### 1.1 Quick Start

```bash
python3 -m pes_dql.ext.train_rl
```

This runs the **full training pipeline** with Bayesian-optimised
hyperparameters (860 000 episodes by default).

### 1.2 Custom Episode Count

Pass the number of episodes as the first argument:

```bash
python3 -m pes_dql.ext.train_rl 1000000
```

### 1.3 What Happens During Training

The pipeline proceeds through these stages:

1. **Load data** — reads `initial_severity.csv` and
   `sequence_lengths.csv` from `inputs/`.
2. **Random player baseline** — runs 64 sequences with uniformly random
   allocations and generates two baseline plots.
3. **Double Q-Learning training** — trains with three algorithmic
   improvements over standard Q-Learning (§ 1.7). Prints average reward
   every 10 000 episodes.
4. **Evaluation** — runs the trained greedy policy on 64 fixed sequences,
   collects meta-cognitive confidence scores.
5. **Save artefacts** — writes Q-table, reward history, config report,
   and visualisations to a dated subdirectory.

### 1.4 Training Output Files

All outputs are saved to `pes_dql/inputs/<YYYY-MM-DD>_RL_TRAIN/`:

| File | Description |
|------|-------------|
| `q_<date>.npy` | Trained Q-table — shape `(31, 11, 10, 11)` |
| `rewards_<date>.npy` | Average-reward history (86 values, sampled every 10 000 eps) |
| `training_config_<date>.txt` | Summary of hyperparameters and settings |
| `confsrl_<date>.npy` | Confidence scores from evaluation |
| `random_player_sequence_performance_<date>.png` | Baseline severity per sequence |
| `random_player_normalised_performance_<date>.png` | Baseline normalised performance |
| `rl_agent_rewards_vs_episodes_<date>.png` | Learning curve |
| `rl_agent_sequence_performance_<date>.png` | Severity per sequence |
| `rl_agent_normalised_performance_<date>.png` | Performance (0–1) |
| `rl_agent_cumulative_performance_<date>.png` | Cumulative trend |
| `rl_agent_confidences_<date>.png` | Raw confidence scatter |
| `rl_agent_remapped_confidences_<date>.png` | Normalised confidence (0–1) |

Additionally, the Q-table and rewards are **copied** to the standard
paths consumed by the experiment runner:

- `pes_dql/inputs/q.npy`
- `pes_dql/inputs/rewards.npy`

### 1.5 Default Hyperparameters (Bayesian-Optimised)

These values were found by Bayesian optimisation (§ 2) and are hard-coded
in `ext/train_rl.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning rate α | 0.2593 | Q-table update step size |
| Discount factor γ | 0.9806 | Future-reward weighting |
| Initial ε | 0.8392 | Starting exploration probability |
| Minimum ε | 0.0799 | Final exploration probability |
| Episodes | 860 000 | Total training episodes |
| Warm-up ratio | 0.0240 | Fraction of episodes at constant ε |
| Target ratio | 0.5174 | Fraction where ε reaches ε_min |
| PBRS coefficient β | 0.2177 | Reward-shaping strength |
| Random seed | 42 | `CONFIG.SEED` |

### 1.6 Q-Table Dimensions

| Axis | Size | Meaning |
|------|------|---------|
| 0 | 31 | Resources left (0–30) |
| 1 | 11 | Trial number (0–10) |
| 2 | 10 | Severity (0–9), `MAX_SEVERITY = 9` |
| 3 | 11 | Actions (0–10 resources) |

**Total entries:** $31 \times 11 \times 10 \times 11 = 37{,}510$

During training, **two Q-tables** are maintained (`Q_A`, `Q_B`). The
saved Q-table is their average: $Q = \frac{Q_A + Q_B}{2}$.

### 1.7 Three Algorithmic Improvements

pes_dql extends the baseline with three improvements that preserve the
optimal policy while accelerating convergence:

#### A. Double Q-Learning (van Hasselt, 2010)

Eliminates maximisation bias by maintaining two independent Q-tables.
Action selection uses one table; value evaluation uses the other:

$$Q_A(s,a) \leftarrow Q_A(s,a) + \alpha \bigl[ r + \gamma \, Q_B\bigl(s', \arg\max_{a'} Q_A(s',a')\bigr) - Q_A(s,a) \bigr]$$

With probability 0.5, the roles of $Q_A$ and $Q_B$ are swapped.

#### B. Exponential ε-Decay with Warm-up

Replaces linear decay with three distinct phases:

| Phase | Episodes | ε behaviour |
|-------|----------|-------------|
| Warm-up | 0 → $W$ | Constant at $\varepsilon_0$ |
| Decay | $W$ → $T$ | Exponential: $\varepsilon_0 \cdot \lambda^{t-W}$ |
| Exploitation | $T$ → $N$ | ≈ $\varepsilon_{\min}$ |

Where $W = \text{warmup\_ratio} \times N$, $T = \text{target\_ratio} \times N$,
and $\lambda = (\varepsilon_{\min} / \varepsilon_0)^{1/((T - W)/N)}$ is
auto-computed.

#### C. Potential-Based Reward Shaping (Ng et al., 1999)

Augments the environment reward with a shaping term:

$$r' = r + \beta \bigl( \gamma \, \Phi(s') - \Phi(s) \bigr)$$

where $\Phi(s) = -\sum_i \max(0, s_i)$ (negative sum of current
severities). By the invariance theorem, the optimal policy is unchanged
for any $\beta \geq 0$.

### 1.8 Key Differences from Other Packages

| Aspect | pes_base | pes_ql | pes_dql |
|--------|----------|--------|---------|
| Algorithm | Q-Learning | Q-Learning | **Double** Q-Learning |
| ε-decay | Linear | Linear | **Exponential + warm-up** |
| Reward shaping | None | None | **PBRS** |
| Hyperparameters | Hand-tuned | Bayesian (5 params) | Bayesian (**8 params**) |
| Default episodes | 1 000 000 | 900 000 | 860 000 |

---

## 2. Bayesian Hyperparameter Optimisation

### 2.1 Quick Start

```bash
python3 -m pes_dql.ext.optimize_rl 50
```

This launches 50 Optuna trials. Each trial trains a Double Q-Learning
agent with sampled hyperparameters and evaluates it on 64 fixed sequences.

### 2.2 Custom Trial Count

```bash
python3 -m pes_dql.ext.optimize_rl 100
```

### 2.3 Resume a Previous Search

```bash
python3 -m pes_dql.ext.optimize_rl 100 --resume 2026-03-02
```

Optuna loads the existing SQLite database and continues from where it
stopped. If the requested total is already reached, no new trials run.

### 2.4 Search Space (8 Parameters)

| Parameter | Range | Scale |
|-----------|-------|-------|
| `learning_rate` | [0.20, 0.40] | Log |
| `discount_factor` | [0.80, 0.99] | Linear |
| `epsilon_initial` | [0.40, 1.00] | Linear |
| `epsilon_min` | [0.05, 0.10] | Linear |
| `num_episodes` | [800 000, 1 200 000] (step 10 000) | Linear |
| `warmup_ratio` | [0.01, 0.10] | Linear |
| `target_ratio` | [0.50, 0.80] | Linear |
| `penalty_coeff` | [0.001, 0.50] | Log |

**Objective:** Maximise mean normalised performance (0–1) over 64
evaluation sequences.

### 2.5 Optimisation Outputs

Saved to `pes_dql/inputs/<YYYY-MM-DD>_BAYESIAN_OPT/`:

| File | Description |
|------|-------------|
| `q_best_<date>.npy` | Q-table from the best trial |
| `rewards_best_<date>.npy` | Reward history from the best trial |
| `optimization_results_<date>.txt` | Full report with all trial results |
| `optimization_history_<date>.png` | Convergence plot |
| `hyperparameter_importances_<date>.png` | Parameter importance ranking (fANOVA) |
| `optuna_study_<date>.db` | SQLite database (resumable) |

### 2.6 Running in the Background

**Linux:**

```bash
nohup python3 -m pes_dql.ext.optimize_rl 100 \
  > pes_dql/inputs/bayesian_opt.log 2>&1 &
```

Or use the helper scripts:

```bash
./utils/run_bayesian_opt.sh qlv2 100
```

**Windows (PowerShell):**

```powershell
.\utils\run_bayesian_opt.ps1 qlv2 100
```

---

## 3. Testing the Agent (Running the Experiment)

### 3.1 Verify Training Files Exist

Before running the experiment, ensure these files are present:

```
pes_dql/inputs/q.npy           ← trained Q-table (31 × 11 × 10 × 11)
pes_dql/inputs/rewards.npy     ← reward history
```

Both are automatically created by the training pipeline (§ 1).

### 3.2 Run the Experiment

```bash
python3 -m pes_dql
```

This launches the full experiment lifecycle:

1. **Validates** the Q-table (loads `q.npy`, checks shape).
2. **Sets up** the experiment session with logging and dated output folders.
3. **Iterates** through 8 blocks × 8 sequences × 3–10 trials per
   sequence (~360 total decisions).
4. At each trial, the RL agent:
   - Indexes the Q-table: `Q[resources_left, trial_no, severity]`.
   - Selects the action with the highest Q-value (greedy argmax).
   - Masks infeasible actions (allocation > resources_left).
   - Computes **meta-cognitive confidence** from the entropy of the
     Q-value distribution.
   - Simulates human-like **response timing** based on confidence.
5. **Saves** results (performance JSON, visualisation PNG, response logs)
   to `pes_dql/outputs/<YYYY-MM-DD>_RL_AGENT/`.

### 3.3 Experiment Outputs

| File | Description |
|------|-------------|
| `PES__<session_id>.txt` | Experiment configuration snapshot |
| `PES_responses_<session_id>.txt` | Trial-by-trial responses (severity, allocation, confidence, timing) |
| `PES_log_<session_id>.txt` | Console log (dual-stream) |
| `PES_movement_log_<session_id>.npy` | Movement data |
| `PES_results_<session_id>.json` | Performance summary JSON |
| `PES_results_<session_id>.png` | Performance plots |

### 3.4 Configuration

All experiment parameters are in `config/CONFIG.py`. Key settings:

```python
PLAYER_TYPE = 'RL_AGENT'
SEED = 42
NUM_BLOCKS = 8
NUM_SEQUENCES = 8
AVAILABLE_RESOURCES_PER_SEQUENCE = 39
MAX_SEVERITY = 9
```

---

## 4. Complete Workflow (3-Stage Pipeline)

The recommended workflow from scratch:

```bash
# Stage 1 — Find optimal hyperparameters
python3 -m pes_dql.ext.optimize_rl 50

# Stage 2 — Train with those hyperparameters (already hard-coded)
python3 -m pes_dql.ext.train_rl

# Stage 3 — Run the experiment
python3 -m pes_dql
```

If you have run a new Bayesian search, update the hyperparameters in
`ext/train_rl.py` with the best values from the optimisation report
before Stage 2.

---

## 5. Troubleshooting

### Q-table not found

```
Q‑table file not found: .../pes_dql/inputs/q.npy
```

**Solution:** Run training first:

```bash
python3 -m pes_dql.ext.train_rl
```

### Poor agent performance

- Run Bayesian optimisation with more trials:
  `python3 -m pes_dql.ext.optimize_rl 100`
- Increase episode count:
  `python3 -m pes_dql.ext.train_rl 1200000`
- Verify `SEED = 42` for reproducible results.

### Optimisation appears stuck or slow

Each trial trains a full Double Q-Learning agent — expect longer per trial
than pes_ql due to the three improvements (especially PBRS).

---

## 6. Quick Reference

| Task | Command |
|------|---------|
| Activate environment (Linux) | `source linux_mpes_env/bin/activate` |
| Activate environment (Windows) | `win_mpes_env\Scripts\Activate.ps1` |
| Train agent (default) | `python3 -m pes_dql.ext.train_rl` |
| Train agent (custom) | `python3 -m pes_dql.ext.train_rl 1000000` |
| Optimise hyperparameters | `python3 -m pes_dql.ext.optimize_rl 50` |
| Resume optimisation | `python3 -m pes_dql.ext.optimize_rl 100 --resume YYYY-MM-DD` |
| Run experiment | `python3 -m pes_dql` |
