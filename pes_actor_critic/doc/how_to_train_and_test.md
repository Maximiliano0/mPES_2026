# How to Train and Test the A2C Model

> Package: **pes_actor_critic** — Advantage Actor-Critic variant of the Pandemic Experiment Scenario

---

## Prerequisites

| Requirement | Detail |
|-------------|--------|
| Python | 3.12 |
| Virtual environment | `linux_mpes_env` (activate before every command) |
| TensorFlow | 2.16.2 (CPU — no GPU required) |
| Input data | `inputs/initial_severity.csv` and `inputs/sequence_lengths.csv` |

Activate the environment once per terminal session:

```bash
source linux_mpes_env/bin/activate
```

---

## 1. Training the A2C Agent

### 1.1 Quick Start

```bash
python3 -m pes_actor_critic.ext.train_ac
```

This runs the **full training pipeline** with default settings (100 000 episodes).

### 1.2 Custom Episode Count

Pass the number of episodes as the first argument:

```bash
python3 -m pes_actor_critic.ext.train_ac 200000
```

### 1.3 What Happens During Training

The pipeline proceeds through these stages:

1. **Load data** — reads `initial_severity.csv` and `sequence_lengths.csv`
   from `pes_actor_critic/inputs/`.
2. **Random-player baseline** — runs 64 evaluation sequences with a random
   agent and saves performance plots.
3. **A2C training** — trains the Actor and Critic networks for the configured
   number of episodes using on-policy batched updates with advantage
   estimation, entropy bonus, and ε-greedy overlay with linear decay.
4. **Save artefacts** — writes the trained Actor Keras model (`.keras`),
   reward history (`rewards.npy`), and a configuration summary to a dated
   subdirectory.
5. **Evaluate** — runs the trained agent on the same 64 sequences and
   generates performance, confidence, and cumulative-performance plots.

### 1.4 Training Output Files

All outputs are saved to `pes_actor_critic/inputs/<YYYY-MM-DD>_AC_TRAIN/`:

| File | Description |
|------|-------------|
| `ac_actor_<date>.keras` | Trained Actor Keras model (date-stamped copy) |
| `rewards_<date>.npy` | Average reward every 10 000 episodes |
| `training_config_<date>.txt` | Full hyperparameter record |
| `random_player_*.png` | Baseline performance plots |
| `ac_agent_rewards_vs_episodes_<date>.png` | Reward convergence curve |
| `ac_agent_sequence_performance_<date>.png` | Final severity per sequence |
| `ac_agent_normalised_performance_<date>.png` | Normalised performance |
| `ac_agent_cumulative_performance_<date>.png` | Cumulative performance trend |
| `ac_agent_confidences_<date>.png` | Entropy-based confidence scores |

Additionally, the Actor model and rewards are **copied** to the standard
paths consumed by the experiment runner:

- `pes_actor_critic/inputs/ac_actor.keras`
- `pes_actor_critic/inputs/rewards.npy`

### 1.5 Default Hyperparameters

These defaults are set in `config/CONFIG.py` and `ext/train_ac.py`:

| Parameter | Default | Source |
|-----------|---------|--------|
| Actor hidden layers | `[64, 64]` | `CONFIG.AC_ACTOR_HIDDEN_UNITS` |
| Critic hidden layers | `[64, 64]` | `CONFIG.AC_CRITIC_HIDDEN_UNITS` |
| Actor learning rate | 3 × 10⁻⁴ | `CONFIG.AC_ACTOR_LR` |
| Critic learning rate | 1 × 10⁻³ | `CONFIG.AC_CRITIC_LR` |
| Entropy coefficient | 0.01 | `CONFIG.AC_ENTROPY_COEFF` |
| Discount γ | 0.99 | `CONFIG.AC_DISCOUNT` |
| Initial ε | 0.679 | `train_ac.py` |
| Min ε | 0.085 | `train_ac.py` |
| Episodes | 100 000 | CLI arg or default |
| Random seed | 42 | `CONFIG.SEED` |

### 1.6 Expected Training Time

On a modest CPU (e.g. Intel i3-6006U @ 2 GHz, 4 threads), approximately
**20–40 minutes** for 100 000 episodes with CPU optimisations enabled
(`@tf.function` compilation, no confidence computation during training,
TensorFlow thread-pool tuning).  Training time scales linearly with
episode count.

A2C may be slightly slower than DQN per episode due to maintaining two
networks and performing a full gradient update per episode (vs. DQN's
periodic mini-batch updates).

### 1.7 CPU Training Optimisations

The following optimisations are active by default and require no
configuration:

| Optimisation | Effect | Location |
|-------------|--------|----------|
| `@tf.function` compiled training | Eliminates eager overhead for per-episode updates | `ext/ac_model.py` |
| Skip confidence computation | `compute_confidence=False` skips entropy/masking per step | `ext/pandemic.py` |
| TF thread-pool tuning | `intra_op=0` (auto), `inter_op=2` for multi-core CPUs | `ext/ac_model.py` |
| OMP_NUM_THREADS | Set to CPU core count before TF import | `__init__.py` |
| Actor-only inference | Only Actor loaded at experiment time (Critic discarded) | `src/pygameMediator.py` |

To re-enable confidence tracking during training (at slightly slower speed):

```python
rewards, actor, critic, confs = A2CTraining(
    env, actor_lr, critic_lr, discount, entropy_coeff,
    eps, min_eps, episodes,
    ...,
    compute_confidence=True,   # ← enables meta-cognitive observation
)
```

---

## 2. Bayesian Hyperparameter Optimisation (Optional)

If the default hyperparameters are not satisfactory, you can run an
automated search using Optuna:

```bash
python3 -m pes_actor_critic.ext.optimize_ac 30
```

The integer argument is the number of trials (default: 30).

### 2.1 Resume a Previous Search

Optimisation state is stored in an SQLite database.  To resume:

```bash
python3 -m pes_actor_critic.ext.optimize_ac 50 --resume 2026-03-02
```

This loads the study from `inputs/<date>_BAYESIAN_OPT/optuna_study_<date>.db`
and runs additional trials until the total reaches 50.

### 2.2 Optimisation Outputs

Saved to `pes_actor_critic/inputs/<YYYY-MM-DD>_BAYESIAN_OPT/`:

| File | Description |
|------|-------------|
| `ac_best_<date>.keras` | Best Actor model found |
| `rewards_best_<date>.npy` | Reward history of the best trial |
| `optimization_results_<date>.txt` | Full report with all trial results |
| `optimization_history_<date>.png` | Convergence plot |
| `hyperparameter_importances_<date>.png` | Parameter importance ranking |
| `optuna_study_<date>.db` | SQLite database (resumable) |

### 2.3 A2C-Specific Search Space

The A2C optimisation searches over 10 hyperparameters, including two
separate learning rates (Actor and Critic) and an entropy coefficient —
parameters that don't exist in the DQN search space.  Conversely, DQN
parameters like `batch_size`, `replay_buffer_size`, and
`target_sync_freq` are absent since A2C is on-policy.

---

## 3. Testing the A2C Agent (Running the Experiment)

### 3.1 Verify Training Files Exist

Before running the experiment, ensure these files are present:

```
pes_actor_critic/inputs/ac_actor.keras    ← trained Actor Keras model
pes_actor_critic/inputs/rewards.npy       ← reward history
```

Both are automatically created by the training pipeline (§ 1).

### 3.2 Run the Experiment

```bash
python3 -m pes_actor_critic
```

This launches the full experiment lifecycle:

1. **Validates** the Actor model (loads it, checks parameter count and output
   shape).
2. **Sets up** the experiment session with logging and dated output folders.
3. **Iterates** through 8 blocks × 8 sequences × 3–10 trials per sequence.
4. At each trial, the A2C agent:
   - Normalises the current state `(resources_left, trial_no, severity)` to
     `[0, 1]³`.
   - Performs a **forward pass** through the Actor to obtain
     $\pi_\theta(a \mid s)$.
   - Selects the action with the highest probability (masking infeasible
     allocations where `action > resources_left`).
   - Computes **meta-cognitive confidence** from the entropy of the policy
     distribution — a theoretically grounded measure.
   - Simulates human-like **response timing** based on confidence.
5. **Saves** results (performance JSON, visualisation PNG, response logs) to
   `pes_actor_critic/outputs/<YYYY-MM-DD>_AC_AGENT/`.

### 3.3 Experiment Outputs

| File | Description |
|------|-------------|
| `PES__<session_id>.txt` | Experiment configuration snapshot |
| `PES_responses_<session_id>.txt` | Trial-by-trial responses |
| `PES_log_<session_id>.txt` | Console log (dual-stream) |
| `PES_movement_log_<session_id>.npy` | Movement data |
| `PES_results_<session_id>.json` | Performance summary JSON |
| `PES_results_<session_id>.png` | Performance plots |

### 3.4 Configuration

All experiment parameters are in `config/CONFIG.py`.  Key settings:

```python
PLAYER_TYPE = {  # Decision maker type - SELECT ONE
    1: 'RL_AGENT',     # Tabular Q-Learning agent (fallback / baseline comparison)
    2: 'DQN_AGENT',    # Deep Q-Network (experience replay + target net)
    3: 'AC_AGENT'      # Advantage Actor-Critic (A2C) agent
}[3]                   # <-- change index to select
AC_MODEL_ACTOR_FILE = 'ac_actor.keras'  # Actor model filename in inputs/
NUM_BLOCKS = 8
NUM_SEQUENCES = 8
```

---

## 4. Troubleshooting

### Model file not found

```
A2C Actor model file not found!
Expected path: .../pes_actor_critic/inputs/ac_actor.keras
```

**Solution:** Run training first:

```bash
python3 -m pes_actor_critic.ext.train_ac
```

### Rewards file not found

Same as above — the training pipeline generates both `ac_actor.keras` and
`rewards.npy`.

### TensorFlow warnings

TensorFlow GPU/CUDA warnings are suppressed by default
(`TF_CPP_MIN_LOG_LEVEL=3`).  If you still see them, they are harmless — the
model runs on CPU.

### Poor agent performance

- Try increasing episodes: `python3 -m pes_actor_critic.ext.train_ac 200000`
- Run Bayesian optimisation to find better hyperparameters:
  `python3 -m pes_actor_critic.ext.optimize_ac 50`
- Adjust `AC_ENTROPY_COEFF` in `CONFIG.py` — higher values encourage more
  exploration, lower values favour exploitation.
- Check that `CONFIG.SEED = 42` for reproducible results.

### Slow training on CPU

The default configuration is already optimised for CPU.  If training is
still too slow:

- Reduce episode count: `python3 -m pes_actor_critic.ext.train_ac 50000`
- Verify `compute_confidence=False` is set (default) in `train_ac.py`
- Check that `OMP_NUM_THREADS` is set to your CPU core count
  (auto-configured in `__init__.py`)

### Actor vs. Critic model confusion

Only the **Actor** model (`ac_actor.keras`) is needed for the experiment.
The Critic is used only during training and is not saved to the standard
inputs path.

---

## 5. Quick Reference

| Task | Command |
|------|---------|
| Activate environment | `source linux_mpes_env/bin/activate` |
| Train A2C (default) | `python3 -m pes_actor_critic.ext.train_ac` |
| Train A2C (custom) | `python3 -m pes_actor_critic.ext.train_ac 200000` |
| Optimise hyperparams | `python3 -m pes_actor_critic.ext.optimize_ac 30` |
| Resume optimisation | `python3 -m pes_actor_critic.ext.optimize_ac 50 --resume YYYY-MM-DD` |
| Run experiment | `python3 -m pes_actor_critic` |
