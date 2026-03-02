# How to Train and Test the DQN Model

> Package: **pes_dqn** — Deep Q-Network variant of the Pandemic Experiment Scenario

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

## 1. Training the DQN Agent

### 1.1 Quick Start

```bash
python3 -m pes_dqn.ext.train_drl
```

This runs the **full training pipeline** with default settings (100 000 episodes).

### 1.2 Custom Episode Count

Pass the number of episodes as the first argument:

```bash
python3 -m pes_dqn.ext.train_drl 200000
```

### 1.3 What Happens During Training

The pipeline proceeds through these stages:

1. **Load data** — reads `initial_severity.csv` and `sequence_lengths.csv`
   from `pes_dqn/inputs/`.
2. **Random-player baseline** — runs 64 evaluation sequences with a random
   agent and saves performance plots.
3. **DQN training** — trains the Deep Q-Network for the configured number of
   episodes using experience replay, a target network, and ε-greedy
   exploration with linear decay.
4. **Save artefacts** — writes the trained Keras model (`.keras`), reward
   history (`rewards.npy`), and a configuration summary to a dated
   subdirectory.
5. **Evaluate** — runs the trained agent on the same 64 sequences and
   generates performance, confidence, and cumulative-performance plots.

### 1.4 Training Output Files

All outputs are saved to `pes_dqn/inputs/<YYYY-MM-DD>_DRL_TRAIN/`:

| File | Description |
|------|-------------|
| `dqn_model_<date>.keras` | Trained Keras model (date-stamped copy) |
| `rewards_<date>.npy` | Average reward every 10 000 episodes |
| `training_config_<date>.txt` | Full hyperparameter record |
| `random_player_*.png` | Baseline performance plots |
| `dqn_agent_rewards_vs_episodes_<date>.png` | Reward convergence curve |
| `dqn_agent_sequence_performance_<date>.png` | Final severity per sequence |
| `dqn_agent_normalised_performance_<date>.png` | Normalised performance |
| `dqn_agent_cumulative_performance_<date>.png` | Cumulative performance trend |
| `dqn_agent_confidences_<date>.png` | Entropy-based confidence scores |

Additionally, the model and rewards are **copied** to the standard paths
consumed by the experiment runner:

- `pes_dqn/inputs/dqn_model.keras`
- `pes_dqn/inputs/rewards.npy`

### 1.5 Default Hyperparameters

These defaults are set in `config/CONFIG.py` and `ext/train_drl.py`:

| Parameter | Default | Source |
|-----------|---------|--------|
| Hidden layers | `[64, 64]` | `CONFIG.DQN_HIDDEN_UNITS` |
| Batch size | 32 | `CONFIG.DQN_BATCH_SIZE` |
| Replay buffer | 50 000 | `CONFIG.DQN_REPLAY_BUFFER_SIZE` |
| Target sync freq | 1 000 steps | `CONFIG.DQN_TARGET_SYNC_FREQ` |
| Learning rate (Adam) | 1 × 10⁻³ | `CONFIG.DQN_LEARNING_RATE` |
| Train freq | every 4 env steps | `CONFIG.DQN_TRAIN_FREQ` |
| Discount γ | 0.865 | `train_drl.py` |
| Initial ε | 0.679 | `train_drl.py` |
| Min ε | 0.085 | `train_drl.py` |
| Episodes | 100 000 | CLI arg or default |
| Random seed | 42 | `CONFIG.SEED` |

### 1.6 Expected Training Time

On a modern CPU (no GPU), approximately **30 minutes to 1 hour** for
100 000 episodes.  Training time scales linearly with episode count.

---

## 2. Bayesian Hyperparameter Optimisation (Optional)

If the default hyperparameters are not satisfactory, you can run an
automated search using Optuna:

```bash
python3 -m pes_dqn.ext.optimize_drl 30
```

The integer argument is the number of trials (default: 30).

### 2.1 Resume a Previous Search

Optimisation state is stored in an SQLite database.  To resume:

```bash
python3 -m pes_dqn.ext.optimize_drl 50 --resume 2026-03-02
```

This loads the study from `inputs/<date>_BAYESIAN_OPT/optuna_study_<date>.db`
and runs additional trials until the total reaches 50.

### 2.2 Optimisation Outputs

Saved to `pes_dqn/inputs/<YYYY-MM-DD>_BAYESIAN_OPT/`:

| File | Description |
|------|-------------|
| `dqn_best_<date>.keras` | Best model found |
| `rewards_best_<date>.npy` | Reward history of the best trial |
| `optimization_results_<date>.txt` | Full report with all trial results |
| `optimization_history_<date>.png` | Convergence plot |
| `hyperparameter_importances_<date>.png` | Parameter importance ranking |
| `optuna_study_<date>.db` | SQLite database (resumable) |

---

## 3. Testing the DQN Agent (Running the Experiment)

### 3.1 Verify Training Files Exist

Before running the experiment, ensure these files are present:

```
pes_dqn/inputs/dqn_model.keras    ← trained Keras model
pes_dqn/inputs/rewards.npy        ← reward history
```

Both are automatically created by the training pipeline (§ 1).

### 3.2 Run the Experiment

```bash
python3 -m pes_dqn
```

This launches the full experiment lifecycle:

1. **Validates** the DQN model (loads it, checks parameter count and output
   shape).
2. **Sets up** the experiment session with logging and dated output folders.
3. **Iterates** through 8 blocks × 8 sequences × 3–10 trials per sequence.
4. At each trial, the DQN agent:
   - Normalises the current state `(resources_left, trial_no, severity)` to
     `[0, 1]³`.
   - Performs a **forward pass** through the trained model to obtain Q-values.
   - Selects the action with the highest Q-value (masking infeasible
     allocations).
   - Computes **meta-cognitive confidence** from the entropy of the Q-value
     distribution.
   - Simulates human-like **response timing** based on confidence.
5. **Saves** results (performance JSON, visualisation PNG, response logs) to
   `pes_dqn/outputs/<YYYY-MM-DD>_DEEP_Q_LEARNING/`.

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
PLAYER_TYPE = 'DEEP_Q_LEARNING'   # Agent type (do not change for DQN)
DQN_MODEL_FILE = 'dqn_model.keras'  # Model filename in inputs/
NUM_BLOCKS = 8
NUM_SEQUENCES = 8
```

---

## 4. Troubleshooting

### Model file not found

```
DQN model file not found!
Expected path: .../pes_dqn/inputs/dqn_model.keras
```

**Solution:** Run training first:

```bash
python3 -m pes_dqn.ext.train_drl
```

### Rewards file not found

Same as above — the training pipeline generates both `dqn_model.keras` and
`rewards.npy`.

### TensorFlow warnings

TensorFlow GPU/CUDA warnings are suppressed by default
(`TF_CPP_MIN_LOG_LEVEL=3`).  If you still see them, they are harmless — the
model runs on CPU.

### Poor agent performance

- Try increasing episodes: `python3 -m pes_dqn.ext.train_drl 200000`
- Run Bayesian optimisation to find better hyperparameters:
  `python3 -m pes_dqn.ext.optimize_drl 50`
- Check that `CONFIG.SEED = 42` for reproducible results.

---

## 5. Quick Reference

| Task | Command |
|------|---------|
| Activate environment | `source linux_mpes_env/bin/activate` |
| Train DQN (default) | `python3 -m pes_dqn.ext.train_drl` |
| Train DQN (custom) | `python3 -m pes_dqn.ext.train_drl 200000` |
| Optimise hyperparams | `python3 -m pes_dqn.ext.optimize_drl 30` |
| Resume optimisation | `python3 -m pes_dqn.ext.optimize_drl 50 --resume YYYY-MM-DD` |
| Run experiment | `python3 -m pes_dqn` |
