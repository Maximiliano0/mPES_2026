# How to Train and Test the DQN Model

> Package: **pes_dqn** — Deep Q-Network variant of the Pandemic Experiment Scenario

---

## Prerequisites

| Requirement | Detail |
|-------------|--------|
| Python | 3.10 (Windows) / 3.12 (Linux) |
| Virtual environment | `win_mpes_env` (Windows) or `linux_mpes_env` (Linux) |
| TensorFlow | 2.16.2 (CPU — no GPU required) |
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

## 1. Training the DQN Agent

### 1.1 Quick Start

```bash
python3 -m pes_dqn.ext.train_dqn
```

This runs the **full training pipeline** with default settings (100 000 episodes).

### 1.2 Custom Episode Count

Pass the number of episodes as the first argument:

```bash
python3 -m pes_dqn.ext.train_dqn 200000
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

All outputs are saved to `pes_dqn/inputs/<YYYY-MM-DD>_DQN_TRAIN/`:

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

These defaults are set in `config/CONFIG.py` and `ext/train_dqn.py`:

| Parameter | Default | Source |
|-----------|---------|--------|
| Hidden layers | `[64, 64]` | `CONFIG.DQN_HIDDEN_UNITS` |
| Batch size | 32 | `CONFIG.DQN_BATCH_SIZE` |
| Replay buffer | 50 000 | `CONFIG.DQN_REPLAY_BUFFER_SIZE` |
| Target sync freq | 1 000 steps | `CONFIG.DQN_TARGET_SYNC_FREQ` |
| Learning rate (Adam) | 1 × 10⁻³ | `CONFIG.DQN_LEARNING_RATE` |
| Train freq | every 4 env steps | `CONFIG.DQN_TRAIN_FREQ` |
| Discount γ | 0.865 | `train_dqn.py` |
| Initial ε | 0.679 | `train_dqn.py` |
| Min ε | 0.085 | `train_dqn.py` |
| Episodes | 100 000 | CLI arg or default |
| Random seed | 42 | `CONFIG.SEED` |

### 1.6 Expected Training Time

On a modest CPU (e.g. Intel i3-6006U @ 2 GHz, 4 threads), approximately
**15–30 minutes** for 100 000 episodes with CPU optimisations enabled
(NumPy replay buffer, no confidence forward pass during training,
TensorFlow thread-pool tuning).  Training time scales linearly with
episode count.

> **Note:** Without the CPU optimisations (`compute_confidence=True` and
> the old `deque`-based replay buffer), training may take 45–60 minutes
> on the same hardware.

### 1.7 CPU Training Optimisations

The following optimisations are active by default and require no
configuration:

| Optimisation | Effect | Location |
|-------------|--------|----------|
| NumPy replay buffer | Pre-allocated arrays with `randint` indexing instead of `deque` + `random.sample` | `ext/dqn_model.py` |
| Skip confidence forward pass | `compute_confidence=False` eliminates 1 of 2 forward passes per step | `ext/pandemic.py` |
| TF thread-pool tuning | `intra_op=0` (auto), `inter_op=2` for multi-core CPUs | `ext/dqn_model.py` |
| OMP_NUM_THREADS | Set to CPU core count before TF import | `__init__.py` |

To re-enable confidence tracking during training (at ~2× slower speed):

```python
rewards, model, confs = DQNTraining(
    env, lr, gamma, eps, min_eps, episodes,
    ...,
    compute_confidence=True,   # ← enables meta-cognitive observation
)
```

---

## 2. Bayesian Hyperparameter Optimisation (Optional)

If the default hyperparameters are not satisfactory, you can run an
automated search using Optuna:

```bash
python3 -m pes_dqn.ext.optimize_dqn 30
```

The integer argument is the number of trials (default: 30).

### 2.1 Resume a Previous Search

Optimisation state is stored in an SQLite database.  To resume:

```bash
python3 -m pes_dqn.ext.optimize_dqn 50 --resume 2026-03-02
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

### 2.3 Running on Google Colab Pro+

If the local machine does not have enough RAM (the full optimisation needs
~900 MB per process), you can run it on Google Colab Pro+ using the notebook
at `colab/Bayesian_Colab.ipynb`.  See `utils/colab_workflow.md`
for the complete step-by-step guide.

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
   `pes_dqn/outputs/<YYYY-MM-DD>_DQN_AGENT/`.

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
    1: 'RL_AGENT',           # DQN agent labelled as RL_AGENT
    2: 'DQN_AGENT'           # DQN agent labelled as DQN_AGENT
}[2]                         # <-- change index to select
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
python3 -m pes_dqn.ext.train_dqn
```

### Rewards file not found

Same as above — the training pipeline generates both `dqn_model.keras` and
`rewards.npy`.

### TensorFlow warnings

TensorFlow GPU/CUDA warnings are suppressed by default
(`TF_CPP_MIN_LOG_LEVEL=3`).  If you still see them, they are harmless — the
model runs on CPU.

### Poor agent performance

- Try increasing episodes: `python3 -m pes_dqn.ext.train_dqn 200000`
- Run Bayesian optimisation to find better hyperparameters:
  `python3 -m pes_dqn.ext.optimize_dqn 50`
- Check that `CONFIG.SEED = 42` for reproducible results.

### Slow training on CPU

The default configuration is already optimised for CPU.  If training is
still too slow:

- Reduce episode count: `python3 -m pes_dqn.ext.train_dqn 50000`
- Verify `compute_confidence=False` is set (default) in `train_dqn.py`
- Check that `OMP_NUM_THREADS` is set to your CPU core count
  (auto-configured in `__init__.py`)

---

## 5. Quick Reference

| Task | Command |
|------|---------|
| Activate environment (Linux) | `source linux_mpes_env/bin/activate` |
| Activate environment (Windows) | `win_mpes_env\Scripts\Activate.ps1` |
| Train DQN (default) | `python3 -m pes_dqn.ext.train_dqn` |
| Train DQN (custom) | `python3 -m pes_dqn.ext.train_dqn 200000` |
| Optimise hyperparams | `python3 -m pes_dqn.ext.optimize_dqn 30` |
| Resume optimisation | `python3 -m pes_dqn.ext.optimize_dqn 50 --resume YYYY-MM-DD` |
| Run experiment | `python3 -m pes_dqn` |
