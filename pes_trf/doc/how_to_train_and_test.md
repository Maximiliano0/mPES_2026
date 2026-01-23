# How to Train and Test the Transformer Model

> Package: **pes_trf** — Causal Transformer variant of the Pandemic Experiment Scenario

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

## 1. Training the Transformer Agent

### 1.1 Quick Start

```bash
python3 -m pes_trf.ext.train_transformer
```

This runs the **full training pipeline** with default settings (1 500
batches × 64 episodes/batch = 96 000 total episodes).

### 1.2 Custom Batch Count

Pass the number of batches as the first argument:

```bash
python3 -m pes_trf.ext.train_transformer 3000
```

### 1.3 What Happens During Training

The pipeline proceeds through these stages:

1. **Configure** — sets up the `PandemicTransformer` model (~77 000
   parameters) with default architecture (d=64, 4 heads, 2 layers).
2. **Training loop** — for each batch:
   - Collects 64 episodes using the current policy (sampling, not greedy).
   - Pads all episodes to `max_seq_len=10` and computes padding masks.
   - Computes discounted returns and advantages via GAE (γ=0.98, λ=0.95).
   - Normalises advantages (mean=0, std=1).
   - Runs a single gradient update with the combined loss:
     $\mathcal{L} = \mathcal{L}_\text{policy} + 0.5 \cdot \mathcal{L}_\text{value} - 0.01 \cdot H(\pi)$
   - Clips gradients by global norm (0.5).
3. **Periodic evaluation** (every 100 batches) — runs the greedy policy on
   64 fixed evaluation sequences and tracks the best-performing weights.
4. **Restore best weights** and run a final evaluation.
5. **Save artefacts** — writes model config/weights, reward history, and
   training visualisations to a dated subdirectory.

### 1.4 Training Output Files

All outputs are saved to `pes_trf/inputs/<YYYY-MM-DD>_TRANSFORMER_TRAIN/`:

| File | Description |
|------|-------------|
| `transformer_config.json` | Architecture hyperparameters (d_model, n_heads, etc.) |
| `transformer_weights.weights.h5` | Trained model weights |
| `rewards_<date>.npy` | Reward history (one value per episode) |
| `training_curves_<date>.png` | Two-panel plot: smoothed rewards + performance distribution |
| `transformer_confidence_distribution.png` | Confidence histogram from final evaluation |
| `training_report_<date>.txt` | Summary of architecture, hyperparameters, and results |

Additionally, the model files are **copied** to the standard paths
consumed by the experiment runner:

- `pes_trf/inputs/transformer_config.json`
- `pes_trf/inputs/transformer_weights.weights.h5`
- `pes_trf/inputs/rewards.npy`

### 1.5 Default Hyperparameters

**Architecture** (in `ext/train_transformer.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 64 | Embedding dimension |
| `n_heads` | 4 | Attention heads |
| `n_layers` | 2 | Transformer blocks |
| `d_ff` | 128 | Feed-forward hidden dimension |
| `max_seq_len` | 10 | Maximum episode length |
| `dropout_rate` | 0.1 | Dropout probability |

**Training** (in `ext/train_transformer.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_batches` | 1 500 | Total gradient updates |
| `batch_size` | 64 | Episodes per batch |
| Learning rate (Adam) | 3 × 10⁻⁴ | Optimizer LR |
| Discount γ | 0.98 | Reward discounting |
| GAE λ | 0.95 | Advantage estimation decay |
| `value_coeff` | 0.5 | Value loss weight |
| `entropy_coeff` | 0.01 | Entropy bonus weight |
| `max_grad_norm` | 0.5 | Gradient clipping threshold |
| `eval_every` | 100 batches | Evaluation interval |
| Random seed | 42 | `CONFIG.SEED` |

### 1.6 Expected Training Time

The Transformer is more expensive per episode than feedforward models due
to the attention mechanism.  On a modest CPU (e.g. Intel i3-6006U @ 2 GHz,
4 threads), expect **30–60 minutes** for 1 500 batches.

Training time scales linearly with `n_batches`.

### 1.7 Key Differences from DQN/A2C Training

| Aspect | DQN / A2C | Transformer |
|--------|-----------|-------------|
| Input | Single state | Full episode trajectory |
| Update frequency | Per step (DQN) / per episode (A2C) | Per batch of 64 episodes |
| Advantage estimator | TD(0) | GAE (λ=0.95) |
| Gradient clipping | None | Global norm (0.5) |
| Advantage normalisation | None | (μ=0, σ=1) |
| Model format | `.keras` | `config.json` + `.weights.h5` |

---

## 2. Bayesian Hyperparameter Optimisation (Q-Learning Baseline)

> **Note:** The optimisation module (`ext/optimize_tr.py`) optimises
> **tabular Q-Learning** hyperparameters as a baseline comparison, **not**
> Transformer hyperparameters.

```bash
python3 -m pes_trf.ext.optimize_tr 30
```

### 2.1 Resume a Previous Search

```bash
python3 -m pes_trf.ext.optimize_tr 50 --resume 2026-03-02
```

### 2.2 Optimisation Outputs

Saved to `pes_trf/inputs/<YYYY-MM-DD>_BAYESIAN_OPT/`:

| File | Description |
|------|-------------|
| `best_<date>.npy` | Best Q-table found |
| `rewards_best_<date>.npy` | Reward history of the best trial |
| `optimization_results_<date>.txt` | Full report with all trial results |
| `optimization_history_<date>.png` | Convergence plot |
| `hyperparameter_importances_<date>.png` | Parameter importance ranking |
| `optuna_study_<date>.db` | SQLite database (resumable) |

---

## 3. Testing the Transformer Agent (Running the Experiment)

### 3.1 Verify Training Files Exist

Before running the experiment, ensure these files are present:

```
pes_trf/inputs/transformer_config.json           ← architecture config
pes_trf/inputs/transformer_weights.weights.h5    ← trained weights
```

Both are automatically created by the training pipeline (§ 1).

### 3.2 Run the Experiment

```bash
python3 -m pes_trf
```

This launches the full experiment lifecycle:

1. **Validates** the Transformer model (loads config, checks d_model,
   n_layers, n_heads).
2. **Sets up** the experiment session with logging and dated output folders.
3. **Iterates** through 8 blocks × 8 sequences × 3–10 trials per sequence.
4. At each trial, the Transformer agent:
   - Appends the current state to the episode trajectory buffer.
   - Feeds the **full trajectory** to the model (not just the current state).
   - Computes policy logits at the last timestep with causal attention.
   - Masks infeasible actions (allocation > resources_left).
   - Selects the action with the highest probability (greedy argmax).
   - Computes **meta-cognitive confidence** from the entropy of the
     softmax distribution.
   - Simulates human-like **response timing** based on confidence.
5. The trajectory buffer **resets** at the start of each new sequence.
6. **Saves** results (performance JSON, visualisation PNG, response logs) to
   `pes_trf/outputs/<YYYY-MM-DD>_RL_AGENT/`.

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
PLAYER_TYPE = {
    1: 'RL_AGENT',       # Transformer agent
    2: 'TRANSFORMER'     # Pre-trained Transformer model
}[2]                     # <-- change index to select
SEED = 42
NUM_BLOCKS = 8
NUM_SEQUENCES = 8
```

---

## 4. Troubleshooting

### Model files not found

```
Transformer config or weights not found!
Expected: .../pes_trf/inputs/transformer_config.json
```

**Solution:** Run training first:

```bash
python3 -m pes_trf.ext.train_transformer
```

### TensorFlow warnings

TensorFlow GPU/CUDA warnings are suppressed by default
(`CUDA_VISIBLE_DEVICES=-1`).  If you still see them, they are harmless —
the model runs on CPU.

### Poor agent performance

- Increase batch count: `python3 -m pes_trf.ext.train_transformer 3000`
- The training loop tracks the best weights automatically — poor early
  performance is expected.
- Check that `SEED = 42` for reproducible results.

### Slow training on CPU

- Reduce batch count: `python3 -m pes_trf.ext.train_transformer 500`
- The Transformer is inherently more expensive than feedforward models
  due to the attention mechanism, but episodes are short (≤10 steps).

### Training appears stuck

The training loop prints progress every 50 batches and evaluates every
100 batches.  If the first 100 batches show low performance, this is
normal — the model needs several hundred batches to converge.

---

## 5. Quick Reference

| Task | Command |
|------|---------|
| Activate environment (Linux) | `source linux_mpes_env/bin/activate` |
| Activate environment (Windows) | `win_mpes_env\Scripts\Activate.ps1` |
| Train Transformer (default) | `python3 -m pes_trf.ext.train_transformer` |
| Train Transformer (custom) | `python3 -m pes_trf.ext.train_transformer 3000` |
| Optimise Q-Learning baseline | `python3 -m pes_trf.ext.optimize_tr 30` |
| Resume optimisation | `python3 -m pes_trf.ext.optimize_tr 50 --resume YYYY-MM-DD` |
| Run experiment | `python3 -m pes_trf` |
