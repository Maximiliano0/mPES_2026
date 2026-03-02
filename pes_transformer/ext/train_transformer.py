'''
PES_Transformer — Training script for the Transformer-based RL agent.

Trains a ``PandemicTransformer`` actor-critic model on the Pandemic
environment using REINFORCE with a learned value-function baseline
(Williams, 1992).  The training loop mirrors the structure of
``PES_QLv2/ext/train_rl.py`` but replaces tabular Q-Learning with
gradient-based policy optimisation through a Transformer encoder.

Pipeline stages
---------------
1. Instantiate Pandemic environment with CONFIG parameters.
2. Build the ``PandemicTransformer`` model (~77 K parameters).
3. Collect episodes in batches, computing discounted returns and
   advantages per timestep.
4. Update the model with the combined loss::

       L = L_policy + c_v · L_value − c_e · H(π)

   where ``L_policy = −log π(a|s) · A(s)`` (policy gradient),
   ``L_value = MSE(V, R)`` (critic), and ``H(π)`` is the entropy
   bonus that encourages exploration.
5. Evaluate the model every ``eval_every`` batches on the 64 fixed
   sequences (same benchmark as Q-Learning).
6. Save the best model weights, reward history, and confidence plots
   to ``INPUTS_PATH/<date>_TRANSFORMER_TRAIN/``.

Default hyperparameters::

    d_model          = 64
    n_heads          = 4
    n_layers         = 2
    d_ff             = 128
    learning_rate    = 3e-4   (Adam)
    gamma            = 0.98
    batch_size       = 64     (episodes per update)
    n_batches        = 1500   (total gradient steps)
    value_coeff      = 0.5
    entropy_coeff    = 0.01
    max_grad_norm    = 0.5

Usage::

    python3 -m PES_Transformer.ext.train_transformer [n_batches]

References
----------
Williams, R. J. (1992). "Simple statistical gradient-following algorithms
for connectionist reinforcement learning." Machine Learning, 8, 229–256.

Schulman, J. et al. (2016). "High-dimensional continuous control using
generalized advantage estimation." ICLR 2016.
'''

# ─────────────────────────────────────────────────────────────────
#  Imports
# ─────────────────────────────────────────────────────────────────
import os
import sys
import numpy
import warnings
import matplotlib.pyplot as plt
from datetime import datetime

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf

from .. import INPUTS_PATH
from ..config.CONFIG import SEED

from .tools import plot_confidences
from ..src.terminal_utils import header, section, success, info, list_item
from .pandemic import (
    Pandemic, rl_agent_meta_cognitive, run_experiment,
)
from PES_Transformer.ext.transformer_model import PandemicTransformer, TransformerAgent

warnings.filterwarnings('ignore', category=UserWarning,
                        message='.*Box bound precision.*')
warnings.filterwarnings('ignore', message='.*A NumPy version.*SciPy.*')


# ─────────────────────────────────────────────────────────────────
#  Trajectory collection
# ─────────────────────────────────────────────────────────────────

def collect_batch(env, model, batch_size, max_seq_len=10):
    """Collect a batch of episodes using the current policy.

    Parameters
    ----------
    env : Pandemic
        Environment instance (will be reset for each episode).
    model : PandemicTransformer
        Current model (used in inference mode).
    batch_size : int
        Number of episodes to collect.
    max_seq_len : int
        Pad / truncate all episodes to this length.

    Returns
    -------
    states : numpy.ndarray, shape (batch, max_seq_len, 3)
    actions : numpy.ndarray, shape (batch, max_seq_len)
    rewards : numpy.ndarray, shape (batch, max_seq_len)
    masks : numpy.ndarray, shape (batch, max_seq_len)
        1.0 for real timesteps, 0.0 for padding.
    log_probs : numpy.ndarray, shape (batch, max_seq_len)
    values : numpy.ndarray, shape (batch, max_seq_len)
    """
    all_s, all_a, all_r, all_m, all_lp, all_v = [], [], [], [], [], []

    agent = TransformerAgent(model, greedy=False)

    for _ in range(batch_size):
        env.random_sequence()
        state = env.reset()
        agent.reset()

        ep_s, ep_a, ep_r, ep_lp, ep_v = [], [], [], [], []
        done = False
        while not done:
            action, _probs, value, log_prob = agent.act(state)
            # Clamp action to available resources
            action = min(action, int(state[0]))
            state2, reward, done, _ = env.step(action)

            ep_s.append(list(state))
            ep_a.append(action)
            ep_r.append(reward)
            ep_lp.append(log_prob)
            ep_v.append(value)
            state = state2

        T = len(ep_s)
        pad = max_seq_len - T

        ep_s += [[0, 0, 0]] * pad
        ep_a += [0] * pad
        ep_r += [0.0] * pad
        ep_lp += [0.0] * pad
        ep_v += [0.0] * pad
        mask = [1.0] * T + [0.0] * pad

        all_s.append(ep_s)
        all_a.append(ep_a)
        all_r.append(ep_r)
        all_m.append(mask)
        all_lp.append(ep_lp)
        all_v.append(ep_v)

    return (
        numpy.array(all_s, dtype=numpy.float32),
        numpy.array(all_a, dtype=numpy.int32),
        numpy.array(all_r, dtype=numpy.float32),
        numpy.array(all_m, dtype=numpy.float32),
        numpy.array(all_lp, dtype=numpy.float32),
        numpy.array(all_v, dtype=numpy.float32),
    )


# ─────────────────────────────────────────────────────────────────
#  Return / advantage computation
# ─────────────────────────────────────────────────────────────────

def compute_returns(rewards, masks, gamma):
    """Compute discounted Monte-Carlo returns per timestep.

    Parameters
    ----------
    rewards : numpy.ndarray, shape (batch, seq_len)
    masks : numpy.ndarray, shape (batch, seq_len)
    gamma : float

    Returns
    -------
    returns : numpy.ndarray, shape (batch, seq_len)
    """
    B, T = rewards.shape
    returns = numpy.zeros_like(rewards)
    for b in range(B):
        R = 0.0
        for t in reversed(range(T)):
            if masks[b, t] == 0.0:
                continue
            R = rewards[b, t] + gamma * R
            returns[b, t] = R
    return returns


def compute_gae(rewards, values, masks, gamma, lam=0.95):
    """Generalised Advantage Estimation (Schulman et al., 2016).

    Parameters
    ----------
    rewards : numpy.ndarray, shape (batch, seq_len)
    values : numpy.ndarray, shape (batch, seq_len)
    masks : numpy.ndarray, shape (batch, seq_len)
    gamma : float
    lam : float

    Returns
    -------
    advantages : numpy.ndarray, shape (batch, seq_len)
    returns : numpy.ndarray, shape (batch, seq_len)
    """
    B, T = rewards.shape
    advantages = numpy.zeros_like(rewards)
    returns = numpy.zeros_like(rewards)

    for b in range(B):
        gae = 0.0
        for t in reversed(range(T)):
            if masks[b, t] == 0.0:
                continue
            # next value (0 if terminal or padding)
            next_val = values[b, t + 1] if (t + 1 < T and masks[b, t + 1] > 0) else 0.0
            delta = rewards[b, t] + gamma * next_val - values[b, t]
            gae = delta + gamma * lam * gae
            advantages[b, t] = gae
            returns[b, t] = gae + values[b, t]

    return advantages, returns


# ─────────────────────────────────────────────────────────────────
#  Training step
# ─────────────────────────────────────────────────────────────────

@tf.function
def train_step(model, optimizer, states, actions, returns, advantages,
               masks, value_coeff, entropy_coeff, max_grad_norm):
    """Single gradient update (policy gradient + value + entropy).

    Parameters
    ----------
    model : PandemicTransformer
    optimizer : tf.keras.optimizers.Optimizer
    states : tf.Tensor (batch, seq, 3)
    actions : tf.Tensor (batch, seq)  int32
    returns : tf.Tensor (batch, seq)
    advantages : tf.Tensor (batch, seq)
    masks : tf.Tensor (batch, seq)
    value_coeff : float
    entropy_coeff : float
    max_grad_norm : float

    Returns
    -------
    total_loss, policy_loss, value_loss, entropy : tf.Tensor (scalars)
    """
    with tf.GradientTape() as tape:
        policy_logits, values = model(states, padding_mask=masks, training=True)

        # Policy distribution
        log_probs_all = tf.nn.log_softmax(policy_logits, axis=-1)  # (B, T, A)
        actions_oh = tf.one_hot(actions, depth=model.n_actions)    # (B, T, A)
        log_probs = tf.reduce_sum(log_probs_all * actions_oh, axis=-1)  # (B, T)

        # Entropy  H(π) = −Σ π log π
        probs_all = tf.nn.softmax(policy_logits, axis=-1)
        entropy = -tf.reduce_sum(probs_all * log_probs_all, axis=-1)  # (B, T)

        # Apply masks
        policy_loss = -tf.reduce_sum(log_probs * advantages * masks) / tf.reduce_sum(masks)
        value_loss = tf.reduce_sum(tf.square(values - returns) * masks) / tf.reduce_sum(masks)
        entropy_mean = tf.reduce_sum(entropy * masks) / tf.reduce_sum(masks)

        total_loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy_mean

    grads = tape.gradient(total_loss, model.trainable_variables)
    # Gradient clipping
    grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return total_loss, policy_loss, value_loss, entropy_mean


# ─────────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────────

def evaluate_model(env, model, trials_per_sequence, initial_severities):
    """Evaluate trained model over 64 fixed sequences.

    Parameters
    ----------
    env : Pandemic
    model : PandemicTransformer
    trials_per_sequence : numpy.ndarray, shape (64,)
    initial_severities : list of lists

    Returns
    -------
    mean_perf : float
        Mean normalised performance across 64 sequences.
    perfs : list
        Per-sequence performance values.
    confs : list
        Confidence values per decision.
    """
    agent = TransformerAgent(model, greedy=True)
    confs = []

    def action_fn(_env_inner, state, _seqid):
        if state[1] == 0:
            agent.reset()
        action, probs, _value, _ = agent.act(state, resources_left=state[0])
        # Compute entropy-based confidence
        _, conf, _, _ = rl_agent_meta_cognitive(probs, state[0], 10000)
        confs.append(conf)
        return action

    _, perfs, _ = run_experiment(
        env, action_fn, RandomSequences=False,
        trials_per_sequence=trials_per_sequence,
        sevs=initial_severities,
        NumberOfIterations=64,
    )

    return float(numpy.mean(perfs)), perfs, confs


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────

def main():
    """Train, evaluate, and save a Transformer agent."""
    header("PES_Transformer — Transformer Training", width=80)

    # ── parse arguments ──────────────────────────────────────────
    n_batches = 1500
    if len(sys.argv) >= 2:
        try:
            n_batches = int(sys.argv[1])
        except ValueError:
            pass

    # ── hyperparameters ──────────────────────────────────────────
    batch_size     = 64
    gamma          = 0.98
    learning_rate  = 3e-4
    value_coeff    = 0.5
    entropy_coeff  = 0.01
    max_grad_norm  = 0.5
    eval_every     = 100       # evaluate every N batches
    max_seq_len    = 10

    # Architecture
    d_model    = 64
    n_heads    = 4
    n_layers   = 2
    d_ff       = 128
    dropout    = 0.1

    total_episodes = n_batches * batch_size

    section("Configuration")
    for label, val in [
        ("Architecture",    f"d={d_model}, h={n_heads}, L={n_layers}, ff={d_ff}"),
        ("Batches",         n_batches),
        ("Batch size",      batch_size),
        ("Total episodes",  f"{total_episodes:,}"),
        ("Learning rate",   learning_rate),
        ("Gamma",           gamma),
        ("Value coeff",     value_coeff),
        ("Entropy coeff",   entropy_coeff),
        ("SEED",            SEED),
    ]:
        list_item(f"{label}: {val}")
    print()

    # ── reproducibility ──────────────────────────────────────────
    tf.random.set_seed(SEED)
    numpy.random.seed(SEED)

    # ── environment ──────────────────────────────────────────────
    env = Pandemic()
    env.verbose = False

    # Load fixed evaluation data
    SequenceLengthsCsv = os.path.join(INPUTS_PATH, 'sequence_lengths.csv')
    InitialSeverityCsv = os.path.join(INPUTS_PATH, 'initial_severity.csv')

    trials_per_seq = numpy.loadtxt(SequenceLengthsCsv, delimiter=',').astype(int)
    all_severities = numpy.loadtxt(InitialSeverityCsv, delimiter=',')

    # Build list-of-lists for evaluation
    sevs_eval = []
    idx = 0
    for tps in trials_per_seq:
        sevs_eval.append(all_severities[idx:idx + tps].tolist())
        idx += tps

    # ── build model ──────────────────────────────────────────────
    model = PandemicTransformer(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        d_ff=d_ff, max_seq_len=max_seq_len, n_actions=11,
        dropout_rate=dropout,
    )
    # Build
    model(tf.zeros((1, 1, 3), dtype=tf.float32))

    total_params = model.count_params()
    info(f"Model parameters: {total_params:,}")
    print()

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # ── training loop ────────────────────────────────────────────
    section("Training")

    reward_history = []
    best_perf = -1.0
    best_weights = None

    for batch_idx in range(1, n_batches + 1):
        # Collect
        states, actions, rewards, masks, _log_probs, values = collect_batch(
            env, model, batch_size, max_seq_len,
        )

        # Track mean episode reward
        ep_rewards = (rewards * masks).sum(axis=1)
        reward_history.extend(ep_rewards.tolist())

        # Returns and advantages (GAE)
        advantages, returns = compute_gae(rewards, values, masks, gamma, lam=0.95)

        # Normalise advantages
        adv_valid = advantages[masks > 0]
        if len(adv_valid) > 1:
            adv_mean = adv_valid.mean()
            adv_std = adv_valid.std() + 1e-8
            advantages = (advantages - adv_mean) / adv_std

        # Convert to tensors
        t_states = tf.constant(states)
        t_actions = tf.constant(actions)
        t_returns = tf.constant(returns, dtype=tf.float32)
        t_advantages = tf.constant(advantages, dtype=tf.float32)
        t_masks = tf.constant(masks)

        total_loss, p_loss, v_loss, ent = train_step(
            model, optimizer, t_states, t_actions, t_returns,
            t_advantages, t_masks, value_coeff, entropy_coeff, max_grad_norm,
        )

        # ── logging ─────────────────────────────────────────────
        if batch_idx % 50 == 0 or batch_idx == 1:
            mean_rew = float(numpy.mean(ep_rewards))
            info(
                f"Batch {batch_idx:>5}/{n_batches}  |  "
                f"loss={float(total_loss):.4f}  "
                f"p={float(p_loss):.4f}  "
                f"v={float(v_loss):.4f}  "
                f"H={float(ent):.3f}  |  "
                f"mean_reward={mean_rew:.2f}"
            )

        # ── periodic evaluation ─────────────────────────────────
        if batch_idx % eval_every == 0 or batch_idx == n_batches:
            env_eval = Pandemic()
            env_eval.verbose = False
            mean_perf, perfs, _confs = evaluate_model(
                env_eval, model, trials_per_seq, sevs_eval,
            )
            success(
                f"Eval @ batch {batch_idx}: "
                f"mean_perf = {mean_perf:.4f}  "
                f"std = {numpy.std(perfs):.4f}"
            )
            if mean_perf > best_perf:
                best_perf = mean_perf
                best_weights = model.get_weights()
                success(f"  ★ New best performance: {best_perf:.4f}")

    # ── restore best weights ─────────────────────────────────────
    if best_weights is not None:
        model.set_weights(best_weights)
        success(f"Restored best weights (perf = {best_perf:.4f})")

    # ── final evaluation ─────────────────────────────────────────
    section("Final Evaluation")
    env_eval = Pandemic()
    env_eval.verbose = False
    final_perf, final_perfs, final_confs = evaluate_model(
        env_eval, model, trials_per_seq, sevs_eval,
    )
    success(f"Final performance: mean = {final_perf:.4f}, std = {numpy.std(final_perfs):.4f}")

    # ── save artefacts ───────────────────────────────────────────
    section("Saving Artefacts")

    today_str = datetime.now().strftime('%Y-%m-%d')
    save_dir = os.path.join(INPUTS_PATH, f'{today_str}_TRANSFORMER_TRAIN')
    os.makedirs(save_dir, exist_ok=True)

    # Save model
    model.save_pretrained(save_dir)
    success(f"Model saved to {save_dir}")

    # Also save to INPUTS_PATH root for experiment execution
    model.save_pretrained(INPUTS_PATH)
    success("Model also saved to INPUTS_PATH (for __main__.py)")

    # Save reward history
    rewards_path = os.path.join(save_dir, f'rewards_{today_str}.npy')
    numpy.save(rewards_path, numpy.array(reward_history))

    # Also save as rewards.npy in INPUTS_PATH for compatibility
    numpy.save(os.path.join(INPUTS_PATH, 'rewards.npy'), numpy.array(reward_history))
    success("Reward history saved")

    # ── plots ────────────────────────────────────────────────────
    # Reward curve (smoothed)
    _fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    window = max(1, len(reward_history) // 100)
    smoothed = numpy.convolve(reward_history,
                           numpy.ones(window) / window, mode='valid')
    axes[0].plot(smoothed, linewidth=0.8)
    axes[0].set_title('Training Reward (smoothed)')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Episode Reward')
    axes[0].grid(True, alpha=0.3)

    # Evaluation performance histogram
    axes[1].hist(final_perfs, bins=20, edgecolor='black', alpha=0.7)
    axes[1].axvline(final_perf, color='red', linestyle='--',
                    label=f'mean = {final_perf:.3f}')
    axes[1].set_title('Final Performance Distribution (64 sequences)')
    axes[1].set_xlabel('Normalised Performance')
    axes[1].set_ylabel('Count')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, f'training_curves_{today_str}.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    success(f"Training plots saved to {plot_path}")

    # Confidence plot (reuse existing utility)
    if len(final_confs) > 0:
        try:
            plot_confidences(final_confs, "Transformer Confidence Distribution", Show=False)
            plt.savefig(os.path.join(save_dir, "transformer_confidence_distribution.png"), dpi=150)
            plt.close()
            success("Confidence plots saved")
        except Exception as e:
            info(f"Confidence plot skipped: {e}")

    # ── summary report ───────────────────────────────────────────
    section("Summary")
    report_lines = [
        f"PES_Transformer — Training Report",
        f"Date: {today_str}",
        f"",
        f"Architecture:",
        f"  d_model = {d_model}",
        f"  n_heads = {n_heads}",
        f"  n_layers = {n_layers}",
        f"  d_ff = {d_ff}",
        f"  total_params = {total_params:,}",
        f"",
        f"Training:",
        f"  batches = {n_batches}",
        f"  batch_size = {batch_size}",
        f"  total_episodes = {total_episodes:,}",
        f"  learning_rate = {learning_rate}",
        f"  gamma = {gamma}",
        f"  value_coeff = {value_coeff}",
        f"  entropy_coeff = {entropy_coeff}",
        f"  SEED = {SEED}",
        f"",
        f"Results:",
        f"  best_perf = {best_perf:.4f}",
        f"  final_mean_perf = {final_perf:.4f}",
        f"  final_std_perf = {numpy.std(final_perfs):.4f}",
        f"  final_min_perf = {numpy.min(final_perfs):.4f}",
        f"  final_max_perf = {numpy.max(final_perfs):.4f}",
    ]

    report_path = os.path.join(save_dir, f'training_report_{today_str}.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    success(f"Report saved to {report_path}")

    print()
    for line in report_lines:
        print(f"  {line}")
    print()

    env.close()
    success("Training complete ✓")


# ─────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    main()
