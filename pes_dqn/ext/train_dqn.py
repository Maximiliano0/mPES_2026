'''
pes_dqn - Pandemic Experiment Scenario: DQN Training Pipeline

Trains a Deep Q-Network agent on the Pandemic environment and evaluates
it against a random-player baseline.

Pipeline stages
---------------
1. Load training data (initial_severity.csv, sequence_lengths.csv)
2. Run random-player baseline and save performance plots
3. Train DQN agent (default 100 000 episodes, configurable via CLI)
4. Save Keras model, rewards history, and training config to a dated directory
5. Evaluate trained agent on the same sequences and generate
   performance / confidence visualisations

Usage
-----
::

    python3 -m pes_dqn.ext.train_dqn [num_episodes]
'''

##########################
##  Imports externos    ##
##########################
import os
import sys
import numpy
import warnings
import matplotlib.pyplot as plt
from datetime import datetime

# Force TensorFlow to use CPU by default before any TF import happens.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

##########################
##  Imports internos    ##
##########################
from .pandemic import Pandemic, dqn_agent_meta_cognitive, run_experiment, DQNTraining
from .dqn_model import normalize_state
from ..src.terminal_utils import header, section, success, info, list_item
from .tools import plot_confidences, convert_globalseq_to_seqs
from ..config.CONFIG import (SEED, DQN_HIDDEN_UNITS, DQN_BATCH_SIZE,
                             DQN_REPLAY_BUFFER_SIZE, DQN_TARGET_SYNC_FREQ,
                             DQN_LEARNING_RATE, DQN_TRAIN_FREQ, DQN_MODEL_FILE)
from .. import INPUTS_PATH


# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*Box bound precision.*')
warnings.filterwarnings('ignore', message='.*A NumPy version.*SciPy.*')

###################################
##             Main             ###
###################################


def main():
    """Run the full DQN training and evaluation pipeline."""

    header("DQN TRAINING PIPELINE", width=80)

    # Configure matplotlib for better aesthetics
    try:
        plt.style.use('ggplot')
    except BaseException:
        pass  # Use default if style is not available

    matplotlib_config = {
        'figure.figsize': (12, 6),
        'figure.dpi': 100,
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'lines.linewidth': 2,
        'lines.markersize': 6
    }
    plt.rcParams.update(matplotlib_config)

    # Create training output directory with date stamp
    train_date = datetime.now().strftime("%Y-%m-%d")
    train_dir = os.path.join(INPUTS_PATH, f'{train_date}_DQN_TRAIN')
    os.makedirs(train_dir, exist_ok=True)
    info(f"Output directory: {train_dir}")

    # Load initial severity and sequence lengths data
    section("Loading Training Data", width=80)
    trials_per_sequence = numpy.loadtxt(os.path.join(INPUTS_PATH, 'sequence_lengths.csv'), delimiter=',')
    all_severities = numpy.loadtxt(os.path.join(INPUTS_PATH, 'initial_severity.csv'), delimiter=',')

    list_item(f"Sequence lengths shape: {trials_per_sequence.shape}")
    list_item(f"Initial severities shape: {all_severities.shape}")
    list_item(f"Total trials: {int(sum(trials_per_sequence))}")
    print()

    sevs = convert_globalseq_to_seqs(trials_per_sequence, all_severities)

    # Calculate probability distributions for number of cities (trials per sequence)
    val_cities, count_cities = numpy.unique(trials_per_sequence, return_counts=True)
    number_cities_prob = numpy.asarray((val_cities, count_cities / len(trials_per_sequence))).T

    # Calculate probability distributions for initial severities
    val_severity, count_severity = numpy.unique(all_severities, return_counts=True)
    severity_prob = numpy.asarray((val_severity, count_severity / len(all_severities))).T

    env = Pandemic()

    def random_qf(_env, _state, _seqid):
        """Return a random action from the environment's action space."""
        return env.sample()

    section("Random Player Baseline", width=80)
    info("Training random agent for comparison...")
    seqs1, perfs1, _ = run_experiment(env, random_qf, False, trials_per_sequence, sevs)
    success("Random player experiment completed")

    __fig, ax = plt.subplots(figsize=(12, 6))
    assert isinstance(ax, plt.Axes)
    ax.plot(seqs1, color='#1f77b4', linewidth=2.5, marker='o', markersize=5, label='Random Player')
    ax.set_xlabel('Trial', fontsize=12, fontweight='bold')
    ax.set_ylabel('Final Severity Achieved', fontsize=12, fontweight='bold')
    ax.set_title('Random Player Baseline: Severity per Sequence', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            train_dir,
            f'random_player_sequence_performance_{train_date}.png'),
        dpi=150,
        bbox_inches='tight')
    plt.close()
    list_item("Saved: random_player_sequence_performance.png")

    __fig, ax = plt.subplots(figsize=(12, 6))
    assert isinstance(ax, plt.Axes)
    ax.plot(perfs1, color='#ff7f0e', linewidth=2.5, marker='s', markersize=5, label='Random Player')
    ax.set_ylabel('Normalised Performance (0-1)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Trial', fontsize=12, fontweight='bold')
    ax.set_title('Random Player Baseline: Normalised Performance per Sequence', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            train_dir,
            f'random_player_normalised_performance_{train_date}.png'),
        dpi=150,
        bbox_inches='tight')
    plt.close()
    list_item("Saved: random_player_normalised_performance.png")
    print()

    # Initialize pandemic environment with the calculated probability distributions
    env = Pandemic()

    env.number_cities_prob = number_cities_prob
    env.severity_prob = severity_prob
    env.verbose = False

    # ---- DQN Training ----
    section("DQN Training", width=80)

    # DQN hyperparameters
    learning_rate = DQN_LEARNING_RATE
    discount_factor = 0.865
    epsilon_initial = 0.679
    epsilon_min = 0.085
    num_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 100000

    info(f"Starting DQN training ({num_episodes:,} episodes)...")
    info(f"  Hidden units:       {DQN_HIDDEN_UNITS}")
    info(f"  Batch size:         {DQN_BATCH_SIZE}")
    info(f"  Replay buffer:      {DQN_REPLAY_BUFFER_SIZE:,}")
    info(f"  Target sync freq:   {DQN_TARGET_SYNC_FREQ}")
    info(f"  Train freq:         {DQN_TRAIN_FREQ}")
    info("(This may take a while on CPU)")
    print()

    rewards, model, confsrl = DQNTraining(
        env, learning_rate, discount_factor,
        epsilon_initial, epsilon_min, num_episodes,
        hidden_units=DQN_HIDDEN_UNITS,
        batch_size=DQN_BATCH_SIZE,
        replay_buffer_size=DQN_REPLAY_BUFFER_SIZE,
        target_sync_freq=DQN_TARGET_SYNC_FREQ,
        train_freq=DQN_TRAIN_FREQ,
        seed=SEED,
        compute_confidence=True,
    )
    print()
    success("Training completed")
    list_item(f"Model parameters: {model.count_params()}")
    list_item(f"Rewards history length: {len(rewards)}")

    info("Saving trained model...")

    # Save Keras model and rewards with date stamp
    model_file = os.path.join(train_dir, f'dqn_model_{train_date}.keras')
    rewards_file = os.path.join(train_dir, f'rewards_{train_date}.npy')
    config_file = os.path.join(train_dir, f'training_config_{train_date}.txt')

    model.save(model_file)
    numpy.save(rewards_file, rewards)

    # Also save to the standard paths consumed by __main__.py / pygameMediator
    model.save(os.path.join(INPUTS_PATH, DQN_MODEL_FILE))
    numpy.save(os.path.join(INPUTS_PATH, 'rewards.npy'), rewards)

    # Create configuration file
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("DQN TRAINING CONFIGURATION\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Training Date: {train_date}\n")
        f.write(f"Training Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("DQN HYPERPARAMETERS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Learning Rate (Adam):        {learning_rate}\n")
        f.write(f"Discount Factor (γ):         {discount_factor}\n")
        f.write(f"Initial Epsilon (ε):         {epsilon_initial}\n")
        f.write(f"Minimum Epsilon (ε_min):     {epsilon_min}\n")
        f.write(f"Number of Episodes:          {num_episodes:,}\n")
        f.write(f"Epsilon Decay:               Linear ({epsilon_initial} → {epsilon_min})\n")
        f.write(f"Hidden Units:                {DQN_HIDDEN_UNITS}\n")
        f.write(f"Batch Size:                  {DQN_BATCH_SIZE}\n")
        f.write(f"Replay Buffer Size:          {DQN_REPLAY_BUFFER_SIZE:,}\n")
        f.write(f"Target Sync Frequency:       {DQN_TARGET_SYNC_FREQ}\n")
        f.write(f"Train Frequency:             {DQN_TRAIN_FREQ}\n\n")

        f.write("NETWORK ARCHITECTURE\n")
        f.write("-" * 80 + "\n")
        f.write(f"State Dimension:             3 (resources, trial, severity)\n")
        f.write(f"Action Dimension:            {model.output_shape[-1]}\n")
        f.write(f"Total Parameters:            {model.count_params()}\n")
        f.write(f"Input normalisation:         [res/30, trial/10, sev/9]\n\n")

        f.write("OUTPUT FILES\n")
        f.write("-" * 80 + "\n")
        f.write(f"Model File:                  dqn_model_{train_date}.keras\n")
        f.write(f"Rewards File:                rewards_{train_date}.npy\n")
        f.write(f"Configuration File:          training_config_{train_date}.txt\n\n")

        f.write("DESCRIPTION\n")
        f.write("-" * 80 + "\n")
        f.write("Deep Q-Network trained on the Pandemic Scenario.\n")
        f.write("The model maps normalised (resources, trial, severity) states\n")
        f.write("to Q-values for 11 possible resource-allocation actions.\n")
        f.write("The rewards file contains average reward progression every 10,000 episodes.\n")

    success(f"Model saved to dqn_model_{train_date}.keras")
    success(f"Rewards saved to rewards_{train_date}.npy")
    success(f"Configuration saved to training_config_{train_date}.txt")
    list_item(f"Training Directory: {train_dir}")
    print()

    section("Training Performance Analysis", width=80)
    info("Generating reward history visualization...")

    __fig, ax = plt.subplots(figsize=(12, 6))
    assert isinstance(ax, plt.Axes)
    ax.plot(100 * (numpy.arange(len(rewards)) + 1), rewards, color='#2ca02c', linewidth=2.5, label='Average Reward')
    ax.fill_between(100 * (numpy.arange(len(rewards)) + 1), rewards, alpha=0.2, color='#2ca02c')
    ax.set_xlabel('Episodes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Reward', fontsize=12, fontweight='bold')
    ax.set_title('DQN Training: Average Reward Progression', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(train_dir, f'dqn_agent_rewards_vs_episodes_{train_date}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    list_item(f"Saved: dqn_agent_rewards_vs_episodes_{train_date}.png")
    print()

    if True:  # pylint: disable=using-constant-test
        section("DQN Agent Evaluation", width=80)
        info("Running evaluation experiment with trained agent...")
        confsrl = []

        # Dimension limits for normalisation (must match training)
        _max_res = env.available_resources_states - 1
        _max_tri = env.trial_no_states - 1
        _max_sev = env.severity_states - 1

        def eval_qf(_env, state, _seqid):
            """Select the best action using the DQN model with meta-cognitive confidence."""
            s_norm = normalize_state(state, _max_res, _max_tri, _max_sev)
            q_vals = model(s_norm[numpy.newaxis, :], training=False)[0].numpy()

            _response, confidence, _rt_hold, _rt_release = dqn_agent_meta_cognitive(
                q_vals, state[0], 10000
            )

            if state[0] == 0:
                confidence = -1.0

            confsrl.append(confidence)
            return _response

        seqs, perfs, _ = run_experiment(env, eval_qf, False, trials_per_sequence, sevs)
        success("Evaluation experiment completed")

        info("Generating performance visualizations...")

        _fig, ax = plt.subplots(figsize=(12, 6))
        assert isinstance(ax, plt.Axes)
        ax.plot(seqs, color='#d62728', linewidth=2.5, marker='o', markersize=5, label='DQN Agent')
        ax.set_xlabel('Trial', fontsize=12, fontweight='bold')
        ax.set_ylabel('Final Severity Achieved', fontsize=12, fontweight='bold')
        ax.set_title('DQN Agent Evaluation: Severity per Sequence', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=11)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                train_dir,
                f'dqn_agent_sequence_performance_{train_date}.png'),
            dpi=150,
            bbox_inches='tight')
        plt.close()
        list_item(f"Saved: dqn_agent_sequence_performance_{train_date}.png")

        _fig, ax = plt.subplots(figsize=(12, 6))
        assert isinstance(ax, plt.Axes)
        ax.plot(perfs, color='#9467bd', linewidth=2.5, marker='s', markersize=5, label='DQN Agent')
        ax.set_ylabel('Normalised Performance (0-1)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Trial', fontsize=12, fontweight='bold')
        ax.set_title('DQN Agent Evaluation: Normalised Performance', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, 64)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=11)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                train_dir,
                f'dqn_agent_normalised_performance_{train_date}.png'),
            dpi=150,
            bbox_inches='tight')
        plt.close()
        list_item(f"Saved: dqn_agent_normalised_performance_{train_date}.png")

        cumperfs = numpy.cumsum(perfs)
        Domain = numpy.arange(1, 1 + 64)
        _fig, ax = plt.subplots(figsize=(12, 6))
        assert isinstance(ax, plt.Axes)
        ax.plot(cumperfs / Domain, color='#8c564b', linewidth=2.5, marker='^', markersize=5, label='DQN Agent')
        ax.set_ylabel('Cumulative Normalised Performance', fontsize=12, fontweight='bold')
        ax.set_xlabel('Trial', fontsize=12, fontweight='bold')
        ax.set_title('DQN Agent Evaluation: Cumulative Performance Trend', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0.5, 1.05)
        ax.set_xlim(0, 64)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=11)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                train_dir,
                f'dqn_agent_cumulative_performance_{train_date}.png'),
            dpi=150,
            bbox_inches='tight')
        plt.close()
        list_item(f"Saved: dqn_agent_cumulative_performance_{train_date}.png")

        _fig, ax = plt.subplots(figsize=(14, 5))
        assert isinstance(ax, plt.Axes)
        ax.scatter(
            numpy.asarray(
                range(
                    len(confsrl))),
            confsrl,
            color='#1f77b4',
            s=40,
            alpha=0.6,
            edgecolors='navy',
            linewidth=0.5)
        ax.set_title('DQN Agent: Decision Confidence Scores During Evaluation', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel('Confidence', fontsize=12, fontweight='bold')
        ax.set_xlabel('Trial Number', fontsize=12, fontweight='bold')
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(-10, 360)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(train_dir, f'dqn_agent_confidences_{train_date}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        list_item(f"Saved: dqn_agent_confidences_{train_date}.png")

        confsrl_arr = numpy.asarray(confsrl, dtype=numpy.float32)

        val_confidences = numpy.arange(11, dtype=numpy.float32) / 10.0
        _confsrl_hist = numpy.histogram(confsrl_arr, bins=val_confidences)

        plot_confidences(confsrl_arr, 'Confidences', Show=False)

        numpy.save(os.path.join(train_dir, f'confsrl_{train_date}.npy'), confsrl_arr)

        confsrl_arr = confsrl_arr[confsrl_arr != -1]

        print(confsrl_arr)

        I = confsrl_arr
        rescaled = (I - numpy.min(I)) * ((1.0 - 0.0) / (numpy.max(I) - numpy.min(I))) + 0.0
        remapconfrl = numpy.clip(rescaled, 0.0, 1.0)

        print(remapconfrl.shape)

        _fig, ax = plt.subplots(figsize=(14, 5))
        assert isinstance(ax, plt.Axes)
        ax.scatter(
            numpy.asarray(
                range(
                    remapconfrl.shape[0])),
            remapconfrl,
            color='#2ca02c',
            s=40,
            alpha=0.6,
            edgecolors='darkgreen',
            linewidth=0.5)
        ax.set_ylabel('Remapped Confidence (0-1)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Trial Number', fontsize=12, fontweight='bold')
        ax.set_title('DQN Agent: Normalised Confidence Scores', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(-10, 360)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                train_dir,
                f'dqn_agent_remapped_confidences_{train_date}.png'),
            dpi=150,
            bbox_inches='tight')
        plt.close()

        plot_confidences(remapconfrl, 'Remapped Confidences', Show=False)

    section("Training Complete", width=80)
    success("DQN training pipeline finished successfully!")
    info(f"Output directory: {train_dir}")
    print()
#
# END OF 'main()


if __name__ == '__main__':
    main()
