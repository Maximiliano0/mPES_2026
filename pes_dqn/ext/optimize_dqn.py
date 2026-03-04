'''
pes_dqn - Pandemic Experiment Scenario

Bayesian Optimization of DQN hyperparameters using Optuna.

Optimizes: learning_rate, discount_factor, epsilon_initial, epsilon_min,
           num_episodes, hidden_units, batch_size, replay_buffer_size,
           target_sync_freq, train_freq
Objective: maximize mean normalised performance over the 64 evaluation sequences.

The evaluation uses infeasible-action masking (actions > available resources are
suppressed before argmax) so that the metric matches the behaviour of the RL agent
in __main__.py.  The best model weights found during the search are preserved in
memory and saved directly, avoiding a lossy re-training step.

Usage:
    python3 -m pes_dqn.ext.optimize_dqn [n_trials] [--resume YYYY-MM-DD]

    n_trials : int, optional
        Number of Bayesian optimization trials (default: 30).
    --resume YYYY-MM-DD : str, optional
        Resume a previous optimization run stored under that date.

Search space:
    learning_rate        ∈ [5e-4, 5e-3]         (log scale)
    discount_factor      ∈ [0.85, 0.95]
    epsilon_initial      ∈ [0.50, 0.90]
    epsilon_min          ∈ [0.02, 0.10]
    num_episodes         ∈ [50000, 1000000]      (step=50000)
    hidden_dim           ∈ [32, 64]             (step=32)
    n_hidden_layers      ∈ {1, 2}
    batch_size           ∈ {32, 64}
    replay_buffer_size   ∈ [20000, 50000]       (step=10000)
    target_sync_freq     ∈ [500, 1500]          (step=500)
    train_freq           ∈ {2, 4}

Outputs (saved to INPUTS_PATH/<date>_BAYESIAN_OPT/):
    - dqn_best_<date>.keras            : Keras model from the best trial
    - rewards_best_<date>.npy          : Reward history of the best training run
    - optimization_results_<date>.txt  : Full report (1-based trial #)
    - optimization_history_<date>.png  : Convergence plot (1-based trial #)
    - hyperparameter_importances_<date>.png: Parameter importance plot
    - optuna_study_<date>.db           : SQLite database for resumable studies

Note:
    Trial numbering in reports and plots uses 1-based indexing to match
    the trial_id in the SQLite database.  Optuna internally uses 0-based
    trial.number; the +1 offset is applied at report-generation time.
'''

##########################
##  Imports externos    ##
##########################
import os
import sys
import time
import numpy
import warnings
import optuna
import matplotlib.pyplot as plt
from datetime import datetime

# Force TensorFlow to use CPU
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf  # noqa: E402  (after env var)

##########################
##  Imports internos    ##
##########################
from .pandemic import Pandemic, run_experiment, DQNTraining
from .dqn_model import build_q_network, normalize_state
from ..src.terminal_utils import header, section, success, info, list_item
from .tools import convert_globalseq_to_seqs
from ..config.CONFIG import SEED
from .. import INPUTS_PATH

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*Box bound precision.*')
warnings.filterwarnings('ignore', message='.*A NumPy version.*SciPy.*')

try:
    from utils.notify import notify
except ImportError:
    notify = lambda *a, **kw: None   # no-op if utils is not on sys.path

# Nombre del paquete para las notificaciones push
_PKG_NAME = __package__.split('.')[0] if __package__ else 'mPES'


###################################
##    Global evaluation data     ##
###################################
# Loaded once at startup and reused by every trial
_trials_per_sequence = None
_sevs = None
_number_cities_prob = None
_severity_prob = None

# Store best model weights/rewards during optimization to avoid lossy retraining
_best_artifacts: dict = {'weights': None, 'hidden_units': None,
                         'rewards': None, 'value': float('-inf')}


def _load_evaluation_data():
    """Load sequence lengths, severities and their probability distributions."""
    global _trials_per_sequence, _sevs, _number_cities_prob, _severity_prob

    _trials_per_sequence = numpy.loadtxt(
        os.path.join(INPUTS_PATH, 'sequence_lengths.csv'), delimiter=','
    )
    all_severities = numpy.loadtxt(
        os.path.join(INPUTS_PATH, 'initial_severity.csv'), delimiter=','
    )
    _sevs = convert_globalseq_to_seqs(_trials_per_sequence, all_severities)

    val_cities, count_cities = numpy.unique(_trials_per_sequence, return_counts=True)
    _number_cities_prob = numpy.asarray((val_cities, count_cities / len(_trials_per_sequence))).T

    val_severity, count_severity = numpy.unique(all_severities, return_counts=True)
    _severity_prob = numpy.asarray((val_severity, count_severity / len(all_severities))).T


###################################
##     Objective function        ##
###################################
def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective: train a DQN with sampled hyperparameters,
    evaluate on the fixed 64 sequences, and return mean normalised performance.
    """
    # --- Sample hyperparameters ---
    learning_rate = trial.suggest_float('learning_rate', 5e-4, 5e-3, log=True)
    discount_factor = trial.suggest_float('discount_factor', 0.85, 0.95)
    epsilon_initial = trial.suggest_float('epsilon_initial', 0.50, 0.90)
    epsilon_min = trial.suggest_float('epsilon_min', 0.02, 0.10)
    num_episodes = trial.suggest_int('num_episodes', 50000, 1000000, step=50000)
    hidden_dim = trial.suggest_int('hidden_dim', 32, 64, step=32)
    n_hidden_layers = trial.suggest_int('n_hidden_layers', 1, 2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64])
    replay_buffer_size = trial.suggest_int('replay_buffer_size', 20000, 50000, step=10000)
    target_sync_freq = trial.suggest_int('target_sync_freq', 500, 1500, step=500)
    train_freq = trial.suggest_categorical('train_freq', [2, 4])

    hidden_units = [hidden_dim] * n_hidden_layers

    # --- Train ---
    env = Pandemic()
    env.number_cities_prob = _number_cities_prob  # type: ignore[assignment]
    env.severity_prob = _severity_prob  # type: ignore[assignment]
    env.verbose = False

    rewards, model, _ = DQNTraining(
        env, learning_rate, discount_factor,
        epsilon_initial, epsilon_min, num_episodes,
        hidden_units=hidden_units,
        batch_size=batch_size,
        replay_buffer_size=replay_buffer_size,
        target_sync_freq=target_sync_freq,
        train_freq=train_freq,
        seed=SEED,
        compute_confidence=True,
    )

    # --- Evaluate on fixed sequences ---
    env_eval = Pandemic()
    env_eval.verbose = False

    max_res = env_eval.available_resources_states - 1
    max_tri = env_eval.trial_no_states - 1
    max_sev = env_eval.severity_states - 1

    def qf(_env, state, _seqid):
        s_norm = normalize_state(state, max_res, max_tri, max_sev)
        q_vals = model(s_norm[numpy.newaxis, :], training=False)[0].numpy()
        # Mask infeasible actions (consistent with dqn_agent_meta_cognitive)
        o = numpy.arange(len(q_vals), dtype=numpy.float32)
        q_vals[o > state[0]] = 0.00001
        return int(numpy.argmax(q_vals))

    _, perfs, _ = run_experiment(env_eval, qf, False, _trials_per_sequence, _sevs)
    mean_perf = float(numpy.mean(perfs))

    # Store extra info for later analysis
    trial.set_user_attr('mean_perf', mean_perf)
    trial.set_user_attr('std_perf', float(numpy.std(perfs)))
    trial.set_user_attr('min_perf', float(numpy.min(perfs)))
    trial.set_user_attr('max_perf', float(numpy.max(perfs)))

    # Preserve the best model weights to avoid lossy retraining at the end
    global _best_artifacts
    if mean_perf > _best_artifacts['value']:
        _best_artifacts['weights'] = model.get_weights()
        _best_artifacts['hidden_units'] = hidden_units
        _best_artifacts['rewards'] = list(rewards)
        _best_artifacts['value'] = mean_perf

    return mean_perf


###################################
##        Reporting              ##
###################################
def _save_report(study, opt_dir, opt_date, best_model, best_rewards):
    """Generate and save optimization results report and visualizations.

    Trial numbers are converted to 1-based indexing (trial.number + 1)
    so they match the trial_id column in the Optuna SQLite database.
    """

    best = study.best_trial

    # --- Text report ---
    report_file = os.path.join(opt_dir, f'optimization_results_{opt_date}.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("BAYESIAN OPTIMIZATION RESULTS — DQN HYPERPARAMETERS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Date:              {opt_date}\n")
        f.write(f"Total trials:      {len(study.trials)}\n")
        f.write(f"Best trial:        #{best.number + 1}\n")
        f.write(f"Best mean perf:    {best.value:.6f}\n\n")

        f.write("BEST HYPERPARAMETERS\n")
        f.write("-" * 80 + "\n")
        for name, val in best.params.items():
            f.write(f"  {name:<25s} = {val}\n")
        f.write("\n")

        f.write("BEST TRIAL STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Mean performance:   {best.user_attrs['mean_perf']:.6f}\n")
        f.write(f"  Std  performance:   {best.user_attrs['std_perf']:.6f}\n")
        f.write(f"  Min  performance:   {best.user_attrs['min_perf']:.6f}\n")
        f.write(f"  Max  performance:   {best.user_attrs['max_perf']:.6f}\n\n")

        f.write("ALL TRIALS\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"{'#':>4s}  {'mean_perf':>10s}  {'lr':>10s}  "
            f"{'gamma':>8s}  {'eps0':>6s}  {'eps_min':>7s}  "
            f"{'episodes':>8s}  {'h_dim':>5s}  {'layers':>6s}\n"
        )
        f.write("-" * 80 + "\n")
        for t in sorted(study.trials, key=lambda t: t.value if t.value is not None else -1, reverse=True):
            if t.value is None:
                continue
            p = t.params
            f.write(
                f"{t.number + 1:4d}  {t.value:10.6f}  "
                f"{p['learning_rate']:10.6f}  {p['discount_factor']:8.4f}  "
                f"{p['epsilon_initial']:6.3f}  {p['epsilon_min']:7.4f}  "
                f"{p['num_episodes']:8d}  {p['hidden_dim']:5d}  "
                f"{p['n_hidden_layers']:6d}\n"
            )

    success(f"Report saved: optimization_results_{opt_date}.txt")

    # --- Convergence plot ---
    try:
        plt.style.use('ggplot')
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(12, 6))
    trial_numbers = [t.number + 1 for t in study.trials if t.value is not None]
    trial_values = [t.value for t in study.trials if t.value is not None]

    # Running best
    running_best: list[float] = []
    current_best = -1.0
    for v in trial_values:
        current_best = max(current_best, v)
        running_best.append(current_best)

    ax.scatter(trial_numbers, trial_values, color='#1f77b4', s=50, alpha=0.6,
               edgecolors='navy', linewidth=0.5, label='Trial performance', zorder=3)
    ax.plot(trial_numbers, running_best, color='#d62728', linewidth=2.5,
            label='Best so far', zorder=4)
    ax.set_xlabel('Trial number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean normalised performance', fontsize=12, fontweight='bold')
    ax.set_title('Bayesian Optimisation: Convergence (DQN)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(opt_dir, f'optimization_history_{opt_date}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    list_item(f"Saved: optimization_history_{opt_date}.png")

    # --- Hyperparameter importance ---
    try:
        importances = optuna.importance.get_param_importances(study)
        names = list(importances.keys())
        values = list(importances.values())

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(names[::-1], values[::-1], color='#2ca02c', edgecolor='darkgreen', linewidth=0.5)
        ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
        ax.set_title('DQN Hyperparameter Importance', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        fig.savefig(os.path.join(opt_dir, f'hyperparameter_importances_{opt_date}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        list_item(f"Saved: hyperparameter_importances_{opt_date}.png")
    except Exception as e:
        info(f"Could not compute importances: {e}")

    # --- Save best model and rewards ---
    best_model.save(os.path.join(opt_dir, f'dqn_best_{opt_date}.keras'))
    numpy.save(os.path.join(opt_dir, f'rewards_best_{opt_date}.npy'), best_rewards)
    success(f"Best model saved: dqn_best_{opt_date}.keras")
    success(f"Best rewards saved: rewards_best_{opt_date}.npy")


###################################
##             Main              ##
###################################
def main():
    """Run Bayesian optimisation of DQN hyperparameters using Optuna."""

    header("BAYESIAN OPTIMISATION — DQN HYPERPARAMETERS", width=80)

    # Parse arguments: [n_trials] [--resume YYYY-MM-DD]
    n_trials = 30
    opt_date = datetime.now().strftime("%Y-%m-%d")

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--resume' and i + 1 < len(args):
            opt_date = args[i + 1]
            i += 2
        else:
            try:
                n_trials = int(args[i])
            except ValueError:
                pass
            i += 1

    opt_dir = os.path.join(INPUTS_PATH, f'{opt_date}_BAYESIAN_OPT')
    os.makedirs(opt_dir, exist_ok=True)

    info(f"Output directory: {opt_dir}")
    info(f"Target number of trials: {n_trials}")
    print()

    # --- Load data ---
    section("Loading Evaluation Data", width=80)
    _load_evaluation_data()
    assert _trials_per_sequence is not None and _sevs is not None
    assert _number_cities_prob is not None and _severity_prob is not None
    list_item(f"Sequence lengths shape: {_trials_per_sequence.shape}")
    list_item(f"Sequences loaded: {len(_sevs)}")
    print()

    # --- Run optimisation ---
    section("Running Bayesian Optimisation", width=80)
    info("Search space (tightened for CPU):")
    list_item("learning_rate        ∈ [5e-4, 5e-3]   (log scale)")
    list_item("discount_factor      ∈ [0.85, 0.95]")
    list_item("epsilon_initial      ∈ [0.50, 0.90]")
    list_item("epsilon_min          ∈ [0.02, 0.10]")
    list_item("num_episodes         ∈ [50000, 1000000]  (step=50000)")
    list_item("hidden_dim           ∈ [32, 64]  (step=32)")
    list_item("n_hidden_layers      ∈ {1, 2}")
    list_item("batch_size           ∈ {32, 64}")
    list_item("replay_buffer_size   ∈ [20000, 50000]  (step=10000)")
    list_item("target_sync_freq     ∈ [500, 1500]  (step=500)")
    list_item("train_freq           ∈ {2, 4}")
    print()

    # Suppress Optuna's verbose default logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Use SQLite storage so the study survives machine suspensions/crashes
    db_path = os.path.join(opt_dir, f'optuna_study_{opt_date}.db')
    storage = f'sqlite:///{db_path}'

    study = optuna.create_study(
        direction='maximize',
        study_name=f'dqn_opt_{opt_date}',
        sampler=optuna.samplers.TPESampler(seed=42),
        storage=storage,
        load_if_exists=True,   # Resume from previous run if DB exists
    )

    # Warm-start: enqueue the known-good defaults from train_dqn.py so
    # that TPE has a strong baseline from trial #1.  If the study already
    # has completed trials (resume) this is harmless — Optuna deduplicates.
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if completed == 0:
        study.enqueue_trial({
            'learning_rate': 1e-3,
            'discount_factor': 0.865,
            'epsilon_initial': 0.679,
            'epsilon_min': 0.085,
            'num_episodes': 100000,
            'hidden_dim': 64,
            'n_hidden_layers': 2,
            'batch_size': 32,
            'replay_buffer_size': 50000,
            'target_sync_freq': 1000,
            'train_freq': 4,
        })
        info("Warm-start: known-good defaults enqueued as first trial")

    # Calculate how many trials remain (allows seamless resume)
    remaining = max(0, n_trials - completed)
    if completed > 0:
        info(f"Resuming: {completed} trials already completed, {remaining} remaining")
    else:
        info(f"Starting fresh: {n_trials} trials to run")

    t_start = time.time()

    # Callback to print progress
    def _progress_callback(study, trial):
        done = len([t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE])
        elapsed = time.time() - t_start
        best_val = study.best_value
        print(
            f"  Trial {done:3d}/{n_trials}  |  "
            f"value={trial.value:.4f}  |  best={best_val:.4f}  |  "
            f"elapsed={elapsed:.0f}s"
        )
        # Notificar cada 10 trials completados
        if done > 0 and done % 10 == 0:
            notify(
                f"[{_PKG_NAME}] {done}/{n_trials} trials",
                f"Se completaron {done} de {n_trials} trials.\n"
                f"Mejor valor hasta ahora: {best_val:.6f}\n"
                f"Último trial: value={trial.value:.4f}\n"
                f"Tiempo transcurrido: {elapsed:.0f}s ({elapsed / 60:.1f} min)",
                tags="chart_with_upwards_trend"
            )

    if remaining > 0:
        _prev_err = numpy.seterr(under='ignore')
        try:
            study.optimize(objective, n_trials=remaining, callbacks=[_progress_callback])
        finally:
            numpy.seterr(**_prev_err)
    else:
        info("All trials already completed. Generating reports from stored results.")

    elapsed_total = time.time() - t_start
    total_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print()
    success(f"Optimisation finished in {elapsed_total:.1f}s ({elapsed_total / 60:.1f} min)")
    info(f"Total completed trials: {total_completed}")
    print()

    # --- Best trial summary ---
    best = study.best_trial
    section("Best Hyperparameters Found", width=80)
    for name, val in best.params.items():
        list_item(f"{name:<25s} = {val}")
    info(f"Mean normalised performance: {best.value:.6f}")
    print()

    # --- Use best model from optimization, or retrain only if resuming ---
    section("Best Model", width=80)

    bp = best.params
    best_hidden = [bp['hidden_dim']] * bp['n_hidden_layers']

    if _best_artifacts['weights'] is not None and _best_artifacts['value'] >= best.value:
        # Reconstruct model and load cached weights
        best_model = build_q_network(3, 11, _best_artifacts['hidden_units'])
        best_model(tf.zeros((1, 3)))  # build
        best_model.set_weights(_best_artifacts['weights'])
        best_rewards = numpy.array(_best_artifacts['rewards'])
        success("Using model from best optimization trial (no retraining needed)")
    else:
        info("Retraining with best hyperparameters (resumed study, original weights not in memory)...")
        env_final = Pandemic()
        env_final.number_cities_prob = _number_cities_prob  # type: ignore[assignment]
        env_final.severity_prob = _severity_prob  # type: ignore[assignment]
        env_final.verbose = False

        best_rewards_list, best_model, _ = DQNTraining(
            env_final,
            bp['learning_rate'],
            bp['discount_factor'],
            bp['epsilon_initial'],
            bp['epsilon_min'],
            bp['num_episodes'],
            hidden_units=best_hidden,
            batch_size=bp['batch_size'],
            replay_buffer_size=bp['replay_buffer_size'],
            target_sync_freq=bp['target_sync_freq'],
            train_freq=bp['train_freq'],
            seed=SEED,
            compute_confidence=True,
        )
        best_rewards = numpy.array(best_rewards_list)
        success(f"Retrained model (deterministic — seed = {SEED})")

    list_item(f"Model parameters: {best_model.count_params()}")
    print()

    # --- Save everything ---
    section("Saving Results", width=80)
    _save_report(study, opt_dir, opt_date, best_model, best_rewards)

    print()
    section("Optimisation Complete", width=80)
    success("All outputs saved!")
    info(f"Output directory: {opt_dir}")
    print()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        import traceback
        notify(
            f"[{_PKG_NAME}] ERROR en optimización",
            f"Se produjo un error durante la optimización:\n\n"
            f"{traceback.format_exc()}",
            priority="urgent", tags="rotating_light"
        )
        raise
