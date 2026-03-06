"""
pes_ac - Advantage Actor-Critic (A2C) model components.

Provides the neural-network building blocks used by the A2C training loop
in :pymod:`pandemic`:

- **build_actor**:  Constructs a Keras model that maps a normalised state
  vector to an action probability distribution π(a|s) via softmax.
- **build_critic**:  Constructs a Keras model that maps a normalised state
  vector to a scalar state-value estimate V(s).
- **normalize_state**:  Scales raw integer state components to the [0, 1]
  range expected by the networks.
- **train_step_actor_critic**:  Single gradient-descent update for both
  Actor and Critic using the Advantage Actor-Critic objective.
  Not decorated with ``@tf.function`` at module level — each
  Optuna trial wraps it locally via ``tf.function`` to scope
  the traced graph per trial.

Architecture
------------
::

    Actor:   Input(3) → Dense(64, ReLU) → Dense(64, ReLU) → Dense(11, softmax)
    Critic:  Input(3) → Dense(64, ReLU) → Dense(64, ReLU) → Dense(1, linear)

The default hidden-layer sizes are configurable via ``config/CONFIG.py``.

CPU Optimisations
-----------------
- TensorFlow intra/inter-op thread pools are configured at import time
  to match the host CPU (defaults to auto-detect / 2 inter-op threads).
"""

##########################
##  Imports externos    ##
##########################
from typing import List, Tuple

import numpy
import tensorflow as tf

##########################
##  Imports internos    ##
##########################


# ---------------------------------------------------------------------------
#  CPU threading optimisation
# ---------------------------------------------------------------------------
tf.config.threading.set_intra_op_parallelism_threads(0)   # 0 = auto-detect
tf.config.threading.set_inter_op_parallelism_threads(2)


# ---------------------------------------------------------------------------
#  Network builders
# ---------------------------------------------------------------------------
def build_actor(state_dim: int, action_dim: int,
                hidden_units: List[int]) -> tf.keras.Model:
    """Build a fully-connected Actor (policy) network with softmax output.

    The Actor maps a normalised state vector to a probability distribution
    over discrete actions.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the (normalised) state vector.
    action_dim : int
        Number of discrete actions (network output width).
    hidden_units : list of int
        Widths of the hidden dense layers (e.g. ``[64, 64]``).

    Returns
    -------
    tf.keras.Model
        Keras ``Sequential`` model with softmax output.
    """
    model = tf.keras.Sequential(name="Actor")
    model.add(tf.keras.layers.Input(shape=(state_dim,)))
    for idx, units in enumerate(hidden_units):
        model.add(tf.keras.layers.Dense(
            units, activation="relu", name=f"actor_hidden_{idx}"))
    model.add(tf.keras.layers.Dense(
        action_dim, activation="softmax", name="policy"))
    return model


def build_critic(state_dim: int,
                 hidden_units: List[int]) -> tf.keras.Model:
    """Build a fully-connected Critic (value) network with linear output.

    The Critic maps a normalised state vector to a scalar state-value
    estimate V(s).

    Parameters
    ----------
    state_dim : int
        Dimensionality of the (normalised) state vector.
    hidden_units : list of int
        Widths of the hidden dense layers (e.g. ``[64, 64]``).

    Returns
    -------
    tf.keras.Model
        Keras ``Sequential`` model with single linear output.
    """
    model = tf.keras.Sequential(name="Critic")
    model.add(tf.keras.layers.Input(shape=(state_dim,)))
    for idx, units in enumerate(hidden_units):
        model.add(tf.keras.layers.Dense(
            units, activation="relu", name=f"critic_hidden_{idx}"))
    model.add(tf.keras.layers.Dense(1, activation="linear", name="value"))
    return model


# ---------------------------------------------------------------------------
#  State normalisation
# ---------------------------------------------------------------------------
def normalize_state(state, max_resources: int,
                    max_trials: int, max_severity: int) -> numpy.ndarray:
    """Scale raw integer state components to [0, 1].

    Parameters
    ----------
    state : array-like
        Raw state ``[resources_left, trial_no, severity]``.
    max_resources : int
        Upper bound of the resources dimension.
    max_trials : int
        Upper bound of the trial-number dimension.
    max_severity : int
        Upper bound of the severity dimension.

    Returns
    -------
    ndarray, shape ``(3,)``, dtype ``float32``
    """
    return numpy.array([
        state[0] / max(max_resources, 1),
        state[1] / max(max_trials, 1),
        state[2] / max(max_severity, 1),
    ], dtype=numpy.float32)


# ---------------------------------------------------------------------------
#  Training step (Actor-Critic)
# ---------------------------------------------------------------------------
def train_step_actor_critic(
    actor: tf.keras.Model,
    critic: tf.keras.Model,
    actor_optimizer: tf.keras.optimizers.Optimizer,
    critic_optimizer: tf.keras.optimizers.Optimizer,
    states: tf.Tensor,
    actions: tf.Tensor,
    rewards: tf.Tensor,
    next_states: tf.Tensor,
    dones: tf.Tensor,
    discount: tf.Tensor,
    entropy_coeff: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Execute a single gradient-descent step for both Actor and Critic.

    Uses the Advantage Actor-Critic (A2C) objective:

    - **Critic loss**: MSE between V(s) and the TD target  r + γ·V(s')·(1-done).
    - **Actor loss**: -log π(a|s) · A(s,a), where A(s,a) = r + γ·V(s') - V(s)
      is the advantage, plus an entropy bonus to encourage exploration.

    This function is **not** decorated with ``@tf.function`` at the
    module level because Optuna calls ``A2CTraining`` multiple times
    with different model/optimiser instances.  Each trial wraps this
    function via ``tf.function`` locally so the traced graph is
    scoped to a single optimisation trial and no ``tf.Variable``
    leaks between trials.

    Parameters
    ----------
    actor : tf.keras.Model
        Policy network whose weights are updated.
    critic : tf.keras.Model
        Value network whose weights are updated.
    actor_optimizer : tf.keras.optimizers.Optimizer
        Optimiser for the Actor (e.g. Adam).
    critic_optimizer : tf.keras.optimizers.Optimizer
        Optimiser for the Critic (e.g. Adam).
    states : tf.Tensor, shape ``(B, state_dim)``
    actions : tf.Tensor, shape ``(B,)``   int32
    rewards : tf.Tensor, shape ``(B,)``
    next_states : tf.Tensor, shape ``(B, state_dim)``
    dones : tf.Tensor, shape ``(B,)``
    discount : tf.Tensor
        Scalar ``tf.Tensor`` with discount factor γ.  Passed as a
        tensor (not a Python float) to avoid ``@tf.function``
        retracing when the value changes across Optuna trials.
    entropy_coeff : tf.Tensor
        Scalar ``tf.Tensor`` with the entropy-bonus weight.  Passed
        as a tensor for the same reason as *discount*.

    Returns
    -------
    actor_loss : tf.Tensor
        Scalar Actor loss (policy gradient + entropy).
    critic_loss : tf.Tensor
        Scalar Critic loss (MSE on TD error).
    entropy : tf.Tensor
        Scalar mean entropy of the policy distribution.
    """
    # ---------- Critic update ----------
    with tf.GradientTape() as critic_tape:
        values = tf.squeeze(critic(states, training=True), axis=1)
        next_values = tf.squeeze(critic(next_states, training=False), axis=1)
        td_targets = rewards + discount * next_values * (1.0 - dones)
        critic_loss = tf.reduce_mean(tf.square(td_targets - values))

    critic_grads = critic_tape.gradient(critic_loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))

    # ---------- Advantage ----------
    # Recompute values after Critic update for a cleaner advantage signal
    values_updated = tf.squeeze(critic(states, training=False), axis=1)
    next_values_updated = tf.squeeze(critic(next_states, training=False), axis=1)
    advantages = rewards + discount * next_values_updated * (1.0 - dones) - values_updated

    # ---------- Actor update ----------
    with tf.GradientTape() as actor_tape:
        probs = actor(states, training=True)
        # Clip probabilities to avoid log(0)
        probs_clipped = tf.clip_by_value(probs, 1e-8, 1.0)
        action_mask = tf.one_hot(actions, depth=tf.shape(probs)[1])
        log_probs = tf.reduce_sum(tf.math.log(probs_clipped) * action_mask, axis=1)

        # Entropy bonus: H(π) = -Σ π(a) log π(a)
        entropy = -tf.reduce_sum(probs_clipped * tf.math.log(probs_clipped), axis=1)
        mean_entropy = tf.reduce_mean(entropy)

        # Policy gradient loss: -E[ log π(a|s) · A(s,a) ] - c_ent · H(π)
        actor_loss = -tf.reduce_mean(log_probs * tf.stop_gradient(advantages))
        actor_loss = actor_loss - entropy_coeff * mean_entropy

    actor_grads = actor_tape.gradient(actor_loss, actor.trainable_variables)
    actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))

    return actor_loss, critic_loss, mean_entropy
