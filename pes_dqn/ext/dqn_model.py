"""
pes_dqn - Deep Q-Network (DQN) model components.

Provides the neural-network building blocks used by the DQN training loop
in :pymod:`pandemic`:

- **ReplayBuffer**:  Fixed-capacity circular buffer that stores experience
  tuples ``(state, action, reward, next_state, done)`` and provides uniform
  random mini-batch sampling for off-policy learning.
- **build_q_network**:  Constructs a fully-connected Keras ``Sequential``
  model that maps a normalised state vector to Q-values for every action.
- **normalize_state**:  Scales raw integer state components to the [0, 1]
  range expected by the network.
- **train_step**:  Single gradient-descent update using a Huber-loss
  objective on a sampled mini-batch (compiled with ``@tf.function``).

Architecture
------------
::

    Input  (3)  →  Dense(64, ReLU)  →  Dense(64, ReLU)  →  Dense(11, linear)
                   (hidden_units[0])    (hidden_units[1])    (action_dim)

The default hidden-layer sizes are configurable via ``config/CONFIG.py``.
"""

##########################
##  Imports externos    ##
##########################
import random
from collections import deque
from typing import List, Tuple

import numpy
import tensorflow as tf

##########################
##  Imports internos    ##
##########################


# ---------------------------------------------------------------------------
#  Replay Buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """Fixed-capacity circular experience-replay buffer.

    Stores ``(state, action, reward, next_state, done)`` tuples and provides
    uniform-random mini-batch sampling for off-policy training.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions stored.  When full, the oldest
        transition is silently discarded.
    """

    def __init__(self, capacity: int) -> None:
        self.buffer: deque[Tuple[numpy.ndarray, int, float,
                                 numpy.ndarray, bool]] = deque(maxlen=capacity)

    # ------------------------------------------------------------------
    def push(self, state: numpy.ndarray, action: int, reward: float,
             next_state: numpy.ndarray, done: bool) -> None:
        """Append one transition, evicting the oldest if at capacity.

        Parameters
        ----------
        state : ndarray
            Normalised state vector, shape ``(state_dim,)``.
        action : int
            Action index taken.
        reward : float
            Immediate scalar reward.
        next_state : ndarray
            Normalised successor state, shape ``(state_dim,)``.
        done : bool
            Whether the episode ended after this transition.
        """
        self.buffer.append((state, action, reward, next_state, done))

    # ------------------------------------------------------------------
    def sample(self, batch_size: int
               ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray,
                           numpy.ndarray, numpy.ndarray]:
        """Return a uniformly sampled mini-batch of transitions.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.

        Returns
        -------
        states : ndarray, shape ``(batch_size, state_dim)``
        actions : ndarray, shape ``(batch_size,)``
        rewards : ndarray, shape ``(batch_size,)``
        next_states : ndarray, shape ``(batch_size, state_dim)``
        dones : ndarray, shape ``(batch_size,)``
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (numpy.array(states, dtype=numpy.float32),
                numpy.array(actions, dtype=numpy.int32),
                numpy.array(rewards, dtype=numpy.float32),
                numpy.array(next_states, dtype=numpy.float32),
                numpy.array(dones, dtype=numpy.float32))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """Return the current number of stored transitions."""
        return len(self.buffer)


# ---------------------------------------------------------------------------
#  Network builder
# ---------------------------------------------------------------------------
def build_q_network(state_dim: int, action_dim: int,
                    hidden_units: List[int]) -> tf.keras.Model:
    """Build a fully-connected Q-network.

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
        Compiled-ready Keras ``Sequential`` model.
    """
    model = tf.keras.Sequential(name="DQN")
    model.add(tf.keras.layers.Input(shape=(state_dim,)))
    for idx, units in enumerate(hidden_units):
        model.add(tf.keras.layers.Dense(
            units, activation="relu", name=f"hidden_{idx}"))
    model.add(tf.keras.layers.Dense(
        action_dim, activation="linear", name="q_values"))
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
#  Compiled training step
# ---------------------------------------------------------------------------
@tf.function
def train_step(online_model: tf.keras.Model,
               target_model: tf.keras.Model,
               optimizer: tf.keras.optimizers.Optimizer,
               states: tf.Tensor,
               actions: tf.Tensor,
               rewards: tf.Tensor,
               next_states: tf.Tensor,
               dones: tf.Tensor,
               discount: float) -> tf.Tensor:
    """Execute a single gradient-descent step on a mini-batch.

    Uses the Huber loss (smooth L1) between the online Q-values at the
    selected actions and the TD targets computed from the *target* network.

    Parameters
    ----------
    online_model : tf.keras.Model
        Network whose weights are updated.
    target_model : tf.keras.Model
        Slowly-updated copy used for target computation.
    optimizer : tf.keras.optimizers.Optimizer
        Optimiser instance (e.g. Adam).
    states : tf.Tensor, shape ``(B, state_dim)``
    actions : tf.Tensor, shape ``(B,)``   int32
    rewards : tf.Tensor, shape ``(B,)``
    next_states : tf.Tensor, shape ``(B, state_dim)``
    dones : tf.Tensor, shape ``(B,)``
    discount : float
        Discount factor γ.

    Returns
    -------
    tf.Tensor
        Scalar Huber loss for logging.
    """
    # TD targets  y = r + γ · max_a' Q_target(s', a') · (1 - done)
    next_q = target_model(next_states, training=False)
    max_next_q = tf.reduce_max(next_q, axis=1)
    targets = rewards + discount * max_next_q * (1.0 - dones)

    with tf.GradientTape() as tape:
        q_all = online_model(states, training=True)
        action_mask = tf.one_hot(actions, depth=tf.shape(q_all)[1])
        q_selected = tf.reduce_sum(q_all * action_mask, axis=1)
        loss = tf.keras.losses.huber(targets, q_selected)
        loss = tf.reduce_mean(loss)

    grads = tape.gradient(loss, online_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, online_model.trainable_variables))
    return loss


# ---------------------------------------------------------------------------
#  Target-network synchronisation
# ---------------------------------------------------------------------------
def sync_target_network(online_model: tf.keras.Model,
                        target_model: tf.keras.Model) -> None:
    """Copy online-network weights to the target network.

    Parameters
    ----------
    online_model : tf.keras.Model
        Source of the weight values.
    target_model : tf.keras.Model
        Destination whose weights are overwritten.
    """
    target_model.set_weights(online_model.get_weights())
