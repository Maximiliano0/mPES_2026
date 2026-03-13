"""
pes_dqn - Deep Q-Network (DQN) model components.

Provides the neural-network building blocks used by the DQN training loop
in :pymod:`pandemic`:

- **ReplayBuffer**:  Fixed-capacity circular buffer backed by pre-allocated
  NumPy arrays.  Stores experience tuples
  ``(state, action, reward, next_state, done)`` and provides uniform random
  mini-batch sampling via ``numpy.random.randint`` + advanced indexing —
  significantly faster than a ``deque``-based buffer on CPU.
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

CPU Optimisations
-----------------
- TensorFlow intra/inter-op thread pools are configured at import time
  to match the host CPU (defaults to auto-detect / 2 inter-op threads).
- The replay buffer uses contiguous NumPy arrays to avoid Python-level
  iteration when sampling mini-batches.
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
#  CPU threading optimisation (skipped when GPU is available)
# ---------------------------------------------------------------------------
#  On CPUs with few cores (e.g. Intel i3-6006U, 4 threads) explicitly
#  tuning the thread pools improves throughput for the small DQN model.
#  Must be called *before* any TF operation creates the thread pool.
if not tf.config.list_physical_devices('GPU'):
    tf.config.threading.set_intra_op_parallelism_threads(0)   # 0 = auto-detect
    tf.config.threading.set_inter_op_parallelism_threads(2)


# ---------------------------------------------------------------------------
#  Replay Buffer  (NumPy-backed, cache-friendly)
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """Fixed-capacity circular experience-replay buffer backed by pre-allocated NumPy arrays.

    Stores ``(state, action, reward, next_state, done)`` transitions in
    contiguous NumPy arrays.  Sampling uses ``numpy.random.randint`` +
    advanced indexing, which is *orders of magnitude* faster than
    ``random.sample`` on a Python ``deque`` — particularly for large
    buffers (50 000+ entries).

    Parameters
    ----------
    capacity : int
        Maximum number of transitions stored.  When full, the oldest
        transition is silently overwritten.
    state_dim : int, optional
        Dimensionality of each state vector (default ``3``).
    """

    def __init__(self, capacity: int, state_dim: int = 3) -> None:
        self.capacity = capacity
        self.size = 0
        self._idx = 0
        self._states = numpy.zeros((capacity, state_dim), dtype=numpy.float32)
        self._actions = numpy.zeros(capacity, dtype=numpy.int32)
        self._rewards = numpy.zeros(capacity, dtype=numpy.float32)
        self._next_states = numpy.zeros((capacity, state_dim), dtype=numpy.float32)
        self._dones = numpy.zeros(capacity, dtype=numpy.float32)

    # ------------------------------------------------------------------
    def push(self, state: numpy.ndarray, action: int, reward: float,
             next_state: numpy.ndarray, done: bool) -> None:
        """Append one transition, overwriting the oldest if at capacity.

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
        i = self._idx
        self._states[i] = state
        self._actions[i] = action
        self._rewards[i] = reward
        self._next_states[i] = next_state
        self._dones[i] = float(done)
        self._idx = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # ------------------------------------------------------------------
    def sample(self, batch_size: int
               ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray,
                           numpy.ndarray, numpy.ndarray]:
        """Return a uniformly sampled mini-batch of transitions.

        Uses ``numpy.random.randint`` with advanced indexing for
        cache-friendly, vectorised sampling — much faster than
        ``random.sample`` on a Python ``deque``.

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
        indices = numpy.random.randint(0, self.size, size=batch_size)
        return (self._states[indices],
                self._actions[indices],
                self._rewards[indices],
                self._next_states[indices],
                self._dones[indices])

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """Return the current number of stored transitions."""
        return self.size


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
        Keras ``Sequential`` model (uncompiled; call ``.compile()`` or use
        a custom training loop with ``tf.GradientTape``).
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
def train_step(online_model: tf.keras.Model,
               target_model: tf.keras.Model,
               optimizer: tf.keras.optimizers.Optimizer,
               states: tf.Tensor,
               actions: tf.Tensor,
               rewards: tf.Tensor,
               next_states: tf.Tensor,
               dones: tf.Tensor,
               discount: tf.Tensor) -> tf.Tensor:
    """Execute a single gradient-descent step on a mini-batch.

    Uses the Huber loss (smooth L1) between the online Q-values at the
    selected actions and the TD targets computed from the *target* network.

    This function is **not** decorated with ``@tf.function`` at the
    module level because Optuna calls ``DQNTraining`` multiple times
    with different model/optimizer instances.  Each trial wraps this
    function via ``tf.function`` locally so the traced graph is
    scoped to a single optimisation trial.

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
    discount : tf.Tensor
        Scalar ``tf.Tensor`` with discount factor γ.  Passed as a
        tensor (not a Python float) to prevent ``@tf.function``
        retracing when the value changes across Optuna trials.

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
