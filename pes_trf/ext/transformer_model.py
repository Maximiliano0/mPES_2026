"""
pes_trf — Transformer Policy Network for the Pandemic Environment.

Implements a small causal Transformer encoder with actor-critic output heads
for decision-making in the Pandemic MDP.  The model processes the trajectory
of states within an episode (max 10 timesteps) and outputs an action
distribution (policy) and a scalar value estimate at each position.

Architecture
------------
::

    Input:  (batch, seq_len, 3)  — [resources_left, trial_no, severity]
      ↓ Normalise to [0, 1]
      ↓ Linear embedding → (batch, seq_len, d_model)
      ↓ + Learned positional encoding
      ↓ L × TransformerBlock (causal multi-head self-attention + FFN)
      ↓ Policy head → (batch, seq_len, 11)   action logits
      ↓ Value head  → (batch, seq_len, 1)    state-value estimate

Default Hyperparameters
-----------------------
    d_model      = 64
    n_heads      = 4
    n_layers     = 2
    d_ff         = 128
    max_seq_len  = 10
    n_actions    = 11
    dropout_rate = 0.1

Parameter Count
---------------
    ~77 000 trainable parameters (vs 37 510 Q-table cells in PES_QLv2).

References
----------
Vaswani, A. et al. (2017). "Attention Is All You Need." NeurIPS 2017.
Chen, L. et al. (2021). "Decision Transformer." NeurIPS 2021.
Schulman, J. et al. (2017). "Proximal Policy Optimization Algorithms."
"""

# ─────────────────────────────────────────────────────────────────
#  Imports
# ─────────────────────────────────────────────────────────────────
import json
import os
import tensorflow as tf
import keras


# ─────────────────────────────────────────────────────────────────
#  Layers
# ─────────────────────────────────────────────────────────────────

class CausalSelfAttention(keras.layers.Layer):
    """Multi-head causal (masked) self-attention.

    Each position can only attend to the current and all previous
    positions in the sequence, enforced by an upper-triangular mask.

    Parameters
    ----------
    d_model : int
        Total embedding dimension (split across heads).
    n_heads : int
        Number of parallel attention heads.
    dropout_rate : float
        Dropout probability applied to attention weights.
    """

    def __init__(self, d_model, n_heads, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.depth = d_model // n_heads
        self.wq = keras.layers.Dense(d_model)
        self.wk = keras.layers.Dense(d_model)
        self.wv = keras.layers.Dense(d_model)
        self.dense = keras.layers.Dense(d_model)
        self.dropout = keras.layers.Dropout(dropout_rate)

    # ── helpers ──────────────────────────────────────────────────

    def _split_heads(self, x, batch_size):
        """Reshape (batch, seq, d_model) → (batch, heads, seq, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.n_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    # ── forward ──────────────────────────────────────────────────

    def call(self, x, padding_mask=None, training=False):
        """Forward pass applying causal multi-head self-attention."""
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        q = self._split_heads(self.wq(x), batch_size)
        k = self._split_heads(self.wk(x), batch_size)
        v = self._split_heads(self.wv(x), batch_size)

        # Scaled dot-product attention
        scale = tf.math.sqrt(tf.cast(self.depth, tf.float32))
        attn_logits = tf.matmul(q, k, transpose_b=True) / scale

        # Causal mask — lower-triangular ones
        causal = tf.linalg.band_part(
            tf.ones((seq_len, seq_len), dtype=tf.float32), -1, 0
        )
        causal = tf.reshape(causal, (1, 1, seq_len, seq_len))
        attn_logits += (1.0 - causal) * (-1e9)

        # Optional padding mask  (batch, seq_len) → (batch, 1, 1, seq_len)
        if padding_mask is not None:
            pm = tf.cast(
                tf.reshape(padding_mask, (batch_size, 1, 1, -1)), tf.float32
            )
            attn_logits += (1.0 - pm) * (-1e9)

        attn_weights = tf.nn.softmax(attn_logits, axis=-1)
        attn_weights = self.dropout(attn_weights, training=training)

        out = tf.matmul(attn_weights, v)                        # (B, H, S, D)
        out = tf.transpose(out, perm=[0, 2, 1, 3])              # (B, S, H, D)
        out = tf.reshape(out, (batch_size, -1, self.d_model))    # (B, S, d_model)
        return self.dense(out)


class TransformerBlock(keras.layers.Layer):
    """Pre-norm Transformer encoder block: LN → Attn → LN → FFN.

    Uses GELU activation in the feed-forward sub-layer (Hendrycks &
    Gimpel, 2016) and pre-norm residual connections (GPT-2 style).

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    n_heads : int
        Number of attention heads.
    d_ff : int
        Hidden dimension of the feed-forward sub-layer.
    dropout_rate : float
        Dropout probability.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout_rate)
        self.ffn = keras.Sequential([
            keras.layers.Dense(d_ff, activation='gelu'),
            keras.layers.Dense(d_model),
        ])
        self.ln1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = keras.layers.Dropout(dropout_rate)
        self.drop2 = keras.layers.Dropout(dropout_rate)

    def call(self, x, padding_mask=None, training=False):
        """Forward pass: pre-norm attention + FFN with residual connections."""
        h = self.attn(self.ln1(x), padding_mask=padding_mask, training=training)
        x = x + self.drop1(h, training=training)
        h = self.ffn(self.ln2(x))
        x = x + self.drop2(h, training=training)
        return x


# ─────────────────────────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────────────────────────

class PandemicTransformer(keras.Model):
    """Transformer-based actor-critic for the Pandemic MDP.

    Processes a sequence of raw states ``[resources_left, trial_no, severity]``
    through a stack of causal Transformer blocks and produces:

    * **policy_logits** — un-normalised log-probabilities over 11 actions at
      each timestep.
    * **values** — scalar state-value estimates at each timestep.

    The state features are normalised internally to [0, 1] using the known
    environment bounds (30, 10, 9).

    Parameters
    ----------
    d_model : int
        Embedding / hidden dimension.  Default 64.
    n_heads : int
        Number of self-attention heads.  Default 4.
    n_layers : int
        Number of Transformer blocks.  Default 2.
    d_ff : int
        Feed-forward hidden dimension.  Default 128.
    max_seq_len : int
        Maximum episode length (padded).  Default 10.
    n_actions : int
        Size of the discrete action space.  Default 11.
    dropout_rate : float
        Dropout probability.  Default 0.1.
    """

    # Environment bounds for state normalisation
    STATE_MAX = tf.constant([30.0, 10.0, 9.0], dtype=tf.float32)

    def __init__(self, d_model=64, n_heads=4, n_layers=2, d_ff=128,
                 max_seq_len=10, n_actions=11, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_actions = n_actions
        self.max_seq_len = max_seq_len

        # Store constructor args for serialisation
        self._config = dict(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            d_ff=d_ff, max_seq_len=max_seq_len, n_actions=n_actions,
            dropout_rate=dropout_rate,
        )

        # --- input projection ------------------------------------------------
        self.state_proj = keras.layers.Dense(d_model, name='state_proj')

        # Learned positional encoding (max 10 positions)
        self.pos_embedding = self.add_weight(
            name='pos_embedding',
            shape=(1, max_seq_len, d_model),
            initializer='glorot_uniform',
            trainable=True,
        )

        self.input_dropout = keras.layers.Dropout(dropout_rate)

        # --- transformer stack ------------------------------------------------
        self.blocks = [
            TransformerBlock(d_model, n_heads, d_ff, dropout_rate,
                             name=f'block_{i}')
            for i in range(n_layers)
        ]
        self.final_ln = keras.layers.LayerNormalization(epsilon=1e-6)

        # --- output heads -----------------------------------------------------
        self.policy_head = keras.Sequential([
            keras.layers.Dense(d_model, activation='relu'),
            keras.layers.Dense(n_actions),
        ], name='policy_head')

        self.value_head = keras.Sequential([
            keras.layers.Dense(d_model, activation='relu'),
            keras.layers.Dense(1),
        ], name='value_head')

    # ── forward ──────────────────────────────────────────────────

    def call(self, states, padding_mask=None, training=False):
        """Forward pass.

        Parameters
        ----------
        states : tf.Tensor, shape (batch, seq_len, 3)
            Raw integer states: ``[resources_left, trial_no, severity]``.
        padding_mask : tf.Tensor, shape (batch, seq_len), optional
            1.0 for real tokens, 0.0 for padding.
        training : bool
            Enables dropout when ``True``.

        Returns
        -------
        policy_logits : tf.Tensor, shape (batch, seq_len, n_actions)
        values : tf.Tensor, shape (batch, seq_len)
        """
        # Normalise states to [0, 1]
        x = tf.cast(states, tf.float32) / self.STATE_MAX

        seq_len = tf.shape(x)[1]

        # Project to d_model and add positional encoding
        x = self.state_proj(x) + self.pos_embedding[:, :seq_len, :]
        x = self.input_dropout(x, training=training)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, padding_mask=padding_mask, training=training)

        x = self.final_ln(x)

        policy_logits = self.policy_head(x)
        values = tf.squeeze(self.value_head(x), axis=-1)

        return policy_logits, values

    # ── action selection helpers ─────────────────────────────────

    def get_action_and_value(self, states, padding_mask=None,
                             greedy=False, resources_left=None):
        """Select action at the **last** valid timestep.

        Parameters
        ----------
        states : tf.Tensor, shape (1, seq_len, 3)
            Single trajectory (un-batched is fine; batch dim = 1).
        padding_mask : tf.Tensor, shape (1, seq_len), optional
        greedy : bool
            ``True`` → argmax;  ``False`` → sample from π.
        resources_left : int or None
            Mask actions that exceed available resources.

        Returns
        -------
        action : int
        probs : np.ndarray, shape (n_actions,)
            Softmax probabilities (after masking).
        value : float
        log_prob : float
            log π(action | s) — useful for training.
        """
        policy_logits, values = self(states, padding_mask=padding_mask,
                                     training=False)

        logits = policy_logits[0, -1]                       # (n_actions,)

        # Mask infeasible actions
        if resources_left is not None:
            rl = int(resources_left)
            if rl < self.n_actions - 1:
                mask_vals = tf.concat([
                    tf.zeros(rl + 1, dtype=tf.float32),
                    tf.fill([self.n_actions - rl - 1], -1e9),
                ], axis=0)
                logits = logits + mask_vals

        probs = tf.nn.softmax(logits).numpy()

        if greedy:
            action = int(tf.argmax(logits).numpy())
        else:
            action = int(
                tf.random.categorical(
                    tf.expand_dims(logits, 0), 1
                )[0, 0].numpy()
            )

        log_prob = float(tf.math.log(probs[action] + 1e-10).numpy())
        value = float(values[0, -1].numpy())

        return action, probs, value, log_prob

    # ── persistence ──────────────────────────────────────────────

    def save_pretrained(self, directory):
        """Save model weights and architecture config to *directory*.

        Creates:
        * ``transformer_config.json``  — constructor kwargs
        * ``transformer_weights.weights.h5`` — Keras weights

        Parameters
        ----------
        directory : str
            Target directory (will be created if needed).
        """
        os.makedirs(directory, exist_ok=True)

        config_path = os.path.join(directory, 'transformer_config.json')
        with open(config_path, 'w') as f:
            json.dump(self._config, f, indent=2)

        weights_path = os.path.join(directory, 'transformer_weights.weights.h5')
        self.save_weights(weights_path)

    @classmethod
    def from_pretrained(cls, directory):
        """Load model from *directory* created by :meth:`save_pretrained`.

        Parameters
        ----------
        directory : str
            Directory containing ``transformer_config.json`` and
            ``transformer_weights.weights.h5``.

        Returns
        -------
        PandemicTransformer
        """
        config_path = os.path.join(directory, 'transformer_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        model = cls(**config)
        # Build by running a dummy forward pass
        model(tf.zeros((1, 1, 3), dtype=tf.float32))

        weights_path = os.path.join(directory, 'transformer_weights.weights.h5')
        model.load_weights(weights_path)
        return model


# ─────────────────────────────────────────────────────────────────
#  Agent wrapper  (trajectory management for inference)
# ─────────────────────────────────────────────────────────────────

class TransformerAgent:
    """High-level wrapper providing trajectory-aware action selection.

    Maintains an internal buffer of states within the current episode so
    that the Transformer receives full episode context at each decision.

    Usage with ``pandemic.run_experiment()``::

        agent = TransformerAgent(model, greedy=True)
        seqs, perfs, evs = run_experiment(
            env, agent.action_function, RandomSequences=False,
            trials_per_sequence=tps, sevs=sevs)

    Parameters
    ----------
    model : PandemicTransformer
        A trained (or randomly initialised) model.
    greedy : bool
        ``True`` for evaluation (argmax), ``False`` for training (sample).
    """

    def __init__(self, model, greedy=True):
        self.model = model
        self.greedy = greedy
        self._trajectory = []

    def reset(self):
        """Clear trajectory buffer for a new episode / sequence."""
        self._trajectory = []

    def act(self, state, resources_left=None):
        """Append *state* to trajectory and return an action.

        Parameters
        ----------
        state : list
            ``[resources_left, trial_no, severity]``.
        resources_left : int or None
            If ``None``, uses ``state[0]``.

        Returns
        -------
        action : int
        probs : np.ndarray
        value : float
        log_prob : float
        """
        self._trajectory.append([int(s) for s in state])

        states = tf.constant([self._trajectory], dtype=tf.float32)

        if resources_left is None:
            resources_left = int(state[0])

        action, probs, value, log_prob = self.model.get_action_and_value(
            states, greedy=self.greedy, resources_left=resources_left,
        )
        return action, probs, value, log_prob

    def action_function(self, _env, state, _seqid):
        """Interface compatible with ``pandemic.run_experiment()``.

        Resets trajectory when ``trial_no == 0`` (start of new sequence).
        """
        if state[1] == 0:
            self.reset()
        action, _, _, _ = self.act(state, resources_left=state[0])
        return action
