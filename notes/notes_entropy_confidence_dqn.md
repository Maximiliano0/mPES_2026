# Entropy-Derived Confidence for DQN: Does It Still Make Sense?

> Date: 2026-03-02

## Question

Now that Q is not a table, does it make sense to use an entropy-derived
metric for `dqn_agent_meta_cognitive`?

## Answer

**Yes, it still makes sense**, but with a nuance worth understanding.

### Why it works

The entropy metric operates on **the Q-value vector output** (11 values, one
per action), not on how Q is stored.  Whether that vector comes from a table
lookup or a neural-network forward pass, the interpretation is the same:

- **Peaked distribution** (one Q-value much larger) → low entropy → high
  confidence → the agent clearly prefers one action.
- **Flat distribution** (similar Q-values) → high entropy → low confidence →
  the agent is indifferent.

This pattern is widely used in deep RL — entropy of action preferences is the
basis of SAC (Soft Actor-Critic), entropy bonuses in policy gradient, and
exploration diagnostics.

### The caveat: calibration

The one meaningful difference is **calibration quality**:

|                          | Tabular Q | DQN |
|--------------------------|-----------|-----|
| Unvisited state          | Q-values ≈ initialisation → naturally flat → high entropy (correctly signals "I don't know") | Network **extrapolates** → can output peaked Q-values for unseen states → false confidence |
| Frequently visited state | Q-values reflect accumulated experience → well-calibrated entropy | Q-values reflect generalisation → usually well-calibrated |

In `pes_dqn` this caveat is **minor** because the state space is small
(3,410 states) and training runs 100k+ episodes — the network sees virtually
every state many times, so its Q-value landscape is well-grounded.

### Bottom line

The entropy-derived confidence is a valid heuristic for DQN in this problem.
If you ever moved to a much larger state space (e.g., continuous states, image
observations), you'd want to consider alternatives like:

- **MC dropout uncertainty**
- **Ensemble disagreement**
- **Distributional Q-learning** (C51 / QR-DQN)

These give probabilistic uncertainty estimates rather than relying on Q-value
spread.  But for the pandemic scenario's discrete, fully-covered state space,
the current approach is adequate.
