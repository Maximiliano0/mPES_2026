'''
pes_actor_critic - Pandemic Experiment Scenario: Gym Environment and A2C Training

Provides the core simulation components:

- **Pandemic** (gym.Env):  OpenAI Gym environment that models a pandemic
  resource-allocation problem.  State = (resources_left, trial_no, severity);
  action = resources to allocate (0-10).
- **ac_agent_meta_cognitive**:  Actor-policy-based meta-cognitive function that
  computes confidence and simulated response times from the Actor's π(a|s)
  probability distribution (a theoretically grounded entropy measure).
- **run_experiment**:  Runs multiple sequences through the environment using
  any action-selection function and collects performance metrics.
- **A2CTraining**:  Advantage Actor-Critic training loop with on-policy
  updates, entropy regularisation, epsilon-greedy exploration overlay,
  and optional seed for reproducibility.

Network architecture
--------------------
::

    Actor:   Input(3)  →  Dense(64, ReLU)  →  Dense(64, ReLU)  →  Dense(11, softmax)
    Critic:  Input(3)  →  Dense(64, ReLU)  →  Dense(64, ReLU)  →  Dense(1, linear)

State normalisation: [resources/30, trial/10, severity/9] → [0, 1]³.
'''

##########################
##  Imports externos    ##
##########################
import numpy
import random
from gym import Env, spaces
import tensorflow as tf

##########################
##  Imports internos    ##
##########################
from .. import AVAILABLE_RESOURCES_PER_SEQUENCE
from .. import MAX_SEVERITY
from .. import MAX_ALLOCATABLE_RESOURCES
from .. import NUM_MAX_TRIALS

from .tools import entropy_from_pdf
from .ac_model import (build_actor, build_critic, normalize_state,
                       train_step_actor_critic)
from ..src.exp_utils import get_updated_severity
from ..src.exp_utils import calculate_normalised_final_severity_performance_metric


class Pandemic(Env):
    """
    Pandemic environment implementing OpenAI Gym's Env interface.

    The Pandemic environment simulates a pandemic response scenario where an agent
    must allocate limited resources across multiple cities to minimize final severity.
    Each episode consists of multiple sequences, and each sequence contains multiple trials.

    Attributes
    ----------
    max_resources : int
        Maximum resources available per sequence (after 9 are pre-assigned)
    available_resources_states : int
        Number of possible resource states (max_resources + 1)
    max_seq_length : int
        Maximum number of trials per sequence
    trial_no_states : int
        Number of possible trial number states (max_seq_length + 1)
    max_severity : int
        Maximum initial severity value
    severity_states : int
        Number of possible severity states (max_severity + 1)
    max_allocation : int
        Maximum resources that can be allocated in a single action
    observation_space : spaces.Box
        3D observation space for [available_resources, trial_number, severity]
    action_space : spaces.Discrete
        Discrete action space representing resource allocations (0 to max_allocation)
    """

    def __init__(self):
        """
        Initialize the Pandemic environment.

        Sets up the state and action spaces, initializes internal variables,
        and configures the environment for simulation.
        """
        # Construct the parent class
        super(Pandemic, self).__init__()

        # Number of available resources at the beginning (9 are preassigned)
        self.max_resources = AVAILABLE_RESOURCES_PER_SEQUENCE - 9
        self.available_resources_states = self.max_resources + 1

        # Ten trials per sequence, from 3 to 10
        self.max_seq_length = NUM_MAX_TRIALS
        self.trial_no_states = self.max_seq_length + 1

        # Ten severities, from 0 to 10
        self.max_severity = MAX_SEVERITY
        self.severity_states = self.max_severity + 1

        # Ten is the max alloc, Eleven choices, from 0 to 10
        self.max_allocation = MAX_ALLOCATABLE_RESOURCES

        # Define a 3-D observation space
        self.observation_shape = (self.available_resources_states,
                                  self.trial_no_states,
                                  self.severity_states)

        self.observation_space = spaces.Box(low=numpy.zeros(self.observation_shape),
                                            high=numpy.ones(self.observation_shape),
                                            dtype=numpy.float16)

        # Define an action space
        self.action_space = spaces.Discrete(self.max_allocation + 1,)

        # Create a canvas to render the environment images upon
        self.canvas = numpy.ones(self.observation_shape)

        # Define elements present inside the environment
        self.elements = []
        self.verbose = True
        self.number_cities_prob = numpy.asarray([], dtype=numpy.float64)
        self.severity_prob = numpy.asarray([], dtype=numpy.float64)

    def random_sequence(self):
        """
        Generate a random sequence with severities and allocations.

        Generates a sequence for simulation with random trial count, severities,
        and allocations. Uses uniform random values if no probability distributions
        are set, otherwise samples from the configured distributions.

        Sets
        ----
        self.seq_length : int
            Length of the randomly generated sequence
        self.initial_severities : list
            Initial severity values for each trial in the sequence
        self.allocations : list
            Resource allocations for each trial in the sequence
        """
        if (self.number_cities_prob.shape[0] == 0):
            self.seq_length = random.randrange(int(3), int(self.max_seq_length))
            self.allocations = [self.action_space.sample() for s in range(self.seq_length)]
            self.initial_severities = [random.randrange(int(0), int(self.max_severity)) for s in range(self.seq_length)]
        else:
            self.seq_length = int(numpy.random.choice(self.number_cities_prob[:, 0], p=(self.number_cities_prob[:, 1])))
            self.initial_severities = numpy.random.choice(
                self.severity_prob[:, 0], size=(self.seq_length,), p=self.severity_prob[:, 1])

    def set_fixed_sequence(self, length, init_severities, allocs=None):
        """
        Set a fixed sequence with specified parameters.

        Configures the environment with a predefined sequence length, initial
        severities, and optionally allocations. If allocations are not provided,
        they are randomly generated.

        Parameters
        ----------
        length : int
            Number of trials in the sequence
        init_severities : array-like
            Initial severity values for each trial
        allocs : array-like, optional
            Resource allocations for each trial. If None, allocations are randomly
            generated. Default: None
        """
        self.seq_length = int(length)
        self.set_initial_severities(init_severities)

        if allocs is None:
            self.allocations = [self.action_space.sample() for s in range(self.seq_length)]
        else:
            self.set_fixed_allocations(allocs)

    def set_fixed_allocations(self, allocs):
        """
        Set fixed resource allocations for the current sequence.

        Parameters
        ----------
        allocs : array-like
            Resource allocations for each trial in the sequence
        """
        self.allocations = allocs

    def set_initial_severities(self, init_severities):
        """
        Set the initial severity values for the current sequence.

        Parameters
        ----------
        init_severities : array-like
            Initial severity value for each trial in the sequence
        """
        self.initial_severities = init_severities

    def new_city(self):
        """
        Get the initial severity for the next city/trial.

        Returns
        -------
        float
            The initial severity value of the current iteration
        """
        return self.initial_severities[self.iteration]

    def sample(self):
        """
        Get the allocated resources for the current trial.

        Returns
        -------
        int
            Resource allocation for the current iteration
        """
        return self.allocations[self.iteration]

    def reset(self, seed=None, options=None):  # type: ignore[override]
        """
        Reset the environment to an initial state.

        Resets all tracking variables, initializes resources and severities,
        and returns an initial observation of the new sequence.

        Parameters
        ----------
        seed : int or None, optional
            Random seed (unused, kept for Gym API compatibility).
        options : dict or None, optional
            Extra reset options (unused, kept for Gym API compatibility).

        Returns
        -------
        list
            Initial observation [available_resources, trial_number, initial_severity]
        """
        # Reload the available resources
        self.available_resources = self.max_resources

        # Reset the reward
        self.ep_return = 0

        # City number
        self.iteration = 0

        self.severities = []
        self.resources = []

        self.severity_evolution = numpy.zeros((len(self.initial_severities) + 1, len(self.initial_severities)))
        self.severity_city_counter = 0

        self.done = False

        # Get a new city with its own severity, and keep going....
        new_severity = self.new_city()
        self.severities.append(new_severity)

        # return the observation
        return [self.available_resources, self.iteration, int(new_severity)]

    def render(self):
        """
        Render the current state of the environment.

        Prints human-readable information about the current episode state,
        including trial number, severities, and actions taken.

        Returns
        -------
        ndarray
            The canvas/observation array
        """
        if (self.done):
            print("--".format(self.iteration + 1), ':',
                  ":".join([" {:5.2f}".format(sev) for sev in self.severities]), '->', ' Done!')
        elif (len(self.resources) > 0):
            print("{:02d}".format(self.iteration + 1), ':',
                  ":".join(["{:5.2f}".format(sev) for sev in self.severities]), '->', self.resources[-1])
        return self.canvas

    def close(self):
        """
        Close the environment and clean up resources.

        Placeholder method for environment cleanup (currently does nothing).
        """

    def get_action_meanings(self):
        """
        Get the mapping between action indices and their meanings.

        Returns
        -------
        dict
            Dictionary mapping action indices (0-10) to resource allocation amounts
        """
        return {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "10"}

    def damage(self):
        """
        Calculate the updated severity based on current allocations.

        Returns
        -------
        ndarray
            Updated severity values for all trials based on resource allocations
        """
        return get_updated_severity(len(self.severities), self.resources, self.severities)

    def step(self, action):
        """
        Execute one step of the environment.

        Applies the specified action, updates the environment state, calculates
        rewards, and determines if the episode is complete.

        Parameters
        ----------
        action : int
            The action to take (resource allocation amount, 0-10)

        Returns
        -------
        tuple
            - observation (list): New state [available_resources, trial_number, severity]
            - reward (float): Reward for this step (negative sum of severities)
            - done (bool): Whether the episode is finished
            - info (list): Additional information (empty list)
        """
        # Flag that marks the termination of an episode
        done = False

        # Assert that it is a valid action
        assert self.action_space.contains(action), f'Invalid Action {action}'

        # Reward for executing a step.
        reward = 0

        if ((self.available_resources - action) <= 0):
            action = self.available_resources

        self.available_resources -= action
        self.resources.append(action)

        if (self.verbose):
            self.render()

        self.severity_evolution[self.severity_city_counter][:len(self.severities)] = self.severities

        self.severities = get_updated_severity(len(self.severities), self.resources, self.severities)

        self.severity_city_counter = self.severity_city_counter + 1

        # Increment the episodic return
        self.ep_return += 1
        self.iteration += 1

        # Get a new city with its own severity, and keep going....
        reward = (-1) * numpy.sum(self.severities)

        # If the length of the sequence was achieved, stop
        if (self.iteration) == self.seq_length:
            done = True
            new_severity = 0

            # Update the evolution of the severity one more time for the final severity of all the cities.
            self.severity_evolution[self.severity_city_counter][:len(self.severities)] = self.severities
        else:
            new_severity = self.new_city()
            self.severities.append(new_severity)

        return [self.available_resources, self.iteration, int(new_severity)], reward, done, False, {}  # type: ignore[override]


def ac_agent_meta_cognitive(policy_probs, resources_left, response_timeout):
    """
    Compute meta-cognitive confidence and response time estimates from Actor policy output.

    Unlike Q-Learning or DQN (which compute entropy over Q-values — a heuristic),
    this function uses the Actor's π(a|s) output **directly**.  Since π(a|s) is
    already a proper probability distribution, its entropy is a theoretically
    grounded measure of decision uncertainty.

    Confidence formula::

        confidence = 1 - (H(π_feasible) / H_max)

    where H_max = log₂(n_actions) and H(π_feasible) is the Shannon entropy of
    the policy restricted to feasible actions (allocation ≤ resources_left).

    Parameters
    ----------
    policy_probs : array-like
        Action probabilities from the Actor's softmax output.  Shape: ``(n_actions,)``.
    resources_left : int
        Number of resources remaining.
    response_timeout : float
        Maximum response time allowed in milliseconds.

    Returns
    -------
    response : int
        The selected action (argmax of feasible probabilities).
    confidence : float
        Normalised confidence score based on entropy (range: typically 0-1).
        Lower entropy → higher confidence.
    rt_hold : float
        Response time for button hold phase (in seconds).
    rt_release : float
        Response time for button release phase (in seconds).

    Notes
    -----
    - Infeasible actions (allocation > resources_left) are set to a tiny value
      before entropy computation and action selection.
    - Confidence is computed as the normalised inverse of entropy based on the
      min/max entropy reference distributions.
    - Response times are sampled from normal distributions parameterised by
      confidence.
    - Both *rt_hold* and *rt_release* are clipped to ``[0, response_timeout / 1000]``.
    """

    # Ensure we work with a numpy copy
    probs = numpy.array(policy_probs, dtype=numpy.float64).flatten()

    # Min entropy from a univalue distribution (0)
    m_entropy = numpy.zeros((len(probs),),)
    m_entropy[0] = 1

    # Max entropy from a uniform distribution (3.55....)
    M_entropy = numpy.ones((len(probs),),)

    # Calculate the entropy of the raw policy distribution
    _entrp1 = entropy_from_pdf(probs)

    o = numpy.arange(len(probs), dtype=numpy.float32)

    # Set options that are not feasible (greater than resources left)
    # to a very small value to avoid them being selected
    probs[o > resources_left] = 0.00001

    # available resources, trial, severity
    dec_entropy = entropy_from_pdf(probs)
    M_entropy = entropy_from_pdf(M_entropy)
    m_entropy = entropy_from_pdf(m_entropy)

    # Calculate confidence as a normalized inverse of entropy
    confidence = (1. / (m_entropy - M_entropy)) * (dec_entropy - M_entropy)

    # Select the action with the highest probability as the response
    response = numpy.argmax(probs)

    # Map confidence to response times using a linear transformation
    def map_to_response_time(x): return x * (-2) + 1
    mu, sigma = int(map_to_response_time(confidence) * 10), 3

    rt_hold = numpy.random.normal(mu, sigma, 1)[0]
    rt_release = rt_hold + numpy.random.normal(mu, 1, 1)[0]

    rt_hold = numpy.clip(rt_hold, 0, response_timeout / 1000.0)
    rt_release = numpy.clip(rt_release, 0, response_timeout / 1000.0)

    return response, confidence, rt_hold, rt_release


def run_experiment(env, actionfunction, RandomSequences=True,
                   trials_per_sequence=None, sevs=None,
                   NumberOfIterations=64):
    """
    Execute a pandemic simulation experiment over multiple sequences.

    Runs the Pandemic environment with a specified action function at each step,
    collecting performance metrics across multiple sequences.  Supports both
    random and fixed sequence generation with optional pre-defined severities.

    Parameters
    ----------
    env : Pandemic
        The Pandemic environment instance to run the experiment on.
    actionfunction : callable
        Function that takes ``(env, state, sequence_id)`` and returns an action (int).
    RandomSequences : bool, optional
        If ``True``, generates random sequences.  If ``False``, uses fixed sequences
        from *trials_per_sequence* and *sevs*.  Default: ``True``.
    trials_per_sequence : array-like, optional
        Number of trials in each sequence.  Required when ``RandomSequences=False``.
        Shape: ``(NumberOfIterations,)``.
    sevs : array-like, optional
        Initial severity values for each trial in each sequence.  Required when
        ``RandomSequences=False``.  Shape: ``(NumberOfIterations, variable_length)``.
    NumberOfIterations : int, optional
        Number of sequences to simulate.  Default: 64.

    Returns
    -------
    seqs : list of float
        Total severity sum for each completed sequence.
    perfs : list of float
        Normalised performance metric for each sequence.
    seq_ev : list
        Severity evolution matrix for each sequence.
    """

    seqid = 0
    if RandomSequences:
        env.random_sequence()
    else:
        assert trials_per_sequence is not None and sevs is not None
        env.set_fixed_sequence(trials_per_sequence[seqid], sevs[seqid])
    state = env.reset()
    seqs = []
    perfs = []
    seq_ev = []
    ITERATIONS = NumberOfIterations
    while seqid < ITERATIONS:
        print(f'State: {state}')
        action = actionfunction(env, state, seqid)
        state2, _reward, done, _truncated, _info = env.step(action)

        if done:
            env.done = True
            env.render()
            seqs.append(numpy.sum(env.severities))
            perf = calculate_normalised_final_severity_performance_metric(env.severities,
                                                                          env.initial_severities)
            perfs.append(perf[0])
            seq_ev.append(env.severity_evolution)
            seqid = seqid + 1

            if seqid < ITERATIONS:
                if RandomSequences:
                    env.random_sequence()
                else:
                    assert trials_per_sequence is not None and sevs is not None
                    env.set_fixed_sequence(trials_per_sequence[seqid], sevs[seqid])
            state2 = env.reset()

        state = state2

    print(seqs)
    env.close()

    return seqs, perfs, seq_ev


def A2CTraining(env, actor_lr, critic_lr, discount, entropy_coeff,
                epsilon, min_eps, episodes,
                actor_hidden, critic_hidden,
                seed=None, compute_confidence=False):
    """
    Train an Advantage Actor-Critic (A2C) agent on the Pandemic environment.

    Uses on-policy updates — each transition is used to update both the Actor
    (policy network) and the Critic (value network) immediately within the
    episode.  An ε-greedy overlay is maintained for additional exploration
    during early training, with linear decay.

    Parameters
    ----------
    env : Pandemic
        Gym environment instance.
    actor_lr : float
        Adam optimiser learning rate for the Actor.
    critic_lr : float
        Adam optimiser learning rate for the Critic.
    discount : float
        Discount factor γ ∈ (0, 1].
    entropy_coeff : float
        Weight for the entropy bonus in the Actor loss.
    epsilon : float
        Initial exploration rate ε₀.
    min_eps : float
        Minimum exploration rate ε_min.
    episodes : int
        Total number of training episodes.
    actor_hidden : list of int
        Widths of the hidden dense layers for the Actor (e.g. ``[64, 64]``).
    critic_hidden : list of int
        Widths of the hidden dense layers for the Critic (e.g. ``[64, 64]``).
    seed : int or None, optional
        Random seed for full reproducibility.  Default: ``None``.
    compute_confidence : bool, optional
        When ``True``, run an extra forward pass every step to record
        meta-cognitive confidence values.  **Disabling this (default)
        reduces computation during training.**  Default: ``False``.

    Returns
    -------
    ave_reward_list : list of float
        Average reward computed every 10 000 episodes.
    actor_model : tf.keras.Model
        Trained Actor (policy) network.
    conf_list : list of float
        Meta-cognitive confidence values (one per environment step).
        Empty list when *compute_confidence* is ``False``.

    Notes
    -----
    - ε decays linearly from *epsilon* to *min_eps* over *episodes*.
    - The state vector is normalised to [0, 1]³ before being fed to the
      networks (see :func:`ac_model.normalize_state`).
    - The function prints average reward every 10 000 episodes to track
      convergence.
    - A2C performs on-policy updates: each transition (s, a, r, s', done)
      is used **once** to update both Actor and Critic via
      :func:`ac_model.train_step_actor_critic`.
    """

    # ----- Reproducibility -----
    if seed is not None:
        numpy.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)

    # ----- Dimensions -----
    state_dim = 3
    action_dim = env.action_space.n
    max_res = env.available_resources_states - 1       # 30
    max_tri = env.trial_no_states - 1                  # 10
    max_sev = env.severity_states - 1                  # 9

    # ----- Networks -----
    actor_model = build_actor(state_dim, action_dim, actor_hidden)
    critic_model = build_critic(state_dim, critic_hidden)

    # Build both networks with a dummy forward pass
    _dummy = tf.zeros((1, state_dim))
    actor_model(_dummy)
    critic_model(_dummy)

    actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

    # ----- Tracking -----
    reward_list: list[float] = []
    ave_reward_list: list[float] = []
    conf_list: list[float] = []

    reduction = (epsilon - min_eps) / episodes

    # ----- Training loop -----
    for i in range(episodes):
        done = False
        tot_reward: float = 0.0
        env.random_sequence()
        state = env.reset()

        # Collect episode transitions for batched update
        ep_states = []
        ep_actions = []
        ep_rewards = []
        ep_next_states = []
        ep_dones = []

        while not done:
            state_norm = normalize_state(state, max_res, max_tri, max_sev)

            # ε-greedy action selection using Actor policy
            if numpy.random.random() < 1 - epsilon and state[0] is not None:
                probs = actor_model(
                    state_norm[numpy.newaxis, :], training=False
                )[0].numpy()
                action = int(numpy.argmax(probs))
            else:
                action = numpy.random.randint(0, action_dim)
                probs = None

            # Meta-cognitive confidence (observational only)
            if compute_confidence:
                if probs is None:
                    probs = actor_model(
                        state_norm[numpy.newaxis, :], training=False
                    )[0].numpy()
                _, confidence, _, _ = ac_agent_meta_cognitive(
                    probs.copy(), state[0], 10000
                )
                conf_list.append(confidence)

            # Step
            state2, reward, done, _trunc, _info = env.step(action)
            state2_norm = normalize_state(state2, max_res, max_tri, max_sev)

            ep_states.append(state_norm)
            ep_actions.append(action)
            ep_rewards.append(float(reward))
            ep_next_states.append(state2_norm)
            ep_dones.append(float(done))

            tot_reward += reward
            state = state2

        # ---------- End-of-episode batch update ----------
        if len(ep_states) > 0:
            train_step_actor_critic(
                actor_model, critic_model,
                actor_optimizer, critic_optimizer,
                tf.constant(numpy.array(ep_states, dtype=numpy.float32)),
                tf.constant(numpy.array(ep_actions, dtype=numpy.int32)),
                tf.constant(numpy.array(ep_rewards, dtype=numpy.float32)),
                tf.constant(numpy.array(ep_next_states, dtype=numpy.float32)),
                tf.constant(numpy.array(ep_dones, dtype=numpy.float32)),
                float(discount),
                float(entropy_coeff),
            )

        # ε-decay
        if epsilon > min_eps:
            epsilon -= reduction

        reward_list.append(tot_reward)

        if (i + 1) % 10000 == 0:
            ave_reward = float(numpy.mean(reward_list))
            ave_reward_list.append(ave_reward)
            reward_list = []
            print(f"Episode {i + 1} Average Reward: {ave_reward:.4f}")

    env.close()
    return ave_reward_list, actor_model, conf_list
