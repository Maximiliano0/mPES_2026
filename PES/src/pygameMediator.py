"""RL Agent Game Display Mediator and Response Handler.

This module bridges the Pygame game interface with the trained RL agent (Q-Learning model),
handling agent decision-making, response timing, and confidence calculations. It manages
the communication between the game display and the RL agent by processing game states
and generating appropriately timed and confident responses.

Key Functions:
    - entropy: Calculate empirical distribution entropy
    - calculate_agent_response_and_confidence: Neural network confidence estimation
    - rl_agent_meta_cognitive: Meta-cognitive decision making with entropy-based confidence
    - provide_rl_agent_response: Main interface returning agent response and timing

Module Dependencies:
    External: numpy, tensorflow, os
    Internal: log_utils, convert_globalseq_to_seqs, CONFIG constants

Global Variables:
    first_severity: Initial severity array loaded by caller
    number_of_trials: Total trial count for experiment

Author: PES Development Team
Version: 1.0
"""

##########################
##  Imports externos    ##
##########################
import numpy
import os
import tensorflow as tf

##########################
##  Imports internos    ##
##########################
from .. import *
from .. ext.tools import convert_globalseq_to_seqs 
from . import log_utils

##########################################################
## Variables requiring initialisation before module use ##
##########################################################
first_severity        = None
number_of_trials      = None

#################################
## Module-Specific constants   ##
#################################
FONT              = 'ubuntumono'   # previously: Arial
BACKGROUND_COLOUR = ANSI.GRAY
RESPONSE_TIMEOUT  = 5000  # in milliseconds

def entropy(x, bins=None):
    """Calculate Shannon entropy of an empirical distribution.
    
    Computes the Shannon entropy H = -Σ(p_i * log2(p_i)) of a discrete distribution,
    where p_i are the probabilities estimated from the input data.
    
    Parameters
    ----------
    x : array_like
        1-D array of values for which to compute entropy
    bins : int, optional
        Number of bins for histogram computation. If None, uses numpy.bincount()
        for integer-valued data
    
    Returns
    -------
    float
        Shannon entropy in bits (log base 2). Higher values indicate more uniform
        distributions; lower values indicate more peaked distributions.
    
    Notes
    -----
    - Zero probabilities are excluded from entropy calculation to avoid log(0)
    - Uses log base 2, so entropy is measured in bits
    - Entropy = 0 for deterministic distributions (single value)
    - Maximum entropy occurs for uniform distributions
    
    Examples
    --------
    >>> x_uniform = np.array([1, 2, 3, 4, 5])
    >>> entrop_uniform = entropy(x_uniform)  # ~2.32 bits
    >>> x_peaked = np.array([5, 5, 5, 5, 1, 1])  
    >>> entropy_peaked = entropy(x_peaked)  # ~0.65 bits (more peaked)
    """

    N = x.shape

    if bins is None:   counts = numpy.bincount( x )
    else           :   counts = numpy.histogram( x, bins = bins )[ 0 ]   # Counts, probs

    p = counts[ numpy.nonzero( counts ) ] / N    # log(0)
    H = -numpy.dot( p, numpy.log2( p ) )

    return H


def calculate_agent_response_and_confidence(model, city_severity, trial_no, resource_remaining):
    """Estimate agent confidence and response using neural network stochasticity.
    
    Converts a neural network into a stochastic policy by adding Gaussian noise to
    input features. The resulting distribution of responses is used to calculate
    entropy, which is normalized to extract a confidence metric (0-1 range).
    
    This function models decision uncertainty as emerging from noisy neural network
    inputs rather than explicit stochasticity in the network itself.
    
    Parameters
    ----------
    model : tf.keras.Model
        Trained neural network that accepts (city_severity, trial_no, resource_remaining)
        and outputs a single resource allocation value
    city_severity : float
        Current severity level of the city (typically 0-5, from Q-table indexing)
    trial_no : int
        Current trial number in sequence
    resource_remaining : float or int
        Number of resources available for allocation
    
    Returns
    -------
    confidence : float
        Normalized confidence metric (typically 0-1 range)
        - 0 = maximum entropy (high uncertainty)
        - 1 = minimum entropy (high certainty)
    response : float
        Mean of the sampled response distribution (mean resource allocation)
    
    Notes
    -----
    - Samples 1000 noisy variations of the input features
    - Gaussian noise with std=1 added to each feature
    - Entropy calculated across resource allocation range (0-20)
    - Normalization requires both maximum and minimum entropy baselines
    - Confidence formula: (1/(m-M)) * (H - M) where M=max, m=min, H=decision entropy
    
    Examples
    --------
    >>> model = load_pretrained_nn_model()
    >>> conf, resp = calculate_agent_response_and_confidence(
    ...     model, city_severity=3.5, trial_no=8, resource_remaining=15)
    >>> print(f"Confidence: {conf:.2f}, Response: {resp:.1f} resources")
    # Output example: Confidence: 0.75, Response: 12.3 resources
    """
    repl = 1000
    M_entropy = entropy(numpy.linspace(MIN_ALLOCATABLE_RESOURCES,MAX_ALLOCATABLE_RESOURCES, repl), bins=MAX_ALLOCATABLE_RESOURCES-MIN_ALLOCATABLE_RESOURCES+1)
    m_entropy = entropy(numpy.ones((repl,)), bins=MAX_ALLOCATABLE_RESOURCES-MIN_ALLOCATABLE_RESOURCES+1)
    allocated_resources = numpy.asarray([])
    for i in range(0,repl):
        r_remaining = resource_remaining + numpy.random.normal(MIN_ALLOCATABLE_RESOURCES,MAX_ALLOCATABLE_RESOURCES,1)[0]
        c_severity = city_severity +  numpy.random.normal( 0,3,1)[0]
        t_no = trial_no + numpy.random.normal(0,3,1)[0]
        resp = model( tf.Variable(c_severity, dtype=tf.float32), tf.Variable(t_no, dtype=tf.float32), tf.Variable(r_remaining, dtype=tf.float32) )
        allocated_resources = numpy.append( allocated_resources, resp.numpy())

    resp = allocated_resources.mean()
    entrp = entropy(allocated_resources, bins=MAX_ALLOCATABLE_RESOURCES-MIN_ALLOCATABLE_RESOURCES+1)
    confidence = (1./(m_entropy-M_entropy)) * (entrp - M_entropy)

    return confidence, resp


def rl_agent_meta_cognitive(options, resources_left, response_timeout):
    """Generate RL agent response with meta-cognitive confidence and timing simulation.
    
    Implements a meta-cognitive decision-making process where the agent:
    1. Evaluates confidence based on entropy of Q-table options
    2. Filters infeasible options (exceeding available resources)
    3. Generates decision entropy from feasible options
    4. Maps confidence to response timing (reaction time, release time)
    
    The meta-cognitive mechanism assumes that higher uncertainty (entropy) in the
    decision space correlates with longer response time to deliberate before committing.
    
    Parameters
    ----------
    options : array_like
        1-D array of Q-values from the Q-table state [resources, trial, severity]
        Typically shape (11,) representing resource allocation probabilities
    resources_left : float or int
        Available resources that constrain feasible actions
    response_timeout : int
        Maximum response time in milliseconds (e.g., 5000 for 5 seconds)
    
    Returns
    -------
    response : int
        Selected resource allocation (argmax of feasible options)
        Guaranteed to not exceed resources_left
    confidence : float
        Meta-cognitive confidence metric (0-1 range)
        - Based on entropy of feasible option distribution
        - 0 = uniform distribution (low confidence)
        - 1 = peaked distribution (high confidence)
    rt_hold : float
        Reaction time from stimulus to response button press (seconds)
        Sampled from Gaussian scaled by confidence
    rt_release : float
        Total time from stimulus to releasing response button (seconds)
        Always >= rt_hold (represents commitment duration)
    
    Notes
    -----
    - Confidence inverted from entropy: (1/(m_entropy - M_entropy)) * (H - M_entropy)
    - High confidence → faster response (lower RT)
    - Low confidence → longer deliberation (higher RT)
    - All response times clipped to [0, response_timeout/1000]
    - Response always clipped to feasible range [0, resources_left]
    
    Examples
    --------
    >>> q_options = np.array([0.1, 0.2, 0.5, 0.15, 0.05, 0, 0, 0, 0, 0, 0])
    >>> response, conf, rt_h, rt_r = rl_agent_meta_cognitive(
    ...     q_options, resources_left=15, response_timeout=5000)
    >>> print(f"Response: {response}, Confidence: {conf:.2f}, "  
    ...       f"Hold time: {rt_h:.3f}s, Release time: {rt_r:.3f}s")
    # Output: Response: 2, Confidence: 0.68, Hold time: 0.234s, Release time: 0.567s
    """

    def entropy_from_pdf(pdf):
        pdf = pdf + numpy.abs(numpy.min( pdf ))
        p =  pdf / numpy.sum(pdf)  # log(0)
        p[ p==0 ] += 0.000001
        print( p )
        H = -numpy.dot( p, numpy.log2( p ) )
        return H

  # Min entropy from a univalue distribution (0)
    m_entropy = numpy.zeros((11,),)
    m_entropy[0] = 1

  # Max entropy from a uniform distribution (3.55....)
    M_entropy = numpy.ones((11,),)

  # Options are the available choices from the Q Table
    log_utils.tee( 'Options:', options )

    entrp1 = entropy_from_pdf(options)

    o = [i for i in range(len(options))]
    o = numpy.asarray(o, dtype=numpy.float32)

    options[o>resources_left] = 0.00001

    log_utils.tee( 'Agent Feasible Options:', options)

  # available resources, trial, severity
    dec_entropy = entropy_from_pdf(options)
    M_entropy = entropy_from_pdf(M_entropy)
    m_entropy = entropy_from_pdf(m_entropy)

    confidence = (1./(m_entropy-M_entropy)) * (dec_entropy - M_entropy)

    response = numpy.argmax(options)
    
    # Ensure response never exceeds available resources
    response = int(numpy.clip(response, 0, int(resources_left)))

    map_to_response_time = lambda x: x * (-2) + 1

    mu, sigma = int(map_to_response_time(confidence) * 10), 3

    rt_hold = numpy.random.normal(mu, sigma, 1)[0]
    rt_release = rt_hold + numpy.random.normal(mu, 1, 1)[0]

    rt_hold = numpy.clip( rt_hold, 0, response_timeout/1000.0)
    rt_release = numpy.clip( rt_release, 0, response_timeout/1000.0)

    return response, confidence, rt_hold, rt_release


def provide_rl_agent_response(
                         resources,
                         resources_left,
                         session_no,
                         sequence_no,
                         trial_no
                          ):
    """Generate RL agent response using trained Q-Learning policy.
    
    Main interface for obtaining agent responses. Loads the trained Q-table from disk,
    retrieves the current game state (severity, resources, trial), selects the
    appropriate Q-value, and generates a response with confidence and timing metadata.
    
    This function handles all file I/O, state indexing, Q-table lookup, and calls
    the meta-cognitive decision function to produce a realistic response.
    
    Parameters
    ----------
    resources : float
        Total resources available in session (informational, not used for indexing)
    resources_left : float or int
        Remaining resources available for allocation (Q-table first dimension)
    session_no : int
        Session identifier (0-indexed) for multi-session experiments
    sequence_no : int
        Sequence within session (0-indexed)
    trial_no : int
        Trial within sequence (0-indexed, Q-table second dimension)
    
    Returns
    -------
    confidence : float
        Decision confidence metric (0-1 range) from meta-cognitive processing
    response : int
        Resource allocation decision (0 to resources_left)
    rt_hold : float
        Reaction time in seconds (stimulus to button press)
    rt_release : float  
        Total response duration in seconds (stimulus to button release)
    movement : list
        Placeholder for movement data (currently empty list)
    
    Raises
    ------
    AssertionError
        If first_severity module variable not initialized by caller
    FileNotFoundError
        If Q-table (q.npy) or rewards file (rewards.npy) not found in INPUTS_PATH
    RuntimeError
        If Q-table or rewards files are corrupted and cannot be loaded
    
    Notes
    -----
    - Requires Q-table pre-training via: python3 -m PES.ext.train_rl
    - Requires first_severity initialized: call before using this function
    - Q-table dimensions: [resources (31) × trials (11) × severity (6)]
    - State indices automatically clamped to valid ranges
    - All Q-table values converted to integers for safe array indexing
    - Uses VERBOSE flag to enable debug output during execution
    
    Examples
    --------
    Initialize module variable before first call::
    
        pygameMediator.first_severity = severity_array  # shape (num_sequences,)
        
    Then use function in game loop::
    
        conf, resp, rth, rtr, mov = provide_rl_agent_response(
            resources=30, resources_left=15, session_no=0,
            sequence_no=2, trial_no=7)
        print(f"Agent allocated {resp} resources with {conf:.2f} confidence")
        print(f"Response timing: {rth:.3f}s to press, {rtr:.3f}s to release")
    """
    
    assert first_severity is not None, \
           "The 'first_severity' module-global variable needs to be set by caller before calling this function"

    # Load and validate Q-Table
    q_file = os.path.join(INPUTS_PATH, 'q.npy')
    rewards_file = os.path.join(INPUTS_PATH, 'rewards.npy')
    
    if not os.path.isfile(q_file):
        raise FileNotFoundError(
            f"\nFATAL ERROR: Q-Table file not found at {q_file}\n"
            f"Please train the RL-Agent first by running: python3 -m PES.ext.train_rl\n"
        )
    
    if not os.path.isfile(rewards_file):
        raise FileNotFoundError(
            f"\nFATAL ERROR: Rewards file not found at {rewards_file}\n"
            f"Please train the RL-Agent first by running: python3 -m PES.ext.train_rl\n"
        )
    
    try:
        Q = numpy.load(q_file)
        rewards = numpy.load(rewards_file)
    except Exception as e:
        raise RuntimeError(
            f"\nFATAL ERROR: Failed to load training files!\n"
            f"Error: {str(e)}\n"
            f"Files may be corrupted. Please retrain by running: python3 -m PES.ext.train_rl\n"
        )

    if VERBOSE:
        print( "Reading preloaded Q-Table for RL-Agent" )

    resources_remaining = tf.Variable(resources_left, dtype=tf.float32)

    if VERBOSE:
        print( 'Resources remaining...' )
        print( int(resources_remaining.numpy()) if hasattr(resources_remaining, 'numpy') else resources_remaining )
        print()

    SequenceLengthsCsv = os.path.join( INPUTS_PATH, SEQ_LENGTHS_FILE )
    sequence_length = numpy.loadtxt( SequenceLengthsCsv , delimiter=',')
    sevs = convert_globalseq_to_seqs(sequence_length, first_severity)

    sever = sevs[ session_no * NUM_SEQUENCES + sequence_no ][ trial_no ]
    city_number = trial_no

    # Convert to integers for array indexing (handle numpy types and tensors)
    try:
        resources_idx = int(resources_left)
    except (ValueError, TypeError):
        resources_idx = int(resources_left.numpy())
    
    try:
        city_idx = int(city_number)
    except (ValueError, TypeError):
        city_idx = int(city_number.numpy())
    
    try:
        sever_idx = int(sever)
    except (ValueError, TypeError):
        sever_idx = int(sever.numpy())
    
    # Clamp indices to valid ranges
    resources_idx = max(0, min(resources_idx, Q.shape[0]-1))
    city_idx = max(0, min(city_idx, Q.shape[1]-1))
    sever_idx = max(0, min(sever_idx, Q.shape[2]-1))
    
    if VERBOSE:
        print(f"State indices - Resources: {resources_idx}, City: {city_idx}, Severity: {sever_idx}")
        print(f"Q-Table shape: {Q.shape}")
        print(f"Q values for this state: {Q[resources_idx, city_idx, sever_idx]}")

    # Calculate the response and confidence feeding the NN with noisy inputs, getting the mean and entropy from the responses.
    resp, confidence, rt_hold, rt_release = rl_agent_meta_cognitive(Q[resources_idx, city_idx, sever_idx], resources_left, RESPONSE_TIMEOUT)
    
    # Final validation: ensure response never exceeds available resources
    resp = int(numpy.clip(resp, 0, int(resources_left)))
    
    if VERBOSE:
        print(f"RL-Agent Response: {resp}, Confidence: {confidence}")
        print(f"Resources available: {int(resources_left)}, Response clamped to: {resp}")

    movement = []

    return confidence, resp, rt_hold, rt_release, movement