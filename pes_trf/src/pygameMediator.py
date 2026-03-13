"""pes_trf — RL Agent Response Handler and Confidence Estimator.

Bridges the experiment loop (__main__.py) with the trained Transformer model,
handling agent decision-making, response timing, and entropy-based confidence
calculations.

Key Functions:
    - rl_agent_meta_cognitive: Entropy-based confidence over action distribution.
    - provide_rl_agent_response: Main interface — loads Transformer model,
      selects greedy action via causal attention over episode trajectory,
      computes confidence, returns formatted response.

Model layout (loaded from ``INPUTS_PATH``):
    Architecture config:  ``transformer_config.json``
    Weights:              ``transformer_weights.weights.h5``

    The model receives the trajectory of states seen so far within the
    current sequence and outputs action logits at the last position.
    A causal mask ensures each position only attends to past context.

Module Dependencies:
    External: numpy, tensorflow, os
    Internal: log_utils, convert_globalseq_to_seqs, CONFIG constants,
              transformer_model.PandemicTransformer

Global Variables:
    first_severity: Initial severity array set by caller before first use.
    number_of_trials: Total trial count for the experiment.
"""

##########################
##  Imports externos    ##
##########################
import numpy
import os

##########################
##  Imports internos    ##
##########################
from .. import ANSI, INPUTS_PATH, NUM_SEQUENCES, SEQ_LENGTHS_FILE, VERBOSE
from . import log_utils
from ..ext.tools import convert_globalseq_to_seqs
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

###############################################
## Transformer model singleton + trajectory  ##
###############################################
_transformer_model = None          # PandemicTransformer (loaded once)
_trajectory_buffer = []            # list of [res, trial, sev] for current sequence
_last_abs_seq      = -1            # detect new sequence → reset trajectory

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
        1-D array of action values or softmax probabilities.
        Typically shape (11,) representing resource allocation options.
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

    o = list(range(len(options)))
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

    def map_to_response_time(x):
        return x * (-2) + 1

    mu, sigma = int(map_to_response_time(confidence) * 10), 3

    rt_hold = numpy.random.normal(mu, sigma, 1)[0]
    rt_release = rt_hold + numpy.random.normal(mu, 1, 1)[0]

    rt_hold = numpy.clip( rt_hold, 0, response_timeout/1000.0)
    rt_release = numpy.clip( rt_release, 0, response_timeout/1000.0)

    return response, confidence, rt_hold, rt_release


def provide_rl_agent_response(
                         _resources,
                         resources_left,
                         session_no,
                         sequence_no,
                         trial_no
                          ):
    """Generate agent response using the trained Transformer policy.

    Loads the Transformer model (once, cached), maintains the episode
    trajectory, feeds all past states through causal self-attention, and
    returns the greedy action at the last position.  Confidence is
    computed from the entropy of the softmax policy — same metric as
    the Q-Learning baseline.

    Parameters
    ----------
    resources : float
        Total resources available in session (informational; prefixed with
        ``_`` in the signature because it is unused).
    resources_left : float or int
        Remaining resources available for allocation.
    session_no : int
        Session identifier (0-indexed).
    sequence_no : int
        Sequence within session (0-indexed).
    trial_no : int
        Trial within sequence (0-indexed).

    Returns
    -------
    confidence : float
        Decision confidence metric (0–1) based on policy entropy.
    response : int
        Resource allocation decision (0 to resources_left).
    rt_hold : float
        Reaction time in seconds (stimulus to button press).
    rt_release : float
        Total response duration in seconds (stimulus to button release).
    movement : list
        Placeholder (empty list, preserved for interface compatibility).

    Raises
    ------
    AssertionError
        If ``first_severity`` module variable not initialised by caller.
    FileNotFoundError
        If model files not found in ``INPUTS_PATH``.

    Notes
    -----
    - Requires pre-training via: ``python3 -m pes_trf.ext.train_transformer``
    - The model is loaded once and cached as a module-level singleton.
    - Trajectory buffer resets automatically at the start of each sequence
      (when ``trial_no == 0``).
    """
    global _transformer_model, _trajectory_buffer, _last_abs_seq

    assert first_severity is not None, \
           "The 'first_severity' module-global variable needs to be set by caller before calling this function"

    # ── load model (singleton) ───────────────────────────────────
    if _transformer_model is None:
        config_file = os.path.join(INPUTS_PATH, 'transformer_config.json')
        weights_file = os.path.join(INPUTS_PATH, 'transformer_weights.weights.h5')

        if not os.path.isfile(config_file) or not os.path.isfile(weights_file):
            raise FileNotFoundError(
                f"\nFATAL ERROR: Transformer model files not found in {INPUTS_PATH}\n"
                f"Expected: transformer_config.json, transformer_weights.weights.h5\n"
                f"Please train the agent first by running: python3 -m pes_trf.ext.train_transformer\n"
            )

        try:
            from ..ext.transformer_model import PandemicTransformer
            _transformer_model = PandemicTransformer.from_pretrained(INPUTS_PATH)
            if VERBOSE:
                print("Transformer model loaded successfully")
                print(f"  Parameters: {_transformer_model.count_params():,}")
        except Exception as e:
            raise RuntimeError(
                f"\nFATAL ERROR: Failed to load Transformer model!\n"
                f"Error: {str(e)}\n"
                f"Please retrain by running: python3 -m pes_trf.ext.train_transformer\n"
            ) from e

    # ── trajectory management ────────────────────────────────────
    abs_seq = session_no * NUM_SEQUENCES + sequence_no
    if abs_seq != _last_abs_seq or trial_no == 0:
        _trajectory_buffer = []
        _last_abs_seq = abs_seq

    # ── resolve severity for current trial ───────────────────────
    SequenceLengthsCsv = os.path.join(INPUTS_PATH, SEQ_LENGTHS_FILE)
    sequence_length = numpy.loadtxt(SequenceLengthsCsv, delimiter=',')
    sevs = convert_globalseq_to_seqs(sequence_length, first_severity)

    sever = sevs[abs_seq][trial_no]

    resources_idx = max(0, min(int(resources_left), 30))
    city_idx = max(0, min(int(trial_no), 10))
    sever_idx = max(0, min(int(sever), 9))

    # ── build trajectory and query model ─────────────────────────
    _trajectory_buffer.append([resources_idx, city_idx, sever_idx])

    import tensorflow as tf_local
    states = tf_local.constant([_trajectory_buffer], dtype=tf_local.float32)

    _, probs, _, _ = _transformer_model.get_action_and_value(
        states, greedy=True, resources_left=resources_idx,
    )

    if VERBOSE:
        print(f"State: resources={resources_idx}, trial={city_idx}, severity={sever_idx}")
        print(f"Policy: {[f'{p:.3f}' for p in probs]}")

    # ── confidence via entropy (same mechanism as Q-Learning) ────
    resp, confidence, rt_hold, rt_release = rl_agent_meta_cognitive(
        probs, resources_left, RESPONSE_TIMEOUT,
    )

    # Final validation: ensure response never exceeds available resources
    resp = int(numpy.clip(resp, 0, int(resources_left)))

    if VERBOSE:
        print(f"Transformer Response: {resp}, Confidence: {confidence:.4f}")

    movement = []
    return confidence, resp, rt_hold, rt_release, movement
