"""
PES - Pandemic Experiment Scenario

This module contains functions that provide a higher-level experiment-specific
interface to respective lower-level pygame engine functions.

Functions defined here:
 • ask_player_to_rate_colleagues_using_sliders
 • calculate_agent_response_and_confidence
 • calculate_agent_response_and_confidence_alternative
 • convert_globalseq_to_seqs
 • divide_circle
 • down_arrow
 • draw_city_marker
 • entropy
 • get_age_from_user
 • get_gender_from_user
 • get_handedness_from_user
 • get_user_input
 • gracefully_quit_pygame
 • hide_mouse_cursor
 • init_pygame_display
 • load_image
 • provide_agent_confidence
 • provide_agent_response
 • provide_confidence
 • provide_replay_confidence
 • provide_replay_response
 • provide_response
 • provide_rl_agent_response
 • provide_user_confidence
 • provide_user_response
 • reset_screen
 • rl_agent_meta_cognitive
 • screen_messages
 • set_window_title
 • show_before_and_after_map
 • show_end_of_trial_feedback
 • show_feedback
 • show_images
 • show_message_and_wait
 • up_arrow
"""

# ----------------
# external imports
# ----------------

import copy
import glob
import math
import numpy
import os
import tensorflow as tf

# ----------------
# internal imports
# ----------------

from . import exp_utils
from . import log_utils
from . import Agent
from .Agent import agent_meta_cognitive, adjust_response_decay, boltzmann_decay, get_random_confidence 
from .exp_utils import chain_ops

from .. import *
from .. ext.tools import convert_globalseq_to_seqs 
# -----------------------------------------------------------
# module variables requiring initialisation before module use
# -----------------------------------------------------------
first_severity        = None
number_of_trials      = None

# -------------------------
# module-specific constants
# -------------------------

# These are constants used throughout the module, which are not required
# elsewhere, and are unlikely to require any adjustments, so there is little
# need to include them in the main CONFIG.py file
FONT              = 'ubuntumono'   # previously: Arial
BACKGROUND_COLOUR = ANSI.GRAY
RESPONSE_TIMEOUT  = 5000  # in milliseconds

######################
## Module functions ##
######################

def entropy(x, bins=None):
    '''
    Returns the entropy of the empiricial distribution of x
    '''

    N = x.shape

    if bins is None:   counts = numpy.bincount( x )
    else           :   counts = numpy.histogram( x, bins = bins )[ 0 ]   # Counts, probs

    p = counts[ numpy.nonzero( counts ) ] / N    # log(0)
    H = -numpy.dot( p, numpy.log2( p ) )

    return H

def calculate_agent_response_and_confidence(model, city_severity, trial_no, resource_remaining):
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

def calculate_agent_response_and_confidence_alternative(model, city_severity, trial_no, resource_remaining):

    r_remaining = resource_remaining
    t_no = trial_no
    c_severity = city_severity

    action = model( tf.Variable(c_severity, dtype=tf.float32), tf.Variable(t_no, dtype=tf.float32), tf.Variable(r_remaining, dtype=tf.float32) )

    response, confidence, rt_hold, rt_release = agent_meta_cognitive(action, MAX_ALLOCATABLE_RESOURCES+1, resource_remaining,RESPONSE_TIMEOUT)


    return confidence, response, rt_hold, rt_release

def rl_agent_meta_cognitive(options, resources_left, response_timeout):

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

