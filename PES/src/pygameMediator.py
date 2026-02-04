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
import pygame
import sys
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

    Q = numpy.load(os.path.join(INPUTS_PATH,'q.npy'))
    rewards = numpy.load(os.path.join(INPUTS_PATH,'rewards.npy'))

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

    print( int(resources_left.numpy()) if hasattr(resources_left, 'numpy') else resources_left )
    print( int(city_number.numpy()) if hasattr(city_number, 'numpy') else city_number )
    print( sever )
  # Calculate the response and confidence feeding the NN with noisy inputs, getting the mean and entropy from the responses.
    resp, confidence, rt_hold, rt_release = rl_agent_meta_cognitive(Q[int(resources_left), int(city_number),int(sever)],resources_left,RESPONSE_TIMEOUT)

    movement = []

    return confidence, resp, rt_hold, rt_release, movement

