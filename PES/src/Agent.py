"""
PES - Pandemic Experiment Scenario

Agent.py 

Utility functions for the RL Agent to generate responses and confidence ratings.
"""

import numpy
import random

from .. import MAX_ALLOCATABLE_RESOURCES  


AGENT_NOISE_VARIANCE = 2.0

randominstance = random.Random()

def initseed(seed, noise):
    randominstance.seed( seed )
    global AGENT_NOISE_VARIANCE 
    AGENT_NOISE_VARIANCE = noise 


def agent_meta_cognitive(action, output_value_range, resources_left, response_timeout):
    # What are we doing here is mapping the distance to the boundary of each decision, as
    # the level of confidence that the NN-Agent has on this decision.
    # This can be done because the output of the NN-Agent is continuos.
    centers = [r for r in range(output_value_range)]
    center = numpy.asarray( centers, dtype=numpy.float32)

    closer = center-action
    closer = numpy.abs( closer )
    closervalue = center[numpy.argmin(closer)]

    action = numpy.clip(action, 1, resources_left)

    f = lambda x: x * (-2) + 1

    distance = numpy.clip(numpy.abs(closervalue-action),0,0.5)

    confidence = f( distance )

    response = int(round(action))

    mu, sigma = int(distance * 10), 3

    rt_hold = numpy.random.normal(mu, sigma, 1)[0]
    rt_release = rt_hold + numpy.random.normal(mu, 1, 1)[0]

    rt_hold = numpy.clip( rt_hold, 0, response_timeout/1000.0)
    rt_release = numpy.clip( rt_release, 0, response_timeout/1000.0)

    return response, confidence, rt_hold, rt_release

def adjust_response_decay( resp, decay, resources_left):
    
    rand = numpy.random.random(1)

    COIN_FLIPPING       = 1
    GAUSSIAN_VARIANCE   = 2
    NON_HUMANISED       = 3

    modality = GAUSSIAN_VARIANCE 

    if (modality == COIN_FLIPPING):
        if (rand > decay):
            resp = randominstance.randrange(0,MAX_ALLOCATABLE_RESOURCES+1)
            resp = numpy.clip( resp, 0, resources_left)

    if (modality == GAUSSIAN_VARIANCE):
        variance = 1.0-decay
        variance = int(variance*AGENT_NOISE_VARIANCE)
        #if (variance>1):
        delta = randominstance.gauss(resp, variance)
        resp = numpy.clip( delta, 0, min(resources_left, MAX_ALLOCATABLE_RESOURCES))
        #else:
            # 0 is a valid response (delegate response)
        #    delta = randominstance.uniform(-1,1+1)
        #    resp = numpy.clip( resp+delta, 0, min(resources_left, MAX_ALLOCATABLE_RESOURCES))
    
    return resp 

def boltzmann_decay( global_seq_no ):
    # 75 and 4.0 here are parameters that push the shape of the boltzmann decay (as function of temperature, the sequence number in this case).
    return numpy.exp( ( -75.0) / (4.0 * global_seq_no ) )





