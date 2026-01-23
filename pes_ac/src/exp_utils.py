"""
pes_ac — Pandemic Experiment Scenario (Advantage Actor-Critic)

Utility functions module providing essential functionality for experiment execution,
severity calculations, resource allocation aggregation, and performance metrics.

Key Components
--------------
• Severity Calculations: Functions to compute and update pandemic severity based on
  resource allocations and time progression
• Performance Metrics: Normalized performance calculation comparing actual results
  against best/worst case allocations
• Confidence Aggregation: Weighted mean/median calculations for combining decisions
  from multiple participants
• Analysis Tools: Utilities for data transformation, sampling, and experiment control

Main Functions
---------------
• calculate_normalised_final_severity_performance_metric: Compute normalized performance
• get_updated_severity: Calculate new severity given resource allocations
• get_confidence_weighted_mean: Aggregate decisions using confidence-weighted mean
• get_confidence_weighted_median: Aggregate decisions using confidence-weighted median
• get_percent_deviation: Measure deviation from optimal outcomes
• random_severity_generator: Generate random initial conditions
• chain_ops: Pipe operations through sequential functions
• exit_experiment_gracefully: Clean shutdown with resource cleanup
"""

##########################
##  Imports externos    ##
##########################
import os
import numpy
import scipy.stats as ss
from statsmodels.stats.weightstats import DescrStatsW as WeightedStats

##########################
##  Imports internos    ##
##########################
from .. import (INPUTS_PATH, MAX_ALLOCATABLE_RESOURCES, MIN_ALLOCATABLE_RESOURCES,
                RESPONSE_MULTIPLIER, SEQ_LENGTHS_FILE, SEVERITY_MULTIPLIER)


def get_sequence_severity_from_allocations(Allocations, InitialSeverities):
    """Compute the total severity of a full sequence given allocations and initial severities."""
    return numpy.sum(get_array_of_sequence_severities_from_allocations(Allocations, InitialSeverities))


def calculate_normalised_final_severity_performance_metric(SeveritiesFromSequence, InitialSequenceSeverities):
    """
    Calculate normalized performance metric comparing actual severity outcome to best/worst case scenarios.

    The metric ranges from 0 (worst case performance) to 1 (best case performance), representing
    how well the participant/agent performed relative to the theoretical bounds.

    Parameters
    ----------
    SeveritiesFromSequence : array-like
        Final severity values achieved for each trial in the sequence
    InitialSequenceSeverities : array-like
        Initial severity values for each trial in the sequence

    Returns
    -------
    tuple
        - Performance (float): Normalized performance metric (0-1)
        - WorstCaseSequenceSeverity (float): Sum of severities if no resources allocated
        - BestCaseSequenceSeverity (float): Sum of severities if max resources allocated
    """

    FinalSequenceSeverity = numpy.sum(SeveritiesFromSequence)
    BestCaseAllocations = numpy.full_like(SeveritiesFromSequence, MAX_ALLOCATABLE_RESOURCES)
    WorstCaseAllocations = numpy.full_like(SeveritiesFromSequence, MIN_ALLOCATABLE_RESOURCES)
    BestCaseSequenceSeverity = get_sequence_severity_from_allocations(BestCaseAllocations, InitialSequenceSeverities)
    WorstCaseSequenceSeverity = get_sequence_severity_from_allocations(WorstCaseAllocations, InitialSequenceSeverities)
    Performance = (WorstCaseSequenceSeverity - FinalSequenceSeverity) / \
        (WorstCaseSequenceSeverity - BestCaseSequenceSeverity)

    return Performance, WorstCaseSequenceSeverity, BestCaseSequenceSeverity


def get_array_of_sequence_severities_from_allocations(Allocations, InitialSeverities):
    """
    Calculate severity progression through a sequence given resource allocations.

    Simulates the pandemic scenario where severity evolves over time as resources
    are sequentially allocated to trials. Each trial's final severity depends on
    initial severity, resource allocation, and the combined effect of previous allocations.

    The severity update formula for each trial is:
        new_severity = max(0, SEVERITY_MULTIPLIER * initial - RESPONSE_MULTIPLIER * allocated)

    Parameters
    ----------
    Allocations : array-like
        Resource allocation amounts for each trial in sequence (0-10)
    InitialSeverities : array-like
        Initial severity value for each trial

    Returns
    -------
    list[float]
        Final severity values for each trial after resource allocation effects

    Examples
    --------
    City damage progression (Pandemic damage model):

    Initial severities: [3, 4, 8]
    Allocations: [5, 6, 4]

    This example shows how city damage is performed by sequentially applying
    resources. The formula at each step is:
        new_severity = max(0, SEVERITY_MULTIPLIER * severity - RESPONSE_MULTIPLIER * allocation)

    Severity evolution across trials:

    City 1          City 2          City 3
    ------          ------          ------
    3
    2.6 (alloc=5)   4
    2.12 (alloc=5)  3.6 (alloc=6)   8
    1.54 (alloc=5)  3.12 (alloc=6)  8.8 (alloc=4)   [Final result]

    Step-by-step explanation:
    • After Trial 0: City 1 evolves from 3 → 2.6 (allocation applied)
    • After Trial 1: City 1 continues 2.6 → 2.12; City 2 starts 4 → 3.6
    • After Trial 2: All cities update; City 3 enters 8 → 8.8

    Final result: [1.54, 3.12, 8.8]

    Note: This example uses SEVERITY_MULTIPLIER=1.2 and RESPONSE_MULTIPLIER=0.2
    for illustration. Actual values depend on PANDEMIC_PARAMETER configuration
    (typically 0.4, giving multipliers 1.4 and 0.4)

    Notes
    -----
    • Severities are clipped to minimum of 0
    • The effect of allocations compounds across the sequence as each trial's outcome
      influences subsequent trials
    • Higher allocations reduce severity more effectively
    • If new_severity < 0, it is clipped to 0 (pandemic eliminated)
    """

    NumTrialsInSequence = len(InitialSeverities)
    severities = []
    resources = []

    for Trial in range(NumTrialsInSequence):

        severities . append(InitialSeverities[Trial])
        resources  . append(Allocations[Trial])

        severities = get_updated_severity(len(severities), resources, severities)

    return severities.copy()


def exit_experiment_gracefully(Message, Filehandles, MovementData, LogUtils, _PygameMediator):
    """
    Clean shutdown of experiment, closing all resources and logging final information.

    Gracefully terminates the experiment by closing files, pygame window, and
    performing final logging operations. Avoids circular import issues by
    accepting LogUtils and PygameMediator as arguments.

    Parameters
    ----------
    Message : str
        Final message to log before exit
    Filehandles : list of file
        Open file handles to close
    MovementData : tuple
        Movement tracking data to save
    LogUtils : module
        Log utilities module for final logging
    PygameMediator : module
        Pygame mediator module (currently unused for AC-Agent mode)
    """

    LogUtils.tee()
    LogUtils.tee(Message)

    numpy.save(*MovementData)
    for Filehandle in Filehandles:
        if Filehandle is not None:
            Filehandle.close()
    LogUtils.close_consolelog_filehandle()


def get_updated_severity(no_of_cities, resource_allocated, initial_severity) -> list[float]:
    """
    Update severity for existing cities given allocated resources.

    Updates the severity of each city based on the resources allocated to it,
    using the pandemic damage formula. This reflects how resource allocation
    reduces the growth/intensity of the pandemic in each location.

    Parameters
    ----------
    no_of_cities : int
        Number of cities/trials to update severity for
    resource_allocated : array-like
        Resources allocated to each city (0 to MAX_ALLOCATABLE_RESOURCES)
    initial_severity : array-like
        Current severity values for each city

    Returns
    -------
    list[float]
        Updated severity values, clipped to minimum of 0

    Notes
    -----
    Uses the formula: new_severity = max(0, SEVERITY_MULTIPLIER * initial - RESPONSE_MULTIPLIER * resources)
    """

    UpdatedSeverity_list = []

    for c in range(no_of_cities):

        InitialSeverityInCity = initial_severity[c]
        ResourcesAllocatedToCity = resource_allocated[c]
        NewSeverityInCity = SEVERITY_MULTIPLIER * InitialSeverityInCity - RESPONSE_MULTIPLIER * ResourcesAllocatedToCity
        NewSeverityInCity = max(NewSeverityInCity, 0)

        UpdatedSeverity_list.append(NewSeverityInCity)

    return UpdatedSeverity_list


def random_severity_generator(number_of_runs, lower_limit, upper_limit):
    """
    Generate random initial severity values following a custom probability distribution.

    Creates a distribution of severity values that can be used to randomly sample
    initial conditions for trials. Uses a normal distribution to weight the
    probability of selecting different severity levels.

    Parameters
    ----------
    number_of_runs : int
        Number of random severity values to generate
    lower_limit : int
        Minimum severity value to consider
    upper_limit : int
        Maximum severity value to consider

    Returns
    -------
    ndarray
        Array of random severity values
    """

    x = numpy.arange(lower_limit, upper_limit)
    xU, xL = x + 0.5, x - 0.5

    prob = ss.norm.cdf(xU, scale=100) - ss.norm.cdf(xL, scale=100)
    prob = prob / prob.sum()

    numpy.random.seed(3)

    nums = numpy.random.choice(x, size=number_of_runs, p=prob)

    return nums


def next_seq_length(index, seq_per_block):
    """
    Retrieve sequence lengths for the next block of sequences.

    Parameters
    ----------
    index : int
        Global sequence index to start from
    seq_per_block : int
        Number of sequences to retrieve

    Returns
    -------
    ndarray
        Array of sequence lengths for the next seq_per_block sequences
    """
    SequenceLengthsCsv = os.path.join(INPUTS_PATH, SEQ_LENGTHS_FILE)
    s = numpy.loadtxt(SequenceLengthsCsv, delimiter=',')
    sequence = s[index: index + seq_per_block]
    return sequence


def sampler(samples, sum_to, range_list, rn=100):
    """
    Distribute trials across sequences in a block with randomized sampling.

    Generates a random distribution of trial counts across multiple sequences
    such that the total number of trials in a block sums to a target value.

    Parameters
    ----------
    samples : int
        Number of sequences in the block (typically NUM_SEQUENCES = 8)
    sum_to : int
        Target total number of trials (typically TOTAL_NUM_TRIALS_IN_BLOCK = 45)
    range_list : list[int]
        [min_trials, max_trials] - Bounds on trials per sequence
    rn : int, optional
        Random seed for reproducibility. Default is 100.

    Returns
    -------
    ndarray
        Array of trial counts for each sequence, summing to ``sum_to``

    Raises
    ------
    ValueError
        If the specified range constraints make it impossible to reach ``sum_to``
    """

    assert range_list[0] < range_list[1], "Range should be a list, the first element of which is smaller than the second"

    numpy.random.seed(rn)

    arr = numpy.random.rand(samples)
    sum_arr = sum(arr)

    new_arr = numpy.array([
                          int(item / sum_arr * sum_to)
                          if (range_list[0] < int(item / sum_arr * sum_to) < range_list[1])
                          else numpy.random.choice(range(range_list[0], range_list[1] + 1))

                          for item in arr
                          ])

    difference = sum(new_arr) - sum_to

    if len(samples) * range_list[1] < sum_to or len(samples) * range_list[0] > sum_to:
        raise ValueError(
            'The specified number of sequences is such that the desired TOTAL_NUM_TRIALS_IN_BLOCK value can never be reached')

    while difference != 0:

        if difference < 0:
            for idx in numpy.random.choice(range(len(new_arr)), abs(difference)):
                if new_arr[idx] != range_list[1]:
                    new_arr[idx] += 1

        if difference > 0:
            for idx in numpy.random.choice(range(len(new_arr)), abs(difference)):
                if new_arr[idx] != 0 and new_arr[idx] != range_list[0]:
                    new_arr[idx] -= 1

        difference = sum(new_arr) - sum_to

    return new_arr


def get_confidence_weighted_mean(all_messages, first_severity, _AbsoluteSequenceIndex, AbsoluteTrialCount):
    """
    Aggregate decisions from multiple participants using confidence-weighted mean.

    Parameters
    ----------
    all_messages : array-like
        3D array of shape (num_participants, num_trials, 2)
    first_severity : array-like
        Initial severity values for the sequence
    AbsoluteSequenceIndex : int
        Index of the current sequence
    AbsoluteTrialCount : int
        Total number of trials completed so far

    Returns
    -------
    tuple
        - AggregatedAllocations (ndarray): Confidence-weighted mean allocations
        - SeverityFromAggregate (ndarray): Resulting severities from aggregated decisions
    """

    NumTrials = numpy.shape(all_messages)[1]
    AggregatedAllocations = []

    for t in range(NumTrials):

        TrialResponses = numpy.array(all_messages)[:, t, 0]
        TrialConfidences = numpy.array(all_messages)[:, t, 1]

        OriginalTrialResponses = TrialResponses.copy()

        TrialResponses = TrialResponses[TrialConfidences != -1]
        TrialConfidences = TrialConfidences[TrialConfidences != -1]

        if (numpy.sum(TrialConfidences) == 0):
            TrialConfidences[:] = 1.0

        if numpy.size(TrialConfidences) == 0:
            ConfidenceWeightedMean = numpy.mean(OriginalTrialResponses)
        else:
            ConfidenceWeightedMean = numpy.average(TrialResponses, weights=TrialConfidences)

        AggregatedAllocations.append(ConfidenceWeightedMean)

    AggregatedAllocations = numpy.array(AggregatedAllocations)
    AggregatedAllocations = numpy.round(AggregatedAllocations)

    SeverityFromAggregate = get_array_of_sequence_severities_from_allocations(
        AggregatedAllocations, first_severity[AbsoluteTrialCount - NumTrials: AbsoluteTrialCount].copy())

    return AggregatedAllocations, SeverityFromAggregate


def get_confidence_weighted_mode():
    """
    Aggregate decisions using confidence-weighted mode (NOT IMPLEMENTED).

    Raises
    ------
    NotImplementedError
        This method is not yet implemented.
    """
    raise NotImplementedError


def get_confidence_weighted_median(all_messages, first_severity, _AbsoluteSequenceIndex, AbsoluteTrialCount):
    """
    Aggregate decisions from multiple participants using confidence-weighted median.

    Parameters
    ----------
    all_messages : array-like
        3D array of shape (num_participants, num_trials, 2)
    first_severity : array-like
        Initial severity values for the sequence
    AbsoluteSequenceIndex : int
        Index of the current sequence
    AbsoluteTrialCount : int
        Total number of trials completed so far

    Returns
    -------
    tuple
        - AggregatedAllocations (ndarray): Confidence-weighted median allocations
        - SeverityFromAggregate (ndarray): Resulting severities from aggregated decisions
    """

    NumTrials = numpy.shape(all_messages)[1]
    AggregatedAllocations = []

    for t in range(NumTrials):

        TrialResponses = numpy.array(all_messages)[:, t, 0]
        TrialConfidences = numpy.array(all_messages)[:, t, 1]

        OriginalTrialResponses = TrialResponses.copy()

        TrialResponses = TrialResponses[TrialConfidences != -1]
        TrialConfidences = TrialConfidences[TrialConfidences != -1]

        if numpy.size(TrialConfidences) == 0:
            ConfidenceWeightedMedian = numpy.median(OriginalTrialResponses)
        else:

            if numpy.size(TrialResponses) == 1:
                TrialResponses = numpy.repeat(TrialResponses, 2)
                TrialConfidences = numpy.repeat(TrialConfidences, 2)

            ConfidenceWeightedMedian = WeightedStats(
                data=TrialResponses,
                weights=TrialConfidences
            ).quantile(
                probs=[0.5],
                return_pandas=False
            )[0]

        AggregatedAllocations.append(ConfidenceWeightedMedian)

    AggregatedAllocations = numpy.array(AggregatedAllocations)
    AggregatedAllocations = numpy.round(AggregatedAllocations)

    SeverityFromAggregate = get_array_of_sequence_severities_from_allocations(
        AggregatedAllocations, first_severity[AbsoluteTrialCount - NumTrials: AbsoluteTrialCount].copy())

    return AggregatedAllocations, SeverityFromAggregate
