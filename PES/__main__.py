"""
This is the package's main module; it defines a main function, acting as the main entry point / execution logic of the
whole package at a high-level, delegating implementation details to the package's specialised modules in a top-down
manner.
"""


# -------------------------------------------------------
# Ensure we're running in package mode before proceeding!
# -------------------------------------------------------

assert len( __package__ ) > 0, """

The '__main__' module does not seem to have been run in the context of a runnable package.
Did you forget to add the '-m' flag?

Usage: python3 -m <packagename>

"""


# ----------------
# external imports
# ----------------

import os
import pygame
import socket
import datetime
import numpy
import pickle
import sys
import getpass


# ----------------
# internal imports
# ----------------

# Import project variables (provided via __init__)
from . import ALLOCATION_TYPE
from . import ANSI
from . import AVAILABLE_RESOURCES_PER_SEQUENCE
from . import BLOCK_MODE_INDICES
from . import DEBUG
from . import INITIAL_SEVERITY_FILE
from . import INIT_NO_OF_CITIES
from . import INPUTS_PATH
from . import MAX_INIT_RESOURCES
from . import MAX_INIT_SEVERITY
from . import MIN_INIT_RESOURCES
from . import MIN_INIT_SEVERITY
from . import NUM_ATTEMPTS_TO_ASSIGN_SEQ
from . import NUM_BLOCKS
from . import NUM_MIN_TRIALS, NUM_MAX_TRIALS
from . import NUM_PREDEFINED_CITY_COORDS
from . import NUM_SEQUENCES
from . import OUTPUTS_PATH, OUTPUT_FILE_PREFIX
from . import RANDOM_INITIAL_SEVERITY
from . import RESOURCES_PATH
from . import SAVE_INITIAL_SEVERITY_TO_FILE
from . import SAVE_RESULTS
from . import STARTING_BLOCK_INDEX
from . import STARTING_SEQ_INDEX
from . import TOTAL_NUM_TRIALS_IN_BLOCK
from . import USE_FIXED_BLOCK_SEQUENCES
from . import VERBOSE
from . import WHITE, YELLOW, BLACK, DARK_RED, DARK_CYAN, DARK_GREEN, GREEN, RED, GRAY

# Import internal modules and functions
from . import printinfo, printstatus
from . src import exp_utils
from . src import log_utils
from . src import pygameMediator

# Import random_severity_generator utility
from .src.exp_utils import random_severity_generator


# -----------------------------------------
# Initialization of Module-global variables
# -----------------------------------------

# Note: In RL-Agent mode, aggregation is not used (single agent)
call_nominated_aggregator = None  # Not used in RL-Agent autonomous mode

Responses_filehandle = None

Starting_sequence_cumulative_index = STARTING_BLOCK_INDEX * NUM_SEQUENCES + STARTING_SEQ_INDEX



########
### Main
########

def main():
    printinfo( "--- Entering main execution block ---\n" )

    global Responses_filehandle

    init_circle_radius  = 20


    # -----------------------------------
    # Initialise experimental environment
    # -----------------------------------

  # Set Subject id automatically in 'minimum three digit' format (e.g. '001')
    MySubjectId = exp_utils.create_subject_id()
    MySubjectId += '_TEST'  # RL-Agent mode always uses TEST suffix

  # Create a log file to tee console output to (see: exp3f.src.log_utils.tee)
    log_utils.create_ConsoleLog_filehandle_singleton( MySubjectId )

  # In RL-Agent mode, BioSemi is not connected
    # No need to initialize BioSemi for autonomous RL training

  # RL-Agent mode: skip pygame window initialization to avoid graphics overhead
    if VERBOSE:   printinfo( "__main__: Skipping pygame initialization for RL-Agent mode" )


    # -------------------------------------------------------
    # Single Agent Setup (Lobby removed for single agent)
    # -------------------------------------------------------



    if VERBOSE:   printstatus( 'Done', ANSI.GREEN )

    if VERBOSE:   printinfo( "__main__: Active player:" )
    log_utils.tee( f'{ANSI.ORANGE} - Player {MySubjectId}{ANSI.RESET} <-- active player' )

    NumPlayers = 1


    # ------------------------------------
    # Save experimental parameters to file
    # ------------------------------------

    if SAVE_RESULTS:

        experiment_date = datetime.date.today().strftime( "%d/%m/%Y" )

      # For RL-Agent, use default values
        age     = 'N/A'
        gender  = 'N/A'
        hand    = 'N/A'

        SubjectInfo_filename   = os.path.join( OUTPUTS_PATH, f'{OUTPUT_FILE_PREFIX}info_{MySubjectId}.txt' )
        SubjectInfo_filehandle = open( SubjectInfo_filename, 'w')

        SubjectInfo_filehandle.write(  '#Age, Gender, Handedness, ExperimentDate\n')
        SubjectInfo_filehandle.write( f'{age:>4}, {gender:>6}, {hand:>10}, {experiment_date:>14}\n' )
        SubjectInfo_filehandle.close()

        Responses_filename = os.path.join(
            OUTPUTS_PATH,
            f'{OUTPUT_FILE_PREFIX}responses_{ MySubjectId }.txt'
        )

        Responses_filehandle = open( Responses_filename, 'w' )

        if VERBOSE:   log_utils.print_settings(
            Date      = str( datetime.datetime.now( tz = datetime.timezone.utc ) ),
            SubjectId = MySubjectId
        )


    # ----------------------------------------------------------------------------
    # Initialize necessary structures for experiment over all blocks and sequences (i.e. chosen maps, no. trials etc)
    # ----------------------------------------------------------------------------

    response               = {}
    hold_response_times    = {}
    release_response_times = {}
    confidence             = {}
    total_movement         = {}


    # Always load coordinates, but load_image() will skip pygame rendering for RL-Agent
    images, all_coordinates = pygameMediator.load_image()

    NumTrials__blocks_x_sequences__2darray = numpy.zeros( (NUM_BLOCKS, NUM_SEQUENCES) )   # each slot contains the
                                                                                          # number of trials (i.e.
                                                                                          # cities) that will be
                                                                                          # presented for a particular
                                                                                          # 'block,sequence' pair. (Note
                                                                                          # that these numbers do not
                                                                                          # include the 'initial cities
                                                                                          # on map'; only cities for
                                                                                          # which the player will have
                                                                                          # to make an allocation
                                                                                          # decision)

    MapIndices__blocks_x_sequences__2darray      = numpy.zeros( (NUM_BLOCKS, NUM_SEQUENCES) )   # each slot contains a
                                                                                                # map index (from 0 to 8
                                                                                                # which background image
                                                                                                # file will be selected
                                                                                                # for a particular
                                                                                                # 'block,sequence' pair

    NumTrialsPerSequence_list = []   # this list contains exactly the same information as the
                                     # NumTrials__blocks_x_sequences__2darray array, but as a flat list of integers,
                                     # rather than as an nblocks-by-nsequences array.

    CoordinateIndicesPerTrial__nestedDict_by_block_and_sequence = {}   # This is a 'dictionary of dictionaries' with the
                                                                       # following format:
                                                                       #  - Outer Key: An index number referring to a
                                                                       #    'block'
                                                                       #  - Inner Key (i.e. per block): An index number
                                                                       #    referring to a sequence
                                                                       #  - Typical entry:
                                                                       #    CoordinateIndicesPerTrial__nestedDict_by_block_and_sequence[
                                                                       #    blk ][ seq ] = L, where L is a list of ints,
                                                                       #    such that len(L) is equal to the number of
                                                                       #    trials for that sequence, and each entry is
                                                                       #    an integer from 0 to 24 (inclusive),
                                                                       #    corresponding to one of the predetermined
                                                                       #    city coordinates obtained from file. In
                                                                       #    other words this dict defined which cities
                                                                       #    will be displayed for a particular sequence
                                                                       # (Note: It is unclear why a dictionary structure
                                                                       # is used instead of a plain list)

    if VERBOSE:   printinfo( f'__main__: Passing {ANSI.ORANGE}NumTrials__blocks_x_sequences__2darray{ANSI.BLUE} to {ANSI.GREEN}pygameMediator{ANSI.BLUE} module' )
    pygameMediator.number_of_trials = NumTrials__blocks_x_sequences__2darray


    for blk in range( NUM_BLOCKS ):

        CoordinateIndicesPerTrial__nestedDict_by_block_and_sequence[ blk ] = {}


        for seq in range( NUM_SEQUENCES ):

            CoordinateIndicesPerTrial__nestedDict_by_block_and_sequence[ blk ][ seq ] = []

            counter_seq = NUM_ATTEMPTS_TO_ASSIGN_SEQ   # This constant dictates how many attempts at assignment will be
                                                       # made per sequence. Each attempt overrides the previous one,
                                                       # unless the same map already exists in the block, or the default
                                                       # map '0' is selected. If no valid attempts occur, map '0' is used.

            numpy.random.seed( 100 + blk )


            # Assign a random map index to each sequence, excluding duplicates within the block.
            # The process is repeated counter_seq times.

            while counter_seq > 0:

                b = numpy.random.randint(0, 9)

                if b not in MapIndices__blocks_x_sequences__2darray[ blk, : ]:
                    MapIndices__blocks_x_sequences__2darray[ blk, seq ] = b

                counter_seq -= 1


            if (USE_FIXED_BLOCK_SEQUENCES):
                NumTrials__blocks_x_sequences__2darray[ blk, : ] = exp_utils.next_seq_length(
                                                                   blk * NUM_SEQUENCES,
                                                                   NUM_SEQUENCES
                                                                 )
            else:
                NumTrials__blocks_x_sequences__2darray[ blk, : ] = exp_utils.sampler(
                                                                   NUM_SEQUENCES,
                                                                   TOTAL_NUM_TRIALS_IN_BLOCK,
                                                                   [NUM_MIN_TRIALS, NUM_MAX_TRIALS],
                                                                   rn = blk
                                                                 )

            counter_trial = INIT_NO_OF_CITIES + NumTrials__blocks_x_sequences__2darray[ blk, seq ]

            NumTrialsPerSequence_list.append( NumTrials__blocks_x_sequences__2darray[ blk, seq ] )


            while counter_trial > 0:

                c = numpy.random.randint( 0, NUM_PREDEFINED_CITY_COORDS )

                if c not in CoordinateIndicesPerTrial__nestedDict_by_block_and_sequence[ blk ][ seq ]:
                    CoordinateIndicesPerTrial__nestedDict_by_block_and_sequence[ blk ][ seq ].append( c )
                    counter_trial -= 1

        #
        ### END OF for seq in range( NUM_SEQUENCES ):

    #
    ### END OF `for blk in range( NUM_BLOCKS )`


    # ----------------------------
    # Start experimental procedure
    # ----------------------------


  ## Generate initial severity for all trials in the experiment (values 2-8 inclusive).
    # Configuration: use prescripted values from CSV file or generate randomly based on RANDOM_INITIAL_SEVERITY flag

    InitialSeverityCsv = os.path.join( INPUTS_PATH, INITIAL_SEVERITY_FILE)

    if RANDOM_INITIAL_SEVERITY:
        first_severity = random_severity_generator( int( sum( sum( NumTrials__blocks_x_sequences__2darray ) ) ), 2, 9 )
        if SAVE_INITIAL_SEVERITY_TO_FILE:   numpy.savetxt( InitialSeverityCsv, first_severity, fmt = '%d', delimiter = ',' )
    else:
        first_severity = numpy.loadtxt( InitialSeverityCsv )
        first_severity = first_severity[ 0 : int( sum( sum( NumTrials__blocks_x_sequences__2darray ) ) ) ]

  # Passing to pygameMediator too, as provide_agent_response requires this to be set
    if VERBOSE: printinfo( f'__main__: Passing {ANSI.ORANGE}first_severity{ANSI.BLUE} to {ANSI.GREEN}pygameMediator{ANSI.BLUE} module' )
    pygameMediator.first_severity = first_severity

  ## Load optimal resource allocation and associated final severity (for reference/comparison)
    # These values are obtained from neural network training and used for performance benchmarking
    optimal_resources_allocated = numpy.load( os.path.join( INPUTS_PATH, 'optimal_resources.npy' ), allow_pickle = True )
    optimal_final_severity      = numpy.load( os.path.join( INPUTS_PATH, 'optimal_severity.npy'  ), allow_pickle = True )


  # Experiment Sessions
    total_number_of_sequences = NUM_BLOCKS * NUM_SEQUENCES


  # Initialise output (i.e. responses file) by creating a header
    Responses_filehandle.write( "#InitialSeverity, Response, Confidence, PressEvent_seconds, ReleaseEvent_seconds\n" )
    Responses_filehandle.flush()

    AbsoluteTrialCount = 0
    AbsoluteSequenceIndex = 0
    ExperimentStartTime = datetime.datetime.now( tz = datetime.timezone.utc )


  # Start the actual experiment.
    if VERBOSE:   print ()
    printinfo( "--- Starting experiment ---" )
    print()


  # Initialise Performance lists
    MyPerformances  = []  # Agent performance per sequence
    AllPerformances = [ [] for i in range( NumPlayers + 1) ]  # Index 0: agent, Index 1: aggregated
    # TrustRatings not used in RL-Agent single-agent mode
    TrustRatings    = numpy.full( (1, max(1, NumPlayers-1)), 100 )   # Placeholder for compatibility

  # Start experimental blocks
    for CurrentBlockIndex in range( NUM_BLOCKS ):

      # Jump to the block and sequence specified in the config, if present
        if CurrentBlockIndex < STARTING_BLOCK_INDEX:
            AbsoluteTrialCount    += int( NumTrials__blocks_x_sequences__2darray[ CurrentBlockIndex, : ].sum() )
            AbsoluteSequenceIndex += NUM_SEQUENCES
            continue
        else:
            pass   # We have (already) reached the correct starting block

        log_utils.tee(
             f'Current session (i.e. block): {CurrentBlockIndex+1} of {NUM_BLOCKS}'
           )


      # Set the gameplay 'mode' for the experiment
        if   CurrentBlockIndex in BLOCK_MODE_INDICES[ 'Joint' ]:   exp_utils.toggle_experiment_mode( 'Joint' )
        elif CurrentBlockIndex in BLOCK_MODE_INDICES[ 'Solo'  ]:   exp_utils.toggle_experiment_mode( 'Solo' )
        else: raise ValueError( "Block index not found in either 'Joint' or 'Solo' modes."  )


      # Obtain current time in experiment
        TimeElapsed        = (datetime.datetime.now( tz = datetime.timezone.utc ) - ExperimentStartTime).total_seconds()
        HoursElapsed       = max( 0, TimeElapsed / 60 / 60 // 1 )
        MinutesElapsed     = max( 0, (TimeElapsed - HoursElapsed * 60 * 60) / 60 // 1 )
        HoursElapsed_str   = f'{HoursElapsed:.0f} hours, and ' if HoursElapsed    > 1 else '1 hour, and ' if HoursElapsed   == 1 else ''
        MinutesElapsed_str = f'{MinutesElapsed:.0f} minutes'   if MinutesElapsed != 1 else '1 minute'     if MinutesElapsed == 1 else ''
        TimeElapsed_str    = f'Time elapsed: {HoursElapsed_str}{MinutesElapsed_str}'

      # Calculate time left based on speed so far and sequences remaining
        if AbsoluteSequenceIndex > Starting_sequence_cumulative_index:
            TimePerSeq      = TimeElapsed / (AbsoluteSequenceIndex - Starting_sequence_cumulative_index)
            SeqsLeft        = total_number_of_sequences - AbsoluteSequenceIndex
            TimeLeft        = SeqsLeft * TimePerSeq
            HoursLeft       = max( 0, TimeLeft / 60 / 60 // 1 )
            MinutesLeft     = max( 0, (TimeLeft - HoursLeft * 60 * 60) / 60 // 1 )
            HoursLeft_str   = f'{HoursLeft:.0f} hours, and ' if HoursLeft    > 1 else '1 hour, and ' if HoursLeft   == 1 else ''
            MinutesLeft_str = f'{MinutesLeft:.0f} minutes'   if MinutesLeft != 1 else '1 minute'     if MinutesLeft == 1 else ''
            TimeLeft_str    = f'Estimated time left: {HoursLeft_str}{MinutesLeft_str}'
            EndOfBlock_str  = f'End of block {CurrentBlockIndex}   ({NUM_BLOCKS - CurrentBlockIndex} blocks to go)\nTake rest if needed.'
        else:
            EndOfBlock_str  = ' '
            TimeLeft_str    = ' '

      # Report timings in console
        log_utils.tee( f'{TimeElapsed_str} -- {TimeLeft_str}' )

      # Determine whether resources will be shared or not during this block
        ResourcesType = 'will be SHARED among all players!' if exp_utils.AllocationType == 'shared' else 'will NOT be shared among players.'

      # Continue to next block without GUI messages


        CurrentBlockMapIndices = MapIndices__blocks_x_sequences__2darray[ CurrentBlockIndex, : ]

        response              [ CurrentBlockIndex ] = {}
        hold_response_times   [ CurrentBlockIndex ] = {}
        release_response_times[ CurrentBlockIndex ] = {}
        confidence            [ CurrentBlockIndex ] = {}
        total_movement        [ CurrentBlockIndex ] = {}


      # Start interactive processing of sequences
        for CurrentSequenceIndex, CurrentSequenceMapIndex in enumerate( CurrentBlockMapIndices ):

            if CurrentBlockIndex == STARTING_BLOCK_INDEX and CurrentSequenceIndex < STARTING_SEQ_INDEX:
                AbsoluteTrialCount    += int( NumTrials__blocks_x_sequences__2darray[ CurrentBlockIndex, CurrentSequenceIndex ] )
                AbsoluteSequenceIndex += 1
                continue

            resources_to_allocate = AVAILABLE_RESOURCES_PER_SEQUENCE

            log_utils.tee( )
            log_utils.tee(
                 f'Current sequence: {AbsoluteSequenceIndex+1} of {total_number_of_sequences}',
                 f'({CurrentSequenceIndex+1} of {NUM_SEQUENCES} in block),',
                 f'using map No. {int(CurrentSequenceMapIndex)}'
               )


            # Starting sequence without showing message to human
            ESC_was_pressed = False

          # Handle "exit game" requests gracefully
            if ESC_was_pressed:
                exp_utils.exit_experiment_gracefully( Message        = f"--- "
                                                                       f"Experiment interrupted "
                                                                       f"at BlockIndex={ CurrentBlockIndex    }, "
                                                                       f"SequenceIndex={ CurrentSequenceIndex } "
                                                                       f"---",
                                                      Filehandles    = [ Responses_filehandle ],
                                                      MovementData   = (os.path.join( OUTPUTS_PATH, f'{OUTPUT_FILE_PREFIX}movement_log_{MySubjectId}.npy' ),  total_movement),
                                                      LogUtils       = log_utils,
                                                      PygameMediator = pygameMediator
                                                    )
                return   # i.e. exit main


          # Initialise image and coordinate structures
            try:
                image = images[ int( CurrentSequenceMapIndex ) ]            # A PyGame surface representing a map
            except (TypeError, IndexError):
                print( 'Try/Catch: Note - image object set to None for RL-Agent mode' )
                image = None

            # Handle missing coordinate files in RL-Agent mode (no GUI resources)
            try:
                img_coordinates   = all_coordinates[ int( CurrentSequenceMapIndex ) ]   # Returns a (25, 2) numpy array of coordinates for all potential cities on that map
            except (IndexError, TypeError):
                # No coordinate files available - generate placeholder coordinates for RL-Agent
                img_coordinates = numpy.random.rand(25, 2) * 100  # Placeholder coordinates
                
            nCitiesInSequence = int( NumTrials__blocks_x_sequences__2darray[ CurrentBlockIndex ][ CurrentSequenceIndex ] + INIT_NO_OF_CITIES )
            coordinates       = numpy.empty( (nCitiesInSequence, 2) )


          # Obtain coordinates for the selected cities in the sequence
            for cidx in range( nCitiesInSequence ):
                SelectedCity = CoordinateIndicesPerTrial__nestedDict_by_block_and_sequence[ CurrentBlockIndex ][ CurrentSequenceIndex ][ cidx ]
                coordinates[ cidx, : ] = img_coordinates[ SelectedCity ]

          # Initialise arrays that will hold the participant's various outputs of interest
            response              [CurrentBlockIndex][CurrentSequenceIndex] = []
            hold_response_times   [CurrentBlockIndex][CurrentSequenceIndex] = []
            release_response_times[CurrentBlockIndex][CurrentSequenceIndex] = []
            confidence            [CurrentBlockIndex][CurrentSequenceIndex] = []
            total_movement        [CurrentBlockIndex][CurrentSequenceIndex] = []

          # Initialise arrays that will hold map-specific information for this sequence
            init_severity = []   # Initial severity allocated to each city in the map BEFORE EACH TRIAL
            ResourceAllocationsAtCurrentlyVisibleCities = []   # Resources currently allocated to each city in the map
            circle_radius = []


          # Initialise the map with the two pre-trial cities and their severities (following random resource allocation)

            numpy.random.seed( 3 )   # NOTE: In practice, this seed, given an INIT_NO_OF_CITIES of 2, always results in
                                     # initial city severities of 4 and 3, and initial resource allocations to those
                                     # cities of 3 and 6. Meaning the player always has 40 resources left to allocate to
                                     # the remainder of the sequence.


            for c in range( INIT_NO_OF_CITIES ):

                init_severity.append( numpy.random.randint( MIN_INIT_SEVERITY,  1 + MAX_INIT_SEVERITY  ) )
                ResourceAllocationsAtCurrentlyVisibleCities.append    ( numpy.random.randint( MIN_INIT_RESOURCES, 1 + MAX_INIT_RESOURCES ) )
                circle_radius.append( init_circle_radius )   # Initialised at the start of the 'main' block


            resources_left = resources_to_allocate - numpy.sum( ResourceAllocationsAtCurrentlyVisibleCities )   # Note: resources_left: i.e. in this sequence (i.e. as opposed to in block)
                                                                              # Note: resources_to_allocate == AVAILABLE_RESOURCES_PER_SEQUENCE == 49

            # Skipping GUI presentation for RL-Agent

            SeveritiesOfCurrentlyVisibleCities  = exp_utils.get_updated_severity( INIT_NO_OF_CITIES, ResourceAllocationsAtCurrentlyVisibleCities, init_severity )
            direction = []

          # Update whether severity has increased or decreased per city
            for i in range( len( SeveritiesOfCurrentlyVisibleCities ) ):
                if SeveritiesOfCurrentlyVisibleCities[ i ] < init_severity[ i ]:
                    direction.append( 2 )   # decrease in severity
                else:
                    direction.append( 1 )   # increase in severity

          # Show End Of Trial Feedback -- will be turned off 'after' resources end.
            ShowEndOfTrialFeedback = True

          # Map initialisation complete; now proceed with Human/AI Agent trial annotations
            for trial_no in range( int( NumTrials__blocks_x_sequences__2darray[ CurrentBlockIndex, CurrentSequenceIndex ] ) ):

                AbsoluteTrialCount += 1
                AbsoluteTrialIndex = AbsoluteTrialCount - 1

                init_severity = SeveritiesOfCurrentlyVisibleCities   # i.e. severity before update (used in the update step)

                log_utils.tee()   # will print an empty line

                log_utils.tee(
                     f'Current trial: {trial_no+1} out of {NumTrials__blocks_x_sequences__2darray[ CurrentBlockIndex, CurrentSequenceIndex ]:.0f} in sequence',
                     f'({AbsoluteTrialCount} out of {TOTAL_NUM_TRIALS_IN_BLOCK * NUM_BLOCKS} overall)'
                   )


                log_utils.tee(
                     'Initial severity values before annotation: ',
                     ", ".join([ f'{i:.2f}' for i in init_severity])
                   )


                log_utils.tee(
                     'Resources remaining: ', resources_left
                   )


                new_locations         = INIT_NO_OF_CITIES + trial_no + 1   # Number of cities to appear on map in this step
                severity_new_location = first_severity[ AbsoluteTrialIndex ]      # I.e. from random pre-initialization

                log_utils.tee( f"Current trial's initial severity: {severity_new_location}" )

                init_severity.append( severity_new_location )   # Severity values of cities on map, before update (used
                                                                # in the update step).

                ResourceAllocationsAtCurrentlyVisibleCities.append( -1 )   # Placeholder for compatibility

                circle_radius.append( init_circle_radius )

                # City radius visualization not used in RL-Agent mode
                # if CITY_RADIUS_REFLECTS_SEVERITY:
                #     circle_radius[ -1 ] -= (init_severity[ -1 ] * 1.5)


                # Skipping GUI presentation for RL-Agent
                new_img = None

# NOTE: A seemingly useless delay of 2000ms was removed from here
# NOTE: A seemingly useless if statement (trial_no < num. trials in sequence) was removed from here


                if resources_left > 0:

                  # Get response from user
                    ( pc,      # corresponds to the value of 'confidence' precalculated on provide_response.  Ignored for online players.
                      r,       # corresponds to the value of 'resp' variable (defined as global within 'provide_response') at the time of return
                      rt_h,    # corresponds to the value of 'rt_hold'       (defined as global within 'provide_response') at the time of return
                      rt_rel,  # corresponds to the value of 'rt_release'    (defined as global within 'provide_response') at the time of return
                      mov      # corresponds to 'movement' array in 'provide_response'
                    ) = pygameMediator.provide_response(
                                         new_img,          # 1-element list with eltype (a : pygame.Surface, a.get_rect())
                                         ResourceAllocationsAtCurrentlyVisibleCities,        # resource allocation to cities currently shown on map
                                         resources_left,   # resources left in this sequence
                                         coordinates[ : new_locations ],
                                         circle_radius,
                                         CurrentBlockIndex,
                                         CurrentSequenceIndex,
                                         trial_no,
                                         image
                                       )

                  # Get confidence rating from user
                    if r == 0.0:
                        c = -1      # c here stands for 'confidence', not 'city' as elsewhere.
                        mov = [0]
                    else:
                        if (pc == -1):
                            c = pygameMediator.provide_confidence( new_img, CurrentBlockIndex, CurrentSequenceIndex, trial_no )
                        else:
                            c = pc


                    confidence[ CurrentBlockIndex ][ CurrentSequenceIndex ].append( c )

                    response              [ CurrentBlockIndex ][ CurrentSequenceIndex ].append( r      )
                    hold_response_times   [ CurrentBlockIndex ][ CurrentSequenceIndex ].append( rt_h   )
                    release_response_times[ CurrentBlockIndex ][ CurrentSequenceIndex ].append( rt_rel )
                    total_movement        [ CurrentBlockIndex ][ CurrentSequenceIndex ].append( mov    )

                    log_utils.tee( f'Response was: {r}'                  )
                    log_utils.tee( f'Confidence was: {c}'                )
                    log_utils.tee( f'PressEvent_seconds was: {rt_h}'     )
                    log_utils.tee( f'ReleaseEvent_seconds was: {rt_rel}' )

                    # Single agent - no need to wait for other players


                else:   # i.e. resources_left NOT > 0

                    log_utils.tee( "No resources left for allocation." )

                    confidence[ CurrentBlockIndex ][ CurrentSequenceIndex ].append( -1 )

                    response              [ CurrentBlockIndex ][ CurrentSequenceIndex ].append(  0  )
                    hold_response_times   [ CurrentBlockIndex ][ CurrentSequenceIndex ].append(  0  )
                    release_response_times[ CurrentBlockIndex ][ CurrentSequenceIndex ].append(  0  )
                    total_movement        [ CurrentBlockIndex ][ CurrentSequenceIndex ].append( [0] )




              ## "Send the event to USB to keep track of the decision-making process of the user"
                # XXX Note: this variable is not used anywhere
                resp_header = 'R=' + str( response[ CurrentBlockIndex ][ CurrentSequenceIndex ][ -1 ] ) + "Conf=" + \
                              str( confidence[ CurrentBlockIndex ][ CurrentSequenceIndex ][ -1 ] )

                if SAVE_RESULTS:

                    Responses_filehandle.write(
                        ", ".join([
                            f"{first_severity                            [ AbsoluteTrialIndex ]: >{len( '#InitialSeverity'     )}}",
                            f"{response              [ CurrentBlockIndex ][ CurrentSequenceIndex ][ -1 ]: >{len( 'Response'             )}}",
                            f"{confidence            [ CurrentBlockIndex ][ CurrentSequenceIndex ][ -1 ]: >{len( 'Confidence'           )}}",
                            f"{hold_response_times   [ CurrentBlockIndex ][ CurrentSequenceIndex ][ -1 ]: >{len( 'PressEvent_seconds'   )}}",
                            f"{release_response_times[ CurrentBlockIndex ][ CurrentSequenceIndex ][ -1 ]: >{len( 'ReleaseEvent_seconds' )}}"
                        ]) + "\n"
                    )

                    Responses_filehandle.flush()


              # Send a message to all other players (contains response, confidence, and final severities).
                NumTrialsInSequence = int( NumTrialsPerSequence_list[ AbsoluteSequenceIndex ] )
                StartingIndex = int( sum( NumTrialsPerSequence_list[ : AbsoluteSequenceIndex ] ) )
                InitialSeveritiesInSequence = first_severity[ StartingIndex : AbsoluteTrialCount ].copy()

                MyMessage = numpy.c_ [ response   [ CurrentBlockIndex ][ CurrentSequenceIndex ],
                                       confidence [ CurrentBlockIndex ][ CurrentSequenceIndex ],
                                       exp_utils.get_array_of_sequence_severities_from_allocations( response[ CurrentBlockIndex ][ CurrentSequenceIndex ], InitialSeveritiesInSequence ),
                                       circle_radius [ INIT_NO_OF_CITIES : ]
                                     ]

              # Optimal final severity for this sequence
                optimal_seq_final_severity = optimal_final_severity[AbsoluteSequenceIndex]
              # Single agent - only use own message
                AllPlayerIds = [ MySubjectId ]
                AllMessages  = [ MyMessage   ]

              # In RL-Agent single-agent mode, no aggregation needed
              # AggregatedAllocations = MyMessage allocations
                if call_nominated_aggregator is not None:
                    AggregatedAllocations, AggregatedFinalSeverities = call_nominated_aggregator( AllMessages, first_severity, AbsoluteSequenceIndex, AbsoluteTrialCount )
                else:
                    # Single RL-Agent mode: use direct allocation without aggregation
                    AggregatedAllocations = MyMessage[ :, 0 ]

                if resources_left > 0:

                  # Calculate severity and update necessary structures
                    ResourceAllocationsAtCurrentlyVisibleCities[ -1 ] = exp_utils.get_actual_allocation_given_allocation_type( AggregatedAllocations, MyMessage, AllMessages )
                    SeveritiesOfCurrentlyVisibleCities = exp_utils.get_updated_severity( new_locations, ResourceAllocationsAtCurrentlyVisibleCities, init_severity )

                    if SeveritiesOfCurrentlyVisibleCities[ -1 ] < init_severity[ -1 ]:   direction.append( 2 )   # decrease in severity
                    else                                                             :   direction.append( 1 )   # increase in severity
                else:
                    ResourceAllocationsAtCurrentlyVisibleCities[ -1 ] = 0

                    SeveritiesOfCurrentlyVisibleCities = exp_utils.get_updated_severity( new_locations, ResourceAllocationsAtCurrentlyVisibleCities, init_severity )

                    if SeveritiesOfCurrentlyVisibleCities[ -1 ] < init_severity[ -1 ]:   direction.append( 2 )   # decrease in severity
                    else                                                             :   direction.append( 1 )   # increase in severity


                resources_left = exp_utils.get_resources_left_given_allocation_type( resources_left, AggregatedAllocations, MyMessage, AllMessages )
                # Skip feedback screen for RL-Agent


            #
            ### END OF `for trial_no in range( int( NumTrials__blocks_x_sequences__2darray[ CurrentBlockIndex, CurrentSequenceIndex ] ) )`

          ## Ok, at this point I need to propagate to all the others my final severity, and get from the others their
            # final severity, so we can compare the performance and show some feedback.

          # Sequence completed, continuing to feedback processing


          ## Send a message to all other players (contains response, confidence, and final severities).
            # This is used at the Feedback Screen
            NumTrialsInSequence = int( NumTrialsPerSequence_list[ AbsoluteSequenceIndex ] )
            StartingIndex = int( sum( NumTrialsPerSequence_list[ : AbsoluteSequenceIndex ] ) )
            InitialSeveritiesInSequence = first_severity[ StartingIndex : AbsoluteTrialCount ].copy()

            MyMessage = numpy.c_ [ response   [ CurrentBlockIndex ][ CurrentSequenceIndex ],
                                   confidence [ CurrentBlockIndex ][ CurrentSequenceIndex ],
                                   exp_utils.get_array_of_sequence_severities_from_allocations( response[ CurrentBlockIndex ][ CurrentSequenceIndex ], InitialSeveritiesInSequence ),
                                   circle_radius [ INIT_NO_OF_CITIES : ]
                                 ]

          # Optimal final severity for this sequence
            optimal_seq_final_severity = optimal_final_severity[AbsoluteSequenceIndex]


          ## Collect your own response and that of all the other players
            # This constant is used to mark the UDP messages with a different number than those
            # used on each trial.

            log_utils.tee()
            log_utils.tee( f'Number of available responses:{len(AllMessages)}')

            # In RL-Agent mode, always process feedback (no GUI display)
            if True:  # Was: if DISPLAY_FEEDBACK

                NumTrialsInSequence = int( NumTrialsPerSequence_list[ AbsoluteSequenceIndex ] )
                StartingIndex = int( sum( NumTrialsPerSequence_list[ : AbsoluteSequenceIndex ] ) )
                InitialSeveritiesInSequence = first_severity[ StartingIndex : AbsoluteTrialCount ].copy()

                ( MyPerformance,
                  WorstCaseSequenceSeverity,
                  BestCaseSequenceSeverity ) = exp_utils.calculate_normalised_final_severity_performance_metric(
                                                   MyMessage[ :, 2 ],
                                                   InitialSeveritiesInSequence
                                               )

                MyPerformances.append( MyPerformance )
                AllPerformances[0].append( MyPerformance )
                log_utils.tee(
                     "My Sequence Performances: ",
                     ", ".join([f'{i:.2f}' for i in MyPerformances])
                   )
                log_utils.tee()

                # Single agent - use index 1 for aggregated performance (same as agent in this case)
                count = 1

              # Add the aggregated performance.
                if call_nominated_aggregator is not None:
                    aggregated_allocations, aggregated_final_severity = call_nominated_aggregator( AllMessages, first_severity, AbsoluteSequenceIndex, AbsoluteTrialCount )
                else:
                    # Single RL-Agent mode: compute aggregated performance directly
                    aggregated_allocations = AllMessages[0][ :, 0 ]  # Use allocations from single agent
                    aggregated_final_severity = SeveritiesOfCurrentlyVisibleCities  # Use computed severities
                
                AggregatedPerformance, *_ = exp_utils.calculate_normalised_final_severity_performance_metric(
                                            aggregated_final_severity,
                                            InitialSeveritiesInSequence
                                        )

                AllPerformances[count].append( AggregatedPerformance )

              # Plot all the accumulated performance for all the players.
                StartingAbsoluteSequence = STARTING_BLOCK_INDEX * NUM_SEQUENCES + STARTING_SEQ_INDEX
                
                # For RL-Agent, print results to console
                log_utils.tee( f"Sequence {AbsoluteSequenceIndex}: Performance = {AggregatedPerformance:.4f}" )

            #
            ### END OF `if feedback`


            AbsoluteSequenceIndex += 1

        #
        ### END OF `for CurrentSequenceIndex, CurrentSequenceMapIndex in enumerate( CurrentBlockMapIndices )`


      # Skip trust ratings for RL-Agent
        log_utils.tee( "Trust Ratings: N/A for RL-Agent" )


    #
    ### END OF `for CurrentBlockIndex in range( NUM_BLOCKS )`


  # Experiment completed
    ExperimentEndTime = datetime.datetime.now( tz = datetime.timezone.utc )
    log_utils.tee( "\n--- Experiment completed successfully ---\n" )

  # Print results summary and generate plots
    exp_utils.print_experiment_results_summary( MyPerformances, AllPerformances, log_utils )
    exp_utils.plot_experiment_results( MyPerformances, AllPerformances, OUTPUTS_PATH, MySubjectId, log_utils )
    
  # Save experiment balance to JSON for model comparison
    exp_utils.save_experiment_balance_json( MyPerformances, OUTPUTS_PATH, MySubjectId, log_utils, ExperimentStartTime, ExperimentEndTime )

  # Insert 'successful completion' marker in logfile.
    exp_utils.exit_experiment_gracefully( Message        = "--- Experiment completed successfully ---",
                                          Filehandles    = [ Responses_filehandle ],
                                          MovementData   = (os.path.join( OUTPUTS_PATH, f'{OUTPUT_FILE_PREFIX}movement_log_{MySubjectId}.npy' ),  total_movement),
                                          LogUtils       = log_utils,
                                          PygameMediator = pygameMediator
                                        )

    return

#
### END OF 'main()




if __name__ == '__main__':  main()
