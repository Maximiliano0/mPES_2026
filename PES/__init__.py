#####################################################################
### Initial imports and sanity checks required for initialization ###
#####################################################################

import os
import sys
import numpy
from config import CONFIG

# Relevant directories and files paths
PKG_ROOT    = os.path.dirname( os.path.abspath( __file__ ) )

DOCUMENTATION_PATH = os.path.join( PKG_ROOT, 'doc' )
OUTPUTS_PATH      = os.path.join( PKG_ROOT, 'outputs' )
INPUTS_PATH       = os.path.join( PKG_ROOT, 'inputs' )

# ANSI colour escape codes (for use in logging/debug messages)
class ANSI:
    BOLD   = '\033[1m'
    RED    = '\033[91m'
    GREEN  = '\033[92m'
    ORANGE = '\033[93m'
    BLUE   = '\033[94m'
    PURPLE = '\033[95m'
    RESET  = '\033[0m'


# Suggest virtual-environment usage if none detected
if not os.getenv( 'VIRTUAL_ENV' ):
    print(
f"""{ANSI.PURPLE}

Warning: No suitable VIRTUAL_ENV environmental variable detected.

In order to ensure consistency / reproducibility between runs, you might want to
consider always running this experiment from within a suitable python virtual
environment, containing the python package versions specified in the package's
requirements.txt file.

Press ENTER if you'd like to continue regardless (or Ctrl-C to abort).

{ANSI.RESET}""" )
    try                     : input()   # i.e. press Enter
    except KeyboardInterrupt: print( '\n\nExiting...' ); exit()


#######################################
### Process selected CONFIG.py file ###
#######################################

AVAILABLE_RESOURCES_PER_SEQUENCE            = CONFIG.AVAILABLE_RESOURCES_PER_SEQUENCE
INIT_NO_OF_CITIES                           = CONFIG.INIT_NO_OF_CITIES
INITIAL_SEVERITY_FILE                       = CONFIG.INITIAL_SEVERITY_FILE
MAX_ALLOCATABLE_RESOURCES                   = CONFIG.MAX_ALLOCATABLE_RESOURCES
MAX_INIT_RESOURCES                          = CONFIG.MAX_INIT_RESOURCES
MAX_INIT_SEVERITY                           = CONFIG.MAX_INIT_SEVERITY
MIN_ALLOCATABLE_RESOURCES                   = CONFIG.MIN_ALLOCATABLE_RESOURCES
MIN_INIT_RESOURCES                          = CONFIG.MIN_INIT_RESOURCES
MIN_INIT_SEVERITY                           = CONFIG.MIN_INIT_SEVERITY
NUM_ATTEMPTS_TO_ASSIGN_SEQ                  = CONFIG.NUM_ATTEMPTS_TO_ASSIGN_SEQ
NUM_BLOCKS                                  = CONFIG.NUM_BLOCKS
NUM_MAX_TRIALS                              = CONFIG.NUM_MAX_TRIALS
NUM_MIN_TRIALS                              = CONFIG.NUM_MIN_TRIALS
NUM_PREDEFINED_CITY_COORDS                  = CONFIG.NUM_PREDEFINED_CITY_COORDS
NUM_SEQUENCES                               = CONFIG.NUM_SEQUENCES
OUTPUT_FILE_PREFIX                          = CONFIG.OUTPUT_FILE_PREFIX
PANDEMIC_PARAMETER                          = CONFIG.PANDEMIC_PARAMETER
PLAYER_TYPE                                 = CONFIG.PLAYER_TYPE
SAVE_RESULTS                                = CONFIG.SAVE_RESULTS
STARTING_BLOCK_INDEX                        = CONFIG.STARTING_BLOCK_INDEX
STARTING_SEQ_INDEX                          = CONFIG.STARTING_SEQ_INDEX
TOTAL_NUM_TRIALS_IN_BLOCK                   = CONFIG.TOTAL_NUM_TRIALS_IN_BLOCK
TRUST_MAX                                   = CONFIG.TRUST_MAX
USE_FIXED_BLOCK_SEQUENCES                   = CONFIG.USE_FIXED_BLOCK_SEQUENCES
VERBOSE                                     = CONFIG.VERBOSE

##############################################
### Process imported configuration options ###
##############################################
# sev_n = * β * sev_(n-1) - α * a  --> a is allocated resources, sev is severity
RESPONSE_MULTIPLIER = PANDEMIC_PARAMETER # α (Alpha)
SEVERITY_MULTIPLIER = 1 + PANDEMIC_PARAMETER # β (Beta)

# The experiment uses tensorflow, which has a nasty habit of dumping lots of
# warning messages for missing nvidia libraries etc. The following environmental
# variable disables these. ( '0': all logs are shown; '1': filter out INFOs and
# below; '2': filter out WARNs; '3': filter out ERRORs, etc )
os.environ[ 'TF_CPP_MIN_LOG_LEVEL' ] ="2"

# Set some nice numpy printing defaults and error handling
numpy.set_printoptions( threshold = numpy.inf, precision = 3, suppress = True,
                        linewidth = 80, nanstr = "--", infstr = "∞"  )
numpy.seterr( all = 'raise' )

# Print all (important) final init variables to terminal
if VERBOSE not in [ True, False ]:
    raise ValueError( 'Bad value given for VERBOSE environmental variable. Needs to be True or False.' )


if VERBOSE:
#               Variable name                           Variable Value                           Suggested value check
    printconfig( 'PKG_ROOT'                             ,                              PKG_ROOT                                      )
    printconfig( 'CONFIG_FILE'                          ,                           CONFIG_FILE                                      )
    printconfig( 'ALLOCATION_TYPE'                      ,                       ALLOCATION_TYPE, 'shared'                            )
    printconfig( 'AVAILABLE_RESOURCES_PER_SEQUENCE'     ,      AVAILABLE_RESOURCES_PER_SEQUENCE, 49                                  )
    printconfig( 'CITY_RADIUS_REFLECTS_SEVERITY'        ,         CITY_RADIUS_REFLECTS_SEVERITY, False                               )
    printconfig( 'CONFIDENCE_TIMEOUT'                   ,                    CONFIDENCE_TIMEOUT, 5000                                )
    printconfig( 'DISPLAY_FEEDBACK'                     ,                      DISPLAY_FEEDBACK, True                                )
    printconfig( 'INIT_NO_OF_CITIES'                    ,                     INIT_NO_OF_CITIES, 2                                   )
    printconfig( 'INPUTS_PATH'                          ,                           INPUTS_PATH, os.path.join( PKG_ROOT, 'inputs' )  )
    printconfig( 'LIVE_EXPERIMENT'                      ,                       LIVE_EXPERIMENT, True                                )
    printconfig( 'MAX_ALLOCATABLE_RESOURCES'            ,             MAX_ALLOCATABLE_RESOURCES, 10                                  )
    printconfig( 'MAX_INIT_RESOURCES'                   ,                    MAX_INIT_RESOURCES, 6                                   )
    printconfig( 'MAX_INIT_SEVERITY'                    ,                     MAX_INIT_SEVERITY, 5                                   )
    printconfig( 'MIN_ALLOCATABLE_RESOURCES'            ,             MIN_ALLOCATABLE_RESOURCES, 0                                   )
    printconfig( 'MIN_INIT_RESOURCES'                   ,                    MIN_INIT_RESOURCES, 3                                   )
    printconfig( 'MIN_INIT_SEVERITY'                    ,                     MIN_INIT_SEVERITY, 2                                   )
    printconfig( 'NUM_BLOCKS'                           ,                            NUM_BLOCKS, 8                                   )
    printconfig( 'NUM_MAX_TRIALS'                       ,                        NUM_MAX_TRIALS, 10                                  )
    printconfig( 'NUM_MIN_TRIALS'                       ,                        NUM_MIN_TRIALS, 3                                   )
    printconfig( 'NUM_SEQUENCES'                        ,                         NUM_SEQUENCES, 8                                   )
    printconfig( 'OUTPUT_FILE_PREFIX'                   ,                    OUTPUT_FILE_PREFIX, 'PES_RL_'                        )
    printconfig( 'OUTPUTS_PATH'                         ,                          OUTPUTS_PATH, os.path.join( PKG_ROOT, 'outputs' ) )
    printconfig( 'PANDEMIC_PARAMETER'                   ,                   PANDEMIC_PARAMETER , 0.6                                 )
    printconfig( 'PLAYER_TYPE'                          ,                           PLAYER_TYPE, 'RL-Agent'                          )
    printconfig( 'RESPONSE_TIMEOUT'                     ,                      RESPONSE_TIMEOUT, 10000                               )
    printconfig( 'RESPONSE_MULTIPLIER'                  ,                   RESPONSE_MULTIPLIER, 0.6                                 )
    printconfig( 'SAVE_RESULTS'                         ,                          SAVE_RESULTS, True                                )
    printconfig( 'SEVERITY_MULTIPLIER'                  ,                   SEVERITY_MULTIPLIER, 1.6                                 )
    printconfig( 'SHOW_BEFORE_AND_AFTER_MAP'            ,             SHOW_BEFORE_AND_AFTER_MAP, False                               )
    printconfig( 'STARTING_BLOCK_INDEX'                 ,                  STARTING_BLOCK_INDEX, 0                                   )
    printconfig( 'STARTING_SEQ_INDEX'                   ,                    STARTING_SEQ_INDEX, 0                                   )
    printconfig( 'TOTAL_NUM_TRIALS_IN_BLOCK'            ,             TOTAL_NUM_TRIALS_IN_BLOCK, 45                                  )
    printconfig( 'TRUST_MAX'                            ,             TRUST_MAX                , 100                                 )
    printconfig( 'USE_FIXED_BLOCK_SEQUENCES'            ,             USE_FIXED_BLOCK_SEQUENCES, True                                )
    printconfig( 'VERBOSE'                              ,                               VERBOSE, True                                )
    printconfig( 'SEQ_LENGTHS_FILE'                     ,                      SEQ_LENGTHS_FILE, 'sequence_lengths.csv'              )
    printconfig( 'INITIAL_SEVERITY_FILE'                ,                 INITIAL_SEVERITY_FILE, 'initial_severity.csv'              )


# Initialise pygame engine
if VERBOSE:
    printinfo( "__init__: Initializing pygame engine ... ", end = '', flush = True )
    pygame.init();
    printstatus( 'Done', ANSI.GREEN )
else:
    pygame.init()


# List of package variables to be made availbale to package modules
__all__ = [
    'PKG_ROOT',
    'CONFIG_FILE',
    'DOCUMENTATION_PATH',
    'RESOURCES_PATH',
    'VERBOSE',
    'BIOSEMI_CONNECTED',
    'SEQ_LENGTHS_FILE',
    'INITIAL_SEVERITY_FILE',
    'WHITE',
    'YELLOW',
    'BLACK',
    'DARK_RED',
    'DARK_CYAN',
    'DARK_GREEN',
    'GREEN',
    'RED',
    'GRAY',
    'LIGHTGRAY',
    'LIGHTBLUE',

    'ANSI',
    'printinfo',
    'printstatus',
    'printconfig',

    'AGGREGATION_METHOD',
    'ALLOCATION_TYPE',
    'AVAILABLE_RESOURCES_PER_SEQUENCE',
    'AVATAR_ICONS_SET',
    'BLOCK_MODE_INDICES',
    'CITY_RADIUS_REFLECTS_SEVERITY',
    'COLORS',
    'CONFIDENCE_TIMEOUT',
    'CONFIDENCE_UPDATE_AMOUNT',
    'DEBUG',
    'DEBUG_RESOLUTION',
    'DETECT_USER_RESOLUTION',
    'DISPLAY_FEEDBACK',
    'FALLBACK_RESOLUTION',
    'FEEDBACK_SHOW_COMBINED_ALLOCATIONS',
    'FEEDBACK_SHOW_GROUP_PERFORMANCE',
    'FEEDBACK_SHOW_INDIVIDUAL_PERFORMANCES',
    'FORCE_MOUSEWHEEL_SCROLL_CONFIDENCE',
    'INIT_NO_OF_CITIES',
    'INPUTS_PATH',
    'LIVE_EXPERIMENT',
    'LOBBY_PLAYERS',
    'LOBBY_TIMEOUT',
    'MAX_ALLOCATABLE_RESOURCES',
    'MAX_INIT_RESOURCES',
    'MAX_INIT_SEVERITY',
    'MIN_ALLOCATABLE_RESOURCES',
    'MIN_INIT_RESOURCES',
    'MIN_INIT_SEVERITY',
    'MOVEMENT_REFRESH_RATE',
    'NUM_ATTEMPTS_TO_ASSIGN_SEQ',
    'NUM_BLOCKS',
    'NUM_MAX_TRIALS',
    'NUM_MIN_TRIALS',
    'NUM_PREDEFINED_CITY_COORDS',
    'NUM_SEQUENCES',
    'OUTPUT_FILE_PREFIX',
    'OUTPUTS_PATH',
    'PANDEMIC_PARAMETER',
    'PLAYBACK_ID',
    'PLAYER_TYPE',
    'RANDOM_INITIAL_SEVERITY',
    'RESPONSE_MULTIPLIER',
    'RESPONSE_TIMEOUT',
    'SAVE_INITIAL_SEVERITY_TO_FILE',
    'SAVE_RESULTS',
    'SEVERITY_MULTIPLIER',
    'SHOW_BEFORE_AND_AFTER_MAP',
    'SHOW_PYGAME_IF_NONHUMAN_PLAYER',
    'STARTING_BLOCK_INDEX',
    'STARTING_SEQ_INDEX',
    'TOTAL_NUM_TRIALS_IN_BLOCK',
    'TRUST_MAX',
    'USE_FIXED_BLOCK_SEQUENCES'
]

if VERBOSE:   print()   # Just to separate initialization messages from rest of execution.
