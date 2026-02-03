"""
Package initialization module for the mPES project.

Handles package setup including:
- Configuration loading from config.py
- Path definitions for documentation, outputs, and inputs
- ANSI color codes for terminal output
- Virtual environment validation
- Numpy and TensorFlow configuration
- Package exports via __all__
"""
######################
## External Imports ##
######################
import os
import sys
import numpy
from config import CONFIG

###########
## PATHs ##
###########
PKG_ROOT    = os.path.dirname( os.path.abspath( __file__ ) )

DOCUMENTATION_PATH = os.path.join( PKG_ROOT, 'doc' )
OUTPUTS_PATH      = os.path.join( PKG_ROOT, 'outputs' )
INPUTS_PATH       = os.path.join( PKG_ROOT, 'inputs' )

############################
## ANSI Color Scape Codes ##
############################
class ANSI:
    BOLD   = '\033[1m'
    RED    = '\033[91m'
    GREEN  = '\033[92m'
    ORANGE = '\033[93m'
    BLUE   = '\033[94m'
    PURPLE = '\033[95m'
    RESET  = '\033[0m'

#################################
## Suggest virtual-environment ##
#################################
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

###########################
### Tensorflow Issue    ###
###########################
# The experiment uses tensorflow, which has a nasty habit of dumping lots of
# warning messages for missing nvidia libraries etc. The following environmental
# variable disables these. ( '0': all logs are shown; '1': filter out INFOs and
# below; '2': filter out WARNs; '3': filter out ERRORs, etc )
os.environ[ 'TF_CPP_MIN_LOG_LEVEL' ] ="2"

# Set some nice numpy printing defaults and error handling
numpy.set_printoptions( threshold = numpy.inf, precision = 3, suppress = True,
                        linewidth = 80, nanstr = "--", infstr = "∞"  )
numpy.seterr( all = 'raise' )

#########################################
### Print final init variables to log ###
#########################################
if VERBOSE:
    print(f"{'Variable Name':<40} {'Variable Value':<30} {ANSI.RED}{'Suggested Value':<20}{ANSI.RESET}")
    print(f"{'AVAILABLE_RESOURCES_PER_SEQUENCE':<40} {str(AVAILABLE_RESOURCES_PER_SEQUENCE):<30} {ANSI.RED}{'49':<20}{ANSI.RESET}")
    print(f"{'INIT_NO_OF_CITIES':<40} {str(INIT_NO_OF_CITIES):<30} {ANSI.RED}{'  ':<20}{ANSI.RESET}")
    print(f"{'NUM_BLOCKS':<40} {str(NUM_BLOCKS):<30} {ANSI.RED}{'8':<20}{ANSI.RESET}")
    print(f"{'NUM_MAX_TRIALS':<40} {str(NUM_MAX_TRIALS):<30} {ANSI.RED}{'10':<20}{ANSI.RESET}")
    print(f"{'NUM_MIN_TRIALS':<40} {str(NUM_MIN_TRIALS):<30} {ANSI.RED}{'3':<20}{ANSI.RESET}")
    print(f"{'NUM_SEQUENCES':<40} {str(NUM_SEQUENCES):<30} {ANSI.RED}{'8':<20}{ANSI.RESET}")
    print(f"{'PANDEMIC_PARAMETER':<40} {str(PANDEMIC_PARAMETER):<30} {ANSI.RED}{'0.6':<20}{ANSI.RESET}")
    print(f"{'PLAYER_TYPE':<40} {str(PLAYER_TYPE):<30} {ANSI.RED}{'  ':<20}{ANSI.RESET}")
    print(f"{'RESPONSE_MULTIPLIER':<40} {str(RESPONSE_MULTIPLIER):<30} {ANSI.RED}{'0.6':<20}{ANSI.RESET}")
    print(f"{'SEVERITY_MULTIPLIER':<40} {str(SEVERITY_MULTIPLIER):<30} {ANSI.RED}{'1.6':<20}{ANSI.RESET}")
    print(f"{'TOTAL_NUM_TRIALS_IN_BLOCK':<40} {str(TOTAL_NUM_TRIALS_IN_BLOCK):<30} {ANSI.RED}{'45':<20}{ANSI.RESET}")
    print(f"{'USE_FIXED_BLOCK_SEQUENCES':<40} {str(USE_FIXED_BLOCK_SEQUENCES):<30} {ANSI.RED}{'  ':<20}{ANSI.RESET}")
    print(f"{'INITIAL_SEVERITY_FILE':<40} {str(INITIAL_SEVERITY_FILE):<30} {ANSI.RED}{' ':<20}{ANSI.RESET}")

##############################
### Define package exports ###
##############################
__all__ = [
    'PKG_ROOT',
    'DOCUMENTATION_PATH',
    'OUTPUTS_PATH',
    'INPUTS_PATH',
    'ANSI',
    'AVAILABLE_RESOURCES_PER_SEQUENCE',
    'INIT_NO_OF_CITIES',
    'INITIAL_SEVERITY_FILE',
    'MAX_ALLOCATABLE_RESOURCES',
    'MAX_INIT_RESOURCES',
    'MAX_INIT_SEVERITY',
    'MIN_ALLOCATABLE_RESOURCES',
    'MIN_INIT_RESOURCES',
    'MIN_INIT_SEVERITY',
    'NUM_ATTEMPTS_TO_ASSIGN_SEQ',
    'NUM_BLOCKS',
    'NUM_MAX_TRIALS',
    'NUM_MIN_TRIALS',
    'NUM_SEQUENCES',
    'OUTPUT_FILE_PREFIX',
    'PANDEMIC_PARAMETER',
    'PLAYER_TYPE',
    'SAVE_RESULTS',
    'STARTING_BLOCK_INDEX',
    'STARTING_SEQ_INDEX',
    'TOTAL_NUM_TRIALS_IN_BLOCK',
    'TRUST_MAX',
    'USE_FIXED_BLOCK_SEQUENCES',
    'VERBOSE',
    'RESPONSE_MULTIPLIER',
    'SEVERITY_MULTIPLIER',
    'CONFIG'
]