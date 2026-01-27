## RL-Agent Configuration File
##
## This file defines configuration parameters for running the Pandemic Experiment Scenario (PES)
## with a single RL-Agent in autonomous mode (no pygame GUI, no human players).
##
## Structure:
##   - Sequence   : A single map with multiple trials (cities for resource allocation)
##   - Trial      : Individual city annotation (user/agent allocates resources)
##   - Block      : Collection of 8 sequences
##   - Experiment : Collection of 8 blocks (~360 trials total)

import os

def get_INPUTS_PATH ( PkgRoot ):   return os.path.join( PkgRoot, 'inputs'  )
def get_OUTPUTS_PATH( PkgRoot ):   return os.path.join( PkgRoot, 'outputs' )

# ===== CORE EXPERIMENT PARAMETERS =====

ALLOCATION_TYPE = 'individual'   # Single RL-Agent resource allocation mode

AVAILABLE_RESOURCES_PER_SEQUENCE = 39    # Total resources per sequence (some consumed by initial cities)

BLOCK_MODE_INDICES = { 'Joint': [],
                       'Solo' : [0,1,2,3,4,5,6,7] }   # All blocks in Solo mode for single agent

# ===== SEVERITY & RESOURCE PARAMETERS =====

INIT_NO_OF_CITIES = 2           # Pre-allocated cities per sequence

MAX_ALLOCATABLE_RESOURCES = 10  # Maximum resources per allocation decision
MAX_INIT_RESOURCES = 6          # Maximum for initial city resource allocation
MAX_INIT_SEVERITY  = 5          # Maximum initial severity

MIN_ALLOCATABLE_RESOURCES = 0   # Minimum resources per allocation
MIN_INIT_RESOURCES = 3          # Minimum for initial city resources
MIN_INIT_SEVERITY  = 2          # Minimum initial severity

PANDEMIC_PARAMETER = 0.4        # Severity/response multiplier (converted at init)

# ===== EXPERIMENT STRUCTURE =====

NUM_BLOCKS = 8                  # Number of blocks in experiment
NUM_SEQUENCES = 8               # Number of sequences (maps) per block
NUM_ATTEMPTS_TO_ASSIGN_SEQ = 8  # Attempts to assign random map per sequence
NUM_MIN_TRIALS = 3              # Minimum trials per sequence
NUM_MAX_TRIALS = 10             # Maximum trials per sequence
NUM_PREDEFINED_CITY_COORDS = 25 # Available coordinate positions per map

TOTAL_NUM_TRIALS_IN_BLOCK  = 45 # Total trials per block (~360 overall)

STARTING_BLOCK_INDEX = 0        # Start from first block
STARTING_SEQ_INDEX   = 0        # Start from first sequence

# ===== INPUT/OUTPUT FILES =====

INITIAL_SEVERITY_FILE = 'initial_severity.csv'   # Pre-computed severity values
SEQ_LENGTHS_FILE = 'sequence_lengths.csv'        # Trial counts per sequence
OUTPUT_FILE_PREFIX = 'PES_'                      # Prefix for output files

# ===== RL-AGENT SPECIFIC =====

PLAYER_TYPE = 'RL-Agent'
AGENT_NOISE_VARIANCE = 8.0      # Gaussian noise for agent responses
RANDOM_INITIAL_SEVERITY = False # Use pre-loaded severity file

# ===== EXECUTION FLAGS =====

DEBUG = False                          # False for production runs
SAVE_INITIAL_SEVERITY_TO_FILE = False  # Save computed severities
SAVE_RESULTS = True                    # Save results to output files
USE_FIXED_BLOCK_SEQUENCES = True       # Use fixed sequence lengths
VERBOSE = True                         # Enable logging output

SHOW_BEFORE_AND_AFTER_MAP = False      # (Legacy: for multi-player feedback)