"""
Configuration file for the PES experiment

Experiment Structure:
    Experimento (1)
    ├─ Bloque (8)
    │    ├─ Secuencia / Mapa (8)
    │    │    ├─ Trial / Ciudad (3~10)
    │    │    │    └─ Decision de Recursos (0-10)

Summary:
    - Total: 1 Experimento
    - 8 Bloques
    - 8 Secuencias por Bloque
    - Entre 3 y 10 Trials por Secuencia

Note:
    Las configuraciones de cada Experimento se definen en este archivo.
"""

#------------------ Configuration Constants ---------------#
AVAILABLE_RESOURCES_PER_SEQUENCE = 39   # Total resources available to be allocated per sequence      

INIT_NO_OF_CITIES = 2   # Initial cities per sequence

INITIAL_SEVERITY_FILE = 'initial_severity.csv'   # Initial severity data file
SEQ_LENGTHS_FILE = 'sequence_lengths.csv'       # Sequence lengths data file
RANDOM_INITIAL_SEVERITY = False # If True, initial severity will be randomly generated instead of using the file
SAVE_INITIAL_SEVERITY_TO_FILE = False # If True, the randomly generated initial severity will be saved to a CSV file for record-keeping

AGGREGATION_METHOD = {  1: 'confidence_weighted_median',
                        2: 'confidence_weighted_mean',
                        3: 'confidence_weighted_mode'
                     }[ 2 ]    # <-- select option here

MAX_ALLOCATABLE_RESOURCES = 10  # Suggested Value: 10
MAX_INIT_RESOURCES = 6          # Suggested Value: 6
MAX_INIT_SEVERITY  = 5          # Suggested Value: 5
MIN_ALLOCATABLE_RESOURCES = 0   # Suggested Value: 0
MIN_INIT_RESOURCES = 3          # Suggested Value: 3
MIN_INIT_SEVERITY  = 2          # Suggested Value: 2       

NUM_ATTEMPTS_TO_ASSIGN_SEQ = 8  # Suggested Value: 8       ## Number of attempts to assign sequences to a block while satisfying constraints
NUM_BLOCKS = 8                  # Suggested Value: 8       ## Number of Blocks in the Experiment (between 6 and 8 according to spec document)
NUM_MAX_TRIALS = 10             # Suggested Value: 10      ## Maximum number of trials that can be allocated to a sequence
NUM_MIN_TRIALS = 3              # Suggested Value: 3       ## Minimum number of trials that can be allocated to a sequence
NUM_SEQUENCES = 8               # Suggested Value: 8       ## Number of Sequences (i.e. 'maps') per Block (between 8 and 12 according to spec)

OUTPUT_FILE_PREFIX = 'PES_'    # Prefix for output files

PLAYER_TYPE = { # Player Configuration - Select player type
    1: 'RL_AGENT' 
    }[1]

PANDEMIC_PARAMETER = 0.4            # (β): severity multiplier and response multiplier at initialisation

STARTING_BLOCK_INDEX = 0   # Default: 0  (i.e. start at the first block of the experiment)
STARTING_SEQ_INDEX   = 0   # Default: 0  (i.e. start at the first sequence of the selected block)

TOTAL_NUM_TRIALS_IN_BLOCK  = 45       # Suggested Value: 45      ## The total trials contained in a Block should be 45; in other words, the sum of trials over all sequences in a block should sum up to 45. Q: Why? A: According to Riccardo, this is probably to ensure that all blocks have more or less the same duration (i.e. so that breaks occur at relatively consistent intervals)
TRUST_MAX = 100                       # Suggested Value: 100     ## The maximum integer value on the trust slider scales. (previously 4, now 100)
USE_FIXED_BLOCK_SEQUENCES = True      # Suggested Value: True    ## If True, the length of each sequence is obtained from 'sequence_lengths.csv' file.
VERBOSE = True                        # Suggested Value: True    ## Set to True to enable informative messages on the terminal (e.g. initialization logs etc)
SAVE_RESULTS = True                   # Suggested Value: True    ## If True, results will be saved to output files after each sequence
