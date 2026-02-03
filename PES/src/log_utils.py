"""
PES - Pandemic Experiment Scenario

Various logging utility functions:
 - create_ConsoleLog_filehandle_singleton
 - get_subject_id   # TODO confirm if necessary
 - print_settings
 - print_with_colour
 - tee
 - TODO update
"""


# ----------------
# external imports
# ----------------

import os
import datetime

# ----------------
# internal imports
# ----------------

from .. import *


# -----------------------------------------
# module variables requiring initialization
# -----------------------------------------

ConsoleLog_filehandle = None



####################
### Module functions
####################

def create_ConsoleLog_filehandle_singleton( SubjectId : str ):
    """
    Returns the existing filehandle to a log file for redirecting console output to.
    If none exists, it creates this instance and makes it available to the module.

    In theory you should never have a reason to 'reset' this singleton. However, if you'd like to do that regardless, you
    can do so by manually overwriting the existing module-global ConsoleLog_filehandle object with None (preferably closing
    the old one first!), which will cause this function to create a fresh one.
    """

    assert SubjectId is not None

    global ConsoleLog_filehandle

    if ConsoleLog_filehandle is None:
        ConsoleLog_filename   = os.path.join( OUTPUTS_PATH, f'{OUTPUT_FILE_PREFIX}log_{SubjectId}.txt' )
        ConsoleLog_filehandle = ConsoleLog_filehandle = open( ConsoleLog_filename, 'w' )


    return ConsoleLog_filehandle




def close_consolelog_filehandle():

    global ConsoleLog_filehandle
    assert ConsoleLog_filehandle is not None, "ConsoleLog_filehandle provided was None. Aborting `close` operation."
    ConsoleLog_filehandle.close()



def _convert_tensorflow_types(value):
    """Convert TensorFlow types to native Python types."""
    if hasattr(value, 'numpy'):  # TensorFlow Variable or Tensor
        return value.numpy().item() if value.numpy().ndim == 0 else value.numpy()
    return value


def tee( *Strings, **kwargs ):
    """
    Utility to simultaneously print a string to the terminal and also write it
    to a filehandle with an associated timestamp. Defaults to ANSI Blue colour
    if no other colours are used.

    Any ansi color sequences are stripped before writing to file, but are kept
    for printing to the terminal.
    """

    global ConsoleLog_filehandle
    assert ConsoleLog_filehandle is not None, "ConsoleLog_filehandle has not been initialised in module"

    if 'file'  in kwargs:   kwargs.pop( 'file' )
    if 'flush' in kwargs:   kwargs.pop( 'flush' )

    # Convert TensorFlow types to native Python types
    Strings = [ _convert_tensorflow_types(Str) for Str in Strings ]
    Strings = [ ANSI.BLUE + str(Str) + ANSI.RESET for Str in Strings ]

    ColorlessStrings = []
    for Str in Strings:
        ColorlessStrings.append( strip_colour( Str ) )

    if VERBOSE: print( *Strings, flush = True, **kwargs )
    print( '[', datetime.datetime.now( tz = datetime.timezone.utc ), ']:', *ColorlessStrings, file = ConsoleLog_filehandle, flush = True, **kwargs )




def strip_colour( Str ):
    AnsiColours = [i for i in dir( ANSI ) if not i.startswith( '_' )]
    for Colour in AnsiColours:  Str = Str.replace( getattr( ANSI, Colour ), '' )
    return Str






