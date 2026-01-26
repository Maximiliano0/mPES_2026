"""
PES - Pandemic Experiment Scenario

SIMPLIFIED VERSION FOR SINGLE AGENT ONLY
"""


# ----------------
# external imports
# ----------------

# ----------------
# internal imports
# ----------------

from . import log_utils

from .. import printinfo
from .. import ANSI
from .. import VERBOSE

# -----------------------
# module-global variables
# -----------------------

Port  = None   # Will be initialised by set_up_lobby
Lobby = {}     # Single agent only

MySubjectId = None   # Will be initialised from set_up_lobby (obtained from __main__)

MessageLog_dict = {}   # Will be initialised by set_up_lobby

####################
### Module functions
####################

def set_up_lobby( SubjectId ):
    """
    Single agent setup - no network communication.
    Simply stores the subject ID in the Lobby dict.
    """
    global Port
    global Lobby
    global MySubjectId
    global MessageLog_dict

    MySubjectId = SubjectId
    
    # For single agent, use a simple port assignment
    Port = "8000"
    
    # Lobby contains only the current agent
    Lobby = { ('127.0.0.1', int(Port)): MySubjectId }
    
    MessageLog_dict = {}
    
    if VERBOSE:
        printinfo(f'lobbyManager: Single agent setup complete for subject {MySubjectId}')
    
    return


def set_up_TCP_server():
    """
    Single agent stub - no TCP server needed for single agent.
    """
    global Port
    
    if Port is None:
        raise RuntimeError("Port has not been initialised yet in module!")
    
    if VERBOSE:
        printinfo(f'lobbyManager: Single agent mode - TCP server not needed')
    
    return


def send_and_request_stream(TrialId, MyMessage):
    """
    Single agent stub - returns only own message.
    
    For a single agent, there are no other players to communicate with.
    This function returns empty lists since there's only one agent.
    
    Args:
        TrialId: Trial identifier (not used in single agent)
        MyMessage: The agent's message (not used here, already processed)
    
    Returns:
        PlayerIds: Empty list (no other players)
        PlayerMessages: Empty list (no other messages)
    """
    global MessageLog_dict
    
    # Store message in log for compatibility
    MessageLog_dict[TrialId] = MyMessage
    
    # For single agent, there are no other players
    PlayerIds = []
    PlayerMessages = []
    
    if VERBOSE:
        printinfo(f'lobbyManager: Single agent - Trial {TrialId} message logged (no other players)')
    
    return PlayerIds, PlayerMessages


def request_data_from_other_players(TrialId=None):
    """
    Deprecated - no other players in single agent mode.
    Kept for compatibility only.
    """
    return [], []
