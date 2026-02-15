"""Orchestration submodule.

Handles high-level workflows like processing participant directories.
"""

from src.orchestration.participant import (
    find_participant_directories,
    process_participant,
    sync_single_audio_file,
)
from src.orchestration.persistent_processor import (
    PersistentParticipantProcessor,
    create_persistent_processor,
)

__all__ = [
    "PersistentParticipantProcessor",
    "create_persistent_processor",
    "find_participant_directories",
    "process_participant",
    "sync_single_audio_file",
]
