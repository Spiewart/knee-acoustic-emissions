"""Orchestration submodule.

Handles high-level workflows like processing participant directories.
"""

from src.orchestration.participant import (
                                           find_participant_directories,
                                           process_participant,
                                           sync_single_audio_file,
)

__all__ = [
    "process_participant",
    "find_participant_directories",
    "sync_single_audio_file",
]
