"""Audio processing submodule.

Handles reading, parsing, and quality control of audio board data.
"""

from src.audio.parsers import (
                               LegendMismatch,
                               MicSetupData,
                               extract_file_name_and_notes,
                               extract_knee_metadata_table,
                               extract_microphone_positions,
                               filter_by_maneuver,
                               find_knee_table_start,
                               get_acoustics_metadata,
                               normalize_maneuver_column,
)
from src.audio.quality_control import (
                               qc_audio_directory,
                               qc_audio_flexion_extension,
                               qc_audio_sit_to_stand,
                               qc_audio_walk,
)
from src.audio.readers import read_audio_board_file

__all__ = [
    # readers
    "read_audio_board_file",
    # parsers
    "get_acoustics_metadata",
    "LegendMismatch",
    "MicSetupData",
    "find_knee_table_start",
    "extract_knee_metadata_table",
    "normalize_maneuver_column",
    "filter_by_maneuver",
    "extract_microphone_positions",
    "extract_file_name_and_notes",
    # quality_control
    "qc_audio_flexion_extension",
    "qc_audio_sit_to_stand",
    "qc_audio_walk",
    "qc_audio_directory",
]
