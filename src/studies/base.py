"""Study configuration protocol.

Defines the interface that all study configurations must implement.
Core pipeline code depends on this protocol, never on concrete study classes.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Protocol

if TYPE_CHECKING:
    from src.audio.parsers import MicSetupData
    from src.models import MicrophonePosition


class StudyConfig(Protocol):
    """Protocol defining the interface for study-specific configurations.

    Each study (AOA, preOA, etc.) implements this protocol to provide
    study-specific directory structures, file naming conventions,
    and metadata formats.
    """

    @property
    def study_name(self) -> str:
        """Short identifier for the study (e.g., 'AOA', 'preOA')."""
        ...

    def get_knee_directory_name(self, knee: Literal["left", "right"]) -> str:
        """Directory name for a given knee side.

        Args:
            knee: Lowercase knee side

        Returns:
            Directory name, e.g. "Left Knee"
        """
        ...

    def get_biomechanics_file_pattern(self, study_id: str | int) -> str:
        """Biomechanics Excel filename stem (without extension).

        Args:
            study_id: Numeric participant ID within the study

        Returns:
            Filename stem, e.g. "AOA1011_Biomechanics_Full_Set"
        """
        ...

    def get_acoustics_sheet_name(self) -> str:
        """Name of the acoustics metadata sheet in the legend Excel file.

        Returns:
            Sheet name, e.g. "Acoustic Notes"
        """
        ...

    def construct_biomechanics_sheet_names(
        self,
        study_id: str,
        maneuver: Literal["walk", "sit_to_stand", "flexion_extension"],
        speed: Optional[str] = None,
    ) -> dict[str, str]:
        """Construct biomechanics Excel sheet names for a given maneuver.

        Args:
            study_id: Full study-prefixed ID, e.g. "AOA1011"
            maneuver: Internal maneuver name
            speed: Speed category for walk maneuvers

        Returns:
            Dict with keys "data_sheet" and "event_sheet"
        """
        ...

    def get_legend_file_pattern(self) -> str:
        """Glob pattern for finding the acoustics file legend.

        Returns:
            Glob pattern, e.g. "*acoustic_file_legend*"
        """
        ...

    def parse_participant_id(self, directory_name: str) -> tuple[str, int]:
        """Parse study name and numeric ID from a participant directory name.

        Args:
            directory_name: e.g. "#AOA1011" or "AOA1011"

        Returns:
            Tuple of (study_name, numeric_id), e.g. ("AOA", 1011)
        """
        ...

    def format_study_prefix(self, study_id: int) -> str:
        """Format the full study-prefixed ID string.

        Args:
            study_id: Numeric participant ID

        Returns:
            Prefixed ID string, e.g. "AOA1011"
        """
        ...

    def find_excel_file(
        self,
        directory: Path,
        filename_pattern: str,
    ) -> Optional[Path]:
        """Find an Excel file (.xlsx or .xlsm) matching the pattern.

        Generic utility — searches for both .xlsx and .xlsm extensions.

        Args:
            directory: Directory to search in
            filename_pattern: Glob-style filename pattern (without extension)

        Returns:
            Path to the first matching file, or None
        """
        ...

    # --- Maneuver directory mapping ---

    def get_maneuver_directory_name(
        self,
        maneuver: Literal["walk", "sit_to_stand", "flexion_extension"],
    ) -> str:
        """Folder name for a maneuver type.

        Args:
            maneuver: Internal maneuver key

        Returns:
            Directory name, e.g. "Walking", "Sit-Stand"
        """
        ...

    def get_maneuver_from_directory(
        self,
        directory_name: str,
    ) -> Optional[Literal["walk", "sit_to_stand", "flexion_extension"]]:
        """Reverse map: folder name → internal maneuver key.

        Args:
            directory_name: Directory name to look up

        Returns:
            Internal maneuver key, or None if not recognized
        """
        ...

    def get_maneuver_search_terms(
        self,
        maneuver: Literal["walk", "sit_to_stand", "flexion_extension"],
    ) -> tuple[str, ...]:
        """Fallback search terms for fuzzy-matching maneuver folders.

        Used when the exact directory name isn't found.

        Args:
            maneuver: Internal maneuver key

        Returns:
            Tuple of lowercase search terms
        """
        ...

    # --- Motion capture ---

    def get_motion_capture_directory_name(self) -> str:
        """Name of the biomechanics/motion capture folder.

        Returns:
            Directory name, e.g. "Motion Capture"
        """
        ...

    # --- Biomechanics event names ---

    def get_stomp_event_name(
        self, foot: Literal["left", "right"],
    ) -> str:
        """Sync stomp event label in biomechanics data.

        Args:
            foot: Which foot performed the stomp

        Returns:
            Event label, e.g. "Sync Left"
        """
        ...

    def get_movement_start_event(
        self,
        maneuver: Literal["walk", "sit_to_stand", "flexion_extension"],
        speed: Optional[str] = None,
        pass_number: int = 1,
    ) -> str:
        """Movement start event name from biomechanics.

        Args:
            maneuver: Internal maneuver key
            speed: Speed category for walk ("slow", "normal", "fast")
            pass_number: Pass number for walk

        Returns:
            Event label, e.g. "Movement Start" or "SS Pass 1 Start"
        """
        ...

    def get_movement_end_event(
        self,
        maneuver: Literal["walk", "sit_to_stand", "flexion_extension"],
        speed: Optional[str] = None,
        pass_number: int = 1,
    ) -> str:
        """Movement end event name from biomechanics.

        Args:
            maneuver: Internal maneuver key
            speed: Speed category for walk ("slow", "normal", "fast")
            pass_number: Pass number for walk

        Returns:
            Event label, e.g. "Movement End" or "SS Pass 1 End"
        """
        ...

    # --- Biomechanics column names ---

    def get_biomechanics_event_column(self) -> str:
        """Column name for event labels in biomechanics Excel.

        Returns:
            Column name, e.g. "Event Info"
        """
        ...

    def get_biomechanics_time_column(self) -> str:
        """Column name for time values in biomechanics Excel.

        Returns:
            Column name, e.g. "Time (sec)"
        """
        ...

    def get_knee_angle_column(self) -> str:
        """Column name for knee flexion angle in biomechanics data.

        Returns:
            Column name, e.g. "Knee Angle Z"
        """
        ...

    # --- Legend fallback ---

    def parse_legend_fallback(
        self,
        metadata_file_path: str,
        scripted_maneuver: Literal["walk", "sit_to_stand", "flexion_extension"],
        knee: Literal["left", "right"],
    ) -> Optional[MicSetupData]:
        """Parse fallback legend data from a study-specific secondary sheet.

        Returns None if the study doesn't support a fallback sheet.
        Default implementation returns None.

        Args:
            metadata_file_path: Path to the legend Excel file
            scripted_maneuver: Maneuver to look up
            knee: Knee side

        Returns:
            Parsed fallback data, or None
        """
        ...

    # --- Default microphones ---

    def get_default_microphones(self) -> dict[int, MicrophonePosition]:
        """Default microphone layout when legend data is unavailable.

        Returns:
            Dict mapping mic number (1-4) to MicrophonePosition
        """
        ...
