"""Study configuration protocol.

Defines the interface that all study configurations must implement.
Core pipeline code depends on this protocol, never on concrete study classes.

Methods are organized by processing stage:

1. **Identity & Directory Structure** — participant IDs, folder layout
2. **Audio Metadata (Legend Parsing)** — acoustics legend sheets, microphones
3. **Biomechanics Import** — sheet names, UID parsing, speed codes
4. **Synchronization (Event Names & Columns)** — stomp events, column names
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Protocol

if TYPE_CHECKING:
    from src.audio.parsers import MicSetupData
    from src.models import BiomechanicsFileMetadata, MicrophonePosition


class StudyConfig(Protocol):
    """Protocol defining the interface for study-specific configurations.

    Each study (AOA, preOA, etc.) implements this protocol to provide
    study-specific directory structures, file naming conventions,
    and metadata formats.
    """

    # ── Identity & Directory Structure ────────────────────────────

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
        """Reverse map: folder name -> internal maneuver key.

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

    def get_motion_capture_directory_name(self) -> str:
        """Name of the biomechanics/motion capture folder.

        Returns:
            Directory name, e.g. "Motion Capture"
        """
        ...

    def find_excel_file(
        self,
        directory: Path,
        filename_pattern: str,
    ) -> Optional[Path]:
        """Find an Excel file (.xlsx or .xlsm) matching the pattern.

        Generic utility -- searches for both .xlsx and .xlsm extensions.

        Args:
            directory: Directory to search in
            filename_pattern: Glob-style filename pattern (without extension)

        Returns:
            Path to the first matching file, or None
        """
        ...

    # ── Audio Metadata (Legend Parsing) ───────────────────────────

    def get_acoustics_sheet_name(self) -> str:
        """Name of the acoustics metadata sheet in the legend Excel file.

        Returns:
            Sheet name, e.g. "Acoustic Notes"
        """
        ...

    def get_legend_file_pattern(self) -> str:
        """Glob pattern for finding the acoustics file legend.

        Returns:
            Glob pattern, e.g. "*acoustic_file_legend*"
        """
        ...

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

    def get_default_microphones(self) -> dict[int, MicrophonePosition]:
        """Default microphone layout when legend data is unavailable.

        Returns:
            Dict mapping mic number (1-4) to MicrophonePosition
        """
        ...

    # ── Biomechanics Import ───────────────────────────────────────

    def get_biomechanics_file_pattern(self, study_id: str | int) -> str:
        """Biomechanics Excel filename stem (without extension).

        Args:
            study_id: Numeric participant ID within the study

        Returns:
            Filename stem, e.g. "AOA1011_Biomechanics_Full_Set"
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

    def get_walk_event_sheet_base_name(self) -> str:
        """Base sheet name for walking events (without study ID prefix).

        Used by audio QC to load walk event data from biomechanics files.

        Returns:
            Sheet name suffix, e.g. "Walk0001"
        """
        ...

    def parse_biomechanics_uid(self, uid: str) -> BiomechanicsFileMetadata:
        """Parse a Visual3D unique identifier into structured metadata.

        Each study uses its own UID format in biomechanics Excel files.
        For example, AOA UIDs look like ``Study123_Walk0001_NSP1_Filt``.

        Args:
            uid: Cleaned unique identifier string (no V3D path or .c3d ext)

        Returns:
            BiomechanicsFileMetadata with maneuver, speed, pass_number, etc.
        """
        ...

    def get_speed_code_map(self) -> dict[str, str]:
        """Map internal speed names to biomechanics event prefix codes.

        Single source of truth for speed code mapping used across the
        pipeline (biomechanics import, audio QC, synchronization).

        Returns:
            Dict mapping speed name to code, e.g.
            ``{"slow": "SS", "normal": "NS", "fast": "FS"}``
        """
        ...

    def get_speed_event_keywords(self) -> dict[str, str]:
        """Map biomechanics event section header keywords to speed codes.

        Used for parsing walk event sheets where speed sections are
        demarcated by keyword rows (e.g. "Slow Speed" header row).

        Returns:
            Dict mapping keyword to speed code, e.g.
            ``{"Slow Speed": "SS", "Normal Speed": "NS", ...}``
        """
        ...

    # ── Synchronization (Event Names & Columns) ──────────────────

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
