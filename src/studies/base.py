"""Study configuration protocol.

Defines the interface that all study configurations must implement.
Core pipeline code depends on this protocol, never on concrete study classes.
"""

from pathlib import Path
from typing import Literal, Optional, Protocol


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
