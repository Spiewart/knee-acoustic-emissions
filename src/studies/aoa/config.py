"""AOA study configuration.

Encapsulates all AOA-specific directory structure, file naming conventions,
biomechanics sheet names, and metadata formats. Core pipeline code uses
the StudyConfig protocol and never references AOA conventions directly.

Directory structure:
    #AOA{id}/
        Left Knee/
            Walking/
            Sit to Stand/
            Flexion-Extension/
        Right Knee/
            ...
        Motion Capture/
            AOA{id}_Biomechanics_Full_Set.xlsx
        *acoustic_file_legend*.xlsx

Biomechanics sheets:
    Walk:   AOA{id}_Slow_Walking, AOA{id}_Medium_Walking, AOA{id}_Fast_Walking
    STS:    AOA{id}_SitToStand
    FE:     AOA{id}_FlexExt

Event sheets:
    Walk:   AOA{id}_Walk0001
    STS:    AOA{id}_StoS_Events
    FE:     AOA{id}_FE_Events
"""

from pathlib import Path
from typing import Literal, Optional


class AOAConfig:
    """AOA study-specific configuration."""

    @property
    def study_name(self) -> str:
        return "AOA"

    def get_knee_directory_name(self, knee: Literal["left", "right"]) -> str:
        """AOA uses 'Left Knee' / 'Right Knee' directory names."""
        return f"{knee.capitalize()} Knee"

    def get_biomechanics_file_pattern(self, study_id: str | int) -> str:
        """AOA biomechanics filename: AOA{id}_Biomechanics_Full_Set."""
        return f"AOA{study_id}_Biomechanics_Full_Set"

    def get_acoustics_sheet_name(self) -> str:
        """AOA acoustics metadata sheet is named 'Acoustic Notes'."""
        return "Acoustic Notes"

    def construct_biomechanics_sheet_names(
        self,
        study_id: str,
        maneuver: Literal["walk", "sit_to_stand", "flexion_extension"],
        speed: Optional[str] = None,
    ) -> dict[str, str]:
        """Construct AOA biomechanics sheet names.

        Args:
            study_id: Full prefixed ID, e.g. "AOA1011"
            maneuver: Internal maneuver name
            speed: Speed for walk maneuvers ("slow", "medium", "fast")

        Returns:
            Dict with "data_sheet" and "event_sheet" keys
        """
        if maneuver == "walk":
            speed_map = {
                "slow": "Slow",
                "medium": "Medium",
                "fast": "Fast",
            }
            if speed not in speed_map:
                raise ValueError(
                    f"Invalid speed '{speed}' for walk. Expected: {list(speed_map.keys())}"
                )
            data_sheet = f"{study_id}_{speed_map[speed]}_Walking"
            event_sheet = f"{study_id}_Walk0001"
        elif maneuver == "sit_to_stand":
            data_sheet = f"{study_id}_SitToStand"
            event_sheet = f"{study_id}_StoS_Events"
        elif maneuver == "flexion_extension":
            data_sheet = f"{study_id}_FlexExt"
            event_sheet = f"{study_id}_FE_Events"
        else:
            raise ValueError(f"Unknown maneuver: {maneuver}")

        return {"data_sheet": data_sheet, "event_sheet": event_sheet}

    def get_legend_file_pattern(self) -> str:
        """AOA legend file pattern: *acoustic_file_legend*."""
        return "*acoustic_file_legend*"

    def parse_participant_id(self, directory_name: str) -> tuple[str, int]:
        """Parse AOA participant ID from directory name.

        Args:
            directory_name: e.g. "#AOA1011", "AOA1011", or "1011"

        Returns:
            ("AOA", 1011)
        """
        cleaned = directory_name.lstrip("#")
        if cleaned.startswith("AOA"):
            numeric = cleaned[3:]
        else:
            numeric = cleaned
        try:
            return "AOA", int(numeric)
        except ValueError:
            raise ValueError(
                f"Cannot parse AOA participant ID from '{directory_name}'. "
                f"Expected format: #AOA1011 or AOA1011"
            )

    def format_study_prefix(self, study_id: int) -> str:
        """Format AOA study prefix: AOA1011."""
        return f"AOA{study_id}"

    def find_excel_file(
        self,
        directory: Path,
        filename_pattern: str,
    ) -> Optional[Path]:
        """Find an Excel file (.xlsx or .xlsm) matching the pattern.

        Searches for both extensions, returns the first match.
        """
        for extension in (".xlsx", ".xlsm"):
            files = sorted(directory.glob(f"{filename_pattern}{extension}"))
            if files:
                return files[0]
        return None
