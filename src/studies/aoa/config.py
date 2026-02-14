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

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from src.audio.parsers import MicSetupData
    from src.models import BiomechanicsFileMetadata, MicrophonePosition


class AOAConfig:
    """AOA study-specific configuration."""

    # ── Identity & Directory Structure ────────────────────────────

    @property
    def study_name(self) -> str:
        return "AOA"

    def get_knee_directory_name(self, knee: Literal["left", "right"]) -> str:
        """AOA uses 'Left Knee' / 'Right Knee' directory names."""
        return f"{knee.capitalize()} Knee"

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

    def get_maneuver_directory_name(
        self,
        maneuver: Literal["walk", "sit_to_stand", "flexion_extension"],
    ) -> str:
        _map = {
            "walk": "Walking",
            "sit_to_stand": "Sit-Stand",
            "flexion_extension": "Flexion-Extension",
        }
        return _map[maneuver]

    def get_maneuver_from_directory(
        self,
        directory_name: str,
    ) -> Optional[Literal["walk", "sit_to_stand", "flexion_extension"]]:
        _reverse_map: dict[
            str, Literal["walk", "sit_to_stand", "flexion_extension"]
        ] = {
            "Walking": "walk",
            "Sit-Stand": "sit_to_stand",
            "Flexion-Extension": "flexion_extension",
        }
        return _reverse_map.get(directory_name)

    def get_maneuver_search_terms(
        self,
        maneuver: Literal["walk", "sit_to_stand", "flexion_extension"],
    ) -> tuple[str, ...]:
        _terms: dict[str, tuple[str, ...]] = {
            "walk": (),
            "sit_to_stand": ("sit", "stand"),
            "flexion_extension": (),
        }
        return _terms[maneuver]

    def get_motion_capture_directory_name(self) -> str:
        return "Motion Capture"

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

    # ── Audio Metadata (Legend Parsing) ───────────────────────────

    def get_acoustics_sheet_name(self) -> str:
        """AOA acoustics metadata sheet is named 'Acoustic Notes'."""
        return "Acoustic Notes"

    def get_legend_file_pattern(self) -> str:
        """AOA legend file pattern: *acoustic_file_legend*."""
        return "*acoustic_file_legend*"

    def parse_legend_fallback(
        self,
        metadata_file_path: str,
        scripted_maneuver: Literal[
            "walk", "sit_to_stand", "flexion_extension"
        ],
        knee: Literal["left", "right"],
    ) -> Optional[MicSetupData]:
        from src.studies.aoa.legend import parse_aoa_mic_setup_sheet

        return parse_aoa_mic_setup_sheet(
            metadata_file_path, scripted_maneuver, knee,
        )

    def get_default_microphones(self) -> dict[int, MicrophonePosition]:
        from src.models import MicrophonePosition

        return {
            1: MicrophonePosition(
                patellar_position="Infrapatellar", laterality="Lateral",
            ),
            2: MicrophonePosition(
                patellar_position="Infrapatellar", laterality="Medial",
            ),
            3: MicrophonePosition(
                patellar_position="Suprapatellar", laterality="Medial",
            ),
            4: MicrophonePosition(
                patellar_position="Suprapatellar", laterality="Lateral",
            ),
        }

    # ── Biomechanics Import ───────────────────────────────────────

    def get_biomechanics_file_pattern(self, study_id: str | int) -> str:
        """AOA biomechanics filename: AOA{id}_Biomechanics_Full_Set."""
        return f"AOA{study_id}_Biomechanics_Full_Set"

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
                    f"Invalid speed '{speed}' for walk. "
                    f"Expected: {list(speed_map.keys())}"
                )
            data_sheet = f"{study_id}_{speed_map[speed]}_Walking"
            event_sheet = (
                f"{study_id}_{self.get_walk_event_sheet_base_name()}"
            )
        elif maneuver == "sit_to_stand":
            data_sheet = f"{study_id}_SitToStand"
            event_sheet = f"{study_id}_StoS_Events"
        elif maneuver == "flexion_extension":
            data_sheet = f"{study_id}_FlexExt"
            event_sheet = f"{study_id}_FE_Events"
        else:
            raise ValueError(f"Unknown maneuver: {maneuver}")

        return {"data_sheet": data_sheet, "event_sheet": event_sheet}

    def get_walk_event_sheet_base_name(self) -> str:
        """AOA walk event sheet suffix (prefixed with study ID in files)."""
        return "Walk0001"

    def parse_biomechanics_uid(self, uid: str) -> BiomechanicsFileMetadata:
        """Parse an AOA Visual3D UID into structured metadata.

        AOA UID format: ``{Study}{ID}_{Maneuver}{Num}_{SpeedPass}_{Filt}``
        Examples:
            - ``AOA1011_Walk0001_NSP1_Filt``  (walk, normal speed, pass 1)
            - ``AOA1011_SitToStand0001_Filt``  (sit-to-stand)
            - ``AOA1011_FlexExt0001_Filt``     (flexion-extension)

        Args:
            uid: Cleaned UID (no V3D path prefix or .c3d extension).

        Returns:
            BiomechanicsFileMetadata with all parsed fields.

        Raises:
            ValueError: If maneuver or speed code is not recognized.
        """
        from src.models import BiomechanicsFileMetadata

        # --- Extract study token (first underscore-delimited segment) ---
        study_token = uid.split("_")[0]
        study_alpha = "".join(filter(str.isalpha, study_token)) or None
        study_digits = "".join(filter(str.isdigit, study_token))
        study_id = int(study_digits) if study_digits else None

        # --- Extract maneuver from second token ---
        maneuver = self._extract_maneuver_from_uid(uid)

        if maneuver != "walk":
            return BiomechanicsFileMetadata(
                scripted_maneuver=maneuver,
                speed=None,
                pass_number=None,
                biomech_file_name=uid,
                study=study_alpha,
                study_id=study_id,
            )

        # --- Walk: extract pass number and speed from penultimate token ---
        pass_number, speed = self._extract_walking_pass_info(uid)

        return BiomechanicsFileMetadata(
            scripted_maneuver="walk",
            speed=speed,
            pass_number=pass_number,
            biomech_file_name=uid,
            study=study_alpha,
            study_id=study_id,
        )

    def get_speed_code_map(self) -> dict[str, str]:
        """AOA speed codes: slow->SS, normal->NS, fast->FS."""
        return {"slow": "SS", "normal": "NS", "fast": "FS"}

    def get_speed_event_keywords(self) -> dict[str, str]:
        """AOA event section header keywords mapped to speed codes.

        In AOA biomechanics Walk event sheets, speed sections are
        demarcated by rows containing these keywords in the Event Info
        column.
        """
        return {
            "Slow Speed": "SS",
            "Normal Speed": "NS",
            "Medium Speed": "NS",
            "Fast Speed": "FS",
        }

    # ── Synchronization (Event Names & Columns) ──────────────────

    def get_stomp_event_name(
        self, foot: Literal["left", "right"],
    ) -> str:
        return f"Sync {foot.capitalize()}"

    def get_movement_start_event(
        self,
        maneuver: Literal["walk", "sit_to_stand", "flexion_extension"],
        speed: Optional[str] = None,
        pass_number: int = 1,
    ) -> str:
        if maneuver == "walk":
            return self._walking_event_name(speed, pass_number, "Start")
        return "Movement Start"

    def get_movement_end_event(
        self,
        maneuver: Literal["walk", "sit_to_stand", "flexion_extension"],
        speed: Optional[str] = None,
        pass_number: int = 1,
    ) -> str:
        if maneuver == "walk":
            return self._walking_event_name(speed, pass_number, "End")
        return "Movement End"

    def get_biomechanics_event_column(self) -> str:
        return "Event Info"

    def get_biomechanics_time_column(self) -> str:
        return "Time (sec)"

    def get_knee_angle_column(self) -> str:
        return "Knee Angle Z"

    # ── Private Helpers ───────────────────────────────────────────

    def _walking_event_name(
        self,
        speed: Optional[str],
        pass_number: int,
        event_type: Literal["Start", "End"],
    ) -> str:
        """Build AOA walking event name, e.g. 'SS Pass 1 Start'."""
        speed_codes = self.get_speed_code_map()
        if speed not in speed_codes:
            raise ValueError(
                f"Invalid speed '{speed}' for walk. "
                f"Expected: {list(speed_codes.keys())}"
            )
        return f"{speed_codes[speed]} Pass {pass_number} {event_type}"

    @staticmethod
    def _extract_maneuver_from_uid(
        uid: str,
    ) -> Literal["walk", "sit_to_stand", "flexion_extension"]:
        """Extract and normalize maneuver type from AOA UID.

        Converts from UID format (CamelCase) to internal format
        (snake_case). E.g. "Walk" -> "walk",
        "SitToStand" -> "sit_to_stand".
        """
        maneuver_raw = "".join(
            filter(str.isalpha, uid.split("_")[1]),
        ).lower()

        maneuver_map: dict[
            str,
            Literal["walk", "sit_to_stand", "flexion_extension"],
        ] = {
            "walk": "walk",
            "sittostand": "sit_to_stand",
            "sitstand": "sit_to_stand",
            "flexext": "flexion_extension",
        }

        maneuver = maneuver_map.get(maneuver_raw)
        if maneuver is None:
            raise ValueError(
                f"Unknown maneuver '{maneuver_raw}' in UID '{uid}'"
            )
        return maneuver

    @staticmethod
    def _extract_walking_pass_info(
        uid: str,
    ) -> tuple[int, Literal["slow", "normal", "fast"]]:
        """Extract pass number and speed from AOA walking UID.

        Parses the penultimate ``_``-delimited token (e.g. ``NSP1``)
        into a pass number and speed name.
        """
        reverse_speed_map: dict[
            str, Literal["slow", "normal", "fast"]
        ] = {
            "SS": "slow",
            "NS": "normal",
            "FS": "fast",
        }

        pass_info = uid.split("_")[-2]
        pass_number = int("".join(filter(str.isdigit, pass_info)))

        pass_speed_code = "".join(filter(str.isalpha, pass_info))
        pass_speed_code = pass_speed_code.replace("P", "").upper()

        speed = reverse_speed_map.get(pass_speed_code)
        if speed is None:
            raise ValueError(
                f"Unknown speed code '{pass_speed_code}' "
                f"in pass info '{pass_info}'"
            )
        return pass_number, speed
