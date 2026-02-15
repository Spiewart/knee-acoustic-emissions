from datetime import datetime, timedelta
from typing import Annotated, ClassVar, Literal

try:  # Python <3.11 compatibility
    from typing import Self
except ImportError:  # pragma: no cover - fallback for older runtimes
    from typing import Self

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import CoreSchema, PydanticCustomError, core_schema

from src.qc_versions import get_audio_qc_version, get_biomech_qc_version, get_cycle_qc_version


class StudyMetadata(BaseModel):
    """Metadata for an acoustic emission study."""

    study: str = Field(..., description="Study name/code")
    study_id: int = Field(..., description="Numeric study identifier")


class MicrophonePosition(BaseModel):
    """Metadata describing a microphone position on the knee."""

    patellar_position: Literal["Infrapatellar", "Suprapatellar"]
    laterality: Literal["Medial", "Lateral"]


class KneeMetadata(BaseModel):
    """Metadata for a knee in an acoustic emission study."""

    knee: Literal["left", "right"]


class PassData(pd.DataFrame):
    """Pass/event metadata exported from biomechanics processing."""

    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type, _handler) -> CoreSchema:
        def validate_dataframe(value: pd.DataFrame) -> Self:  # type: ignore[type-var]
            if not isinstance(value, pd.DataFrame):
                raise PydanticCustomError("not_dataframe", "Value is not a pandas DataFrame")

            required_columns: list[str] = ["Event Info", "Time (sec)"]
            if not all(col in value.columns for col in required_columns):
                missing: set[str] = set(required_columns) - set(value.columns)
                raise PydanticCustomError(
                    "missing_column",
                    f"PassData DataFrame is missing required columns: {', '.join(missing)}",
                )
            return cls(value)

        return core_schema.no_info_after_validator_function(
            validate_dataframe,
            core_schema.any_schema(),
        )


class ScriptedManeuverMetadata(BaseModel):
    """Common fields for a scripted maneuver.

    `scripted_maneuver` is the canonical field name, but `maneuver` is
    accepted as an alias for backwards compatibility.
    """

    model_config = ConfigDict(populate_by_name=True)
    require_walk_details: ClassVar[bool] = False

    scripted_maneuver: Literal["walk", "sit_to_stand", "flexion_extension"] = Field(alias="maneuver")
    speed: Literal["slow", "normal", "fast", "medium"] | None = None
    pass_number: int | None = None
    pass_data: PassData | None = None

    @model_validator(mode="before")
    @classmethod
    def harmonize_maneuver_alias(cls, values: dict) -> dict:
        """Allow both `scripted_maneuver` and its alias `maneuver`.

        If both are provided they must agree; otherwise fill the canonical
        `scripted_maneuver` field from the alias.
        """
        alias_val = values.get("maneuver")
        canonical_val = values.get("scripted_maneuver")
        if alias_val is not None and canonical_val is not None and alias_val != canonical_val:
            raise ValueError("maneuver and scripted_maneuver must match if both are provided")
        if canonical_val is None and alias_val is not None:
            values["scripted_maneuver"] = alias_val
        return values

    @property
    def maneuver(self) -> Literal["walk", "sit_to_stand", "flexion_extension"]:
        return self.scripted_maneuver

    @field_validator("speed")
    @classmethod
    def validate_speed_for_maneuver(cls, value: str | None, info) -> str | None:
        maneuver = info.data.get("scripted_maneuver") or info.data.get("maneuver")
        if maneuver == "walk" and cls.require_walk_details:
            if value is None:
                raise ValueError("speed is required when maneuver is 'walk'")
        elif maneuver != "walk" and value is not None:
            raise ValueError(f"speed must be None when maneuver is '{maneuver}'")
        return value

    @field_validator("pass_number")
    @classmethod
    def validate_pass_number_for_maneuver(cls, value: int | None, info) -> int | None:
        maneuver = info.data.get("scripted_maneuver") or info.data.get("maneuver")
        if maneuver == "walk" and cls.require_walk_details:
            if value is None:
                raise ValueError("pass_number is required when maneuver is 'walk'")
            if value < 0:
                raise ValueError("pass_number must be non-negative")
        elif maneuver != "walk" and value is not None:
            raise ValueError(f"pass_number must be None when maneuver is '{maneuver}'")
        return value


class AcousticsFileMetadata(
    ScriptedManeuverMetadata,
    KneeMetadata,
    StudyMetadata,
):
    """Metadata for an acoustics recording."""

    model_config = ConfigDict(populate_by_name=True)

    audio_file_name: str = Field(alias="file_name")
    audio_serial_number: str = "unknown"
    audio_firmware_version: int = 0
    date_of_recording: datetime = Field(default_factory=lambda: datetime.min)
    microphones: Annotated[
        dict[Literal[1, 2, 3, 4], MicrophonePosition],
        "Microphone index (1-4) mapped to position metadata",
    ]
    microphone_notes: dict[Literal[1, 2, 3, 4], str] | None = None
    # Time from start of recording to audio sync event
    audio_sync_time: timedelta | None = None
    audio_qc_pass: bool = False
    audio_qc_mic_1_pass: bool = True  # Per-microphone QC results
    audio_qc_mic_2_pass: bool = True
    audio_qc_mic_3_pass: bool = True
    audio_qc_mic_4_pass: bool = True
    audio_qc_version: int = Field(default_factory=get_audio_qc_version)
    audio_notes: str | None = None

    @property
    def file_name(self) -> str:
        return self.audio_file_name

    @property
    def notes(self) -> str | None:
        """Backward-compatible alias for audio_notes."""
        return self.audio_notes

    @field_validator("microphone_notes")
    @classmethod
    def validate_microphone_notes_keys(cls, value: dict | None) -> dict | None:
        """Validate microphone_notes keys are within {1,2,3,4}."""
        if value is None:
            return value
        allowed_keys: set[int] = {1, 2, 3, 4}
        if not set(value.keys()).issubset(allowed_keys):
            raise ValueError(f"microphone_notes keys must be in {{1, 2, 3, 4}}, got {set(value.keys())}")
        return value

    @field_validator("microphones")
    @classmethod
    def validate_microphone_keys(cls, value: dict) -> dict:
        """Ensure microphones dict has exactly keys {1,2,3,4}."""
        required_keys: set[int] = {1, 2, 3, 4}
        if set(value.keys()) != required_keys:
            raise ValueError(f"microphones must contain exactly keys {required_keys}, got {set(value.keys())}")
        return value


class AcousticsData(pd.DataFrame):
    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type, _handler) -> CoreSchema:
        def validate_dataframe(value: pd.DataFrame) -> Self:  # type: ignore[type-var]
            if not isinstance(value, pd.DataFrame):
                raise PydanticCustomError("not_dataframe", "Value is not a pandas DataFrame")
            required_columns: list[str] = [
                "tt",
                "ch1",
                "ch2",
                "ch3",
                "ch4",
                "f_ch1",
                "f_ch2",
                "f_ch3",
                "f_ch4",
            ]
            if not all(col in value.columns for col in required_columns):
                missing_cols: set[str] = set(required_columns) - set(value.columns)
                msg = f"DataFrame is missing required columns: {', '.join(missing_cols)}"
                raise PydanticCustomError("missing_column", msg)
            return cls(value)

        return core_schema.no_info_after_validator_function(
            validate_dataframe,
            core_schema.any_schema(),
        )


class AcousticsRecording(AcousticsFileMetadata):
    """Acoustics data paired with its metadata."""

    data: AcousticsData


class BiomechanicsFileMetadata(ScriptedManeuverMetadata, StudyMetadata):
    """Metadata for a biomechanics recording."""

    model_config = ConfigDict(populate_by_name=True)
    require_walk_details: ClassVar[bool] = True

    biomech_file_name: str = Field(alias="file_name")
    biomech_system: Literal["Vicon", "Qualisys"] = "Qualisys"
    date_of_recording: datetime | None = None
    # Time from start of recording to biomech sync event
    biomech_sync_left_time: timedelta | None = None
    biomech_sync_right_time: timedelta | None = None
    biomech_qc_pass: bool = False
    biomech_qc_version: int = Field(default_factory=get_biomech_qc_version)
    biomech_notes: str | None = None

    @property
    def file_name(self) -> str:
        return self.biomech_file_name


class BiomechanicsData(pd.DataFrame):
    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type, _handler) -> CoreSchema:
        def validate_dataframe(value: pd.DataFrame) -> Self:  # type: ignore[type-var]
            if not isinstance(value, pd.DataFrame):
                raise PydanticCustomError("not_dataframe", "Value is not a pandas DataFrame")
            if "TIME" not in value.columns:
                raise PydanticCustomError("missing_column", "DataFrame must contain 'TIME' column")
            return cls(value)

        return core_schema.no_info_after_validator_function(
            validate_dataframe,
            core_schema.any_schema(),
        )


class BiomechanicsRecording(BiomechanicsFileMetadata):
    """Biomechanics data paired with metadata and sync details."""

    data: BiomechanicsData


class FullMovementCycleMetadata(
    AcousticsFileMetadata,
    BiomechanicsFileMetadata,
):
    """Metadata for a knee acoustic emission recording for a single movement cycle.

    This is a complete model that inherits from file metadata classes
    (AcousticsFileMetadata and BiomechanicsFileMetadata) and contains all
    the requisite information for saving to a postgres database.

    Note: This is different from MovementCycle in metadata.py, which is used for
    processing log metadata with embedded upstream metadata objects.
    FullMovementCycleMetadata inherits all fields from parent classes
    and is used in synchronization quality control and data processing workflows.
    """

    # Core cycle identification
    id: int
    cycle_index: int

    # Sync times (inherited fields made required)
    audio_sync_time: timedelta
    biomech_sync_left_time: timedelta
    biomech_sync_right_time: timedelta

    # Cycle-specific measurements
    cycle_acoustic_energy: float
    cycle_qc_pass: bool
    cycle_qc_version: int = Field(default_factory=get_cycle_qc_version)
    cycle_notes: str | None = None

    # Periodic noise detection results (per-channel)
    periodic_noise_detected: bool = False
    periodic_noise_ch1: bool = False
    periodic_noise_ch2: bool = False
    periodic_noise_ch3: bool = False
    periodic_noise_ch4: bool = False

    # Sync quality results (cross-modal validation)
    sync_quality_score: float | None = None
    sync_qc_pass: bool | None = None

    require_walk_details: ClassVar[bool] = True

    @field_validator("cycle_index")
    @classmethod
    def validate_cycle_index(cls, value: int) -> int:
        """Validate cycle index is non-negative."""
        if value < 0:
            raise ValueError("cycle_index must be non-negative")
        return value

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: int) -> int:
        """Validate id is non-negative."""
        if value < 0:
            raise ValueError("id must be non-negative")
        return value

    @field_validator(
        "audio_sync_time",
        "biomech_sync_left_time",
        "biomech_sync_right_time",
    )
    @classmethod
    def validate_sync_times(cls, value: timedelta) -> timedelta:
        """Validate sync times are provided (required for movement cycles)."""
        if value is None:
            raise ValueError("sync times are required for movement cycles")
        return value


class SynchronizedData(pd.DataFrame):
    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type, _handler) -> CoreSchema:
        def validate_dataframe(value: pd.DataFrame) -> Self:  # type: ignore[type-var]
            if not isinstance(value, pd.DataFrame):
                raise PydanticCustomError("not_dataframe", "Value is not a pandas DataFrame")
            required_columns: list[str] = [
                "tt",
                "ch1",
                "ch2",
                "ch3",
                "ch4",
                "f_ch1",
                "f_ch2",
                "f_ch3",
                "f_ch4",
                "TIME",
            ]
            if not all(col in value.columns for col in required_columns):
                missing_cols: set[str] = set(required_columns) - set(value.columns)
                msg = f"DataFrame is missing required columns: {', '.join(missing_cols)}"
                raise PydanticCustomError("missing_column", msg)
            return cls(value)

        return core_schema.no_info_after_validator_function(
            validate_dataframe,
            core_schema.any_schema(),
        )


class SynchronizedRecording(
    AcousticsFileMetadata,
    BiomechanicsFileMetadata,
):
    """Synchronized acoustics and biomechanics data."""

    require_walk_details: ClassVar[bool] = True

    data: SynchronizedData


# Note: FullMovementCycleMetadata has been moved to src/metadata.py as a Pydantic @dataclass
# Import it from there: from src.metadata import FullMovementCycleMetadata


class MovementCycle(BaseModel):
    """Single movement cycle with synchronized data.

    This class combines FullMovementCycleMetadata (now in src/metadata.py) with
    the synchronized data slice for the cycle. It's used in synchronization QC
    workflows where the actual data needs to be carried along with metadata.

    For metadata-only operations, use FullMovementCycleMetadata from src/metadata.py."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Core cycle identification
    id: int
    cycle_index: int

    # Study metadata (from StudyMetadata)
    study: str
    study_id: int

    # Knee metadata (from KneeMetadata)
    knee: Literal["left", "right"]

    # Scripted maneuver metadata (from ScriptedManeuverMetadata)
    scripted_maneuver: Literal["walk", "sit_to_stand", "flexion_extension"]
    speed: Literal["slow", "normal", "fast", "medium"] | None = None
    pass_number: int | None = None

    # Acoustics file metadata fields
    audio_file_name: str
    audio_serial_number: str = "unknown"
    audio_firmware_version: int = 0
    date_of_recording: datetime = Field(default_factory=lambda: datetime.min)
    microphones: dict[Literal[1, 2, 3, 4], MicrophonePosition] | None = None
    microphone_notes: dict[Literal[1, 2, 3, 4], str] | None = None
    audio_qc_pass: bool = False
    audio_qc_mic_1_pass: bool = True
    audio_qc_mic_2_pass: bool = True
    audio_qc_mic_3_pass: bool = True
    audio_qc_mic_4_pass: bool = True
    audio_qc_version: int = Field(default_factory=get_audio_qc_version)
    audio_notes: str | None = None

    # Biomechanics file metadata fields
    biomech_file_name: str
    biomech_system: Literal["Vicon", "Qualisys"] = "Qualisys"
    biomech_qc_pass: bool = False
    biomech_qc_version: int = Field(default_factory=get_biomech_qc_version)
    biomech_notes: str | None = None

    # Sync times
    audio_sync_time: timedelta
    biomech_sync_left_time: timedelta
    biomech_sync_right_time: timedelta

    # Cycle measurements
    cycle_acoustic_energy: float
    cycle_qc_pass: bool
    cycle_qc_version: int = Field(default_factory=get_cycle_qc_version)
    cycle_notes: str | None = None

    # Periodic noise detection
    periodic_noise_detected: bool = False
    periodic_noise_ch1: bool = False
    periodic_noise_ch2: bool = False
    periodic_noise_ch3: bool = False
    periodic_noise_ch4: bool = False

    # Sync quality
    sync_quality_score: float | None = None
    sync_qc_pass: bool | None = None

    # The synchronized data for this cycle
    data: SynchronizedData

    @field_validator("cycle_index")
    @classmethod
    def validate_cycle_index(cls, value: int) -> int:
        """Validate cycle index is non-negative."""
        if value < 0:
            raise ValueError("cycle_index must be non-negative")
        return value

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: int) -> int:
        """Validate id is non-negative."""
        if value < 0:
            raise ValueError("id must be non-negative")
        return value
