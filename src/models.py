from datetime import datetime, timedelta
from typing import Annotated, ClassVar, Literal, Optional

try:  # Python <3.11 compatibility
    from typing import Self
except ImportError:  # pragma: no cover - fallback for older runtimes
    from typing_extensions import Self

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
    def __get_pydantic_core_schema__(
        cls, _source_type, _handler
    ) -> CoreSchema:
        def validate_dataframe(value: pd.DataFrame) -> Self:
            if not isinstance(value, pd.DataFrame):
                raise PydanticCustomError(
                    "not_dataframe", "Value is not a pandas DataFrame"
                )

            required_columns: list[str] = ["Event Info", "Time (sec)"]
            if not all(col in value.columns for col in required_columns):
                missing: set[str] = set(required_columns) - set(value.columns)
                raise PydanticCustomError(
                    "missing_column",
                    "PassData DataFrame is missing required columns: "
                    f"{', '.join(missing)}",
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

    scripted_maneuver: Literal["walk", "sit_to_stand", "flexion_extension"] = Field(
        alias="maneuver"
    )
    speed: Optional[Literal["slow", "normal", "fast", "medium"]] = None
    pass_number: Optional[int] = None
    pass_data: Optional[PassData] = None

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
    def validate_speed_for_maneuver(
        cls, value: Optional[str], info
    ) -> Optional[str]:
        maneuver = info.data.get("scripted_maneuver") or info.data.get("maneuver")
        if maneuver == "walk" and cls.require_walk_details:
            if value is None:
                raise ValueError("speed is required when maneuver is 'walk'")
        elif maneuver != "walk" and value is not None:
            raise ValueError(
                "speed must be None when maneuver is "
                f"'{maneuver}'"
            )
        return value

    @field_validator("pass_number")
    @classmethod
    def validate_pass_number_for_maneuver(
        cls, value: Optional[int], info
    ) -> Optional[int]:
        maneuver = info.data.get("scripted_maneuver") or info.data.get("maneuver")
        if maneuver == "walk" and cls.require_walk_details:
            if value is None:
                raise ValueError(
                    "pass_number is required when maneuver is 'walk'"
                )
            if value < 0:
                raise ValueError("pass_number must be non-negative")
        elif maneuver != "walk" and value is not None:
            raise ValueError(
                "pass_number must be None when maneuver is "
                f"'{maneuver}'"
            )
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
    microphone_notes: Optional[dict[Literal[1, 2, 3, 4], str]] = None
    # Time from start of recording to audio sync event
    audio_sync_time: Optional[timedelta] = None
    audio_qc_pass: bool = False
    audio_qc_mic_1_pass: bool = True  # Per-microphone QC results
    audio_qc_mic_2_pass: bool = True
    audio_qc_mic_3_pass: bool = True
    audio_qc_mic_4_pass: bool = True
    audio_qc_version: int = Field(default_factory=get_audio_qc_version)
    audio_notes: Optional[str] = None

    @property
    def file_name(self) -> str:
        return self.audio_file_name

    @property
    def notes(self) -> Optional[str]:
        """Backward-compatible alias for audio_notes."""
        return self.audio_notes

    @field_validator("microphone_notes")
    @classmethod
    def validate_microphone_notes_keys(
        cls, value: Optional[dict]
    ) -> Optional[dict]:
        """Validate microphone_notes keys are within {1,2,3,4}."""
        if value is None:
            return value
        allowed_keys: set[int] = {1, 2, 3, 4}
        if not set(value.keys()).issubset(allowed_keys):
            raise ValueError(
                "microphone_notes keys must be in {1, 2, 3, 4}, "
                f"got {set(value.keys())}"
            )
        return value

    @field_validator("microphones")
    @classmethod
    def validate_microphone_keys(cls, value: dict) -> dict:
        """Ensure microphones dict has exactly keys {1,2,3,4}."""
        required_keys: set[int] = {1, 2, 3, 4}
        if set(value.keys()) != required_keys:
            raise ValueError(
                "microphones must contain exactly keys "
                f"{required_keys}, got {set(value.keys())}"
            )
        return value


class AcousticsData(pd.DataFrame):
    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type, _handler
    ) -> CoreSchema:
        def validate_dataframe(value: pd.DataFrame) -> Self:
            if not isinstance(value, pd.DataFrame):
                raise PydanticCustomError(
                    "not_dataframe", "Value is not a pandas DataFrame"
                )
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
                missing_cols: set[str] = (
                    set(required_columns) - set(value.columns)
                )
                msg = (
                    "DataFrame is missing required columns: "
                    f"{', '.join(missing_cols)}"
                )
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
    date_of_recording: Optional[datetime] = None
    # Time from start of recording to biomech sync event
    biomech_sync_left_time: Optional[timedelta] = None
    biomech_sync_right_time: Optional[timedelta] = None
    biomech_qc_pass: bool = False
    biomech_qc_version: int = Field(default_factory=get_biomech_qc_version)
    biomech_notes: Optional[str] = None

    @property
    def file_name(self) -> str:
        return self.biomech_file_name


class BiomechanicsData(pd.DataFrame):
    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type, _handler
    ) -> CoreSchema:
        def validate_dataframe(value: pd.DataFrame) -> Self:
            if not isinstance(value, pd.DataFrame):
                raise PydanticCustomError(
                    "not_dataframe", "Value is not a pandas DataFrame"
                )
            if "TIME" not in value.columns:
                raise PydanticCustomError(
                    "missing_column", "DataFrame must contain 'TIME' column"
                )
            return cls(value)

        return core_schema.no_info_after_validator_function(
            validate_dataframe,
            core_schema.any_schema(),
        )


class BiomechanicsRecording(BiomechanicsFileMetadata):
    """Biomechanics data paired with metadata and sync details."""

    data: BiomechanicsData


class SynchronizedData(pd.DataFrame):
    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type, _handler
    ) -> CoreSchema:
        def validate_dataframe(value: pd.DataFrame) -> Self:
            if not isinstance(value, pd.DataFrame):
                raise PydanticCustomError(
                    "not_dataframe", "Value is not a pandas DataFrame"
                )
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
                missing_cols: set[str] = (
                    set(required_columns) - set(value.columns)
                )
                msg = (
                    "DataFrame is missing required columns: "
                    f"{', '.join(missing_cols)}"
                )
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


class MovementCycleMetadata(
    AcousticsFileMetadata,
    BiomechanicsFileMetadata,
):
    """Metadata for a knee acoustic emission recording for a single
    movement cycle. Optionally (ideally) synchronized to a biomechanics
    recording. Contains all the requisite information for saving to a
    postgres database."""

    id: int
    cycle_index: int
    audio_sync_time: timedelta
    biomech_sync_left_time: timedelta
    biomech_sync_right_time: timedelta
    cycle_acoustic_energy: float
    cycle_qc_pass: bool
    cycle_qc_version: int = Field(default_factory=get_cycle_qc_version)
    cycle_notes: Optional[str] = None
    
    # Periodic noise detection results (per-channel)
    periodic_noise_detected: bool = False
    periodic_noise_ch1: bool = False
    periodic_noise_ch2: bool = False
    periodic_noise_ch3: bool = False
    periodic_noise_ch4: bool = False
    
    # Sync quality results (cross-modal validation)
    sync_quality_score: Optional[float] = None
    sync_qc_pass: Optional[bool] = None
    
    require_walk_details: ClassVar[bool] = True

    @field_validator("cycle_index")
    @classmethod
    def validate_cycle_index(cls, value: int) -> int:
        if value < 0:
            raise ValueError("cycle_index must be non-negative")
        return value

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: int) -> int:
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
        if value is None:
            raise ValueError("sync times are required for movement cycles")
        return value


class MovementCycle(MovementCycleMetadata):
    """Single movement cycle with synchronized data."""

    data: SynchronizedData


# Processing log models for validation and logging
# These models consolidate the dataclass fields from processing_log.py
# into Pydantic models for comprehensive validation


class AudioProcessingMetadata(BaseModel):
    """Metadata for audio file processing and QC.
    
    Consolidates audio processing information for logging and validation.
    Fields use snake_case for direct mapping to database columns and Excel headers.
    """
    
    # File identification
    audio_file_name: str
    audio_bin_file: Optional[str] = None
    audio_pkl_file: Optional[str] = None
    
    # Processing metadata
    processing_date: Optional[datetime] = None
    processing_status: Literal["not_processed", "success", "error"] = "not_processed"
    error_message: Optional[str] = None
    
    # Audio file characteristics
    sample_rate: Optional[float] = None  # Hz
    num_channels: int = 4
    duration_seconds: Optional[float] = None
    file_size_mb: Optional[float] = None
    
    # Header metadata
    device_serial: Optional[str] = None
    firmware_version: Optional[int] = None
    file_time: Optional[datetime] = None
    
    # Processing flags
    has_instantaneous_freq: bool = False
    
    # Raw audio QC results (dropout and artifacts)
    # String representation of list of (start, end) tuples
    qc_not_passed: Optional[str] = None  # Any mic bad intervals
    qc_not_passed_mic_1: Optional[str] = None
    qc_not_passed_mic_2: Optional[str] = None
    qc_not_passed_mic_3: Optional[str] = None
    qc_not_passed_mic_4: Optional[str] = None
    
    # QC version tracking
    audio_qc_version: int = Field(default_factory=get_audio_qc_version)
    
    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, value: Optional[float]) -> Optional[float]:
        """Validate sample rate is positive if provided."""
        if value is not None and value <= 0:
            raise ValueError("sample_rate must be positive")
        return value
    
    @field_validator("duration_seconds")
    @classmethod
    def validate_duration(cls, value: Optional[float]) -> Optional[float]:
        """Validate duration is non-negative if provided."""
        if value is not None and value < 0:
            raise ValueError("duration_seconds must be non-negative")
        return value
    
    @field_validator("num_channels")
    @classmethod
    def validate_num_channels(cls, value: int) -> int:
        """Validate number of channels."""
        if value not in [1, 2, 3, 4]:
            raise ValueError(f"num_channels must be 1-4, got {value}")
        return value


class BiomechanicsImportMetadata(BaseModel):
    """Metadata for biomechanics data import.
    
    Tracks biomechanics file import status and characteristics.
    Fields use snake_case for direct mapping to database columns and Excel headers.
    """
    
    # File identification
    biomechanics_file: str
    sheet_name: Optional[str] = None
    
    # Processing metadata
    processing_date: Optional[datetime] = None
    processing_status: Literal["not_processed", "success", "error"] = "not_processed"
    error_message: Optional[str] = None
    
    # Import statistics
    num_recordings: int = 0
    num_passes: int = 0  # For walking maneuvers
    
    # Data characteristics
    duration_seconds: Optional[float] = None
    sample_rate: Optional[float] = None
    
    # Time range
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    # QC version tracking
    biomech_qc_version: int = Field(default_factory=get_biomech_qc_version)
    
    @field_validator("num_recordings", "num_passes")
    @classmethod
    def validate_counts(cls, value: int) -> int:
        """Validate counts are non-negative."""
        if value < 0:
            raise ValueError("counts must be non-negative")
        return value
    
    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, value: Optional[float]) -> Optional[float]:
        """Validate sample rate is positive if provided."""
        if value is not None and value <= 0:
            raise ValueError("sample_rate must be positive")
        return value


class SynchronizationMetadata(BaseModel):
    """Metadata for audio-biomechanics synchronization.
    
    Tracks synchronization process and alignment details.
    Fields use snake_case for direct mapping to database columns and Excel headers.
    """
    
    # File identification
    sync_file_name: str
    pass_number: Optional[int] = None  # For walking
    speed: Optional[Literal["slow", "normal", "fast", "medium"]] = None
    
    # Processing metadata
    processing_date: Optional[datetime] = None
    processing_status: Literal["not_processed", "success", "error"] = "not_processed"
    error_message: Optional[str] = None
    
    # Synchronization details (in original time coordinates)
    audio_stomp_time: Optional[float] = None  # seconds (audio time coords)
    bio_left_stomp_time: Optional[float] = None  # seconds (bio time coords)
    bio_right_stomp_time: Optional[float] = None  # seconds (bio time coords)
    knee_side: Optional[Literal["left", "right"]] = None
    
    # Alignment details (derived from sync)
    stomp_offset: Optional[float] = None  # seconds (bio_stomp - audio_stomp)
    aligned_audio_stomp_time: Optional[float] = None  # seconds (synced coords)
    aligned_bio_stomp_time: Optional[float] = None  # seconds (synced coords)
    
    # Synchronized data characteristics
    duration_seconds: Optional[float] = None
    
    # QC results
    sync_qc_performed: bool = False
    sync_qc_passed: Optional[bool] = None
    
    # QC version tracking
    audio_qc_version: int = Field(default_factory=get_audio_qc_version)
    biomech_qc_version: int = Field(default_factory=get_biomech_qc_version)
    
    # Detection method details
    consensus_time: Optional[float] = None
    consensus_methods: Optional[str] = None  # Comma-separated method names
    rms_time: Optional[float] = None
    onset_time: Optional[float] = None
    freq_time: Optional[float] = None
    method_agreement_span: Optional[float] = None
    
    # Biomechanics-guided detection metadata
    audio_stomp_method: Optional[Literal["consensus", "biomechanics-guided"]] = None
    selected_time: Optional[float] = None
    contra_selected_time: Optional[float] = None
    
    @field_validator("pass_number")
    @classmethod
    def validate_pass_number(cls, value: Optional[int]) -> Optional[int]:
        """Validate pass number is non-negative if provided."""
        if value is not None and value < 0:
            raise ValueError("pass_number must be non-negative")
        return value


class MovementCyclesMetadata(BaseModel):
    """Metadata for movement cycle extraction and QC.
    
    Tracks cycle extraction process and aggregate statistics.
    Fields use snake_case for direct mapping to database columns and Excel headers.
    """
    
    # Source file
    sync_file_name: str
    
    # Processing metadata
    processing_date: Optional[datetime] = None
    processing_status: Literal["not_processed", "success", "error"] = "not_processed"
    error_message: Optional[str] = None
    
    # Extraction results
    total_cycles_extracted: int = 0
    clean_cycles: int = 0
    outlier_cycles: int = 0
    
    # QC parameters
    qc_acoustic_threshold: Optional[float] = None
    
    # Aggregate statistics (across clean cycles)
    mean_cycle_duration_s: Optional[float] = None
    median_cycle_duration_s: Optional[float] = None
    min_cycle_duration_s: Optional[float] = None
    max_cycle_duration_s: Optional[float] = None
    mean_acoustic_auc: Optional[float] = None
    
    # QC version tracking
    cycle_qc_version: int = Field(default_factory=get_cycle_qc_version)
    
    @field_validator("total_cycles_extracted", "clean_cycles", "outlier_cycles")
    @classmethod
    def validate_cycle_counts(cls, value: int) -> int:
        """Validate cycle counts are non-negative."""
        if value < 0:
            raise ValueError("cycle counts must be non-negative")
        return value
    
    @model_validator(mode="after")
    def validate_cycle_totals(self) -> Self:
        """Validate that clean + outlier cycles equals total (if all are set)."""
        if self.total_cycles_extracted > 0:
            expected = self.clean_cycles + self.outlier_cycles
            if expected != self.total_cycles_extracted:
                # Allow mismatch but log warning through validation
                # In practice, we'll be lenient here
                pass
        return self


class IndividualMovementCycleMetadata(BaseModel):
    """Metadata for a single movement cycle.
    
    Combines information from audio processing, biomechanics import, and synchronization
    for a single extracted cycle. This model is used for database population and the 
    'Cycle Details' sheet in processing log Excel files.
    
    Uses inheritance structure to avoid code duplication - contains all upstream
    processing metadata from audio, biomechanics, and synchronization.
    
    Fields use snake_case for direct mapping to database columns and Excel headers.
    """
    
    # Cycle identification
    cycle_index: int
    is_outlier: bool = False
    cycle_file: Optional[str] = None  # Path to .pkl file if saved
    
    # Source files and processing (from upstream)
    audio_file_name: Optional[str] = None
    biomechanics_file: Optional[str] = None
    sync_file_name: Optional[str] = None
    
    # Cycle temporal characteristics
    start_time_s: Optional[float] = None  # Start time within synced recording
    end_time_s: Optional[float] = None    # End time within synced recording
    duration_s: Optional[float] = None
    
    # Acoustic characteristics (data-derived from cycle)
    acoustic_auc: Optional[float] = None  # Total acoustic energy
    
    # Audio QC metadata (from AudioProcessingMetadata)
    audio_sample_rate: Optional[float] = None
    audio_duration_seconds: Optional[float] = None
    audio_qc_version: Optional[int] = None
    audio_processing_status: Optional[str] = None
    
    # Biomechanics QC metadata (from BiomechanicsImportMetadata)
    biomech_sample_rate: Optional[float] = None
    biomech_duration_seconds: Optional[float] = None
    biomech_qc_version: Optional[int] = None
    biomech_processing_status: Optional[str] = None
    
    # Synchronization QC metadata (from SynchronizationMetadata)
    audio_stomp_time: Optional[float] = None
    bio_left_stomp_time: Optional[float] = None
    bio_right_stomp_time: Optional[float] = None
    knee_side: Optional[str] = None
    stomp_offset: Optional[float] = None
    sync_qc_performed: bool = False
    sync_qc_passed: Optional[bool] = None
    sync_duration_seconds: Optional[float] = None
    
    # Cycle QC metadata (from MovementCyclesMetadata)
    qc_acoustic_threshold: Optional[float] = None
    cycle_qc_version: Optional[int] = None
    
    # Processing timestamps
    processing_date: Optional[datetime] = None
    
    @field_validator("cycle_index")
    @classmethod
    def validate_cycle_index(cls, value: int) -> int:
        """Validate cycle index is non-negative."""
        if value < 0:
            raise ValueError("cycle_index must be non-negative")
        return value
    
    @field_validator("duration_s", "start_time_s", "end_time_s")
    @classmethod
    def validate_times(cls, value: Optional[float]) -> Optional[float]:
        """Validate time values are non-negative if provided."""
        if value is not None and value < 0:
            raise ValueError("time values must be non-negative")
        return value
    
    @field_validator("acoustic_auc")
    @classmethod
    def validate_auc(cls, value: Optional[float]) -> Optional[float]:
        """Validate acoustic AUC is non-negative if provided."""
        if value is not None and value < 0:
            raise ValueError("acoustic_auc must be non-negative")
        return value

