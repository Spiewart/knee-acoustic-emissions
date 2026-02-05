"""Pydantic dataclasses for normalized database schema.

NORMALIZED STRUCTURE - Uses foreign key references instead of inheritance.

Each model represents a standalone entity:
- AudioProcessing: Audio file metadata + QC
- BiomechanicsImport: Biomechanics import metadata + import stats
- Synchronization: Audio-biomechanics sync results (FK to both)
- MovementCycle: Individual extracted cycle (FK to audio, optional biomech, optional sync)

This structure eliminates data redundancy and enables efficient relational queries.
"""

import dataclasses
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, field_validator, model_validator
from pydantic.dataclasses import dataclass

from src.qc_versions import (
    get_audio_qc_version,
    get_biomech_qc_version,
    get_cycle_qc_version,
)

# ============================================================================
# BASE CLASS - Keep minimal
# ============================================================================

@dataclass(kw_only=True)
class StudyMetadata:
    """Base metadata class containing study information.

    Provides consistent study and participant identification across all models.
    """

    study: Literal["AOA", "preOA", "SMoCK"]
    study_id: int  # Participant ID within the study (e.g., 1011 from #AOA1011)

    @field_validator("study_id")
    @classmethod
    def validate_study_id(cls, value: int) -> int:
        """Validate study ID is non-negative."""
        if value < 0:
            raise ValueError("study_id must be non-negative")
        return value


# ============================================================================
# STANDALONE MODELS - No inheritance beyond StudyMetadata
# ============================================================================

@dataclass(kw_only=True)
class AudioProcessing(StudyMetadata):
    """Audio processing metadata.

    Represents an audio file with all associated QC information.
    Does NOT contain biomechanics or synchronization data.

    Optional FK to biomechanics_import_id indicates which biomechanics file
    was recorded simultaneously with this audio (if any).
    """

    # ===== Foreign Key References (optional, from recording time) =====
    biomechanics_import_id: Optional[int] = None  # Biomechanics recorded with this audio

    # ===== File Identification =====
    audio_file_name: str = Field(...)
    device_serial: str = Field(...)
    firmware_version: int = Field(...)
    file_time: datetime = Field(...)
    file_size_mb: float = Field(...)

    # ===== Recording Metadata =====
    recording_date: datetime = Field(...)
    recording_time: datetime = Field(...)
    recording_timezone: str = Field(default="UTC")

    # ===== Maneuver Identification =====
    knee: Literal["right", "left"] = Field(...)
    maneuver: Literal["fe", "sts", "walk"] = Field(...)

    # ===== Audio Characteristics =====
    num_channels: int = Field(...)
    sample_rate: float = 46875.0

    # ===== Microphone Positions =====
    mic_1_position: Literal["IPM", "IPL", "SPM", "SPL"] = Field(...)
    mic_2_position: Literal["IPM", "IPL", "SPM", "SPL"] = Field(...)
    mic_3_position: Literal["IPM", "IPL", "SPM", "SPL"] = Field(...)
    mic_4_position: Literal["IPM", "IPL", "SPM", "SPL"] = Field(...)

    # ===== Optional Notes =====
    mic_1_notes: Optional[str] = None
    mic_2_notes: Optional[str] = None
    mic_3_notes: Optional[str] = None
    mic_4_notes: Optional[str] = None
    notes: Optional[str] = None

    # ===== Pickle File Storage =====
    pkl_file_path: Optional[str] = None

    # ===== QC Version =====
    audio_qc_version: int = Field(default_factory=get_audio_qc_version)

    # ===== QC Fail Segments (overall + per channel) =====
    qc_fail_segments: List[tuple] = Field(default_factory=list)
    qc_fail_segments_ch1: List[tuple] = Field(default_factory=list)
    qc_fail_segments_ch2: List[tuple] = Field(default_factory=list)
    qc_fail_segments_ch3: List[tuple] = Field(default_factory=list)
    qc_fail_segments_ch4: List[tuple] = Field(default_factory=list)

    # ===== Signal Dropout QC =====
    qc_signal_dropout: bool = False
    qc_signal_dropout_segments: List[tuple] = Field(default_factory=list)
    qc_signal_dropout_ch1: bool = False
    qc_signal_dropout_segments_ch1: List[tuple] = Field(default_factory=list)
    qc_signal_dropout_ch2: bool = False
    qc_signal_dropout_segments_ch2: List[tuple] = Field(default_factory=list)
    qc_signal_dropout_ch3: bool = False
    qc_signal_dropout_segments_ch3: List[tuple] = Field(default_factory=list)
    qc_signal_dropout_ch4: bool = False
    qc_signal_dropout_segments_ch4: List[tuple] = Field(default_factory=list)

    # ===== Artifact QC =====
    qc_artifact: bool = False
    qc_artifact_type: Optional[List[Literal["Intermittent", "Continuous"]]] = None
    qc_artifact_segments: List[tuple] = Field(default_factory=list)
    qc_artifact_ch1: bool = False
    qc_artifact_type_ch1: Optional[List[Literal["Intermittent", "Continuous"]]] = None
    qc_artifact_segments_ch1: List[tuple] = Field(default_factory=list)
    qc_artifact_ch2: bool = False
    qc_artifact_type_ch2: Optional[List[Literal["Intermittent", "Continuous"]]] = None
    qc_artifact_segments_ch2: List[tuple] = Field(default_factory=list)
    qc_artifact_ch3: bool = False
    qc_artifact_type_ch3: Optional[List[Literal["Intermittent", "Continuous"]]] = None
    qc_artifact_segments_ch3: List[tuple] = Field(default_factory=list)
    qc_artifact_ch4: bool = False
    qc_artifact_type_ch4: Optional[List[Literal["Intermittent", "Continuous"]]] = None
    qc_artifact_segments_ch4: List[tuple] = Field(default_factory=list)

    # ===== QC Status (auto-populated from segments) =====
    qc_not_passed: bool = False
    qc_not_passed_mic_1: bool = False
    qc_not_passed_mic_2: bool = False
    qc_not_passed_mic_3: bool = False
    qc_not_passed_mic_4: bool = False

    # ===== Processing Metadata =====
    processing_date: datetime = Field(default_factory=datetime.now)
    processing_status: Literal["not_processed", "success", "error"] = "not_processed"
    error_message: Optional[str] = None
    duration_seconds: Optional[float] = None

    # ===== Log Tracking =====
    log_updated: Optional[datetime] = None

    @model_validator(mode="after")
    def populate_qc_not_passed_fields(self) -> "AudioProcessing":
        """Auto-populate qc_not_passed boolean fields from qc_fail_segments."""
        self.qc_not_passed = len(self.qc_fail_segments) > 0
        self.qc_not_passed_mic_1 = len(self.qc_fail_segments_ch1) > 0
        self.qc_not_passed_mic_2 = len(self.qc_fail_segments_ch2) > 0
        self.qc_not_passed_mic_3 = len(self.qc_fail_segments_ch3) > 0
        self.qc_not_passed_mic_4 = len(self.qc_fail_segments_ch4) > 0
        return self

    @field_validator("biomechanics_import_id")
    @classmethod
    def validate_biomechanics_import_id(cls, value: Optional[int]) -> Optional[int]:
        """Validate biomechanics_import_id is positive if provided."""
        if value is not None and value <= 0:
            raise ValueError("biomechanics_import_id must be positive")
        return value

    @field_validator("recording_time")
    @classmethod
    def validate_recording_time_matches_date(cls, value: datetime, info) -> datetime:
        """Validate recording_time date component matches recording_date."""
        recording_date = info.data.get("recording_date")
        if recording_date and value.date() != recording_date.date():
            raise ValueError("recording_time date component must match recording_date")
        return value

    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, value: float) -> float:
        """Validate sample rate is positive."""
        if value <= 0:
            raise ValueError("sample_rate must be positive")
        return value

    @field_validator("num_channels")
    @classmethod
    def validate_num_channels(cls, value: int) -> int:
        """Validate number of channels."""
        if value not in [1, 2, 3, 4]:
            raise ValueError(f"num_channels must be 1-4, got {value}")
        return value

    @field_validator("file_size_mb")
    @classmethod
    def validate_file_size(cls, value: float) -> float:
        """Validate file size is non-negative."""
        if value < 0:
            raise ValueError("file_size_mb must be non-negative")
        return value

    @field_validator("firmware_version")
    @classmethod
    def validate_firmware_version(cls, value: int) -> int:
        """Validate firmware version is non-negative."""
        if value < 0:
            raise ValueError("firmware_version must be non-negative")
        return value

    @field_validator("duration_seconds")
    @classmethod
    def validate_duration(cls, value: Optional[float]) -> Optional[float]:
        """Validate duration is non-negative if provided."""
        if value is not None and value < 0:
            raise ValueError("duration_seconds must be non-negative")
        return value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        # Use dataclasses.asdict for Pydantic dataclasses
        result = dataclasses.asdict(self)
        # Add friendly column names for Excel export
        result.update({
            "Audio File": self.audio_file_name,
            "Device Serial": self.device_serial,
            "Firmware Version": self.firmware_version,
            "File Time": self.file_time,
            "File Size (MB)": self.file_size_mb,
            "Recording Date": self.recording_date,
            "Recording Time": self.recording_time,
            "Knee": self.knee,
            "Maneuver": self.maneuver,
            "Num Channels": self.num_channels,
            "Sample Rate (Hz)": self.sample_rate,
            "Mic 1 Position": self.mic_1_position,
            "Mic 2 Position": self.mic_2_position,
            "Mic 3 Position": self.mic_3_position,
            "Mic 4 Position": self.mic_4_position,
            "Processing Date": self.processing_date,
            "Processing Status": self.processing_status,
            "Error Message": self.error_message,
            "Duration (s)": self.duration_seconds,
            "Audio QC Version": self.audio_qc_version,
            "Audio QC Not Passed": self.qc_not_passed,
            "Biomechanics Import ID": self.biomechanics_import_id,
            # Raw field names for backwards compatibility
            "QC_not_passed": self.qc_not_passed,
            "QC_not_passed_mic_1": self.qc_not_passed_mic_1,
            "QC_not_passed_mic_2": self.qc_not_passed_mic_2,
            "QC_not_passed_mic_3": self.qc_not_passed_mic_3,
            "QC_not_passed_mic_4": self.qc_not_passed_mic_4,
            # QC fields
            "QC Fail Segments": self.qc_fail_segments,
            "QC Fail Segments Ch1": self.qc_fail_segments_ch1,
            "QC Fail Segments Ch2": self.qc_fail_segments_ch2,
            "QC Fail Segments Ch3": self.qc_fail_segments_ch3,
            "QC Fail Segments Ch4": self.qc_fail_segments_ch4,
            "QC Signal Dropout": self.qc_signal_dropout,
            "QC Signal Dropout Segments": self.qc_signal_dropout_segments,
            "QC Signal Dropout Ch1": self.qc_signal_dropout_ch1,
            "QC Signal Dropout Segments Ch1": self.qc_signal_dropout_segments_ch1,
            "QC Signal Dropout Ch2": self.qc_signal_dropout_ch2,
            "QC Signal Dropout Segments Ch2": self.qc_signal_dropout_segments_ch2,
            "QC Signal Dropout Ch3": self.qc_signal_dropout_ch3,
            "QC Signal Dropout Segments Ch3": self.qc_signal_dropout_segments_ch3,
            "QC Signal Dropout Ch4": self.qc_signal_dropout_ch4,
            "QC Signal Dropout Segments Ch4": self.qc_signal_dropout_segments_ch4,
            "QC Artifact": self.qc_artifact,
            "QC Artifact Type": self.qc_artifact_type,
            "QC Artifact Segments": self.qc_artifact_segments,
            "QC Artifact Ch1": self.qc_artifact_ch1,
            "QC Artifact Type Ch1": self.qc_artifact_type_ch1,
            "QC Artifact Segments Ch1": self.qc_artifact_segments_ch1,
            "QC Artifact Ch2": self.qc_artifact_ch2,
            "QC Artifact Type Ch2": self.qc_artifact_type_ch2,
            "QC Artifact Segments Ch2": self.qc_artifact_segments_ch2,
            "QC Artifact Ch3": self.qc_artifact_ch3,
            "QC Artifact Type Ch3": self.qc_artifact_type_ch3,
            "QC Artifact Segments Ch3": self.qc_artifact_segments_ch3,
            "QC Artifact Ch4": self.qc_artifact_ch4,
            "QC Artifact Type Ch4": self.qc_artifact_type_ch4,
            "QC Artifact Segments Ch4": self.qc_artifact_segments_ch4,
        })
        return result


@dataclass(kw_only=True)
class BiomechanicsImport(StudyMetadata):
    """Biomechanics import metadata.

    Represents a biomechanics file import with all associated metadata and QC.
    Does NOT contain synchronization data.

    Optional FK to audio_processing_id indicates which audio file was recorded
    simultaneously with this biomechanics data (if any).
    """

    # ===== Foreign Key References (optional, from recording time) =====
    audio_processing_id: Optional[int] = None  # Audio recorded with this biomechanics

    # ===== File Identification =====
    biomechanics_file: str = Field(...)
    sheet_name: Optional[str] = None
    biomechanics_type: Literal["Gonio", "IMU", "Motion Analysis"] = Field(...)

    # ===== Maneuver Identification =====
    knee: Literal["right", "left"] = Field(...)
    maneuver: Literal["fe", "sts", "walk"] = Field(...)

    # ===== Walk-Specific Metadata (optional, required for walk) =====
    pass_number: Optional[int] = None
    speed: Optional[Literal["slow", "fast", "medium", "comfortable"]] = None

    # ===== Biomechanics Characteristics =====
    biomechanics_sync_method: Literal["flick", "stomp"] = Field(...)
    biomechanics_sample_rate: float = Field(...)
    biomechanics_notes: Optional[str] = None

    # ===== Import Statistics =====
    num_sub_recordings: int = Field(...)
    duration_seconds: float = Field(...)
    num_data_points: int = Field(...)
    num_passes: int = 0

    # ===== QC Version =====
    biomech_qc_version: int = Field(default_factory=get_biomech_qc_version)

    @field_validator("audio_processing_id")
    @classmethod
    def validate_audio_processing_id(cls, value: Optional[int]) -> Optional[int]:
        """Validate audio_processing_id is positive if provided."""
        if value is not None and value <= 0:
            raise ValueError("audio_processing_id must be positive")
        return value

    # ===== QC Metadata =====
    biomechanics_qc_fail: bool = False
    biomechanics_qc_notes: Optional[str] = None

    # ===== Processing Metadata =====
    processing_date: datetime = Field(...)
    processing_status: Literal["not_processed", "success", "error"] = "not_processed"
    error_message: Optional[str] = None

    @field_validator("pass_number")
    @classmethod
    def validate_pass_number(cls, value: Optional[int], info) -> Optional[int]:
        """Validate pass number: required and positive for walk, must be None for non-walk."""
        maneuver = info.data.get("maneuver")
        if maneuver == "walk":
            if value is None:
                raise ValueError("pass_number is required for walk maneuvers")
            if value <= 0:
                raise ValueError("pass_number must be positive (>0)")
        else:
            if value is not None:
                raise ValueError("pass_number must be None for non-walk maneuvers")
        return value

    @field_validator("speed")
    @classmethod
    def validate_speed(cls, value: Optional[str], info) -> Optional[str]:
        """Validate speed: required for walk, must be None for non-walk."""
        maneuver = info.data.get("maneuver")
        if maneuver == "walk":
            if value is None:
                raise ValueError("speed is required for walk maneuvers")
        else:
            if value is not None:
                raise ValueError("speed must be None for non-walk maneuvers")
        return value

    @field_validator("biomechanics_sample_rate")
    @classmethod
    def validate_biomechanics_sample_rate(cls, value: float) -> float:
        """Validate biomechanics sample rate is positive."""
        if value <= 0:
            raise ValueError("biomechanics_sample_rate must be positive")
        return value

    @field_validator("num_sub_recordings", "num_passes", "num_data_points")
    @classmethod
    def validate_counts(cls, value: int) -> int:
        """Validate counts are non-negative."""
        if value < 0:
            raise ValueError("counts must be non-negative")
        return value

    @field_validator("duration_seconds")
    @classmethod
    def validate_duration(cls, value: float) -> float:
        """Validate duration is non-negative."""
        if value < 0:
            raise ValueError("duration_seconds must be non-negative")
        return value


@dataclass(kw_only=True)
class Synchronization(StudyMetadata):
    """Synchronization metadata.

    Represents audio-biomechanics synchronization results.
    Uses foreign keys to reference AudioProcessing and BiomechanicsImport.
    """

    # ===== Foreign Key References =====
    audio_processing_id: int = Field(...)  # ID of related AudioProcessing record
    biomechanics_import_id: int = Field(...)  # ID of related BiomechanicsImport record

    # ===== Walk-Specific Metadata =====
    pass_number: Optional[int] = None
    speed: Optional[Literal["slow", "fast", "medium", "comfortable"]] = None

    # ===== Stomp Time Data (Core Synchronization Results) =====
    # Note: Biomechanics is synchronized to audio data (audio time 0 = sync time 0)
    bio_left_sync_time: Optional[float] = None
    bio_right_sync_time: Optional[float] = None
    bio_sync_offset: Optional[float] = None  # Biomechanics sync offset between legs
    aligned_sync_time: Optional[float] = None  # Unified aligned sync time on merged dataframes

    # ===== Sync Method Details =====
    sync_method: Optional[Literal["consensus", "biomechanics"]] = None
    consensus_methods: Optional[str] = None
    consensus_time: Optional[float] = None
    rms_time: Optional[float] = None
    onset_time: Optional[float] = None
    freq_time: Optional[float] = None

    # ===== Biomechanics-Guided Detection (Optional) =====
    bio_selected_sync_time: Optional[float] = None  # Renamed from selected_audio_sync_time
    contra_bio_selected_sync_time: Optional[float] = None  # Renamed from contra_selected_audio_sync_time

    # ===== Audio-Visual Sync (Optional) =====
    audio_visual_sync_time: Optional[float] = None
    audio_visual_sync_time_contralateral: Optional[float] = None
    
    # ===== Audio Sync Times (Optional) =====
    # Time between turning microphones on and participant stopping for each leg
    audio_sync_time_left: Optional[float] = None
    audio_sync_time_right: Optional[float] = None
    audio_sync_offset: Optional[float] = None  # Required if both left and right are present
    
    # ===== Audio-Based Sync Fields (Optional) =====
    # Different from bio-based sync - required if 'audio' in stomp_detection_methods
    selected_audio_sync_time: Optional[float] = None
    contra_selected_audio_sync_time: Optional[float] = None

    # ===== Detection Method Details =====
    stomp_detection_methods: Optional[List[Literal["audio", "consensus", "biomechanics"]]] = None
    selected_stomp_method: Optional[Literal["audio", "consensus", "biomechanics"]] = None

    # ===== Pickle File Storage =====
    sync_file_name: str = Field(...)
    sync_file_path: Optional[str] = None

    # ===== Duration & Aggregate Statistics =====
    sync_duration: Optional[float] = None
    total_cycles_extracted: int = 0
    clean_cycles: int = 0
    outlier_cycles: int = 0

    # ===== Cycle Duration Statistics =====
    mean_cycle_duration_s: Optional[float] = None
    median_cycle_duration_s: Optional[float] = None
    min_cycle_duration_s: Optional[float] = None
    max_cycle_duration_s: Optional[float] = None
    method_agreement_span: Optional[float] = None

    # ===== QC Status (removed individual QC version fields - QC done at other stages) =====
    sync_qc_fail: bool = False

    # ===== Processing Metadata =====
    processing_date: datetime = Field(default_factory=datetime.now)
    processing_status: Literal["not_processed", "success", "error"] = "not_processed"
    error_message: Optional[str] = None

    @field_validator("audio_processing_id", "biomechanics_import_id")
    @classmethod
    def validate_ids(cls, value: int) -> int:
        """Validate IDs are positive."""
        if value <= 0:
            raise ValueError("IDs must be positive")
        return value

    @field_validator("total_cycles_extracted", "clean_cycles", "outlier_cycles")
    @classmethod
    def validate_cycle_counts(cls, value: int) -> int:
        """Validate cycle counts are non-negative."""
        if value < 0:
            raise ValueError("cycle counts must be non-negative")
        return value


@dataclass(kw_only=True)
class MovementCycle(StudyMetadata):
    """Individual movement cycle metadata.

    Represents a single extracted movement cycle.
    Uses foreign keys to reference AudioProcessing, BiomechanicsImport, and Synchronization.

    If synchronization_id is set, pass_number and speed must also be set and should
    match the associated Synchronization record (validated by repository layer).
    """

    # ===== Foreign Key References =====
    audio_processing_id: int = Field(...)  # ID of related AudioProcessing record
    biomechanics_import_id: Optional[int] = None  # ID of related BiomechanicsImport (optional)
    synchronization_id: Optional[int] = None  # ID of related Synchronization (optional)

    # ===== Walk-Specific Metadata (required if synchronization_id is set) =====
    pass_number: Optional[int] = None
    speed: Optional[Literal["slow", "fast", "medium", "comfortable"]] = None

    # ===== Cycle Identification =====
    cycle_file: str = Field(...)
    cycle_index: int = Field(...)
    is_outlier: bool = Field(...)

    # ===== Cycle Temporal Characteristics =====
    start_time_s: float = Field(...)
    end_time_s: float = Field(...)
    duration_s: float = Field(...)

    # ===== Audio Timestamps (always present) =====
    audio_start_time: datetime = Field(...)
    audio_end_time: datetime = Field(...)

    # ===== Biomechanics Timestamps (optional, required if biomechanics_import_id is set) =====
    bio_start_time: Optional[datetime] = None
    bio_end_time: Optional[datetime] = None

    # ===== Cycle-Level QC Flags =====
    biomechanics_qc_fail: bool = Field(...)
    sync_qc_fail: bool = Field(...)

    # ===== QC Version Tracking =====
    biomechanics_qc_version: int = Field(default_factory=get_biomech_qc_version)
    sync_qc_version: int = Field(default_factory=get_cycle_qc_version)

    @field_validator("audio_processing_id")
    @classmethod
    def validate_audio_processing_id(cls, value: int) -> int:
        """Validate audio_processing_id is positive."""
        if value <= 0:
            raise ValueError("audio_processing_id must be positive")
        return value

    @field_validator("biomechanics_import_id")
    @classmethod
    def validate_biomechanics_import_id(cls, value: Optional[int]) -> Optional[int]:
        """Validate biomechanics_import_id is positive if provided."""
        if value is not None and value <= 0:
            raise ValueError("biomechanics_import_id must be positive")
        return value

    @field_validator("synchronization_id")
    @classmethod
    def validate_synchronization_id(cls, value: Optional[int]) -> Optional[int]:
        """Validate synchronization_id is positive if provided."""
        if value is not None and value <= 0:
            raise ValueError("synchronization_id must be positive")
        return value

    @field_validator("cycle_index")
    @classmethod
    def validate_cycle_index(cls, value: int) -> int:
        """Validate cycle index is non-negative."""
        if value < 0:
            raise ValueError("cycle_index must be non-negative")
        return value

    @field_validator("duration_s", "start_time_s", "end_time_s")
    @classmethod
    def validate_times(cls, value: float) -> float:
        """Validate time values are non-negative."""
        if value < 0:
            raise ValueError("time values must be non-negative")
        return value

    @model_validator(mode="after")
    def validate_biomechanics_timestamps(self):
        """Validate bio_start_time and bio_end_time are provided when biomechanics_import_id is set."""
        if self.biomechanics_import_id is not None:
            if self.bio_start_time is None or self.bio_end_time is None:
                raise ValueError(
                    "bio_start_time and bio_end_time are required when biomechanics_import_id is set"
                )
        return self

    @model_validator(mode="after")
    def validate_walk_metadata_with_sync(self):
        """Validate walk metadata is present when synchronization_id is set.

        If this cycle is part of a Synchronization, it must have pass_number and speed set.
        The repository layer will validate these match the actual Synchronization record.
        """
        if self.synchronization_id is not None:
            if self.pass_number is None or self.speed is None:
                raise ValueError(
                    "pass_number and speed are required when synchronization_id is set"
                )
        return self
