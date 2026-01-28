"""Unified Pydantic dataclasses for metadata validation and Excel export.

This module provides Pydantic dataclasses that combine the validation
capabilities of Pydantic models with the convenience of dataclasses.
These classes are used for both input validation and Excel export
throughout the acoustic emissions processing pipeline.

The classes consolidate the previous separation between:
- Pydantic models in src.models (for validation)
- Python dataclasses in src.orchestration.processing_log (for Excel export)

Now using Pydantic's @dataclass decorator, we get both validation AND
convenient to_dict() methods for Excel export in a single definition.
"""

from datetime import datetime, timedelta
from typing import Any, ClassVar, Dict, List, Literal, Optional

from pydantic import Field, field_validator, model_validator
from pydantic.dataclasses import dataclass

from src.qc_versions import (
    get_audio_qc_version,
    get_biomech_qc_version,
    get_cycle_qc_version,
)


@dataclass(kw_only=True)
class StudyMetadata:
    """Base metadata class containing study information.

    All other metadata classes inherit from this to ensure consistent
    study and participant identification across all processing stages.
    """

    # Study identification
    study: Literal["AOA", "preOA", "SMoCK"]
    study_id: int  # Participant ID within the study (e.g., 1011 from #AOA1011)

    @field_validator("study_id")
    @classmethod
    def validate_study_id(cls, value: int) -> int:
        """Validate study ID is non-negative."""
        if value < 0:
            raise ValueError("study_id must be non-negative")
        return value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Excel export."""
        return {
            "Study": self.study,
            "Study ID": self.study_id,
        }


@dataclass(kw_only=True)
class BiomechanicsMetadata(StudyMetadata):
    """Metadata for biomechanics data linked to acoustic recording.

    Inherits from StudyMetadata. Fields are conditionally required based on
    whether biomechanics data is linked to the acoustic recording.
    """

    # Biomechanics linkage (with default to avoid inheritance issues)
    linked_biomechanics: bool = False

    # Conditionally required fields (required if linked_biomechanics is True)
    biomechanics_file: Optional[str] = None
    biomechanics_type: Optional[Literal["Gonio", "IMU", "Motion Analysis"]] = None
    biomechanics_sync_method: Optional[Literal["flick", "stomp"]] = None
    biomechanics_sample_rate: Optional[float] = None
    biomechanics_notes: Optional[str] = None

    @field_validator("biomechanics_file", "biomechanics_type", "biomechanics_sync_method")
    @classmethod
    def validate_biomechanics_fields(cls, value: Optional[Any], info) -> Optional[Any]:
        """Validate biomechanics fields are provided when linked_biomechanics is True."""
        field_name = info.field_name

        if info.data.get("linked_biomechanics") is True:
            if value is None:
                raise ValueError(f"{field_name} is required when linked_biomechanics is True")
        else:
            if value is not None:
                raise ValueError(f"{field_name} must be None when linked_biomechanics is False")
        return value

    @field_validator("biomechanics_notes")
    @classmethod
    def validate_biomechanics_notes(cls, value: Optional[str], info) -> Optional[str]:
        """Validate biomechanics_notes is optional regardless of linked status."""
        # Notes can be provided or not, regardless of linkage status
        return value

    @field_validator("biomechanics_sample_rate")
    @classmethod
    def validate_biomechanics_sample_rate(cls, value: Optional[float]) -> Optional[float]:
        """Validate biomechanics sample rate is positive if provided."""
        if value is not None and value <= 0:
            raise ValueError("biomechanics_sample_rate must be positive")
        return value

    @field_validator("biomechanics_sync_method")
    @classmethod
    def validate_biomechanics_sync_method_for_type(cls, value: Optional[str], info) -> Optional[str]:
        """Validate biomechanics_sync_method matches biomechanics_type requirements."""
        biomech_type = info.data.get("biomechanics_type")
        if biomech_type and value:
            if biomech_type in ["Motion Analysis", "IMU"] and value != "stomp":
                raise ValueError(f"biomechanics_sync_method must be 'stomp' for biomechanics_type '{biomech_type}'")
            elif biomech_type == "Gonio" and value != "flick":
                raise ValueError("biomechanics_sync_method must be 'flick' for biomechanics_type 'Gonio'")
        return value

    @field_validator("biomechanics_sample_rate")
    @classmethod
    def validate_biomechanics_sample_rate(cls, value: Optional[float]) -> Optional[float]:
        """Validate biomechanics sample rate is positive if provided."""
        if value is not None and value <= 0:
            raise ValueError("biomechanics_sample_rate must be positive")
        return value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Excel export."""
        result = super().to_dict()
        result.update({
            "Linked Biomechanics": self.linked_biomechanics,
            "Biomechanics File": self.biomechanics_file,
            "Biomechanics Type": self.biomechanics_type,
            "Biomechanics Sync Method": self.biomechanics_sync_method,
            "Biomechanics Sample Rate (Hz)": self.biomechanics_sample_rate,
            "Biomechanics Notes": self.biomechanics_notes,
        })
        return result


@dataclass(kw_only=True)
class WalkMetadata:
    """Mixin providing optional walk-specific metadata fields.

    These fields (pass_number and speed) are populated when the maneuver is "walk"
    and are conditionally required for walk maneuvers. This mixin is used to provide
    these fields to multiple classes (Synchronization, MovementCycle) without duplication.
    """

    # Walk-specific metadata (optional for non-walk maneuvers, required for walk)
    pass_number: Optional[int] = None
    speed: Optional[Literal["slow", "normal", "fast", "medium", "comfortable"]] = None

    @field_validator("pass_number")
    @classmethod
    def validate_pass_number(cls, value: Optional[int]) -> Optional[int]:
        """Validate pass number is non-negative if provided."""
        if value is not None and value < 0:
            raise ValueError("pass_number must be non-negative")
        return value


@dataclass(kw_only=True)
class AcousticsFile(BiomechanicsMetadata):
    """Audio file metadata.

    Inherits from BiomechanicsMetadata and StudyMetadata.
    Contains file-level information extracted from audio files.
    """

    # File identification
    audio_file_name: str = Field(...)
    device_serial: str = Field(...)  # Pulled from file name
    firmware_version: int = Field(...)  # Pulled from file name
    file_time: datetime = Field(...)  # Pulled from file name
    file_size_mb: float = Field(...)

    # Recording metadata
    recording_date: datetime = Field(...)  # Date from audio file recording
    recording_time: datetime = Field(...)  # Full datetime from audio file recording

    # Maneuver metadata
    knee: Literal["right", "left"] = Field(...)
    maneuver: Literal["fe", "sts", "walk"] = Field(...)

    # Audio characteristics (required fields first)
    num_channels: int = Field(...)  # Must be inferred from audio file metadata or data

    # Microphone positions
    mic_1_position: Literal["IPM", "IPL", "SPM", "SPL"] = Field(...)  # I=infra, S=supra, P=patellar, M=medial, L=lateral
    mic_2_position: Literal["IPM", "IPL", "SPM", "SPL"] = Field(...)
    mic_3_position: Literal["IPM", "IPL", "SPM", "SPL"] = Field(...)
    mic_4_position: Literal["IPM", "IPL", "SPM", "SPL"] = Field(...)

    # Audio characteristics (optional fields with defaults)
    sample_rate: float = 46875.0

    # Optional notes
    mic_1_notes: Optional[str] = None
    mic_2_notes: Optional[str] = None
    mic_3_notes: Optional[str] = None
    mic_4_notes: Optional[str] = None
    notes: Optional[str] = None

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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Excel export."""
        result = super().to_dict()
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
            "Sample Rate (Hz)": self.sample_rate,
            "Channels": self.num_channels,
            "Mic 1 Position": self.mic_1_position,
            "Mic 2 Position": self.mic_2_position,
            "Mic 3 Position": self.mic_3_position,
            "Mic 4 Position": self.mic_4_position,
            "Mic 1 Notes": self.mic_1_notes,
            "Mic 2 Notes": self.mic_2_notes,
            "Mic 3 Notes": self.mic_3_notes,
            "Mic 4 Notes": self.mic_4_notes,
            "Notes": self.notes,
        })
        return result


@dataclass(kw_only=True)
class SynchronizationMetadata(AcousticsFile):
    """Synchronization metadata for audio-biomechanics alignment.

    Inherits from AcousticsFile (audio file details).
    Contains sync event times, detection method details, and walk-specific metadata.
    
    Note: While pass_number and speed logically belong in WalkMetadata, they are
    defined here for practical dataclass field ordering reasons. These fields follow
    WalkMetadata semantics: optional for non-walk maneuvers, required for walk."""

    # ===== Walk-specific metadata =====
    # Optional for non-walk maneuvers, required for walk
    pass_number: Optional[int] = None
    speed: Optional[Literal["slow", "normal", "fast", "medium", "comfortable"]] = None

    # ===== Stomp Time Data (Core Synchronization Results) =====
    # Primary stomp times - optional (float in seconds, not timedelta)
    audio_sync_time: Optional[float] = None
    sync_offset: Optional[float] = None
    aligned_audio_sync_time: Optional[float] = None
    aligned_biomechanics_sync_time: Optional[float] = None

    # ===== Sync Method Details =====
    sync_method: Optional[Literal["consensus", "biomechanics"]] = None
    consensus_methods: Optional[str] = None  # Comma-separated if sync_method is "consensus"

    # ===== Detection Method Times =====
    # Individual detection algorithm results (in seconds, not timedelta)
    consensus_time: Optional[float] = None
    rms_time: Optional[float] = None
    onset_time: Optional[float] = None
    freq_time: Optional[float] = None

    # ===== Biomechanics-Guided Detection (Optional) =====
    # Used when sync is determined via biomechanics guidance
    selected_audio_sync_time: Optional[float] = None  # Selected audio stomp from biomechanics
    contra_selected_audio_sync_time: Optional[float] = None  # Contralateral selection (requires selected_audio_sync_time)

    # ===== Audio-Visual Sync (Optional) =====
    # Legacy fields for visual/audio sync
    audio_visual_sync_time: Optional[float] = None
    audio_visual_sync_time_contralateral: Optional[float] = None

    # ===== Biomechanics-specific sync times (Optional) =====
    # bio_left_sync_time and bio_right_sync_time represent the visual offsets between
    # the start of the recording and the sync event (e.g., stomp) for each side.
    # These are the ground truth from biomechanics visual data.
    bio_left_sync_time: Optional[float] = None  # Left side visual sync offset
    bio_right_sync_time: Optional[float] = None  # Right side visual sync offset

    # ===== Detection Method Details =====
    audio_stomp_method: Optional[str] = None  # Detection method used for audio sync (required if sync_type is "stomp")

    @field_validator("pass_number")
    @classmethod
    def validate_pass_number(cls, value: Optional[int]) -> Optional[int]:
        """Validate pass number is non-negative if provided."""
        if value is not None and value < 0:
            raise ValueError("pass_number must be non-negative")
        return value

    @field_validator("consensus_methods")
    @classmethod
    def validate_consensus_methods(cls, value: Optional[str], info) -> Optional[str]:
        """Validate consensus_methods is provided when sync_method is 'consensus'."""
        sync_method = info.data.get("sync_method")
        if sync_method == "consensus" and value is None:
            raise ValueError("consensus_methods is required when sync_method is 'consensus'")
        return value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Excel export."""
        result = super().to_dict()

        result.update({
            "Audio Sync Time (s)": self.audio_sync_time,
            "Sync Offset (s)": self.sync_offset,
            "Aligned Audio Sync Time (s)": self.aligned_audio_sync_time,
            "Aligned Biomechanics Sync Time (s)": self.aligned_biomechanics_sync_time,
            "Sync Method": self.sync_method,
            "Consensus Methods": self.consensus_methods,
            "Consensus Time (s)": self.consensus_time,
            "RMS Time (s)": self.rms_time,
            "Onset Time (s)": self.onset_time,
            "Freq Time (s)": self.freq_time,
            "Selected Audio Sync Time (s)": self.selected_audio_sync_time,
            "Contra Selected Audio Sync Time (s)": self.contra_selected_audio_sync_time,
            "Audio Visual Sync Time (s)": self.audio_visual_sync_time,
            "Audio Visual Sync Time Contralateral (s)": self.audio_visual_sync_time_contralateral,
            "Bio Left Sync Time (s)": self.bio_left_sync_time,
            "Bio Right Sync Time (s)": self.bio_right_sync_time,
            "Audio Stomp Method": self.audio_stomp_method,
        })
        return result


@dataclass(kw_only=True)
class AudioProcessing(AcousticsFile):
    """Audio file processing and QC metadata.

    Inherits from AcousticsFile (which inherits from BiomechanicsMetadata and StudyMetadata).
    Consolidates audio processing information for logging and validation.
    Fields use snake_case for direct mapping to database columns and Excel headers.
    """

    # REQUIRED FIELDS (no defaults) - must come first in Python 3.9
    # Processing metadata
    processing_date: datetime = Field(...)

    # Raw audio QC results - Overall fail segments (consolidated)
    qc_fail_segments: List[tuple[float, float]] = Field(...)  # Consolidated contiguous bad intervals
    qc_fail_segments_ch1: List[tuple[float, float]] = Field(...)
    qc_fail_segments_ch2: List[tuple[float, float]] = Field(...)
    qc_fail_segments_ch3: List[tuple[float, float]] = Field(...)
    qc_fail_segments_ch4: List[tuple[float, float]] = Field(...)

    # Signal dropout QC
    qc_signal_dropout: bool = Field(...)
    qc_signal_dropout_segments: List[tuple[float, float]] = Field(...)
    qc_signal_dropout_ch1: bool = Field(...)
    qc_signal_dropout_segments_ch1: List[tuple[float, float]] = Field(...)
    qc_signal_dropout_ch2: bool = Field(...)
    qc_signal_dropout_segments_ch2: List[tuple[float, float]] = Field(...)
    qc_signal_dropout_ch3: bool = Field(...)
    qc_signal_dropout_segments_ch3: List[tuple[float, float]] = Field(...)
    qc_signal_dropout_ch4: bool = Field(...)
    qc_signal_dropout_segments_ch4: List[tuple[float, float]] = Field(...)

    # Artifact QC
    qc_artifact: bool = Field(...)
    qc_artifact_segments: List[tuple[float, float]] = Field(...)
    qc_artifact_ch1: bool = Field(...)
    qc_artifact_segments_ch1: List[tuple[float, float]] = Field(...)
    qc_artifact_ch2: bool = Field(...)
    qc_artifact_segments_ch2: List[tuple[float, float]] = Field(...)
    qc_artifact_ch3: bool = Field(...)
    qc_artifact_segments_ch3: List[tuple[float, float]] = Field(...)
    qc_artifact_ch4: bool = Field(...)
    qc_artifact_segments_ch4: List[tuple[float, float]] = Field(...)

    # FIELDS WITH DEFAULTS - must come after required fields
    processing_status: Literal["not_processed", "success", "error"] = "not_processed"
    error_message: Optional[str] = None

    # Audio file characteristics
    duration_seconds: Optional[float] = None

    # QC version tracking
    audio_qc_version: int = Field(default_factory=get_audio_qc_version)

    # Artifact types (optional)
    qc_artifact_type: Optional[Literal["intermittent", "continuous"]] = None
    qc_artifact_type_ch1: Optional[Literal["intermittent", "continuous"]] = None
    qc_artifact_type_ch2: Optional[Literal["intermittent", "continuous"]] = None
    qc_artifact_type_ch3: Optional[Literal["intermittent", "continuous"]] = None
    qc_artifact_type_ch4: Optional[Literal["intermittent", "continuous"]] = None

    # Boolean QC status indicators (auto-populated from fail segments)
    qc_not_passed: bool = False  # True if any qc_fail_segments exist
    qc_not_passed_mic_1: bool = False
    qc_not_passed_mic_2: bool = False
    qc_not_passed_mic_3: bool = False
    qc_not_passed_mic_4: bool = False

    # Log tracking
    log_updated: Optional[datetime] = None

    @model_validator(mode="after")
    def populate_qc_not_passed_fields(self) -> "AudioProcessing":
        """Auto-populate qc_not_passed boolean fields from qc_fail_segments.

        These fields indicate whether QC failed (True) or passed (False)
        based on whether any fail segments exist.
        """
        # Set overall qc_not_passed based on whether any fail segments exist
        self.qc_not_passed = len(self.qc_fail_segments) > 0

        # Set per-mic qc_not_passed fields based on per-channel fail segments
        self.qc_not_passed_mic_1 = len(self.qc_fail_segments_ch1) > 0
        self.qc_not_passed_mic_2 = len(self.qc_fail_segments_ch2) > 0
        self.qc_not_passed_mic_3 = len(self.qc_fail_segments_ch3) > 0
        self.qc_not_passed_mic_4 = len(self.qc_fail_segments_ch4) > 0

        return self

    @field_validator("duration_seconds")
    @classmethod
    def validate_duration(cls, value: Optional[float]) -> Optional[float]:
        """Validate duration is non-negative if provided."""
        if value is not None and value < 0:
            raise ValueError("duration_seconds must be non-negative")
        return value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Excel export."""
        result = super().to_dict()
        result.update({
            "Processing Date": self.processing_date,
            "Status": self.processing_status,
            "Error": self.error_message,
            "Duration (s)": self.duration_seconds,
            "Audio QC Version": self.audio_qc_version,
            # QC fail segments
            "QC Fail Segments": str(self.qc_fail_segments),
            "QC Fail Segments Ch1": str(self.qc_fail_segments_ch1),
            "QC Fail Segments Ch2": str(self.qc_fail_segments_ch2),
            "QC Fail Segments Ch3": str(self.qc_fail_segments_ch3),
            "QC Fail Segments Ch4": str(self.qc_fail_segments_ch4),
            # Signal dropout
            "QC Signal Dropout": self.qc_signal_dropout,
            "QC Signal Dropout Segments": str(self.qc_signal_dropout_segments),
            "QC Signal Dropout Ch1": self.qc_signal_dropout_ch1,
            "QC Signal Dropout Segments Ch1": str(self.qc_signal_dropout_segments_ch1),
            "QC Signal Dropout Ch2": self.qc_signal_dropout_ch2,
            "QC Signal Dropout Segments Ch2": str(self.qc_signal_dropout_segments_ch2),
            "QC Signal Dropout Ch3": self.qc_signal_dropout_ch3,
            "QC Signal Dropout Segments Ch3": str(self.qc_signal_dropout_segments_ch3),
            "QC Signal Dropout Ch4": self.qc_signal_dropout_ch4,
            "QC Signal Dropout Segments Ch4": str(self.qc_signal_dropout_segments_ch4),
            # Artifacts
            "QC Artifact": self.qc_artifact,
            "QC Artifact Type": self.qc_artifact_type,
            "QC Artifact Segments": str(self.qc_artifact_segments),
            "QC Artifact Ch1": self.qc_artifact_ch1,
            "QC Artifact Type Ch1": self.qc_artifact_type_ch1,
            "QC Artifact Segments Ch1": str(self.qc_artifact_segments_ch1),
            "QC Artifact Ch2": self.qc_artifact_ch2,
            "QC Artifact Type Ch2": self.qc_artifact_type_ch2,
            "QC Artifact Segments Ch2": str(self.qc_artifact_segments_ch2),
            "QC Artifact Ch3": self.qc_artifact_ch3,
            "QC Artifact Type Ch3": self.qc_artifact_type_ch3,
            "QC Artifact Segments Ch3": str(self.qc_artifact_segments_ch3),
            "QC Artifact Ch4": self.qc_artifact_ch4,
            "QC Artifact Type Ch4": self.qc_artifact_type_ch4,
            "QC Artifact Segments Ch4": str(self.qc_artifact_segments_ch4),
            # Boolean QC status indicators
            "QC_not_passed": self.qc_not_passed,
            "QC_not_passed_mic_1": self.qc_not_passed_mic_1,
            "QC_not_passed_mic_2": self.qc_not_passed_mic_2,
            "QC_not_passed_mic_3": self.qc_not_passed_mic_3,
            "QC_not_passed_mic_4": self.qc_not_passed_mic_4,
            # Log tracking
            "Log Updated": self.log_updated.isoformat() if self.log_updated else None,
        })
        return result



@dataclass(kw_only=True)
class BiomechanicsImport(StudyMetadata):
    """Biomechanics data import metadata.

    Inherits from StudyMetadata.
    Tracks biomechanics file import status and characteristics.
    Fields use snake_case for direct mapping to database columns and Excel headers.
    """

    # REQUIRED FIELDS (no defaults) - must come first in Python 3.9
    # File identification
    biomechanics_file: str = Field(...)
    sheet_name: str = Field(...)

    # Processing metadata
    processing_date: datetime = Field(...)
    processing_status: Literal["not_processed", "success", "error"] = Field(...)

    # Import statistics
    num_sub_recordings: int = Field(...)

    # Data characteristics
    duration_seconds: float = Field(...)  # Total duration of entire dataset (all passes)
    sample_rate: float = Field(...)
    num_data_points: int = Field(...)  # Total data points across entire dataset (all passes)

    # FIELDS WITH DEFAULTS - must come after required fields
    # Maneuver context (optional to allow legacy records without this field)
    maneuver: Optional[Literal["fe", "sts", "walk"]] = None

    error_message: Optional[str] = None

    # Number of passes in the biomechanics data (only relevant for walking)
    num_passes: int = 0

    # Biomechanics type (for tracking which modality was used)
    biomechanics_type: Optional[Literal["Gonio", "IMU", "Motion Analysis"]] = None

    @field_validator("num_sub_recordings")
    @classmethod
    def validate_num_sub_recordings(cls, value: int, info) -> int:
        """Validate num_sub_recordings based on maneuver type.

        For sit-to-stand and flexion-extension maneuvers, num_sub_recordings must equal 1.
        For walking maneuvers, num_sub_recordings must be >= 1.

        The number of sub-recordings represents usable biomechanics data segments that
        meet quality criteria (e.g., sufficient heel strikes for walking).
        """
        if value < 0:
            raise ValueError("num_sub_recordings must be non-negative")

        # Get maneuver from parent class data if available
        maneuver = info.data.get("maneuver")
        if maneuver in ["sts", "fe"]:
            if value != 1:
                raise ValueError(f"num_sub_recordings must equal 1 for {maneuver} maneuvers, got {value}")
        elif maneuver == "walk":
            if value < 1:
                raise ValueError(f"num_sub_recordings must be >= 1 for walking maneuvers, got {value}")

        return value

    @field_validator("num_passes", "num_data_points")
    @classmethod
    def validate_counts(cls, value: int) -> int:
        """Validate counts are non-negative."""
        if value < 0:
            raise ValueError("counts must be non-negative")
        return value

    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, value: float) -> float:
        """Validate sample rate is positive."""
        if value <= 0:
            raise ValueError("sample_rate must be positive")
        return value

    @field_validator("duration_seconds")
    @classmethod
    def validate_duration(cls, value: float) -> float:
        """Validate duration is non-negative."""
        if value < 0:
            raise ValueError("duration_seconds must be non-negative")
        return value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Excel export."""
        result = super().to_dict()
        result.update({
            "Biomechanics File": self.biomechanics_file,
            "Sheet Name": self.sheet_name,
            "Processing Date": self.processing_date,
            "Status": self.processing_status,
            "Error": self.error_message,
            "Maneuver": self.maneuver,
            "Num Sub-Recordings": self.num_sub_recordings,
            "Num Passes": self.num_passes,
            "Duration (s)": self.duration_seconds,
            "Num Data Points": self.num_data_points,
            "Sample Rate (Hz)": self.sample_rate,
        })
        return result


@dataclass(kw_only=True)
class Synchronization(SynchronizationMetadata):
    """Synchronization process record.

    Inherits from SynchronizationMetadata (which inherits from AcousticsFile and WalkMetadata).
    Contains synchronization process metadata, aggregate statistics across all cycles,
    but NOT per-cycle details (those are stored in MovementCycle records).
    
    Note: This class requires linked_biomechanics to be True (biomechanics data must be present).
    """

    # ===== Sync Process Identification =====
    sync_file_name: str = Field(...)
    processing_date: datetime = Field(...)

    # ===== Processing Details =====
    sync_duration: Optional[float] = None  # Duration in seconds
    processing_status: Literal["not_processed", "success", "error"] = "not_processed"
    error_message: Optional[str] = None

    # ===== Aggregate Statistics (from all cycles) =====
    total_cycles_extracted: int = 0
    clean_cycles: int = 0
    outlier_cycles: int = 0

    # ===== Cycle Duration Statistics =====
    mean_cycle_duration_s: Optional[float] = None
    median_cycle_duration_s: Optional[float] = None
    min_cycle_duration_s: Optional[float] = None
    max_cycle_duration_s: Optional[float] = None

    # ===== Detection Method Details =====
    method_agreement_span: Optional[float] = None  # Time span between methods in consensus

    # ===== Constraints =====
    # Override linked_biomechanics from parent - always True for synchronization
    linked_biomechanics: Literal[True] = True

    # ===== QC Version Tracking =====
    audio_qc_version: int = Field(default_factory=get_audio_qc_version)
    biomech_qc_version: int = Field(default_factory=get_biomech_qc_version)
    cycle_qc_version: int = Field(default_factory=get_cycle_qc_version)

    @field_validator("total_cycles_extracted", "clean_cycles", "outlier_cycles")
    @classmethod
    def validate_cycle_counts(cls, value: int) -> int:
        """Validate cycle counts are non-negative."""
        if value < 0:
            raise ValueError("cycle counts must be non-negative")
        return value

    @property
    def acoustic_threshold(self) -> Optional[float]:
        """Alias for qc_acoustic_threshold for convenience."""
        return self.qc_acoustic_threshold

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Excel export."""
        result = super().to_dict()
        result.update({
            "Sync File": self.sync_file_name,
            "Processing Date": self.processing_date,
            "Processing Status": self.processing_status,
            "Error Message": self.error_message,
            "Sync Duration (s)": self.sync_duration,
            "Total Cycles Extracted": self.total_cycles_extracted,
            "Clean Cycles": self.clean_cycles,
            "Outlier Cycles": self.outlier_cycles,
            "Mean Cycle Duration (s)": self.mean_cycle_duration_s,
            "Median Cycle Duration (s)": self.median_cycle_duration_s,
            "Min Cycle Duration (s)": self.min_cycle_duration_s,
            "Max Cycle Duration (s)": self.max_cycle_duration_s,
            "Method Agreement Span (s)": self.method_agreement_span,
            "Audio QC Version": self.audio_qc_version,
            "Biomech QC Version": self.biomech_qc_version,
            "Cycle QC Version": self.cycle_qc_version,
        })
        return result


@dataclass(kw_only=True)
class MovementCycle(SynchronizationMetadata, AudioProcessing):
    """Single movement cycle metadata.

    Inherits from:
    - SynchronizationMetadata: sync times and sync method data
    - AudioProcessing: all audio QC fields (qc_artifact, qc_signal_dropout, etc.)
    
    Represents a single extracted movement cycle with all upstream processing context via inheritance.
    Fields use snake_case for direct mapping to database columns and Excel headers.
    
    Note: This class automatically inherits:
    - All sync times and methods from SynchronizationMetadata
    - All audio QC fields from AudioProcessing
    - Walk metadata (pass_number, speed) from WalkMetadata (via SynchronizationMetadata)
    """

    # ===== Cycle Identification =====
    cycle_file: str = Field(...)
    cycle_index: int = Field(...)
    is_outlier: bool = Field(...)  # REQUIRED - set by processing, not auto-calculated

    # ===== Cycle Temporal Characteristics =====
    start_time_s: float = Field(...)
    end_time_s: float = Field(...)
    duration_s: float = Field(...)
    audio_start_time: datetime = Field(...)  # REQUIRED - all files have audio
    audio_end_time: datetime = Field(...)  # REQUIRED - all files have audio

    # ===== Biomechanics Timestamps (conditional on linked_biomechanics) =====
    bio_start_time: Optional[datetime] = None  # REQUIRED if linked_biomechanics is True
    bio_end_time: Optional[datetime] = None  # REQUIRED if linked_biomechanics is True

    # ===== Cycle-Level QC Flags (CYCLE-LEVEL ONLY - NOT inherited) =====
    # These should NOT have defaults - must be explicitly set by processing
    biomechanics_qc_fail: bool = Field(...)  # REQUIRED - populated by processing
    sync_qc_fail: bool = Field(...)  # REQUIRED - populated by processing

    # ===== QC Version Tracking =====
    biomechanics_qc_version: int = Field(default_factory=get_biomech_qc_version)
    sync_qc_version: int = Field(default_factory=get_cycle_qc_version)

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
        """Validate bio_start_time and bio_end_time are provided when linked_biomechanics is True.
        
        If biomechanics data is linked to this cycle, both timestamps must be provided.
        """
        if self.linked_biomechanics is True:
            if self.bio_start_time is None or self.bio_end_time is None:
                raise ValueError(
                    "bio_start_time and bio_end_time are required when linked_biomechanics is True"
                )
        return self

    @field_validator("pass_number", "speed")
    @classmethod
    def validate_walk_fields(cls, value: Optional[Any], info) -> Optional[Any]:
        """Validate pass_number and speed for walk maneuver."""
        maneuver = info.data.get("maneuver")
        if maneuver == "walk" and value is None:
            field_name = info.field_name
            raise ValueError(f"{field_name} is required when maneuver is 'walk'")
        return value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Excel export (Cycle Details sheet)."""
        result = super().to_dict()
        result.update({
            "Cycle File": self.cycle_file,
            "Cycle Index": self.cycle_index,
            "Is Outlier": self.is_outlier,
            "Start (s)": self.start_time_s,
            "End (s)": self.end_time_s,
            "Duration (s)": self.duration_s,
            "Audio Start Time": self.audio_start_time,
            "Audio End Time": self.audio_end_time,
            "Bio Start Time": self.bio_start_time,
            "Bio End Time": self.bio_end_time,
            "Biomechanics QC Fail": self.biomechanics_qc_fail,
            "Sync QC Fail": self.sync_qc_fail,
            "Biomechanics QC Version": self.biomechanics_qc_version,
            "Sync QC Version": self.sync_qc_version,
        })
        return result
