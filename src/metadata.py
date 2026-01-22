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

from pydantic import Field, field_validator
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
    
    # Study identification (optional with defaults for incremental record creation)
    study: Optional[Literal["AOA", "preOA", "SMoCK"]] = None
    study_id: Optional[int] = None  # Participant ID within the study (e.g., 1011 from #AOA1011)
    
    @field_validator("study_id")
    @classmethod
    def validate_study_id(cls, value: Optional[int]) -> Optional[int]:
        """Validate study ID is positive if provided."""
        if value is not None and value <= 0:
            raise ValueError("study_id must be positive")
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
    sync_method: Optional[Literal["flick", "stomp"]] = None
    biomechanics_sample_rate: Optional[float] = None
    biomechanics_notes: Optional[str] = None
    
    @field_validator("biomechanics_file", "biomechanics_type", "sync_method", "biomechanics_sample_rate")
    @classmethod
    def validate_biomechanics_fields(cls, value: Optional[Any], info) -> Optional[Any]:
        """Validate biomechanics fields are provided when linked_biomechanics is True."""
        if info.data.get("linked_biomechanics") is True:
            if value is None and info.field_name != "biomechanics_notes":
                raise ValueError(f"{info.field_name} is required when linked_biomechanics is True")
        else:
            if value is not None:
                raise ValueError(f"{info.field_name} must be None when linked_biomechanics is False")
        return value
    
    @field_validator("sync_method")
    @classmethod
    def validate_sync_method_for_type(cls, value: Optional[str], info) -> Optional[str]:
        """Validate sync_method matches biomechanics_type requirements."""
        biomech_type = info.data.get("biomechanics_type")
        if biomech_type and value:
            if biomech_type in ["Motion Analysis", "IMU"] and value != "stomp":
                raise ValueError(f"sync_method must be 'stomp' for biomechanics_type '{biomech_type}'")
            elif biomech_type == "Gonio" and value != "flick":
                raise ValueError("sync_method must be 'flick' for biomechanics_type 'Gonio'")
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
            "Sync Method": self.sync_method,
            "Biomechanics Sample Rate (Hz)": self.biomechanics_sample_rate,
            "Biomechanics Notes": self.biomechanics_notes,
        })
        return result


@dataclass(kw_only=True)
class AcousticsFile(BiomechanicsMetadata):
    """Audio file metadata.
    
    Inherits from BiomechanicsMetadata and StudyMetadata.
    Contains file-level information extracted from audio files.
    Fields are optional to allow incremental record creation.
    """
    
    # File identification (optional for incremental creation)
    audio_file_name: Optional[str] = None
    device_serial: Optional[str] = None  # Pulled from file name
    firmware_version: Optional[int] = None  # Pulled from file name
    file_time: Optional[datetime] = None  # Pulled from file name
    file_size_mb: Optional[float] = None
    
    # Recording metadata
    recording_date: Optional[datetime] = None  # Date from audio file recording
    recording_time: Optional[datetime] = None  # Full datetime from audio file recording
    
    # Maneuver metadata
    knee: Optional[Literal["right", "left"]] = None
    maneuver: Optional[Literal["fe", "sts", "walk"]] = None
    
    # Audio characteristics
    sample_rate: float = 46875.0
    num_channels: int = 4
    
    # Microphone positions
    mic_1_position: Optional[Literal["IPM", "IPL", "SPM", "SPL"]] = None  # I=infra, S=supra, P=patellar, M=medial, L=lateral
    mic_2_position: Optional[Literal["IPM", "IPL", "SPM", "SPL"]] = None
    mic_3_position: Optional[Literal["IPM", "IPL", "SPM", "SPL"]] = None
    mic_4_position: Optional[Literal["IPM", "IPL", "SPM", "SPL"]] = None
    
    # Optional notes
    mic_1_notes: Optional[str] = None
    mic_2_notes: Optional[str] = None
    mic_3_notes: Optional[str] = None
    mic_4_notes: Optional[str] = None
    notes: Optional[str] = None
    
    @field_validator("recording_time")
    @classmethod
    def validate_recording_time_matches_date(cls, value: Optional[datetime], info) -> Optional[datetime]:
        """Validate recording_time date component matches recording_date if both provided."""
        if value is None:
            return value
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
    def validate_file_size(cls, value: Optional[float]) -> Optional[float]:
        """Validate file size is non-negative if provided."""
        if value is not None and value < 0:
            raise ValueError("file_size_mb must be non-negative")
        return value
    
    @field_validator("firmware_version")
    @classmethod
    def validate_firmware_version(cls, value: Optional[int]) -> Optional[int]:
        """Validate firmware version is non-negative if provided."""
        if value is not None and value < 0:
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
    
    Inherits from AcousticsFile. Contains sync event times and
    detection method details. Fields are optional to allow incremental creation.
    """
    
    # Sync times (in original time coordinates) - optional for incremental creation
    audio_sync_time: Optional[timedelta] = None
    bio_left_sync_time: Optional[timedelta] = None
    bio_right_sync_time: Optional[timedelta] = None
    
    # Visual sync times (from biomechanics QC)
    audio_visual_sync_time: Optional[timedelta] = None
    audio_visual_sync_time_contralateral: Optional[timedelta] = None
    
    # Alignment details - optional for incremental creation
    sync_offset: Optional[timedelta] = None
    aligned_audio_sync_time: Optional[timedelta] = None
    aligned_bio_sync_time: Optional[timedelta] = None
    
    # Detection method - optional for incremental creation
    sync_method: Optional[Literal["consensus", "biomechanics"]] = None
    consensus_methods: Optional[str] = None  # Comma-separated if sync_method is "consensus"
    
    # Detection times - optional for incremental creation
    consensus_time: Optional[timedelta] = None
    rms_time: Optional[timedelta] = None
    onset_time: Optional[timedelta] = None
    freq_time: Optional[timedelta] = None
    biomechanics_time: Optional[timedelta] = None
    biomechanics_time_contralateral: Optional[timedelta] = None
    
    @field_validator("bio_left_sync_time", "bio_right_sync_time")
    @classmethod
    def validate_bio_sync_time_for_knee(cls, value: Optional[timedelta], info) -> Optional[timedelta]:
        """Validate bio sync time is provided for the recorded knee if knee is specified."""
        knee = info.data.get("knee")
        if knee is None:
            return value
            
        field_name = info.field_name
        
        if knee == "left" and field_name == "bio_left_sync_time":
            if value is None and info.data.get("audio_sync_time") is not None:
                # Only require if we're creating a full sync record
                pass  # Allow None for partial records
        elif knee == "right" and field_name == "bio_right_sync_time":
            if value is None and info.data.get("audio_sync_time") is not None:
                pass  # Allow None for partial records
        
        return value
    
    @field_validator("consensus_methods")
    @classmethod
    def validate_consensus_methods(cls, value: Optional[str], info) -> Optional[str]:
        """Validate consensus_methods is provided when sync_method is 'consensus'."""
        sync_method = info.data.get("sync_method")
        if sync_method == "consensus" and value is None:
            # Allow None for partial records
            pass
        return value
    
    @field_validator("biomechanics_time")
    @classmethod
    def validate_biomechanics_time(cls, value: Optional[timedelta], info) -> Optional[timedelta]:
        """Validate biomechanics_time is provided when sync_method is 'biomechanics'."""
        sync_method = info.data.get("sync_method")
        if sync_method == "biomechanics" and value is None:
            # Allow None for partial records
            pass
        return value
    
    @field_validator("biomechanics_time_contralateral")
    @classmethod
    def validate_biomechanics_time_contralateral(cls, value: Optional[timedelta], info) -> Optional[timedelta]:
        """Validate biomechanics_time_contralateral when contralateral knee has bio_sync_time."""
        knee = info.data.get("knee")
        if knee is None:
            return value
            
        if knee == "left":
            bio_sync = info.data.get("bio_right_sync_time")
        else:
            bio_sync = info.data.get("bio_left_sync_time")
        
        if bio_sync is not None and value is None:
            # Allow None for partial records
            pass
        
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Excel export."""
        result = super().to_dict()
        result.update({
            "Audio Sync Time": self.audio_sync_time,
            "Bio Left Sync Time": self.bio_left_sync_time,
            "Bio Right Sync Time": self.bio_right_sync_time,
            "Audio Visual Sync Time": self.audio_visual_sync_time,
            "Audio Visual Sync Time Contralateral": self.audio_visual_sync_time_contralateral,
            "Sync Offset": self.sync_offset,
            "Aligned Audio Sync Time": self.aligned_audio_sync_time,
            "Aligned Bio Sync Time": self.aligned_bio_sync_time,
            "Sync Method": self.sync_method,
            "Consensus Methods": self.consensus_methods,
            "Consensus Time": self.consensus_time,
            "RMS Time": self.rms_time,
            "Onset Time": self.onset_time,
            "Freq Time": self.freq_time,
            "Biomechanics Time": self.biomechanics_time,
            "Biomechanics Time Contralateral": self.biomechanics_time_contralateral,
        })
        return result


@dataclass(kw_only=True)
class AudioProcessing(AcousticsFile):
    """Audio file processing and QC metadata.

    Inherits from AcousticsFile (which inherits from BiomechanicsMetadata and StudyMetadata).
    Consolidates audio processing information for logging and validation.
    Fields use snake_case for direct mapping to database columns and Excel headers.
    """

    # Processing metadata - optional for incremental creation
    processing_date: Optional[datetime] = None
    processing_status: Literal["not_processed", "success", "error"] = "not_processed"
    error_message: Optional[str] = None

    # Audio file characteristics
    duration_seconds: Optional[float] = None

    # QC version tracking
    audio_qc_version: int = Field(default_factory=get_audio_qc_version)

    # Raw audio QC results - Overall fail segments (consolidated) - optional with defaults
    qc_fail_segments: List[tuple[float, float]] = Field(default_factory=list)
    qc_fail_segments_ch1: List[tuple[float, float]] = Field(default_factory=list)
    qc_fail_segments_ch2: List[tuple[float, float]] = Field(default_factory=list)
    qc_fail_segments_ch3: List[tuple[float, float]] = Field(default_factory=list)
    qc_fail_segments_ch4: List[tuple[float, float]] = Field(default_factory=list)

    # Signal dropout QC - optional with defaults
    qc_signal_dropout: bool = False
    qc_signal_dropout_segments: List[tuple[float, float]] = Field(default_factory=list)
    qc_signal_dropout_ch1: bool = False
    qc_signal_dropout_segments_ch1: List[tuple[float, float]] = Field(default_factory=list)
    qc_signal_dropout_ch2: bool = False
    qc_signal_dropout_segments_ch2: List[tuple[float, float]] = Field(default_factory=list)
    qc_signal_dropout_ch3: bool = False
    qc_signal_dropout_segments_ch3: List[tuple[float, float]] = Field(default_factory=list)
    qc_signal_dropout_ch4: bool = False
    qc_signal_dropout_segments_ch4: List[tuple[float, float]] = Field(default_factory=list)

    # Artifact QC - optional with defaults
    qc_artifact: bool = False
    qc_artifact_type: Optional[Literal["intermittent", "continuous"]] = None
    qc_artifact_segments: List[tuple[float, float]] = Field(default_factory=list)
    qc_artifact_ch1: bool = False
    qc_artifact_type_ch1: Optional[Literal["intermittent", "continuous"]] = None
    qc_artifact_segments_ch1: List[tuple[float, float]] = Field(default_factory=list)
    qc_artifact_ch2: bool = False
    qc_artifact_type_ch2: Optional[Literal["intermittent", "continuous"]] = None
    qc_artifact_segments_ch2: List[tuple[float, float]] = Field(default_factory=list)
    qc_artifact_ch3: bool = False
    qc_artifact_type_ch3: Optional[Literal["intermittent", "continuous"]] = None
    qc_artifact_segments_ch3: List[tuple[float, float]] = Field(default_factory=list)
    qc_artifact_ch4: bool = False
    qc_artifact_type_ch4: Optional[Literal["intermittent", "continuous"]] = None
    qc_artifact_segments_ch4: List[tuple[float, float]] = Field(default_factory=list)

    # Legacy QC fields (for backward compatibility, populated from new fields)
    qc_not_passed: Optional[str] = None  # String repr of qc_fail_segments
    qc_not_passed_mic_1: Optional[str] = None
    qc_not_passed_mic_2: Optional[str] = None
    qc_not_passed_mic_3: Optional[str] = None
    qc_not_passed_mic_4: Optional[str] = None

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
            # Legacy fields
            "QC_not_passed": self.qc_not_passed,
            "QC_not_passed_mic_1": self.qc_not_passed_mic_1,
            "QC_not_passed_mic_2": self.qc_not_passed_mic_2,
            "QC_not_passed_mic_3": self.qc_not_passed_mic_3,
            "QC_not_passed_mic_4": self.qc_not_passed_mic_4,
        })
        return result


@dataclass(kw_only=True)
class BiomechanicsImport(StudyMetadata):
    """Biomechanics data import metadata.

    Inherits from StudyMetadata.
    Tracks biomechanics file import status and characteristics.
    Fields use snake_case for direct mapping to database columns and Excel headers.
    """

    # File identification (required)
    biomechanics_file: str
    sheet_name: str

    # Processing metadata (required)
    processing_date: datetime
    processing_status: Literal["not_processed", "success", "error"]
    error_message: Optional[str] = None

    # Import statistics
    num_recordings: int = 0
    num_passes: int = 0  # For walking maneuvers

    # Data characteristics (required)
    duration_seconds: float  # Total duration of entire dataset (all passes)
    sample_rate: float
    num_data_points: int  # Total data points across entire dataset (all passes)

    @field_validator("num_recordings", "num_passes", "num_data_points")
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
            "Num Recordings": self.num_recordings,
            "Num Passes": self.num_passes,
            "Duration (s)": self.duration_seconds,
            "Num Data Points": self.num_data_points,
            "Sample Rate (Hz)": self.sample_rate,
        })
        return result


@dataclass(kw_only=True)
class Synchronization(SynchronizationMetadata):
    """Audio-biomechanics synchronization and movement cycle extraction metadata.

    Inherits from SynchronizationMetadata (which inherits from AcousticsFile, BiomechanicsMetadata, StudyMetadata).
    Merges synchronization tracking with movement cycle extraction (previously MovementCycles).
    Tracks synchronization process, alignment details, and cycle extraction results.
    Fields use snake_case for direct mapping to database columns and Excel headers.
    """

    # File identification
    sync_file_name: str
    pass_number: Optional[int] = None  # For walking
    speed: Optional[Literal["slow", "normal", "fast", "medium"]] = None

    # Processing metadata - optional for incremental creation
    processing_date: Optional[datetime] = None
    processing_status: Literal["not_processed", "success", "error"] = "not_processed"
    error_message: Optional[str] = None

    # Synchronized data characteristics - optional for incremental creation
    sync_duration: Optional[timedelta] = None  # Total time of overlapped recording/biomechanics

    # Movement cycle extraction results - optional since not always extracted
    total_cycles_extracted: int = 0
    clean_cycles: int = 0
    outlier_cycles: int = 0

    # QC parameters
    qc_acoustic_threshold: Optional[float] = None

    # Per-cycle details list (for Cycle Details sheet in Excel)
    per_cycle_details: List['MovementCycle'] = Field(default_factory=list)

    # Aggregate statistics (across clean cycles) - optional since not always available
    mean_cycle_duration_s: Optional[float] = None
    median_cycle_duration_s: Optional[float] = None
    min_cycle_duration_s: Optional[float] = None
    max_cycle_duration_s: Optional[float] = None
    mean_acoustic_auc: Optional[float] = None

    # QC version tracking
    audio_qc_version: int = Field(default_factory=get_audio_qc_version)
    biomech_qc_version: int = Field(default_factory=get_biomech_qc_version)
    cycle_qc_version: int = Field(default_factory=get_cycle_qc_version)

    @field_validator("pass_number")
    @classmethod
    def validate_pass_number(cls, value: Optional[int]) -> Optional[int]:
        """Validate pass number is non-negative if provided."""
        if value is not None and value < 0:
            raise ValueError("pass_number must be non-negative")
        return value

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

    # Backward compatibility properties for old field names
    @property
    def stomp_offset(self) -> Optional[float]:
        """Backward compatibility: return sync_offset in seconds."""
        return self.sync_offset.total_seconds() if self.sync_offset else None
    
    @property
    def audio_stomp_time(self) -> Optional[float]:
        """Backward compatibility: return audio_sync_time in seconds."""
        return self.audio_sync_time.total_seconds() if self.audio_sync_time else None
    
    @property
    def bio_left_stomp_time(self) -> Optional[float]:
        """Backward compatibility: return bio_left_sync_time in seconds."""
        return self.bio_left_sync_time.total_seconds() if self.bio_left_sync_time else None
    
    @property
    def bio_right_stomp_time(self) -> Optional[float]:
        """Backward compatibility: return bio_right_sync_time in seconds."""
        return self.bio_right_sync_time.total_seconds() if self.bio_right_sync_time else None
    
    @property
    def aligned_audio_stomp_time(self) -> Optional[float]:
        """Backward compatibility: return aligned_audio_sync_time in seconds."""
        return self.aligned_audio_sync_time.total_seconds() if self.aligned_audio_sync_time else None
    
    @property
    def aligned_bio_stomp_time(self) -> Optional[float]:
        """Backward compatibility: return aligned_bio_sync_time in seconds."""
        return self.aligned_bio_sync_time.total_seconds() if self.aligned_bio_sync_time else None
    
    @property
    def knee_side(self) -> Optional[str]:
        """Backward compatibility: return knee field."""
        return self.knee

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Excel export."""
        result = super().to_dict()
        result.update({
            "Sync File": self.sync_file_name,
            "Pass Number": self.pass_number,
            "Speed": self.speed,
            "Processing Date": self.processing_date,
            "Status": self.processing_status,
            "Error": self.error_message,
            "Sync Duration": self.sync_duration,
            "Total Cycles": self.total_cycles_extracted,
            "Clean Cycles": self.clean_cycles,
            "Outlier Cycles": self.outlier_cycles,
            "Acoustic Threshold": self.qc_acoustic_threshold,
            "Mean Duration (s)": self.mean_cycle_duration_s,
            "Median Duration (s)": self.median_cycle_duration_s,
            "Min Duration (s)": self.min_cycle_duration_s,
            "Max Duration (s)": self.max_cycle_duration_s,
            "Mean Acoustic AUC": self.mean_acoustic_auc,
            "Audio QC Version": self.audio_qc_version,
            "Biomech QC Version": self.biomech_qc_version,
            "Cycle QC Version": self.cycle_qc_version,
            # Backward compatibility: add old field names with seconds
            "Stomp Offset (s)": self.stomp_offset,
            "Audio Stomp (s)": self.audio_stomp_time,
            "Bio Left Stomp (s)": self.bio_left_stomp_time,
            "Bio Right Stomp (s)": self.bio_right_stomp_time,
            "Aligned Audio Stomp (s)": self.aligned_audio_stomp_time,
            "Aligned Bio Stomp (s)": self.aligned_bio_stomp_time,
            "Knee Side": self.knee_side,
        })
        return result


@dataclass(kw_only=True)
class MovementCycle(AudioProcessing):
    """Single movement cycle metadata.

    Inherits from AudioProcessing (which inherits from AcousticsFile, BiomechanicsMetadata, StudyMetadata).
    Represents a single extracted movement cycle with all upstream processing context via inheritance.
    Fields use snake_case for direct mapping to database columns and Excel headers.
    """

    # Cycle identification - optional for incremental creation
    cycle_file: Optional[str] = None  # Path to .pkl file
    cycle_index: int = 0
    is_outlier: bool = False  # True if failed any QC (audio, biomech, or sync)

    # Cycle temporal characteristics - optional for incremental creation
    start_time_s: Optional[float] = None  # Start time within synced recording
    end_time_s: Optional[float] = None    # End time within synced recording
    duration_s: Optional[float] = None

    # Audio timestamps - optional for incremental creation
    audio_start_time: Optional[datetime] = None
    audio_end_time: Optional[datetime] = None

    # Biomechanics timestamps - optional for incremental creation
    bio_start_time: Optional[datetime] = None
    bio_end_time: Optional[datetime] = None

    # Biomechanics context (conditionally required for walk maneuver)
    pass_number: Optional[int] = None
    speed: Optional[Literal["slow", "normal", "fast", "medium"]] = None

    # QC tracking
    biomechanics_qc_version: int = Field(default_factory=get_biomech_qc_version)
    biomechanics_qc_fail: bool = False
    sync_qc_version: int = Field(default_factory=get_cycle_qc_version)
    sync_qc_fail: bool = False

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

    @field_validator("is_outlier")
    @classmethod
    def validate_is_outlier(cls, value: bool, info) -> bool:
        """Update is_outlier based on QC failures."""
        # If any QC failed, should be marked as outlier
        biomech_qc_fail = info.data.get("biomechanics_qc_fail", False)
        sync_qc_fail = info.data.get("sync_qc_fail", False)
        
        # Get audio QC info from parent class fields
        qc_signal_dropout = info.data.get("qc_signal_dropout", False)
        qc_artifact = info.data.get("qc_artifact", False)
        
        if biomech_qc_fail or sync_qc_fail or qc_signal_dropout or qc_artifact:
            return True
        return value

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
            "Pass Number": self.pass_number,
            "Speed": self.speed,
            "Biomechanics QC Version": self.biomechanics_qc_version,
            "Biomechanics QC Fail": self.biomechanics_qc_fail,
            "Sync QC Version": self.sync_qc_version,
            "Sync QC Fail": self.sync_qc_fail,
        })
        return result


# Type alias for backward compatibility
# MovementCycles was merged into Synchronization per Issue #69
MovementCycles = Synchronization
