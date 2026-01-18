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




@dataclass
class AudioProcessing:
    """Audio file processing and QC metadata.
    
    Consolidates audio processing information for logging and validation.
    Combines AudioProcessingMetadata (validation) and AudioProcessingRecord (Excel export).
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
    
    # Data-derived fields (pure statistics from audio data)
    channel_1_rms: Optional[float] = None
    channel_2_rms: Optional[float] = None
    channel_3_rms: Optional[float] = None
    channel_4_rms: Optional[float] = None
    channel_1_peak: Optional[float] = None
    channel_2_peak: Optional[float] = None
    channel_3_peak: Optional[float] = None
    channel_4_peak: Optional[float] = None
    
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
    

    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Excel export."""
        return {
            "Audio File": self.audio_file_name,
            "Bin File": self.audio_bin_file,
            "Pickle File": self.audio_pkl_file,
            "Processing Date": self.processing_date,
            "Status": self.processing_status,
            "Error": self.error_message,
            "Sample Rate (Hz)": self.sample_rate,
            "Channels": self.num_channels,
            "Duration (s)": self.duration_seconds,
            "File Size (MB)": self.file_size_mb,
            "Device Serial": self.device_serial,
            "Firmware Version": self.firmware_version,
            "Recording Time": self.file_time,
            "Ch1 RMS": self.channel_1_rms,
            "Ch2 RMS": self.channel_2_rms,
            "Ch3 RMS": self.channel_3_rms,
            "Ch4 RMS": self.channel_4_rms,
            "Ch1 Peak": self.channel_1_peak,
            "Ch2 Peak": self.channel_2_peak,
            "Ch3 Peak": self.channel_3_peak,
            "Ch4 Peak": self.channel_4_peak,
            "Has Inst. Freq": self.has_instantaneous_freq,
            "QC_not_passed": self.qc_not_passed,
            "QC_not_passed_mic_1": self.qc_not_passed_mic_1,
            "QC_not_passed_mic_2": self.qc_not_passed_mic_2,
            "QC_not_passed_mic_3": self.qc_not_passed_mic_3,
            "QC_not_passed_mic_4": self.qc_not_passed_mic_4,
            "Audio QC Version": self.audio_qc_version,
        }


@dataclass
class BiomechanicsImport:
    """Biomechanics data import metadata.
    
    Tracks biomechanics file import status and characteristics.
    Combines BiomechanicsImportMetadata (validation) and BiomechanicsImportRecord (Excel export).
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
    
    # Data-derived field (pure statistic from biomech data)
    num_data_points: Optional[int] = None
    
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
    


    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Excel export."""
        return {
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
            "Start Time (s)": self.start_time,
            "End Time (s)": self.end_time,
            "Biomech QC Version": self.biomech_qc_version,
        }


@dataclass
class Synchronization:
    """Audio-biomechanics synchronization metadata.
    
    Tracks synchronization process and alignment details.
    Combines SynchronizationMetadata (validation) and SynchronizationRecord (Excel export).
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
    
    # Data-derived field (pure statistic from synced data)
    num_synced_samples: Optional[int] = None
    
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
    
    # Data-derived fields (pure statistics from detection)
    rms_energy: Optional[float] = None
    onset_magnitude: Optional[float] = None
    freq_energy: Optional[float] = None
    
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
    


    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Excel export."""
        return {
            "Sync File": self.sync_file_name,
            "Pass Number": self.pass_number,
            "Speed": self.speed,
            "Processing Date": self.processing_date,
            "Status": self.processing_status,
            "Error": self.error_message,
            "Audio Stomp (s)": self.audio_stomp_time,
            "Bio Left Stomp (s)": self.bio_left_stomp_time,
            "Bio Right Stomp (s)": self.bio_right_stomp_time,
            "Knee Side": self.knee_side,
            "Stomp Offset (s)": self.stomp_offset,
            "Aligned Audio Stomp (s)": self.aligned_audio_stomp_time,
            "Aligned Bio Stomp (s)": self.aligned_bio_stomp_time,
            "Num Samples": self.num_synced_samples,
            "Duration (s)": self.duration_seconds,
            "Sync QC Done": self.sync_qc_performed,
            "Sync QC Passed": self.sync_qc_passed,
            "Audio QC Version": self.audio_qc_version,
            "Biomech QC Version": self.biomech_qc_version,
            "Consensus (s)": self.consensus_time,
            "Consensus Methods": self.consensus_methods,
            "RMS Detect (s)": self.rms_time,
            "Onset Detect (s)": self.onset_time,
            "Freq Detect (s)": self.freq_time,
            "RMS Energy": self.rms_energy,
            "Onset Magnitude": self.onset_magnitude,
            "Freq Energy": self.freq_energy,
            "Method Agreement Span (s)": self.method_agreement_span,
            "Detection Method": self.audio_stomp_method,
            "Selected Time (s)": self.selected_time,
            "Contra Selected Time (s)": self.contra_selected_time,
        }


@dataclass
class MovementCycles:
    """Movement cycle extraction and QC metadata.
    
    Tracks cycle extraction process and aggregate statistics.
    Combines MovementCyclesMetadata (validation) and MovementCyclesRecord (Excel export).
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
    
    # Data-derived fields (filesystem info)
    output_directory: Optional[str] = None
    plots_created: bool = False
    
    # Per-cycle details list (for Cycle Details sheet in Excel)
    per_cycle_details: List['MovementCycle'] = Field(default_factory=list)
    
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
    
    @property
    def acoustic_threshold(self) -> Optional[float]:
        """Alias for qc_acoustic_threshold for convenience."""
        return self.qc_acoustic_threshold

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Excel export."""
        return {
            "Source Sync File": self.sync_file_name,
            "Processing Date": self.processing_date,
            "Status": self.processing_status,
            "Error": self.error_message,
            "Total Cycles": self.total_cycles_extracted,
            "Clean Cycles": self.clean_cycles,
            "Outlier Cycles": self.outlier_cycles,
            "Acoustic Threshold": self.qc_acoustic_threshold,
            "Output Directory": self.output_directory,
            "Plots Created": self.plots_created,
            "Mean Duration (s)": self.mean_cycle_duration_s,
            "Median Duration (s)": self.median_cycle_duration_s,
            "Min Duration (s)": self.min_cycle_duration_s,
            "Max Duration (s)": self.max_cycle_duration_s,
            "Mean Acoustic AUC": self.mean_acoustic_auc,
            "Cycle QC Version": self.cycle_qc_version,
        }


@dataclass
class MovementCycle:
    """Single movement cycle metadata with comprehensive upstream context.
    
    Combines information from audio processing, biomechanics import, and synchronization
    for a single extracted cycle. Uses composition pattern with embedded metadata objects
    to avoid field duplication while maintaining access to all upstream processing context.
    
    Combines MovementCycleMetadata (validation) and MovementCycleRecord (Excel export).
    Fields use snake_case for direct mapping to database columns and Excel headers.
    """
    
    # Cycle identification
    cycle_index: int
    is_outlier: bool = False
    cycle_file: Optional[str] = None  # Path to .pkl file if saved
    
    # Cycle temporal characteristics
    start_time_s: Optional[float] = None  # Start time within synced recording
    end_time_s: Optional[float] = None    # End time within synced recording
    duration_s: Optional[float] = None
    
    # Acoustic characteristics (data-derived from cycle)
    acoustic_auc: Optional[float] = None  # Total acoustic energy
    
    # Data-derived per-channel RMS values
    ch1_rms: Optional[float] = None
    ch2_rms: Optional[float] = None
    ch3_rms: Optional[float] = None
    ch4_rms: Optional[float] = None
    
    # Upstream processing metadata (embedded objects for composition)
    audio_metadata: Optional[AudioProcessing] = None
    biomech_metadata: Optional[BiomechanicsImport] = None
    sync_metadata: Optional[Synchronization] = None
    cycles_metadata: Optional[MovementCycles] = None
    
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
    
    # Helper properties for easy access to flattened data
    @property
    def audio_file_name(self) -> Optional[str]:
        """Get audio file name from embedded metadata."""
        return self.audio_metadata.audio_file_name if self.audio_metadata else None
    
    @property
    def biomechanics_file(self) -> Optional[str]:
        """Get biomechanics file name from embedded metadata."""
        return self.biomech_metadata.biomechanics_file if self.biomech_metadata else None
    
    @property
    def sync_file_name(self) -> Optional[str]:
        """Get sync file name from embedded metadata."""
        return self.sync_metadata.sync_file_name if self.sync_metadata else None
    
    def to_flat_dict(self) -> Dict[str, Any]:
        """Convert to flattened dictionary for internal use.
        
        Extracts key fields from embedded metadata objects and combines them with
        cycle-specific fields into a single flat dictionary.
        """
        result = {
            "cycle_index": self.cycle_index,
            "is_outlier": self.is_outlier,
            "cycle_file": self.cycle_file,
            "start_time_s": self.start_time_s,
            "end_time_s": self.end_time_s,
            "duration_s": self.duration_s,
            "acoustic_auc": self.acoustic_auc,
        }
        
        # Add audio metadata fields
        if self.audio_metadata:
            result.update({
                "audio_file_name": self.audio_metadata.audio_file_name,
                "audio_sample_rate": self.audio_metadata.sample_rate,
                "audio_duration_seconds": self.audio_metadata.duration_seconds,
                "audio_qc_version": self.audio_metadata.audio_qc_version,
                "audio_processing_status": self.audio_metadata.processing_status,
            })
        
        # Add biomechanics metadata fields
        if self.biomech_metadata:
            result.update({
                "biomechanics_file": self.biomech_metadata.biomechanics_file,
                "biomech_sample_rate": self.biomech_metadata.sample_rate,
                "biomech_duration_seconds": self.biomech_metadata.duration_seconds,
                "biomech_qc_version": self.biomech_metadata.biomech_qc_version,
                "biomech_processing_status": self.biomech_metadata.processing_status,
            })
        
        # Add synchronization metadata fields
        if self.sync_metadata:
            result.update({
                "sync_file_name": self.sync_metadata.sync_file_name,
                "audio_stomp_time": self.sync_metadata.audio_stomp_time,
                "bio_left_stomp_time": self.sync_metadata.bio_left_stomp_time,
                "bio_right_stomp_time": self.sync_metadata.bio_right_stomp_time,
                "knee_side": self.sync_metadata.knee_side,
                "stomp_offset": self.sync_metadata.stomp_offset,
                "sync_qc_performed": self.sync_metadata.sync_qc_performed,
                "sync_qc_passed": self.sync_metadata.sync_qc_passed,
                "sync_duration_seconds": self.sync_metadata.duration_seconds,
            })
        
        # Add cycles metadata fields
        if self.cycles_metadata:
            result.update({
                "qc_acoustic_threshold": self.cycles_metadata.qc_acoustic_threshold,
                "cycle_qc_version": self.cycles_metadata.cycle_qc_version,
            })
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Excel export (Cycle Details sheet).
        
        Uses the flattened representation and maps to Excel column names.
        """
        flat_dict = self.to_flat_dict()
        
        result = {
            "Cycle Index": flat_dict.get("cycle_index"),
            "Is Outlier": flat_dict.get("is_outlier"),
            "Cycle File": flat_dict.get("cycle_file"),
            "Audio File": flat_dict.get("audio_file_name"),
            "Biomechanics File": flat_dict.get("biomechanics_file"),
            "Sync File": flat_dict.get("sync_file_name"),
            "Start (s)": flat_dict.get("start_time_s"),
            "End (s)": flat_dict.get("end_time_s"),
            "Duration (s)": flat_dict.get("duration_s"),
            "Acoustic AUC": flat_dict.get("acoustic_auc"),
            # Add data-derived channel RMS values
            "Ch1 RMS": self.ch1_rms,
            "Ch2 RMS": self.ch2_rms,
            "Ch3 RMS": self.ch3_rms,
            "Ch4 RMS": self.ch4_rms,
            "Audio Sample Rate (Hz)": flat_dict.get("audio_sample_rate"),
            "Audio Duration (s)": flat_dict.get("audio_duration_seconds"),
            "Audio QC Version": flat_dict.get("audio_qc_version"),
            "Audio Status": flat_dict.get("audio_processing_status"),
            "Biomech Sample Rate (Hz)": flat_dict.get("biomech_sample_rate"),
            "Biomech Duration (s)": flat_dict.get("biomech_duration_seconds"),
            "Biomech QC Version": flat_dict.get("biomech_qc_version"),
            "Biomech Status": flat_dict.get("biomech_processing_status"),
            "Audio Stomp (s)": flat_dict.get("audio_stomp_time"),
            "Bio Left Stomp (s)": flat_dict.get("bio_left_stomp_time"),
            "Bio Right Stomp (s)": flat_dict.get("bio_right_stomp_time"),
            "Knee Side": flat_dict.get("knee_side"),
            "Stomp Offset (s)": flat_dict.get("stomp_offset"),
            "Sync QC Performed": flat_dict.get("sync_qc_performed"),
            "Sync QC Passed": flat_dict.get("sync_qc_passed"),
            "Sync Duration (s)": flat_dict.get("sync_duration_seconds"),
            "QC Acoustic Threshold": flat_dict.get("qc_acoustic_threshold"),
            "Cycle QC Version": flat_dict.get("cycle_qc_version"),
        }
        return result
