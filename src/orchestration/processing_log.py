"""Processing log module for tracking QC, statistics, and metadata.

This module provides a comprehensive logging system for participant processing,
capturing information about audio conversion, biomechanics import, synchronization,
and movement cycle extraction. Logs are saved as Excel files (.xlsx) in the
maneuver directory and can be incrementally updated when re-processing.

This module uses Pydantic models from src.models for validation and consistency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from src.models import (
    AudioProcessingMetadata,
    BiomechanicsImportMetadata,
    MovementCycleMetadata,
    MovementCyclesMetadata,
    SynchronizationMetadata,
)
from src.qc_versions import (
    get_audio_qc_version,
    get_biomech_qc_version,
    get_cycle_qc_version,
)

if TYPE_CHECKING:
    from src.models import BiomechanicsRecording

logger = logging.getLogger(__name__)


@dataclass
class AudioProcessingRecord:
    """Record of audio file processing and QC.
    
    Wraps AudioProcessingMetadata Pydantic model for Excel export compatibility.
    Use the Pydantic model for validation, this dataclass for logging.
    
    Note: _metadata is required - records can only be created from validated data.
    """
    
    # Pydantic model for validation (required)
    _metadata: AudioProcessingMetadata
    
    # Keep original fields for backward compatibility
    audio_file_name: str = ""
    audio_bin_file: Optional[str] = None
    audio_pkl_file: Optional[str] = None
    processing_date: Optional[datetime] = None
    processing_status: str = "not_processed"
    error_message: Optional[str] = None
    sample_rate: Optional[float] = None
    num_channels: int = 4
    duration_seconds: Optional[float] = None
    file_size_mb: Optional[float] = None
    device_serial: Optional[str] = None
    firmware_version: Optional[int] = None
    file_time: Optional[datetime] = None
    channel_1_rms: Optional[float] = None
    channel_2_rms: Optional[float] = None
    channel_3_rms: Optional[float] = None
    channel_4_rms: Optional[float] = None
    channel_1_peak: Optional[float] = None
    channel_2_peak: Optional[float] = None
    channel_3_peak: Optional[float] = None
    channel_4_peak: Optional[float] = None
    has_instantaneous_freq: bool = False
    QC_not_passed: Optional[str] = None
    QC_not_passed_mic_1: Optional[str] = None
    QC_not_passed_mic_2: Optional[str] = None
    QC_not_passed_mic_3: Optional[str] = None
    QC_not_passed_mic_4: Optional[str] = None
    audio_qc_version: int = field(default_factory=get_audio_qc_version)
    
    @classmethod
    def from_metadata(cls, metadata: AudioProcessingMetadata) -> 'AudioProcessingRecord':
        """Create record from validated Pydantic model."""
        data = metadata.model_dump()
        # Map qc_not_passed fields to QC_not_passed (uppercase convention)
        return cls(
            _metadata=metadata,
            audio_file_name=data["audio_file_name"],
            audio_bin_file=data.get("audio_bin_file"),
            audio_pkl_file=data.get("audio_pkl_file"),
            processing_date=data.get("processing_date"),
            processing_status=data["processing_status"],
            error_message=data.get("error_message"),
            sample_rate=data.get("sample_rate"),
            num_channels=data["num_channels"],
            duration_seconds=data.get("duration_seconds"),
            file_size_mb=data.get("file_size_mb"),
            device_serial=data.get("device_serial"),
            firmware_version=data.get("firmware_version"),
            file_time=data.get("file_time"),
            # channel RMS/peak values removed from metadata - set separately from actual data
            has_instantaneous_freq=data["has_instantaneous_freq"],
            QC_not_passed=data.get("qc_not_passed"),
            QC_not_passed_mic_1=data.get("qc_not_passed_mic_1"),
            QC_not_passed_mic_2=data.get("qc_not_passed_mic_2"),
            QC_not_passed_mic_3=data.get("qc_not_passed_mic_3"),
            QC_not_passed_mic_4=data.get("qc_not_passed_mic_4"),
            audio_qc_version=data["audio_qc_version"],
        )

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
            "QC_not_passed": self.QC_not_passed,
            "QC_not_passed_mic_1": self.QC_not_passed_mic_1,
            "QC_not_passed_mic_2": self.QC_not_passed_mic_2,
            "QC_not_passed_mic_3": self.QC_not_passed_mic_3,
            "QC_not_passed_mic_4": self.QC_not_passed_mic_4,
            "Audio QC Version": self.audio_qc_version,
        }


@dataclass
class BiomechanicsImportRecord:
    """Record of biomechanics data import.
    
    Wraps BiomechanicsImportMetadata Pydantic model for Excel export compatibility.
    
    Note: _metadata is required - records can only be created from validated data.
    """
    
    _metadata: BiomechanicsImportMetadata
    
    biomechanics_file: str = ""
    sheet_name: Optional[str] = None
    processing_date: Optional[datetime] = None
    processing_status: str = "not_processed"
    error_message: Optional[str] = None
    num_recordings: int = 0
    num_passes: int = 0
    duration_seconds: Optional[float] = None
    num_data_points: Optional[int] = None
    sample_rate: Optional[float] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    biomech_qc_version: int = field(default_factory=get_biomech_qc_version)
    
    @classmethod
    def from_metadata(cls, metadata: BiomechanicsImportMetadata) -> 'BiomechanicsImportRecord':
        """Create record from validated Pydantic model."""
        data = metadata.model_dump()
        return cls(
            _metadata=metadata,
            biomechanics_file=data["biomechanics_file"],
            sheet_name=data.get("sheet_name"),
            processing_date=data.get("processing_date"),
            processing_status=data["processing_status"],
            error_message=data.get("error_message"),
            num_recordings=data["num_recordings"],
            num_passes=data["num_passes"],
            duration_seconds=data.get("duration_seconds"),
            # num_data_points removed from metadata - will be set separately from actual data
            sample_rate=data.get("sample_rate"),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            biomech_qc_version=data["biomech_qc_version"],
        )

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
class SynchronizationRecord:
    """Record of audio-biomechanics synchronization.
    
    Wraps SynchronizationMetadata Pydantic model for Excel export compatibility.
    
    Note: _metadata is required - records can only be created from validated data.
    """
    
    _metadata: SynchronizationMetadata
    
    sync_file_name: str = ""
    pass_number: Optional[int] = None
    speed: Optional[str] = None
    processing_date: Optional[datetime] = None
    processing_status: str = "not_processed"
    error_message: Optional[str] = None
    audio_stomp_time: Optional[float] = None
    bio_left_stomp_time: Optional[float] = None
    bio_right_stomp_time: Optional[float] = None
    knee_side: Optional[str] = None
    stomp_offset: Optional[float] = None
    aligned_audio_stomp_time: Optional[float] = None
    aligned_bio_stomp_time: Optional[float] = None
    num_synced_samples: Optional[int] = None
    duration_seconds: Optional[float] = None
    sync_qc_performed: bool = False
    sync_qc_passed: Optional[bool] = None
    audio_qc_version: int = field(default_factory=get_audio_qc_version)
    biomech_qc_version: int = field(default_factory=get_biomech_qc_version)
    consensus_time: Optional[float] = None
    consensus_methods: Optional[str] = None
    rms_time: Optional[float] = None
    onset_time: Optional[float] = None
    freq_time: Optional[float] = None
    rms_energy: Optional[float] = None
    onset_magnitude: Optional[float] = None
    freq_energy: Optional[float] = None
    method_agreement_span: Optional[float] = None
    audio_stomp_method: Optional[str] = None
    selected_time: Optional[float] = None
    contra_selected_time: Optional[float] = None
    
    @classmethod
    def from_metadata(cls, metadata: SynchronizationMetadata) -> 'SynchronizationRecord':
        """Create record from validated Pydantic model."""
        data = metadata.model_dump()
        return cls(
            _metadata=metadata,
            sync_file_name=data["sync_file_name"],
            pass_number=data.get("pass_number"),
            speed=data.get("speed"),
            processing_date=data.get("processing_date"),
            processing_status=data["processing_status"],
            error_message=data.get("error_message"),
            audio_stomp_time=data.get("audio_stomp_time"),
            bio_left_stomp_time=data.get("bio_left_stomp_time"),
            bio_right_stomp_time=data.get("bio_right_stomp_time"),
            knee_side=data.get("knee_side"),
            stomp_offset=data.get("stomp_offset"),
            aligned_audio_stomp_time=data.get("aligned_audio_stomp_time"),
            aligned_bio_stomp_time=data.get("aligned_bio_stomp_time"),
            # num_synced_samples removed from metadata - set separately from actual data
            duration_seconds=data.get("duration_seconds"),
            sync_qc_performed=data["sync_qc_performed"],
            sync_qc_passed=data.get("sync_qc_passed"),
            audio_qc_version=data["audio_qc_version"],
            biomech_qc_version=data["biomech_qc_version"],
            consensus_time=data.get("consensus_time"),
            consensus_methods=data.get("consensus_methods"),
            rms_time=data.get("rms_time"),
            onset_time=data.get("onset_time"),
            freq_time=data.get("freq_time"),
            # rms_energy, onset_magnitude, freq_energy removed from metadata
            method_agreement_span=data.get("method_agreement_span"),
            audio_stomp_method=data.get("audio_stomp_method"),
            selected_time=data.get("selected_time"),
            contra_selected_time=data.get("contra_selected_time"),
        )

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
class MovementCyclesRecord:
    """Record of movement cycle extraction and QC.
    
    Wraps MovementCyclesMetadata Pydantic model for Excel export compatibility.
    
    Note: _metadata is required - records can only be created from validated data.
    """
    
    _metadata: MovementCyclesMetadata
    
    sync_file_name: str = ""
    processing_date: Optional[datetime] = None
    processing_status: str = "not_processed"
    error_message: Optional[str] = None
    total_cycles_extracted: int = 0
    clean_cycles: int = 0
    outlier_cycles: int = 0
    acoustic_threshold: Optional[float] = None
    output_directory: Optional[str] = None
    plots_created: bool = False
    mean_cycle_duration_s: Optional[float] = None
    median_cycle_duration_s: Optional[float] = None
    min_cycle_duration_s: Optional[float] = None
    max_cycle_duration_s: Optional[float] = None
    mean_acoustic_auc: Optional[float] = None
    per_cycle_details: list[MovementCycleRecord] = field(default_factory=list)
    cycle_qc_version: int = field(default_factory=get_cycle_qc_version)
    
    @classmethod
    def from_metadata(cls, metadata: MovementCyclesMetadata, per_cycle_details: Optional[list[MovementCycleRecord]] = None) -> 'MovementCyclesRecord':
        """Create record from validated Pydantic model."""
        data = metadata.model_dump()
        return cls(
            _metadata=metadata,
            sync_file_name=data["sync_file_name"],
            processing_date=data.get("processing_date"),
            processing_status=data["processing_status"],
            error_message=data.get("error_message"),
            total_cycles_extracted=data["total_cycles_extracted"],
            clean_cycles=data["clean_cycles"],
            outlier_cycles=data["outlier_cycles"],
            acoustic_threshold=data.get("qc_acoustic_threshold"),  # Renamed field
            # output_directory and plots_created removed from metadata
            mean_cycle_duration_s=data.get("mean_cycle_duration_s"),
            median_cycle_duration_s=data.get("median_cycle_duration_s"),
            min_cycle_duration_s=data.get("min_cycle_duration_s"),
            max_cycle_duration_s=data.get("max_cycle_duration_s"),
            mean_acoustic_auc=data.get("mean_acoustic_auc"),
            per_cycle_details=per_cycle_details or [],
            cycle_qc_version=data["cycle_qc_version"],
        )

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
            "Acoustic Threshold": self.acoustic_threshold,
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
class MovementCycleRecord:
    """Record of a single movement cycle with comprehensive metadata.
    
    Wraps MovementCycleMetadata Pydantic model for Excel export compatibility.
    The metadata model uses composition to include upstream processing metadata,
    ensuring tight coupling and avoiding field duplication.
    
    Note: _metadata is required - records can only be created from validated data.
    """
    
    _metadata: MovementCycleMetadata
    
    # Per-channel RMS values (data-derived, not in metadata)
    ch1_rms: Optional[float] = None
    ch2_rms: Optional[float] = None
    ch3_rms: Optional[float] = None
    ch4_rms: Optional[float] = None
    
    @classmethod
    def from_metadata(cls, metadata: MovementCycleMetadata, 
                     ch1_rms: Optional[float] = None,
                     ch2_rms: Optional[float] = None,
                     ch3_rms: Optional[float] = None,
                     ch4_rms: Optional[float] = None) -> 'MovementCycleRecord':
        """Create record from validated Pydantic model.
        
        Args:
            metadata: Validated Pydantic metadata model with embedded upstream metadata
            ch1_rms: Channel 1 RMS (data-derived, not in metadata)
            ch2_rms: Channel 2 RMS (data-derived, not in metadata)
            ch3_rms: Channel 3 RMS (data-derived, not in metadata)
            ch4_rms: Channel 4 RMS (data-derived, not in metadata)
        """
        return cls(
            _metadata=metadata,
            ch1_rms=ch1_rms,
            ch2_rms=ch2_rms,
            ch3_rms=ch3_rms,
            ch4_rms=ch4_rms,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Excel export (Cycle Details sheet).
        
        Uses the flattened representation from the metadata model.
        """
        # Start with flattened metadata fields
        flat_dict = self._metadata.to_flat_dict()
        
        # Map to Excel column names
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


@dataclass
class ManeuverProcessingLog:
    """Complete processing log for a knee/maneuver combination."""

    # Identification
    study_id: str
    knee_side: Literal["Left", "Right"]
    maneuver: Literal["walk", "sit_to_stand", "flexion_extension"]
    maneuver_directory: Path

    # Processing stages
    audio_record: Optional[AudioProcessingRecord] = None
    biomechanics_record: Optional[BiomechanicsImportRecord] = None
    synchronization_records: List[SynchronizationRecord] = field(default_factory=list)
    movement_cycles_records: List[MovementCyclesRecord] = field(default_factory=list)

    # Overall metadata
    log_created: Optional[datetime] = None
    log_updated: Optional[datetime] = None

    def update_audio_record(self, record: AudioProcessingRecord) -> None:
        """Update audio processing record."""
        self.audio_record = record
        self.log_updated = datetime.now()

    def update_biomechanics_record(self, record: BiomechanicsImportRecord) -> None:
        """Update biomechanics import record."""
        self.biomechanics_record = record
        self.log_updated = datetime.now()

    def add_synchronization_record(self, record: SynchronizationRecord) -> None:
        """Add or update a synchronization record."""
        # Check if we already have a record for this file
        for i, existing in enumerate(self.synchronization_records):
            if existing.sync_file_name == record.sync_file_name:
                self.synchronization_records[i] = record
                self.log_updated = datetime.now()
                return

        # New record
        self.synchronization_records.append(record)
        self.log_updated = datetime.now()

    def add_movement_cycles_record(self, record: MovementCyclesRecord) -> None:
        """Add or update a movement cycles record."""
        # Normalize names so re-processing replaces existing rows even if
        # extensions differ (e.g., "file.pkl" vs "file").
        def _norm(name: str) -> str:
            return Path(str(name).strip()).stem.lower()

        incoming = _norm(record.sync_file_name)
        for i, existing in enumerate(self.movement_cycles_records):
            if _norm(existing.sync_file_name) == incoming:
                record.sync_file_name = existing.sync_file_name  # preserve original label
                self.movement_cycles_records[i] = record
                self.log_updated = datetime.now()
                return

        # New record
        self.movement_cycles_records.append(record)
        # Dedupe by normalized sync file name, keep the latest
        seen: dict[str, MovementCyclesRecord] = {}
        for rec in self.movement_cycles_records:
            seen[_norm(rec.sync_file_name)] = rec
        self.movement_cycles_records = list(seen.values())
        self.log_updated = datetime.now()

    def save_to_excel(self, filepath: Optional[Path] = None) -> Path:
        """Save processing log to Excel file.

        Args:
            filepath: Optional custom path. If None, saves to maneuver directory.

        Returns:
            Path to saved Excel file.
        """
        if filepath is None:
            filepath = self.maneuver_directory / f"processing_log_{self.study_id}_{self.knee_side}_{self.maneuver}.xlsx"

        filepath = Path(filepath)

        # Create DataFrames for each section
        sheets: Dict[str, pd.DataFrame] = {}

        # Summary sheet
        summary_data = {
            "Study ID": [self.study_id],
            "Knee Side": [self.knee_side],
            "Maneuver": [self.maneuver],
            "Maneuver Directory": [str(self.maneuver_directory)],
            "Log Created": [self.log_created],
            "Log Updated": [self.log_updated],
            "Audio Processed": [self.audio_record is not None and self.audio_record.processing_status == "success"],
            "Biomechanics Imported": [self.biomechanics_record is not None and self.biomechanics_record.processing_status == "success"],
            "Num Synced Files": [len(self.synchronization_records)],
            "Num Movement Cycles": [
                sum(rec.total_cycles_extracted for rec in self.movement_cycles_records)
            ],
        }
        sheets["Summary"] = pd.DataFrame(summary_data)

        # Audio processing sheet
        if self.audio_record:
            sheets["Audio"] = pd.DataFrame([self.audio_record.to_dict()])
        else:
            sheets["Audio"] = pd.DataFrame([{"Note": "No audio processing recorded"}])

        # Biomechanics sheet
        if self.biomechanics_record:
            sheets["Biomechanics"] = pd.DataFrame([self.biomechanics_record.to_dict()])
        else:
            sheets["Biomechanics"] = pd.DataFrame([{"Note": "No biomechanics import recorded"}])

        # Synchronization sheet
        if self.synchronization_records:
            sync_data = [rec.to_dict() for rec in self.synchronization_records]
            sheets["Synchronization"] = pd.DataFrame(sync_data)
        else:
            sheets["Synchronization"] = pd.DataFrame([{"Note": "No synchronization recorded"}])

        # Movement cycles sheet
        if self.movement_cycles_records:
            cycles_data = [rec.to_dict() for rec in self.movement_cycles_records]
            sheets["Movement Cycles"] = pd.DataFrame(cycles_data)
        else:
            sheets["Movement Cycles"] = pd.DataFrame([{"Note": "No movement cycles recorded"}])

        # Optional: Per-cycle details aggregated across all movement cycle analyses
        details_rows: list[Dict[str, Any]] = []
        for rec in self.movement_cycles_records:
            for cycle_record in getattr(rec, "per_cycle_details", []) or []:
                # Convert MovementCycleRecord to dict for Excel export
                details_rows.append(cycle_record.to_dict())
        if details_rows:
            sheets["Cycle Details"] = pd.DataFrame(details_rows)

        # Write to Excel
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            for sheet_name, df in sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        logger.info(f"Saved processing log to {filepath}")
        return filepath

    @classmethod
    def load_from_excel(cls, filepath: Path) -> Optional['ManeuverProcessingLog']:
        """Load processing log from Excel file.

        Args:
            filepath: Path to Excel file.

        Returns:
            ManeuverProcessingLog instance or None if file doesn't exist.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            return None

        try:
            # Read summary sheet
            summary_df = pd.read_excel(filepath, sheet_name="Summary")

            if len(summary_df) == 0:
                return None

            row = summary_df.iloc[0]

            # Create log instance
            log = cls(
                study_id=str(row["Study ID"]),
                knee_side=row["Knee Side"],
                maneuver=row["Maneuver"],
                maneuver_directory=Path(row["Maneuver Directory"]),
                log_created=pd.to_datetime(row["Log Created"]) if pd.notna(row["Log Created"]) else None,
                log_updated=pd.to_datetime(row["Log Updated"]) if pd.notna(row["Log Updated"]) else None,
            )

            # Load audio record if present
            try:
                audio_df = pd.read_excel(filepath, sheet_name="Audio")
                if len(audio_df) > 0 and "Audio File" in audio_df.columns:
                    row = audio_df.iloc[0]
                    metadata = AudioProcessingMetadata(
                        audio_file_name=row.get("Audio File", ""),
                        audio_bin_file=str(row.get("Bin File")) if pd.notna(row.get("Bin File")) else None,
                        audio_pkl_file=str(row.get("Pickle File")) if pd.notna(row.get("Pickle File")) else None,
                        processing_date=pd.to_datetime(row["Processing Date"]) if pd.notna(row.get("Processing Date")) else None,
                        processing_status=row.get("Status", "not_processed"),
                        error_message=str(row.get("Error")) if pd.notna(row.get("Error")) else None,
                        sample_rate=row.get("Sample Rate (Hz)") if pd.notna(row.get("Sample Rate (Hz)")) else None,
                        num_channels=int(row.get("Channels", 4)),
                        duration_seconds=row.get("Duration (s)") if pd.notna(row.get("Duration (s)")) else None,
                        file_size_mb=row.get("File Size (MB)") if pd.notna(row.get("File Size (MB)")) else None,
                        device_serial=str(row.get("Device Serial")) if pd.notna(row.get("Device Serial")) else None,
                        firmware_version=int(row["Firmware Version"]) if pd.notna(row.get("Firmware Version")) else None,
                        file_time=pd.to_datetime(row["Recording Time"]) if pd.notna(row.get("Recording Time")) else None,
                        has_instantaneous_freq=bool(row.get("Has Inst. Freq", False)),
                        qc_not_passed=str(row.get("QC_not_passed")) if pd.notna(row.get("QC_not_passed")) else None,
                        qc_not_passed_mic_1=str(row.get("QC_not_passed_mic_1")) if pd.notna(row.get("QC_not_passed_mic_1")) else None,
                        qc_not_passed_mic_2=str(row.get("QC_not_passed_mic_2")) if pd.notna(row.get("QC_not_passed_mic_2")) else None,
                        qc_not_passed_mic_3=str(row.get("QC_not_passed_mic_3")) if pd.notna(row.get("QC_not_passed_mic_3")) else None,
                        qc_not_passed_mic_4=str(row.get("QC_not_passed_mic_4")) if pd.notna(row.get("QC_not_passed_mic_4")) else None,
                    )
                    record = AudioProcessingRecord.from_metadata(metadata)
                    # Set data-derived fields directly
                    record.channel_1_rms = row.get("Ch1 RMS") if pd.notna(row.get("Ch1 RMS")) else None
                    record.channel_2_rms = row.get("Ch2 RMS") if pd.notna(row.get("Ch2 RMS")) else None
                    record.channel_3_rms = row.get("Ch3 RMS") if pd.notna(row.get("Ch3 RMS")) else None
                    record.channel_4_rms = row.get("Ch4 RMS") if pd.notna(row.get("Ch4 RMS")) else None
                    record.channel_1_peak = row.get("Ch1 Peak") if pd.notna(row.get("Ch1 Peak")) else None
                    record.channel_2_peak = row.get("Ch2 Peak") if pd.notna(row.get("Ch2 Peak")) else None
                    record.channel_3_peak = row.get("Ch3 Peak") if pd.notna(row.get("Ch3 Peak")) else None
                    record.channel_4_peak = row.get("Ch4 Peak") if pd.notna(row.get("Ch4 Peak")) else None
                    log.audio_record = record
            except Exception as e:
                logger.warning(f"Could not load audio record: {e}")

            # Load biomechanics record if present
            try:
                bio_df = pd.read_excel(filepath, sheet_name="Biomechanics")
                if len(bio_df) > 0 and "Biomechanics File" in bio_df.columns:
                    row = bio_df.iloc[0]
                    metadata = BiomechanicsImportMetadata(
                        biomechanics_file=row.get("Biomechanics File", ""),
                        sheet_name=str(row.get("Sheet Name")) if pd.notna(row.get("Sheet Name")) else None,
                        processing_date=pd.to_datetime(row["Processing Date"]) if pd.notna(row.get("Processing Date")) else None,
                        processing_status=row.get("Status", "not_processed"),
                        error_message=str(row.get("Error")) if pd.notna(row.get("Error")) else None,
                        num_recordings=int(row.get("Num Recordings", 0)),
                        num_passes=int(row.get("Num Passes", 0)),
                        duration_seconds=row.get("Duration (s)") if pd.notna(row.get("Duration (s)")) else None,
                        num_data_points=int(row["Num Data Points"]) if pd.notna(row.get("Num Data Points")) else None,
                        sample_rate=row.get("Sample Rate (Hz)") if pd.notna(row.get("Sample Rate (Hz)")) else None,
                        start_time=row.get("Start Time (s)") if pd.notna(row.get("Start Time (s)")) else None,
                        end_time=row.get("End Time (s)") if pd.notna(row.get("End Time (s)")) else None,
                    )
                    log.biomechanics_record = BiomechanicsImportRecord.from_metadata(metadata)
            except Exception as e:
                logger.warning(f"Could not load biomechanics record: {e}")

            # Load synchronization records if present
            try:
                sync_df = pd.read_excel(filepath, sheet_name="Synchronization")
                if len(sync_df) > 0 and "Sync File" in sync_df.columns:
                    for _, row in sync_df.iterrows():
                        metadata = SynchronizationMetadata(
                            sync_file_name=row.get("Sync File", ""),
                            pass_number=int(row["Pass Number"]) if pd.notna(row.get("Pass Number")) else None,
                            speed=str(row.get("Speed")) if pd.notna(row.get("Speed")) else None,
                            processing_date=pd.to_datetime(row["Processing Date"]) if pd.notna(row.get("Processing Date")) else None,
                            processing_status=row.get("Status", "not_processed"),
                            error_message=str(row.get("Error")) if pd.notna(row.get("Error")) else None,
                            audio_stomp_time=row.get("Audio Stomp (s)") if pd.notna(row.get("Audio Stomp (s)")) else None,
                            bio_left_stomp_time=row.get("Bio Left Stomp (s)") if pd.notna(row.get("Bio Left Stomp (s)")) else None,
                            bio_right_stomp_time=row.get("Bio Right Stomp (s)") if pd.notna(row.get("Bio Right Stomp (s)")) else None,
                            knee_side=str(row.get("Knee Side")) if pd.notna(row.get("Knee Side")) else None,
                            stomp_offset=row.get("Stomp Offset (s)") if pd.notna(row.get("Stomp Offset (s)")) else None,
                            aligned_audio_stomp_time=row.get("Aligned Audio Stomp (s)") if pd.notna(row.get("Aligned Audio Stomp (s)")) else None,
                            aligned_bio_stomp_time=row.get("Aligned Bio Stomp (s)") if pd.notna(row.get("Aligned Bio Stomp (s)")) else None,
                            duration_seconds=row.get("Duration (s)") if pd.notna(row.get("Duration (s)")) else None,
                            sync_qc_performed=bool(row.get("Sync QC Done", False)),
                            sync_qc_passed=bool(row["Sync QC Passed"]) if pd.notna(row.get("Sync QC Passed")) else None,
                            audio_qc_version=int(row["Audio QC Version"]) if pd.notna(row.get("Audio QC Version")) else get_audio_qc_version(),
                            biomech_qc_version=int(row["Biomech QC Version"]) if pd.notna(row.get("Biomech QC Version")) else get_biomech_qc_version(),
                            # Load detection method details if present (backward compatible with older logs)
                            consensus_time=row.get("Consensus (s)") if pd.notna(row.get("Consensus (s)")) else None,
                            consensus_methods=str(row.get("Consensus Methods")) if pd.notna(row.get("Consensus Methods")) else None,
                            rms_time=row.get("RMS Detect (s)") if pd.notna(row.get("RMS Detect (s)")) else None,
                            onset_time=row.get("Onset Detect (s)") if pd.notna(row.get("Onset Detect (s)")) else None,
                            freq_time=row.get("Freq Detect (s)") if pd.notna(row.get("Freq Detect (s)")) else None,
                            method_agreement_span=row.get("Method Agreement Span (s)") if pd.notna(row.get("Method Agreement Span (s)")) else None,
                            audio_stomp_method=str(row.get("Detection Method")) if pd.notna(row.get("Detection Method")) else None,
                            selected_time=row.get("Selected Time (s)") if pd.notna(row.get("Selected Time (s)")) else None,
                            contra_selected_time=row.get("Contra Selected Time (s)") if pd.notna(row.get("Contra Selected Time (s)")) else None,
                        )
                        record = SynchronizationRecord.from_metadata(metadata)
                        # Set data-derived fields directly
                        record.num_synced_samples = int(row["Num Samples"]) if pd.notna(row.get("Num Samples")) else None
                        record.rms_energy = row.get("RMS Energy") if pd.notna(row.get("RMS Energy")) else None
                        record.onset_magnitude = row.get("Onset Magnitude") if pd.notna(row.get("Onset Magnitude")) else None
                        record.freq_energy = row.get("Freq Energy") if pd.notna(row.get("Freq Energy")) else None
                        log.synchronization_records.append(record)
            except Exception as e:
                logger.warning(f"Could not load synchronization records: {e}")

            # Load movement cycles records if present
            try:
                cycles_df = pd.read_excel(filepath, sheet_name="Movement Cycles")
                if len(cycles_df) > 0 and "Source Sync File" in cycles_df.columns:
                    for _, row in cycles_df.iterrows():
                        metadata = MovementCyclesMetadata(
                            sync_file_name=row.get("Source Sync File", ""),
                            processing_date=pd.to_datetime(row["Processing Date"]) if pd.notna(row.get("Processing Date")) else None,
                            processing_status=row.get("Status", "not_processed"),
                            error_message=str(row.get("Error")) if pd.notna(row.get("Error")) else None,
                            total_cycles_extracted=int(row.get("Total Cycles", 0)),
                            clean_cycles=int(row.get("Clean Cycles", 0)),
                            outlier_cycles=int(row.get("Outlier Cycles", 0)),
                            qc_acoustic_threshold=row.get("Acoustic Threshold") if pd.notna(row.get("Acoustic Threshold")) else None,
                            mean_cycle_duration_s=float(row.get("Mean Duration (s)")) if pd.notna(row.get("Mean Duration (s)")) else None,
                            median_cycle_duration_s=float(row.get("Median Duration (s)")) if pd.notna(row.get("Median Duration (s)")) else None,
                            min_cycle_duration_s=float(row.get("Min Duration (s)")) if pd.notna(row.get("Min Duration (s)")) else None,
                            max_cycle_duration_s=float(row.get("Max Duration (s)")) if pd.notna(row.get("Max Duration (s)")) else None,
                            mean_acoustic_auc=float(row.get("Mean Acoustic AUC")) if pd.notna(row.get("Mean Acoustic AUC")) else None,
                        )
                        record = MovementCyclesRecord.from_metadata(metadata)
                        # Set data-derived fields directly
                        record.output_directory = str(row.get("Output Directory")) if pd.notna(row.get("Output Directory")) else None
                        record.plots_created = bool(row.get("Plots Created", False))
                        log.movement_cycles_records.append(record)
            except Exception as e:
                logger.warning(f"Could not load movement cycles records: {e}")

            # Load optional Cycle Details sheet
            try:
                details_df = pd.read_excel(filepath, sheet_name="Cycle Details")
                if len(details_df) > 0:
                    # Create MovementCycleRecord objects from Excel rows
                    for _, row in details_df.iterrows():
                        # Create Pydantic metadata from Excel row
                        metadata = MovementCycleMetadata(
                            cycle_index=int(row["Cycle Index"]) if pd.notna(row.get("Cycle Index")) else 0,
                            is_outlier=bool(row.get("Is Outlier", False)),
                            cycle_file=str(row.get("Cycle File")) if pd.notna(row.get("Cycle File")) else None,
                            audio_file_name=str(row.get("Audio File")) if pd.notna(row.get("Audio File")) else None,
                            biomechanics_file=str(row.get("Biomechanics File")) if pd.notna(row.get("Biomechanics File")) else None,
                            sync_file_name=str(row.get("Sync File")) if pd.notna(row.get("Sync File")) else None,
                            start_time_s=row.get("Start (s)") if pd.notna(row.get("Start (s)")) else None,
                            end_time_s=row.get("End (s)") if pd.notna(row.get("End (s)")) else None,
                            duration_s=row.get("Duration (s)") if pd.notna(row.get("Duration (s)")) else None,
                            acoustic_auc=row.get("Acoustic AUC") if pd.notna(row.get("Acoustic AUC")) else None,
                            audio_sample_rate=row.get("Audio Sample Rate (Hz)") if pd.notna(row.get("Audio Sample Rate (Hz)")) else None,
                            audio_duration_seconds=row.get("Audio Duration (s)") if pd.notna(row.get("Audio Duration (s)")) else None,
                            audio_qc_version=int(row["Audio QC Version"]) if pd.notna(row.get("Audio QC Version")) else None,
                            audio_processing_status=str(row.get("Audio Status")) if pd.notna(row.get("Audio Status")) else None,
                            biomech_sample_rate=row.get("Biomech Sample Rate (Hz)") if pd.notna(row.get("Biomech Sample Rate (Hz)")) else None,
                            biomech_duration_seconds=row.get("Biomech Duration (s)") if pd.notna(row.get("Biomech Duration (s)")) else None,
                            biomech_qc_version=int(row["Biomech QC Version"]) if pd.notna(row.get("Biomech QC Version")) else None,
                            biomech_processing_status=str(row.get("Biomech Status")) if pd.notna(row.get("Biomech Status")) else None,
                            audio_stomp_time=row.get("Audio Stomp (s)") if pd.notna(row.get("Audio Stomp (s)")) else None,
                            bio_left_stomp_time=row.get("Bio Left Stomp (s)") if pd.notna(row.get("Bio Left Stomp (s)")) else None,
                            bio_right_stomp_time=row.get("Bio Right Stomp (s)") if pd.notna(row.get("Bio Right Stomp (s)")) else None,
                            knee_side=str(row.get("Knee Side")) if pd.notna(row.get("Knee Side")) else None,
                            stomp_offset=row.get("Stomp Offset (s)") if pd.notna(row.get("Stomp Offset (s)")) else None,
                            sync_qc_performed=bool(row.get("Sync QC Performed", False)),
                            sync_qc_passed=bool(row.get("Sync QC Passed")) if pd.notna(row.get("Sync QC Passed")) else None,
                            sync_duration_seconds=row.get("Sync Duration (s)") if pd.notna(row.get("Sync Duration (s)")) else None,
                            qc_acoustic_threshold=row.get("QC Acoustic Threshold") if pd.notna(row.get("QC Acoustic Threshold")) else None,
                            cycle_qc_version=int(row["Cycle QC Version"]) if pd.notna(row.get("Cycle QC Version")) else None,
                            processing_date=pd.to_datetime(row["Processing Date"]) if pd.notna(row.get("Processing Date")) else None,
                        )
                        
                        # Create record from metadata with channel RMS (data-derived)
                        cycle_record = MovementCycleRecord.from_metadata(
                            metadata,
                            ch1_rms=row.get("Ch1 RMS") if pd.notna(row.get("Ch1 RMS")) else None,
                            ch2_rms=row.get("Ch2 RMS") if pd.notna(row.get("Ch2 RMS")) else None,
                            ch3_rms=row.get("Ch3 RMS") if pd.notna(row.get("Ch3 RMS")) else None,
                            ch4_rms=row.get("Ch4 RMS") if pd.notna(row.get("Ch4 RMS")) else None,
                        )
                        
                        # Attach to corresponding MovementCyclesRecord by sync file name
                        for rec in log.movement_cycles_records:
                            if rec.sync_file_name == cycle_record.sync_file_name:
                                rec.per_cycle_details.append(cycle_record)
                                break
            except Exception as e:
                # Optional sheet may not exist or have different format
                logger.debug(f"Could not load Cycle Details sheet: {e}")
                pass

            return log

        except Exception as e:
            logger.error(f"Failed to load processing log from {filepath}: {e}")
            return None

    @classmethod
    def get_or_create(
        cls,
        study_id: str,
        knee_side: Literal["Left", "Right"],
        maneuver: Literal["walk", "sit_to_stand", "flexion_extension"],
        maneuver_directory: Path,
    ) -> 'ManeuverProcessingLog':
        """Get existing log or create a new one.

        Args:
            study_id: Study participant ID
            knee_side: "Left" or "Right"
            maneuver: Maneuver type
            maneuver_directory: Path to maneuver directory

        Returns:
            ManeuverProcessingLog instance (loaded or newly created)
        """
        maneuver_directory = Path(maneuver_directory)
        log_path = maneuver_directory / f"processing_log_{study_id}_{knee_side}_{maneuver}.xlsx"

        # Try to load existing log
        existing_log = cls.load_from_excel(log_path)
        if existing_log:
            logger.info(f"Loaded existing processing log from {log_path}")
            return existing_log

        # Create new log
        logger.info(f"Creating new processing log for {study_id} {knee_side} {maneuver}")
        return cls(
            study_id=study_id,
            knee_side=knee_side,
            maneuver=maneuver,
            maneuver_directory=maneuver_directory,
            log_created=datetime.now(),
            log_updated=datetime.now(),
        )


@dataclass
class KneeProcessingLog:
    """Master processing log for a knee (Left or Right) tracking all maneuvers."""

    # Identification
    study_id: str
    knee_side: Literal["Left", "Right"]
    knee_directory: Path

    # Overall metadata
    log_created: Optional[datetime] = None
    log_updated: Optional[datetime] = None

    # Maneuver summaries with stomp times
    maneuver_summaries: List[Dict[str, Any]] = field(default_factory=list)

    def update_maneuver_summary(
        self,
        maneuver: Literal["walk", "sit_to_stand", "flexion_extension"],
        maneuver_log: ManeuverProcessingLog,
    ) -> None:
        """Update or add a maneuver summary to the knee log.

        Args:
            maneuver: Maneuver type
            maneuver_log: The maneuver-level processing log
        """
        # Remove existing summary for this maneuver
        self.maneuver_summaries = [
            s for s in self.maneuver_summaries if s.get("Maneuver") != maneuver
        ]

        # Build new summary
        summary: Dict[str, Any] = {
            "Maneuver": maneuver,
            "Last Updated": datetime.now(),
            "Audio Processed": maneuver_log.audio_record is not None and maneuver_log.audio_record.processing_status == "success",
            "Biomechanics Imported": maneuver_log.biomechanics_record is not None and maneuver_log.biomechanics_record.processing_status == "success",
            "Num Synced Files": len(maneuver_log.synchronization_records),
            "Num Movement Cycles": sum(rec.total_cycles_extracted for rec in maneuver_log.movement_cycles_records),
        }

        # Add stomp time details from synchronization records
        if maneuver_log.synchronization_records:
            # Collect all stomp times across sync records
            audio_stomps = [r.audio_stomp_time for r in maneuver_log.synchronization_records if r.audio_stomp_time is not None]
            bio_stomps = []
            for r in maneuver_log.synchronization_records:
                if r.knee_side and r.knee_side.lower() == "left":
                    if r.bio_left_stomp_time is not None:
                        bio_stomps.append(r.bio_left_stomp_time)
                elif r.knee_side and r.knee_side.lower() == "right":
                    if r.bio_right_stomp_time is not None:
                        bio_stomps.append(r.bio_right_stomp_time)

            offsets = [r.stomp_offset for r in maneuver_log.synchronization_records if r.stomp_offset is not None]
            aligned_audio = [r.aligned_audio_stomp_time for r in maneuver_log.synchronization_records if r.aligned_audio_stomp_time is not None]
            aligned_bio = [r.aligned_bio_stomp_time for r in maneuver_log.synchronization_records if r.aligned_bio_stomp_time is not None]

            # Detection method aggregates
            consensus_times = [r.consensus_time for r in maneuver_log.synchronization_records if r.consensus_time is not None]
            rms_times = [r.rms_time for r in maneuver_log.synchronization_records if r.rms_time is not None]
            onset_times = [r.onset_time for r in maneuver_log.synchronization_records if r.onset_time is not None]
            freq_times = [r.freq_time for r in maneuver_log.synchronization_records if r.freq_time is not None]
            agreement_spans = [r.method_agreement_span for r in maneuver_log.synchronization_records if r.method_agreement_span is not None]

            summary.update({
                # Representative stomp metrics (aggregated across sync records)
                "Audio Stomp (s)": float(np.median(audio_stomps)) if audio_stomps else None,
                "Bio Stomp (s)": float(np.median(bio_stomps)) if bio_stomps else None,
                "Stomp Offset (s)": float(np.median(offsets)) if offsets else None,
                "Aligned Audio Stomp (s)": float(np.median(aligned_audio)) if aligned_audio else None,
                "Aligned Bio Stomp (s)": float(np.median(aligned_bio)) if aligned_bio else None,
                # Detection method aggregates (median for robustness)
                "Consensus (s)": float(np.median(consensus_times)) if consensus_times else None,
                "RMS Detect (s)": float(np.median(rms_times)) if rms_times else None,
                "Onset Detect (s)": float(np.median(onset_times)) if onset_times else None,
                "Freq Detect (s)": float(np.median(freq_times)) if freq_times else None,
                "Method Agreement Span (s)": float(np.median(agreement_spans)) if agreement_spans else None,
            })

        self.maneuver_summaries.append(summary)
        self.log_updated = datetime.now()

    def save_to_excel(self, filepath: Optional[Path] = None) -> Path:
        """Save knee-level processing log to Excel file.

        Args:
            filepath: Optional custom path. If None, saves to knee directory.

        Returns:
            Path to saved Excel file.
        """
        if filepath is None:
            filepath = self.knee_directory / f"knee_processing_log_{self.study_id}_{self.knee_side}.xlsx"

        filepath = Path(filepath)

        # Create DataFrames for each section
        sheets: Dict[str, pd.DataFrame] = {}

        # Summary sheet
        summary_data = {
            "Study ID": [self.study_id],
            "Knee Side": [self.knee_side],
            "Knee Directory": [str(self.knee_directory)],
            "Log Created": [self.log_created],
            "Log Updated": [self.log_updated],
            "Total Maneuvers": [len(self.maneuver_summaries)],
        }
        sheets["Summary"] = pd.DataFrame(summary_data)

        # Maneuver summaries sheet
        if self.maneuver_summaries:
            sheets["Maneuver Summaries"] = pd.DataFrame(self.maneuver_summaries)
        else:
            sheets["Maneuver Summaries"] = pd.DataFrame([{"Note": "No maneuvers processed"}])

        # Write to Excel
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            for sheet_name, df in sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        logger.info(f"Saved knee processing log to {filepath}")
        return filepath

    @classmethod
    def load_from_excel(cls, filepath: Path) -> Optional['KneeProcessingLog']:
        """Load knee processing log from Excel file.

        Args:
            filepath: Path to Excel file.

        Returns:
            KneeProcessingLog instance or None if file doesn't exist.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            return None

        try:
            # Read summary sheet
            summary_df = pd.read_excel(filepath, sheet_name="Summary")

            if len(summary_df) == 0:
                return None

            row = summary_df.iloc[0]

            # Create log instance
            log = cls(
                study_id=str(row["Study ID"]),
                knee_side=row["Knee Side"],
                knee_directory=Path(row["Knee Directory"]),
                log_created=pd.to_datetime(row["Log Created"]) if pd.notna(row["Log Created"]) else None,
                log_updated=pd.to_datetime(row["Log Updated"]) if pd.notna(row["Log Updated"]) else None,
            )

            # Load maneuver summaries if present
            try:
                summaries_df = pd.read_excel(filepath, sheet_name="Maneuver Summaries")
                if len(summaries_df) > 0 and "Maneuver" in summaries_df.columns:
                    for _, row in summaries_df.iterrows():
                        summary = {
                            "Maneuver": row.get("Maneuver"),
                            "Last Updated": pd.to_datetime(row["Last Updated"]) if pd.notna(row.get("Last Updated")) else None,
                            "Audio Processed": bool(row.get("Audio Processed", False)),
                            "Biomechanics Imported": bool(row.get("Biomechanics Imported", False)),
                            "Num Synced Files": int(row.get("Num Synced Files", 0)),
                            "Num Movement Cycles": int(row.get("Num Movement Cycles", 0)),
                            "Audio Stomp (s)": float(row["Audio Stomp (s)"]) if pd.notna(row.get("Audio Stomp (s)")) else None,
                            "Bio Stomp (s)": float(row["Bio Stomp (s)"]) if pd.notna(row.get("Bio Stomp (s)")) else None,
                            "Stomp Offset (s)": float(row["Stomp Offset (s)"]) if pd.notna(row.get("Stomp Offset (s)")) else None,
                            "Aligned Audio Stomp (s)": float(row["Aligned Audio Stomp (s)"]) if pd.notna(row.get("Aligned Audio Stomp (s)")) else None,
                            "Aligned Bio Stomp (s)": float(row["Aligned Bio Stomp (s)"]) if pd.notna(row.get("Aligned Bio Stomp (s)")) else None,
                            # Detection method aggregates
                            "Consensus (s)": float(row["Consensus (s)"]) if pd.notna(row.get("Consensus (s)")) else None,
                            "RMS Detect (s)": float(row["RMS Detect (s)"]) if pd.notna(row.get("RMS Detect (s)")) else None,
                            "Onset Detect (s)": float(row["Onset Detect (s)"]) if pd.notna(row.get("Onset Detect (s)")) else None,
                            "Freq Detect (s)": float(row["Freq Detect (s)"]) if pd.notna(row.get("Freq Detect (s)")) else None,
                            "Method Agreement Span (s)": float(row["Method Agreement Span (s)"]) if pd.notna(row.get("Method Agreement Span (s)")) else None,
                        }
                        log.maneuver_summaries.append(summary)
            except Exception as e:
                logger.warning(f"Could not load maneuver summaries: {e}")

            return log

        except Exception as e:
            logger.error(f"Failed to load knee processing log from {filepath}: {e}")
            return None

    @classmethod
    def get_or_create(
        cls,
        study_id: str,
        knee_side: Literal["Left", "Right"],
        knee_directory: Path,
    ) -> 'KneeProcessingLog':
        """Get existing log or create a new one.

        Args:
            study_id: Study participant ID
            knee_side: "Left" or "Right"
            knee_directory: Path to knee directory

        Returns:
            KneeProcessingLog instance (loaded or newly created)
        """
        knee_directory = Path(knee_directory)
        log_path = knee_directory / f"knee_processing_log_{study_id}_{knee_side}.xlsx"

        # Try to load existing log
        existing_log = cls.load_from_excel(log_path)
        if existing_log:
            logger.info(f"Loaded existing knee processing log from {log_path}")
            return existing_log

        # Create new log
        logger.info(f"Creating new knee processing log for {study_id} {knee_side}")
        return cls(
            study_id=study_id,
            knee_side=knee_side,
            knee_directory=knee_directory,
            log_created=datetime.now(),
            log_updated=datetime.now(),
        )


# Helper functions to create records from processing data
# These functions now use Pydantic models for validation
#
# IMPORTANT: Distinction between metadata fields and data-derived fields:
# - Metadata fields: Go into Pydantic model, represent recording properties/QC decisions
#   Examples: sample_rate, duration_seconds, processing_status
# - Data-derived fields: Set directly on record after creation, pure statistics from data
#   Examples: channel_1_rms, channel_1_peak, num_synced_samples
# This separation keeps Pydantic models focused on validation, not computation.


def create_audio_record_from_data(
    audio_file_name: str,
    audio_df: Optional[pd.DataFrame] = None,
    audio_bin_path: Optional[Path] = None,
    audio_pkl_path: Optional[Path] = None,
    metadata: Optional[Dict[str, Any]] = None,
    error: Optional[Exception] = None,
) -> AudioProcessingRecord:
    """Create an AudioProcessingRecord from processing data with Pydantic validation.

    Args:
        audio_file_name: Name of the audio file
        audio_df: Processed audio DataFrame
        audio_bin_path: Path to binary file
        audio_pkl_path: Path to pickle file
        metadata: Optional metadata dictionary from audio reader
        error: Exception if processing failed

    Returns:
        AudioProcessingRecord instance (validated through Pydantic)
        
    Raises:
        ValidationError: If data doesn't meet Pydantic model requirements
    """
    # Build data dict for Pydantic validation
    data: Dict[str, Any] = {
        "audio_file_name": audio_file_name,
        "audio_bin_file": str(audio_bin_path) if audio_bin_path else None,
        "audio_pkl_file": str(audio_pkl_path) if audio_pkl_path else None,
        "processing_date": datetime.now(),
        "processing_status": "error" if error else "not_processed",
        "error_message": str(error) if error else None,
    }

    if error:
        # Early return on error - validate and return
        validated = AudioProcessingMetadata(**data)
        return AudioProcessingRecord.from_metadata(validated)

    data["processing_status"] = "success"

    # Extract metadata if available
    if metadata:
        # Normalize sample rate from metadata
        meta_fs = metadata.get("fs")
        if isinstance(meta_fs, (int, float)):
            data["sample_rate"] = float(meta_fs)
        # Normalize firmware version
        fv = metadata.get("devFirmwareVersion")
        if isinstance(fv, (int, float)):
            data["firmware_version"] = int(fv)
        # Normalize device serial: handle list/tuple
        ds = metadata.get("deviceSerial")
        if isinstance(ds, (list, tuple)) and ds:
            data["device_serial"] = str(ds[0])
        elif ds is not None:
            data["device_serial"] = str(ds)
        # Recording time passthrough
        data["file_time"] = metadata.get("fileTime")

    # Calculate statistics from DataFrame
    channel_stats = {}
    if audio_df is not None:
        # Duration (metadata field - used for QC, represents recording metadata)
        # Unlike RMS/peak (pure statistics), duration is about the recording's extent
        if "tt" in audio_df.columns:
            dur = audio_df["tt"].iloc[-1] - audio_df["tt"].iloc[0]
            if hasattr(dur, 'total_seconds'):
                data["duration_seconds"] = dur.total_seconds()
            else:
                data["duration_seconds"] = float(dur)

        # Sample rate from data (robust median) if not in metadata
        if (data.get("sample_rate") is None) and (len(audio_df) > 1) and ("tt" in audio_df.columns):
            try:
                tt = audio_df["tt"]
                if hasattr(tt, "dt"):
                    tt_s = tt.dt.total_seconds().astype(float).to_numpy()
                else:
                    tt_s = pd.to_numeric(tt, errors="coerce").astype(float).to_numpy()
                tt_s = tt_s[~np.isnan(tt_s)]
                if tt_s.size >= 2:
                    dt_med = float(np.median(np.diff(tt_s)))
                    sr = (1.0 / dt_med) if dt_med > 0 else None
                    # Round to nearest 100 Hz for stability (typical audio board sample rates: 46800, 46900, 47000)
                    SAMPLE_RATE_ROUNDING = 100
                    data["sample_rate"] = float(int(round(sr / SAMPLE_RATE_ROUNDING)) * SAMPLE_RATE_ROUNDING) if sr else None
            except Exception:
                pass

        # Channel statistics (pure data-derived, not metadata)
        for ch_num in range(1, 5):
            ch_name = f"ch{ch_num}"
            if ch_name in audio_df.columns:
                ch_data = audio_df[ch_name].values
                rms = float(np.sqrt(np.mean(ch_data ** 2)))
                peak = float(np.max(np.abs(ch_data)))
                channel_stats[f"channel_{ch_num}_rms"] = rms
                channel_stats[f"channel_{ch_num}_peak"] = peak

        # Check for instantaneous frequency columns
        data["has_instantaneous_freq"] = any(
            col.startswith("f_ch") for col in audio_df.columns
        )

    # File size
    if audio_bin_path and audio_bin_path.exists():
        data["file_size_mb"] = audio_bin_path.stat().st_size / (1024 * 1024)

    # Validate using Pydantic model and create record
    validated = AudioProcessingMetadata(**data)
    record = AudioProcessingRecord.from_metadata(validated)
    
    # Set pure data-derived fields directly (not in metadata)
    for field, value in channel_stats.items():
        setattr(record, field, value)
    
    return record


def create_biomechanics_record_from_data(
    biomechanics_file: Path,
    recordings: List['BiomechanicsRecording'],
    sheet_name: Optional[str] = None,
    error: Optional[Exception] = None,
) -> BiomechanicsImportRecord:
    """Create a BiomechanicsImportRecord from import data.

    Args:
        biomechanics_file: Path to biomechanics Excel file
        recordings: List of BiomechanicsRecording objects
        sheet_name: Name of the sheet that was read
        error: Exception if import failed

    Returns:
        BiomechanicsImportRecord instance
    """
    # Build data dictionary
    data = {
        "biomechanics_file": str(biomechanics_file),
        "sheet_name": sheet_name,
        "processing_date": datetime.now(),
    }

    if error:
        data["processing_status"] = "error"
        data["error_message"] = str(error)
        # Validate and return early for error case
        validated = BiomechanicsImportMetadata(**data)
        return BiomechanicsImportRecord.from_metadata(validated)

    data["processing_status"] = "success"
    data["num_recordings"] = len(recordings)

    # Calculate data-derived fields (not stored in metadata)
    num_data_points = None
    
    if recordings:
        # Count passes (for walking)
        pass_numbers = set()
        for rec in recordings:
            if hasattr(rec, 'pass_number') and rec.pass_number is not None:
                pass_numbers.add(rec.pass_number)
        data["num_passes"] = len(pass_numbers)

        # Get statistics from first recording
        first_rec = recordings[0]
        if hasattr(first_rec, 'data') and isinstance(first_rec.data, pd.DataFrame):
            df = first_rec.data
            # num_data_points is pure data-derived (like RMS/peak), set on record not metadata
            num_data_points = len(df)

            # Identify time-like column
            time_col: Optional[str] = None
            for cand in ["TIME", "Time (sec)", "Time", "tt"]:
                if cand in df.columns:
                    time_col = cand
                    break

            if time_col is not None:
                def _seconds_array(series: pd.Series) -> np.ndarray:
                    """Convert a time-like series to seconds as float."""
                    try:
                        if np.issubdtype(series.dtype, np.timedelta64):
                            vals = series.dt.total_seconds().astype(float).to_numpy()
                        elif np.issubdtype(series.dtype, np.datetime64):
                            vals = (series.view('int64') / 1e9).astype(float).to_numpy()
                        else:
                            vals = pd.to_numeric(series, errors="coerce").astype(float).to_numpy()
                    except Exception:
                        vals = pd.to_numeric(series, errors="coerce").astype(float).to_numpy()
                    vals = vals[~np.isnan(vals)]
                    return vals

                times = _seconds_array(df[time_col])
                if times.size >= 2:
                    # Normalize to start at zero to avoid huge absolute timestamps
                    times = times - times[0]
                    data["start_time"] = 0.0
                    data["end_time"] = float(times[-1])
                    data["duration_seconds"] = float(times[-1])
                    dt = float(np.median(np.diff(times)))
                    if dt and dt > 0:
                        sr = float(1.0 / dt)
                        # Round to nearest common biomechanics sampling rates
                        candidates = np.array([60, 100, 120], dtype=float)
                        nearest = int(candidates[np.argmin(np.abs(candidates - sr))])
                        data["sample_rate"] = float(nearest)
            # Final fallback: infer duration from rows if sample_rate known elsewhere (rare)
            if data.get("duration_seconds") is None and data.get("sample_rate") and data["sample_rate"] > 0:
                data["duration_seconds"] = float(len(df) / data["sample_rate"])

    # Validate using Pydantic model and create record
    validated = BiomechanicsImportMetadata(**data)
    record = BiomechanicsImportRecord.from_metadata(validated)
    
    # Set data-derived fields directly (not in metadata)
    record.num_data_points = num_data_points
    
    return record


def create_sync_record_from_data(
    sync_file_name: str,
    synced_df: pd.DataFrame,
    audio_stomp_time: Optional[float] = None,
    bio_left_stomp_time: Optional[float] = None,
    bio_right_stomp_time: Optional[float] = None,
    knee_side: Optional[str] = None,
    pass_number: Optional[int] = None,
    speed: Optional[str] = None,
    detection_results: Optional[Dict[str, Any]] = None,
    error: Optional[Exception] = None,
) -> SynchronizationRecord:
    """Create a SynchronizationRecord from sync data.

    Args:
        sync_file_name: Name of the synchronized file
        synced_df: Synchronized DataFrame
        audio_stomp_time: Audio stomp time in seconds (in original audio time coords)
        bio_left_stomp_time: Biomechanics left stomp time in seconds (in bio time coords)
        bio_right_stomp_time: Biomechanics right stomp time in seconds (in bio time coords)
        knee_side: "left" or "right"
        pass_number: Pass number (for walking)
        speed: Speed (for walking)
        error: Exception if sync failed

    Returns:
        SynchronizationRecord instance with alignment details
    """
    def _to_seconds(value: Any) -> Optional[float]:
        if value is None:
            return None
        # Fast path: plain numbers
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        # Timedelta-like
        if hasattr(value, "total_seconds"):
            try:
                return float(value.total_seconds())
            except Exception:
                pass
        # Strings or other objects that to_timedelta can parse
        try:
            td = pd.to_timedelta(value)
            return float(td.total_seconds())
        except Exception:
            try:
                return float(value)
            except Exception:
                return None

    audio_stomp_s = _to_seconds(audio_stomp_time)
    bio_left_s = _to_seconds(bio_left_stomp_time)
    bio_right_s = _to_seconds(bio_right_stomp_time)

    # Determine the recorded knee's bio stomp time
    bio_stomp_s: Optional[float] = None
    if knee_side and knee_side.lower() == "left":
        bio_stomp_s = bio_left_s
    elif knee_side and knee_side.lower() == "right":
        bio_stomp_s = bio_right_s

    # Calculate alignment offset and aligned stomp times
    # Offset = bio_stomp - audio_stomp (the shift applied to audio timestamps)
    stomp_offset: Optional[float] = None
    aligned_audio_stomp: Optional[float] = None
    aligned_bio_stomp: Optional[float] = None

    if audio_stomp_s is not None and bio_stomp_s is not None:
        stomp_offset = bio_stomp_s - audio_stomp_s
        # After alignment, audio_stomp appears at: audio_stomp + offset = bio_stomp
        aligned_audio_stomp = audio_stomp_s + stomp_offset  # Should equal bio_stomp_s
        # Bio stomp remains at its original position (synced timeline is in bio coords)
        aligned_bio_stomp = bio_stomp_s

    # Build data dictionary
    data = {
        "sync_file_name": sync_file_name,
        "pass_number": pass_number,
        "speed": speed,
        "processing_date": datetime.now(),
        "audio_stomp_time": audio_stomp_s,
        "bio_left_stomp_time": bio_left_s,
        "bio_right_stomp_time": bio_right_s,
        "knee_side": knee_side,
        "stomp_offset": stomp_offset,
        "aligned_audio_stomp_time": aligned_audio_stomp,
        "aligned_bio_stomp_time": aligned_bio_stomp,
    }

    # Populate detection method details if provided
    # Store energy/magnitude as separate data-derived fields (not metadata)
    rms_energy = None
    onset_magnitude = None
    freq_energy = None
    
    if detection_results:
        try:
            data["consensus_time"] = _to_seconds(detection_results.get("consensus_time"))
            # Extract which methods contributed to consensus
            consensus_methods_list = detection_results.get("consensus_methods", [])
            if consensus_methods_list:
                data["consensus_methods"] = ", ".join(consensus_methods_list)

            data["rms_time"] = _to_seconds(detection_results.get("rms_time"))
            data["onset_time"] = _to_seconds(detection_results.get("onset_time"))
            data["freq_time"] = _to_seconds(detection_results.get("freq_time"))
            
            # Energies/magnitudes are data-derived, not metadata
            rms_energy = (
                float(detection_results.get("rms_energy"))
                if detection_results.get("rms_energy") is not None else None
            )
            onset_magnitude = (
                float(detection_results.get("onset_magnitude"))
                if detection_results.get("onset_magnitude") is not None else None
            )
            freq_energy = (
                float(detection_results.get("freq_energy"))
                if detection_results.get("freq_energy") is not None else None
            )

            # Method agreement span: only for methods that contributed to consensus
            # Get times for methods that contributed
            method_times_used = []
            if consensus_methods_list:
                if "rms" in consensus_methods_list and data.get("rms_time") is not None:
                    method_times_used.append(data["rms_time"])
                if "onset" in consensus_methods_list and data.get("onset_time") is not None:
                    method_times_used.append(data["onset_time"])
                if "freq" in consensus_methods_list and data.get("freq_time") is not None:
                    method_times_used.append(data["freq_time"])

            if method_times_used:
                data["method_agreement_span"] = float(max(method_times_used) - min(method_times_used))

            # Biomechanics-guided detection metadata
            data["audio_stomp_method"] = detection_results.get("audio_stomp_method")
            data["selected_time"] = _to_seconds(detection_results.get("selected_time"))
            data["contra_selected_time"] = _to_seconds(detection_results.get("contra_selected_time"))
        except Exception as e:
            logger.debug(f"Failed to populate detection_results in sync record: {e}")

    if error:
        data["processing_status"] = "error"
        data["error_message"] = str(error)
        # Validate and return early for error case
        validated = SynchronizationMetadata(**data)
        record = SynchronizationRecord.from_metadata(validated)
        # Set data-derived fields
        record.rms_energy = rms_energy
        record.onset_magnitude = onset_magnitude
        record.freq_energy = freq_energy
        return record

    data["processing_status"] = "success"
    # num_synced_samples is pure data-derived, duration_seconds is metadata
    num_synced_samples = len(synced_df)

    if "tt" in synced_df.columns and len(synced_df) > 0:
        duration = synced_df["tt"].iloc[-1] - synced_df["tt"].iloc[0]
        if hasattr(duration, 'total_seconds'):
            data["duration_seconds"] = duration.total_seconds()
        else:
            data["duration_seconds"] = float(duration)

    # Validate using Pydantic model and create record
    validated = SynchronizationMetadata(**data)
    record = SynchronizationRecord.from_metadata(validated)
    
    # Set pure data-derived fields directly (not in metadata)
    record.num_synced_samples = num_synced_samples
    record.rms_energy = rms_energy
    record.onset_magnitude = onset_magnitude
    record.freq_energy = freq_energy
    
    return record


def create_cycles_record_from_data(
    sync_file_name: str,
    clean_cycles: List[pd.DataFrame],
    outlier_cycles: List[pd.DataFrame],
    output_dir: Optional[Path] = None,
    acoustic_threshold: Optional[float] = None,
    plots_created: bool = False,
    error: Optional[Exception] = None,
    # Optional context from upstream processing
    audio_record: Optional[AudioProcessingRecord] = None,
    biomech_record: Optional[BiomechanicsImportRecord] = None,
    sync_record: Optional[SynchronizationRecord] = None,
) -> MovementCyclesRecord:
    """Create a MovementCyclesRecord from cycle extraction data.

    Args:
        sync_file_name: Name of the source sync file
        clean_cycles: List of clean cycle DataFrames
        outlier_cycles: List of outlier cycle DataFrames
        output_dir: Output directory path
        acoustic_threshold: Acoustic threshold used for QC
        plots_created: Whether plots were created
        error: Exception if extraction failed
        audio_record: Optional audio processing record for context
        biomech_record: Optional biomechanics import record for context
        sync_record: Optional synchronization record for context

    Returns:
        MovementCyclesRecord instance
    """
    # Build data dictionary
    data = {
        "sync_file_name": sync_file_name,
        "processing_date": datetime.now(),
        "qc_acoustic_threshold": acoustic_threshold,  # Renamed field
    }

    if error:
        data["processing_status"] = "error"
        data["error_message"] = str(error)
        # Validate and return early for error case
        validated = MovementCyclesMetadata(**data)
        # Set output_directory and plots_created directly on record (not in metadata)
        record = MovementCyclesRecord.from_metadata(validated, per_cycle_details=[])
        record.output_directory = str(output_dir) if output_dir else None
        record.plots_created = plots_created
        return record

    data["processing_status"] = "success"

    # If output_dir provided, derive cycles from .pkl files only to avoid counting plots
    pkl_cycles: list[Path] = []
    if output_dir is not None:
        try:
            # Only include cycles generated from this specific synced file/pass.
            # Cycle files are saved as "<sync_stem>_cycle_###.pkl" or "<sync_stem>_outlier_###.pkl".
            all_pkls = sorted(set(Path(output_dir).rglob("*.pkl")))
            pkl_cycles = [p for p in all_pkls if p.name.startswith(f"{Path(sync_file_name).stem}_")]
        except Exception:
            pkl_cycles = []

    if pkl_cycles:
        # Count total cycles for this synced file only
        data["total_cycles_extracted"] = len(pkl_cycles)
        # Preserve clean/outlier counts from inputs if available
        data["clean_cycles"] = len(clean_cycles)
        data["outlier_cycles"] = len(outlier_cycles)
    else:
        # Fallback to in-memory lists
        data["clean_cycles"] = len(clean_cycles)
        data["outlier_cycles"] = len(outlier_cycles)
        data["total_cycles_extracted"] = len(clean_cycles) + len(outlier_cycles)

    # Helper to compute duration (s) and acoustic AUC for one cycle
    def _cycle_metrics(cycle_df: pd.DataFrame) -> tuple[float, float, float, float]:
        """Compute start/end (absolute within synced file), duration, and AUC."""
        tt_seconds: np.ndarray
        if "tt" in cycle_df.columns:
            series = cycle_df["tt"]
            try:
                if np.issubdtype(series.dtype, np.timedelta64):
                    tt_seconds = series.dt.total_seconds().astype(float).to_numpy()
                else:
                    tt_seconds = pd.to_numeric(series, errors="coerce").astype(float).to_numpy()
            except Exception:
                tt_seconds = pd.to_numeric(series, errors="coerce").astype(float).to_numpy()
            tt_seconds = tt_seconds[~np.isnan(tt_seconds)]
            if tt_seconds.size == 0:
                tt_seconds = np.arange(len(cycle_df), dtype=float)
        else:
            tt_seconds = np.arange(len(cycle_df), dtype=float)

        start_s = float(tt_seconds[0])
        end_s = float(tt_seconds[-1]) if len(tt_seconds) else start_s
        rel_tt = tt_seconds - start_s
        dur = float(end_s - start_s) if len(tt_seconds) else 0.0

        # Prefer filtered channels
        ch_names = [c for c in ["f_ch1", "f_ch2", "f_ch3", "f_ch4"] if c in cycle_df.columns]
        if not ch_names:
            ch_names = [c for c in ["ch1", "ch2", "ch3", "ch4"] if c in cycle_df.columns]

        auc_total = 0.0
        for c in ch_names:
            y = np.abs(pd.to_numeric(cycle_df[c], errors="coerce").to_numpy())
            try:
                auc_total += np.trapezoid(y, rel_tt)
            except AttributeError:
                auc_total += np.trapz(y, rel_tt)

        return start_s, end_s, float(dur), float(auc_total)

    # Build per-cycle details and aggregates for clean cycles
    details: list[MovementCycleRecord] = []
    durations: list[float] = []
    aucs: list[float] = []
    
    # Extract context information from upstream records and create embedded metadata
    audio_metadata = audio_record._metadata if audio_record else None
    biomech_metadata = biomech_record._metadata if biomech_record else None
    sync_metadata = sync_record._metadata if sync_record else None
    
    # We'll create cycles_metadata after calculating aggregates
    # For now, keep it as None for individual cycles

    def _append_cycles(cycles: List[pd.DataFrame], is_outlier: bool) -> None:
        for idx, cdf in enumerate(cycles):
            s, e, d, auc = _cycle_metrics(cdf)
            
            # Calculate channel RMS values (data-derived)
            ch_rms = {}
            for n in [1, 2, 3, 4]:
                col = f"f_ch{n}" if f"f_ch{n}" in cdf.columns else (f"ch{n}" if f"ch{n}" in cdf.columns else None)
                if col:
                    arr = pd.to_numeric(cdf[col], errors="coerce").to_numpy()
                    ch_rms[n] = float(np.sqrt(np.nanmean(arr ** 2)))
            
            # Create Pydantic metadata model with embedded upstream metadata
            # Note: cycles_metadata will be None here, filled in after aggregates
            cycle_metadata = MovementCycleMetadata(
                cycle_index=idx,
                is_outlier=is_outlier,
                start_time_s=s,
                end_time_s=e,
                duration_s=d,
                acoustic_auc=auc,
                # Embed upstream metadata models
                audio_metadata=audio_metadata,
                biomech_metadata=biomech_metadata,
                sync_metadata=sync_metadata,
                cycles_metadata=None,  # Will be set after aggregate calculation
            )
            
            # Create record from validated metadata
            cycle_record = MovementCycleRecord.from_metadata(
                cycle_metadata,
                ch1_rms=ch_rms.get(1),
                ch2_rms=ch_rms.get(2),
                ch3_rms=ch_rms.get(3),
                ch4_rms=ch_rms.get(4),
            )
            
            details.append(cycle_record)
            if not is_outlier:
                durations.append(d)
                aucs.append(auc)
                
    # Build details from .pkl files if present; otherwise from provided cycles
    if pkl_cycles:
        for pidx, p in enumerate(sorted(pkl_cycles)):
            try:
                cdf = pd.read_pickle(p)
            except Exception:
                continue
            s, e, d, auc = _cycle_metrics(cdf)
            is_outlier = "outlier" in p.name.lower()
            
            # Calculate channel RMS values (data-derived)
            ch_rms = {}
            for n in [1, 2, 3, 4]:
                col = f"f_ch{n}" if f"f_ch{n}" in cdf.columns else (f"ch{n}" if f"ch{n}" in cdf.columns else None)
                if col:
                    arr = pd.to_numeric(cdf[col], errors="coerce").to_numpy()
                    ch_rms[n] = float(np.sqrt(np.nanmean(arr ** 2)))
            
            # Create Pydantic metadata model with embedded upstream metadata
            # Note: cycles_metadata will be None here, filled in after aggregates
            cycle_metadata = MovementCycleMetadata(
                cycle_index=pidx,
                is_outlier=is_outlier,
                cycle_file=str(p),
                start_time_s=s,
                end_time_s=e,
                duration_s=d,
                acoustic_auc=auc,
                # Embed upstream metadata models
                audio_metadata=audio_metadata,
                biomech_metadata=biomech_metadata,
                sync_metadata=sync_metadata,
                cycles_metadata=None,  # Will be set after aggregate calculation
            )
            
            # Create record from validated metadata
            cycle_record = MovementCycleRecord.from_metadata(
                cycle_metadata,
                ch1_rms=ch_rms.get(1),
                ch2_rms=ch_rms.get(2),
                ch3_rms=ch_rms.get(3),
                ch4_rms=ch_rms.get(4),
            )
            
            details.append(cycle_record)
            if not is_outlier:
                durations.append(d)
                aucs.append(auc)
    else:
        _append_cycles(clean_cycles, is_outlier=False)
        _append_cycles(outlier_cycles, is_outlier=True)

    # Calculate aggregate statistics
    if durations:
        data["mean_cycle_duration_s"] = float(np.nanmean(durations))
        data["median_cycle_duration_s"] = float(np.nanmedian(durations))
        data["min_cycle_duration_s"] = float(np.nanmin(durations))
        data["max_cycle_duration_s"] = float(np.nanmax(durations))
    if aucs:
        data["mean_acoustic_auc"] = float(np.nanmean(aucs))

    # Validate using Pydantic model and create record
    validated = MovementCyclesMetadata(**data)
    
    # Now update all cycle records with the complete cycles_metadata
    for detail in details:
        detail._metadata.cycles_metadata = validated
    
    record = MovementCyclesRecord.from_metadata(validated, per_cycle_details=details)
    # Set data-derived fields directly (not in metadata)
    record.output_directory = str(output_dir) if output_dir else None
    record.plots_created = plots_created
    return record
