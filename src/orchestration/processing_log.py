"""Processing log module for tracking QC, statistics, and metadata.

This module provides a comprehensive logging system for participant processing,
capturing information about audio conversion, biomechanics import, synchronization,
and movement cycle extraction. Logs are saved as Excel files (.xlsx) in the
maneuver directory and can be incrementally updated when re-processing.

This module uses Pydantic dataclasses from src.metadata for validation and Excel export.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from src.metadata import (
    AudioProcessing,
    BiomechanicsImport,
    MovementCycle,
    Synchronization,
)
from src.qc_versions import (
    get_audio_qc_version,
    get_biomech_qc_version,
    get_cycle_qc_version,
)

if TYPE_CHECKING:
    from src.models import BiomechanicsRecording

logger = logging.getLogger(__name__)


def _infer_num_channels_from_df(df: pd.DataFrame) -> int:
    """Infer number of audio channels from a DataFrame by counting ch columns."""
    if df is None or df.empty:
        return 4  # Default to 4 channels

    # Count columns that match channel pattern (ch1, ch2, ch3, ch4)
    channel_cols = [col for col in df.columns if isinstance(col, str) and col.lower().startswith('ch') and col[2:].isdigit()]
    if channel_cols:
        return len(channel_cols)

    return 4  # Default to 4 channels if not found


def _get_sync_method_defaults(row: Any) -> tuple[str, str]:
    """Extract sync method and consensus methods with proper defaults.

    Returns:
        Tuple of (sync_method, consensus_methods)
    """
    sync_method = str(row.get("Sync Method", "consensus")).lower()
    consensus_methods = str(row.get("Consensus Methods")) if pd.notna(row.get("Consensus Methods")) else None

    # Set default consensus_methods if sync_method is consensus but consensus_methods is None
    if sync_method == "consensus" and consensus_methods is None:
        consensus_methods = "consensus"

    return sync_method, consensus_methods


def _get_audio_processing_qc_defaults() -> Dict[str, Any]:
    """Return all AudioProcessing QC fields with default values.

    These fields are required by MovementCycle (which inherits from AudioProcessing)
    but not present in Synchronization records.
    """
    return {
        # Cycle-level QC flags (supplied during MovementCycle creation)
        'biomechanics_qc_fail': False,
        'sync_qc_fail': False,
        # QC type fields
        'qc_artifact_type': None,
        'qc_artifact_type_ch1': None,
        'qc_artifact_type_ch2': None,
        'qc_artifact_type_ch3': None,
        'qc_artifact_type_ch4': None,
        # QC fail segment fields
        'qc_fail_segments': [],
        'qc_fail_segments_ch1': [],
        'qc_fail_segments_ch2': [],
        'qc_fail_segments_ch3': [],
        'qc_fail_segments_ch4': [],
        # QC signal dropout fields
        'qc_signal_dropout': False,
        'qc_signal_dropout_segments': [],
        'qc_signal_dropout_ch1': False,
        'qc_signal_dropout_segments_ch1': [],
        'qc_signal_dropout_ch2': False,
        'qc_signal_dropout_segments_ch2': [],
        'qc_signal_dropout_ch3': False,
        'qc_signal_dropout_segments_ch3': [],
        'qc_signal_dropout_ch4': False,
        'qc_signal_dropout_segments_ch4': [],
        # QC artifact fields
        'qc_artifact': False,
        'qc_artifact_segments': [],
        'qc_artifact_ch1': False,
        'qc_artifact_segments_ch1': [],
        'qc_artifact_ch2': False,
        'qc_artifact_segments_ch2': [],
        'qc_artifact_ch3': False,
        'qc_artifact_segments_ch3': [],
        'qc_artifact_ch4': False,
        'qc_artifact_segments_ch4': [],
    }


def _infer_biomechanics_type_from_study(study: Optional[str]) -> str:
    """Infer biomechanics system type from study name.

    Provides a small mapping for known studies; defaults to 'Gonio'.
    """
    if study is None:
        return "Gonio"
    s = str(study).strip().lower()
    mapping = {
        "aoa": "Motion Analysis",
    }
    return mapping.get(s, "Gonio")


def _default_biomechanics_sync_method(biomechanics_type: Optional[str]) -> str:
    """Return the default biomechanics sync method for a given biomechanics_type.

    - Motion Analysis or IMU -> 'stomp'
    - Gonio -> 'flick'
    - Unknown -> 'flick'
    """
    if biomechanics_type is None:
        return "flick"
    bt = str(biomechanics_type).strip().lower()
    if bt in ("motion analysis", "imu"):
        return "stomp"
    return "flick"

def _normalize_maneuver_code(maneuver: Optional[str]) -> Optional[str]:
    """Map human-readable maneuver names to internal short codes.

    Accepts legacy names like "sit_to_stand" and "flexion_extension" and
    converts them to the literals expected by metadata models ("sts", "fe", "walk").
    Returns None when input is None.
    """
    if maneuver is None:
        return None

    # Handle NaN/empty strings coming from Excel
    try:
        if pd.isna(maneuver):
            return None
    except Exception:
        pass

    m = str(maneuver).strip().lower()
    if m == "":
        return None

    mapping = {
        "sit_to_stand": "sts",
        "sts": "sts",
        "flexion_extension": "fe",
        "fe": "fe",
        "walk": "walk",
    }
    return mapping.get(m, m)


@dataclass
class ManeuverProcessingLog:
    """Complete processing log for a knee/maneuver combination."""

    # Identification
    study_id: str
    knee_side: Literal["Left", "Right"]
    maneuver: Literal["walk", "sit_to_stand", "flexion_extension"]
    maneuver_directory: Path

    # Processing stages
    audio_record: Optional[AudioProcessing] = None
    biomechanics_record: Optional[BiomechanicsImport] = None
    synchronization_records: List[Synchronization] = field(default_factory=list)
    movement_cycles_records: List[Synchronization] = field(default_factory=list)

    # Overall metadata
    log_created: Optional[datetime] = None
    log_updated: Optional[datetime] = None

    def update_audio_record(self, record: AudioProcessing) -> None:
        """Update audio processing record."""
        self.audio_record = record
        self.log_updated = datetime.now()

    def update_biomechanics_record(self, record: BiomechanicsImport) -> None:
        """Update biomechanics import record."""
        self.biomechanics_record = record
        self.log_updated = datetime.now()

    def add_synchronization_record(self, record: Synchronization) -> None:
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

    def add_movement_cycles_record(self, record: Synchronization) -> None:
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
        seen: dict[str, Synchronization] = {}
        for rec in self.movement_cycles_records:
            seen[_norm(rec.sync_file_name)] = rec
        self.movement_cycles_records = list(seen.values())
        self.log_updated = datetime.now()

    # ------------------------------------------------------------------
    # Reusable summary helpers
    # ------------------------------------------------------------------
    def is_audio_processed(self) -> bool:
        """Return True when audio processing succeeded or a processed .pkl exists.

        This combines in-memory status with an on-disk fallback so downstream
        stages (e.g., starting from sync) still report processed audio.
        """
        processed_in_memory = bool(
            self.audio_record and self.audio_record.processing_status == "success"
        )
        return processed_in_memory or self._check_processed_audio_exists()

    def is_biomechanics_imported(self) -> bool:
        """Return True when biomechanics import completed successfully."""
        return bool(
            self.biomechanics_record
            and self.biomechanics_record.processing_status == "success"
        )

    def count_synced_files(self) -> int:
        """Count synchronization records currently attached to this log."""
        return len(self.synchronization_records)

    def count_movement_cycles(self) -> int:
        """Sum total cycles across all movement cycle records."""
        return sum(
            getattr(rec, "total_cycles_extracted", 0) or 0
            for rec in self.movement_cycles_records
        )

    def build_summary_row(self) -> Dict[str, Any]:
        """Build a single summary row for Excel export and reuse elsewhere."""
        return {
            "Study ID": self.study_id,
            "Knee Side": self.knee_side,
            "Maneuver": self.maneuver,
            "Maneuver Directory": str(self.maneuver_directory),
            "Log Created": self.log_created,
            "Log Updated": self.log_updated,
            "Audio Processed": self.is_audio_processed(),
            "Biomechanics Imported": self.is_biomechanics_imported(),
            "Num Synced Files": self.count_synced_files(),
            "Num Movement Cycles": self.count_movement_cycles(),
        }

    def _check_processed_audio_exists(self) -> bool:
        """Check if processed audio .pkl file exists on disk.

        Returns True if the processed audio .pkl file exists, even if
        audio_record is not in memory (e.g., when starting from 'sync' entrypoint).
        """
        try:
            from src.audio.readers import get_audio_file_name

            # Get audio file name with frequency info
            audio_file_with_freq = get_audio_file_name(self.maneuver_directory, with_freq=True)
            if not audio_file_with_freq:
                return False

            # Get base audio file name (without _with_freq)
            audio_file = get_audio_file_name(self.maneuver_directory, with_freq=False)
            if not audio_file:
                return False

            audio_base = Path(audio_file).name
            pickle_base = Path(audio_file_with_freq).name

            # Check for .pkl file in outputs directory
            processed_audio_outputs = self.maneuver_directory / f"{audio_base}_outputs"
            pkl_file = processed_audio_outputs / f"{pickle_base}.pkl"

            return pkl_file.exists()
        except Exception:
            return False

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
        sheets["Summary"] = pd.DataFrame([self.build_summary_row()])

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

        # Regenerate per-cycle details from disk for all movement cycle records
        # This ensures all passes are included even if per_cycle_details were not
        # persisted in memory across save/load cycles
        details_rows: list[Dict[str, Any]] = []
        for rec in self.movement_cycles_records:
            # Try to regenerate from disk if output_directory exists
            # Note: output_directory was removed during metadata refactoring (#55)
            # Keep this code for backward compatibility with old logs
            if hasattr(rec, 'output_directory') and rec.output_directory and Path(rec.output_directory).exists():
                try:
                    output_dir = Path(rec.output_directory)
                    all_pkls = sorted(output_dir.rglob("*.pkl"))
                    cycle_pkls = [p for p in all_pkls if p.name.startswith(f"{rec.sync_file_name}_")]

                    for pkl_path in cycle_pkls:
                        try:
                            cycle_df = pd.read_pickle(pkl_path)
                            is_outlier = "outlier" in pkl_path.name.lower()

                            # Extract cycle metrics
                            tt_exists = "tt" in cycle_df.columns
                            if tt_exists:
                                tt_seconds = pd.to_numeric(cycle_df["tt"], errors="coerce").to_numpy()
                                tt_seconds = tt_seconds[~np.isnan(tt_seconds)]
                                if tt_seconds.size > 0:
                                    start_s = float(tt_seconds[0])
                                    end_s = float(tt_seconds[-1])
                                    duration_s = float(end_s - start_s)
                                else:
                                    start_s = end_s = duration_s = 0.0
                                    tt_exists = False
                            else:
                                start_s = end_s = duration_s = 0.0

                            # Calculate AUC (only if we have valid time data)
                            auc_total = 0.0
                            if tt_exists:
                                rel_tt = tt_seconds - start_s
                                ch_names = [c for c in ["f_ch1", "f_ch2", "f_ch3", "f_ch4"] if c in cycle_df.columns]
                                if not ch_names:
                                    ch_names = [c for c in ["ch1", "ch2", "ch3", "ch4"] if c in cycle_df.columns]
                                for c in ch_names:
                                    y = np.abs(pd.to_numeric(cycle_df[c], errors="coerce").to_numpy())
                                    try:
                                        auc_total += np.trapezoid(y, rel_tt)
                                    except AttributeError:
                                        with np.errstate(all='ignore'):
                                            auc_total += np.trapz(y, rel_tt)

                            # Calculate channel RMS
                            ch_rms = {}
                            for n in [1, 2, 3, 4]:
                                col = f"f_ch{n}" if f"f_ch{n}" in cycle_df.columns else (f"ch{n}" if f"ch{n}" in cycle_df.columns else None)
                                if col:
                                    arr = pd.to_numeric(cycle_df[col], errors="coerce").to_numpy()
                                    ch_rms[n] = float(np.sqrt(np.nanmean(arr ** 2)))

                            # Build row with context
                            row = {
                                "Study ID": self.study_id,
                                "Knee Side": self.knee_side,
                                "Maneuver": self.maneuver,
                                "Sync File": rec.sync_file_name,
                                "Pass Number": rec.pass_number,
                                "Speed": rec.speed,
                                "Cycle File": str(pkl_path),
                                "Is Outlier": is_outlier,
                                "Start (s)": start_s,
                                "End (s)": end_s,
                                "Duration (s)": duration_s,
                                "Acoustic AUC": auc_total,
                                "Ch1 RMS": ch_rms.get(1),
                                "Ch2 RMS": ch_rms.get(2),
                                "Ch3 RMS": ch_rms.get(3),
                                "Ch4 RMS": ch_rms.get(4),
                            }
                            details_rows.append(row)
                        except Exception as e:
                            logger.warning(f"Failed to load cycle from {pkl_path}: {e}")
                            continue
                except Exception as e:
                    logger.warning(f"Failed to regenerate cycles for {rec.sync_file_name}: {e}")
            else:
                # Fallback to in-memory per_cycle_details if available
                for cycle_record in getattr(rec, "per_cycle_details", []) or []:
                    row = cycle_record.to_dict()
                    row["Study ID"] = self.study_id
                    row["Knee Side"] = self.knee_side
                    row["Maneuver"] = self.maneuver
                    details_rows.append(row)

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

                    # Check if biomechanics fields are actually present
                    # If linked_biomechanics is True but required fields are None, set it to False
                    raw_linked = bool(row.get("Linked Biomechanics", False))
                    biomech_type = str(row.get("Biomechanics Type")) if pd.notna(row.get("Biomechanics Type")) else None
                    biomech_file = str(row.get("Biomechanics File")) if pd.notna(row.get("Biomechanics File")) else None

                    # If marked as linked but missing biomechanics_type or biomechanics_file, treat as not linked
                    # Note: biomechanics_sample_rate may be None if the recording doesn't have that attribute
                    if raw_linked and (biomech_type is None or biomech_file is None):
                        linked_biomechanics = False
                    else:
                        linked_biomechanics = raw_linked

                    # When not linked, all biomechanics fields must be None
                    if not linked_biomechanics:
                        biomechanics_file = None
                        biomechanics_type = None
                        biomechanics_sync_method = None
                        biomechanics_sample_rate = None
                        biomechanics_notes = None
                    else:
                        biomechanics_file = biomech_file
                        biomechanics_type = biomech_type
                        biomechanics_sync_method = str(row.get("Biomechanics Sync Method")) if pd.notna(row.get("Biomechanics Sync Method")) else None
                        biomechanics_sample_rate = float(row.get("Biomechanics Sample Rate (Hz)")) if pd.notna(row.get("Biomechanics Sample Rate (Hz)")) else None
                        biomechanics_notes = str(row.get("Biomechanics Notes")) if pd.notna(row.get("Biomechanics Notes")) else None

                    log.audio_record = AudioProcessing(
                        # StudyMetadata
                        study=row.get("Study", "AOA"),
                        study_id=int(row.get("Study ID", 1)),
                        # BiomechanicsMetadata
                        linked_biomechanics=linked_biomechanics,
                        biomechanics_file=biomechanics_file,
                        biomechanics_type=biomechanics_type,
                        biomechanics_sync_method=biomechanics_sync_method,
                        biomechanics_sample_rate=biomechanics_sample_rate,
                        biomechanics_notes=biomechanics_notes,
                        # AcousticsFile
                        audio_file_name=row.get("Audio File", ""),
                        device_serial=str(row.get("Device Serial", "")),
                        firmware_version=int(row.get("Firmware Version", 1)),
                        file_time=pd.to_datetime(row.get("File Time")) if pd.notna(row.get("File Time")) else datetime.now(),
                        file_size_mb=float(row.get("File Size (MB)", 0.0)),
                        recording_date=pd.to_datetime(row.get("Recording Date")) if pd.notna(row.get("Recording Date")) else datetime.now(),
                        recording_time=pd.to_datetime(row.get("Recording Time")) if pd.notna(row.get("Recording Time")) else datetime.now(),
                        knee=str(row.get("Knee", "left")).lower(),
                        maneuver=_normalize_maneuver_code(str(row.get("Maneuver", "walk"))),
                        sample_rate=float(row.get("Sample Rate (Hz)", 46875.0)),
                        num_channels=int(row.get("Channels", 4)),
                        mic_1_position=str(row.get("Mic 1 Position", "IPM")),
                        mic_2_position=str(row.get("Mic 2 Position", "IPL")),
                        mic_3_position=str(row.get("Mic 3 Position", "SPM")),
                        mic_4_position=str(row.get("Mic 4 Position", "SPL")),
                        mic_1_notes=str(row.get("Mic 1 Notes")) if pd.notna(row.get("Mic 1 Notes")) else None,
                        mic_2_notes=str(row.get("Mic 2 Notes")) if pd.notna(row.get("Mic 2 Notes")) else None,
                        mic_3_notes=str(row.get("Mic 3 Notes")) if pd.notna(row.get("Mic 3 Notes")) else None,
                        mic_4_notes=str(row.get("Mic 4 Notes")) if pd.notna(row.get("Mic 4 Notes")) else None,
                        notes=str(row.get("Notes")) if pd.notna(row.get("Notes")) else None,
                        # AudioProcessing
                        processing_date=pd.to_datetime(row.get("Processing Date")) if pd.notna(row.get("Processing Date")) else datetime.now(),
                        processing_status=row.get("Status", "not_processed"),
                        error_message=str(row.get("Error")) if pd.notna(row.get("Error")) else None,
                        duration_seconds=float(row.get("Duration (s)")) if pd.notna(row.get("Duration (s)")) else None,
                        # QC fields
                        qc_fail_segments=[],
                        qc_fail_segments_ch1=[],
                        qc_fail_segments_ch2=[],
                        qc_fail_segments_ch3=[],
                        qc_fail_segments_ch4=[],
                        qc_signal_dropout=False,
                        qc_signal_dropout_segments=[],
                        qc_signal_dropout_ch1=False,
                        qc_signal_dropout_segments_ch1=[],
                        qc_signal_dropout_ch2=False,
                        qc_signal_dropout_segments_ch2=[],
                        qc_signal_dropout_ch3=False,
                        qc_signal_dropout_segments_ch3=[],
                        qc_signal_dropout_ch4=False,
                        qc_signal_dropout_segments_ch4=[],
                        qc_artifact=False,
                        qc_artifact_segments=[],
                        qc_artifact_ch1=False,
                        qc_artifact_segments_ch1=[],
                        qc_artifact_ch2=False,
                        qc_artifact_segments_ch2=[],
                        qc_artifact_ch3=False,
                        qc_artifact_segments_ch3=[],
                        qc_artifact_ch4=False,
                        qc_artifact_segments_ch4=[],
                        # Log tracking
                        log_updated=pd.to_datetime(row.get("Log Updated")) if pd.notna(row.get("Log Updated")) else None,
                    )
            except Exception as e:
                logger.warning(f"Could not load audio record: {e}")

            # Load biomechanics record if present
            try:
                bio_df = pd.read_excel(filepath, sheet_name="Biomechanics")
                if len(bio_df) > 0 and "Biomechanics File" in bio_df.columns:
                    row = bio_df.iloc[0]
                    log.biomechanics_record = BiomechanicsImport(
                        # StudyMetadata
                        study=row.get("Study", "AOA"),
                        study_id=int(row.get("Study ID", 1)),
                        # BiomechanicsImport
                        biomechanics_file=row.get("Biomechanics File", ""),
                        sheet_name=str(row.get("Sheet Name")) if pd.notna(row.get("Sheet Name")) else None,
                        processing_date=pd.to_datetime(row.get("Processing Date")) if pd.notna(row.get("Processing Date")) else datetime.now(),
                        processing_status=row.get("Status", "not_processed"),
                        error_message=str(row.get("Error")) if pd.notna(row.get("Error")) else None,
                        maneuver=_normalize_maneuver_code(row.get("Maneuver")),
                        num_sub_recordings=int(row.get("Num Sub-Recordings", row.get("Num Recordings", 1))),
                        num_passes=int(row.get("Num Passes", 1)),
                        duration_seconds=float(row.get("Duration (s)")) if pd.notna(row.get("Duration (s)")) else None,
                        sample_rate=float(row.get("Sample Rate (Hz)")) if pd.notna(row.get("Sample Rate (Hz)")) else None,
                        num_data_points=int(row.get("Num Data Points", 0)) if pd.notna(row.get("Num Data Points")) else None,
                    )
            except Exception as e:
                logger.warning(f"Could not load biomechanics record: {e}")

            # Load synchronization records if present
            try:
                sync_df = pd.read_excel(filepath, sheet_name="Synchronization")
                if len(sync_df) > 0 and "Sync File" in sync_df.columns:
                    for _, row in sync_df.iterrows():
                        # Helper function to convert seconds to timedelta
                        def to_timedelta(value):
                            if pd.isna(value):
                                return None
                            return pd.Timedelta(seconds=float(value))

                        try:
                            # Get biomechanics metadata, with defaults if not provided
                            biomechanics_file = str(row.get("Biomechanics File")) if pd.notna(row.get("Biomechanics File")) else "unknown.xlsx"
                            biomechanics_type = str(row.get("Biomechanics Type")) if pd.notna(row.get("Biomechanics Type")) else _infer_biomechanics_type_from_study(row.get("Study", "AOA"))
                            biomechanics_sync_method = (
                                str(row.get("Biomechanics Sync Method"))
                                if pd.notna(row.get("Biomechanics Sync Method"))
                                else _default_biomechanics_sync_method(biomechanics_type)
                            )
                            biomechanics_sample_rate = float(row.get("Biomechanics Sample Rate (Hz)")) if pd.notna(row.get("Biomechanics Sample Rate (Hz)")) else 100.0
                            sync_method = str(row.get("Sync Method", "consensus")).lower()
                            consensus_methods = str(row.get("Consensus Methods")) if pd.notna(row.get("Consensus Methods")) else ("consensus" if sync_method == "consensus" else None)

                            record = Synchronization(
                                # StudyMetadata
                                study=row.get("Study", "AOA"),
                                study_id=int(row.get("Study ID", 1)),
                                # BiomechanicsMetadata
                                linked_biomechanics=True,
                                biomechanics_file=biomechanics_file,
                                biomechanics_type=biomechanics_type,
                                biomechanics_sync_method=biomechanics_sync_method,
                                biomechanics_sample_rate=biomechanics_sample_rate,
                                biomechanics_notes=str(row.get("Biomechanics Notes")) if pd.notna(row.get("Biomechanics Notes")) else None,
                                # AcousticsFile
                                audio_file_name=str(row.get("Audio File", "")),
                                device_serial=str(row.get("Device Serial", "")),
                                firmware_version=int(row.get("Firmware Version", 1)),
                                file_time=pd.to_datetime(row.get("File Time")) if pd.notna(row.get("File Time")) else datetime.now(),
                                file_size_mb=float(row.get("File Size (MB)", 0.0)),
                                recording_date=pd.to_datetime(row.get("Recording Date")) if pd.notna(row.get("Recording Date")) else datetime.now(),
                                recording_time=pd.to_datetime(row.get("Recording Time")) if pd.notna(row.get("Recording Time")) else datetime.now(),
                                knee=str(row.get("Knee", "left")).lower(),
                                maneuver=_normalize_maneuver_code(str(row.get("Maneuver", "walk"))),
                                sample_rate=float(row.get("Sample Rate (Hz)", 46875.0)),
                                num_channels=int(row.get("Channels", 4)),
                                mic_1_position=str(row.get("Mic 1 Position", "IPM")),
                                mic_2_position=str(row.get("Mic 2 Position", "IPL")),
                                mic_3_position=str(row.get("Mic 3 Position", "SPM")),
                                mic_4_position=str(row.get("Mic 4 Position", "SPL")),
                                mic_1_notes=str(row.get("Mic 1 Notes")) if pd.notna(row.get("Mic 1 Notes")) else None,
                                mic_2_notes=str(row.get("Mic 2 Notes")) if pd.notna(row.get("Mic 2 Notes")) else None,
                                mic_3_notes=str(row.get("Mic 3 Notes")) if pd.notna(row.get("Mic 3 Notes")) else None,
                                mic_4_notes=str(row.get("Mic 4 Notes")) if pd.notna(row.get("Mic 4 Notes")) else None,
                                notes=str(row.get("Notes")) if pd.notna(row.get("Notes")) else None,
                                # SynchronizationMetadata
                                audio_sync_time=to_timedelta(row.get("Audio Sync Time")) or pd.Timedelta(0),
                                bio_left_sync_time=to_timedelta(row.get("Bio Left Sync Time")),
                                bio_right_sync_time=to_timedelta(row.get("Bio Right Sync Time")),
                                audio_visual_sync_time=to_timedelta(row.get("Audio Visual Sync Time")),
                                audio_visual_sync_time_contralateral=to_timedelta(row.get("Audio Visual Sync Time Contralateral")),
                                sync_offset=to_timedelta(row.get("Sync Offset")) or pd.Timedelta(0),
                                aligned_audio_sync_time=to_timedelta(row.get("Aligned Audio Sync Time")) or pd.Timedelta(0),
                                aligned_bio_sync_time=to_timedelta(row.get("Aligned Bio Sync Time")) or pd.Timedelta(0),
                                sync_method=sync_method,
                                consensus_methods=consensus_methods,
                                consensus_time=to_timedelta(row.get("Consensus Time")) or pd.Timedelta(0),
                                rms_time=to_timedelta(row.get("RMS Time")) or pd.Timedelta(0),
                                onset_time=to_timedelta(row.get("Onset Time")) or pd.Timedelta(0),
                                freq_time=to_timedelta(row.get("Freq Time")) or pd.Timedelta(0),
                                biomechanics_time=to_timedelta(row.get("Biomechanics Time")),
                                biomechanics_time_contralateral=to_timedelta(row.get("Biomechanics Time Contralateral")),
                                # Synchronization
                                sync_file_name=row.get("Sync File", ""),
                                pass_number=int(row.get("Pass Number")) if pd.notna(row.get("Pass Number")) else None,
                                speed=str(row.get("Speed")) if pd.notna(row.get("Speed")) else None,
                                processing_date=pd.to_datetime(row.get("Processing Date")) if pd.notna(row.get("Processing Date")) else datetime.now(),
                                processing_status=row.get("Status", "not_processed"),
                                error_message=str(row.get("Error")) if pd.notna(row.get("Error")) else None,
                                sync_duration=to_timedelta(row.get("Sync Duration")) or pd.Timedelta(0),
                                total_cycles_extracted=int(row.get("Total Cycles", 0)),
                                clean_cycles=int(row.get("Clean Cycles", 0)),
                                outlier_cycles=int(row.get("Outlier Cycles", 0)),
                                qc_acoustic_threshold=float(row.get("Acoustic Threshold")) if pd.notna(row.get("Acoustic Threshold")) else None,
                                mean_cycle_duration_s=float(row.get("Mean Duration (s)")) if pd.notna(row.get("Mean Duration (s)")) else None,
                                median_cycle_duration_s=float(row.get("Median Duration (s)")) if pd.notna(row.get("Median Duration (s)")) else None,
                                min_cycle_duration_s=float(row.get("Min Duration (s)")) if pd.notna(row.get("Min Duration (s)")) else None,
                                max_cycle_duration_s=float(row.get("Max Duration (s)")) if pd.notna(row.get("Max Duration (s)")) else None,
                                mean_acoustic_auc=float(row.get("Mean Acoustic AUC")) if pd.notna(row.get("Mean Acoustic AUC")) else None,
                                method_agreement_span=float(row.get("Method Agreement Span (s)")) if pd.notna(row.get("Method Agreement Span (s)")) else None,
                                audio_stomp_method=str(row.get("Detection Method")) if pd.notna(row.get("Detection Method")) else None,
                                selected_time=float(row.get("Selected Time (s)")) if pd.notna(row.get("Selected Time (s)")) else None,
                                contra_selected_time=float(row.get("Contra Selected Time (s)")) if pd.notna(row.get("Contra Selected Time (s)")) else None,
                                audio_qc_version=int(row.get("Audio QC Version", 1)),
                                biomech_qc_version=int(row.get("Biomech QC Version", 1)),
                                cycle_qc_version=int(row.get("Cycle QC Version", 1)),
                            )

                            # If speed is None, try to infer from sync_file_name
                            if record.speed is None and record.maneuver == "walk":
                                filename_lower = record.sync_file_name.lower()
                                if "slow" in filename_lower:
                                    record.speed = "slow"
                                elif "medium" in filename_lower or "normal" in filename_lower:
                                    record.speed = "medium"
                                elif "fast" in filename_lower:
                                    record.speed = "fast"

                            log.synchronization_records.append(record)
                        except Exception as e:
                            logger.debug(f"Could not load synchronization record from row: {e}")
            except Exception as e:
                logger.warning(f"Could not load synchronization records: {e}")

            # Load movement cycles records if present - if present, also use as synchronization records
            try:
                cycles_df = pd.read_excel(filepath, sheet_name="Movement Cycles")
                if len(cycles_df) > 0:
                    # Try both old and new column name conventions
                    # Check for actual columns: "Sync File Name", "Source Sync File", or "Sync File"
                    sync_file_col = (
                        "Sync File Name" if "Sync File Name" in cycles_df.columns else
                        ("Source Sync File" if "Source Sync File" in cycles_df.columns else
                         ("Sync File" if "Sync File" in cycles_df.columns else None))
                    )

                    if sync_file_col and cycles_df[sync_file_col].notna().any():
                        for _, row in cycles_df.iterrows():
                            # Helper function to convert seconds to timedelta
                            def to_timedelta(value):
                                if pd.isna(value):
                                    return None
                                return pd.Timedelta(seconds=float(value))

                            try:
                                # Get biomechanics metadata, with defaults if not provided
                                biomechanics_file = str(row.get("Biomechanics File")) if pd.notna(row.get("Biomechanics File")) else "unknown.xlsx"
                                biomechanics_type = str(row.get("Biomechanics Type")) if pd.notna(row.get("Biomechanics Type")) else _infer_biomechanics_type_from_study(row.get("Study", "AOA"))
                                biomechanics_sync_method = (
                                    str(row.get("Biomechanics Sync Method"))
                                    if pd.notna(row.get("Biomechanics Sync Method"))
                                    else _default_biomechanics_sync_method(biomechanics_type)
                                )
                                biomechanics_sample_rate = float(row.get("Biomechanics Sample Rate (Hz)")) if pd.notna(row.get("Biomechanics Sample Rate (Hz)")) else 100.0
                                sync_method = str(row.get("Sync Method", "consensus")).lower()
                                consensus_methods = str(row.get("Consensus Methods")) if pd.notna(row.get("Consensus Methods")) else ("consensus" if sync_method == "consensus" else None)

                                record = Synchronization(
                                    # StudyMetadata
                                    study=row.get("Study", "AOA"),
                                    study_id=int(row.get("Study ID", 1)),
                                    # BiomechanicsMetadata
                                    linked_biomechanics=True,
                                    biomechanics_file=biomechanics_file,
                                    biomechanics_type=biomechanics_type,
                                    biomechanics_sync_method=biomechanics_sync_method,
                                    biomechanics_sample_rate=biomechanics_sample_rate,
                                    biomechanics_notes=str(row.get("Biomechanics Notes")) if pd.notna(row.get("Biomechanics Notes")) else None,
                                    # AcousticsFile
                                    audio_file_name=str(row.get("Audio File", "")),
                                    device_serial=str(row.get("Device Serial", "")),
                                    firmware_version=int(row.get("Firmware Version", 1)),
                                    file_time=pd.to_datetime(row.get("File Time")) if pd.notna(row.get("File Time")) else datetime.now(),
                                    file_size_mb=float(row.get("File Size (MB)", 0.0)),
                                    recording_date=pd.to_datetime(row.get("Recording Date")) if pd.notna(row.get("Recording Date")) else datetime.now(),
                                    recording_time=pd.to_datetime(row.get("Recording Time")) if pd.notna(row.get("Recording Time")) else datetime.now(),
                                    knee=str(row.get("Knee", "left")).lower(),
                                    maneuver=_normalize_maneuver_code(str(row.get("Maneuver", "walk"))),
                                    sample_rate=float(row.get("Sample Rate (Hz)", 46875.0)),
                                    num_channels=int(row.get("Channels", 4)),
                                    mic_1_position=str(row.get("Mic 1 Position", "IPM")),
                                    mic_2_position=str(row.get("Mic 2 Position", "IPL")),
                                    mic_3_position=str(row.get("Mic 3 Position", "SPM")),
                                    mic_4_position=str(row.get("Mic 4 Position", "SPL")),
                                    mic_1_notes=str(row.get("Mic 1 Notes")) if pd.notna(row.get("Mic 1 Notes")) else None,
                                    mic_2_notes=str(row.get("Mic 2 Notes")) if pd.notna(row.get("Mic 2 Notes")) else None,
                                    mic_3_notes=str(row.get("Mic 3 Notes")) if pd.notna(row.get("Mic 3 Notes")) else None,
                                    mic_4_notes=str(row.get("Mic 4 Notes")) if pd.notna(row.get("Mic 4 Notes")) else None,
                                    notes=str(row.get("Notes")) if pd.notna(row.get("Notes")) else None,
                                    # SynchronizationMetadata
                                    audio_sync_time=to_timedelta(row.get("Audio Sync Time")) or pd.Timedelta(0),
                                    bio_left_sync_time=to_timedelta(row.get("Bio Left Sync Time")),
                                    bio_right_sync_time=to_timedelta(row.get("Bio Right Sync Time")),
                                    audio_visual_sync_time=to_timedelta(row.get("Audio Visual Sync Time")),
                                    audio_visual_sync_time_contralateral=to_timedelta(row.get("Audio Visual Sync Time Contralateral")),
                                    sync_offset=to_timedelta(row.get("Sync Offset")) or pd.Timedelta(0),
                                    aligned_audio_sync_time=to_timedelta(row.get("Aligned Audio Sync Time")) or pd.Timedelta(0),
                                    aligned_bio_sync_time=to_timedelta(row.get("Aligned Bio Sync Time")) or pd.Timedelta(0),
                                    sync_method=sync_method,
                                    consensus_methods=consensus_methods,
                                    consensus_time=to_timedelta(row.get("Consensus Time")) or pd.Timedelta(0),
                                    rms_time=to_timedelta(row.get("RMS Time")) or pd.Timedelta(0),
                                    onset_time=to_timedelta(row.get("Onset Time")) or pd.Timedelta(0),
                                    freq_time=to_timedelta(row.get("Freq Time")) or pd.Timedelta(0),
                                    biomechanics_time=to_timedelta(row.get("Biomechanics Time")),
                                    biomechanics_time_contralateral=to_timedelta(row.get("Biomechanics Time Contralateral")),
                                    # Synchronization
                                    sync_file_name=row.get(sync_file_col, ""),
                                    pass_number=int(row.get("Pass Number")) if pd.notna(row.get("Pass Number")) else None,
                                    speed=str(row.get("Speed")) if pd.notna(row.get("Speed")) else None,
                                    processing_date=pd.to_datetime(row.get("Processing Date")) if pd.notna(row.get("Processing Date")) else datetime.now(),
                                    processing_status=row.get("Status", "not_processed"),
                                    error_message=str(row.get("Error")) if pd.notna(row.get("Error")) else None,
                                    sync_duration=to_timedelta(row.get("Sync Duration")) or pd.Timedelta(0),
                                    total_cycles_extracted=int(row.get("Total Cycles", 0)),
                                    clean_cycles=int(row.get("Clean Cycles", 0)),
                                    outlier_cycles=int(row.get("Outlier Cycles", 0)),
                                    qc_acoustic_threshold=float(row.get("Acoustic Threshold")) if pd.notna(row.get("Acoustic Threshold")) else None,
                                    mean_cycle_duration_s=float(row.get("Mean Duration (s)")) if pd.notna(row.get("Mean Duration (s)")) else None,
                                    median_cycle_duration_s=float(row.get("Median Duration (s)")) if pd.notna(row.get("Median Duration (s)")) else None,
                                    min_cycle_duration_s=float(row.get("Min Duration (s)")) if pd.notna(row.get("Min Duration (s)")) else None,
                                    max_cycle_duration_s=float(row.get("Max Duration (s)")) if pd.notna(row.get("Max Duration (s)")) else None,
                                    mean_acoustic_auc=float(row.get("Mean Acoustic AUC")) if pd.notna(row.get("Mean Acoustic AUC")) else None,
                                    method_agreement_span=float(row.get("Method Agreement Span (s)")) if pd.notna(row.get("Method Agreement Span (s)")) else None,
                                    audio_stomp_method=str(row.get("Detection Method")) if pd.notna(row.get("Detection Method")) else None,
                                    selected_time=float(row.get("Selected Time (s)")) if pd.notna(row.get("Selected Time (s)")) else None,
                                    contra_selected_time=float(row.get("Contra Selected Time (s)")) if pd.notna(row.get("Contra Selected Time (s)")) else None,
                                    audio_qc_version=int(row.get("Audio QC Version", 1)),
                                    biomech_qc_version=int(row.get("Biomech QC Version", 1)),
                                    cycle_qc_version=int(row.get("Cycle QC Version", 1)),
                                )

                                # If speed is None, try to infer from sync_file_name
                                if record.speed is None and record.maneuver == "walk":
                                    filename_lower = record.sync_file_name.lower()
                                    if "slow" in filename_lower:
                                        record.speed = "slow"
                                    elif "medium" in filename_lower or "normal" in filename_lower:
                                        record.speed = "medium"
                                    elif "fast" in filename_lower:
                                        record.speed = "fast"

                                log.movement_cycles_records.append(record)
                            except Exception as e:
                                logger.debug(f"Could not load movement cycles record from row: {e}")
            except Exception as e:
                logger.warning(f"Could not load movement cycles records: {e}")

            # If synchronization_records are empty but movement_cycles_records exist,
            # use movement cycles records as synchronization records (they contain all the data)
            if not log.synchronization_records and log.movement_cycles_records:
                log.synchronization_records = log.movement_cycles_records.copy()

            # Note: Do NOT load Cycle Details from Excel. Cycle details are always regenerated
            # from cycle pickle files during the cycles stage to ensure proper metadata
            # (pass_number, speed, knee_side context). Loading from Excel can result in
            # orphaned rows with missing context if they were created before all metadata
            # fields were available. Cycle Details will be regenerated fresh when cycles
            # are processed and the log is re-saved.

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
            # Convert timedeltas to seconds
            audio_stomps = [
                r.audio_sync_time.total_seconds()
                for r in maneuver_log.synchronization_records
                if r.audio_sync_time is not None
            ]
            bio_stomps = []
            for r in maneuver_log.synchronization_records:
                if r.knee and r.knee.lower() == "left":
                    if r.bio_left_sync_time is not None:
                        bio_stomps.append(r.bio_left_sync_time.total_seconds())
                elif r.knee and r.knee.lower() == "right":
                    if r.bio_right_sync_time is not None:
                        bio_stomps.append(r.bio_right_sync_time.total_seconds())

            offsets = [
                r.sync_offset.total_seconds()
                for r in maneuver_log.synchronization_records
                if r.sync_offset is not None
            ]
            aligned_audio = [
                r.aligned_audio_sync_time.total_seconds()
                for r in maneuver_log.synchronization_records
                if r.aligned_audio_sync_time is not None
            ]
            aligned_bio = [
                r.aligned_bio_sync_time.total_seconds()
                for r in maneuver_log.synchronization_records
                if r.aligned_bio_sync_time is not None
            ]

            # Detection method aggregates
            consensus_times = [
                r.consensus_time.total_seconds()
                for r in maneuver_log.synchronization_records
                if r.consensus_time is not None
            ]
            rms_times = [
                r.rms_time.total_seconds()
                for r in maneuver_log.synchronization_records
                if r.rms_time is not None
            ]
            onset_times = [
                r.onset_time.total_seconds()
                for r in maneuver_log.synchronization_records
                if r.onset_time is not None
            ]
            freq_times = [
                r.freq_time.total_seconds()
                for r in maneuver_log.synchronization_records
                if r.freq_time is not None
            ]
            agreement_spans = [
                r.method_agreement_span
                for r in maneuver_log.synchronization_records
                if r.method_agreement_span is not None
            ]

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
# These functions now use Pydantic dataclasses for validation and Excel export


def create_audio_record_from_data(
    audio_file_name: str,
    audio_df: Optional[pd.DataFrame] = None,
    audio_bin_path: Optional[Path] = None,
    audio_pkl_path: Optional[Path] = None,
    metadata: Optional[Dict[str, Any]] = None,
    error: Optional[Exception] = None,
    qc_data: Optional[Dict[str, Any]] = None,
    biomechanics_type: Optional[str] = None,
) -> AudioProcessing:
    """Create an AudioProcessing record from processing data with Pydantic validation.

    Args:
        audio_file_name: Name of the audio file
        audio_df: Processed audio DataFrame
        audio_bin_path: Path to binary file
        audio_pkl_path: Path to pickle file
        metadata: Optional metadata dictionary from audio reader
        error: Exception if processing failed
        qc_data: Optional QC data dictionary containing QC results

    Returns:
        AudioProcessing instance (validated through Pydantic)

    Raises:
        ValidationError: If data doesn't meet Pydantic dataclass requirements
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

    # Initialize QC fields with default values (empty lists and False)
    # These will be populated from qc_data if available
    data.update({
        # Overall fail segments
        "qc_fail_segments": [],
        "qc_fail_segments_ch1": [],
        "qc_fail_segments_ch2": [],
        "qc_fail_segments_ch3": [],
        "qc_fail_segments_ch4": [],
        # Signal dropout QC
        "qc_signal_dropout": False,
        "qc_signal_dropout_segments": [],
        "qc_signal_dropout_ch1": False,
        "qc_signal_dropout_segments_ch1": [],
        "qc_signal_dropout_ch2": False,
        "qc_signal_dropout_segments_ch2": [],
        "qc_signal_dropout_ch3": False,
        "qc_signal_dropout_segments_ch3": [],
        "qc_signal_dropout_ch4": False,
        "qc_signal_dropout_segments_ch4": [],
        # Artifact QC
        "qc_artifact": False,
        "qc_artifact_type": None,
        "qc_artifact_segments": [],
        "qc_artifact_ch1": False,
        "qc_artifact_type_ch1": None,
        "qc_artifact_segments_ch1": [],
        "qc_artifact_ch2": False,
        "qc_artifact_type_ch2": None,
        "qc_artifact_segments_ch2": [],
        "qc_artifact_ch3": False,
        "qc_artifact_type_ch3": None,
        "qc_artifact_segments_ch3": [],
        "qc_artifact_ch4": False,
        "qc_artifact_type_ch4": None,
        "qc_artifact_segments_ch4": [],
    })

    # Extract metadata if available (needed even for error cases)
    if metadata:
        # Parent class fields (StudyMetadata, BiomechanicsMetadata, AcousticsFile)
        for field in ['study', 'study_id', 'recording_date', 'recording_time', 'knee', 'maneuver',
                      'mic_1_position', 'mic_2_position', 'mic_3_position', 'mic_4_position',
                      'file_size_mb', 'linked_biomechanics', 'biomechanics_file', 'biomechanics_type',
                      'biomechanics_sync_method', 'biomechanics_sample_rate', 'biomechanics_notes', 'num_channels']:
            if field in metadata:
                if field == "maneuver":
                    data[field] = _normalize_maneuver_code(metadata[field])
                else:
                    data[field] = metadata[field]


        # Normalize sample rate from metadata
        meta_fs = metadata.get("fs")
        if isinstance(meta_fs, (int, float)):
            data["sample_rate"] = float(meta_fs)
        # Normalize firmware version from metadata
        fv = metadata.get("devFirmwareVersion")
        if isinstance(fv, (int, float)):
            data["firmware_version"] = int(fv)
        # firmware_version can also be provided directly (takes precedence)
        if "firmware_version" in metadata:
            data["firmware_version"] = int(metadata["firmware_version"])
        # Normalize device serial: handle list/tuple from deviceSerial
        ds = metadata.get("deviceSerial")
        if isinstance(ds, (list, tuple)) and ds:
            data["device_serial"] = str(ds[0])
        elif ds is not None:
            data["device_serial"] = str(ds)
        # Device serial can also be provided directly (takes precedence)
        if "device_serial" in metadata:
            data["device_serial"] = str(metadata["device_serial"])
        # Recording time passthrough
        data["file_time"] = metadata.get("fileTime")
        if "file_time" in metadata:
            data["file_time"] = metadata["file_time"]

    # Provide defaults only for non-critical fields
    # Critical fields (recording_date, recording_time, mic positions) should come from metadata
    # but we provide minimal fallbacks for error cases and testing
    if "study" not in data:
        data["study"] = "AOA"  # Default study
    if "study_id" not in data:
        data["study_id"] = 1  # Default ID
    if "linked_biomechanics" not in data:
        data["linked_biomechanics"] = False
    # Recording date/time: prefer metadata, fallback to file_time or datetime.now() for errors/tests
    if "recording_date" not in data:
        if "file_time" in data and data["file_time"] is not None:
            data["recording_date"] = data["file_time"]
        else:
            data["recording_date"] = datetime.now()
    if "recording_time" not in data:
        if "file_time" in data and data["file_time"] is not None:
            data["recording_time"] = data["file_time"]
        else:
            data["recording_time"] = datetime.now()
    if "file_time" not in data or data["file_time"] is None:
        # file_time can default to recording_time if available
        if "recording_time" in data:
            data["file_time"] = data["recording_time"]
        else:
            data["file_time"] = datetime.now()
    if "knee" not in data:
        data["knee"] = "left"  # Default knee
    if "maneuver" not in data:
        data["maneuver"] = "walk"  # Default maneuver
    # Mic positions: prefer metadata, provide defaults as last resort for errors/tests
    if "mic_1_position" not in data:
        data["mic_1_position"] = "IPM"
    if "mic_2_position" not in data:
        data["mic_2_position"] = "IPL"
    if "mic_3_position" not in data:
        data["mic_3_position"] = "SPM"
    if "mic_4_position" not in data:
        data["mic_4_position"] = "SPL"
    if "device_serial" not in data:
        data["device_serial"] = "unknown"
    if "firmware_version" not in data:
        data["firmware_version"] = 0
    if "file_size_mb" not in data:
        data["file_size_mb"] = 0.0
    if "num_channels" not in data:
        # Try to infer from audio_df, otherwise default to 4
        if audio_df is not None:
            data["num_channels"] = _infer_num_channels_from_df(audio_df)
        else:
            data["num_channels"] = 4

    # If caller provided a biomechanics_type override, prefer it when metadata
    # explicitly indicates a linkage to biomechanics (do not set if not linked).
    if (
        biomechanics_type is not None
        and "biomechanics_type" not in data
        and data.get("linked_biomechanics", False)
    ):
        data["biomechanics_type"] = biomechanics_type


    if error:
        # Early return on error - validate and return
        return AudioProcessing(**data)

    data["processing_status"] = "success"

    # Extract QC data if available
    if qc_data:
        # Overall fail segments
        if "qc_fail_segments" in qc_data:
            data["qc_fail_segments"] = qc_data["qc_fail_segments"]
        if "qc_fail_segments_ch1" in qc_data:
            data["qc_fail_segments_ch1"] = qc_data["qc_fail_segments_ch1"]
        if "qc_fail_segments_ch2" in qc_data:
            data["qc_fail_segments_ch2"] = qc_data["qc_fail_segments_ch2"]
        if "qc_fail_segments_ch3" in qc_data:
            data["qc_fail_segments_ch3"] = qc_data["qc_fail_segments_ch3"]
        if "qc_fail_segments_ch4" in qc_data:
            data["qc_fail_segments_ch4"] = qc_data["qc_fail_segments_ch4"]

        # Signal dropout QC
        if "qc_signal_dropout" in qc_data:
            data["qc_signal_dropout"] = qc_data["qc_signal_dropout"]
        if "qc_signal_dropout_segments" in qc_data:
            data["qc_signal_dropout_segments"] = qc_data["qc_signal_dropout_segments"]
        for ch_num in range(1, 5):
            if f"qc_signal_dropout_ch{ch_num}" in qc_data:
                data[f"qc_signal_dropout_ch{ch_num}"] = qc_data[f"qc_signal_dropout_ch{ch_num}"]
            if f"qc_signal_dropout_segments_ch{ch_num}" in qc_data:
                data[f"qc_signal_dropout_segments_ch{ch_num}"] = qc_data[f"qc_signal_dropout_segments_ch{ch_num}"]

        # Artifact QC
        if "qc_artifact" in qc_data:
            data["qc_artifact"] = qc_data["qc_artifact"]
        if "qc_artifact_type" in qc_data:
            data["qc_artifact_type"] = qc_data["qc_artifact_type"]
        if "qc_artifact_segments" in qc_data:
            data["qc_artifact_segments"] = qc_data["qc_artifact_segments"]
        for ch_num in range(1, 5):
            if f"qc_artifact_ch{ch_num}" in qc_data:
                data[f"qc_artifact_ch{ch_num}"] = qc_data[f"qc_artifact_ch{ch_num}"]
            if f"qc_artifact_type_ch{ch_num}" in qc_data:
                data[f"qc_artifact_type_ch{ch_num}"] = qc_data[f"qc_artifact_type_ch{ch_num}"]
            if f"qc_artifact_segments_ch{ch_num}" in qc_data:
                data[f"qc_artifact_segments_ch{ch_num}"] = qc_data[f"qc_artifact_segments_ch{ch_num}"]

    # Calculate statistics from DataFrame
    if audio_df is not None:
        # Infer num_channels from audio data
        # Check which channel columns are present in the data
        channels_present = []
        for ch_num in range(1, 5):
            ch_name = f"ch{ch_num}"
            if ch_name in audio_df.columns:
                channels_present.append(ch_num)

        if channels_present:
            data["num_channels"] = len(channels_present)
        elif "num_channels" not in data:
            # If no channel columns found and not in metadata, raise error
            raise ValueError("Cannot determine num_channels: no channel data found in audio_df")

        # Duration (metadata field - used for QC, represents recording metadata)
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
                    # Round to nearest 100 Hz for stability
                    # (typical audio board sample rates: 46800, 46900, 47000)
                    SAMPLE_RATE_ROUNDING = 100
                    data["sample_rate"] = float(int(round(sr / SAMPLE_RATE_ROUNDING)) * SAMPLE_RATE_ROUNDING) if sr else None
            except Exception:
                pass

        # Channel statistics (pure data-derived)
        # Note: These are computed from the actual audio data and stored directly
        # in the dataclass fields rather than in a separate metadata model.
        # This design consolidates all information (metadata + derived stats) in one place.
        for ch_num in range(1, 5):
            ch_name = f"ch{ch_num}"
            if ch_name in audio_df.columns:
                ch_data = audio_df[ch_name].values
                data[f"channel_{ch_num}_rms"] = float(np.sqrt(np.mean(ch_data ** 2)))
                data[f"channel_{ch_num}_peak"] = float(np.max(np.abs(ch_data)))

        # Check for instantaneous frequency columns
        data["has_instantaneous_freq"] = any(
            col.startswith("f_ch") for col in audio_df.columns
        )

    # File size
    if audio_bin_path and audio_bin_path.exists():
        data["file_size_mb"] = audio_bin_path.stat().st_size / (1024 * 1024)

    # Create and return validated Pydantic dataclass
    return AudioProcessing(**data)


def create_biomechanics_record_from_data(
    biomechanics_file: Path,
    recordings: List['BiomechanicsRecording'],
    sheet_name: Optional[str] = None,
    maneuver: Optional[str] = None,
    error: Optional[Exception] = None,
    biomechanics_type: Optional[str] = None,
) -> BiomechanicsImport:
    """Create a BiomechanicsImport record from import data.

    Args:
        biomechanics_file: Path to biomechanics Excel file
        recordings: List of BiomechanicsRecording objects
        sheet_name: Name of the sheet that was read
        error: Exception if import failed

    Returns:
        BiomechanicsImport instance
    """
    # Build data dictionary
    data = {
        "biomechanics_file": str(biomechanics_file),
        "sheet_name": sheet_name,
        "processing_date": datetime.now(),
        "maneuver": _normalize_maneuver_code(maneuver),
        # Provide defaults for required StudyMetadata fields
        "study": "AOA",
        "study_id": 1,
    }

    # Add biomechanics_type if provided
    if biomechanics_type is not None:
        data["biomechanics_type"] = biomechanics_type

    if error:
        data["processing_status"] = "error"
        data["error_message"] = str(error)
        # For error case, need to provide default values for remaining required fields
        data["num_sub_recordings"] = 0
        data["duration_seconds"] = 0.0
        data["sample_rate"] = 100.0
        data["num_data_points"] = 0
        return BiomechanicsImport(**data)

    data["processing_status"] = "success"
    # Number of sub-recordings: usable biomechanics segments that meet quality criteria
    # For walking: passes with sufficient clean heel strikes
    # For sts/fe: typically 1 recording
    data["num_sub_recordings"] = len(recordings)

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
            data["num_data_points"] = len(df)

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

    return BiomechanicsImport(**data)


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
    # Additional context for required fields
    audio_record: Optional[AudioProcessing] = None,
    biomech_record: Optional[BiomechanicsImport] = None,
    metadata: Optional[Dict[str, Any]] = None,
    study: str = "AOA",  # Default to AOA
    study_id: int = 1,  # Default to 1 (must be positive)
    biomechanics_type: Optional[str] = None,
) -> Synchronization:
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
        detection_results: Detection results dictionary
        error: Exception if sync failed
        audio_record: Optional AudioProcessing record for metadata context
        biomech_record: Optional BiomechanicsImport record for metadata context
        metadata: Optional metadata dictionary
        study: Study name (AOA, preOA, SMoCK)
        study_id: Participant ID

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

    # Preserve any explicitly provided biomechanics_type parameter
    provided_biomechanics_type = biomechanics_type

    # Build data dictionary with all required fields from inheritance chain
    # Start with required StudyMetadata fields
    if audio_record:
        data_study = audio_record.study
        data_study_id = audio_record.study_id
    elif metadata:
        data_study = metadata.get("study", study)
        data_study_id = metadata.get("study_id", study_id)
    else:
        data_study = study
        data_study_id = study_id

    # BiomechanicsMetadata fields
    # Note: sync_method is redefined in SynchronizationMetadata, so we don't set it here
    # Note: Synchronization REQUIRES linked_biomechanics=True
    if biomech_record:
        linked_biomechanics = True
        biomechanics_file = biomech_record.biomechanics_file
        biomechanics_type = biomech_record.biomechanics_type if hasattr(biomech_record, 'biomechanics_type') else _infer_biomechanics_type_from_study(data_study)
        biomechanics_sample_rate = biomech_record.sample_rate if hasattr(biomech_record, 'sample_rate') else 100.0
    elif metadata and metadata.get("linked_biomechanics"):
        linked_biomechanics = True
        biomechanics_file = metadata.get("biomechanics_file", "unknown")
        biomechanics_type = metadata.get("biomechanics_type", _infer_biomechanics_type_from_study(data_study))
        biomechanics_sample_rate = metadata.get("biomechanics_sample_rate", 100.0)
    else:
        # Synchronization requires linked_biomechanics=True, so default to True with placeholder values
        linked_biomechanics = True
        biomechanics_file = "unknown"
        biomechanics_type = _infer_biomechanics_type_from_study(data_study)
        biomechanics_sample_rate = 100.0  # Default sample rate for tests

    # If caller provided an explicit biomechanics_type, prefer it
    if provided_biomechanics_type is not None:
        biomechanics_type = provided_biomechanics_type

    # AcousticsFile fields
    if audio_record:
        audio_file_name = audio_record.audio_file_name
        device_serial = audio_record.device_serial
        firmware_version = audio_record.firmware_version
        file_time = audio_record.file_time
        file_size_mb = audio_record.file_size_mb
        recording_date = audio_record.recording_date
        recording_time = audio_record.recording_time
        knee = audio_record.knee
        maneuver = _normalize_maneuver_code(audio_record.maneuver)
        sample_rate = audio_record.sample_rate
        num_channels = audio_record.num_channels
        mic_1_position = audio_record.mic_1_position
        mic_2_position = audio_record.mic_2_position
        mic_3_position = audio_record.mic_3_position
        mic_4_position = audio_record.mic_4_position
    elif metadata:
        audio_file_name = metadata.get("audio_file_name", sync_file_name)
        device_serial = metadata.get("device_serial", "unknown")
        firmware_version = metadata.get("firmware_version", 0)
        file_time = metadata.get("file_time", datetime.now())
        file_size_mb = metadata.get("file_size_mb", 0.0)
        recording_date = metadata.get("recording_date", datetime.now())
        recording_time = metadata.get("recording_time", datetime.now())
        knee = metadata.get("knee", knee_side if knee_side else "left")
        maneuver = _normalize_maneuver_code(metadata.get("maneuver", "walk"))
        sample_rate = metadata.get("sample_rate", 46875.0)
        num_channels = metadata.get("num_channels", _infer_num_channels_from_df(synced_df))
        mic_1_position = metadata.get("mic_1_position", "IPM")
        mic_2_position = metadata.get("mic_2_position", "IPL")
        mic_3_position = metadata.get("mic_3_position", "SPM")
        mic_4_position = metadata.get("mic_4_position", "SPL")
    else:
        # Use defaults when no context is available
        audio_file_name = sync_file_name
        device_serial = "unknown"
        firmware_version = 0
        file_time = datetime.now()
        file_size_mb = 0.0
        recording_date = datetime.now()
        recording_time = datetime.now()
        knee = knee_side if knee_side else "left"
        maneuver = "walk"
        sample_rate = 46875.0
        num_channels = _infer_num_channels_from_df(synced_df)
        mic_1_position = "IPM"
        mic_2_position = "IPL"
        mic_3_position = "SPM"
        mic_4_position = "SPL"

    # SynchronizationMetadata fields - sync times as timedeltas
    # Convert from seconds to timedelta
    def _to_timedelta(seconds: Optional[float]) -> timedelta:
        """Convert seconds to timedelta, defaulting to 0 if None."""
        return timedelta(seconds=seconds) if seconds is not None else timedelta(0)

    # If metadata provides a more specific maneuver, override now
    meta_maneuver = _normalize_maneuver_code(metadata.get("maneuver")) if metadata else None
    if meta_maneuver:
        maneuver = meta_maneuver

    audio_sync_time = _to_timedelta(audio_stomp_s)

    # bio_left/right_sync_time is required based on knee side
    # If not provided, use a default value to avoid validation error
    if knee == "left":
        bio_left_sync_time = _to_timedelta(bio_left_s) if bio_left_s is not None else timedelta(0)
        bio_right_sync_time = _to_timedelta(bio_right_s) if bio_right_s is not None else None
    elif knee == "right":
        bio_right_sync_time = _to_timedelta(bio_right_s) if bio_right_s is not None else timedelta(0)
        bio_left_sync_time = _to_timedelta(bio_left_s) if bio_left_s is not None else None
    else:
        # Default case - assume left
        bio_left_sync_time = _to_timedelta(bio_left_s) if bio_left_s is not None else timedelta(0)
        bio_right_sync_time = _to_timedelta(bio_right_s) if bio_right_s is not None else None

    sync_offset_td = _to_timedelta(stomp_offset)
    aligned_audio_sync_time = _to_timedelta(aligned_audio_stomp)
    aligned_bio_sync_time = _to_timedelta(aligned_bio_stomp)

    # Synchronization-specific fields
    # Calculate sync_duration from synced_df
    if "tt" in synced_df.columns and len(synced_df) > 0:
        duration = synced_df["tt"].iloc[-1] - synced_df["tt"].iloc[0]
        if hasattr(duration, 'total_seconds'):
            sync_duration = duration
        else:
            sync_duration = timedelta(seconds=float(duration))
    else:
        sync_duration = timedelta(0)

    # Initialize aggregate statistics with defaults (will be populated from cycles if available)
    mean_cycle_duration_s = 0.0
    median_cycle_duration_s = 0.0
    min_cycle_duration_s = 0.0
    max_cycle_duration_s = 0.0
    mean_acoustic_auc = 0.0

    # Build complete data dictionary
    data = {
        # StudyMetadata
        "study": data_study,
        "study_id": data_study_id,
        # BiomechanicsMetadata (note: sync_method is redefined in SynchronizationMetadata)
        "linked_biomechanics": linked_biomechanics,
        "biomechanics_file": biomechanics_file,
        "biomechanics_type": biomechanics_type,
        "biomechanics_sample_rate": biomechanics_sample_rate,
        # AcousticsFile
        "audio_file_name": audio_file_name,
        "device_serial": device_serial,
        "firmware_version": firmware_version,
        "file_time": file_time,
        "file_size_mb": file_size_mb,
        "recording_date": recording_date,
        "recording_time": recording_time,
        "knee": knee,
        "maneuver": maneuver,
        "sample_rate": sample_rate,
        "num_channels": num_channels,
        "mic_1_position": mic_1_position,
        "mic_2_position": mic_2_position,
        "mic_3_position": mic_3_position,
        "mic_4_position": mic_4_position,
        # SynchronizationMetadata
        "audio_sync_time": audio_sync_time,
        "bio_left_sync_time": bio_left_sync_time,
        "bio_right_sync_time": bio_right_sync_time,
        "sync_offset": sync_offset_td,
        "aligned_audio_sync_time": aligned_audio_sync_time,
        "aligned_bio_sync_time": aligned_bio_sync_time,
        "sync_method": "consensus",  # Detection method (consensus or biomechanics); will be overridden from detection_results if provided
        # Synchronization
        "sync_file_name": sync_file_name,
        "pass_number": pass_number,
        "speed": speed,
        "processing_date": datetime.now(),
        "sync_duration": sync_duration,
        "mean_cycle_duration_s": mean_cycle_duration_s,
        "median_cycle_duration_s": median_cycle_duration_s,
        "min_cycle_duration_s": min_cycle_duration_s,
        "max_cycle_duration_s": max_cycle_duration_s,
        "mean_acoustic_auc": mean_acoustic_auc,
        # Cycle extraction counts (defaults, will be populated when cycles are extracted)
        "total_cycles_extracted": 0,
        "clean_cycles": 0,
        "outlier_cycles": 0,
    }

    # Populate detection method details if provided
    # These times need to be timedeltas for SynchronizationMetadata
    # Store energy/magnitude as separate data-derived fields (not metadata)
    rms_energy = None
    onset_magnitude = None
    freq_energy = None

    if detection_results:
        try:
            consensus_time_s = _to_seconds(detection_results.get("consensus_time"))
            rms_time_s = _to_seconds(detection_results.get("rms_time"))
            onset_time_s = _to_seconds(detection_results.get("onset_time"))
            freq_time_s = _to_seconds(detection_results.get("freq_time"))

            # Convert to timedeltas for metadata fields
            data["consensus_time"] = _to_timedelta(consensus_time_s)
            data["rms_time"] = _to_timedelta(rms_time_s)
            data["onset_time"] = _to_timedelta(onset_time_s)
            data["freq_time"] = _to_timedelta(freq_time_s)

            # Extract which methods contributed to consensus
            consensus_methods_list = detection_results.get("consensus_methods", [])
            if consensus_methods_list:
                data["consensus_methods"] = ", ".join(consensus_methods_list)

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
            # Get times for methods that contributed (in seconds for calculation)
            method_times_used = []
            if consensus_methods_list:
                if "rms" in consensus_methods_list and rms_time_s is not None:
                    method_times_used.append(rms_time_s)
                if "onset" in consensus_methods_list and onset_time_s is not None:
                    method_times_used.append(onset_time_s)
                if "freq" in consensus_methods_list and freq_time_s is not None:
                    method_times_used.append(freq_time_s)

            if method_times_used:
                data["method_agreement_span"] = float(max(method_times_used) - min(method_times_used))

            # Biomechanics-guided detection metadata
            data["audio_stomp_method"] = detection_results.get("audio_stomp_method")
            data["selected_time"] = _to_seconds(detection_results.get("selected_time"))
            data["contra_selected_time"] = _to_seconds(detection_results.get("contra_selected_time"))
        except Exception as e:
            logger.debug(f"Failed to populate detection_results in sync record: {e}")
    else:
        # If no detection_results provided, set required detection times to defaults
        data["consensus_time"] = timedelta(0)
        data["rms_time"] = timedelta(0)
        data["onset_time"] = timedelta(0)
        data["freq_time"] = timedelta(0)


    if error:
        data["processing_status"] = "error"
        data["error_message"] = str(error)
        # Include data-derived fields in unified class
        data["rms_energy"] = rms_energy
        data["onset_magnitude"] = onset_magnitude
        data["freq_energy"] = freq_energy
        return Synchronization(**data)

    data["processing_status"] = "success"
    # num_synced_samples is pure data-derived, duration_seconds is metadata
    data["num_synced_samples"] = len(synced_df)

    if "tt" in synced_df.columns and len(synced_df) > 0:
        duration = synced_df["tt"].iloc[-1] - synced_df["tt"].iloc[0]
        if hasattr(duration, 'total_seconds'):
            data["duration_seconds"] = duration.total_seconds()
        else:
            data["duration_seconds"] = float(duration)

    # Include all data-derived fields in unified Synchronization class
    data["rms_energy"] = rms_energy
    data["onset_magnitude"] = onset_magnitude
    data["freq_energy"] = freq_energy

    return Synchronization(**data)


def create_cycles_record_from_data(
    sync_file_name: str,
    clean_cycles: List[pd.DataFrame],
    outlier_cycles: List[pd.DataFrame],
    output_dir: Optional[Path] = None,
    acoustic_threshold: Optional[float] = None,
    plots_created: bool = False,
    error: Optional[Exception] = None,
    # Optional context from upstream processing
    audio_record: Optional[AudioProcessing] = None,
    biomech_record: Optional[BiomechanicsImport] = None,
    sync_record: Optional[Synchronization] = None,
    metadata: Optional[Dict[str, Any]] = None,
    study: str = "AOA",
    study_id: int = 1,
    biomechanics_type: Optional[str] = None,
) -> Synchronization:
    """Create a Synchronization record from cycle extraction data.

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
        sync_record: Optional synchronization record for context (preferred)
        metadata: Optional metadata dictionary
        study: Study name (AOA, preOA, SMoCK)
        study_id: Participant ID

    Returns:
        Synchronization instance
    """

    def _infer_maneuver_from_name(name: str) -> Optional[str]:
        """Best-effort inference of maneuver from a file name/path."""
        lower = str(name).lower()
        if "sit" in lower and "stand" in lower:
            return "sts"
        if "flex" in lower:
            return "fe"
        if "walk" in lower:
            return "walk"
        return None
    def _infer_pass_number_from_name(name: str) -> Optional[int]:
        match = re.search(r"Pass(\d{2,4})", name, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None

    # If we have a sync_record, use it as the base and update cycle-specific fields
    if sync_record:
        # Convert sync_record to dict and update cycle-specific fields
        data = {
            # Copy all fields from sync_record
            **{k: getattr(sync_record, k) for k in sync_record.__dataclass_fields__.keys()},
            # Update with cycle-specific values
            "processing_date": datetime.now(),
            "qc_acoustic_threshold": acoustic_threshold,
        }
    else:
        # Build from scratch like create_sync_record_from_data
        # Get metadata from audio_record or use defaults
        if audio_record:
            data = {
                "study": audio_record.study,
                "study_id": audio_record.study_id,
                "linked_biomechanics": audio_record.linked_biomechanics,
                "biomechanics_file": audio_record.biomechanics_file,
                "biomechanics_type": audio_record.biomechanics_type if audio_record.biomechanics_type is not None else biomechanics_type,
                "biomechanics_sample_rate": audio_record.biomechanics_sample_rate,
                "audio_file_name": audio_record.audio_file_name,
                "device_serial": audio_record.device_serial,
                "firmware_version": audio_record.firmware_version,
                "file_time": audio_record.file_time,
                "file_size_mb": audio_record.file_size_mb,
                "recording_date": audio_record.recording_date,
                "recording_time": audio_record.recording_time,
                "knee": audio_record.knee,
                "maneuver": _normalize_maneuver_code(audio_record.maneuver),
                "sample_rate": audio_record.sample_rate,
                "mic_1_position": audio_record.mic_1_position,
                "mic_2_position": audio_record.mic_2_position,
                "mic_3_position": audio_record.mic_3_position,
                "mic_4_position": audio_record.mic_4_position,
            }
        else:
            # Use defaults
            chosen_biomechanics_type = biomechanics_type if biomechanics_type is not None else _infer_biomechanics_type_from_study(metadata.get("study", study) if metadata else study)
            data = {
                "study": metadata.get("study", study) if metadata else study,
                "study_id": metadata.get("study_id", study_id) if metadata else study_id,
                "linked_biomechanics": True,  # Synchronization requires this to be True
                "biomechanics_file": "unknown_biomechanics.xlsx",
                "biomechanics_type": chosen_biomechanics_type,
                "biomechanics_sync_method": _default_biomechanics_sync_method(chosen_biomechanics_type),
                "biomechanics_sample_rate": 100.0,
                "biomechanics_notes": None,
                "audio_file_name": sync_file_name,
                "device_serial": "unknown",
                "firmware_version": 0,
                "file_time": datetime.now(),
                "file_size_mb": 0.0,
                "recording_date": datetime.now(),
                "recording_time": datetime.now(),
                "knee": "left",
                "maneuver": _normalize_maneuver_code(metadata.get("maneuver")) if metadata else "walk",
                "sample_rate": 46875.0,
                "num_channels": 4,
                "mic_1_position": "IPM",
                "mic_2_position": "IPL",
                "mic_3_position": "SPM",
                "mic_4_position": "SPL",
            }

        # Add SynchronizationMetadata fields
        data.update({
            "audio_sync_time": timedelta(0),
            "bio_left_sync_time": timedelta(0),
            "bio_right_sync_time": None,
            "sync_offset": timedelta(0),
            "aligned_audio_sync_time": timedelta(0),
            "aligned_bio_sync_time": timedelta(0),
            "sync_method": "consensus",
            "consensus_time": timedelta(0),
            "rms_time": timedelta(0),
            "onset_time": timedelta(0),
            "freq_time": timedelta(0),
        })

        # Add Synchronization fields
        data.update({
            "sync_file_name": sync_file_name,
            "processing_date": datetime.now(),
            "sync_duration": timedelta(0),
            "qc_acoustic_threshold": acoustic_threshold,
        })

    # Update pass/speed/knee context
    if not sync_record:
        data["pass_number"] = _infer_pass_number_from_name(sync_file_name)
        # Extract speed from filename
        filename_lower = sync_file_name.lower()
        if "slow" in filename_lower:
            data["speed"] = "slow"
        elif "medium" in filename_lower or "normal" in filename_lower:
            data["speed"] = "medium"
        elif "fast" in filename_lower:
            data["speed"] = "fast"
        else:
            data["speed"] = None

    # Normalize maneuver and prefer provided metadata/sync context when current is walk/None
    normalized_maneuver = _normalize_maneuver_code(data.get("maneuver"))
    meta_maneuver = _normalize_maneuver_code(metadata.get("maneuver")) if metadata else None
    sync_maneuver = _normalize_maneuver_code(getattr(sync_record, "maneuver", None)) if sync_record else None

    for candidate in (meta_maneuver, sync_maneuver):
        if candidate and (normalized_maneuver is None or normalized_maneuver == "walk"):
            normalized_maneuver = candidate

    if normalized_maneuver in (None, "walk"):
        inferred = _infer_maneuver_from_name(sync_file_name)
        if inferred:
            normalized_maneuver = inferred

    if normalized_maneuver is None:
        normalized_maneuver = "walk"
    data["maneuver"] = normalized_maneuver

    if error:
        data["processing_status"] = "error"
        data["error_message"] = str(error)
        # Set defaults for required cycle stats
        data["total_cycles_extracted"] = 0
        data["clean_cycles"] = 0
        data["outlier_cycles"] = 0
        data["mean_cycle_duration_s"] = 0.0
        data["median_cycle_duration_s"] = 0.0
        data["min_cycle_duration_s"] = 0.0
        data["max_cycle_duration_s"] = 0.0
        data["mean_acoustic_auc"] = 0.0
        data["per_cycle_details"] = []
        return Synchronization(**data)

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
                with np.errstate(all='ignore'):
                    auc_total += np.trapz(y, rel_tt)

        return start_s, end_s, float(dur), float(auc_total)

    # Build per-cycle details and aggregates for clean cycles
    details: list[MovementCycle] = []
    durations: list[float] = []
    aucs: list[float] = []

    def _append_cycles(cycles: List[pd.DataFrame], is_outlier: bool) -> None:
        for idx, cdf in enumerate(cycles):
            s, e, d, auc = _cycle_metrics(cdf)

            # Calculate channel RMS values (data-derived) - Note: these are not stored in MovementCycle
            ch_rms = {}
            for n in [1, 2, 3, 4]:
                col = f"f_ch{n}" if f"f_ch{n}" in cdf.columns else (f"ch{n}" if f"ch{n}" in cdf.columns else None)
                if col:
                    arr = pd.to_numeric(cdf[col], errors="coerce").to_numpy()
                    ch_rms[n] = float(np.sqrt(np.nanmean(arr ** 2)))

            # Build cycle_record fields from sync_record if available
            cycle_fields = {}
            # Normalize maneuver context, preferring sync_record then metadata
            maneuver_ctx = None
            if sync_record:
                maneuver_ctx = getattr(sync_record, "maneuver", None)
            elif metadata:
                maneuver_ctx = metadata.get("maneuver")

            normalized_maneuver = _normalize_maneuver_code(maneuver_ctx)

            # Prefer metadata maneuver when available and more specific
            if metadata:
                meta_maneuver = _normalize_maneuver_code(metadata.get("maneuver"))
                if meta_maneuver and (normalized_maneuver is None or normalized_maneuver == "walk"):
                    normalized_maneuver = meta_maneuver

            # Heuristic: infer from file name if still unknown/walk
            if normalized_maneuver in (None, "walk"):
                inferred = _infer_maneuver_from_name(sync_file_name)
                if inferred:
                    normalized_maneuver = inferred

            if normalized_maneuver is None:
                normalized_maneuver = "walk"

            if sync_record:
                # Copy all fields from sync_record that exist in both models
                for field_name in sync_record.__dataclass_fields__.keys():
                    # Skip fields that are specific to Synchronization (not MovementCycle)
                    if field_name not in ['sync_file_name', 'sync_duration', 'total_cycles_extracted',
                                          'clean_cycles', 'outlier_cycles', 'mean_cycle_duration_s',
                                          'median_cycle_duration_s', 'min_cycle_duration_s',
                                          'max_cycle_duration_s', 'mean_acoustic_auc', 'per_cycle_details',
                                          'output_directory', 'plots_created', 'qc_acoustic_threshold',
                                          'audio_sync_time', 'bio_left_sync_time', 'bio_right_sync_time',
                                          'audio_visual_sync_time', 'audio_visual_sync_time_contralateral',
                                          'sync_offset', 'aligned_audio_sync_time', 'aligned_bio_sync_time',
                                          'sync_method', 'consensus_methods', 'consensus_time', 'rms_time',
                                          'onset_time', 'freq_time', 'biomechanics_time', 'biomechanics_time_contralateral']:
                        cycle_fields[field_name] = getattr(sync_record, field_name)

                    # Ensure maneuver uses normalized code
                    cycle_fields["maneuver"] = normalized_maneuver

                # Add missing AudioProcessing QC type fields if sync_record doesn't have them
                # (Synchronization doesn't inherit from AudioProcessing, but MovementCycle does)
                for field_name, default_value in _get_audio_processing_qc_defaults().items():
                    if field_name not in cycle_fields:
                        cycle_fields[field_name] = default_value

                # Derive timestamps
                audio_start = sync_record.recording_time + timedelta(seconds=s)
                audio_end = sync_record.recording_time + timedelta(seconds=e)
                bio_start = sync_record.recording_time + timedelta(seconds=s)
                bio_end = sync_record.recording_time + timedelta(seconds=e)
            else:
                # Minimal fallback if no sync_record
                now = datetime.now()
                audio_start = now
                audio_end = now + timedelta(seconds=d)
                bio_start = now
                bio_end = now + timedelta(seconds=d)

                # Provide minimal required fields
                cycle_fields = {
                    'study': study,
                    'study_id': study_id,
                    'linked_biomechanics': False,
                    'audio_file_name': 'unknown.bin',
                    'device_serial': 'UNKNOWN',
                    'firmware_version': 0,
                    'file_time': now,
                    'file_size_mb': 0.0,
                    'recording_date': now,
                    'recording_time': now,
                    'knee': 'left',
                    'maneuver': normalized_maneuver,
                    'num_channels': 4,
                    'mic_1_position': 'IPL',
                    'mic_2_position': 'IPM',
                    'mic_3_position': 'SPM',
                    'mic_4_position': 'SPL',
                    'processing_date': now,
                    **_get_audio_processing_qc_defaults(),
                }

            # Create MovementCycle with all required fields
            cycle_record = MovementCycle(
                **cycle_fields,
                cycle_index=idx,
                is_outlier=is_outlier,
                cycle_file=f"cycle_{idx}.pkl",
                start_time_s=s,
                end_time_s=e,
                duration_s=d,
                audio_start_time=audio_start,
                audio_end_time=audio_end,
                bio_start_time=bio_start,
                bio_end_time=bio_end,
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

            # Calculate channel RMS values (data-derived) - Note: these are not stored in MovementCycle
            ch_rms = {}
            for n in [1, 2, 3, 4]:
                col = f"f_ch{n}" if f"f_ch{n}" in cdf.columns else (f"ch{n}" if f"ch{n}" in cdf.columns else None)
                if col:
                    arr = pd.to_numeric(cdf[col], errors="coerce").to_numpy()
                    ch_rms[n] = float(np.sqrt(np.nanmean(arr ** 2)))

            # Determine maneuver context
            maneuver_ctx = None
            if sync_record:
                maneuver_ctx = getattr(sync_record, "maneuver", None)
            elif metadata:
                maneuver_ctx = metadata.get("maneuver")

            normalized_maneuver = _normalize_maneuver_code(maneuver_ctx)
            if metadata:
                meta_maneuver = _normalize_maneuver_code(metadata.get("maneuver"))
                if meta_maneuver and (normalized_maneuver is None or normalized_maneuver == "walk"):
                    normalized_maneuver = meta_maneuver
            if normalized_maneuver in (None, "walk"):
                inferred = _infer_maneuver_from_name(sync_file_name)
                if inferred:
                    normalized_maneuver = inferred
            if normalized_maneuver is None:
                normalized_maneuver = "walk"

            # Build cycle_record fields from sync_record if available
            cycle_fields = {}
            if sync_record:
                # Copy all fields from sync_record that exist in both models
                for field_name in sync_record.__dataclass_fields__.keys():
                    # Skip fields that are specific to Synchronization (not MovementCycle)
                    if field_name not in ['sync_file_name', 'sync_duration', 'total_cycles_extracted',
                                          'clean_cycles', 'outlier_cycles', 'mean_cycle_duration_s',
                                          'median_cycle_duration_s', 'min_cycle_duration_s',
                                          'max_cycle_duration_s', 'mean_acoustic_auc', 'per_cycle_details',
                                          'output_directory', 'plots_created', 'qc_acoustic_threshold',
                                          'audio_sync_time', 'bio_left_sync_time', 'bio_right_sync_time',
                                          'audio_visual_sync_time', 'audio_visual_sync_time_contralateral',
                                          'sync_offset', 'aligned_audio_sync_time', 'aligned_bio_sync_time',
                                          'sync_method', 'consensus_methods', 'consensus_time', 'rms_time',
                                          'onset_time', 'freq_time', 'biomechanics_time', 'biomechanics_time_contralateral']:
                        cycle_fields[field_name] = getattr(sync_record, field_name)

                    # Ensure maneuver uses normalized code
                    cycle_fields["maneuver"] = normalized_maneuver

                # Add missing AudioProcessing QC type fields if sync_record doesn't have them
                # (Synchronization doesn't inherit from AudioProcessing, but MovementCycle does)
                for field_name, default_value in _get_audio_processing_qc_defaults().items():
                    if field_name not in cycle_fields:
                        cycle_fields[field_name] = default_value

                # Derive timestamps
                audio_start = sync_record.recording_time + timedelta(seconds=s)
                audio_end = sync_record.recording_time + timedelta(seconds=e)
                bio_start = sync_record.recording_time + timedelta(seconds=s)
                bio_end = sync_record.recording_time + timedelta(seconds=e)
            else:
                # Minimal fallback if no sync_record
                now = datetime.now()
                audio_start = now
                audio_end = now + timedelta(seconds=d)
                bio_start = now
                bio_end = now + timedelta(seconds=d)

                # Provide minimal required fields
                cycle_fields = {
                    'study': study,
                    'study_id': study_id,
                    'linked_biomechanics': False,
                    'audio_file_name': 'unknown.bin',
                    'device_serial': 'UNKNOWN',
                    'firmware_version': 0,
                    'file_time': now,
                    'file_size_mb': 0.0,
                    'recording_date': now,
                    'recording_time': now,
                    'knee': 'left',
                    'maneuver': normalized_maneuver,
                    'num_channels': 4,
                    'mic_1_position': 'IPL',
                    'mic_2_position': 'IPM',
                    'mic_3_position': 'SPM',
                    'mic_4_position': 'SPL',
                    'processing_date': now,
                    **_get_audio_processing_qc_defaults(),
                }

            # Create MovementCycle with all required fields
            cycle_record = MovementCycle(
                **cycle_fields,
                cycle_index=pidx,
                is_outlier=is_outlier,
                cycle_file=str(p),
                start_time_s=s,
                end_time_s=e,
                duration_s=d,
                audio_start_time=audio_start,
                audio_end_time=audio_end,
                bio_start_time=bio_start,
                bio_end_time=bio_end,
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
    else:
        data["mean_cycle_duration_s"] = 0.0
        data["median_cycle_duration_s"] = 0.0
        data["min_cycle_duration_s"] = 0.0
        data["max_cycle_duration_s"] = 0.0

    if aucs:
        data["mean_acoustic_auc"] = float(np.nanmean(aucs))
    else:
        data["mean_acoustic_auc"] = 0.0

    # Include data-derived fields in unified Synchronization class
    data["output_directory"] = str(output_dir) if output_dir else None
    data["plots_created"] = plots_created
    data["per_cycle_details"] = details

    # Create unified Synchronization object
    cycles = Synchronization(**data)

    return cycles
