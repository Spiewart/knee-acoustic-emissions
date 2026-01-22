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
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from src.metadata import (
    AudioProcessing,
    BiomechanicsImport,
    MovementCycle,
    MovementCycles,
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
    movement_cycles_records: List[MovementCycles] = field(default_factory=list)

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

    def add_movement_cycles_record(self, record: MovementCycles) -> None:
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
        seen: dict[str, MovementCycles] = {}
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

        # Regenerate per-cycle details from disk for all movement cycle records
        # This ensures all passes are included even if per_cycle_details were not
        # persisted in memory across save/load cycles
        details_rows: list[Dict[str, Any]] = []
        for rec in self.movement_cycles_records:
            # Try to regenerate from disk if output_directory exists
            if rec.output_directory and Path(rec.output_directory).exists():
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
                    log.audio_record = AudioProcessing(
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
                        channel_1_rms=row.get("Ch1 RMS") if pd.notna(row.get("Ch1 RMS")) else None,
                        channel_2_rms=row.get("Ch2 RMS") if pd.notna(row.get("Ch2 RMS")) else None,
                        channel_3_rms=row.get("Ch3 RMS") if pd.notna(row.get("Ch3 RMS")) else None,
                        channel_4_rms=row.get("Ch4 RMS") if pd.notna(row.get("Ch4 RMS")) else None,
                        channel_1_peak=row.get("Ch1 Peak") if pd.notna(row.get("Ch1 Peak")) else None,
                        channel_2_peak=row.get("Ch2 Peak") if pd.notna(row.get("Ch2 Peak")) else None,
                        channel_3_peak=row.get("Ch3 Peak") if pd.notna(row.get("Ch3 Peak")) else None,
                        channel_4_peak=row.get("Ch4 Peak") if pd.notna(row.get("Ch4 Peak")) else None,
                    )
            except Exception as e:
                logger.warning(f"Could not load audio record: {e}")

            # Load biomechanics record if present
            try:
                bio_df = pd.read_excel(filepath, sheet_name="Biomechanics")
                if len(bio_df) > 0 and "Biomechanics File" in bio_df.columns:
                    row = bio_df.iloc[0]
                    log.biomechanics_record = BiomechanicsImport(
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
            except Exception as e:
                logger.warning(f"Could not load biomechanics record: {e}")

            # Load synchronization records if present
            try:
                sync_df = pd.read_excel(filepath, sheet_name="Synchronization")
                if len(sync_df) > 0 and "Sync File" in sync_df.columns:
                    for _, row in sync_df.iterrows():
                        record = Synchronization(
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
                            num_synced_samples=int(row["Num Samples"]) if pd.notna(row.get("Num Samples")) else None,
                            duration_seconds=row.get("Duration (s)") if pd.notna(row.get("Duration (s)")) else None,
                            sync_qc_performed=bool(row.get("Sync QC Done", False)),
                            sync_qc_passed=bool(row["Sync QC Passed"]) if pd.notna(row.get("Sync QC Passed")) else None,
                            audio_qc_version=int(row["Audio QC Version"]) if pd.notna(row.get("Audio QC Version")) else get_audio_qc_version(),
                            biomech_qc_version=int(row["Biomech QC Version"]) if pd.notna(row.get("Biomech QC Version")) else get_biomech_qc_version(),
                            consensus_time=row.get("Consensus (s)") if pd.notna(row.get("Consensus (s)")) else None,
                            consensus_methods=str(row.get("Consensus Methods")) if pd.notna(row.get("Consensus Methods")) else None,
                            rms_time=row.get("RMS Detect (s)") if pd.notna(row.get("RMS Detect (s)")) else None,
                            onset_time=row.get("Onset Detect (s)") if pd.notna(row.get("Onset Detect (s)")) else None,
                            freq_time=row.get("Freq Detect (s)") if pd.notna(row.get("Freq Detect (s)")) else None,
                            rms_energy=row.get("RMS Energy") if pd.notna(row.get("RMS Energy")) else None,
                            onset_magnitude=row.get("Onset Magnitude") if pd.notna(row.get("Onset Magnitude")) else None,
                            freq_energy=row.get("Freq Energy") if pd.notna(row.get("Freq Energy")) else None,
                            method_agreement_span=row.get("Method Agreement Span (s)") if pd.notna(row.get("Method Agreement Span (s)")) else None,
                            audio_stomp_method=str(row.get("Detection Method")) if pd.notna(row.get("Detection Method")) else None,
                            selected_time=row.get("Selected Time (s)") if pd.notna(row.get("Selected Time (s)")) else None,
                            contra_selected_time=row.get("Contra Selected Time (s)") if pd.notna(row.get("Contra Selected Time (s)")) else None,
                        )
                        log.synchronization_records.append(record)
            except Exception as e:
                logger.warning(f"Could not load synchronization records: {e}")

            # Load movement cycles records if present
            try:
                cycles_df = pd.read_excel(filepath, sheet_name="Movement Cycles")
                if len(cycles_df) > 0 and "Source Sync File" in cycles_df.columns:
                    for _, row in cycles_df.iterrows():
                        record = MovementCycles(
                            sync_file_name=row.get("Source Sync File", ""),
                            pass_number=int(row["Pass Number"]) if pd.notna(row.get("Pass Number")) else None,
                            speed=str(row.get("Speed")) if pd.notna(row.get("Speed")) else None,
                            knee_side=str(row.get("Knee Side")) if pd.notna(row.get("Knee Side")) else None,
                            processing_date=pd.to_datetime(row["Processing Date"]) if pd.notna(row.get("Processing Date")) else None,
                            processing_status=row.get("Status", "not_processed"),
                            error_message=str(row.get("Error")) if pd.notna(row.get("Error")) else None,
                            total_cycles_extracted=int(row.get("Total Cycles", 0)),
                            clean_cycles=int(row.get("Clean Cycles", 0)),
                            outlier_cycles=int(row.get("Outlier Cycles", 0)),
                            qc_acoustic_threshold=row.get("Acoustic Threshold") if pd.notna(row.get("Acoustic Threshold")) else None,
                            output_directory=str(row.get("Output Directory")) if pd.notna(row.get("Output Directory")) else None,
                            plots_created=bool(row.get("Plots Created", False)),
                            mean_cycle_duration_s=float(row.get("Mean Duration (s)")) if pd.notna(row.get("Mean Duration (s)")) else None,
                            median_cycle_duration_s=float(row.get("Median Duration (s)")) if pd.notna(row.get("Median Duration (s)")) else None,
                            min_cycle_duration_s=float(row.get("Min Duration (s)")) if pd.notna(row.get("Min Duration (s)")) else None,
                            max_cycle_duration_s=float(row.get("Max Duration (s)")) if pd.notna(row.get("Max Duration (s)")) else None,
                            mean_acoustic_auc=float(row.get("Mean Acoustic AUC")) if pd.notna(row.get("Mean Acoustic AUC")) else None,
                        )
                        log.movement_cycles_records.append(record)
            except Exception as e:
                logger.warning(f"Could not load movement cycles records: {e}")

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
# These functions now use Pydantic dataclasses for validation and Excel export


def create_audio_record_from_data(
    audio_file_name: str,
    audio_df: Optional[pd.DataFrame] = None,
    audio_bin_path: Optional[Path] = None,
    audio_pkl_path: Optional[Path] = None,
    metadata: Optional[Dict[str, Any]] = None,
    error: Optional[Exception] = None,
) -> AudioProcessing:
    """Create an AudioProcessing record from processing data with Pydantic validation.

    Args:
        audio_file_name: Name of the audio file
        audio_df: Processed audio DataFrame
        audio_bin_path: Path to binary file
        audio_pkl_path: Path to pickle file
        metadata: Optional metadata dictionary from audio reader
        error: Exception if processing failed

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

    if error:
        # Early return on error - validate and return
        return AudioProcessing(**data)

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
    if audio_df is not None:
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
    error: Optional[Exception] = None,
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
    }

    if error:
        data["processing_status"] = "error"
        data["error_message"] = str(error)
        return BiomechanicsImport(**data)

    data["processing_status"] = "success"
    data["num_recordings"] = len(recordings)

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
) -> MovementCycles:
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
    def _infer_pass_number_from_name(name: str) -> Optional[int]:
        match = re.search(r"Pass(\d{2,4})", name, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None

    # Build data dictionary
    data = {
        "sync_file_name": sync_file_name,
        "processing_date": datetime.now(),
        "qc_acoustic_threshold": acoustic_threshold,  # Renamed field
    }

    # Propagate pass/speed/knee context when available
    if sync_record:
        data["pass_number"] = sync_record.pass_number
        data["speed"] = sync_record.speed
        data["knee_side"] = sync_record.knee_side
    else:
        data["pass_number"] = _infer_pass_number_from_name(sync_file_name)
        data["speed"] = None
        data["knee_side"] = None

    if error:
        data["processing_status"] = "error"
        data["error_message"] = str(error)
        # Include data-derived fields in unified class
        data["output_directory"] = str(output_dir) if output_dir else None
        data["plots_created"] = plots_created
        data["per_cycle_details"] = []
        return MovementCycles(**data)

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

    # Extract context information from upstream records
    audio_metadata = audio_record if audio_record else None
    biomech_metadata = biomech_record if biomech_record else None
    sync_metadata = sync_record if sync_record else None

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

            # Create MovementCycle directly with embedded upstream metadata
            cycle_record = MovementCycle(
                cycle_index=idx,
                is_outlier=is_outlier,
                start_time_s=s,
                end_time_s=e,
                duration_s=d,
                acoustic_auc=auc,
                ch1_rms=ch_rms.get(1),
                ch2_rms=ch_rms.get(2),
                ch3_rms=ch_rms.get(3),
                ch4_rms=ch_rms.get(4),
                # Embed upstream metadata models
                audio_metadata=audio_metadata,
                biomech_metadata=biomech_metadata,
                sync_metadata=sync_metadata,
                cycles_metadata=None,  # Will be set after aggregate calculation
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

            # Create MovementCycle directly with embedded upstream metadata
            cycle_record = MovementCycle(
                cycle_index=pidx,
                is_outlier=is_outlier,
                cycle_file=str(p),
                start_time_s=s,
                end_time_s=e,
                duration_s=d,
                acoustic_auc=auc,
                ch1_rms=ch_rms.get(1),
                ch2_rms=ch_rms.get(2),
                ch3_rms=ch_rms.get(3),
                ch4_rms=ch_rms.get(4),
                # Embed upstream metadata models
                audio_metadata=audio_metadata,
                biomech_metadata=biomech_metadata,
                sync_metadata=sync_metadata,
                cycles_metadata=None,  # Will be set after aggregate calculation
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

    # Include data-derived fields in unified MovementCycles class
    data["output_directory"] = str(output_dir) if output_dir else None
    data["plots_created"] = plots_created
    data["per_cycle_details"] = details

    # Create unified MovementCycles object
    cycles = MovementCycles(**data)

    # Now update all cycle records with the complete cycles_metadata
    for detail in details:
        detail.cycles_metadata = cycles

    return cycles
