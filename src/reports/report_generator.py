"""Database-backed Excel report generation.

Queries the database to populate all Excel sheets on-demand.
This is the single source of truth for all reporting.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates Excel reports by querying the database.

    Supports generating sheets for:
    - Summary (statistics and metrics)
    - Audio Processing
    - Biomechanics Import
    - Synchronization
    - Movement Cycles

    All data is queried on-demand from the database.
    """

    def __init__(self, session: Session):
        """Initialize report generator with database session.

        Args:
            session: SQLAlchemy session for database queries
        """
        self.session = session

    def _coerce_excel_compatible(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame contains Excel-compatible datetime values.

        Excel does not support timezone-aware datetimes. Convert any timezone-aware
        datetime columns (or objects) to naive UTC timestamps.
        """
        if df.empty:
            return df

        df = df.copy()
        for col in df.columns:
            series = df[col]
            if isinstance(series.dtype, pd.DatetimeTZDtype):
                df[col] = series.dt.tz_convert(None)
            elif pd.api.types.is_object_dtype(series):
                df[col] = series.apply(
                    lambda v: v.replace(tzinfo=None)
                    if isinstance(v, datetime) and v.tzinfo is not None
                    else v
                )
        return df

    @staticmethod
    def _format_list_for_excel(value) -> Optional[str]:
        """Convert list to comma-separated string for Excel.

        PostgreSQL ARRAY columns return as Python lists, but pandas/Excel
        will convert these to string representations like "['value']".
        This method properly converts them to readable strings like "value".

        Args:
            value: List, string, or other value

        Returns:
            Comma-separated string or None
        """
        if value is None:
            return None
        if isinstance(value, list):
            if not value:
                return None
            # Join list items with comma
            return ", ".join(str(v) for v in value)
        # Already a string or other type
        return value

    def generate_audio_sheet(self, study_id: int, maneuver: str,
                           knee: str) -> pd.DataFrame:
        """Query database and generate Audio Processing sheet.

        Args:
            study_id: Study enrollment ID (studies.id)
            maneuver: Maneuver code (walk, sts, fe)
            knee: Knee side (left, right)

        Returns:
            DataFrame with audio processing data
        """
        from src.db.models import AudioProcessingRecord

        query = self.session.query(AudioProcessingRecord).filter(
            AudioProcessingRecord.study_id == study_id,
            AudioProcessingRecord.maneuver == maneuver,
            AudioProcessingRecord.knee == knee,
            AudioProcessingRecord.is_active == True,  # noqa: E712
        )

        records = query.all()

        data = []
        for record in records:
            # Get biomechanics info if linked
            biomech_record = None
            if record.biomechanics_import_id:
                from src.db.models import BiomechanicsImportRecord
                biomech_record = self.session.query(BiomechanicsImportRecord).filter(
                    BiomechanicsImportRecord.id == record.biomechanics_import_id
                ).first()

            # Get participant number to show in Excel (not database ID)
            study_record = record.study
            participant_number = study_record.study_participant_id if study_record else record.study_id

            # Parse recording datetime from audio file name (most reliable source)
            from src.orchestration.processing_log import _parse_audio_filename
            audio_file_name = record.audio_file_name
            if audio_file_name.endswith("_with_freq"):
                audio_file_name = audio_file_name.replace("_with_freq", "")
            if audio_file_name.lower().endswith(".bin"):
                audio_file_name = Path(audio_file_name).stem
            _, recording_dt_parsed = _parse_audio_filename(audio_file_name)
            recording_datetime = (
                recording_dt_parsed
                or record.recording_date
                or record.recording_time
                or record.file_time
            )
            recording_timezone = getattr(record, "recording_timezone", None) or "UTC"

            data.append({
                'Audio Processing ID': record.id,
                'Participant ID': participant_number,
                'Knee': record.knee,
                'Maneuver': record.maneuver,
                'Audio File Name': audio_file_name,
                'Device Serial': record.device_serial,
                'Firmware Version': record.firmware_version,
                'Recording Datetime': recording_datetime,
                'Recording Timezone': recording_timezone,
                'Recording Length (s)': record.duration_seconds,
                'File Size MB': record.file_size_mb,
                'Num Channels': record.num_channels,
                'Sample Rate': record.sample_rate,
                'Linked Biomechanics': record.biomechanics_import_id is not None,
                'Biomechanics Type': biomech_record.biomechanics_type if biomech_record else None,
                'Biomechanics Sync Method': biomech_record.biomechanics_sync_method if biomech_record else None,
                'Biomechanics Sample Rate (Hz)': biomech_record.biomechanics_sample_rate if biomech_record else None,
                'Biomechanics Import ID': record.biomechanics_import_id,
                'Mic 1 Position': record.mic_1_position,
                'Mic 2 Position': record.mic_2_position,
                'Mic 3 Position': record.mic_3_position,
                'Mic 4 Position': record.mic_4_position,
                'Processing Date': record.processing_date,
                'Processing Status': record.processing_status,
                'QC Fail': record.qc_signal_dropout or record.qc_continuous_artifact,
                'QC Fail Segments': record.qc_fail_segments,
                'QC Fail Segments Ch1': record.qc_fail_segments_ch1,
                'QC Fail Segments Ch2': record.qc_fail_segments_ch2,
                'QC Fail Segments Ch3': record.qc_fail_segments_ch3,
                'QC Fail Segments Ch4': record.qc_fail_segments_ch4,
                'QC Signal Dropout': record.qc_signal_dropout,
                'QC Signal Dropout Segments': record.qc_signal_dropout_segments,
                'QC Signal Dropout Ch1': record.qc_signal_dropout_ch1,
                'QC Signal Dropout Segments Ch1': record.qc_signal_dropout_segments_ch1,
                'QC Signal Dropout Ch2': record.qc_signal_dropout_ch2,
                'QC Signal Dropout Segments Ch2': record.qc_signal_dropout_segments_ch2,
                'QC Signal Dropout Ch3': record.qc_signal_dropout_ch3,
                'QC Signal Dropout Segments Ch3': record.qc_signal_dropout_segments_ch3,
                'QC Signal Dropout Ch4': record.qc_signal_dropout_ch4,
                'QC Signal Dropout Segments Ch4': record.qc_signal_dropout_segments_ch4,
                'QC Continuous Artifact': record.qc_continuous_artifact,
                'QC Continuous Artifact Segments': record.qc_continuous_artifact_segments,
                'QC Continuous Artifact Ch1': record.qc_continuous_artifact_ch1,
                'QC Continuous Artifact Segments Ch1': record.qc_continuous_artifact_segments_ch1,
                'QC Continuous Artifact Ch2': record.qc_continuous_artifact_ch2,
                'QC Continuous Artifact Segments Ch2': record.qc_continuous_artifact_segments_ch2,
                'QC Continuous Artifact Ch3': record.qc_continuous_artifact_ch3,
                'QC Continuous Artifact Segments Ch3': record.qc_continuous_artifact_segments_ch3,
                'QC Continuous Artifact Ch4': record.qc_continuous_artifact_ch4,
                'QC Continuous Artifact Segments Ch4': record.qc_continuous_artifact_segments_ch4,
                'Audio QC Version': record.audio_qc_version,
            })

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)

        base_columns = [
            "Audio Processing ID",
            "Participant ID",
            "Knee",
            "Maneuver",
            "Audio File Name",
            "Device Serial",
            "Firmware Version",
            "Recording Datetime",
            "Recording Timezone",
            "Recording Length (s)",
            "File Size MB",
            "Num Channels",
            "Sample Rate",
            "Linked Biomechanics",
            "Biomechanics Type",
            "Biomechanics Sync Method",
            "Biomechanics Sample Rate (Hz)",
            "Biomechanics Import ID",
            "Mic 1 Position",
            "Mic 2 Position",
            "Mic 3 Position",
            "Mic 4 Position",
            "Processing Date",
            "Processing Status",
        ]

        qc_columns = [
            "QC Fail",
            "QC Fail Segments",
            "QC Fail Segments Ch1",
            "QC Fail Segments Ch2",
            "QC Fail Segments Ch3",
            "QC Fail Segments Ch4",
            "QC Signal Dropout",
            "QC Signal Dropout Segments",
            "QC Signal Dropout Ch1",
            "QC Signal Dropout Segments Ch1",
            "QC Signal Dropout Ch2",
            "QC Signal Dropout Segments Ch2",
            "QC Signal Dropout Ch3",
            "QC Signal Dropout Segments Ch3",
            "QC Signal Dropout Ch4",
            "QC Signal Dropout Segments Ch4",
            "QC Continuous Artifact",
            "QC Continuous Artifact Segments",
            "QC Continuous Artifact Ch1",
            "QC Continuous Artifact Segments Ch1",
            "QC Continuous Artifact Ch2",
            "QC Continuous Artifact Segments Ch2",
            "QC Continuous Artifact Ch3",
            "QC Continuous Artifact Segments Ch3",
            "QC Continuous Artifact Ch4",
            "QC Continuous Artifact Segments Ch4",
            "QC Artifact Type",
            "QC Artifact Type Ch1",
            "QC Artifact Type Ch2",
            "QC Artifact Type Ch3",
            "QC Artifact Type Ch4",
            "Audio QC Version",
        ]

        ordered_cols = [c for c in base_columns if c in df.columns]
        ordered_cols += [c for c in qc_columns if c in df.columns]
        ordered_cols += [c for c in df.columns if c not in ordered_cols]

        return df[ordered_cols]

    def generate_biomechanics_sheet(self, study_id: int, maneuver: str,
                                   knee: str) -> pd.DataFrame:
        """Query database and generate Biomechanics Import sheet.

        Args:
            study_id: Study enrollment ID (studies.id)
            maneuver: Maneuver code (walk, sts, fe)
            knee: Knee side (left, right)

        Returns:
            DataFrame with biomechanics import data
        """
        from src.db.models import BiomechanicsImportRecord

        query = self.session.query(BiomechanicsImportRecord).filter(
            BiomechanicsImportRecord.study_id == study_id,
            BiomechanicsImportRecord.maneuver == maneuver,
            BiomechanicsImportRecord.knee == knee,
            BiomechanicsImportRecord.is_active == True,  # noqa: E712
        )

        records = query.all()

        data = []
        for record in records:
            # Get participant number to show in Excel (not database ID)
            study_record = record.study
            participant_number = study_record.study_participant_id if study_record else record.study_id

            data.append({
                'Biomechanics Import ID': record.id,
                'Participant ID': participant_number,
                'Knee': record.knee,
                'Maneuver': record.maneuver,
                'Biomechanics File': record.biomechanics_file,
                'Sheet Name': record.sheet_name,
                'Biomechanics Type': record.biomechanics_type,
                'Sync Method': record.biomechanics_sync_method,
                'Sample Rate': record.biomechanics_sample_rate,
                'Num Sub-Recordings': record.num_sub_recordings,
                'Num Passes': record.num_passes,
                'Duration (s)': record.duration_seconds,
                'Num Data Points': record.num_data_points,
                'Processing Date': record.processing_date,
                'Processing Status': record.processing_status,
            })

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if "Sync File" in df.columns:
            df = df[df["Sync File"].notna()].drop_duplicates(subset=["Sync File"])
        return df

    def generate_synchronization_sheet(self, study_id: int, maneuver: str,
                                      knee: str) -> pd.DataFrame:
        """Query database and generate Synchronization sheet.

        Args:
            study_id: Study enrollment ID (studies.id)
            maneuver: Maneuver code (walk, sts, fe)
            knee: Knee side (left, right)

        Returns:
            DataFrame with synchronization data
        """
        from src.db.models import AudioProcessingRecord, SynchronizationRecord

        query = (
            self.session.query(SynchronizationRecord)
            .join(
                AudioProcessingRecord,
                SynchronizationRecord.audio_processing_id == AudioProcessingRecord.id,
            )
            .filter(
                SynchronizationRecord.study_id == study_id,
                AudioProcessingRecord.maneuver == maneuver,
                AudioProcessingRecord.knee == knee,
                SynchronizationRecord.is_active == True,  # noqa: E712
            )
            .order_by(
                SynchronizationRecord.pass_number,
                SynchronizationRecord.id,
            )
        )

        records = query.all()

        data = []
        for record in records:
            # Get participant number to show in Excel (not database ID)
            study_record = record.study
            participant_number = study_record.study_participant_id if study_record else record.study_id

            data.append({
                'Synchronization ID': record.id,
                'Participant ID': participant_number,
                'Pass Number': record.pass_number,
                'Speed': record.speed,
                'Audio Processing ID': record.audio_processing_id,
                'Biomechanics Import ID': record.biomechanics_import_id,
                'Sync File': record.sync_file_name,

                # Synchronization times (biomechanics synced to audio, audio t=0 = sync t=0)
                'Bio Left Sync Time': record.bio_left_sync_time,
                'Bio Right Sync Time': record.bio_right_sync_time,
                'Bio Sync Offset': record.bio_sync_offset,
                'Aligned Sync Time': record.aligned_sync_time,

                # Sync method details
                'Sync Method': record.sync_method,
                'Consensus Methods': record.consensus_methods,
                'Consensus Time': record.consensus_time,
                'Method Agreement Span': record.method_agreement_span,  # Moved next to consensus
                'RMS Time': record.rms_time,
                'Onset Time': record.onset_time,
                'Freq Time': record.freq_time,

                # Detection methods
                'Stomp Detection Methods': self._format_list_for_excel(record.stomp_detection_methods),
                'Selected Stomp Method': record.selected_stomp_method,

                # Biomechanics-based sync times
                'Bio Selected Sync Time': record.bio_selected_sync_time,
                'Contra Bio Selected Sync Time': record.contra_bio_selected_sync_time,

                # Audio sync times (optional - mic on to participant stopping)
                'Audio Sync Time Left': record.audio_sync_time_left,
                'Audio Sync Time Right': record.audio_sync_time_right,
                'Audio Sync Offset': record.audio_sync_offset,

                # Audio-based sync times (different from bio-based)
                'Audio Selected Sync Time': record.audio_selected_sync_time,
                'Contra Audio Selected Sync Time': record.contra_audio_selected_sync_time,

                # Cycle statistics
                'Sync Duration': record.sync_duration,
                'Total Cycles Extracted': record.total_cycles_extracted,
                'Clean Cycles': record.clean_cycles,
                'Outlier Cycles': record.outlier_cycles,
                'Mean Cycle Duration': record.mean_cycle_duration_s,
                'Median Cycle Duration': record.median_cycle_duration_s,
                'Min Cycle Duration': record.min_cycle_duration_s,
                'Max Cycle Duration': record.max_cycle_duration_s,

                # Periodic artifact detection (sync-level)
                'Periodic Artifact Detected': record.periodic_artifact_detected,
                'Periodic Artifact Ch1': record.periodic_artifact_detected_ch1,
                'Periodic Artifact Ch2': record.periodic_artifact_detected_ch2,
                'Periodic Artifact Ch3': record.periodic_artifact_detected_ch3,
                'Periodic Artifact Ch4': record.periodic_artifact_detected_ch4,
                'Periodic Artifact Segments': record.periodic_artifact_segments,
                'Periodic Artifact Segments Ch1': record.periodic_artifact_segments_ch1,
                'Periodic Artifact Segments Ch2': record.periodic_artifact_segments_ch2,
                'Periodic Artifact Segments Ch3': record.periodic_artifact_segments_ch3,
                'Periodic Artifact Segments Ch4': record.periodic_artifact_segments_ch4,

                # Processing
                'Processing Date': record.processing_date,
                'Processing Status': record.processing_status,
            })

        return pd.DataFrame(data) if data else pd.DataFrame()

    def generate_movement_cycles_sheet(self, study_id: int, maneuver: str,
                                      knee: str) -> pd.DataFrame:
        """Query database and generate Movement Cycles sheet.

        Args:
            study_id: Study enrollment ID (studies.id)
            maneuver: Maneuver code (walk, sts, fe)
            knee: Knee side (left, right)

        Returns:
            DataFrame with movement cycle data
        """
        from sqlalchemy import func, select
        from sqlalchemy.orm import joinedload

        from src.db.models import (
            AudioProcessingRecord,
            MovementCycleRecord,
            SynchronizationRecord,
        )

        # Query movement cycles with related audio and sync data
        query = self.session.query(MovementCycleRecord).join(
            AudioProcessingRecord, MovementCycleRecord.audio_processing_id == AudioProcessingRecord.id
        ).outerjoin(
            SynchronizationRecord, MovementCycleRecord.synchronization_id == SynchronizationRecord.id
        ).filter(
            MovementCycleRecord.study_id == study_id,
            AudioProcessingRecord.maneuver == maneuver,
            AudioProcessingRecord.knee == knee,
            MovementCycleRecord.is_active == True,  # noqa: E712
        ).order_by(
            func.coalesce(SynchronizationRecord.pass_number, 0),
            MovementCycleRecord.cycle_index
        )

        records = query.all()

        data = []
        for record in records:
            # Get sync data if available
            sync_record = record.synchronization
            pass_number = sync_record.pass_number if sync_record else None
            speed = sync_record.speed if sync_record else None

            # Get participant number to show in Excel (not database ID)
            study_record = record.study
            participant_number = study_record.study_participant_id if study_record else record.study_id

            data.append({
                'Movement Cycle ID': record.id,
                'Participant ID': participant_number,
                'Pass Number': pass_number,
                'Speed': speed,
                'Cycle Index': record.cycle_index,
                'Is Outlier': record.is_outlier,
                'Audio Processing ID': record.audio_processing_id,
                'Synchronization ID': record.synchronization_id,
                'Cycle File': record.cycle_file,
                'Start Time (s)': record.start_time_s,
                'End Time (s)': record.end_time_s,
                'Duration (s)': record.duration_s,
                'Start Time': record.start_time,
                'End Time': record.end_time,
                'Biomechanics QC Fail': record.biomechanics_qc_fail,
                'Sync QC Fail': record.sync_qc_fail,
                'Audio QC Fail': record.audio_qc_fail,
                'Audio QC Failures': self._format_list_for_excel(record.audio_qc_failures),
                'Audio Artifact Intermittent Fail': record.audio_artifact_intermittent_fail,
                'Audio Artifact Intermittent Fail Ch1': record.audio_artifact_intermittent_fail_ch1,
                'Audio Artifact Intermittent Fail Ch2': record.audio_artifact_intermittent_fail_ch2,
                'Audio Artifact Intermittent Fail Ch3': record.audio_artifact_intermittent_fail_ch3,
                'Audio Artifact Intermittent Fail Ch4': record.audio_artifact_intermittent_fail_ch4,
                'Audio Artifact Timestamps': self._format_list_for_excel(record.audio_artifact_timestamps),
                'Audio Artifact Timestamps Ch1': self._format_list_for_excel(record.audio_artifact_timestamps_ch1),
                'Audio Artifact Timestamps Ch2': self._format_list_for_excel(record.audio_artifact_timestamps_ch2),
                'Audio Artifact Timestamps Ch3': self._format_list_for_excel(record.audio_artifact_timestamps_ch3),
                'Audio Artifact Timestamps Ch4': self._format_list_for_excel(record.audio_artifact_timestamps_ch4),
                'Audio Artifact Dropout Fail': record.audio_artifact_dropout_fail,
                'Audio Artifact Dropout Fail Ch1': record.audio_artifact_dropout_fail_ch1,
                'Audio Artifact Dropout Fail Ch2': record.audio_artifact_dropout_fail_ch2,
                'Audio Artifact Dropout Fail Ch3': record.audio_artifact_dropout_fail_ch3,
                'Audio Artifact Dropout Fail Ch4': record.audio_artifact_dropout_fail_ch4,
                'Audio Artifact Dropout Timestamps': self._format_list_for_excel(record.audio_artifact_dropout_timestamps),
                'Audio Artifact Dropout Timestamps Ch1': self._format_list_for_excel(record.audio_artifact_dropout_timestamps_ch1),
                'Audio Artifact Dropout Timestamps Ch2': self._format_list_for_excel(record.audio_artifact_dropout_timestamps_ch2),
                'Audio Artifact Dropout Timestamps Ch3': self._format_list_for_excel(record.audio_artifact_dropout_timestamps_ch3),
                'Audio Artifact Dropout Timestamps Ch4': self._format_list_for_excel(record.audio_artifact_dropout_timestamps_ch4),
                'Audio Artifact Continuous Fail': record.audio_artifact_continuous_fail,
                'Audio Artifact Continuous Fail Ch1': record.audio_artifact_continuous_fail_ch1,
                'Audio Artifact Continuous Fail Ch2': record.audio_artifact_continuous_fail_ch2,
                'Audio Artifact Continuous Fail Ch3': record.audio_artifact_continuous_fail_ch3,
                'Audio Artifact Continuous Fail Ch4': record.audio_artifact_continuous_fail_ch4,
                'Audio Artifact Continuous Timestamps': self._format_list_for_excel(record.audio_artifact_continuous_timestamps),
                'Audio Artifact Continuous Timestamps Ch1': self._format_list_for_excel(record.audio_artifact_continuous_timestamps_ch1),
                'Audio Artifact Continuous Timestamps Ch2': self._format_list_for_excel(record.audio_artifact_continuous_timestamps_ch2),
                'Audio Artifact Continuous Timestamps Ch3': self._format_list_for_excel(record.audio_artifact_continuous_timestamps_ch3),
                'Audio Artifact Continuous Timestamps Ch4': self._format_list_for_excel(record.audio_artifact_continuous_timestamps_ch4),
                'Audio Artifact Periodic Fail': record.audio_artifact_periodic_fail,
                'Audio Artifact Periodic Fail Ch1': record.audio_artifact_periodic_fail_ch1,
                'Audio Artifact Periodic Fail Ch2': record.audio_artifact_periodic_fail_ch2,
                'Audio Artifact Periodic Fail Ch3': record.audio_artifact_periodic_fail_ch3,
                'Audio Artifact Periodic Fail Ch4': record.audio_artifact_periodic_fail_ch4,
                'Audio Artifact Periodic Timestamps': self._format_list_for_excel(record.audio_artifact_periodic_timestamps),
                'Audio Artifact Periodic Timestamps Ch1': self._format_list_for_excel(record.audio_artifact_periodic_timestamps_ch1),
                'Audio Artifact Periodic Timestamps Ch2': self._format_list_for_excel(record.audio_artifact_periodic_timestamps_ch2),
                'Audio Artifact Periodic Timestamps Ch3': self._format_list_for_excel(record.audio_artifact_periodic_timestamps_ch3),
                'Audio Artifact Periodic Timestamps Ch4': self._format_list_for_excel(record.audio_artifact_periodic_timestamps_ch4),
                'Processing Date': record.created_at,
            })

        return pd.DataFrame(data) if data else pd.DataFrame()

    def generate_summary_sheet(self, study_id: int, maneuver: str,
                             knee: str) -> pd.DataFrame:
        """Generate summary statistics sheet from database queries.

        Args:
            study_id: Study enrollment ID (studies.id)
            maneuver: Maneuver code (walk, sts, fe)
            knee: Knee side (left, right)

        Returns:
            DataFrame with summary statistics
        """
        from sqlalchemy import func

        from src.db.models import (
            AudioProcessingRecord,
            BiomechanicsImportRecord,
            MovementCycleRecord,
            StudyRecord,
            SynchronizationRecord,
        )

        audio_count = self.session.query(AudioProcessingRecord).filter(
            AudioProcessingRecord.study_id == study_id,
            AudioProcessingRecord.maneuver == maneuver,
            AudioProcessingRecord.knee == knee,
            AudioProcessingRecord.is_active == True,  # noqa: E712
        ).count()

        biomech_count = self.session.query(BiomechanicsImportRecord).filter(
            BiomechanicsImportRecord.study_id == study_id,
            BiomechanicsImportRecord.maneuver == maneuver,
            BiomechanicsImportRecord.knee == knee,
            BiomechanicsImportRecord.is_active == True,  # noqa: E712
        ).count()

        sync_count = self.session.query(
            func.count(func.distinct(SynchronizationRecord.sync_file_name))
        ).join(
            AudioProcessingRecord,
            SynchronizationRecord.audio_processing_id == AudioProcessingRecord.id,
        ).filter(
            SynchronizationRecord.study_id == study_id,
            AudioProcessingRecord.maneuver == maneuver,
            AudioProcessingRecord.knee == knee,
            SynchronizationRecord.sync_file_name.isnot(None),
            SynchronizationRecord.is_active == True,  # noqa: E712
        ).scalar()

        if sync_count is None:
            sync_count = 0

        cycle_count = self.session.query(MovementCycleRecord).join(
            AudioProcessingRecord,
            MovementCycleRecord.audio_processing_id == AudioProcessingRecord.id,
        ).filter(
            MovementCycleRecord.study_id == study_id,
            AudioProcessingRecord.maneuver == maneuver,
            AudioProcessingRecord.knee == knee,
            MovementCycleRecord.is_active == True,  # noqa: E712
        ).count()

        # Get participant number for display
        study_record = self.session.query(StudyRecord).filter(
            StudyRecord.id == study_id
        ).first()
        participant_number = study_record.study_participant_id if study_record else study_id

        summary_data = {
            'Metric': [
                'Participant ID',
                'Knee',
                'Maneuver',
                'Audio Records',
                'Biomechanics Records',
                'Synchronization Records',
                'Movement Cycles',
                'Generated',
            ],
            'Value': [
                participant_number,
                knee,
                maneuver,
                audio_count,
                biomech_count,
                sync_count,
                cycle_count,
                datetime.now().isoformat(),
            ]
        }

        return pd.DataFrame(summary_data)

    @staticmethod
    def generate_legend_validation_sheet(
        legend_mismatches: List,
    ) -> pd.DataFrame:
        """Generate Legend Validation sheet from cross-sheet mismatches.

        Only produces output when mismatches exist between Acoustic Notes
        and Mic Setup sheets.

        Args:
            legend_mismatches: List of LegendMismatch instances.

        Returns:
            DataFrame with mismatch details, or empty DataFrame.
        """
        if not legend_mismatches:
            return pd.DataFrame()

        data = []
        for mm in legend_mismatches:
            data.append({
                'Knee': mm.knee,
                'Maneuver': mm.maneuver,
                'Field': mm.field,
                'Acoustic Notes Value': mm.acoustic_notes_value,
                'Mic Setup Value': mm.mic_setup_value,
            })

        return pd.DataFrame(data)

    def save_to_excel(self, output_path: Path, participant_id: int,
                     maneuver: str, knee: str,
                     legend_mismatches: Optional[List] = None) -> Path:
        """Generate all sheets and save to Excel file.

        Args:
            output_path: Path to save Excel file
            participant_id: Study enrollment ID (studies.id) — legacy param name
            maneuver: Maneuver code
            knee: Knee side
            legend_mismatches: Optional list of LegendMismatch instances for
                cross-sheet validation reporting.

        Returns:
            Path to generated Excel file
        """
        study_id = participant_id  # Callers pass studies.id under legacy name
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating Excel report: {output_path}")

        # Generate all sheets
        sheets = {
            'Summary': self.generate_summary_sheet(study_id, maneuver, knee),
            'Audio': self.generate_audio_sheet(study_id, maneuver, knee),
            'Biomechanics': self.generate_biomechanics_sheet(study_id, maneuver, knee),
            'Synchronization': self.generate_synchronization_sheet(study_id, maneuver, knee),
            'Cycles': self.generate_movement_cycles_sheet(study_id, maneuver, knee),
        }

        # Add Legend Validation sheet if mismatches exist
        if legend_mismatches:
            sheets['Legend Validation'] = self.generate_legend_validation_sheet(
                legend_mismatches
            )

        # Write to Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for sheet_name, df in sheets.items():
                if not df.empty:
                    df = self._coerce_excel_compatible(df)
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    logger.debug(f"Wrote {len(df)} rows to '{sheet_name}' sheet")
                else:
                    logger.debug(f"No data for '{sheet_name}' sheet")

        logger.info(f"Report saved: {output_path}")
        return output_path
