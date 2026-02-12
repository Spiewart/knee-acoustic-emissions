"""Processing log module for tracking processing status and generating reports.

This module provides metadata tracking for participant processing. All data is stored
in the PostgreSQL database, and reports are generated on-demand by querying the DB.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import select

from src.db.models import ParticipantRecord, StudyRecord
from src.metadata import AudioProcessing, BiomechanicsImport, Synchronization
from src.orchestration.cli_db_helpers import close_db_session, create_db_session

logger = logging.getLogger(__name__)


def _normalize_maneuver(maneuver: str) -> str:
    """Normalize maneuver values to DB-friendly codes."""
    maneuver_lower = maneuver.lower()
    if maneuver_lower in ("fe", "flexion_extension"):
        return "fe"
    if maneuver_lower in ("sts", "sit_to_stand"):
        return "sts"
    return "walk"


def _normalize_knee(knee_side: str) -> str:
    """Normalize knee side to lowercase values used in DB."""
    return "left" if knee_side.lower().startswith("l") else "right"


def _parse_audio_filename(audio_file_name: str) -> Tuple[Optional[str], Optional[datetime]]:
    """Parse device serial and recording datetime from audio filename.

    Expected format: HP_W<hw_rev>-<serial>-<date>_<time>.bin
    Example: HP_W11.2-1-20240312_124051.bin

    Note: The W<hw_rev> portion (e.g., W11.2) encodes hardware revision, NOT firmware version.
    Firmware version must be read from the binary file header (devFirmwareVersion).

    Returns:
        (device_serial, recording_datetime) tuple. Values may be None if not parseable.
    """
    import re

    stem = Path(audio_file_name).stem if "/" in audio_file_name or "\\" in audio_file_name else audio_file_name.replace(".bin", "")

    device_serial = None
    recording_date = None

    # Parse pattern: HP_W<hw_rev>-<serial>-<date>_<time>
    # Example: HP_W11.2-1-20240312_124051
    # Group 1 is hardware revision (ignored), group 2 is serial
    pattern = r"HP_W([\d.]+)-([\d]+)-(20\d{2})(0\d|1[0-2])([0-3]\d)_(\d{6})"
    match = re.search(pattern, stem)

    if match:
        try:
            device_serial = match.group(2)
            year = int(match.group(3))
            month = int(match.group(4))
            day = int(match.group(5))
            time_str = match.group(6)
            hour = int(time_str[0:2])
            minute = int(time_str[2:4])
            second = int(time_str[4:6])
            recording_date = datetime(year, month, day, hour, minute, second)
        except (ValueError, IndexError):
            pass

    return device_serial, recording_date


def _parse_study_and_participant(
    study_id: str | int,
    study_name: Optional[str] = None,
) -> Tuple[str, int]:
    """Parse study name and participant number from an ID string or int."""
    if isinstance(study_id, int):
        return study_name or "AOA", study_id

    study_id_str = str(study_id).lstrip("#")
    if study_name:
        try:
            return study_name, int(study_id_str)
        except ValueError:
            return study_name, 1

    if study_id_str.startswith("AOA"):
        return "AOA", int(study_id_str[3:]) if len(study_id_str) > 3 else 1
    if study_id_str.startswith("preOA"):
        return "preOA", int(study_id_str[5:]) if len(study_id_str) > 5 else 1
    if study_id_str.startswith("SMoCK"):
        return "SMoCK", int(study_id_str[5:]) if len(study_id_str) > 5 else 1
    try:
        return "AOA", int(study_id_str)
    except ValueError:
        return "AOA", 1


def _get_participant_db_id(
    study_name: str,
    participant_number: int,
    session,
) -> int:
    """Resolve database participant ID from study name and participant study_id."""
    study = session.execute(
        select(StudyRecord).where(StudyRecord.name == study_name)
    ).scalar_one_or_none()
    if study is None:
        study = StudyRecord(name=study_name)
        session.add(study)
        session.flush()

    participant = session.execute(
        select(ParticipantRecord).where(
            ParticipantRecord.study_participant_id == study.id,
            ParticipantRecord.study_id == participant_number,
        )
    ).scalar_one_or_none()
    if participant is None:
        participant = ParticipantRecord(
            study_participant_id=study.id,
            study_id=participant_number,
        )
        session.add(participant)
        session.flush()

    return participant.id


@dataclass
class ManeuverProcessingLog:
    """Metadata about a processed knee/maneuver combination.

    This log tracks what was processed. All data lives in PostgreSQL,
    and Excel reports are generated from DB queries.
    """

    study_id: str
    knee_side: str
    maneuver: str
    maneuver_directory: Path
    log_created: datetime = field(default_factory=datetime.now)
    log_updated: datetime = field(default_factory=datetime.now)
    study: Optional[str] = None

    audio_record: Optional[AudioProcessing] = None
    biomechanics_record: Optional[BiomechanicsImport] = None
    synchronization_records: List[Synchronization] = field(default_factory=list)
    movement_cycles_records: List[Synchronization] = field(default_factory=list)

    @classmethod
    def get_or_create(
        cls,
        study_id: str,
        knee_side: str,
        maneuver: str,
        maneuver_directory: Path,
    ) -> "ManeuverProcessingLog":
        """Return a new in-memory log.

        Persistent state is stored in PostgreSQL, so this no longer reads Excel.
        """
        return cls(
            study_id=study_id,
            knee_side=knee_side,
            maneuver=maneuver,
            maneuver_directory=maneuver_directory,
            log_created=datetime.now(),
            log_updated=datetime.now(),
        )

    def update_audio_record(self, record: AudioProcessing) -> None:
        """Update audio record metadata."""
        self.audio_record = record
        self.study = record.study
        self.log_updated = datetime.now()

    def update_biomechanics_record(self, record: BiomechanicsImport) -> None:
        """Update biomechanics record metadata."""
        self.biomechanics_record = record
        self.study = record.study
        self.log_updated = datetime.now()

    def add_synchronization_record(self, record: Synchronization) -> None:
        """Append a synchronization record."""
        self.synchronization_records.append(record)
        self.study = record.study
        self.log_updated = datetime.now()

    def add_movement_cycles_record(self, record: Synchronization) -> None:
        """Append a movement cycles summary record."""
        self.movement_cycles_records.append(record)
        self.study = record.study
        self.log_updated = datetime.now()

    def _default_output_path(self) -> Path:
        knee_label = "Left" if self.knee_side.lower().startswith("l") else "Right"
        file_name = f"processing_log_{self.study_id}_{knee_label}_{self.maneuver}.xlsx"
        return self.maneuver_directory / file_name

    def save_to_excel(
        self,
        output_path: Optional[Path] = None,
        session=None,
        db_url: Optional[str] = None,
    ) -> Path:
        """Generate Excel report from database queries.

        Requires a PostgreSQL connection (AE_DATABASE_URL).
        """
        from src.reports.report_generator import ReportGenerator

        created_session = False
        if session is None:
            session = create_db_session(db_url=db_url)
            if session is None:
                raise RuntimeError("Database connection required to generate reports")
            created_session = True

        try:
            study_name, participant_number = _parse_study_and_participant(
                self.study_id, self.study
            )
            participant_id = _get_participant_db_id(
                study_name, participant_number, session
            )
            report_gen = ReportGenerator(session)
            knee = _normalize_knee(self.knee_side)
            maneuver = _normalize_maneuver(self.maneuver)
            output_path = output_path or self._default_output_path()
            return report_gen.save_to_excel(
                output_path,
                participant_id=participant_id,
                maneuver=maneuver,
                knee=knee,
            )
        finally:
            if created_session:
                close_db_session(session)


@dataclass
class KneeProcessingLog:
    """Knee-level processing summary.

    Generates a knee-level Excel summary from DB queries.
    """

    study_id: str
    knee_side: str
    knee_directory: Path
    log_created: datetime = field(default_factory=datetime.now)
    log_updated: datetime = field(default_factory=datetime.now)
    study: Optional[str] = None
    maneuver_logs: Dict[str, ManeuverProcessingLog] = field(default_factory=dict)

    @classmethod
    def get_or_create(
        cls,
        study_id: str,
        knee_side: str,
        knee_directory: Path,
    ) -> "KneeProcessingLog":
        """Return a new in-memory knee log."""
        return cls(
            study_id=study_id,
            knee_side=knee_side,
            knee_directory=knee_directory,
            log_created=datetime.now(),
            log_updated=datetime.now(),
        )

    def update_maneuver_summary(
        self,
        maneuver: str,
        maneuver_log: ManeuverProcessingLog,
    ) -> None:
        """Update the maneuver log for this knee."""
        self.maneuver_logs[maneuver] = maneuver_log
        if maneuver_log.study:
            self.study = maneuver_log.study
        self.log_updated = datetime.now()

    def _default_output_path(self) -> Path:
        knee_label = "Left" if self.knee_side.lower().startswith("l") else "Right"
        file_name = f"knee_processing_log_{self.study_id}_{knee_label}.xlsx"
        return self.knee_directory / file_name

    def save_to_excel(
        self,
        output_path: Optional[Path] = None,
        session=None,
        db_url: Optional[str] = None,
    ) -> Path:
        """Generate knee-level Excel summary from database queries.

        Requires a PostgreSQL connection (AE_DATABASE_URL).
        """
        created_session = False
        if session is None:
            session = create_db_session(db_url=db_url)
            if session is None:
                raise RuntimeError("Database connection required to generate reports")
            created_session = True

        try:
            study_name, participant_number = _parse_study_and_participant(
                self.study_id, self.study
            )
            participant_id = _get_participant_db_id(
                study_name, participant_number, session
            )
            knee = _normalize_knee(self.knee_side)
            maneuvers = list(self.maneuver_logs.keys()) or [
                "walk",
                "sit_to_stand",
                "flexion_extension",
            ]

            summary_rows = []
            from src.reports.report_generator import ReportGenerator

            report_gen = ReportGenerator(session)
            for maneuver in maneuvers:
                db_maneuver = _normalize_maneuver(maneuver)
                summary = report_gen.generate_summary_sheet(
                    participant_id, db_maneuver, knee
                )
                if not summary.empty:
                    summary_rows.append({
                        "Maneuver": db_maneuver,
                        "Audio Records": summary.loc[summary["Metric"] == "Audio Records", "Value"].iloc[0],
                        "Biomechanics Records": summary.loc[summary["Metric"] == "Biomechanics Records", "Value"].iloc[0],
                        "Synchronization Records": summary.loc[summary["Metric"] == "Synchronization Records", "Value"].iloc[0],
                        "Movement Cycles": summary.loc[summary["Metric"] == "Movement Cycles", "Value"].iloc[0],
                    })
                else:
                    summary_rows.append({
                        "Maneuver": db_maneuver,
                        "Audio Records": 0,
                        "Biomechanics Records": 0,
                        "Synchronization Records": 0,
                        "Movement Cycles": 0,
                    })

            output_path = output_path or self._default_output_path()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            summary_df = pd.DataFrame(summary_rows)
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                summary_df.to_excel(writer, sheet_name="Knee Summary", index=False)

            return output_path
        finally:
            if created_session:
                close_db_session(session)


def _parse_study_from_file_name(file_name: str) -> Optional[str]:
    for prefix in ("AOA", "preOA", "SMoCK"):
        if file_name.startswith(prefix):
            return prefix
    return None


def _infer_biomechanics_type_from_study(study_id: str | int) -> str:
    """Infer biomechanics type from study identifier.

    Defaults to Motion Analysis when uncertain.
    """
    study_name, _ = _parse_study_and_participant(study_id)
    if study_name == "SMoCK":
        return "IMU"
    return "Motion Analysis"


def create_audio_record_from_data(
    audio_file_name: str,
    audio_df: pd.DataFrame,
    audio_bin_path: Optional[Path] = None,
    audio_pkl_path: Optional[Path] = None,
    metadata: Optional[Dict] = None,
    biomechanics_type: Optional[str] = None,
    qc_data: Optional[Dict] = None,
    knee: Optional[str] = None,
    maneuver: Optional[str] = None,
    **kwargs,
) -> AudioProcessing:
    """Create AudioProcessing record from data and metadata."""
    metadata = metadata or {}
    study_name = metadata.get("study") or _parse_study_from_file_name(audio_file_name) or "AOA"
    study_id = metadata.get("study_id") or 1
    knee_value = (knee or metadata.get("knee") or "left").lower()
    maneuver_value = _normalize_maneuver(maneuver or metadata.get("maneuver") or "walk")

    def _parse_datetime_value(value: Optional[object]) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return None
        return None

    # Parse device_serial and recording datetime from audio file name (format: HP_W<hw_rev>-<serial>-<date>_<time>)
    device_serial, recording_dt_parsed = _parse_audio_filename(audio_file_name)

    # Firmware version MUST come from file header, not filename (filename has hardware revision)
    firmware_version = (
        metadata.get("devFirmwareVersion")
        or metadata.get("firmware_version")
        or 0
    )
    # Convert firmware_version to int, handling both float and string inputs
    try:
        if isinstance(firmware_version, str):
            firmware_version = int(float(firmware_version))
        elif isinstance(firmware_version, float):
            firmware_version = int(firmware_version)
        else:
            firmware_version = int(firmware_version)
    except (ValueError, TypeError):
        firmware_version = 0

    # Extract device serial, handling various formats
    # Try metadata first, but skip invalid formats like lists
    device_serial_meta = metadata.get("devSerial") or metadata.get("device_serial")
    if device_serial_meta and isinstance(device_serial_meta, str):
        device_serial = device_serial_meta
    else:
        # Try HP_Serial field from raw metadata
        device_serial_meta = metadata.get("HP_Serial")
        if device_serial_meta:
            device_serial = str(device_serial_meta)
        else:
            # Fall back to parsed value from filename or UNKNOWN
            device_serial = device_serial or "UNKNOWN"
    file_time = _parse_datetime_value(metadata.get("file_time") or metadata.get("fileTime"))
    recording_date = _parse_datetime_value(
        metadata.get("recording_date") or metadata.get("recordingDate")
    )
    if recording_date is None:
        recording_date = recording_dt_parsed or file_time
    recording_time = _parse_datetime_value(
        metadata.get("recording_time") or metadata.get("recordingTime")
    )
    if recording_time is None:
        recording_time = file_time or recording_dt_parsed

    if file_time is None:
        file_time = recording_time

    # If all datetime fields are still None, use a default timestamp
    if all(dt is None for dt in [file_time, recording_date, recording_time]):
        default_dt = datetime.now()
        file_time = default_dt
        recording_date = default_dt
        recording_time = default_dt
    elif file_time is None:
        file_time = recording_date or recording_time or datetime.now()
    elif recording_date is None:
        recording_date = file_time or recording_time or datetime.now()
    elif recording_time is None:
        recording_time = file_time or recording_date or datetime.now()

    file_size_mb = metadata.get("file_size_mb")
    if file_size_mb is None and audio_pkl_path:
        try:
            file_size_mb = audio_pkl_path.stat().st_size / (1024 * 1024)
        except Exception:
            file_size_mb = 0.0

    num_channels = metadata.get("num_channels")
    if num_channels is None:
        num_channels = len([col for col in audio_df.columns if str(col).startswith("ch")])

    mic_positions = metadata.get("mic_positions") or {}
    mic_1_position = mic_positions.get("mic_1_position") or metadata.get("mic_1_position")
    mic_2_position = mic_positions.get("mic_2_position") or metadata.get("mic_2_position")
    mic_3_position = mic_positions.get("mic_3_position") or metadata.get("mic_3_position")
    mic_4_position = mic_positions.get("mic_4_position") or metadata.get("mic_4_position")

    if not all([mic_1_position, mic_2_position, mic_3_position, mic_4_position]):
        raise ValueError("Microphone positions must be provided from the acoustics file legend")

    sample_rate = metadata.get("fs") or metadata.get("sample_rate") or 46875.0
    recording_timezone = metadata.get("recording_timezone") or "UTC"
    qc_data = qc_data or {}

    duration_seconds = metadata.get("duration_seconds")
    if duration_seconds is None and audio_df is not None:
        try:
            if "tt" in audio_df.columns:
                tt = pd.to_numeric(audio_df["tt"], errors="coerce").dropna().to_numpy()
                if len(tt) >= 2:
                    duration_seconds = float(tt[-1] - tt[0])
            if duration_seconds is None:
                sample_rate_val = float(sample_rate) if sample_rate else 0.0
                if sample_rate_val > 0 and len(audio_df) > 1:
                    duration_seconds = float((len(audio_df) - 1) / sample_rate_val)
        except Exception:
            duration_seconds = None

    return AudioProcessing(
        study=study_name,
        study_id=int(study_id),
        audio_file_name=audio_file_name,
        device_serial=str(device_serial),
        firmware_version=int(firmware_version),
        file_time=file_time,
        file_size_mb=float(file_size_mb or 0.0),
        recording_date=recording_date,
        recording_time=recording_time,
        knee=knee_value,
        maneuver=maneuver_value,
        num_channels=int(num_channels),
        sample_rate=float(sample_rate),
        mic_1_position=mic_1_position,
        mic_2_position=mic_2_position,
        mic_3_position=mic_3_position,
        mic_4_position=mic_4_position,
        pkl_file_path=str(audio_pkl_path) if audio_pkl_path else None,
        audio_qc_fail=bool(qc_data.get("audio_qc_fail", False)),
        qc_fail_segments=qc_data.get("qc_fail_segments", []),
        qc_fail_segments_ch1=qc_data.get("qc_fail_segments_ch1", []),
        qc_fail_segments_ch2=qc_data.get("qc_fail_segments_ch2", []),
        qc_fail_segments_ch3=qc_data.get("qc_fail_segments_ch3", []),
        qc_fail_segments_ch4=qc_data.get("qc_fail_segments_ch4", []),
        qc_signal_dropout=qc_data.get("qc_signal_dropout", False),
        qc_signal_dropout_segments=qc_data.get("qc_signal_dropout_segments", []),
        qc_signal_dropout_ch1=qc_data.get("qc_signal_dropout_ch1", False),
        qc_signal_dropout_segments_ch1=qc_data.get("qc_signal_dropout_segments_ch1", []),
        qc_signal_dropout_ch2=qc_data.get("qc_signal_dropout_ch2", False),
        qc_signal_dropout_segments_ch2=qc_data.get("qc_signal_dropout_segments_ch2", []),
        qc_signal_dropout_ch3=qc_data.get("qc_signal_dropout_ch3", False),
        qc_signal_dropout_segments_ch3=qc_data.get("qc_signal_dropout_segments_ch3", []),
        qc_signal_dropout_ch4=qc_data.get("qc_signal_dropout_ch4", False),
        qc_signal_dropout_segments_ch4=qc_data.get("qc_signal_dropout_segments_ch4", []),
        qc_artifact=qc_data.get("qc_artifact", False),
        qc_artifact_type=qc_data.get("qc_artifact_type"),
        qc_artifact_segments=qc_data.get("qc_artifact_segments", []),
        qc_artifact_ch1=qc_data.get("qc_artifact_ch1", False),
        qc_artifact_type_ch1=qc_data.get("qc_artifact_type_ch1"),
        qc_artifact_segments_ch1=qc_data.get("qc_artifact_segments_ch1", []),
        qc_artifact_ch2=qc_data.get("qc_artifact_ch2", False),
        qc_artifact_type_ch2=qc_data.get("qc_artifact_type_ch2"),
        qc_artifact_segments_ch2=qc_data.get("qc_artifact_segments_ch2", []),
        qc_artifact_ch3=qc_data.get("qc_artifact_ch3", False),
        qc_artifact_type_ch3=qc_data.get("qc_artifact_type_ch3"),
        qc_artifact_segments_ch3=qc_data.get("qc_artifact_segments_ch3", []),
        qc_artifact_ch4=qc_data.get("qc_artifact_ch4", False),
        qc_artifact_type_ch4=qc_data.get("qc_artifact_type_ch4"),
        qc_artifact_segments_ch4=qc_data.get("qc_artifact_segments_ch4", []),
        qc_continuous_artifact=qc_data.get("qc_continuous_artifact", False),
        qc_continuous_artifact_segments=qc_data.get("qc_continuous_artifact_segments", []),
        qc_continuous_artifact_ch1=qc_data.get("qc_continuous_artifact_ch1", False),
        qc_continuous_artifact_segments_ch1=qc_data.get("qc_continuous_artifact_segments_ch1", []),
        qc_continuous_artifact_ch2=qc_data.get("qc_continuous_artifact_ch2", False),
        qc_continuous_artifact_segments_ch2=qc_data.get("qc_continuous_artifact_segments_ch2", []),
        qc_continuous_artifact_ch3=qc_data.get("qc_continuous_artifact_ch3", False),
        qc_continuous_artifact_segments_ch3=qc_data.get("qc_continuous_artifact_segments_ch3", []),
        qc_continuous_artifact_ch4=qc_data.get("qc_continuous_artifact_ch4", False),
        qc_continuous_artifact_segments_ch4=qc_data.get("qc_continuous_artifact_segments_ch4", []),
        processing_date=metadata.get("processing_date"),
        processing_status=metadata.get("processing_status"),
        duration_seconds=duration_seconds,
        recording_timezone=recording_timezone,
        linked_biomechanics=bool(metadata.get("linked_biomechanics", False)),
        biomechanics_file=metadata.get("biomechanics_file"),
        biomechanics_type=biomechanics_type or metadata.get("biomechanics_type"),
        biomechanics_sync_method=metadata.get("biomechanics_sync_method"),
        biomechanics_sample_rate=metadata.get("biomechanics_sample_rate"),
        biomechanics_notes=metadata.get("biomechanics_notes"),
        **{k: v for k, v in kwargs.items() if k in AudioProcessing.__dataclass_fields__},
    )


def create_biomechanics_record_from_data(
    biomechanics_file: Path,
    recordings: list,
    sheet_name: Optional[str],
    maneuver: str,
    biomechanics_type: Optional[str],
    knee: str,
    biomechanics_sync_method: Optional[str],
    biomechanics_sample_rate: Optional[float],
    study_id: int,
    study: Optional[str] = None,
    **kwargs,
) -> BiomechanicsImport:
    """Create BiomechanicsImport record from data and metadata."""
    study_name = study or _parse_study_from_file_name(biomechanics_file.stem) or "AOA"
    num_sub_recordings = len(recordings)
    num_data_points = 0
    duration_seconds = 0.0
    num_passes = 0

    try:
        for rec in recordings:
            if hasattr(rec, "data") and rec.data is not None:
                data_obj = rec.data
                df = data_obj.data if hasattr(data_obj, "data") else data_obj
                num_data_points += len(df)
                if "TIME" in df.columns and len(df) > 1:
                    duration_seconds = max(duration_seconds, (df["TIME"].iloc[-1] - df["TIME"].iloc[0]).total_seconds())
            if hasattr(rec, "speed") and rec.speed:
                num_passes += 1
    except Exception:
        pass

    return BiomechanicsImport(
        study=study_name,
        study_id=int(study_id),
        biomechanics_file=str(biomechanics_file),
        sheet_name=sheet_name,
        biomechanics_type=biomechanics_type or "Motion Analysis",
        knee=knee.lower(),
        maneuver=_normalize_maneuver(maneuver),
        biomechanics_sync_method=biomechanics_sync_method or "stomp",
        biomechanics_sample_rate=float(biomechanics_sample_rate or 0.0),
        num_sub_recordings=num_sub_recordings,
        duration_seconds=float(duration_seconds),
        num_data_points=int(num_data_points),
        num_passes=int(num_passes),
        processing_date=datetime.now(),
        processing_status="success",
        **{k: v for k, v in kwargs.items() if k in BiomechanicsImport.__dataclass_fields__},
    )


def create_sync_record_from_data(
    sync_file_name: str,
    synced_df: pd.DataFrame,
    audio_stomp_time: Optional[float],
    knee_side: str,  # Required: "left" or "right"
    maneuver: str = "walk",  # Required: DB maneuver code ("fe", "sts", "walk")
    bio_left_stomp_time: Optional[float] = None,
    bio_right_stomp_time: Optional[float] = None,
    pass_number: Optional[int] = None,
    speed: Optional[str] = None,
    detection_results: Optional[Dict] = None,
    audio_record: Optional[AudioProcessing] = None,
    metadata: Optional[Dict] = None,
    study: Optional[str] = None,
    study_id: Optional[int] = None,
    **kwargs,
) -> Synchronization:
    """Create Synchronization record from sync data and detection results.

    Args:
        sync_file_name: Name of synchronization file
        synced_df: Synchronized dataframe
        audio_stomp_time: Audio stomp time
        knee_side: Side of knee being processed ("left" or "right") - REQUIRED
        bio_left_stomp_time: Left biomechanics stomp time
        bio_right_stomp_time: Right biomechanics stomp time
        ... other parameters
    """
    detection_results = detection_results or {}
    metadata = metadata or {}

    # Validate knee_side (normalize to lowercase)
    knee_side_lower = knee_side.lower()
    if knee_side_lower not in ["left", "right"]:
        raise ValueError(f"knee_side must be 'left' or 'right', got: {knee_side}")

    speed_value = speed
    if speed_value and speed_value.lower() == "normal":
        speed_value = "medium"

    consensus_methods = detection_results.get("consensus_methods")
    if isinstance(consensus_methods, str):
        consensus_methods = [m.strip() for m in consensus_methods.split(",") if m.strip()]

    method_times = []
    if consensus_methods:
        for method in consensus_methods:
            time_key = f"{method}_time"
            if detection_results.get(time_key) is not None:
                method_times.append(detection_results[time_key])
    method_agreement_span = (
        max(method_times) - min(method_times) if len(method_times) >= 2 else (0.0 if len(method_times) == 1 else None)
    )

    audio_processing_id = metadata.get("audio_processing_id") or kwargs.get("audio_processing_id") or 1
    biomechanics_import_id = metadata.get("biomechanics_import_id") or kwargs.get("biomechanics_import_id") or 1

    study_name = study or metadata.get("study") or (audio_record.study if audio_record else None) or "AOA"
    participant_number = study_id or metadata.get("study_id") or (audio_record.study_id if audio_record else 1)

    # Calculate bio_sync_offset based on knee_side
    bio_stomp_time = bio_left_stomp_time if knee_side_lower == "left" else bio_right_stomp_time
    bio_sync_offset = (
        _timedelta_to_seconds(audio_stomp_time - bio_stomp_time)
        if audio_stomp_time is not None and bio_stomp_time is not None
        else None
    )

    return Synchronization(
        study=study_name,
        study_id=int(participant_number),
        audio_processing_id=int(audio_processing_id),
        biomechanics_import_id=int(biomechanics_import_id),
        pass_number=pass_number,
        speed=speed_value,
        knee=knee_side_lower,
        maneuver=_normalize_maneuver(maneuver),
        # Biomechanics sync times (biomechanics is synced to audio: audio t=0 = sync t=0)
        bio_left_sync_time=_timedelta_to_seconds(bio_left_stomp_time),
        bio_right_sync_time=_timedelta_to_seconds(bio_right_stomp_time),
        bio_sync_offset=bio_sync_offset,
        aligned_sync_time=_timedelta_to_seconds(detection_results.get("consensus_time") or audio_stomp_time),
        # Sync method details
        sync_method=detection_results.get("selected_stomp_method") if detection_results.get("selected_stomp_method") else ("biomechanics" if detection_results.get("audio_stomp_method") else "consensus"),
        consensus_methods=", ".join(consensus_methods) if consensus_methods else None,
        consensus_time=_timedelta_to_seconds(detection_results.get("consensus_time")),
        rms_time=_timedelta_to_seconds(detection_results.get("rms_time")),
        onset_time=_timedelta_to_seconds(detection_results.get("onset_time")),
        freq_time=_timedelta_to_seconds(detection_results.get("freq_time")),
        method_agreement_span=method_agreement_span,
        # Detection methods
        stomp_detection_methods=detection_results.get("stomp_detection_methods"),
        selected_stomp_method=detection_results.get("selected_stomp_method"),
        # Biomechanics-based sync times
        bio_selected_sync_time=_timedelta_to_seconds(detection_results.get("bio_selected_time") or detection_results.get("selected_time")),
        contra_bio_selected_sync_time=_timedelta_to_seconds(detection_results.get("contra_bio_selected_time") or detection_results.get("contra_selected_time")),
        # Audio sync times (optional - mic on to participant stopping)
        audio_sync_time_left=_timedelta_to_seconds(detection_results.get("audio_sync_time_left")),
        audio_sync_time_right=_timedelta_to_seconds(detection_results.get("audio_sync_time_right")),
        audio_sync_offset=_timedelta_to_seconds(detection_results.get("audio_sync_offset")),
        # Audio-based sync times (different from bio-based) - renamed for consistency
        audio_selected_sync_time=_timedelta_to_seconds(detection_results.get("audio_selected_sync_time") or detection_results.get("selected_audio_sync_time")),
        contra_audio_selected_sync_time=_timedelta_to_seconds(detection_results.get("contra_audio_selected_sync_time") or detection_results.get("contra_selected_audio_sync_time")),
        # File and processing
        sync_file_name=sync_file_name,
        sync_file_path=str(kwargs.get("sync_file_path")) if kwargs.get("sync_file_path") else None,
        sync_duration=_timedelta_to_seconds(synced_df["tt"].iloc[-1]) if "tt" in synced_df.columns and len(synced_df) > 0 else None,
        processing_date=datetime.now(),
        processing_status="success",
        total_cycles_extracted=0,
        clean_cycles=0,
        outlier_cycles=0,
        # Cycle statistics - will be populated during cycle extraction
        mean_cycle_duration_s=None,
        median_cycle_duration_s=None,
        min_cycle_duration_s=None,
        max_cycle_duration_s=None,
        **{k: v for k, v in kwargs.items() if k in Synchronization.__dataclass_fields__},
    )



def _timedelta_to_seconds(value) -> Optional[float]:
    """Convert various time formats to seconds (float).

    Handles:
    - pd.Timedelta: converts to seconds
    - datetime.timedelta: converts to seconds
    - float/int: returns as-is
    - str: attempts to parse as float
    - None: returns None

    Returns None for unparseable values.
    """
    if value is None:
        return None

    if isinstance(value, pd.Timedelta):
        return value.total_seconds()

    # Also handle standard library timedelta
    if hasattr(value, 'total_seconds'):
        return value.total_seconds()

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None

    return None