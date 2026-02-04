"""Integration between processing_log and database persistence.

Provides hooks to optionally save records created by the processing log
to the PostgreSQL database via the persistence layer.
"""

import logging
from typing import Optional

from src.orchestration.database_persistence import (
    OrchestrationDatabasePersistence,
    RecordTracker,
)
from src.orchestration.processing_log import (
    create_audio_record_from_data,
    create_biomechanics_record_from_data,
    create_cycles_record_from_data,
    create_sync_record_from_data,
)

logger = logging.getLogger(__name__)


class PersistentProcessingLog:
    """Wraps processing log record creation with optional database persistence.

    Usage:
        persistence = OrchestrationDatabasePersistence(session)
        tracker = RecordTracker()
        ppl = PersistentProcessingLog(persistence, tracker)

        audio_record = ppl.create_audio_record_from_data(...)
        audio_id = tracker.get_audio_processing()  # For FK references
    """

    def __init__(
        self,
        persistence: Optional[OrchestrationDatabasePersistence] = None,
        tracker: Optional[RecordTracker] = None,
    ):
        """Initialize persistent processing log.

        Args:
            persistence: Optional database persistence layer
            tracker: Optional record tracker to maintain FK relationships
        """
        self.persistence = persistence
        self.tracker = tracker

    def create_audio_record_from_data(
        self,
        audio_file_name: str,
        device_serial: str,
        firmware_version: int,
        file_time,
        file_size_mb: float,
        recording_date,
        recording_time,
        knee: str,
        maneuver: str,
        num_channels: int,
        sample_rate: float,
        mic_positions: dict,
        study: str,
        study_id: int,
        pkl_file_path: Optional[str] = None,
        biomechanics_file: Optional[str] = None,
        biomechanics_import_id: Optional[int] = None,
        **kwargs,
    ):
        """Create and optionally persist audio processing record.

        Returns audio record and saves to DB if persistence enabled.
        """
        # Create the record using the standard function
        audio_record = create_audio_record_from_data(
            audio_file_name=audio_file_name,
            device_serial=device_serial,
            firmware_version=firmware_version,
            file_time=file_time,
            file_size_mb=file_size_mb,
            recording_date=recording_date,
            recording_time=recording_time,
            knee=knee,
            maneuver=maneuver,
            num_channels=num_channels,
            sample_rate=sample_rate,
            mic_positions=mic_positions,
            study=study,
            study_id=study_id,
            **kwargs,
        )

        # Optionally persist to database
        if self.persistence is not None:
            audio_id = self.persistence.save_audio_processing(
                audio_record,
                pkl_file_path=pkl_file_path,
                biomechanics_import_id=biomechanics_import_id,
            )
            if audio_id is not None and self.tracker is not None:
                self.tracker.set_audio_processing(audio_id)

        return audio_record

    def create_biomechanics_record_from_data(
        self,
        biomechanics_file: str,
        sheet_name: Optional[str],
        biomechanics_type: str,
        knee: str,
        maneuver: str,
        pass_number: Optional[int],
        speed: Optional[str],
        biomechanics_sync_method: str,
        biomechanics_sample_rate: float,
        study: str,
        study_id: int,
        num_sub_recordings: int,
        duration_seconds: float,
        num_data_points: int,
        num_passes: int,
        audio_processing_id: Optional[int] = None,
        **kwargs,
    ):
        """Create and optionally persist biomechanics import record.

        Returns biomechanics record and saves to DB if persistence enabled.
        """
        # Create the record using the standard function
        biomech_record = create_biomechanics_record_from_data(
            biomechanics_file=biomechanics_file,
            sheet_name=sheet_name,
            biomechanics_type=biomechanics_type,
            knee=knee,
            maneuver=maneuver,
            pass_number=pass_number,
            speed=speed,
            biomechanics_sync_method=biomechanics_sync_method,
            biomechanics_sample_rate=biomechanics_sample_rate,
            study=study,
            study_id=study_id,
            num_sub_recordings=num_sub_recordings,
            duration_seconds=duration_seconds,
            num_data_points=num_data_points,
            num_passes=num_passes,
            **kwargs,
        )

        # Optionally persist to database
        if self.persistence is not None:
            biomech_id = self.persistence.save_biomechanics_import(
                biomech_record,
                audio_processing_id=audio_processing_id,
            )
            if biomech_id is not None and self.tracker is not None:
                self.tracker.set_biomechanics_import(biomech_id)

        return biomech_record

    def create_sync_record_from_data(
        self,
        audio_processing_id: int,
        biomechanics_import_id: int,
        sync_file_path: Optional[str] = None,
        pass_number: Optional[int] = None,
        **kwargs,
    ):
        """Create and optionally persist synchronization record.

        Returns sync record and saves to DB if persistence enabled.
        """
        # Create the record using the standard function
        sync_record = create_sync_record_from_data(**kwargs)

        # Optionally persist to database
        if self.persistence is not None:
            sync_id = self.persistence.save_synchronization(
                sync_record,
                audio_processing_id=audio_processing_id,
                biomechanics_import_id=biomechanics_import_id,
                sync_file_path=sync_file_path,
            )
            if sync_id is not None and self.tracker is not None:
                self.tracker.set_synchronization(pass_number, sync_id)

        return sync_record

    def create_cycles_record_from_data(
        self,
        audio_processing_id: int,
        biomechanics_import_id: Optional[int] = None,
        synchronization_id: Optional[int] = None,
        cycles_file_path: Optional[str] = None,
        **kwargs,
    ):
        """Create and optionally persist movement cycle record.

        Returns cycle record and saves to DB if persistence enabled.
        """
        # Create the record using the standard function
        cycle_record = create_cycles_record_from_data(**kwargs)

        # Optionally persist to database
        if self.persistence is not None:
            cycle_id = self.persistence.save_movement_cycle(
                cycle_record,
                audio_processing_id=audio_processing_id,
                biomechanics_import_id=biomechanics_import_id,
                synchronization_id=synchronization_id,
                cycles_file_path=cycles_file_path,
            )
            if cycle_id is not None and self.tracker is not None:
                self.tracker.add_movement_cycle(cycle_id)

        return cycle_record
