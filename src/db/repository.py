"""Repository layer for database operations.

Provides high-level CRUD operations for all metadata entities,
abstracting away SQLAlchemy session management and query construction.
"""

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import and_, select
from sqlalchemy.orm import Session, joinedload

from src.db.models import (
    AudioProcessingRecord,
    BiomechanicsImportRecord,
    MovementCycleRecord,
    ParticipantRecord,
    StudyRecord,
    SynchronizationRecord,
)
from src.metadata import (
    AudioProcessing,
    BiomechanicsImport,
    MovementCycle,
    Synchronization,
)


class Repository:
    """Repository for metadata CRUD operations."""

    def __init__(self, session: Session):
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy session
        """
        self.session = session

    @property
    def _session(self) -> Session:
        """Compatibility accessor for tests expecting _session."""
        return self.session

    @property
    def _models(self) -> dict:
        """Compatibility accessor for tests expecting model mapping."""
        return {
            "StudyRecord": StudyRecord,
            "ParticipantRecord": ParticipantRecord,
            "AudioProcessingRecord": AudioProcessingRecord,
            "BiomechanicsImportRecord": BiomechanicsImportRecord,
            "SynchronizationRecord": SynchronizationRecord,
            "MovementCycleRecord": MovementCycleRecord,
        }

    # ========================================================================
    # Study operations
    # ========================================================================

    def get_or_create_study(self, study_name: str) -> StudyRecord:
        """Get existing study or create new one.

        Args:
            study_name: Study name (AOA, preOA, SMoCK)

        Returns:
            Study record
        """
        study = self.session.execute(
            select(StudyRecord).where(StudyRecord.name == study_name)
        ).scalar_one_or_none()

        if study is None:
            study = StudyRecord(name=study_name)
            self.session.add(study)
            self.session.flush()

        return study

    # ========================================================================
    # Participant operations
    # ========================================================================

    def get_or_create_participant(
        self, study_name: str, participant_number: int
    ) -> ParticipantRecord:
        """Get existing participant or create new one.

        Args:
            study_name: Study name (AOA, preOA, SMoCK)
            participant_number: Participant ID within study (study_id column)

        Returns:
            Participant record
        """
        study = self.get_or_create_study(study_name)

        participant = self.session.execute(
            select(ParticipantRecord).where(
                and_(
                    ParticipantRecord.study_participant_id == study.id,
                    ParticipantRecord.study_id == participant_number,
                )
            )
        ).scalar_one_or_none()

        if participant is None:
            participant = ParticipantRecord(
                study_participant_id=study.id, study_id=participant_number
            )
            self.session.add(participant)
            self.session.flush()

        return participant

    # ========================================================================
    # Audio processing operations
    # ========================================================================

    def save_audio_processing(
        self,
        audio: AudioProcessing,
        pkl_file_path: Optional[str] = None,
        biomechanics_import_id: Optional[int] = None,
    ) -> AudioProcessingRecord:
        """Save or update audio processing record.

        Args:
            audio: AudioProcessing Pydantic model (audio-specific fields only)
            pkl_file_path: Optional path to .pkl file on disk
            biomechanics_import_id: Optional FK to BiomechanicsImportRecord
                (the biomechanics recorded at the same time as this audio)

        Returns:
            Audio processing record
        """
        participant = self.get_or_create_participant(audio.study, audio.study_id)

        # Check if record already exists
        existing = self.session.execute(
            select(AudioProcessingRecord).where(
                and_(
                    AudioProcessingRecord.participant_id == participant.id,
                    AudioProcessingRecord.audio_file_name == audio.audio_file_name,
                    AudioProcessingRecord.knee == audio.knee,
                    AudioProcessingRecord.maneuver == audio.maneuver,
                )
            )
        ).scalar_one_or_none()

        if existing:
            # Update existing record
            self._update_audio_processing_record(existing, audio, pkl_file_path)
            # Update FK if provided
            if biomechanics_import_id is not None:
                existing.biomechanics_import_id = biomechanics_import_id
            record = existing
        else:
            # Create new record
            record = self._create_audio_processing_record(
                participant.id, audio, pkl_file_path, biomechanics_import_id
            )
            self.session.add(record)

        self.session.flush()
        return record

    def _create_audio_processing_record(
        self,
        participant_id: int,
        audio: AudioProcessing,
        pkl_file_path: Optional[str],
        biomechanics_import_id: Optional[int] = None,
    ) -> AudioProcessingRecord:
        """Create new audio processing record from Pydantic model.

        Args:
            participant_id: ID of participant
            audio: AudioProcessing Pydantic model (audio-specific fields only)
            pkl_file_path: Optional path to .pkl file on disk
            biomechanics_import_id: Optional FK to BiomechanicsImportRecord (recorded simultaneously)

        Returns:
            New AudioProcessingRecord
        """
        return AudioProcessingRecord(
            participant_id=participant_id,
            biomechanics_import_id=biomechanics_import_id,
            audio_file_name=audio.audio_file_name,
            device_serial=audio.device_serial,
            firmware_version=audio.firmware_version,
            file_time=audio.file_time,
            file_size_mb=audio.file_size_mb,
            recording_date=audio.recording_date,
            recording_time=audio.recording_time,
            recording_timezone=audio.recording_timezone,
            knee=audio.knee,
            maneuver=audio.maneuver,
            num_channels=audio.num_channels,
            sample_rate=audio.sample_rate,
            mic_1_position=audio.mic_1_position,
            mic_2_position=audio.mic_2_position,
            mic_3_position=audio.mic_3_position,
            mic_4_position=audio.mic_4_position,
            mic_1_notes=audio.mic_1_notes,
            mic_2_notes=audio.mic_2_notes,
            mic_3_notes=audio.mic_3_notes,
            mic_4_notes=audio.mic_4_notes,
            notes=audio.notes,
            pkl_file_path=pkl_file_path,
            audio_qc_version=audio.audio_qc_version,
            audio_qc_fail=audio.qc_not_passed,
            audio_qc_fail_ch1=audio.qc_not_passed_mic_1,
            audio_qc_fail_ch2=audio.qc_not_passed_mic_2,
            audio_qc_fail_ch3=audio.qc_not_passed_mic_3,
            audio_qc_fail_ch4=audio.qc_not_passed_mic_4,
            # ===== QC Fail Segments =====
            qc_fail_segments=audio.qc_fail_segments,
            qc_fail_segments_ch1=audio.qc_fail_segments_ch1,
            qc_fail_segments_ch2=audio.qc_fail_segments_ch2,
            qc_fail_segments_ch3=audio.qc_fail_segments_ch3,
            qc_fail_segments_ch4=audio.qc_fail_segments_ch4,
            # ===== Signal Dropout QC =====
            qc_signal_dropout=audio.qc_signal_dropout,
            qc_signal_dropout_segments=audio.qc_signal_dropout_segments,
            qc_signal_dropout_ch1=audio.qc_signal_dropout_ch1,
            qc_signal_dropout_segments_ch1=audio.qc_signal_dropout_segments_ch1,
            qc_signal_dropout_ch2=audio.qc_signal_dropout_ch2,
            qc_signal_dropout_segments_ch2=audio.qc_signal_dropout_segments_ch2,
            qc_signal_dropout_ch3=audio.qc_signal_dropout_ch3,
            qc_signal_dropout_segments_ch3=audio.qc_signal_dropout_segments_ch3,
            qc_signal_dropout_ch4=audio.qc_signal_dropout_ch4,
            qc_signal_dropout_segments_ch4=audio.qc_signal_dropout_segments_ch4,
            # ===== Artifact QC =====
            qc_artifact=audio.qc_artifact,
            qc_artifact_type=audio.qc_artifact_type,
            qc_artifact_segments=audio.qc_artifact_segments,
            qc_artifact_ch1=audio.qc_artifact_ch1,
            qc_artifact_type_ch1=audio.qc_artifact_type_ch1,
            qc_artifact_segments_ch1=audio.qc_artifact_segments_ch1,
            qc_artifact_ch2=audio.qc_artifact_ch2,
            qc_artifact_type_ch2=audio.qc_artifact_type_ch2,
            qc_artifact_segments_ch2=audio.qc_artifact_segments_ch2,
            qc_artifact_ch3=audio.qc_artifact_ch3,
            qc_artifact_type_ch3=audio.qc_artifact_type_ch3,
            qc_artifact_segments_ch3=audio.qc_artifact_segments_ch3,
            qc_artifact_ch4=audio.qc_artifact_ch4,
            qc_artifact_type_ch4=audio.qc_artifact_type_ch4,
            qc_artifact_segments_ch4=audio.qc_artifact_segments_ch4,
            processing_date=audio.processing_date,
            processing_status=audio.processing_status,
            error_message=audio.error_message,
            duration_seconds=audio.duration_seconds,
        )

    def _update_audio_processing_record(
        self,
        record: AudioProcessingRecord,
        audio: AudioProcessing,
        pkl_file_path: Optional[str],
    ) -> None:
        """Update existing audio processing record.

        Args:
            record: Existing AudioProcessingRecord
            audio: AudioProcessing Pydantic model (audio-specific fields only)
            pkl_file_path: Optional new path to .pkl file
        """
        record.file_size_mb = audio.file_size_mb
        record.device_serial = audio.device_serial
        record.firmware_version = audio.firmware_version
        record.recording_timezone = audio.recording_timezone
        record.mic_1_notes = audio.mic_1_notes
        record.mic_2_notes = audio.mic_2_notes
        record.mic_3_notes = audio.mic_3_notes
        record.mic_4_notes = audio.mic_4_notes
        record.notes = audio.notes
        record.pkl_file_path = pkl_file_path or record.pkl_file_path
        record.audio_qc_version = audio.audio_qc_version
        record.audio_qc_fail = audio.qc_not_passed
        record.audio_qc_fail_ch1 = audio.qc_not_passed_mic_1
        record.audio_qc_fail_ch2 = audio.qc_not_passed_mic_2
        record.audio_qc_fail_ch3 = audio.qc_not_passed_mic_3
        record.audio_qc_fail_ch4 = audio.qc_not_passed_mic_4
        # ===== QC Fail Segments =====
        record.qc_fail_segments = audio.qc_fail_segments
        record.qc_fail_segments_ch1 = audio.qc_fail_segments_ch1
        record.qc_fail_segments_ch2 = audio.qc_fail_segments_ch2
        record.qc_fail_segments_ch3 = audio.qc_fail_segments_ch3
        record.qc_fail_segments_ch4 = audio.qc_fail_segments_ch4
        # ===== Signal Dropout QC =====
        record.qc_signal_dropout = audio.qc_signal_dropout
        record.qc_signal_dropout_segments = audio.qc_signal_dropout_segments
        record.qc_signal_dropout_ch1 = audio.qc_signal_dropout_ch1
        record.qc_signal_dropout_segments_ch1 = audio.qc_signal_dropout_segments_ch1
        record.qc_signal_dropout_ch2 = audio.qc_signal_dropout_ch2
        record.qc_signal_dropout_segments_ch2 = audio.qc_signal_dropout_segments_ch2
        record.qc_signal_dropout_ch3 = audio.qc_signal_dropout_ch3
        record.qc_signal_dropout_segments_ch3 = audio.qc_signal_dropout_segments_ch3
        record.qc_signal_dropout_ch4 = audio.qc_signal_dropout_ch4
        record.qc_signal_dropout_segments_ch4 = audio.qc_signal_dropout_segments_ch4
        # ===== Artifact QC =====
        record.qc_artifact = audio.qc_artifact
        record.qc_artifact_type = audio.qc_artifact_type
        record.qc_artifact_segments = audio.qc_artifact_segments
        record.qc_artifact_ch1 = audio.qc_artifact_ch1
        record.qc_artifact_type_ch1 = audio.qc_artifact_type_ch1
        record.qc_artifact_segments_ch1 = audio.qc_artifact_segments_ch1
        record.qc_artifact_ch2 = audio.qc_artifact_ch2
        record.qc_artifact_type_ch2 = audio.qc_artifact_type_ch2
        record.qc_artifact_segments_ch2 = audio.qc_artifact_segments_ch2
        record.qc_artifact_ch3 = audio.qc_artifact_ch3
        record.qc_artifact_type_ch3 = audio.qc_artifact_type_ch3
        record.qc_artifact_segments_ch3 = audio.qc_artifact_segments_ch3
        record.qc_artifact_ch4 = audio.qc_artifact_ch4
        record.qc_artifact_type_ch4 = audio.qc_artifact_type_ch4
        record.qc_artifact_segments_ch4 = audio.qc_artifact_segments_ch4
        record.processing_date = audio.processing_date
        record.processing_status = audio.processing_status
        record.error_message = audio.error_message
        record.duration_seconds = audio.duration_seconds
        record.updated_at = datetime.now(timezone.utc)

    # ========================================================================
    # Biomechanics import operations
    # ========================================================================

    def save_biomechanics_import(
        self,
        biomech: BiomechanicsImport,
        audio_processing_id: Optional[int] = None,
    ) -> BiomechanicsImportRecord:
        """Save or update biomechanics import record.

        Args:
            biomech: BiomechanicsImport Pydantic model (biomechanics-specific fields only)
            audio_processing_id: Optional FK to AudioProcessingRecord
                (the audio recorded at the same time as this biomechanics)

        Returns:
            Biomechanics import record
        """
        participant = self.get_or_create_participant(biomech.study, biomech.study_id)

        # Check if record already exists
        existing = self.session.execute(
            select(BiomechanicsImportRecord).where(
                and_(
                    BiomechanicsImportRecord.participant_id == participant.id,
                    BiomechanicsImportRecord.biomechanics_file == biomech.biomechanics_file,
                    BiomechanicsImportRecord.knee == biomech.knee,
                    BiomechanicsImportRecord.maneuver == biomech.maneuver,
                )
            )
        ).scalar_one_or_none()

        if existing:
            # Update existing record
            existing.biomechanics_qc_fail = biomech.biomechanics_qc_fail
            existing.biomechanics_qc_notes = biomech.biomechanics_qc_notes
            existing.processing_date = biomech.processing_date
            existing.processing_status = biomech.processing_status
            if audio_processing_id is not None:
                existing.audio_processing_id = audio_processing_id
            existing.updated_at = datetime.now(timezone.utc)
            record = existing
        else:
            # Create new record
            record = BiomechanicsImportRecord(
                participant_id=participant.id,
                biomechanics_file=biomech.biomechanics_file,
                sheet_name=biomech.sheet_name,
                biomechanics_type=biomech.biomechanics_type,
                knee=biomech.knee,
                maneuver=biomech.maneuver,
                biomechanics_sync_method=biomech.biomechanics_sync_method,
                biomechanics_sample_rate=biomech.biomechanics_sample_rate,
                biomechanics_notes=biomech.biomechanics_notes,
                num_sub_recordings=biomech.num_sub_recordings,
                duration_seconds=biomech.duration_seconds,
                num_data_points=biomech.num_data_points,
                num_passes=biomech.num_passes,
                biomech_qc_version=biomech.biomech_qc_version,
                biomechanics_qc_fail=biomech.biomechanics_qc_fail,
                biomechanics_qc_notes=biomech.biomechanics_qc_notes,
                processing_date=biomech.processing_date,
                processing_status=biomech.processing_status,
                audio_processing_id=audio_processing_id,
            )
            self.session.add(record)

        self.session.flush()
        return record

    # ========================================================================
    # Synchronization operations
    # ========================================================================

    def save_synchronization(
        self,
        sync: Synchronization,
        audio_processing_id: int,
        biomechanics_import_id: int,
        sync_file_path: Optional[str] = None,
    ) -> SynchronizationRecord:
        """Save or update synchronization record.

        Args:
            sync: Synchronization Pydantic model (sync-specific fields only)
            audio_processing_id: FK to AudioProcessingRecord
            biomechanics_import_id: FK to BiomechanicsImportRecord
            sync_file_path: Optional path to sync .pkl file on disk

        Returns:
            Synchronization record
        """
        participant = self.get_or_create_participant(sync.study, sync.study_id)

        # Check if record already exists by sync file name (unique constraint)
        existing = None
        if sync.sync_file_name:
            existing = self.session.execute(
                select(SynchronizationRecord).where(
                    and_(
                        SynchronizationRecord.participant_id == participant.id,
                        SynchronizationRecord.sync_file_name == sync.sync_file_name,
                    )
                )
            ).scalar_one_or_none()

        if existing:
            # Update existing record
            self._update_synchronization_record(existing, sync, sync_file_path)
            record = existing
        else:
            # Create new record
            record = self._create_synchronization_record(
                participant.id, sync, audio_processing_id, biomechanics_import_id, sync_file_path
            )
            self.session.add(record)

        self.session.flush()
        return record

    def _create_synchronization_record(
        self,
        participant_id: int,
        sync: Synchronization,
        audio_processing_id: int,
        biomechanics_import_id: int,
        sync_file_path: Optional[str],
    ) -> SynchronizationRecord:
        """Create new synchronization record from Pydantic model.

        Args:
            participant_id: ID of participant
            sync: Synchronization Pydantic model (sync-specific fields only)
            audio_processing_id: FK to AudioProcessingRecord
            biomechanics_import_id: FK to BiomechanicsImportRecord
            sync_file_path: Optional path to sync .pkl file

        Returns:
            New SynchronizationRecord
        """
        return SynchronizationRecord(
            participant_id=participant_id,
            audio_processing_id=audio_processing_id,
            biomechanics_import_id=biomechanics_import_id,
            pass_number=sync.pass_number,
            speed=sync.speed,
            audio_sync_time=sync.audio_sync_time,
            bio_left_sync_time=sync.bio_left_sync_time,
            bio_right_sync_time=sync.bio_right_sync_time,
            sync_offset=sync.sync_offset,
            aligned_audio_sync_time=sync.aligned_audio_sync_time,
            aligned_biomechanics_sync_time=sync.aligned_biomechanics_sync_time,
            sync_method=sync.sync_method,
            consensus_methods=sync.consensus_methods,
            consensus_time=getattr(sync, "consensus_time", None),
            rms_time=sync.rms_time,
            onset_time=sync.onset_time,
            freq_time=sync.freq_time,
            selected_audio_sync_time=getattr(sync, 'selected_audio_sync_time', None),
            contra_selected_audio_sync_time=getattr(sync, 'contra_selected_audio_sync_time', None),
            audio_visual_sync_time=getattr(sync, 'audio_visual_sync_time', None),
            audio_visual_sync_time_contralateral=getattr(sync, 'audio_visual_sync_time_contralateral', None),
            audio_stomp_method=getattr(sync, 'audio_stomp_method', None),
            sync_file_name=sync.sync_file_name,
            sync_file_path=sync_file_path,
            sync_duration=sync.sync_duration,
            total_cycles_extracted=getattr(sync, "total_cycles_extracted", None),
            clean_cycles=getattr(sync, "clean_cycles", None),
            outlier_cycles=getattr(sync, "outlier_cycles", None),
            mean_cycle_duration_s=getattr(sync, "mean_cycle_duration_s", None),
            median_cycle_duration_s=getattr(sync, "median_cycle_duration_s", None),
            min_cycle_duration_s=getattr(sync, "min_cycle_duration_s", None),
            max_cycle_duration_s=getattr(sync, "max_cycle_duration_s", None),
            method_agreement_span=getattr(sync, "method_agreement_span", None),
            sync_qc_fail=sync.sync_qc_fail,
            processing_date=sync.processing_date,
            processing_status=sync.processing_status,
            error_message=sync.error_message,
        )

    def _update_synchronization_record(
        self,
        record: SynchronizationRecord,
        sync: Synchronization,
        sync_file_path: Optional[str],
    ) -> None:
        """Update existing synchronization record.

        Args:
            record: Existing SynchronizationRecord
            sync: Synchronization Pydantic model (sync-specific fields only)
            sync_file_path: Optional new path to sync .pkl file
        """
        record.pass_number = sync.pass_number
        record.speed = sync.speed
        record.audio_sync_time = sync.audio_sync_time
        record.bio_left_sync_time = sync.bio_left_sync_time
        record.bio_right_sync_time = sync.bio_right_sync_time
        record.sync_offset = sync.sync_offset
        record.aligned_audio_sync_time = sync.aligned_audio_sync_time
        record.aligned_biomechanics_sync_time = sync.aligned_biomechanics_sync_time
        record.sync_method = sync.sync_method
        record.consensus_methods = sync.consensus_methods
        record.consensus_time = getattr(sync, "consensus_time", None)
        record.rms_time = sync.rms_time
        record.onset_time = sync.onset_time
        record.freq_time = sync.freq_time
        if hasattr(sync, 'selected_audio_sync_time'):
            record.selected_audio_sync_time = sync.selected_audio_sync_time
        if hasattr(sync, 'contra_selected_audio_sync_time'):
            record.contra_selected_audio_sync_time = sync.contra_selected_audio_sync_time
        if hasattr(sync, 'audio_visual_sync_time'):
            record.audio_visual_sync_time = sync.audio_visual_sync_time
        if hasattr(sync, 'audio_visual_sync_time_contralateral'):
            record.audio_visual_sync_time_contralateral = sync.audio_visual_sync_time_contralateral
        if hasattr(sync, 'audio_stomp_method'):
            record.audio_stomp_method = sync.audio_stomp_method
        record.sync_file_name = sync.sync_file_name
        record.sync_file_path = sync_file_path or record.sync_file_path
        record.sync_duration = sync.sync_duration
        record.total_cycles_extracted = getattr(sync, "total_cycles_extracted", None)
        record.clean_cycles = getattr(sync, "clean_cycles", None)
        record.outlier_cycles = getattr(sync, "outlier_cycles", None)
        record.mean_cycle_duration_s = getattr(sync, "mean_cycle_duration_s", None)
        record.median_cycle_duration_s = getattr(sync, "median_cycle_duration_s", None)
        record.min_cycle_duration_s = getattr(sync, "min_cycle_duration_s", None)
        record.max_cycle_duration_s = getattr(sync, "max_cycle_duration_s", None)
        record.method_agreement_span = getattr(sync, "method_agreement_span", None)
        record.sync_qc_fail = sync.sync_qc_fail
        record.processing_date = sync.processing_date
        record.processing_status = sync.processing_status
        record.error_message = sync.error_message
        record.processing_date = sync.processing_date
        record.updated_at = datetime.now(timezone.utc)

    # ========================================================================
    # Movement cycle operations
    # ========================================================================

    def save_movement_cycle(
        self,
        cycle: MovementCycle,
        audio_processing_id: int,
        biomechanics_import_id: Optional[int] = None,
        synchronization_id: Optional[int] = None,
        cycles_file_path: Optional[str] = None,
    ) -> MovementCycleRecord:
        """Save or update movement cycle record.

        Args:
            cycle: MovementCycle Pydantic model (cycle-specific fields only)
            audio_processing_id: FK to AudioProcessingRecord (required)
            biomechanics_import_id: Optional FK to BiomechanicsImportRecord
            synchronization_id: Optional FK to SynchronizationRecord
            cycles_file_path: Optional path to cycles .pkl file on disk

        Returns:
            Movement cycle record
        """
        participant = self.get_or_create_participant(cycle.study, cycle.study_id)

        # Check if record already exists by cycle file name (unique constraint)
        existing = self.session.execute(
            select(MovementCycleRecord).where(
                and_(
                    MovementCycleRecord.participant_id == participant.id,
                    MovementCycleRecord.cycle_file == cycle.cycle_file,
                )
            )
        ).scalar_one_or_none()

        if existing:
            # Update existing record
            self._update_movement_cycle_record(
                existing, cycle, biomechanics_import_id, synchronization_id, cycles_file_path
            )
            record = existing
        else:
            # Create new record
            record = self._create_movement_cycle_record(
                participant.id,
                cycle,
                audio_processing_id,
                biomechanics_import_id,
                synchronization_id,
                cycles_file_path,
            )
            self.session.add(record)

        self.session.flush()
        return record

    def _create_movement_cycle_record(
        self,
        participant_id: int,
        cycle: MovementCycle,
        audio_processing_id: int,
        biomechanics_import_id: Optional[int],
        synchronization_id: Optional[int],
        cycles_file_path: Optional[str],
    ) -> MovementCycleRecord:
        """Create new movement cycle record from Pydantic model.

        Args:
            participant_id: ID of participant
            cycle: MovementCycle Pydantic model (cycle-specific fields only)
            audio_processing_id: FK to AudioProcessingRecord
            biomechanics_import_id: Optional FK to BiomechanicsImportRecord
            synchronization_id: Optional FK to SynchronizationRecord
            cycles_file_path: Optional path to cycles .pkl file

        Returns:
            New MovementCycleRecord
        """
        return MovementCycleRecord(
            participant_id=participant_id,
            audio_processing_id=audio_processing_id,
            biomechanics_import_id=biomechanics_import_id,
            synchronization_id=synchronization_id,
            cycle_file=cycle.cycle_file,
            cycle_index=cycle.cycle_index,
            is_outlier=cycle.is_outlier,
            start_time_s=cycle.start_time_s,
            end_time_s=cycle.end_time_s,
            duration_s=cycle.duration_s,
            audio_start_time=cycle.audio_start_time,
            audio_end_time=cycle.audio_end_time,
            bio_start_time=cycle.bio_start_time,
            bio_end_time=cycle.bio_end_time,
            biomechanics_qc_fail=cycle.biomechanics_qc_fail,
            sync_qc_fail=cycle.sync_qc_fail,
            cycle_file_path=cycles_file_path,
        )

    def _update_movement_cycle_record(
        self,
        record: MovementCycleRecord,
        cycle: MovementCycle,
        biomechanics_import_id: Optional[int],
        synchronization_id: Optional[int],
        cycles_file_path: Optional[str],
    ) -> None:
        """Update existing movement cycle record.

        Args:
            record: Existing MovementCycleRecord
            cycle: MovementCycle Pydantic model (cycle-specific fields only)
            biomechanics_import_id: Optional FK to BiomechanicsImportRecord
            synchronization_id: Optional FK to SynchronizationRecord
            cycles_file_path: Optional new path to cycles .pkl file
        """
        record.is_outlier = cycle.is_outlier
        record.start_time_s = cycle.start_time_s
        record.end_time_s = cycle.end_time_s
        record.duration_s = cycle.duration_s
        record.audio_start_time = cycle.audio_start_time
        record.audio_end_time = cycle.audio_end_time
        record.bio_start_time = cycle.bio_start_time
        record.bio_end_time = cycle.bio_end_time
        record.biomechanics_qc_fail = cycle.biomechanics_qc_fail
        record.sync_qc_fail = cycle.sync_qc_fail
        if cycles_file_path:
            record.cycle_file_path = cycles_file_path
        if biomechanics_import_id is not None:
            record.biomechanics_import_id = biomechanics_import_id
        if synchronization_id is not None:
            record.synchronization_id = synchronization_id
        record.updated_at = datetime.now(timezone.utc)

    # ========================================================================
    # Query operations
    # ========================================================================

    def get_audio_processing_records(
        self,
        study_name: Optional[str] = None,
        participant_number: Optional[int] = None,
        maneuver: Optional[str] = None,
        knee: Optional[str] = None,
    ) -> List[AudioProcessingRecord]:
        """Query audio processing records with filters.

        Args:
            study_name: Filter by study name
            participant_number: Filter by participant study_id
            maneuver: Filter by maneuver
            knee: Filter by knee

        Returns:
            List of matching audio processing records
        """
        query = select(AudioProcessingRecord).join(ParticipantRecord)

        if study_name:
            query = query.join(StudyRecord).where(StudyRecord.name == study_name)
        if participant_number is not None:
            query = query.where(ParticipantRecord.study_id == participant_number)
        if maneuver:
            query = query.where(AudioProcessingRecord.maneuver == maneuver)
        if knee:
            query = query.where(AudioProcessingRecord.knee == knee)

        return list(self.session.execute(query).scalars().all())

    def get_biomechanics_imports(
        self,
        study_name: Optional[str] = None,
        participant_number: Optional[int] = None,
        maneuver: Optional[str] = None,
        knee: Optional[str] = None,
    ) -> List[BiomechanicsImportRecord]:
        """Query biomechanics import records with filters.

        Args:
            study_name: Filter by study name
            participant_number: Filter by participant study_id
            maneuver: Filter by maneuver
            knee: Filter by knee

        Returns:
            List of matching biomechanics import records
        """
        query = select(BiomechanicsImportRecord).join(ParticipantRecord)

        if study_name:
            query = query.join(StudyRecord).where(StudyRecord.name == study_name)
        if participant_number is not None:
            query = query.where(ParticipantRecord.study_id == participant_number)
        if maneuver:
            query = query.where(BiomechanicsImportRecord.maneuver == maneuver)
        if knee:
            query = query.where(BiomechanicsImportRecord.knee == knee)

        return list(self.session.execute(query).scalars().all())

    def get_synchronization_records(
        self,
        study_name: Optional[str] = None,
        participant_number: Optional[int] = None,
        maneuver: Optional[str] = None,
        knee: Optional[str] = None,
    ) -> List[SynchronizationRecord]:
        """Query synchronization records with filters.

        Args:
            study_name: Filter by study name
            participant_number: Filter by participant study_id
            maneuver: Filter by maneuver
            knee: Filter by knee

        Returns:
            List of matching synchronization records
        """
        query = select(SynchronizationRecord).join(ParticipantRecord)

        if study_name:
            query = query.join(StudyRecord).where(StudyRecord.name == study_name)
        if participant_number is not None:
            query = query.where(ParticipantRecord.study_id == participant_number)
        if maneuver:
            query = query.where(SynchronizationRecord.maneuver == maneuver)
        if knee:
            query = query.where(SynchronizationRecord.knee == knee)

        return list(self.session.execute(query).scalars().all())

    def get_movement_cycle_records(
        self,
        study_name: Optional[str] = None,
        participant_number: Optional[int] = None,
        maneuver: Optional[str] = None,
        knee: Optional[str] = None,
    ) -> List[MovementCycleRecord]:
        """Query movement cycle records with filters.

        Args:
            study_name: Filter by study name
            participant_number: Filter by participant study_id
            maneuver: Filter by maneuver
            knee: Filter by knee

        Returns:
            List of matching movement cycle records
        """
        query = select(MovementCycleRecord).join(ParticipantRecord)

        if study_name:
            query = query.join(StudyRecord).where(StudyRecord.name == study_name)
        if participant_number is not None:
            query = query.where(ParticipantRecord.study_id == participant_number)
        if maneuver:
            query = query.where(MovementCycleRecord.maneuver == maneuver)
        if knee:
            query = query.where(MovementCycleRecord.knee == knee)

        return list(self.session.execute(query).scalars().all())
