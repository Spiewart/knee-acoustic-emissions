"""Repository layer for database operations.

Provides high-level CRUD operations for all metadata entities,
abstracting away SQLAlchemy session management and query construction.
"""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import and_, select
from sqlalchemy.orm import Session, joinedload

from src.db.models import (
    AudioProcessingRecord,
    BiomechanicsImportRecord,
    MovementCycleDetailRecord,
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
            participant_number: Participant ID within study

        Returns:
            Participant record
        """
        study = self.get_or_create_study(study_name)

        participant = self.session.execute(
            select(ParticipantRecord).where(
                and_(
                    ParticipantRecord.study_id == study.id,
                    ParticipantRecord.participant_number == participant_number,
                )
            )
        ).scalar_one_or_none()

        if participant is None:
            participant = ParticipantRecord(
                study_id=study.id, participant_number=participant_number
            )
            self.session.add(participant)
            self.session.flush()

        return participant

    # ========================================================================
    # Audio processing operations
    # ========================================================================

    def save_audio_processing(
        self, audio: AudioProcessing, pkl_file_path: Optional[str] = None
    ) -> AudioProcessingRecord:
        """Save or update audio processing record.

        Args:
            audio: AudioProcessing Pydantic model
            pkl_file_path: Optional path to .pkl file on disk

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
            record = existing
        else:
            # Create new record
            record = self._create_audio_processing_record(
                participant.id, audio, pkl_file_path
            )
            self.session.add(record)

        self.session.flush()
        return record

    def _create_audio_processing_record(
        self, participant_id: int, audio: AudioProcessing, pkl_file_path: Optional[str]
    ) -> AudioProcessingRecord:
        """Create new audio processing record from Pydantic model."""
        return AudioProcessingRecord(
            participant_id=participant_id,
            audio_file_name=audio.audio_file_name,
            device_serial=audio.device_serial,
            firmware_version=audio.firmware_version,
            file_time=audio.file_time,
            file_size_mb=audio.file_size_mb,
            recording_date=audio.recording_date,
            recording_time=audio.recording_time,
            knee=audio.knee,
            maneuver=audio.maneuver,
            pass_number=getattr(audio, 'pass_number', None),
            speed=getattr(audio, 'speed', None),
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
            linked_biomechanics=audio.linked_biomechanics,
            biomechanics_file=audio.biomechanics_file,
            biomechanics_type=audio.biomechanics_type,
            biomechanics_sync_method=audio.biomechanics_sync_method,
            biomechanics_sample_rate=audio.biomechanics_sample_rate,
            biomechanics_notes=audio.biomechanics_notes,
            pkl_file_path=pkl_file_path,
            audio_qc_version=audio.audio_qc_version,
            audio_qc_fail=audio.qc_not_passed,
            audio_qc_fail_ch1=audio.qc_not_passed_mic_1,
            audio_qc_fail_ch2=audio.qc_not_passed_mic_2,
            audio_qc_fail_ch3=audio.qc_not_passed_mic_3,
            audio_qc_fail_ch4=audio.qc_not_passed_mic_4,
            qc_artifact_type=audio.qc_artifact_type,
            qc_artifact_type_ch1=audio.qc_artifact_type_ch1,
            qc_artifact_type_ch2=audio.qc_artifact_type_ch2,
            qc_artifact_type_ch3=audio.qc_artifact_type_ch3,
            qc_artifact_type_ch4=audio.qc_artifact_type_ch4,
            processing_date=audio.processing_date,
        )

    def _update_audio_processing_record(
        self,
        record: AudioProcessingRecord,
        audio: AudioProcessing,
        pkl_file_path: Optional[str],
    ) -> None:
        """Update existing audio processing record."""
        record.file_size_mb = audio.file_size_mb
        record.pass_number = audio.pass_number
        record.speed = audio.speed
        record.mic_1_notes = audio.mic_1_notes
        record.mic_2_notes = audio.mic_2_notes
        record.mic_3_notes = audio.mic_3_notes
        record.mic_4_notes = audio.mic_4_notes
        record.notes = audio.notes
        record.biomechanics_notes = audio.biomechanics_notes
        record.pkl_file_path = pkl_file_path or record.pkl_file_path
        record.audio_qc_version = audio.audio_qc_version
        record.audio_qc_fail = audio.audio_qc_fail
        record.audio_qc_fail_ch1 = audio.audio_qc_fail_ch1
        record.audio_qc_fail_ch2 = audio.audio_qc_fail_ch2
        record.audio_qc_fail_ch3 = audio.audio_qc_fail_ch3
        record.audio_qc_fail_ch4 = audio.audio_qc_fail_ch4
        record.qc_artifact_type = audio.qc_artifact_type
        record.qc_artifact_type_ch1 = audio.qc_artifact_type_ch1
        record.qc_artifact_type_ch2 = audio.qc_artifact_type_ch2
        record.qc_artifact_type_ch3 = audio.qc_artifact_type_ch3
        record.qc_artifact_type_ch4 = audio.qc_artifact_type_ch4
        record.processing_date = audio.processing_date
        record.updated_at = datetime.utcnow()

    # ========================================================================
    # Biomechanics import operations
    # ========================================================================

    def save_biomechanics_import(
        self, biomech: BiomechanicsImport
    ) -> BiomechanicsImportRecord:
        """Save or update biomechanics import record.

        Args:
            biomech: BiomechanicsImport Pydantic model

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
            existing.updated_at = datetime.utcnow()
            record = existing
        else:
            # Create new record
            record = BiomechanicsImportRecord(
                participant_id=participant.id,
                biomechanics_file=biomech.biomechanics_file,
                biomechanics_type=biomech.biomechanics_type,
                knee=biomech.knee,
                maneuver=biomech.maneuver,
                pass_number=biomech.pass_number,
                speed=biomech.speed,
                biomechanics_sync_method=biomech.biomechanics_sync_method,
                biomechanics_sample_rate=biomech.biomechanics_sample_rate,
                biomech_qc_version=biomech.biomech_qc_version,
                biomechanics_qc_fail=biomech.biomechanics_qc_fail,
                biomechanics_qc_notes=biomech.biomechanics_qc_notes,
                processing_date=biomech.processing_date,
            )
            self.session.add(record)

        self.session.flush()
        return record

    # ========================================================================
    # Synchronization operations
    # ========================================================================

    def save_synchronization(
        self, sync: Synchronization, sync_file_path: Optional[str] = None
    ) -> SynchronizationRecord:
        """Save or update synchronization record.

        Args:
            sync: Synchronization Pydantic model
            sync_file_path: Optional path to sync .pkl file on disk

        Returns:
            Synchronization record
        """
        participant = self.get_or_create_participant(sync.study, sync.study_id)

        # Check if record already exists
        existing = self.session.execute(
            select(SynchronizationRecord).where(
                and_(
                    SynchronizationRecord.participant_id == participant.id,
                    SynchronizationRecord.audio_file_name == sync.audio_file_name,
                    SynchronizationRecord.knee == sync.knee,
                    SynchronizationRecord.maneuver == sync.maneuver,
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
                participant.id, sync, sync_file_path
            )
            self.session.add(record)

        self.session.flush()
        return record

    def _create_synchronization_record(
        self, participant_id: int, sync: Synchronization, sync_file_path: Optional[str]
    ) -> SynchronizationRecord:
        """Create new synchronization record from Pydantic model."""
        return SynchronizationRecord(
            participant_id=participant_id,
            audio_file_name=sync.audio_file_name,
            device_serial=sync.device_serial,
            firmware_version=sync.firmware_version,
            file_time=sync.file_time,
            file_size_mb=sync.file_size_mb,
            recording_date=sync.recording_date,
            recording_time=sync.recording_time,
            knee=sync.knee,
            maneuver=sync.maneuver,
            pass_number=sync.pass_number,
            speed=sync.speed,
            num_channels=sync.num_channels,
            sample_rate=sync.sample_rate,
            mic_1_position=sync.mic_1_position,
            mic_2_position=sync.mic_2_position,
            mic_3_position=sync.mic_3_position,
            mic_4_position=sync.mic_4_position,
            linked_biomechanics=sync.linked_biomechanics,
            biomechanics_file=sync.biomechanics_file,
            biomechanics_type=sync.biomechanics_type,
            biomechanics_sync_method=sync.biomechanics_sync_method,
            biomechanics_sample_rate=sync.biomechanics_sample_rate,
            audio_sync_time=sync.audio_sync_time,
            bio_left_sync_time=sync.bio_left_sync_time,
            bio_right_sync_time=sync.bio_right_sync_time,
            sync_offset=sync.sync_offset,
            aligned_audio_sync_time=sync.aligned_audio_sync_time,
            aligned_biomechanics_sync_time=sync.aligned_biomechanics_sync_time,
            sync_method=sync.sync_method,
            consensus_methods=sync.consensus_methods,
            rms_time=sync.rms_time,
            onset_time=sync.onset_time,
            freq_time=sync.freq_time,
            sync_file_name=sync.sync_file_name,
            sync_file_path=sync_file_path,
            sync_duration=sync.sync_duration,
            sync_qc_fail=sync.sync_qc_fail,
            processing_date=sync.processing_date,
        )

    def _update_synchronization_record(
        self,
        record: SynchronizationRecord,
        sync: Synchronization,
        sync_file_path: Optional[str],
    ) -> None:
        """Update existing synchronization record."""
        record.audio_sync_time = sync.audio_sync_time
        record.bio_left_sync_time = sync.bio_left_sync_time
        record.bio_right_sync_time = sync.bio_right_sync_time
        record.sync_offset = sync.sync_offset
        record.aligned_audio_sync_time = sync.aligned_audio_sync_time
        record.aligned_biomechanics_sync_time = sync.aligned_biomechanics_sync_time
        record.sync_method = sync.sync_method
        record.consensus_methods = sync.consensus_methods
        record.rms_time = sync.rms_time
        record.onset_time = sync.onset_time
        record.freq_time = sync.freq_time
        record.sync_file_name = sync.sync_file_name
        record.sync_file_path = sync_file_path or record.sync_file_path
        record.sync_duration = sync.sync_duration
        record.sync_qc_fail = sync.sync_qc_fail
        record.processing_date = sync.processing_date
        record.updated_at = datetime.utcnow()

    # ========================================================================
    # Movement cycle operations
    # ========================================================================

    def save_movement_cycle(
        self, cycle: MovementCycle, cycles_file_path: Optional[str] = None
    ) -> MovementCycleRecord:
        """Save or update movement cycle record.

        Args:
            cycle: MovementCycle Pydantic model
            cycles_file_path: Optional path to cycles .pkl file on disk

        Returns:
            Movement cycle record
        """
        participant = self.get_or_create_participant(cycle.study, cycle.study_id)

        # Check if record already exists
        existing = self.session.execute(
            select(MovementCycleRecord).where(
                and_(
                    MovementCycleRecord.participant_id == participant.id,
                    MovementCycleRecord.audio_file_name == cycle.audio_file_name,
                    MovementCycleRecord.knee == cycle.knee,
                    MovementCycleRecord.maneuver == cycle.maneuver,
                )
            )
        ).scalar_one_or_none()

        if existing:
            # Update existing record
            self._update_movement_cycle_record(existing, cycle, cycles_file_path)
            record = existing
        else:
            # Create new record
            record = self._create_movement_cycle_record(
                participant.id, cycle, cycles_file_path
            )
            self.session.add(record)

        self.session.flush()
        return record

    def _create_movement_cycle_record(
        self, participant_id: int, cycle: MovementCycle, cycles_file_path: Optional[str]
    ) -> MovementCycleRecord:
        """Create new movement cycle record from Pydantic model."""
        return MovementCycleRecord(
            participant_id=participant_id,
            audio_file_name=cycle.audio_file_name,
            device_serial=cycle.device_serial,
            firmware_version=cycle.firmware_version,
            file_time=cycle.file_time,
            file_size_mb=cycle.file_size_mb,
            recording_date=cycle.recording_date,
            recording_time=cycle.recording_time,
            knee=cycle.knee,
            maneuver=cycle.maneuver,
            pass_number=cycle.pass_number,
            speed=cycle.speed,
            num_channels=cycle.num_channels,
            sample_rate=cycle.sample_rate,
            mic_1_position=cycle.mic_1_position,
            mic_2_position=cycle.mic_2_position,
            mic_3_position=cycle.mic_3_position,
            mic_4_position=cycle.mic_4_position,
            linked_biomechanics=cycle.linked_biomechanics,
            biomechanics_file=cycle.biomechanics_file,
            biomechanics_type=cycle.biomechanics_type,
            biomechanics_sync_method=cycle.biomechanics_sync_method,
            biomechanics_sample_rate=cycle.biomechanics_sample_rate,
            num_cycles=cycle.num_cycles,
            cycles_file_name=cycle.cycles_file_name,
            cycles_file_path=cycles_file_path,
            cycle_qc_version=cycle.cycle_qc_version,
            biomechanics_qc_fail=cycle.biomechanics_qc_fail,
            sync_qc_fail=cycle.sync_qc_fail,
            processing_date=cycle.processing_date,
        )

    def _update_movement_cycle_record(
        self,
        record: MovementCycleRecord,
        cycle: MovementCycle,
        cycles_file_path: Optional[str],
    ) -> None:
        """Update existing movement cycle record."""
        record.num_cycles = cycle.num_cycles
        record.cycles_file_name = cycle.cycles_file_name
        record.cycles_file_path = cycles_file_path or record.cycles_file_path
        record.cycle_qc_version = cycle.cycle_qc_version
        record.biomechanics_qc_fail = cycle.biomechanics_qc_fail
        record.sync_qc_fail = cycle.sync_qc_fail
        record.processing_date = cycle.processing_date
        record.updated_at = datetime.utcnow()

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
            participant_number: Filter by participant number
            maneuver: Filter by maneuver
            knee: Filter by knee

        Returns:
            List of matching audio processing records
        """
        query = select(AudioProcessingRecord).join(ParticipantRecord)

        if study_name:
            query = query.join(StudyRecord).where(StudyRecord.name == study_name)
        if participant_number is not None:
            query = query.where(ParticipantRecord.participant_number == participant_number)
        if maneuver:
            query = query.where(AudioProcessingRecord.maneuver == maneuver)
        if knee:
            query = query.where(AudioProcessingRecord.knee == knee)

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
            participant_number: Filter by participant number
            maneuver: Filter by maneuver
            knee: Filter by knee

        Returns:
            List of matching synchronization records
        """
        query = select(SynchronizationRecord).join(ParticipantRecord)

        if study_name:
            query = query.join(StudyRecord).where(StudyRecord.name == study_name)
        if participant_number is not None:
            query = query.where(ParticipantRecord.participant_number == participant_number)
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
            participant_number: Filter by participant number
            maneuver: Filter by maneuver
            knee: Filter by knee

        Returns:
            List of matching movement cycle records
        """
        query = select(MovementCycleRecord).join(ParticipantRecord)

        if study_name:
            query = query.join(StudyRecord).where(StudyRecord.name == study_name)
        if participant_number is not None:
            query = query.where(ParticipantRecord.participant_number == participant_number)
        if maneuver:
            query = query.where(MovementCycleRecord.maneuver == maneuver)
        if knee:
            query = query.where(MovementCycleRecord.knee == knee)

        return list(self.session.execute(query).scalars().all())
