"""Tests for database models and repository layer.

These tests require PostgreSQL to be running and AE_DATABASE_URL configured.
For safety, use a dedicated test database (e.g., acoustic_emissions_test).
"""

import os
from datetime import datetime
from pathlib import Path

import pytest
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.db import Base, ParticipantRecord, StudyRecord, init_db
from src.db.repository import Repository
from src.metadata import (
    AudioProcessing,
    BiomechanicsImport,
    MovementCycle,
    Synchronization,
)

# Load environment variables from .env.local
load_dotenv(Path(__file__).parent.parent / ".env.local")


def create_test_audio_processing(**kwargs):
    """Helper to create AudioProcessing with all required fields."""
    defaults = {
        "study": "AOA",
        "study_id": 1011,
        "linked_biomechanics": False,
        "audio_file_name": "test_audio.bin",
        "device_serial": "SN12345",
        "firmware_version": 1,
        "file_time": datetime(2024, 1, 1, 10, 0, 0),
        "file_size_mb": 100.0,
        "recording_date": datetime(2024, 1, 1),
        "recording_time": datetime(2024, 1, 1, 10, 0, 0),
        "knee": "left",
        "maneuver": "walk",
        "num_channels": 4,
        "mic_1_position": "IPM",
        "mic_2_position": "IPL",
        "mic_3_position": "SPM",
        "mic_4_position": "SPL",
        "processing_date": datetime(2024, 1, 1, 12, 0, 0),
        # QC fields - all empty/passing
        "qc_fail_segments": [],
        "qc_fail_segments_ch1": [],
        "qc_fail_segments_ch2": [],
        "qc_fail_segments_ch3": [],
        "qc_fail_segments_ch4": [],
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
        "qc_continuous_artifact": False,
        "qc_continuous_artifact_segments": [],
        "qc_continuous_artifact_ch1": False,
        "qc_continuous_artifact_segments_ch1": [],
        "qc_continuous_artifact_ch2": False,
        "qc_continuous_artifact_segments_ch2": [],
        "qc_continuous_artifact_ch3": False,
        "qc_continuous_artifact_segments_ch3": [],
        "qc_continuous_artifact_ch4": False,
        "qc_continuous_artifact_segments_ch4": [],
    }
    defaults.update(kwargs)
    return AudioProcessing(**defaults)


def create_test_synchronization(**kwargs):
    """Helper to create Synchronization with all required fields."""
    defaults = {
        "study": "AOA",
        "study_id": 1012,
        "pass_number": 1,  # Required for walk
        "speed": "medium",  # Required for walk (removed "normal")
        "audio_processing_id": 1,  # Dummy FK, will be overridden by caller
        "biomechanics_import_id": 1,  # Dummy FK, will be overridden by caller
        "audio_sync_time": 5.0,
        "bio_left_sync_time": 10.0,
        "bio_right_sync_time": None,
        "sync_offset": 5.0,
        "aligned_audio_sync_time": 10.0,
        "aligned_biomechanics_sync_time": 10.0,
        "sync_method": "consensus",
        "consensus_methods": "rms,onset,freq",
        "consensus_time": 5.0,
        "rms_time": 5.0,
        "onset_time": 5.1,
        "freq_time": 4.9,
        "sync_file_name": "test_sync.pkl",
        "sync_duration": 120.0,
        "processing_date": datetime(2024, 1, 2, 12, 0, 0),
    }
    defaults.update(kwargs)
    return Synchronization(**defaults)


def create_test_biomechanics_import(**kwargs):
    """Helper to create BiomechanicsImport with all required fields."""
    defaults = {
        "study": "AOA",
        "study_id": 1013,
        "biomechanics_file": "test_biomech.xlsx",
        "biomechanics_type": "Motion Analysis",
        "knee": "left",
        "maneuver": "walk",
        "pass_number": 1,
        "speed": "medium",
        "biomechanics_sync_method": "stomp",
        "biomechanics_sample_rate": 100.0,
        "num_sub_recordings": 1,
        "duration_seconds": 120.0,
        "num_data_points": 12000,
        "num_passes": 1,
        "processing_date": datetime(2024, 1, 2, 12, 0, 0),
    }
    defaults.update(kwargs)
    return BiomechanicsImport(**defaults)


def create_test_movement_cycle(**kwargs):
    """Helper to create MovementCycle with all required fields."""
    defaults = {
        "study": "AOA",
        "study_id": 1014,
        "audio_processing_id": 1,  # Dummy FK, will be overridden by caller
        "biomechanics_import_id": None,
        "synchronization_id": None,
        "cycle_file": "test_cycle.pkl",
        "cycle_index": 0,
        "is_outlier": False,
        "start_time_s": 5.0,
        "end_time_s": 7.5,
        "duration_s": 2.5,
        "start_time": datetime(2024, 1, 2, 10, 0, 5),
        "end_time": datetime(2024, 1, 2, 10, 0, 7, 500000),
        "biomechanics_qc_fail": False,
        "sync_qc_fail": False,
        "audio_qc_fail": False,
        # Intermittent artifact QC
        "audio_artifact_intermittent_fail": False,
        "audio_artifact_intermittent_fail_ch1": False,
        "audio_artifact_intermittent_fail_ch2": False,
        "audio_artifact_intermittent_fail_ch3": False,
        "audio_artifact_intermittent_fail_ch4": False,
        # Periodic artifact QC
        "audio_artifact_periodic_fail": False,
        "audio_artifact_periodic_fail_ch1": False,
        "audio_artifact_periodic_fail_ch2": False,
        "audio_artifact_periodic_fail_ch3": False,
        "audio_artifact_periodic_fail_ch4": False,
        "processing_date": datetime(2024, 1, 2, 12, 0, 0),
    }
    defaults.update(kwargs)
    return MovementCycle(**defaults)



def get_test_db_url():
    """Get test database URL, defaulting to a test database.

    Checks AE_TEST_DATABASE_URL first, then falls back to appending _test
    to AE_DATABASE_URL, or uses a default test database.
    """
    # Check for explicit test database URL
    test_url = os.getenv("AE_TEST_DATABASE_URL")
    if test_url:
        return test_url

    # Fall back to production URL with _test suffix
    prod_url = os.getenv("AE_DATABASE_URL")
    if prod_url and "acoustic_emissions" in prod_url:
        return prod_url.replace("acoustic_emissions", "acoustic_emissions_test")

    # Default test database
    return "postgresql+psycopg://postgres@localhost/acoustic_emissions_test"


@pytest.fixture(scope="module")
def test_db_engine():
    """Create a test PostgreSQL database.

    Requires PostgreSQL to be running and AE_DATABASE_URL set.
    Tests will create/drop tables in the configured database.
    """
    try:
        db_url = get_test_db_url()
        engine = create_engine(db_url, echo=False)

        # Test connection
        with engine.connect() as conn:
            pass

# Reset schema to avoid FK naming mismatches
        with engine.begin() as conn:
            conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
            conn.execute(text("CREATE SCHEMA public"))

        # Create all tables
        init_db(engine)

        yield engine

        # Clean up - drop schema after tests
        with engine.begin() as conn:
            conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
            conn.execute(text("CREATE SCHEMA public"))
        engine.dispose()

    except Exception as e:
        pytest.skip(f"PostgreSQL not available: {e}\nMake sure PostgreSQL is running and AE_DATABASE_URL is set.")


@pytest.fixture
def test_session(test_db_engine):
    """Create a test database session that rolls back after each test.

    Uses a transaction to ensure test isolation - all changes are rolled back.
    """
    connection = test_db_engine.connect()
    transaction = connection.begin()
    SessionLocal = sessionmaker(bind=connection)
    session = SessionLocal()

    try:
        yield session
    finally:
        session.close()
        if transaction.is_active:
            transaction.rollback()
        connection.close()


@pytest.fixture
def repository(test_session):
    """Create a repository instance for testing."""
    return Repository(test_session)


class TestDatabaseModels:
    """Test SQLAlchemy ORM models."""

    def test_create_study(self, test_session):
        """Test creating a study record."""
        study = StudyRecord(name="AOA")
        test_session.add(study)
        test_session.commit()

        assert study.id is not None
        assert study.name == "AOA"
        assert study.created_at is not None

    def test_create_participant(self, test_session):
        """Test creating a participant record."""
        study = StudyRecord(name="AOA")
        test_session.add(study)
        test_session.flush()

        participant = ParticipantRecord(
            study_participant_id=study.id,
            study_id=1011
        )
        test_session.add(participant)
        test_session.commit()

        assert participant.id is not None
        assert participant.study_id == 1011
        assert participant.study.name == "AOA"

    def test_unique_study_participant(self, test_session):
        """Test that study+study_id is unique."""
        study = StudyRecord(name="AOA")
        test_session.add(study)
        test_session.flush()

        participant1 = ParticipantRecord(
            study_participant_id=study.id,
            study_id=1011
        )
        test_session.add(participant1)
        test_session.commit()

        # Try to create duplicate
        participant2 = ParticipantRecord(
            study_participant_id=study.id,
            study_id=1011
        )
        test_session.add(participant2)

        with pytest.raises(Exception):  # Should raise integrity error
            test_session.commit()


class TestRepository:
    """Test repository layer."""

    def test_get_or_create_study(self, repository):
        """Test getting or creating a study."""
        study1 = repository.get_or_create_study("AOA")
        assert study1.name == "AOA"

        # Should return same study on second call
        study2 = repository.get_or_create_study("AOA")
        assert study1.id == study2.id

    def test_get_or_create_participant(self, repository):
        """Test getting or creating a participant."""
        participant1 = repository.get_or_create_participant("AOA", 1011)
        assert participant1.study_id == 1011

        # Should return same participant on second call
        participant2 = repository.get_or_create_participant("AOA", 1011)
        assert participant1.id == participant2.id

    def test_save_audio_processing(self, repository):
        """Test saving audio processing record (without biomechanics FK)."""
        audio = create_test_audio_processing()

        # Save without FK reference (recording alone)
        record = repository.save_audio_processing(audio)
        assert record.id is not None
        assert record.audio_file_name == "test_audio.bin"
        assert record.knee == "left"
        assert record.maneuver == "walk"
        assert record.biomechanics_import_id is None

    def test_save_audio_processing_with_biomechanics_fk(self, repository):
        """Test saving audio processing record with biomechanics FK."""
        # First save biomechanics import
        biomech = create_test_biomechanics_import(knee="left", maneuver="walk")
        biomech_record = repository.save_biomechanics_import(biomech)

        # Now save audio with FK reference
        audio = create_test_audio_processing()
        record = repository.save_audio_processing(
            audio,
            biomechanics_import_id=biomech_record.id
        )

        assert record.id is not None
        assert record.audio_file_name == "test_audio.bin"
        assert record.biomechanics_import_id == biomech_record.id

    def test_save_synchronization(self, repository):
        """Test saving synchronization record with FK references."""
        # First save audio and biomechanics records
        audio = create_test_audio_processing(maneuver="walk")
        audio_record = repository.save_audio_processing(audio)

        biomech = create_test_biomechanics_import(maneuver="walk")
        biomech_record = repository.save_biomechanics_import(biomech)

        # Now save sync with FK references
        sync = create_test_synchronization(
            pass_number=1,
            speed="medium",
            aligned_sync_time=10.0,
            bio_left_sync_time=10.0,
            bio_sync_offset=5.0,
            sync_method="consensus",
            consensus_methods="rms,onset,freq",
            rms_time=5.0,
            onset_time=5.1,
            freq_time=4.9,
            sync_duration=120.0,
        )

        record = repository.save_synchronization(
            sync,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id
        )
        assert record.id is not None
        assert record.audio_processing_id == audio_record.id
        assert record.biomechanics_import_id == biomech_record.id
        assert record.sync_method == "consensus"
        assert record.bio_sync_offset == 5.0

    def test_query_audio_processing_records(self, repository):
        """Test querying audio processing records with filters."""
        # Create test records
        audio1 = create_test_audio_processing(
            audio_file_name="test1.bin",
            knee="left",
            maneuver="walk",
        )
        repository.save_audio_processing(audio1)

        audio2 = create_test_audio_processing(
            audio_file_name="test2.bin",
            knee="right",
            maneuver="fe",
        )
        repository.save_audio_processing(audio2)

        # Query all records
        all_records = repository.get_audio_processing_records()
        assert len(all_records) == 2

        # Query by maneuver
        walk_records = repository.get_audio_processing_records(maneuver="walk")
        assert len(walk_records) == 1
        assert walk_records[0].maneuver == "walk"

        # Query by knee
        left_records = repository.get_audio_processing_records(knee="left")
        assert len(left_records) == 1
        assert left_records[0].knee == "left"
    def test_save_biomechanics_import(self, repository):
        """Test saving biomechanics import record."""
        biomech = create_test_biomechanics_import()
        record = repository.save_biomechanics_import(biomech)

        assert record.id is not None
        assert record.biomechanics_file == "test_biomech.xlsx"
        assert record.biomechanics_type == "Motion Analysis"
        assert record.knee == "left"
        assert record.maneuver == "walk"
        assert record.audio_processing_id is None

    def test_save_biomechanics_import_with_audio_fk(self, repository):
        """Test saving biomechanics import with audio FK."""
        # First save audio
        audio = create_test_audio_processing(maneuver="walk")
        audio_record = repository.save_audio_processing(audio)

        # Now save biomech with FK reference
        biomech = create_test_biomechanics_import(maneuver="walk")
        record = repository.save_biomechanics_import(
            biomech,
            audio_processing_id=audio_record.id
        )

        assert record.id is not None
        assert record.biomechanics_file == "test_biomech.xlsx"
        assert record.audio_processing_id == audio_record.id

    def test_save_movement_cycle(self, repository):
        """Test saving movement cycle record with FKs."""
        # First save audio record
        audio = create_test_audio_processing()
        audio_record = repository.save_audio_processing(audio)

        # Now save cycle with audio FK
        cycle = create_test_movement_cycle()
        record = repository.save_movement_cycle(
            cycle,
            audio_processing_id=audio_record.id
        )

        assert record.id is not None
        assert record.audio_processing_id == audio_record.id
        assert record.cycle_file == "test_cycle.pkl"
        assert record.cycle_index == 0
        assert record.biomechanics_import_id is None
        assert record.synchronization_id is None

    def test_save_movement_cycle_with_all_fks(self, repository):
        """Test saving movement cycle with all FK references."""
        # Create and save audio
        audio = create_test_audio_processing(maneuver="walk")
        audio_record = repository.save_audio_processing(audio)

        # Create and save biomechanics
        biomech = create_test_biomechanics_import(maneuver="walk")
        biomech_record = repository.save_biomechanics_import(biomech)

        # Create and save synchronization
        sync = create_test_synchronization()
        sync_record = repository.save_synchronization(
            sync,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id
        )

        # Now save cycle with all FKs
        cycle = create_test_movement_cycle()
        record = repository.save_movement_cycle(
            cycle,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
            synchronization_id=sync_record.id
        )

        assert record.id is not None
        assert record.audio_processing_id == audio_record.id
        assert record.biomechanics_import_id == biomech_record.id
        assert record.synchronization_id == sync_record.id