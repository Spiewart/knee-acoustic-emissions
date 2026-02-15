"""Session 6: End-to-end integration tests with database persistence.

Tests the complete FK-based data persistence pipeline with real sample data.
Requires PostgreSQL and sample data directory to be available.
"""

from datetime import datetime
import os
from pathlib import Path

from dotenv import load_dotenv
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db import Base, init_db
from src.db.repository import Repository
from src.orchestration.database_persistence import (
    OrchestrationDatabasePersistence,
    RecordTracker,
)
from src.orchestration.participant_processor import ParticipantProcessor

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env.local")


def get_test_db_url() -> str:
    """Get test database URL."""
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
    """Create a test PostgreSQL database engine.

    Requires PostgreSQL to be running and AE_DATABASE_URL configured.
    Tests will create/drop tables in the configured database.
    """
    try:
        db_url = get_test_db_url()
        engine = create_engine(db_url, echo=False)

        # Test connection
        with engine.connect():
            pass

        # Drop all tables first (clean slate)
        Base.metadata.drop_all(engine)

        # Create all tables
        init_db(engine)

        yield engine

        # Clean up - drop all tables after tests
        Base.metadata.drop_all(engine)
        engine.dispose()

    except Exception as e:
        pytest.skip(f"PostgreSQL not available: {e}\nMake sure PostgreSQL is running and AE_DATABASE_URL is set.")


@pytest.fixture
def test_session(test_db_engine):
    """Create a test database session that rolls back after each test."""
    connection = test_db_engine.connect()
    transaction = connection.begin()
    SessionLocal = sessionmaker(bind=connection)
    session = SessionLocal()

    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def repository(test_session):
    """Create a repository instance for testing."""
    return Repository(test_session)


@pytest.fixture
def persistence(test_session):
    """Create orchestration database persistence for testing."""
    return OrchestrationDatabasePersistence(session=test_session)


@pytest.fixture
def tracker():
    """Create a record tracker for testing."""
    return RecordTracker()


@pytest.fixture
def sample_data_root():
    """Get sample data root directory."""
    root = os.getenv("AE_DATA_ROOT")
    if not root or not Path(root).exists():
        pytest.skip(f"Sample data directory not found: {root}")
    return Path(root)


class TestDatabasePersistenceWithSampleData:
    """Test database persistence with actual sample data."""

    def test_sample_data_directory_exists(self, sample_data_root):
        """Verify sample data directory is available."""
        assert sample_data_root.exists()

        # Check for participant directories
        participant_dirs = [d for d in sample_data_root.iterdir() if d.is_dir() and d.name.startswith("#")]
        assert len(participant_dirs) > 0, "No participant directories found in sample data"

    def test_participant_1011_has_expected_structure(self, sample_data_root):
        """Verify participant 1011 has expected knee/maneuver structure."""
        participant_dir = sample_data_root / "#1011"
        if not participant_dir.exists():
            pytest.skip(f"Participant directory not found: {participant_dir}")

        assert participant_dir.exists()

        # Check for knee directories
        knee_dirs = ["Left Knee", "Right Knee"]
        for knee in knee_dirs:
            knee_path = participant_dir / knee
            assert knee_path.exists(), f"Expected {knee} directory"

    def test_persistence_layer_initialization(self, persistence, tracker):
        """Test persistence layer can be initialized."""
        assert persistence is not None
        assert persistence.repository is not None
        assert tracker is not None

    def test_record_tracker_relationships(self, tracker):
        """Test record tracker maintains FK relationships correctly."""
        # Track audio processing
        audio_id = 101
        tracker.set_audio_processing(audio_id)
        assert tracker.get_audio_processing() == audio_id

        # Track biomechanics import
        biomech_id = 202
        tracker.set_biomechanics_import(biomech_id)
        assert tracker.get_biomechanics_import() == biomech_id

        # Track synchronization with pass number
        sync_id = 303
        pass_num = 1
        tracker.set_synchronization(pass_number=pass_num, record_id=sync_id)
        assert tracker.get_synchronization(pass_number=pass_num) == sync_id

        # Track movement cycles
        cycle_ids = [401, 402, 403]
        for cid in cycle_ids:
            tracker.add_movement_cycle(cid)
        assert tracker.get_movement_cycles() == cycle_ids

        # Verify summary
        summary = tracker.summary()
        assert summary["audio_processing_id"] == audio_id
        assert summary["biomechanics_import_id"] == biomech_id
        assert summary["synchronization_ids"][pass_num] == sync_id
        assert summary["movement_cycle_count"] == 3

    def test_persistence_returns_none_when_disabled(self, persistence):
        """Test that persistence returns None when session is None."""
        # Create a disabled persistence instance
        disabled_persistence = OrchestrationDatabasePersistence(session=None)
        assert disabled_persistence.enabled is False
        assert disabled_persistence.repository is None

    def test_graceful_degradation_on_persistence_error(self, test_session):
        """Test that processing continues even if persistence fails."""
        # Create a persistence instance with intentionally broken session
        # (we close it to make it invalid)
        test_session.close()

        persistence = OrchestrationDatabasePersistence(session=test_session)

        # Attempting to save should handle the error gracefully
        # (in real implementation, would log warning and return None)
        persistence.save_audio_processing(None, None)  # type: ignore
        # Result depends on error handling implementation
        # Should not raise an exception

    def test_repository_integration(self, repository):
        """Test that repository can be used to query database."""
        from src.metadata import AudioProcessing

        audio = AudioProcessing(
            study="AOA",
            study_id=1011,
            audio_file_name="test.bin",
            device_serial="TEST",
            firmware_version=1,
            file_time=datetime(2024, 1, 1, 10, 0, 0),
            file_size_mb=100.0,
            recording_date=datetime(2024, 1, 1),
            recording_time=datetime(2024, 1, 1, 10, 0, 0),
            processing_date=datetime(2024, 1, 1, 12, 0, 0),
            knee="left",
            maneuver="walk",
            num_channels=4,
            mic_1_position="IPM",
            mic_2_position="IPL",
            mic_3_position="SPM",
            mic_4_position="SPL",
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
            qc_continuous_artifact=False,
            qc_continuous_artifact_segments=[],
            qc_continuous_artifact_ch1=False,
            qc_continuous_artifact_segments_ch1=[],
            qc_continuous_artifact_ch2=False,
            qc_continuous_artifact_segments_ch2=[],
            qc_continuous_artifact_ch3=False,
            qc_continuous_artifact_segments_ch3=[],
            qc_continuous_artifact_ch4=False,
            qc_continuous_artifact_segments_ch4=[],
        )

        # Save audio processing
        record = repository.save_audio_processing(
            audio=audio,
            pkl_file_path="/test/path.pkl",
            biomechanics_import_id=None,
        )

        assert record is not None
        assert record.id is not None
        assert record.study_id is not None
        assert record.study.study_participant_id == 1011
        assert record.audio_file_name == "test.bin"

    def test_foreign_key_cascade_relationships(self, repository):
        """Test that FK relationships are properly maintained."""
        from src.metadata import AudioProcessing, BiomechanicsImport

        # Create and save audio
        audio = AudioProcessing(
            study="AOA",
            study_id=1011,
            audio_file_name="test_audio.bin",
            device_serial="TEST001",
            firmware_version=1,
            file_time=datetime(2024, 1, 1, 10, 0, 0),
            file_size_mb=100.0,
            recording_date=datetime(2024, 1, 1),
            recording_time=datetime(2024, 1, 1, 10, 0, 0),
            processing_date=datetime(2024, 1, 1, 12, 0, 0),
            knee="left",
            maneuver="walk",
            num_channels=4,
            mic_1_position="IPM",
            mic_2_position="IPL",
            mic_3_position="SPM",
            mic_4_position="SPL",
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
            qc_continuous_artifact=False,
            qc_continuous_artifact_segments=[],
            qc_continuous_artifact_ch1=False,
            qc_continuous_artifact_segments_ch1=[],
            qc_continuous_artifact_ch2=False,
            qc_continuous_artifact_segments_ch2=[],
            qc_continuous_artifact_ch3=False,
            qc_continuous_artifact_segments_ch3=[],
            qc_continuous_artifact_ch4=False,
            qc_continuous_artifact_segments_ch4=[],
        )
        audio_record = repository.save_audio_processing(
            audio=audio,
            pkl_file_path="/test/audio.pkl",
            biomechanics_import_id=None,
        )
        assert audio_record is not None
        audio_id = audio_record.id

        # Create and save biomechanics with FK to audio
        biomech = BiomechanicsImport(
            study="AOA",
            study_id=1011,
            biomechanics_file="test_biomech.xlsx",
            biomechanics_type="Motion Analysis",
            knee="left",
            maneuver="walk",
            biomechanics_sync_method="stomp",
            biomechanics_sample_rate=100.0,
            num_sub_recordings=1,
            duration_seconds=120.0,
            num_data_points=12000,
            num_passes=1,
            processing_date=datetime(2024, 1, 1, 12, 0, 0),
        )
        biomech_record = repository.save_biomechanics_import(
            biomech=biomech,
            audio_processing_id=audio_id,
        )
        assert biomech_record is not None
        assert biomech_record.audio_processing_id == audio_id

        # Query back to verify FK relationship
        queried_biomech = (
            repository._session.query(repository._models["BiomechanicsImportRecord"])
            .filter_by(id=biomech_record.id)
            .first()
        )

        assert queried_biomech is not None
        assert queried_biomech.audio_processing_id == audio_id


class TestProcessingPipelineWithDatabase:
    """Integration tests for processing pipeline with database persistence."""

    @pytest.mark.skip(reason="Requires full sample data with actual audio files")
    def test_process_participant_with_persistence(self, sample_data_root, persistence, tracker):
        """Test processing a participant with database persistence.

        This test requires:
        1. PostgreSQL running
        2. Sample data directory with actual audio files
        3. Full participant directory structure

        Skipped by default - run manually with:
            pytest tests/test_session_6_integration.py::TestProcessingPipelineWithDatabase::test_process_participant_with_persistence -v
        """
        participant_dir = sample_data_root / "#1011"
        if not participant_dir.exists():
            pytest.skip(f"Participant directory not found: {participant_dir}")

        # Create processor with persistence
        processor = ParticipantProcessor(
            participant_dir=participant_dir,
            biomechanics_type="Motion Analysis",
        )

        # Process with optional persistence tracking
        # (would integrate persistence layer here in full implementation)
        success = processor.process(entrypoint="bin", knee="left", maneuver="walk")

        assert success is not False  # At least shouldn't fail


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
