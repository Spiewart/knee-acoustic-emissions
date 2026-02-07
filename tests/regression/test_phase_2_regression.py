"""Tests for Phase 2A (Full Record Extraction) and Phase 2B (Dual-Write Pattern)."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.orchestration.dual_write_persistence import (
    DualWritePersistence,
    LocalStorageIndex,
)
from src.orchestration.persistent_processor import PersistentParticipantProcessor


class TestLocalStorageIndex:
    """Test local storage indexing for Phase 2B."""

    @pytest.fixture
    def temp_index_dir(self):
        """Create temporary directory for index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def local_index(self, temp_index_dir):
        """Create LocalStorageIndex instance."""
        return LocalStorageIndex(temp_index_dir)

    def test_initialization_creates_root_directory(self, temp_index_dir):
        """Test that initialization creates root directory."""
        index = LocalStorageIndex(temp_index_dir)
        assert index.index_root == temp_index_dir
        assert index.index_root.exists()

    def test_record_audio_processing_creates_file(self, local_index, temp_index_dir):
        """Test recording audio processing creates index file."""
        local_index.record_audio_processing(
            study_id="1011",
            audio_file="test_audio.bin",
            pkl_path="/path/to/test.pkl",
            db_saved=True,
            db_record_id=101,
        )

        # Check file was created
        index_file = temp_index_dir / "participants" / "1011" / "index.json"
        assert index_file.exists()

        # Verify content
        with open(index_file) as f:
            data = json.load(f)

        assert len(data["entries"]) == 1
        assert data["entries"][0]["type"] == "audio_processing"
        assert data["entries"][0]["audio_file"] == "test_audio.bin"
        assert data["entries"][0]["db_saved"] is True
        assert data["entries"][0]["db_record_id"] == 101

    def test_record_multiple_entries_appends(self, local_index, temp_index_dir):
        """Test that multiple entries are appended to same file."""
        local_index.record_audio_processing(
            study_id="1011",
            audio_file="audio1.bin",
            pkl_path="/path/1.pkl",
            db_saved=True,
            db_record_id=101,
        )
        local_index.record_biomechanics_import(
            study_id="1011",
            biomech_file="biomech.xlsx",
            db_saved=True,
            db_record_id=201,
            audio_id=101,
        )

        index_file = temp_index_dir / "participants" / "1011" / "index.json"
        with open(index_file) as f:
            data = json.load(f)

        assert len(data["entries"]) == 2
        assert data["entries"][0]["type"] == "audio_processing"
        assert data["entries"][1]["type"] == "biomechanics_import"

    def test_get_unsynced_entries_returns_failed_saves(self, local_index, temp_index_dir):
        """Test retrieving entries with failed DB saves."""
        local_index.record_audio_processing(
            study_id="1011",
            audio_file="audio1.bin",
            pkl_path="/path/1.pkl",
            db_saved=True,
            db_record_id=101,
        )
        local_index.record_audio_processing(
            study_id="1011",
            audio_file="audio2.bin",
            pkl_path="/path/2.pkl",
            db_saved=False,  # This one failed
            db_record_id=None,
        )

        unsynced = local_index.get_unsynced_entries("1011")

        assert len(unsynced) == 1
        assert unsynced[0]["audio_file"] == "audio2.bin"
        assert unsynced[0]["db_saved"] is False

    def test_record_synchronization_with_fks(self, local_index, temp_index_dir):
        """Test recording synchronization with FK relationships."""
        local_index.record_synchronization(
            study_id="1011",
            maneuver="walk",
            pass_number=1,
            db_saved=True,
            db_record_id=301,
            audio_id=101,
            biomech_id=201,
        )

        index_file = temp_index_dir / "participants" / "1011" / "index.json"
        with open(index_file) as f:
            data = json.load(f)

        entry = data["entries"][0]
        assert entry["type"] == "synchronization"
        assert entry["audio_id"] == 101
        assert entry["biomech_id"] == 201
        assert entry["pass_number"] == 1

    def test_record_movement_cycle_with_fks(self, local_index, temp_index_dir):
        """Test recording movement cycle with FK relationships."""
        local_index.record_movement_cycle(
            study_id="1011",
            cycle_number=1,
            maneuver="walk",
            db_saved=True,
            db_record_id=401,
            sync_id=301,
        )

        index_file = temp_index_dir / "participants" / "1011" / "index.json"
        with open(index_file) as f:
            data = json.load(f)

        entry = data["entries"][0]
        assert entry["type"] == "movement_cycle"
        assert entry["cycle_number"] == 1
        assert entry["sync_id"] == 301
        assert entry["db_record_id"] == 401


class TestDualWritePersistence:
    """Test dual-write persistence for Phase 2B."""

    @pytest.fixture
    def temp_storage_dir(self):
        """Create temporary directory for storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def dual_write(self, temp_storage_dir):
        """Create DualWritePersistence instance without DB."""
        return DualWritePersistence(
            db_session=None,
            local_storage_root=temp_storage_dir,
        )

    def test_initialization_without_database(self, dual_write):
        """Test initialization without database."""
        assert not dual_write.enabled
        assert dual_write.db_persistence is not None
        assert dual_write.local_index is not None

    def test_initialization_with_database(self, temp_storage_dir):
        """Test initialization with database session."""
        mock_session = MagicMock()
        dual_write = DualWritePersistence(
            db_session=mock_session,
            local_storage_root=temp_storage_dir,
        )

        assert dual_write.enabled
        assert dual_write.tracker is not None

    def test_save_audio_processing_saves_locally_when_db_disabled(
        self, dual_write, temp_storage_dir
    ):
        """Test audio save creates local index even without database."""
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
        )

        result = dual_write.save_audio_processing(audio, "/test/path.pkl")

        # DB save not attempted (no session)
        assert result is None

        # But local save should have occurred
        index_file = temp_storage_dir / "participants" / "1011" / "index.json"
        assert index_file.exists()

    @patch("src.orchestration.dual_write_persistence.OrchestrationDatabasePersistence")
    def test_save_audio_processing_returns_db_id_when_successful(
        self, mock_db_class, dual_write, temp_storage_dir
    ):
        """Test audio save returns DB ID when database save succeeds."""
        from src.metadata import AudioProcessing

        # Mock successful DB save
        mock_db = MagicMock()
        mock_db.enabled = True
        mock_db.save_audio_processing.return_value = 101
        dual_write.db_persistence = mock_db

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
        )

        result = dual_write.save_audio_processing(audio, "/test/path.pkl")

        assert result == 101

        # Verify DB was called
        mock_db.save_audio_processing.assert_called_once()

    @patch("src.orchestration.dual_write_persistence.OrchestrationDatabasePersistence")
    def test_save_audio_processing_continues_on_db_failure(
        self, mock_db_class, dual_write, temp_storage_dir
    ):
        """Test audio save continues and saves locally even if DB fails."""
        from src.metadata import AudioProcessing

        # Mock DB failure
        mock_db = MagicMock()
        mock_db.enabled = True
        mock_db.save_audio_processing.side_effect = Exception("DB connection failed")
        dual_write.db_persistence = mock_db

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
        )

        result = dual_write.save_audio_processing(audio, "/test/path.pkl")

        # Should return None (DB save failed)
        assert result is None

        # But local save should still have occurred
        index_file = temp_storage_dir / "participants" / "1011" / "index.json"
        assert index_file.exists()

        with open(index_file) as f:
            data = json.load(f)

        # Local entry shows DB failed
        assert data["entries"][0]["db_saved"] is False


class TestPhase2AIntegration:
    """Integration tests for Phase 2A record extraction."""

    @patch("src.orchestration.persistent_processor.ParticipantProcessor")
    def test_persist_audio_record_extraction(self, mock_processor_class):
        """Test extracting and persisting audio records."""
        from src.metadata import AudioProcessing

        # Mock processor with audio record
        mock_processor = MagicMock()
        mock_maneuver = MagicMock()
        mock_audio_data = MagicMock()

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
        )

        mock_audio_data.record = audio
        mock_audio_data.pkl_path = Path("/test/audio.pkl")
        mock_maneuver.audio = mock_audio_data
        mock_processor_class.return_value = mock_processor

        persistent_processor = PersistentParticipantProcessor(
            participant_dir=Path("/test"),
            db_session=None,
        )

        # Test extraction method
        result = persistent_processor._persist_audio_record(mock_maneuver)

        # Should return None (no DB session), but method should work
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
