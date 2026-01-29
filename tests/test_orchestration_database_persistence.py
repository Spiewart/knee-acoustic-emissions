"""Tests for orchestration database persistence.

Tests the integration between the orchestration pipeline and the
PostgreSQL database via the persistence layer.
"""

import pytest
from datetime import datetime
from pathlib import Path

from src.orchestration.database_persistence import (
    OrchestrationDatabasePersistence,
    RecordTracker,
)
from src.orchestration.processing_log_persistence import PersistentProcessingLog
from src.metadata import AudioProcessing, BiomechanicsImport, Synchronization, MovementCycle


class TestRecordTracker:
    """Test record tracking functionality."""

    def test_track_audio_processing(self):
        """Test tracking audio processing record ID."""
        tracker = RecordTracker()
        assert tracker.get_audio_processing() is None

        tracker.set_audio_processing(123)
        assert tracker.get_audio_processing() == 123

    def test_track_biomechanics_import(self):
        """Test tracking biomechanics import record ID."""
        tracker = RecordTracker()
        assert tracker.get_biomechanics_import() is None

        tracker.set_biomechanics_import(456)
        assert tracker.get_biomechanics_import() == 456

    def test_track_synchronization_with_pass_number(self):
        """Test tracking synchronization records by pass number."""
        tracker = RecordTracker()

        tracker.set_synchronization(pass_number=1, record_id=789)
        tracker.set_synchronization(pass_number=2, record_id=790)

        assert tracker.get_synchronization(pass_number=1) == 789
        assert tracker.get_synchronization(pass_number=2) == 790

    def test_track_synchronization_without_pass_number(self):
        """Test tracking synchronization record without pass number."""
        tracker = RecordTracker()

        tracker.set_synchronization(pass_number=None, record_id=789)
        assert tracker.get_synchronization(pass_number=None) == 789

    def test_track_movement_cycles(self):
        """Test tracking multiple movement cycle records."""
        tracker = RecordTracker()

        assert len(tracker.get_movement_cycles()) == 0

        tracker.add_movement_cycle(1001)
        tracker.add_movement_cycle(1002)
        tracker.add_movement_cycle(1003)

        cycles = tracker.get_movement_cycles()
        assert len(cycles) == 3
        assert 1001 in cycles
        assert 1002 in cycles
        assert 1003 in cycles

    def test_tracker_summary(self):
        """Test getting tracker summary."""
        tracker = RecordTracker()

        tracker.set_audio_processing(123)
        tracker.set_biomechanics_import(456)
        tracker.set_synchronization(pass_number=1, record_id=789)
        tracker.add_movement_cycle(1001)

        summary = tracker.summary()
        assert summary["audio_processing_id"] == 123
        assert summary["biomechanics_import_id"] == 456
        assert summary["synchronization_ids"][1] == 789
        assert summary["movement_cycle_count"] == 1


class TestOrchestrationDatabasePersistence:
    """Test persistence layer without real database."""

    def test_persistence_disabled_when_no_session(self):
        """Test that persistence is disabled when session is None."""
        persistence = OrchestrationDatabasePersistence(session=None)
        assert not persistence.enabled
        assert persistence.repository is None

    def test_save_audio_processing_disabled(self):
        """Test that save returns None when persistence disabled."""
        persistence = OrchestrationDatabasePersistence(session=None)

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
            processing_date=datetime(2024, 1, 1, 10, 0, 1),
            knee="left",
            maneuver="walk",
            num_channels=4,
            mic_1_position="IPM",
            mic_2_position="IPL",
            mic_3_position="SPM",
            mic_4_position="SPL",
        )

        result = persistence.save_audio_processing(audio)
        assert result is None

    def test_save_biomechanics_import_disabled(self):
        """Test that save returns None when persistence disabled."""
        persistence = OrchestrationDatabasePersistence(session=None)

        biomech = BiomechanicsImport(
            study="AOA",
            study_id=1012,
            biomechanics_file="test.xlsx",
            biomechanics_type="Motion Analysis",
            knee="left",
            maneuver="walk",
            biomechanics_sync_method="stomp",
            biomechanics_sample_rate=100.0,
            num_sub_recordings=1,
            duration_seconds=120.0,
            num_data_points=12000,
            num_passes=1,
            processing_date=datetime(2024, 1, 1, 10, 0, 1),
        )

        result = persistence.save_biomechanics_import(biomech)
        assert result is None


class TestPersistentProcessingLog:
    """Test persistent processing log wrapper."""

    def test_initialization_without_persistence(self):
        """Test creating persistent processing log without persistence."""
        ppl = PersistentProcessingLog(persistence=None, tracker=None)
        assert ppl.persistence is None
        assert ppl.tracker is None

    def test_initialization_with_tracker(self):
        """Test creating persistent processing log with tracker."""
        tracker = RecordTracker()
        ppl = PersistentProcessingLog(persistence=None, tracker=tracker)
        assert ppl.tracker is tracker


# Integration tests with real database (skipped if PostgreSQL not available)
@pytest.mark.skip(reason="Requires live PostgreSQL database - run from test_database.py for integration tests")
class TestOrchestrationDatabaseIntegration:
    """Integration tests with PostgreSQL database."""

    def test_persistence_enabled_with_session(self):
        """Test that persistence is enabled when session provided.
        
        Skipped - requires live database. Run from test_database.py instead.
        """
        pass

    def test_full_integration_workflow(self):
        """Test complete persistence workflow with FK relationships.
        
        Skipped - requires live database. Run from test_database.py instead.
        """
        pass

