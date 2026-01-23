"""Tests for the processing log system.

Tests cover:
- Record creation and data models
- ManeuverProcessingLog CRUD operations
- Excel save/load functionality
- Helper functions for creating records from data
- Incremental updates
"""

from datetime import datetime

import numpy as np
import pandas as pd

from src.metadata import (
    AudioProcessing,
    BiomechanicsImport,
    Synchronization,
    MovementCycles,
)
from src.orchestration.processing_log import (
    ManeuverProcessingLog,
    create_audio_record_from_data,
    create_biomechanics_record_from_data,
    create_cycles_record_from_data,
    create_sync_record_from_data,
)


# Helper functions for creating minimal test records with all required fields

def create_minimal_audio_record(**overrides):
    """Create a minimal AudioProcessing record for testing."""
    defaults = {
        # StudyMetadata
        "study": "AOA",
        "study_id": 1,
        # BiomechanicsMetadata
        "linked_biomechanics": False,
        # AcousticsFile
        "audio_file_name": "test_audio.bin",
        "device_serial": "TEST123",
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
        # AudioProcessing
        "processing_date": datetime(2024, 1, 1, 12, 0, 0),
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
        "qc_artifact": False,
        "qc_artifact_segments": [],
        "qc_artifact_ch1": False,
        "qc_artifact_segments_ch1": [],
        "qc_artifact_ch2": False,
        "qc_artifact_segments_ch2": [],
        "qc_artifact_ch3": False,
        "qc_artifact_segments_ch3": [],
        "qc_artifact_ch4": False,
        "qc_artifact_segments_ch4": [],
    }
    defaults.update(overrides)
    return AudioProcessing(**defaults)


def create_minimal_biomech_record(**overrides):
    """Create a minimal BiomechanicsImport record for testing."""
    defaults = {
        # StudyMetadata
        "study": "AOA",
        "study_id": 1,
        # BiomechanicsImport
        "biomechanics_file": "test_biomech.xlsx",
        "sheet_name": "Walk0001",
        "processing_date": datetime(2024, 1, 1, 12, 0, 0),
        "processing_status": "success",
        "num_sub_recordings": 1,
        "num_passes": 3,
        "duration_seconds": 120.0,
        "sample_rate": 100.0,
        "num_data_points": 12000,
    }
    defaults.update(overrides)
    return BiomechanicsImport(**defaults)


def create_minimal_sync_record(**overrides):
    """Create a minimal Synchronization record for testing."""
    from datetime import timedelta
    
    defaults = {
        # StudyMetadata
        "study": "AOA",
        "study_id": 1,
        # BiomechanicsMetadata
        "linked_biomechanics": True,
        "biomechanics_file": "test_biomech.xlsx",
        "biomechanics_type": "Gonio",
        "biomechanics_sample_rate": 100.0,
        # AcousticsFile
        "audio_file_name": "test_audio.bin",
        "device_serial": "TEST123",
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
        # SynchronizationMetadata
        "audio_sync_time": timedelta(seconds=5.0),
        "bio_left_sync_time": timedelta(seconds=10.0),
        "sync_offset": timedelta(seconds=5.0),
        "aligned_audio_sync_time": timedelta(seconds=10.0),
        "aligned_bio_sync_time": timedelta(seconds=10.0),
        "sync_method": "consensus",
        "consensus_time": timedelta(seconds=5.0),
        "rms_time": timedelta(seconds=5.0),
        "onset_time": timedelta(seconds=5.0),
        "freq_time": timedelta(seconds=5.0),
        # Synchronization
        "sync_file_name": "test_sync.pkl",
        "processing_date": datetime(2024, 1, 1, 12, 0, 0),
        "sync_duration": timedelta(seconds=120.0),
        "total_cycles_extracted": 0,
        "clean_cycles": 0,
        "outlier_cycles": 0,
        "mean_cycle_duration_s": 0.0,
        "median_cycle_duration_s": 0.0,
        "min_cycle_duration_s": 0.0,
        "max_cycle_duration_s": 0.0,
        "mean_acoustic_auc": 0.0,
        "qc_fail_segments": [],
        "qc_signal_dropout": False,
        "qc_signal_dropout_segments": [],
        "qc_artifact": False,
        "qc_artifact_segments": [],
    }
    defaults.update(overrides)
    return Synchronization(**defaults)


class TestAudioProcessing:
    """Tests for AudioProcessing metadata."""

    def test_create_record(self):
        """Test creating an audio processing record."""
        record = create_minimal_audio_record(
            audio_file_name="test_audio",
            processing_status="success",
            sample_rate=46875.0,
            duration_seconds=120.5,
        )

        assert record.audio_file_name == "test_audio"
        assert record.processing_status == "success"
        assert record.sample_rate == 46875.0
        assert record.duration_seconds == 120.5

    def test_to_dict(self):
        """Test converting record to dictionary."""
        record = create_minimal_audio_record(
            audio_file_name="test_audio",
            processing_status="success",
            sample_rate=46875.0,
        )

        data = record.to_dict()

        assert isinstance(data, dict)
        assert data["Audio File"] == "test_audio"
        assert data["Status"] == "success"
        assert data["Sample Rate (Hz)"] == 46875.0
        assert "Audio QC Version" in data
        assert data["Audio QC Version"] == 1

    def test_default_values(self):
        """Test that default values are set correctly."""
        record = create_minimal_audio_record(
            audio_file_name="test",
            sample_rate=None,
        )

        assert record.processing_status == "not_processed"
        assert record.num_channels == 4
        assert record.sample_rate is None
        assert record.audio_qc_version == 1


class TestBiomechanicsImport:
    """Tests for BiomechanicsImport metadata."""

    def test_create_record(self):
        """Test creating a biomechanics import record."""
        record = create_minimal_biomech_record(
            biomechanics_file="AOA1011_Biomechanics_Full_Set.xlsx",
            processing_status="success",
            num_sub_recordings=3,
            num_passes=9,
        )

        assert record.biomechanics_file == "AOA1011_Biomechanics_Full_Set.xlsx"
        assert record.processing_status == "success"
        assert record.num_sub_recordings == 3
        assert record.num_passes == 9

    def test_to_dict(self):
        """Test converting record to dictionary."""
        record = create_minimal_biomech_record(
            biomechanics_file="test.xlsx",
            sheet_name="Walk0001",
            num_sub_recordings=5,
        )

        data = record.to_dict()

        assert data["Biomechanics File"] == "test.xlsx"
        assert data["Sheet Name"] == "Walk0001"
        assert data["Num Sub-Recordings"] == 5


class TestSynchronization:
    """Tests for Synchronization metadata."""

    def test_create_record(self):
        """Test creating a synchronization record."""
        from datetime import timedelta
        record = create_minimal_sync_record(
            sync_file_name="left_walk_slow_pass1",
            pass_number=1,
            speed="slow",
            audio_sync_time=timedelta(seconds=10.5),
            bio_left_sync_time=timedelta(seconds=5.2),
            knee="left",
        )

        assert record.sync_file_name == "left_walk_slow_pass1"
        assert record.pass_number == 1
        assert record.speed == "slow"
        assert record.audio_sync_time.total_seconds() == 10.5
        assert record.knee == "left"

    def test_to_dict(self):
        """Test converting record to dictionary."""
        record = create_minimal_sync_record(
            sync_file_name="test_sync",
            processing_status="success",
        )

        data = record.to_dict()

        assert data["Sync File"] == "test_sync"
        assert data["Status"] == "success"
        assert "Audio QC Version" in data
        assert data["Audio QC Version"] == 1


class TestMovementCycles:
    """Tests for MovementCycles metadata (alias for Synchronization)."""

    def test_create_record(self):
        """Test creating a movement cycles record."""
        record = create_minimal_sync_record(
            sync_file_name="test_sync",
            total_cycles_extracted=12,
            clean_cycles=10,
            outlier_cycles=2,
        )

        assert record.sync_file_name == "test_sync"
        assert record.total_cycles_extracted == 12
        assert record.clean_cycles == 10
        assert record.outlier_cycles == 2

    def test_to_dict(self):
        """Test converting record to dictionary."""
        record = create_minimal_sync_record(
            sync_file_name="test_sync",
            clean_cycles=8,
            outlier_cycles=2,
            total_cycles_extracted=10,
        )

        data = record.to_dict()

        assert data["Sync File"] == "test_sync"
        assert data["Clean Cycles"] == 8
        assert data["Outlier Cycles"] == 2


class TestManeuverProcessingLog:
    """Tests for ManeuverProcessingLog."""

    def test_create_log(self, tmp_path):
        """Test creating a processing log."""
        log = ManeuverProcessingLog(
            study_id="1011",
            knee_side="Left",
            maneuver="walk",
            maneuver_directory=tmp_path,
            log_created=datetime.now(),
            log_updated=datetime.now(),
        )

        assert log.study_id == "1011"
        assert log.knee_side == "Left"
        assert log.maneuver == "walk"
        assert log.maneuver_directory == tmp_path

    def test_update_audio_record(self, tmp_path):
        """Test updating audio record."""
        log = ManeuverProcessingLog(
            study_id="1011",
            knee_side="Left",
            maneuver="walk",
            maneuver_directory=tmp_path,
        )

        record = create_minimal_audio_record(
            audio_file_name="test_audio",
            processing_status="success",
        )

        log.update_audio_record(record)

        assert log.audio_record is not None
        assert log.audio_record.audio_file_name == "test_audio"
        assert log.log_updated is not None

    def test_add_synchronization_record(self, tmp_path):
        """Test adding synchronization record."""
        log = ManeuverProcessingLog(
            study_id="1011",
            knee_side="Left",
            maneuver="walk",
            maneuver_directory=tmp_path,
        )

        sync_record = create_minimal_sync_record(
            sync_file_name="test_sync",
            processing_status="success",
        )

        log.add_synchronization_record(sync_record)

        assert len(log.synchronization_records) == 1
        assert log.synchronization_records[0].sync_file_name == "test_sync"

    def test_update_existing_synchronization_record(self, tmp_path):
        """Test that adding a record with same filename updates existing."""
        log = ManeuverProcessingLog(
            study_id="1011",
            knee_side="Left",
            maneuver="walk",
            maneuver_directory=tmp_path,
        )

        # Add first record
        sync_record1 = create_minimal_sync_record(
            sync_file_name="test_sync",
        )
        log.add_synchronization_record(sync_record1)

        # Add second record with same name
        sync_record2 = create_minimal_sync_record(
            sync_file_name="test_sync",
            total_cycles_extracted=10,
        )
        log.add_synchronization_record(sync_record2)

        # Should only have one record, with updated values
        assert len(log.synchronization_records) == 1
        assert log.synchronization_records[0].total_cycles_extracted == 10

    def test_add_movement_cycles_record(self, tmp_path):
        """Test adding movement cycles record."""
        log = ManeuverProcessingLog(
            study_id="1011",
            knee_side="Left",
            maneuver="walk",
            maneuver_directory=tmp_path,
        )

        cycles_record = create_minimal_sync_record(
            sync_file_name="test_sync",
            clean_cycles=10,
            outlier_cycles=2,
            total_cycles_extracted=12,
        )

        log.add_movement_cycles_record(cycles_record)

        assert len(log.movement_cycles_records) == 1
        assert log.movement_cycles_records[0].clean_cycles == 10

    def test_save_to_excel(self, tmp_path):
        """Test saving log to Excel file."""
        log = ManeuverProcessingLog(
            study_id="1011",
            knee_side="Left",
            maneuver="walk",
            maneuver_directory=tmp_path,
            log_created=datetime.now(),
            log_updated=datetime.now(),
        )

        # Add some data
        audio = create_minimal_audio_record(
            audio_file_name="test_audio",
            processing_status="success",
        )
        log.update_audio_record(audio)

        sync = create_minimal_sync_record(
            sync_file_name="test_sync",
            processing_status="success",
        )
        log.add_synchronization_record(sync)

        # Save to Excel
        excel_path = tmp_path / "test_log.xlsx"
        saved_path = log.save_to_excel(excel_path)

        assert saved_path.exists()
        assert saved_path == excel_path

    def test_load_from_excel(self, tmp_path):
        """Test loading log from Excel file."""
        # Create and save a log
        original_log = ManeuverProcessingLog(
            study_id="1011",
            knee_side="Left",
            maneuver="walk",
            maneuver_directory=tmp_path,
            log_created=datetime.now(),
            log_updated=datetime.now(),
        )

        audio = create_minimal_audio_record(
            audio_file_name="test_audio",
            sample_rate=46875.0,
            processing_status="success",
        )
        original_log.update_audio_record(audio)

        sync = create_minimal_sync_record(
            sync_file_name="test_sync",
        )
        original_log.add_synchronization_record(sync)

        excel_path = tmp_path / "test_log.xlsx"
        original_log.save_to_excel(excel_path)

        # Load the log
        loaded_log = ManeuverProcessingLog.load_from_excel(excel_path)

        assert loaded_log is not None
        assert loaded_log.study_id == "1011"
        assert loaded_log.knee_side == "Left"
        assert loaded_log.maneuver == "walk"
        assert loaded_log.audio_record is not None
        assert loaded_log.audio_record.audio_file_name == "test_audio"
        assert loaded_log.audio_record.sample_rate == 46875.0
        assert len(loaded_log.synchronization_records) == 1

    def test_load_from_nonexistent_file(self, tmp_path):
        """Test loading from non-existent file returns None."""
        excel_path = tmp_path / "nonexistent.xlsx"
        loaded_log = ManeuverProcessingLog.load_from_excel(excel_path)

        assert loaded_log is None

    def test_get_or_create_creates_new(self, tmp_path):
        """Test get_or_create creates new log when file doesn't exist."""
        log = ManeuverProcessingLog.get_or_create(
            study_id="1011",
            knee_side="Left",
            maneuver="walk",
            maneuver_directory=tmp_path,
        )

        assert log is not None
        assert log.study_id == "1011"
        assert log.log_created is not None

    def test_get_or_create_loads_existing(self, tmp_path):
        """Test get_or_create loads existing log when file exists."""
        # Create and save a log
        original_log = ManeuverProcessingLog(
            study_id="1011",
            knee_side="Left",
            maneuver="walk",
            maneuver_directory=tmp_path,
            log_created=datetime.now(),
        )
        audio = create_minimal_audio_record(
            audio_file_name="original_audio",
        )
        original_log.update_audio_record(audio)
        original_log.save_to_excel()

        # Get or create should load existing
        log = ManeuverProcessingLog.get_or_create(
            study_id="1011",
            knee_side="Left",
            maneuver="walk",
            maneuver_directory=tmp_path,
        )

        assert log is not None
        assert log.audio_record is not None
        assert log.audio_record.audio_file_name == "original_audio"


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_audio_record_from_data(self, tmp_path):
        """Test creating audio record from DataFrame and metadata."""
        # Create sample audio DataFrame
        n_samples = 1000
        audio_df = pd.DataFrame({
            'tt': pd.date_range('2024-01-01', periods=n_samples, freq='21.333us'),
            'ch1': np.random.randn(n_samples) * 150,
            'ch2': np.random.randn(n_samples) * 148,
            'ch3': np.random.randn(n_samples) * 152,
            'ch4': np.random.randn(n_samples) * 149,
            'f_ch1': np.random.randn(n_samples) * 50,
        })

        metadata = {
            'fs': 46875.0,
            'devFirmwareVersion': 2,
            'deviceSerial': '123456',
        }

        bin_path = tmp_path / "test.bin"
        bin_path.write_bytes(b"test data")

        record = create_audio_record_from_data(
            audio_file_name="test_audio",
            audio_df=audio_df,
            audio_bin_path=bin_path,
            audio_pkl_path=tmp_path / "test.pkl",
            metadata=metadata,
        )

        assert record.processing_status == "success"
        assert record.sample_rate == 46875.0
        assert record.firmware_version == 2
        assert record.device_serial == "123456"
        assert record.duration_seconds is not None
        assert record.file_size_mb is not None

    def test_create_audio_record_with_error(self):
        """Test creating audio record when processing failed."""
        error = ValueError("Processing failed")

        record = create_audio_record_from_data(
            audio_file_name="test_audio",
            error=error,
        )

        assert record.processing_status == "error"
        assert record.error_message == "Processing failed"

    def test_create_biomechanics_record_from_data(self, tmp_path):
        """Test creating biomechanics record from recordings."""
        # Create mock recordings
        class MockRecording:
            def __init__(self, pass_num=None):
                self.pass_number = pass_num
                self.data = pd.DataFrame({
                    'Time (sec)': np.linspace(0, 10, 1000),
                    'Knee Angle Z': np.random.randn(1000),
                })

        recordings = [
            MockRecording(pass_num=1),
            MockRecording(pass_num=2),
            MockRecording(pass_num=3),
        ]

        bio_file = tmp_path / "biomechanics.xlsx"

        record = create_biomechanics_record_from_data(
            biomechanics_file=bio_file,
            recordings=recordings,
            sheet_name="Walk0001",
        )

        assert record.processing_status == "success"
        assert record.num_sub_recordings == 3
        assert record.num_passes == 3
        assert record.duration_seconds is not None
        assert record.sample_rate is not None

    def test_create_sync_record_from_data(self):
        """Test creating sync record from synchronized DataFrame."""
        synced_df = pd.DataFrame({
            'tt': pd.date_range('2024-01-01', periods=500, freq='10ms'),
            'ch1': np.random.randn(500) * 150,
            'Knee Angle Z': np.sin(np.linspace(0, 4*np.pi, 500)) * 30 + 40,
        })

        record = create_sync_record_from_data(
            sync_file_name="left_walk_medium_pass1",
            synced_df=synced_df,
            audio_stomp_time=10.5,
            bio_left_stomp_time=5.2,
            bio_right_stomp_time=5.1,
            knee_side="left",
            pass_number=1,
            speed="medium",
        )

        assert record.processing_status == "success"
        assert record.sync_file_name == "left_walk_medium_pass1"
        assert record.audio_sync_time.total_seconds() == 10.5
        assert record.bio_left_sync_time.total_seconds() == 5.2
        assert record.knee == "left"
        assert record.pass_number == 1
        assert record.speed == "medium"
        assert record.duration_seconds is not None

    def test_create_cycles_record_from_data(self, tmp_path):
        """Test creating cycles record from cycle extraction results."""
        # Create mock cycle DataFrames
        clean_cycles = [
            pd.DataFrame({'tt': np.arange(100), 'ch1': np.random.randn(100)})
            for _ in range(10)
        ]
        outlier_cycles = [
            pd.DataFrame({'tt': np.arange(100), 'ch1': np.random.randn(100)})
            for _ in range(2)
        ]

        output_dir = tmp_path / "MovementCycles"

        record = create_cycles_record_from_data(
            sync_file_name="left_walk_medium_pass1",
            clean_cycles=clean_cycles,
            outlier_cycles=outlier_cycles,
            output_dir=output_dir,
            acoustic_threshold=100.0,
        )

        assert record.processing_status == "success"
        assert record.sync_file_name == "left_walk_medium_pass1"
        assert record.total_cycles_extracted == 12
        assert record.clean_cycles == 10
        assert record.outlier_cycles == 2
        assert record.qc_acoustic_threshold == 100.0


class TestIncrementalUpdates:
    """Tests for incremental update behavior."""

    def test_update_only_audio_preserves_other_records(self, tmp_path):
        """Test that updating audio doesn't affect other records."""
        log = ManeuverProcessingLog(
            study_id="1011",
            knee_side="Left",
            maneuver="walk",
            maneuver_directory=tmp_path,
        )

        # Add sync record
        sync_record = create_minimal_sync_record(
            sync_file_name="test_sync",
        )
        log.add_synchronization_record(sync_record)

        # Update audio record
        audio = create_minimal_audio_record(
            audio_file_name="new_audio",
        )
        log.update_audio_record(audio)

        # Sync record should still be there
        assert len(log.synchronization_records) == 1
        assert log.synchronization_records[0].sync_file_name == "test_sync"
        # Audio record should be updated
        assert log.audio_record.audio_file_name == "new_audio"

    def test_update_sync_record_replaces_existing(self, tmp_path):
        """Test that updating a sync record replaces the old one."""
        log = ManeuverProcessingLog(
            study_id="1011",
            knee_side="Left",
            maneuver="walk",
            maneuver_directory=tmp_path,
        )

        # Add first sync record
        from datetime import timedelta
        sync_record1 = create_minimal_sync_record(
            sync_file_name="test_sync",
            sync_duration=timedelta(seconds=10.0),
        )
        log.add_synchronization_record(sync_record1)

        # Update with new data for same file
        sync_record2 = create_minimal_sync_record(
            sync_file_name="test_sync",
            sync_duration=timedelta(seconds=20.0),
        )
        log.add_synchronization_record(sync_record2)

        # Should only have one record with updated values
        assert len(log.synchronization_records) == 1
        assert log.synchronization_records[0].sync_duration.total_seconds() == 20.0

    def test_roundtrip_preserves_data(self, tmp_path):
        """Test that save and load preserves all data."""
        # Create a comprehensive log
        original_log = ManeuverProcessingLog(
            study_id="1011",
            knee_side="Left",
            maneuver="walk",
            maneuver_directory=tmp_path,
            log_created=datetime(2024, 1, 1, 12, 0, 0),
        )

        # Add all types of records
        audio = create_minimal_audio_record(
            audio_file_name="test_audio",
            sample_rate=46875.0,
        )
        original_log.update_audio_record(audio)

        bio = create_minimal_biomech_record(
            biomechanics_file="test.xlsx",
            num_sub_recordings=3,
            num_passes=9,
        )
        original_log.update_biomechanics_record(bio)

        for i in range(3):
            sync = create_minimal_sync_record(
                sync_file_name=f"sync_{i}",
            )
            original_log.add_synchronization_record(sync)

        for i in range(3):
            cycles = create_minimal_sync_record(
                sync_file_name=f"cycle_{i}",
                clean_cycles=10 + i,
                outlier_cycles=2,
                total_cycles_extracted=12 + i,
            )
            original_log.add_movement_cycles_record(cycles)

        # Save and load
        excel_path = tmp_path / "test_log.xlsx"
        original_log.save_to_excel(excel_path)
        loaded_log = ManeuverProcessingLog.load_from_excel(excel_path)

        # Verify all data is preserved
        assert loaded_log.study_id == "1011"
        assert loaded_log.audio_record.sample_rate == 46875.0
        assert loaded_log.biomechanics_record.num_sub_recordings == 3
        assert len(loaded_log.synchronization_records) == 3
        assert len(loaded_log.movement_cycles_records) == 3
        assert loaded_log.movement_cycles_records[0].clean_cycles == 10
        assert loaded_log.movement_cycles_records[1].clean_cycles == 11
