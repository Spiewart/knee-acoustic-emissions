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

from src.orchestration.processing_log import (
    AudioProcessingRecord,
    BiomechanicsImportRecord,
    ManeuverProcessingLog,
    MovementCyclesRecord,
    SynchronizationRecord,
    create_audio_record_from_data,
    create_biomechanics_record_from_data,
    create_cycles_record_from_data,
    create_sync_record_from_data,
)


class TestAudioProcessingRecord:
    """Tests for AudioProcessingRecord."""

    def test_create_record(self):
        """Test creating an audio processing record from validated metadata."""
        from src.models import AudioProcessingMetadata
        
        metadata = AudioProcessingMetadata(
            audio_file_name="test_audio",
            processing_status="success",
            sample_rate=46875.0,
            duration_seconds=120.5,
        )
        record = AudioProcessingRecord.from_metadata(metadata)

        assert record.audio_file_name == "test_audio"
        assert record.processing_status == "success"
        assert record.sample_rate == 46875.0
        assert record.duration_seconds == 120.5
        assert record._metadata is not None

    def test_to_dict(self):
        """Test converting record to dictionary."""
        from src.models import AudioProcessingMetadata
        
        metadata = AudioProcessingMetadata(
            audio_file_name="test_audio",
            processing_status="success",
            sample_rate=46875.0,
            channel_1_rms=150.3,
        )
        record = AudioProcessingRecord.from_metadata(metadata)

        data = record.to_dict()

        assert isinstance(data, dict)
        assert data["Audio File"] == "test_audio"
        assert data["Status"] == "success"
        assert data["Sample Rate (Hz)"] == 46875.0
        assert data["Ch1 RMS"] == 150.3
        assert "Audio QC Version" in data
        assert data["Audio QC Version"] == 1

    def test_default_values(self):
        """Test that default values are set correctly."""
        from src.models import AudioProcessingMetadata
        
        metadata = AudioProcessingMetadata(audio_file_name="test")
        record = AudioProcessingRecord.from_metadata(metadata)

        assert record.processing_status == "not_processed"
        assert record.num_channels == 4
        assert record.sample_rate is None
        assert record.has_instantaneous_freq is False
        assert record.audio_qc_version == 1


class TestBiomechanicsImportRecord:
    """Tests for BiomechanicsImportRecord."""

    def test_create_record(self):
        """Test creating a biomechanics import record from validated metadata."""
        from src.models import BiomechanicsImportMetadata
        
        metadata = BiomechanicsImportMetadata(
            biomechanics_file="AOA1011_Biomechanics_Full_Set.xlsx",
            processing_status="success",
            num_recordings=3,
            num_passes=9,
        )
        record = BiomechanicsImportRecord.from_metadata(metadata)

        assert record.biomechanics_file == "AOA1011_Biomechanics_Full_Set.xlsx"
        assert record.processing_status == "success"
        assert record.num_recordings == 3
        assert record.num_passes == 9
        assert record._metadata is not None

    def test_to_dict(self):
        """Test converting record to dictionary."""
        from src.models import BiomechanicsImportMetadata
        
        metadata = BiomechanicsImportMetadata(
            biomechanics_file="test.xlsx",
            sheet_name="Walk0001",
            num_recordings=5,
        )
        record = BiomechanicsImportRecord.from_metadata(metadata)

        data = record.to_dict()

        assert data["Biomechanics File"] == "test.xlsx"
        assert data["Sheet Name"] == "Walk0001"
        assert data["Num Recordings"] == 5
        assert "Biomech QC Version" in data
        assert data["Biomech QC Version"] == 1


class TestSynchronizationRecord:
    """Tests for SynchronizationRecord."""

    def test_create_record(self):
        """Test creating a synchronization record from validated metadata."""
        from src.models import SynchronizationMetadata
        
        metadata = SynchronizationMetadata(
            sync_file_name="left_walk_slow_pass1",
            pass_number=1,
            speed="slow",
            audio_stomp_time=10.5,
            bio_left_stomp_time=5.2,
            knee_side="left",
        )
        record = SynchronizationRecord.from_metadata(metadata)

        assert record.sync_file_name == "left_walk_slow_pass1"
        assert record.pass_number == 1
        assert record.speed == "slow"
        assert record.audio_stomp_time == 10.5
        assert record.knee_side == "left"

    def test_to_dict(self):
        """Test converting record to dictionary."""
        from src.models import SynchronizationMetadata
        metadata = SynchronizationMetadata(
            sync_file_name="test_sync",
            processing_status="success",
            num_synced_samples=1000,
        )
        record = SynchronizationRecord.from_metadata(metadata)

        data = record.to_dict()

        assert data["Sync File"] == "test_sync"
        assert data["Status"] == "success"
        assert data["Num Samples"] == 1000
        assert "Audio QC Version" in data
        assert data["Audio QC Version"] == 1
        assert "Biomech QC Version" in data
        assert data["Biomech QC Version"] == 1


class TestMovementCyclesRecord:
    """Tests for MovementCyclesRecord."""

    def test_create_record(self):
        """Test creating a movement cycles record."""
        from src.models import MovementCyclesMetadata
        metadata = MovementCyclesMetadata(
            sync_file_name="test_sync",
            total_cycles_extracted=12,
            clean_cycles=10,
            outlier_cycles=2,
        )
        record = MovementCyclesRecord.from_metadata(metadata)

        assert record.sync_file_name == "test_sync"
        assert record.total_cycles_extracted == 12
        assert record.clean_cycles == 10
        assert record.outlier_cycles == 2

    def test_to_dict(self):
        """Test converting record to dictionary."""
        from src.models import MovementCyclesMetadata
        metadata = MovementCyclesMetadata(
            sync_file_name="test_sync",
            clean_cycles=8,
            outlier_cycles=2,
        )
        record = MovementCyclesRecord.from_metadata(metadata)

        data = record.to_dict()

        assert data["Source Sync File"] == "test_sync"
        assert data["Clean Cycles"] == 8
        assert data["Outlier Cycles"] == 2
        assert "Cycle QC Version" in data
        assert data["Cycle QC Version"] == 1


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
        from src.models import AudioProcessingMetadata
        log = ManeuverProcessingLog(
            study_id="1011",
            knee_side="Left",
            maneuver="walk",
            maneuver_directory=tmp_path,
        )

        metadata = AudioProcessingMetadata(
            audio_file_name="test_audio",
            processing_status="success",
        )
        audio_record = AudioProcessingRecord.from_metadata(metadata)

        log.update_audio_record(audio_record)

        assert log.audio_record is not None
        assert log.audio_record.audio_file_name == "test_audio"
        assert log.log_updated is not None

    def test_add_synchronization_record(self, tmp_path):
        """Test adding synchronization record."""
        from src.models import SynchronizationMetadata
        log = ManeuverProcessingLog(
            study_id="1011",
            knee_side="Left",
            maneuver="walk",
            maneuver_directory=tmp_path,
        )

        metadata = SynchronizationMetadata(
            sync_file_name="test_sync",
            processing_status="success",
        )
        sync_record = SynchronizationRecord.from_metadata(metadata)

        log.add_synchronization_record(sync_record)

        assert len(log.synchronization_records) == 1
        assert log.synchronization_records[0].sync_file_name == "test_sync"

    def test_update_existing_synchronization_record(self, tmp_path):
        """Test that adding a record with same filename updates existing."""
        from src.models import SynchronizationMetadata
        log = ManeuverProcessingLog(
            study_id="1011",
            knee_side="Left",
            maneuver="walk",
            maneuver_directory=tmp_path,
        )

        # Add first record
        metadata1 = SynchronizationMetadata(
            sync_file_name="test_sync",
            num_synced_samples=1000,
        )
        sync_record1 = SynchronizationRecord.from_metadata(metadata1)
        log.add_synchronization_record(sync_record1)

        # Add second record with same name
        metadata2 = SynchronizationMetadata(
            sync_file_name="test_sync",
            num_synced_samples=2000,
        )
        sync_record2 = SynchronizationRecord.from_metadata(metadata2)
        log.add_synchronization_record(sync_record2)

        # Should only have one record, with updated values
        assert len(log.synchronization_records) == 1
        assert log.synchronization_records[0].num_synced_samples == 2000

    def test_add_movement_cycles_record(self, tmp_path):
        """Test adding movement cycles record."""
        from src.models import MovementCyclesMetadata
        log = ManeuverProcessingLog(
            study_id="1011",
            knee_side="Left",
            maneuver="walk",
            maneuver_directory=tmp_path,
        )

        metadata = MovementCyclesMetadata(
            sync_file_name="test_sync",
            clean_cycles=10,
            outlier_cycles=2,
        )
        cycles_record = MovementCyclesRecord.from_metadata(metadata)

        log.add_movement_cycles_record(cycles_record)

        assert len(log.movement_cycles_records) == 1
        assert log.movement_cycles_records[0].clean_cycles == 10

    def test_save_to_excel(self, tmp_path):
        """Test saving log to Excel file."""
        from src.models import AudioProcessingMetadata, SynchronizationMetadata
        log = ManeuverProcessingLog(
            study_id="1011",
            knee_side="Left",
            maneuver="walk",
            maneuver_directory=tmp_path,
            log_created=datetime.now(),
            log_updated=datetime.now(),
        )

        # Add some data
        audio_metadata = AudioProcessingMetadata(
            audio_file_name="test_audio",
            processing_status="success",
        )
        log.update_audio_record(AudioProcessingRecord.from_metadata(audio_metadata))

        sync_metadata = SynchronizationMetadata(
            sync_file_name="test_sync",
            processing_status="success",
        )
        log.add_synchronization_record(SynchronizationRecord.from_metadata(sync_metadata))

        # Save to Excel
        excel_path = tmp_path / "test_log.xlsx"
        saved_path = log.save_to_excel(excel_path)

        assert saved_path.exists()
        assert saved_path == excel_path

    def test_load_from_excel(self, tmp_path):
        """Test loading log from Excel file."""
        from src.models import AudioProcessingMetadata, SynchronizationMetadata
        # Create and save a log
        original_log = ManeuverProcessingLog(
            study_id="1011",
            knee_side="Left",
            maneuver="walk",
            maneuver_directory=tmp_path,
            log_created=datetime.now(),
            log_updated=datetime.now(),
        )

        audio_metadata = AudioProcessingMetadata(
            audio_file_name="test_audio",
            sample_rate=46875.0,
            processing_status="success",
        )
        original_log.update_audio_record(AudioProcessingRecord.from_metadata(audio_metadata))

        sync_metadata = SynchronizationMetadata(
            sync_file_name="test_sync",
            num_synced_samples=1000,
        )
        original_log.add_synchronization_record(SynchronizationRecord.from_metadata(sync_metadata))

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
        assert loaded_log.synchronization_records[0].num_synced_samples == 1000

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
        from src.models import AudioProcessingMetadata
        # Create and save a log
        original_log = ManeuverProcessingLog(
            study_id="1011",
            knee_side="Left",
            maneuver="walk",
            maneuver_directory=tmp_path,
            log_created=datetime.now(),
        )
        metadata = AudioProcessingMetadata(
            audio_file_name="original_audio",
        )
        original_log.update_audio_record(AudioProcessingRecord.from_metadata(metadata))
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
        assert record.channel_1_rms is not None
        assert record.channel_1_peak is not None
        assert record.has_instantaneous_freq is True
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
        assert record.num_recordings == 3
        assert record.num_passes == 3
        assert record.start_time is not None
        assert record.end_time is not None
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
        assert record.audio_stomp_time == 10.5
        assert record.bio_left_stomp_time == 5.2
        assert record.knee_side == "left"
        assert record.pass_number == 1
        assert record.speed == "medium"
        assert record.num_synced_samples == 500
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
            plots_created=True,
        )

        assert record.processing_status == "success"
        assert record.sync_file_name == "left_walk_medium_pass1"
        assert record.total_cycles_extracted == 12
        assert record.clean_cycles == 10
        assert record.outlier_cycles == 2
        assert record.acoustic_threshold == 100.0
        assert record.plots_created is True


class TestIncrementalUpdates:
    """Tests for incremental update behavior."""

    def test_update_only_audio_preserves_other_records(self, tmp_path):
        """Test that updating audio doesn't affect other records."""
        from src.models import AudioProcessingMetadata, SynchronizationMetadata
        log = ManeuverProcessingLog(
            study_id="1011",
            knee_side="Left",
            maneuver="walk",
            maneuver_directory=tmp_path,
        )

        # Add sync record
        sync_metadata = SynchronizationMetadata(
            sync_file_name="test_sync",
            num_synced_samples=1000,
        )
        log.add_synchronization_record(SynchronizationRecord.from_metadata(sync_metadata))

        # Update audio record
        audio_metadata = AudioProcessingMetadata(
            audio_file_name="new_audio",
        )
        log.update_audio_record(AudioProcessingRecord.from_metadata(audio_metadata))

        # Sync record should still be there
        assert len(log.synchronization_records) == 1
        assert log.synchronization_records[0].num_synced_samples == 1000
        # Audio record should be updated
        assert log.audio_record.audio_file_name == "new_audio"

    def test_update_sync_record_replaces_existing(self, tmp_path):
        """Test that updating a sync record replaces the old one."""
        from src.models import SynchronizationMetadata
        log = ManeuverProcessingLog(
            study_id="1011",
            knee_side="Left",
            maneuver="walk",
            maneuver_directory=tmp_path,
        )

        # Add first sync record
        metadata1 = SynchronizationMetadata(
            sync_file_name="test_sync",
            num_synced_samples=1000,
            duration_seconds=10.0,
        )
        log.add_synchronization_record(SynchronizationRecord.from_metadata(metadata1))

        # Update with new data for same file
        metadata2 = SynchronizationMetadata(
            sync_file_name="test_sync",
            num_synced_samples=2000,
            duration_seconds=20.0,
        )
        log.add_synchronization_record(SynchronizationRecord.from_metadata(metadata2))

        # Should only have one record with updated values
        assert len(log.synchronization_records) == 1
        assert log.synchronization_records[0].num_synced_samples == 2000
        assert log.synchronization_records[0].duration_seconds == 20.0

    def test_roundtrip_preserves_data(self, tmp_path):
        """Test that save and load preserves all data."""
        from src.models import AudioProcessingMetadata, BiomechanicsImportMetadata, SynchronizationMetadata, MovementCyclesMetadata
        # Create a comprehensive log
        original_log = ManeuverProcessingLog(
            study_id="1011",
            knee_side="Left",
            maneuver="walk",
            maneuver_directory=tmp_path,
            log_created=datetime(2024, 1, 1, 12, 0, 0),
        )

        # Add all types of records
        audio_metadata = AudioProcessingMetadata(
            audio_file_name="test_audio",
            sample_rate=46875.0,
            channel_1_rms=150.3,
            has_instantaneous_freq=True,
        )
        original_log.update_audio_record(AudioProcessingRecord.from_metadata(audio_metadata))

        bio_metadata = BiomechanicsImportMetadata(
            biomechanics_file="test.xlsx",
            num_recordings=3,
            num_passes=9,
        )
        original_log.update_biomechanics_record(BiomechanicsImportRecord.from_metadata(bio_metadata))

        for i in range(3):
            sync_metadata = SynchronizationMetadata(
                sync_file_name=f"sync_{i}",
                num_synced_samples=1000 + i,
            )
            original_log.add_synchronization_record(SynchronizationRecord.from_metadata(sync_metadata))

        for i in range(3):
            cycles_metadata = MovementCyclesMetadata(
                sync_file_name=f"sync_{i}",
                clean_cycles=10 + i,
                outlier_cycles=2,
            )
            original_log.add_movement_cycles_record(MovementCyclesRecord.from_metadata(cycles_metadata))

        # Save and load
        excel_path = tmp_path / "test_log.xlsx"
        original_log.save_to_excel(excel_path)
        loaded_log = ManeuverProcessingLog.load_from_excel(excel_path)

        # Verify all data is preserved
        assert loaded_log.study_id == "1011"
        assert loaded_log.audio_record.sample_rate == 46875.0
        assert loaded_log.audio_record.channel_1_rms == 150.3
        assert loaded_log.audio_record.has_instantaneous_freq is True
        assert loaded_log.biomechanics_record.num_recordings == 3
        assert len(loaded_log.synchronization_records) == 3
        assert len(loaded_log.movement_cycles_records) == 3
        assert loaded_log.synchronization_records[0].num_synced_samples == 1000
        assert loaded_log.synchronization_records[1].num_synced_samples == 1001
        assert loaded_log.movement_cycles_records[0].clean_cycles == 10
        assert loaded_log.movement_cycles_records[1].clean_cycles == 11
