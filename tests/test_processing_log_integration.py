"""Integration tests for processing log within the orchestration layer.

These tests verify that the processing log is correctly integrated into
the participant processing workflow.
"""

import json

import numpy as np
import pandas as pd

from src.orchestration.participant import _save_or_update_processing_log
from src.orchestration.processing_log import ManeuverProcessingLog


class TestProcessingLogIntegration:
    """Integration tests for processing log."""

    def test_save_or_update_creates_log(self, tmp_path):
        """Test that _save_or_update_processing_log creates a log file."""
        maneuver_dir = tmp_path / "Left Knee" / "Walking"
        maneuver_dir.mkdir(parents=True)

        # Create sample audio data
        audio_df = pd.DataFrame({
            'tt': pd.date_range('2024-01-01', periods=100, freq='21.333us'),
            'ch1': np.random.randn(100) * 150,
            'ch2': np.random.randn(100) * 148,
        })

        audio_pkl_file = maneuver_dir / "audio.pkl"
        audio_df.to_pickle(audio_pkl_file)

        # Create metadata
        metadata = {'fs': 46875.0, 'devFirmwareVersion': 2}
        meta_file = maneuver_dir / "audio_meta.json"
        with open(meta_file, 'w') as f:
            json.dump(metadata, f)

        # Call the function
        _save_or_update_processing_log(
            study_id="1011",
            knee_side="Left",
            maneuver_key="walk",
            maneuver_dir=maneuver_dir,
            audio_pkl_file=audio_pkl_file,
            audio_df=audio_df,
            audio_metadata=metadata,
        )

        # Verify log was created
        log_file = maneuver_dir / "processing_log_1011_Left_walk.xlsx"
        assert log_file.exists()

        # Verify log contains data
        loaded_log = ManeuverProcessingLog.load_from_excel(log_file)
        assert loaded_log is not None
        assert loaded_log.study_id == "1011"
        assert loaded_log.audio_record is not None
        assert loaded_log.audio_record.sample_rate == 46875.0

    def test_incremental_update_preserves_existing_data(self, tmp_path):
        """Test that updating a log preserves existing data."""
        maneuver_dir = tmp_path / "Left Knee" / "Walking"
        maneuver_dir.mkdir(parents=True)

        # Create initial log with audio data
        audio_df = pd.DataFrame({
            'tt': pd.date_range('2024-01-01', periods=100, freq='21.333us'),
            'ch1': np.random.randn(100) * 150,
        })

        audio_pkl_file = maneuver_dir / "audio.pkl"

        _save_or_update_processing_log(
            study_id="1011",
            knee_side="Left",
            maneuver_key="walk",
            maneuver_dir=maneuver_dir,
            audio_pkl_file=audio_pkl_file,
            audio_df=audio_df,
            audio_metadata={'fs': 46875.0},
        )

        # Now update with sync data
        synced_df = pd.DataFrame({
            'tt': pd.date_range('2024-01-01', periods=50, freq='10ms'),
            'ch1': np.random.randn(50),
        })

        output_path = maneuver_dir / "Synced" / "left_walk_slow_pass1.pkl"
        output_path.parent.mkdir(parents=True)

        synced_data = [
            (output_path, synced_df, (10.5, 5.2, None))
        ]

        _save_or_update_processing_log(
            study_id="1011",
            knee_side="Left",
            maneuver_key="walk",
            maneuver_dir=maneuver_dir,
            synced_data=synced_data,
        )

        # Verify both audio and sync data are present
        log_file = maneuver_dir / "processing_log_1011_Left_walk.xlsx"
        loaded_log = ManeuverProcessingLog.load_from_excel(log_file)

        assert loaded_log is not None
        assert loaded_log.audio_record is not None  # Original data preserved
        assert loaded_log.audio_record.sample_rate == 46875.0
        assert len(loaded_log.synchronization_records) == 1  # New data added
        # audio_sync_time is in timedeltas, convert to seconds for comparison
        assert loaded_log.synchronization_records[0].audio_sync_time.total_seconds() == 10.5

    def test_log_survives_multiple_updates(self, tmp_path):
        """Test that log survives multiple update cycles."""
        maneuver_dir = tmp_path / "Left Knee" / "Walking"
        maneuver_dir.mkdir(parents=True)

        # Update 1: Audio
        _save_or_update_processing_log(
            study_id="1011",
            knee_side="Left",
            maneuver_key="walk",
            maneuver_dir=maneuver_dir,
            audio_pkl_file=maneuver_dir / "audio.pkl",
            audio_df=pd.DataFrame({
                'tt': pd.date_range('2024-01-01', periods=100, freq='21.333us'),
                'ch1': np.random.randn(100),
            }),
            audio_metadata={'fs': 46875.0},
        )

        # Update 2: Add sync data
        synced_data = [
            (
                maneuver_dir / "Synced" / "left_walk_slow_pass1.pkl",
                pd.DataFrame({'tt': pd.date_range('2024-01-01', periods=50, freq='10ms')}),
                (10.5, 5.2, None)
            )
        ]
        _save_or_update_processing_log(
            study_id="1011",
            knee_side="Left",
            maneuver_key="walk",
            maneuver_dir=maneuver_dir,
            synced_data=synced_data,
        )

        # Update 3: Update the same sync file with new data
        synced_data_updated = [
            (
                maneuver_dir / "Synced" / "left_walk_slow_pass1.pkl",
                pd.DataFrame({'tt': pd.date_range('2024-01-01', periods=100, freq='10ms')}),
                (11.0, 5.5, None)  # Different stomp times
            )
        ]
        _save_or_update_processing_log(
            study_id="1011",
            knee_side="Left",
            maneuver_key="walk",
            maneuver_dir=maneuver_dir,
            synced_data=synced_data_updated,
        )

        # Verify final state
        log_file = maneuver_dir / "processing_log_1011_Left_walk.xlsx"
        loaded_log = ManeuverProcessingLog.load_from_excel(log_file)

        assert loaded_log is not None
        assert loaded_log.audio_record is not None
        assert len(loaded_log.synchronization_records) == 1  # Still just one
        # Should have updated stomp times
        assert loaded_log.synchronization_records[0].audio_sync_time.total_seconds() == 11.0
        # Verify the sync duration changed (more samples)
        assert loaded_log.synchronization_records[0].sync_duration.total_seconds() > 0

    def test_get_or_create_behavior(self, tmp_path):
        """Test get_or_create loads existing log correctly."""
        maneuver_dir = tmp_path / "Left Knee" / "Walking"
        maneuver_dir.mkdir(parents=True)

        # Create initial log with audio data
        audio_df = pd.DataFrame({
            'tt': pd.date_range('2024-01-01', periods=100, freq='21.333us'),
            'ch1': np.random.randn(100) * 150,
        })

        _save_or_update_processing_log(
            study_id="1011",
            knee_side="Left",
            maneuver_key="walk",
            maneuver_dir=maneuver_dir,
            audio_pkl_file=maneuver_dir / "audio.pkl",
            audio_df=audio_df,
            audio_metadata={'fs': 46875.0},
        )

        # Get or create should load the existing log
        log2 = ManeuverProcessingLog.get_or_create(
            study_id="1011",
            knee_side="Left",
            maneuver="walk",
            maneuver_directory=maneuver_dir,
        )

        # Should have the original data
        assert log2.audio_record is not None
        assert log2.audio_record.audio_file_name == "audio"
        assert log2.audio_record.sample_rate == 46875.0
