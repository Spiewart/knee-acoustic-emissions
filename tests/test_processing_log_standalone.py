#!/usr/bin/env python3
"""Simple standalone test to verify processing log functionality.

This test can be run directly without pytest to validate the core
functionality of the processing log system.
"""

import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Add parent directory to path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.orchestration.processing_log import (
    AudioProcessingRecord,
    BiomechanicsImportRecord,
    ManeuverProcessingLog,
    MovementCyclesRecord,
    SynchronizationRecord,
    create_audio_record_from_data,
    create_sync_record_from_data,
)


def test_basic_functionality():
    """Test basic processing log functionality."""
    print("Testing basic processing log functionality...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Test 1: Create a log
        print("  ✓ Creating log...")
        log = ManeuverProcessingLog(
            study_id="1011",
            knee_side="Left",
            maneuver="walk",
            maneuver_directory=tmp_path,
            log_created=datetime.now(),
        )
        assert log.study_id == "1011"

        # Test 2: Add audio record
        print("  ✓ Adding audio record...")
        from src.models import AudioProcessingMetadata
        audio_metadata = AudioProcessingMetadata(
            audio_file_name="test_audio",
            processing_status="success",
            sample_rate=46875.0,
        )
        audio_record = AudioProcessingRecord.from_metadata(audio_metadata)
        log.update_audio_record(audio_record)
        assert log.audio_record is not None

        # Test 3: Add biomechanics record
        print("  ✓ Adding biomechanics record...")
        from src.models import BiomechanicsImportMetadata
        bio_metadata = BiomechanicsImportMetadata(
            biomechanics_file="test.xlsx",
            num_recordings=3,
        )
        bio_record = BiomechanicsImportRecord.from_metadata(bio_metadata)
        log.update_biomechanics_record(bio_record)
        assert log.biomechanics_record is not None

        # Test 4: Add sync records
        print("  ✓ Adding synchronization records...")
        from src.models import SynchronizationMetadata
        for i in range(3):
            sync_metadata = SynchronizationMetadata(
                sync_file_name=f"sync_{i}",
            )
            sync_record = SynchronizationRecord.from_metadata(sync_metadata)
            sync_record.num_synced_samples = 1000
            log.add_synchronization_record(sync_record)
        assert len(log.synchronization_records) == 3

        # Test 5: Add cycles records
        print("  ✓ Adding movement cycles records...")
        from src.models import MovementCyclesMetadata
        for i in range(3):
            cycles_metadata = MovementCyclesMetadata(
                sync_file_name=f"sync_{i}",
                clean_cycles=10,
                outlier_cycles=2,
            )
            cycles_record = MovementCyclesRecord.from_metadata(cycles_metadata)
            log.add_movement_cycles_record(cycles_record)
        assert len(log.movement_cycles_records) == 3

        # Test 6: Save to Excel
        print("  ✓ Saving to Excel...")
        excel_path = tmp_path / "test_log.xlsx"
        saved_path = log.save_to_excel(excel_path)
        assert saved_path.exists()

        # Test 7: Load from Excel
        print("  ✓ Loading from Excel...")
        loaded_log = ManeuverProcessingLog.load_from_excel(excel_path)
        assert loaded_log is not None
        assert loaded_log.study_id == "1011"
        assert loaded_log.audio_record is not None
        assert len(loaded_log.synchronization_records) == 3

        # Test 8: Update existing sync record
        print("  ✓ Testing incremental update...")
        sync_metadata_update = SynchronizationMetadata(
            sync_file_name="sync_0",
        )
        sync_record_update = SynchronizationRecord.from_metadata(sync_metadata_update)
        sync_record_update.num_synced_samples = 2000  # Updated value
        loaded_log.add_synchronization_record(sync_record_update)
        assert len(loaded_log.synchronization_records) == 3  # Still 3
        assert loaded_log.synchronization_records[0].num_synced_samples == 2000

        print("✅ All basic tests passed!")


def test_helper_functions():
    """Test helper functions for creating records."""
    print("\nTesting helper functions...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Test 1: Create audio record from DataFrame
        print("  ✓ Testing create_audio_record_from_data...")
        audio_df = pd.DataFrame({
            'tt': pd.date_range('2024-01-01', periods=1000, freq='21.333us'),
            'ch1': np.random.randn(1000) * 150,
            'ch2': np.random.randn(1000) * 148,
            'ch3': np.random.randn(1000) * 152,
            'ch4': np.random.randn(1000) * 149,
            'f_ch1': np.random.randn(1000) * 50,
        })

        metadata = {
            'fs': 46875.0,
            'devFirmwareVersion': 2,
        }

        record = create_audio_record_from_data(
            audio_file_name="test_audio",
            audio_df=audio_df,
            metadata=metadata,
        )

        assert record.processing_status == "success"
        assert record.sample_rate == 46875.0
        assert record.channel_1_rms is not None
        assert record.has_instantaneous_freq is True

        # Test 2: Create sync record from DataFrame
        print("  ✓ Testing create_sync_record_from_data...")
        synced_df = pd.DataFrame({
            'tt': pd.date_range('2024-01-01', periods=500, freq='10ms'),
            'ch1': np.random.randn(500) * 150,
        })

        sync_record = create_sync_record_from_data(
            sync_file_name="test_sync",
            synced_df=synced_df,
            audio_stomp_time=10.5,
            bio_left_stomp_time=5.2,
            knee_side="left",
        )

        assert sync_record.processing_status == "success"
        assert sync_record.num_synced_samples == 500
        assert sync_record.audio_stomp_time == 10.5

        print("✅ All helper function tests passed!")


def test_error_handling():
    """Test error handling."""
    print("\nTesting error handling...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Test 1: Create audio record with error
        print("  ✓ Testing error recording...")
        error = ValueError("Processing failed")
        record = create_audio_record_from_data(
            audio_file_name="test_audio",
            error=error,
        )

        assert record.processing_status == "error"
        assert record.error_message == "Processing failed"

        # Test 2: Load non-existent file
        print("  ✓ Testing load from non-existent file...")
        loaded_log = ManeuverProcessingLog.load_from_excel(
            tmp_path / "nonexistent.xlsx"
        )
        assert loaded_log is None

        print("✅ All error handling tests passed!")


def main():
    """Run all tests."""
    print("="*60)
    print("Processing Log System Validation")
    print("="*60)

    try:
        test_basic_functionality()
        test_helper_functions()
        test_error_handling()

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nThe processing log system is working correctly.")
        print("Run with pytest for comprehensive test coverage:")
        print("  pytest tests/test_processing_log.py -v")
        return 0

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
