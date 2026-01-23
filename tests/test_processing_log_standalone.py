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

from src.metadata import (
    AudioProcessing,
    BiomechanicsImport,
    Synchronization,
    MovementCycles,
)
from src.orchestration.processing_log import (
    ManeuverProcessingLog,
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
        now = datetime.now()
        audio_record = AudioProcessing(
            audio_file_name="test_audio",
            processing_status="success",
            processing_date=now,
            sample_rate=46875.0,
            num_channels=4,
            study='AOA',
            study_id=1011,
            recording_date=now,
            recording_time=now,
            knee='right',
            maneuver='fe',
            mic_1_position='IPM',
            mic_2_position='IPL',
            mic_3_position='SPM',
            mic_4_position='SPL',
            device_serial='12345',
            firmware_version=2,
            file_time=now,
            file_size_mb=1.0,
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
            qc_artifact=False,
            qc_artifact_segments=[],
            qc_artifact_ch1=False,
            qc_artifact_segments_ch1=[],
            qc_artifact_ch2=False,
            qc_artifact_segments_ch2=[],
            qc_artifact_ch3=False,
            qc_artifact_segments_ch3=[],
            qc_artifact_ch4=False,
            qc_artifact_segments_ch4=[],
        )
        log.update_audio_record(audio_record)
        assert log.audio_record is not None

        # Test 3: Add biomechanics record
        print("  ✓ Adding biomechanics record...")
        now = datetime.now()
        bio_record = BiomechanicsImport(
            study='AOA',
            study_id=1011,
            biomechanics_file="test.xlsx",
            sheet_name="Sheet1",
            num_sub_recordings=3,
            num_passes=1,
            duration_seconds=10.0,
            sample_rate=100.0,
            num_data_points=1000,
            processing_status="success",
            processing_date=now,
        )
        log.update_biomechanics_record(bio_record)
        assert log.biomechanics_record is not None

        # Test 4: Add sync records
        print("  ✓ Adding synchronization records...")
        from datetime import timedelta
        for i in range(3):
            sync_record = Synchronization(
                study='AOA',
                study_id=1011,
                linked_biomechanics=True,
                biomechanics_file="test.xlsx",
                biomechanics_type="Gonio",
                biomechanics_sample_rate=100.0,
                audio_file_name="test_audio",
                device_serial="12345",
                firmware_version=2,
                file_time=now,
                file_size_mb=1.0,
                recording_date=now,
                recording_time=now,
                knee='right',
                maneuver='fe',
                sample_rate=46875.0,
                num_channels=4,
                mic_1_position='IPM',
                mic_2_position='IPL',
                mic_3_position='SPM',
                mic_4_position='SPL',
                audio_sync_time=timedelta(seconds=0),
                bio_right_sync_time=timedelta(seconds=1),
                sync_offset=timedelta(seconds=0),
                aligned_audio_sync_time=timedelta(seconds=0),
                aligned_bio_sync_time=timedelta(seconds=1),
                sync_method="consensus",
                consensus_time=timedelta(seconds=0),
                rms_time=timedelta(seconds=0),
                onset_time=timedelta(seconds=0),
                freq_time=timedelta(seconds=0),
                sync_file_name=f"sync_{i}",
                sync_duration=timedelta(seconds=10),
                total_cycles_extracted=1000,
                clean_cycles=900,
                outlier_cycles=100,
                mean_cycle_duration_s=0.5,
                median_cycle_duration_s=0.5,
                min_cycle_duration_s=0.4,
                max_cycle_duration_s=0.6,
                mean_acoustic_auc=100.0,
                processing_status="success",
                processing_date=now,
            )
            log.add_synchronization_record(sync_record)
        assert len(log.synchronization_records) == 3

        # Test 5: Add cycles records
        print("  ✓ Adding movement cycles records...")
        for i in range(3):
            cycles_record = Synchronization(
                study='AOA',
                study_id=1011,
                linked_biomechanics=True,
                biomechanics_file="test.xlsx",
                biomechanics_type="Gonio",
                biomechanics_sample_rate=100.0,
                audio_file_name="test_audio",
                device_serial="12345",
                firmware_version=2,
                file_time=now,
                file_size_mb=1.0,
                recording_date=now,
                recording_time=now,
                knee='right',
                maneuver='fe',
                sample_rate=46875.0,
                num_channels=4,
                mic_1_position='IPM',
                mic_2_position='IPL',
                mic_3_position='SPM',
                mic_4_position='SPL',
                audio_sync_time=timedelta(seconds=0),
                bio_right_sync_time=timedelta(seconds=1),
                sync_offset=timedelta(seconds=0),
                aligned_audio_sync_time=timedelta(seconds=0),
                aligned_bio_sync_time=timedelta(seconds=1),
                sync_method="consensus",
                consensus_time=timedelta(seconds=0),
                rms_time=timedelta(seconds=0),
                onset_time=timedelta(seconds=0),
                freq_time=timedelta(seconds=0),
                sync_file_name=f"sync_{i}",
                sync_duration=timedelta(seconds=10),
                total_cycles_extracted=12,
                clean_cycles=10,
                outlier_cycles=2,
                mean_cycle_duration_s=0.5,
                median_cycle_duration_s=0.5,
                min_cycle_duration_s=0.4,
                max_cycle_duration_s=0.6,
                mean_acoustic_auc=100.0,
                processing_status="success",
                processing_date=now,
            )
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
        sync_record_update = Synchronization(
            study='AOA',
            study_id=1011,
            linked_biomechanics=True,
            biomechanics_file="test.xlsx",
            biomechanics_type="Gonio",
            biomechanics_sample_rate=100.0,
            audio_file_name="test_audio",
            device_serial="12345",
            firmware_version=2,
            file_time=now,
            file_size_mb=1.0,
            recording_date=now,
            recording_time=now,
            knee='right',
            maneuver='fe',
            sample_rate=46875.0,
            num_channels=4,
            mic_1_position='IPM',
            mic_2_position='IPL',
            mic_3_position='SPM',
            mic_4_position='SPL',
            audio_sync_time=timedelta(seconds=0),
            bio_right_sync_time=timedelta(seconds=1),
            sync_offset=timedelta(seconds=0),
            aligned_audio_sync_time=timedelta(seconds=0),
            aligned_bio_sync_time=timedelta(seconds=1),
            sync_method="consensus",
            consensus_time=timedelta(seconds=0),
            rms_time=timedelta(seconds=0),
            onset_time=timedelta(seconds=0),
            freq_time=timedelta(seconds=0),
            sync_file_name="sync_0",
            sync_duration=timedelta(seconds=20),  # Updated value
            total_cycles_extracted=2000,
            clean_cycles=1900,
            outlier_cycles=100,
            mean_cycle_duration_s=0.5,
            median_cycle_duration_s=0.5,
            min_cycle_duration_s=0.4,
            max_cycle_duration_s=0.6,
            mean_acoustic_auc=150.0,
            processing_status="success",
            processing_date=now,
        )
        loaded_log.add_synchronization_record(sync_record_update)
        assert len(loaded_log.synchronization_records) == 3  # Still 3
        assert loaded_log.synchronization_records[0].sync_duration.total_seconds() == 20.0

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

        now = datetime.now()
        metadata = {
            'fs': 46875.0,
            'devFirmwareVersion': 2,
            'deviceSerial': [12345],
            'fileTime': now,
            'study': 'AOA',
            'study_id': 1011,
            'recording_date': now,
            'recording_time': now,
            'knee': 'right',
            'maneuver': 'fe',
            'mic_1_position': 'IPM',
            'mic_2_position': 'IPL',
            'mic_3_position': 'SPM',
            'mic_4_position': 'SPL',
            'file_size_mb': 1.0,
        }

        record = create_audio_record_from_data(
            audio_file_name="test_audio",
            audio_df=audio_df,
            metadata=metadata,
        )

        assert record.processing_status == "success"
        assert record.sample_rate == 46875.0

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
            bio_right_stomp_time=5.2,
            knee_side="right",
        )

        assert sync_record.processing_status == "success"
        # Sync duration depends on exact calculation from DataFrame timestamps
        # The calculation: last_tt - first_tt. With 500 samples @ 10ms freq, the duration
        # is from sample 0 to sample 499, which is 499 * 10ms = 4.99 seconds
        assert sync_record.sync_duration.total_seconds() > 4.98
        assert sync_record.audio_sync_time.total_seconds() == 10.5

        print("✅ All helper function tests passed!")


def test_error_handling():
    """Test error handling."""
    print("\nTesting error handling...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Test 1: Create audio record with error
        print("  ✓ Testing error recording...")
        error = ValueError("Processing failed")
        now = datetime.now()
        metadata = {
            'study': 'AOA',
            'study_id': 1011,
            'recording_date': now,
            'recording_time': now,
            'knee': 'right',
            'maneuver': 'fe',
            'mic_1_position': 'IPM',
            'mic_2_position': 'IPL',
            'mic_3_position': 'SPM',
            'mic_4_position': 'SPL',
            'device_serial': '12345',
            'firmware_version': 2,
            'file_size_mb': 1.0,
            'file_time': now,
        }
        record = create_audio_record_from_data(
            audio_file_name="test_audio",
            metadata=metadata,
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
