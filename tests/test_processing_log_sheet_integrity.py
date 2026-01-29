"""Integration tests for processing log sheet integrity and update behavior.

These tests ensure that:
1. Method Agreement Span persists through save/load and update cycles
2. Movement Cycles sheet has correct columns (Knee, not Knee Side)
3. Log Updated timestamp is present in Movement Cycles
"""

from datetime import datetime

import numpy as np
import pandas as pd

from src.metadata import Synchronization
from src.orchestration.processing_log import (
    ManeuverProcessingLog,
    create_cycles_record_from_data,
    create_sync_record_from_data,
)


class TestProcessingLogSheetIntegrity:
    """Test Excel sheet integrity through save/load/update cycles."""

    def test_method_agreement_span_persists_through_updates(self, tmp_path):
        """Test that method_agreement_span is preserved when updating existing records."""
        maneuver_dir = tmp_path / "Left Knee" / "Walking"
        maneuver_dir.mkdir(parents=True)

        # Create initial log with sync record
        log = ManeuverProcessingLog(
            study_id="1011",
            knee_side="Left",
            maneuver="walk",
            maneuver_directory=maneuver_dir,
            log_created=datetime(2024, 1, 1),
        )

        # Create synced dataframe with detection results
        synced_df = pd.DataFrame({
            'tt': np.arange(0, 10.0, 0.01),
            'ch1': np.random.randn(1000),
        })

        # Create sync record with method_agreement_span
        detection_results = {
            "consensus_time": 5.0,
            "rms_time": 4.8,
            "onset_time": 5.1,
            "freq_time": 5.2,
            "consensus_methods": ["rms", "onset", "freq"],
        }

        sync_record = create_sync_record_from_data(
            sync_file_name="test_sync",
            synced_df=synced_df,
            pass_number=1,
            speed="normal",
            detection_results=detection_results,
            error=None,
            audio_record=None,
            biomech_record=None,
            metadata={"maneuver": "walk"},
            study="AOA",
            study_id=1011,
        )

        # Verify method_agreement_span was calculated
        expected_span = 5.2 - 4.8  # 0.4
        assert abs(sync_record.method_agreement_span - expected_span) < 0.001, \
            f"Initial method_agreement_span should be {expected_span}, got {sync_record.method_agreement_span}"

        # Add to log and save
        log.add_synchronization_record(sync_record)
        excel_path = maneuver_dir / "test_log.xlsx"
        log.save_to_excel(excel_path)

        # Verify in Excel
        sync_df = pd.read_excel(excel_path, sheet_name="Synchronization")
        assert abs(sync_df["Method Agreement Span (s)"].iloc[0] - expected_span) < 0.001, \
            f"Method Agreement Span should be {expected_span} in saved Excel"

        # SIMULATE REPROCESSING: Load log and add cycles record
        loaded_log = ManeuverProcessingLog.load_from_excel(excel_path)
        assert loaded_log is not None, "Failed to load log from Excel"

        # Verify method_agreement_span survived load
        assert abs(loaded_log.synchronization_records[0].method_agreement_span - expected_span) < 0.001, \
            f"Method Agreement Span should be {expected_span} after load, got {loaded_log.synchronization_records[0].method_agreement_span}"

        # Create cycle data and cycles record (simulating cycle extraction)
        cycle_data = pd.DataFrame({
            'tt': np.arange(0, 1.0, 0.01),
            'ch1': np.random.randn(100),
        })

        cycles_record = create_cycles_record_from_data(
            sync_file_name="test_sync",
            clean_cycles=[cycle_data],
            outlier_cycles=[],
            pass_number=1,
            speed="normal",
            output_dir=maneuver_dir,
            plots_created=False,
            error=None,
            audio_record=None,
            biomech_record=None,
            sync_record=sync_record,
            metadata={},
            study="AOA",
            study_id=1011,
        )

        # Verify cycles_record inherited method_agreement_span
        assert abs(cycles_record.method_agreement_span - expected_span) < 0.001, \
            f"Cycles record should inherit method_agreement_span={expected_span}, got {cycles_record.method_agreement_span}"

        # Add cycles record to loaded log (simulating update)
        loaded_log.add_movement_cycles_record(cycles_record)

        # CRITICAL: Verify method_agreement_span in sync record wasn't lost
        sync_rec_after_update = loaded_log.synchronization_records[0]
        assert abs(sync_rec_after_update.method_agreement_span - expected_span) < 0.001, \
            f"Method Agreement Span should still be {expected_span} after adding cycles, got {sync_rec_after_update.method_agreement_span}"

        # Save updated log
        loaded_log.save_to_excel(excel_path)

        # Reload and verify method_agreement_span persists
        final_log = ManeuverProcessingLog.load_from_excel(excel_path)
        assert abs(final_log.synchronization_records[0].method_agreement_span - expected_span) < 0.001, \
            f"Method Agreement Span should be {expected_span} after full cycle, got {final_log.synchronization_records[0].method_agreement_span}"

        # Verify in final Excel
        final_sync_df = pd.read_excel(excel_path, sheet_name="Synchronization")
        assert abs(final_sync_df["Method Agreement Span (s)"].iloc[0] - expected_span) < 0.001, \
            f"Method Agreement Span should be {expected_span} in final Excel"

    def test_movement_cycles_has_correct_columns(self, tmp_path):
        """Test that Movement Cycles sheet has 'Knee' but not 'Knee Side'."""
        maneuver_dir = tmp_path / "Left Knee" / "Walking"
        maneuver_dir.mkdir(parents=True)

        log = ManeuverProcessingLog(
            study_id="1011",
            knee_side="Left",
            maneuver="walk",
            maneuver_directory=maneuver_dir,
            log_created=datetime(2024, 1, 1),
        )

        # Create sync record
        sync_record = Synchronization(
            study="AOA",
            study_id=1011,
            linked_biomechanics=True,
            biomechanics_file="test.xlsx",
            biomechanics_type="Motion Analysis",
            biomechanics_sync_method="stomp",
            biomechanics_sample_rate=100.0,
            audio_file_name="test_audio.bin",
            device_serial="TEST123",
            firmware_version=1,
            file_time=datetime(2024, 1, 1),
            file_size_mb=100.0,
            recording_date=datetime(2024, 1, 1),
            recording_time=datetime(2024, 1, 1, 10, 0, 0),
            knee="left",
            maneuver="walk",
            pass_number=1,
            speed="normal",
            num_channels=4,
            mic_1_position="IPM",
            mic_2_position="IPL",
            mic_3_position="SPM",
            mic_4_position="SPL",
            audio_sync_time=5.0,
            bio_left_sync_time=10.0,
            sync_offset=5.0,
            aligned_audio_sync_time=10.0,
            aligned_biomechanics_sync_time=10.0,
            sync_method="consensus",
            consensus_methods="RMS,Onset,Frequency",
            consensus_time=5.0,
            rms_time=5.0,
            onset_time=5.0,
            freq_time=5.0,
            sync_file_name="test_sync",
            processing_date=datetime(2024, 1, 1),
            sync_duration=120.0,
            total_cycles_extracted=0,
            clean_cycles=0,
            outlier_cycles=0,
        )

        # Create cycles record
        cycle_data = pd.DataFrame({
            'tt': np.arange(0, 1.0, 0.01),
            'ch1': np.random.randn(100),
        })

        cycles_record = create_cycles_record_from_data(
            sync_file_name="test_sync",
            clean_cycles=[cycle_data],
            outlier_cycles=[],
            pass_number=1,
            speed="normal",
            output_dir=maneuver_dir,
            plots_created=False,
            error=None,
            audio_record=None,
            biomech_record=None,
            sync_record=sync_record,
            metadata={},
            study="AOA",
            study_id=1011,
        )

        log.add_movement_cycles_record(cycles_record)

        # Save and check columns
        excel_path = maneuver_dir / "test_log.xlsx"
        log.save_to_excel(excel_path)

        mc_df = pd.read_excel(excel_path, sheet_name="Movement Cycles")

        # CRITICAL ASSERTIONS:
        assert "Knee" in mc_df.columns, "Movement Cycles should have 'Knee' column"
        assert "Knee Side" not in mc_df.columns, \
            "Movement Cycles should NOT have redundant 'Knee Side' column"

        # Verify knee value is present
        assert mc_df["Knee"].iloc[0] == "left", \
            f"Knee should be 'left', got {mc_df['Knee'].iloc[0]}"

    def test_movement_cycles_has_log_updated(self, tmp_path):
        """Test that Movement Cycles sheet has Log Updated timestamp."""
        maneuver_dir = tmp_path / "Left Knee" / "Walking"
        maneuver_dir.mkdir(parents=True)

        log = ManeuverProcessingLog(
            study_id="1011",
            knee_side="Left",
            maneuver="walk",
            maneuver_directory=maneuver_dir,
            log_created=datetime(2024, 1, 1),
        )

        # Create sync record
        sync_record = Synchronization(
            study="AOA",
            study_id=1011,
            linked_biomechanics=True,
            biomechanics_file="test.xlsx",
            biomechanics_type="Motion Analysis",
            biomechanics_sync_method="stomp",
            biomechanics_sample_rate=100.0,
            audio_file_name="test_audio.bin",
            device_serial="TEST123",
            firmware_version=1,
            file_time=datetime(2024, 1, 1),
            file_size_mb=100.0,
            recording_date=datetime(2024, 1, 1),
            recording_time=datetime(2024, 1, 1, 10, 0, 0),
            knee="left",
            maneuver="walk",
            pass_number=1,
            speed="normal",
            num_channels=4,
            mic_1_position="IPM",
            mic_2_position="IPL",
            mic_3_position="SPM",
            mic_4_position="SPL",
            audio_sync_time=5.0,
            bio_left_sync_time=10.0,
            sync_offset=5.0,
            aligned_audio_sync_time=10.0,
            aligned_biomechanics_sync_time=10.0,
            sync_method="consensus",
            consensus_methods="RMS,Onset,Frequency",
            consensus_time=5.0,
            rms_time=5.0,
            onset_time=5.0,
            freq_time=5.0,
            sync_file_name="test_sync",
            processing_date=datetime(2024, 1, 1),
            sync_duration=120.0,
            total_cycles_extracted=0,
            clean_cycles=0,
            outlier_cycles=0,
        )

        # Create cycles record
        cycle_data = pd.DataFrame({
            'tt': np.arange(0, 1.0, 0.01),
            'ch1': np.random.randn(100),
        })

        cycles_record = create_cycles_record_from_data(
            sync_file_name="test_sync",
            clean_cycles=[cycle_data],
            outlier_cycles=[],
            pass_number=1,
            speed="normal",
            output_dir=maneuver_dir,
            plots_created=False,
            error=None,
            audio_record=None,
            biomech_record=None,
            sync_record=sync_record,
            metadata={},
            study="AOA",
            study_id=1011,
        )

        log.add_movement_cycles_record(cycles_record)

        # Save and check Log Updated
        excel_path = maneuver_dir / "test_log.xlsx"
        log.save_to_excel(excel_path)

        mc_df = pd.read_excel(excel_path, sheet_name="Movement Cycles")

        # CRITICAL ASSERTIONS:
        assert "Log Updated" in mc_df.columns, \
            "Movement Cycles should have 'Log Updated' column"

        log_updated_value = mc_df["Log Updated"].iloc[0]
        assert pd.notna(log_updated_value), \
            "Log Updated should have a value (not NaN)"

        # Should be a datetime
        assert isinstance(pd.to_datetime(log_updated_value), pd.Timestamp), \
            "Log Updated should be a valid datetime"


class TestMethodAgreementSpanCalculation:
    """Test method_agreement_span calculation in various scenarios."""

    def test_single_method_span_is_zero(self, tmp_path):
        """Test that method_agreement_span is 0 when only one method contributes."""
        maneuver_dir = tmp_path / "Left Knee" / "Walking"
        maneuver_dir.mkdir(parents=True)

        # Create synced dataframe
        synced_df = pd.DataFrame({
            'tt': np.arange(0, 10.0, 0.01),
            'ch1': np.random.randn(1000),
        })

        # Only RMS method contributes
        detection_results = {
            "consensus_time": 5.0,
            "rms_time": 5.0,
            "onset_time": 4.5,
            "freq_time": 5.5,
            "consensus_methods": ["rms"],  # Only RMS contributed
        }

        sync_record = create_sync_record_from_data(
            sync_file_name="test_sync",
            synced_df=synced_df,
            pass_number=1,
            speed="normal",
            detection_results=detection_results,
            audio_record=None,
            biomech_record=None,
            metadata={},
            study="AOA",
            study_id=1011,
        )

        # CRITICAL: With only one method, span should be 0
        assert sync_record.method_agreement_span == 0.0, \
            f"Method agreement span should be 0 with single method, got {sync_record.method_agreement_span}"
        assert sync_record.consensus_methods == "rms", \
            f"Consensus methods should be 'rms', got {sync_record.consensus_methods}"

    def test_two_methods_span_is_difference(self, tmp_path):
        """Test that method_agreement_span is the difference between two methods."""
        maneuver_dir = tmp_path / "Left Knee" / "Walking"
        maneuver_dir.mkdir(parents=True)

        synced_df = pd.DataFrame({
            'tt': np.arange(0, 10.0, 0.01),
            'ch1': np.random.randn(1000),
        })

        # RMS and Onset methods contribute
        detection_results = {
            "consensus_time": 5.0,
            "rms_time": 4.7,      # Earlier
            "onset_time": 5.2,    # Later
            "freq_time": 6.0,     # Not used
            "consensus_methods": ["rms", "onset"],
        }

        sync_record = create_sync_record_from_data(
            sync_file_name="test_sync",
            synced_df=synced_df,
            detection_results=detection_results,
            audio_record=None,
            biomech_record=None,
            metadata={"pass_number": 1, "speed": "normal"},
            study="AOA",
            study_id=1011,
            pass_number=1,
            speed="normal",
        )

        expected_span = 5.2 - 4.7  # 0.5
        assert abs(sync_record.method_agreement_span - expected_span) < 0.001, \
            f"Method agreement span should be {expected_span}, got {sync_record.method_agreement_span}"
        assert "rms" in sync_record.consensus_methods and "onset" in sync_record.consensus_methods, \
            f"Consensus methods should contain 'rms' and 'onset', got {sync_record.consensus_methods}"

    def test_three_methods_span_is_range(self, tmp_path):
        """Test that method_agreement_span is max-min with three methods."""
        maneuver_dir = tmp_path / "Left Knee" / "Walking"
        maneuver_dir.mkdir(parents=True)

        synced_df = pd.DataFrame({
            'tt': np.arange(0, 10.0, 0.01),
            'ch1': np.random.randn(1000),
        })

        # All three methods contribute with different times
        detection_results = {
            "consensus_time": 5.0,
            "rms_time": 4.8,      # Earliest
            "onset_time": 5.1,    # Middle
            "freq_time": 5.2,     # Latest
            "consensus_methods": ["rms", "onset", "freq"],
        }

        sync_record = create_sync_record_from_data(
            sync_file_name="test_sync",
            synced_df=synced_df,
            detection_results=detection_results,
            audio_record=None,
            biomech_record=None,
            metadata={"pass_number": 1, "speed": "normal"},
            study="AOA",
            study_id=1011,
            pass_number=1,
            speed="normal",
        )

        expected_span = 5.2 - 4.8  # 0.4
        assert abs(sync_record.method_agreement_span - expected_span) < 0.001, \
            f"Method agreement span should be {expected_span}, got {sync_record.method_agreement_span}"
        assert all(method in sync_record.consensus_methods for method in ["rms", "onset", "freq"]), \
            f"Consensus methods should contain all three methods, got {sync_record.consensus_methods}"

    def test_all_methods_same_time_span_is_zero(self, tmp_path):
        """Test that method_agreement_span is 0 when all methods agree perfectly."""
        maneuver_dir = tmp_path / "Left Knee" / "Walking"
        maneuver_dir.mkdir(parents=True)

        synced_df = pd.DataFrame({
            'tt': np.arange(0, 10.0, 0.01),
            'ch1': np.random.randn(1000),
        })

        # All methods at exactly the same time (perfect agreement)
        detection_results = {
            "consensus_time": 5.0,
            "rms_time": 5.0,
            "onset_time": 5.0,
            "freq_time": 5.0,
            "consensus_methods": ["rms", "onset", "freq"],
        }

        sync_record = create_sync_record_from_data(
            sync_file_name="test_sync",
            synced_df=synced_df,
            detection_results=detection_results,
            audio_record=None,
            biomech_record=None,
            metadata={"pass_number": 1, "speed": "normal"},
            study="AOA",
            study_id=1011,
            pass_number=1,
            speed="normal",
        )

        # Perfect agreement = 0 span
        assert sync_record.method_agreement_span == 0.0, \
            f"Method agreement span should be 0 with perfect agreement, got {sync_record.method_agreement_span}"

    def test_large_disagreement_span(self, tmp_path):
        """Test method_agreement_span with large disagreement between methods."""
        maneuver_dir = tmp_path / "Left Knee" / "Walking"
        maneuver_dir.mkdir(parents=True)

        synced_df = pd.DataFrame({
            'tt': np.arange(0, 30.0, 0.01),
            'ch1': np.random.randn(3000),
        })

        # Large disagreement (could indicate poor signal quality)
        detection_results = {
            "consensus_time": 15.0,
            "rms_time": 10.0,     # 5 seconds early
            "onset_time": 15.0,   # On time
            "freq_time": 20.0,    # 5 seconds late
            "consensus_methods": ["rms", "onset", "freq"],
        }

        sync_record = create_sync_record_from_data(
            sync_file_name="test_sync",
            synced_df=synced_df,
            detection_results=detection_results,
            audio_record=None,
            biomech_record=None,
            metadata={"pass_number": 1, "speed": "normal"},
            study="AOA",
            study_id=1011,
            pass_number=1,
            speed="normal",
        )

        expected_span = 20.0 - 10.0  # 10.0 seconds
        assert abs(sync_record.method_agreement_span - expected_span) < 0.001, \
            f"Method agreement span should be {expected_span}, got {sync_record.method_agreement_span}"

    def test_method_agreement_span_persists_in_cycles_record(self, tmp_path):
        """Test that method_agreement_span is inherited by cycles records."""
        maneuver_dir = tmp_path / "Left Knee" / "Walking"
        maneuver_dir.mkdir(parents=True)

        synced_df = pd.DataFrame({
            'tt': np.arange(0, 10.0, 0.01),
            'ch1': np.random.randn(1000),
        })

        # Create sync record with method_agreement_span
        detection_results = {
            "consensus_time": 5.0,
            "rms_time": 4.8,
            "onset_time": 5.1,
            "freq_time": 5.3,
            "consensus_methods": ["rms", "onset", "freq"],
        }

        sync_record = create_sync_record_from_data(
            sync_file_name="test_sync",
            synced_df=synced_df,
            detection_results=detection_results,
            audio_record=None,
            biomech_record=None,
            metadata={"maneuver": "walk"},
            pass_number=1,
            speed="normal",
            study="AOA",
            study_id=1011,
        )

        expected_span = 5.3 - 4.8  # 0.5
        assert abs(sync_record.method_agreement_span - expected_span) < 0.001, \
            f"Sync record method_agreement_span should be {expected_span}"

        # Create cycles record from sync record
        cycle_data = pd.DataFrame({
            'tt': np.arange(0, 1.0, 0.01),
            'ch1': np.random.randn(100),
        })

        cycles_record = create_cycles_record_from_data(
            sync_file_name="test_sync",
            clean_cycles=[cycle_data],
            outlier_cycles=[],
            pass_number=1,
            speed="normal",
            output_dir=maneuver_dir,
            plots_created=False,
            error=None,
            audio_record=None,
            biomech_record=None,
            sync_record=sync_record,
            metadata={},
            study="AOA",
            study_id=1011,
        )

        # Cycles record should inherit method_agreement_span
        assert abs(cycles_record.method_agreement_span - expected_span) < 0.001, \
            f"Cycles record should inherit method_agreement_span={expected_span}, got {cycles_record.method_agreement_span}"

    def test_method_agreement_span_in_excel_output(self, tmp_path):
        """Test that method_agreement_span appears correctly in Excel output."""
        maneuver_dir = tmp_path / "Left Knee" / "Walking"
        maneuver_dir.mkdir(parents=True)

        log = ManeuverProcessingLog(
            study_id="1011",
            knee_side="Left",
            maneuver="walk",
            maneuver_directory=maneuver_dir,
            log_created=datetime(2024, 1, 1),
        )

        synced_df = pd.DataFrame({
            'tt': np.arange(0, 10.0, 0.01),
            'ch1': np.random.randn(1000),
        })

        # Create sync with measurable method disagreement
        detection_results = {
            "consensus_time": 5.0,
            "rms_time": 4.6,      # 0.4s early
            "onset_time": 5.0,    # On time
            "freq_time": 5.3,     # 0.3s late
            "consensus_methods": ["rms", "onset", "freq"],
        }

        sync_record = create_sync_record_from_data(
            sync_file_name="test_sync",
            synced_df=synced_df,
            detection_results=detection_results,
            audio_record=None,
            biomech_record=None,
            metadata={"pass_number": 1, "speed": "normal"},
            study="AOA",
            study_id=1011,
            pass_number=1,
            speed="normal",
        )

        log.add_synchronization_record(sync_record)

        # Save and verify in Excel
        excel_path = maneuver_dir / "test_log.xlsx"
        log.save_to_excel(excel_path)

        sync_df = pd.read_excel(excel_path, sheet_name="Synchronization")

        expected_span = 5.3 - 4.6  # 0.7
        actual_span = sync_df["Method Agreement Span (s)"].iloc[0]

        assert abs(actual_span - expected_span) < 0.001, \
            f"Excel should show method_agreement_span={expected_span}, got {actual_span}"

        # Verify consensus methods are also present
        consensus_methods = sync_df["Consensus Methods"].iloc[0]
        assert "rms" in consensus_methods and "onset" in consensus_methods and "freq" in consensus_methods, \
            f"Excel should show all three consensus methods, got {consensus_methods}"
