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
        # audio_sync_time is in seconds (float)
        assert loaded_log.synchronization_records[0].audio_sync_time == 10.5

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
        assert loaded_log.synchronization_records[0].audio_sync_time == 11.0
        # Verify the sync duration changed (more samples)
        assert loaded_log.synchronization_records[0].sync_duration > 0

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

    def test_cycles_record_has_biomechanics_sync_method(self, tmp_path):
        """Test that cycle records get proper biomechanics_sync_method when linked."""
        from src.orchestration.processing_log import create_cycles_record_from_data

        maneuver_dir = tmp_path / "Left Knee" / "Walking"
        maneuver_dir.mkdir(parents=True)

        # Create test cycles data
        cycle_data = pd.DataFrame({
            'tt': pd.date_range('2024-01-01', periods=100, freq='10ms'),
            'ch1': np.random.randn(100),
        })

        # Create cycles record with linked biomechanics
        cycles_record = create_cycles_record_from_data(
            sync_file_name="test_sync",
            clean_cycles=[cycle_data],
            outlier_cycles=[],
            output_dir=maneuver_dir,
            plots_created=False,
            error=None,
            audio_record=None,  # No audio record, should use defaults
            biomech_record=None,
            sync_record=None,
            metadata={},
            study="AOA",
            study_id=1,
                        pass_number=1,  # Required for walk maneuvers
                        speed="normal",  # Required for walk maneuvers
            biomechanics_type="Motion Analysis",
        )

        # Verify biomechanics_sync_method is set
        assert cycles_record.linked_biomechanics is True
        assert cycles_record.biomechanics_type == "Motion Analysis"
        assert cycles_record.biomechanics_sync_method == "stomp", \
            f"Expected 'stomp' for Motion Analysis, got {cycles_record.biomechanics_sync_method}"

    def test_cycles_record_with_gonio_biomechanics(self, tmp_path):
        """Test that Gonio biomechanics uses flick sync method."""
        from src.orchestration.processing_log import create_cycles_record_from_data

        maneuver_dir = tmp_path / "Left Knee" / "Walking"
        maneuver_dir.mkdir(parents=True)

        # Create test cycles data
        cycle_data = pd.DataFrame({
            'tt': pd.date_range('2024-01-01', periods=100, freq='10ms'),
            'ch1': np.random.randn(100),
        })

        # Create cycles record with Gonio biomechanics
        cycles_record = create_cycles_record_from_data(
            sync_file_name="test_sync",
            clean_cycles=[cycle_data],
            outlier_cycles=[],
            output_dir=maneuver_dir,
            plots_created=False,
            error=None,
            audio_record=None,
            biomech_record=None,
            sync_record=None,
            metadata={},
            study="AOA",
            study_id=1,
            biomechanics_type="Gonio",
                    pass_number=1,  # Required for walk maneuvers
                    speed="normal",  # Required for walk maneuvers
        )

        # Verify Gonio uses flick
        assert cycles_record.biomechanics_type == "Gonio"
        assert cycles_record.biomechanics_sync_method == "flick", \
            f"Expected 'flick' for Gonio, got {cycles_record.biomechanics_sync_method}"

    def test_synchronization_sheet_contains_cycle_counts(self, tmp_path):
        """Test that Synchronization sheet gets updated with cycle counts when cycles are added."""
        from datetime import datetime

        from src.metadata import Synchronization
        from src.orchestration.processing_log import (
            ManeuverProcessingLog,
            create_cycles_record_from_data,
        )

        maneuver_dir = tmp_path / "Left Knee" / "Walking"
        maneuver_dir.mkdir(parents=True)

        # Create initial log with a synchronization record (from sync stage)
        log = ManeuverProcessingLog(
            study_id="1011",
            knee_side="Left",
            maneuver="walk",
            maneuver_directory=maneuver_dir,
            log_created=datetime(2024, 1, 1),
        )

        # Add a sync record without cycle counts
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
            recording_time=datetime(2024, 1, 1),
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
            consensus_time=5.0,
            consensus_methods="RMS,Frequency",
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
        log.add_synchronization_record(sync_record)

        # Verify sync record has no cycles initially
        assert log.synchronization_records[0].total_cycles_extracted == 0
        assert log.synchronization_records[0].clean_cycles == 0

        # Create cycles data with relative seconds (not timestamps)
        cycle_data = pd.DataFrame({
            'tt': np.arange(0, 1.0, 0.01),  # 0 to 1 second in 0.01s steps
            'ch1': np.random.randn(100),
        })

        # Modify sync_record to have realistic times that won't overflow
        sync_record.recording_time = datetime(2024, 1, 1, 10, 0, 0)
        sync_record.recording_date = datetime(2024, 1, 1)

        # Create cycles record with detected cycles
        cycles_record = create_cycles_record_from_data(
            sync_file_name="test_sync",
            clean_cycles=[cycle_data, cycle_data],  # 2 clean cycles
            outlier_cycles=[cycle_data],  # 1 outlier cycle
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

        # Verify cycles record has correct counts
        assert cycles_record.total_cycles_extracted == 3
        assert cycles_record.clean_cycles == 2
        assert cycles_record.outlier_cycles == 1

        # Add cycles record to log
        log.add_movement_cycles_record(cycles_record)

        # CRITICAL: Verify the synchronization record was updated with cycle counts
        assert log.synchronization_records[0].total_cycles_extracted == 3, \
            f"Sync record should have 3 total cycles, got {log.synchronization_records[0].total_cycles_extracted}"
        assert log.synchronization_records[0].clean_cycles == 2, \
            f"Sync record should have 2 clean cycles, got {log.synchronization_records[0].clean_cycles}"
        assert log.synchronization_records[0].outlier_cycles == 1, \
            f"Sync record should have 1 outlier cycle, got {log.synchronization_records[0].outlier_cycles}"

        # Save and reload to verify persistence
        excel_path = maneuver_dir / "test_log.xlsx"
        log.save_to_excel(excel_path)
        loaded_log = ManeuverProcessingLog.load_from_excel(excel_path)

        # Verify the loaded log has cycle counts in synchronization records
        assert len(loaded_log.synchronization_records) == 1
        assert loaded_log.synchronization_records[0].total_cycles_extracted == 3
        assert loaded_log.synchronization_records[0].clean_cycles == 2
        assert loaded_log.synchronization_records[0].outlier_cycles == 1
    def test_movement_cycles_sheet_contains_individual_cycles(self, tmp_path):
        """Test that Movement Cycles sheet contains row-by-row list of individual cycles, not aggregates."""
        from datetime import datetime

        import pandas as pd

        from src.metadata import Synchronization
        from src.orchestration.processing_log import (
            ManeuverProcessingLog,
            create_cycles_record_from_data,
        )

        maneuver_dir = tmp_path / "Left Knee" / "Walking"
        maneuver_dir.mkdir(parents=True)

        # Create log with a sync record
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
            consensus_time=5.0,
            consensus_methods="RMS,Frequency",
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
        log.add_synchronization_record(sync_record)

        # Create cycle data
        cycle_data_1 = pd.DataFrame({
            'tt': np.arange(0, 1.0, 0.01),
            'ch1': np.random.randn(100),
        })
        cycle_data_2 = pd.DataFrame({
            'tt': np.arange(0, 0.9, 0.01),
            'ch1': np.random.randn(90),
        })
        cycle_data_3 = pd.DataFrame({
            'tt': np.arange(0, 1.1, 0.01),
            'ch1': np.random.randn(110),
        })

        # Create cycles record with 3 individual cycles (2 clean, 1 outlier)
        cycles_record = create_cycles_record_from_data(
            sync_file_name="test_sync",
            clean_cycles=[cycle_data_1, cycle_data_2],
            outlier_cycles=[cycle_data_3],
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

        # Add cycles record to log
        log.add_movement_cycles_record(cycles_record)

        # Save to Excel
        excel_path = maneuver_dir / "test_log.xlsx"
        log.save_to_excel(excel_path)

        # Read the Movement Cycles sheet directly
        movement_cycles_df = pd.read_excel(excel_path, sheet_name="Movement Cycles")

        # CRITICAL ASSERTIONS:
        # 1. Movement Cycles sheet should have 3 rows (one per cycle), not 1 row (aggregate)
        assert len(movement_cycles_df) == 3, \
            f"Movement Cycles sheet should have 3 rows (individual cycles), got {len(movement_cycles_df)}"

        # 2. Each row should have cycle-specific fields (not aggregate fields like "Total Cycles Extracted")
        assert "Cycle Index" in movement_cycles_df.columns, \
            "Movement Cycles sheet should have 'Cycle Index' column for individual cycles"
        assert "Is Outlier" in movement_cycles_df.columns, \
            "Movement Cycles sheet should have 'Is Outlier' column for individual cycles"
        assert "Duration (s)" in movement_cycles_df.columns, \
            "Movement Cycles sheet should have 'Duration (s)' column for individual cycles"

        # 3. Should NOT have aggregate fields that belong in Synchronization sheet
        assert "Total Cycles Extracted" not in movement_cycles_df.columns, \
            "Movement Cycles sheet should NOT have aggregate field 'Total Cycles Extracted'"
        assert "Clean Cycles" not in movement_cycles_df.columns, \
            "Movement Cycles sheet should NOT have aggregate field 'Clean Cycles'"

        # 4. Verify outlier marking
        outlier_count = movement_cycles_df["Is Outlier"].sum()
        assert outlier_count == 1, f"Should have 1 outlier cycle, got {outlier_count}"
        clean_count = (~movement_cycles_df["Is Outlier"]).sum()
        assert clean_count == 2, f"Should have 2 clean cycles, got {clean_count}"

        # 5. Verify cycle indices are sequential
        cycle_indices = sorted(movement_cycles_df["Cycle Index"].values)
        assert cycle_indices == [0, 1, 2], f"Cycle indices should be [0, 1, 2], got {cycle_indices}"

        # 6. Read Synchronization sheet and verify it has aggregate data
        sync_df = pd.read_excel(excel_path, sheet_name="Synchronization")
        assert len(sync_df) == 1, f"Synchronization sheet should have 1 row (aggregate), got {len(sync_df)}"
        assert sync_df["Total Cycles Extracted"].iloc[0] == 3, \
            "Synchronization sheet should show total_cycles_extracted=3"
        assert sync_df["Clean Cycles"].iloc[0] == 2, \
            "Synchronization sheet should show clean_cycles=2"
        assert sync_df["Outlier Cycles"].iloc[0] == 1, \
            "Synchronization sheet should show outlier_cycles=1"

    def test_method_agreement_span_persists_correctly(self, tmp_path):
        """Test that method_agreement_span is calculated from consensus methods and persists through save/load."""
        from datetime import datetime

        import pandas as pd

        from src.orchestration.processing_log import (
            ManeuverProcessingLog,
            create_sync_record_from_data,
        )

        maneuver_dir = tmp_path / "Left Knee" / "Walking"
        maneuver_dir.mkdir(parents=True)

        # Create log
        log = ManeuverProcessingLog(
            study_id="1011",
            knee_side="Left",
            maneuver="walk",
            maneuver_directory=maneuver_dir,
            log_created=datetime(2024, 1, 1),
        )

        # Create synced dataframe
        synced_df = pd.DataFrame({
            'tt': np.arange(0, 10.0, 0.01),
            'ch1': np.random.randn(1000),
        })

        # Create detection results with consensus methods
        # Simulate different methods detecting at different times
        detection_results = {
            "consensus_time": 5.0,
            "rms_time": 4.8,      # 0.2s before consensus
            "onset_time": 5.1,    # 0.1s after consensus
            "freq_time": 5.2,     # 0.2s after consensus
            "consensus_methods": ["rms", "onset", "freq"],  # All 3 methods contributed
        }

        # Create sync record using the factory function (which calculates method_agreement_span)
        sync_record = create_sync_record_from_data(
            sync_file_name="test_sync.pkl",
            synced_df=synced_df,
            pass_number=1,
            speed="normal",
            detection_results=detection_results,
            error=None,
            audio_record=None,
            biomech_record=None,
            metadata={},
            study="AOA",
            study_id=1011,
        )

        # CRITICAL: Verify method_agreement_span is calculated correctly
        # Span should be max(4.8, 5.1, 5.2) - min(4.8, 5.1, 5.2) = 5.2 - 4.8 = 0.4
        expected_span = 0.4
        assert abs(sync_record.method_agreement_span - expected_span) < 0.001, \
            f"Method agreement span should be ~{expected_span}, got {sync_record.method_agreement_span}"

        # Add to log and save
        log.add_synchronization_record(sync_record)
        excel_path = maneuver_dir / "test_log.xlsx"
        log.save_to_excel(excel_path)

        # Reload and verify persistence
        loaded_log = ManeuverProcessingLog.load_from_excel(excel_path)
        assert len(loaded_log.synchronization_records) == 1
        loaded_span = loaded_log.synchronization_records[0].method_agreement_span
        assert abs(loaded_span - expected_span) < 0.001, \
            f"Loaded method agreement span should be ~{expected_span}, got {loaded_span}"

        # Verify it's in the Excel file
        sync_df = pd.read_excel(excel_path, sheet_name="Synchronization")
        excel_span = sync_df["Method Agreement Span (s)"].iloc[0]
        assert abs(excel_span - expected_span) < 0.001, \
            f"Excel method agreement span should be ~{expected_span}, got {excel_span}"

    def test_method_agreement_span_defaults_to_zero_when_not_calculated(self, tmp_path):
        """Test that method_agreement_span defaults to 0.0 when no consensus methods are provided."""
        from datetime import datetime

        import pandas as pd

        from src.orchestration.processing_log import (
            ManeuverProcessingLog,
            create_sync_record_from_data,
        )

        maneuver_dir = tmp_path / "Left Knee" / "Walking"
        maneuver_dir.mkdir(parents=True)

        # Create log
        log = ManeuverProcessingLog(
            study_id="1011",
            knee_side="Left",
            maneuver="walk",
            maneuver_directory=maneuver_dir,
            log_created=datetime(2024, 1, 1),
        )

        # Create synced dataframe
        synced_df = pd.DataFrame({
            'tt': np.arange(0, 10.0, 0.01),
            'ch1': np.random.randn(1000),
        })

        # Create detection results WITHOUT consensus methods (e.g., manual selection)
        detection_results = {
            "consensus_time": 5.0,
            "rms_time": 4.8,
            "onset_time": 5.1,
            "freq_time": 5.2,
            "consensus_methods": [],  # No methods contributed to consensus
        }

        # Create sync record
        sync_record = create_sync_record_from_data(
            sync_file_name="test_sync.pkl",
            synced_df=synced_df,
            pass_number=1,
            speed="normal",
            detection_results=detection_results,
            error=None,
            audio_record=None,
            biomech_record=None,
            metadata={},
            study="AOA",
            study_id=1011,
        )

        # CRITICAL: Verify method_agreement_span defaults to 0.0 (not None)
        assert sync_record.method_agreement_span == 0.0, \
            f"Method agreement span should default to 0.0, got {sync_record.method_agreement_span}"

        # Add to log and verify persistence
        log.add_synchronization_record(sync_record)
        excel_path = maneuver_dir / "test_log.xlsx"
        log.save_to_excel(excel_path)

        # Verify in Excel
        sync_df = pd.read_excel(excel_path, sheet_name="Synchronization")
        excel_span = sync_df["Method Agreement Span (s)"].iloc[0]
        assert excel_span == 0.0, \
            f"Excel method agreement span should be 0.0, got {excel_span}"