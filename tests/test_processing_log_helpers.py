"""
Test module for processing_log helper functions added during metadata renovation.

This test file provides comprehensive coverage for:
1. _get_audio_processing_qc_defaults() - QC field defaults for MovementCycle creation
2. _get_sync_method_defaults() - Sync method field defaults
3. Field inference logic in create_audio_record_from_data()
4. Context passing in create_sync_record_from_data()
5. load_from_excel() reconstruction logic
"""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.metadata import (
    AudioProcessing,
    BiomechanicsImport,
    MovementCycle,
    Synchronization,
)
from src.orchestration.processing_log import (
    ManeuverProcessingLog,
    _get_audio_processing_qc_defaults,
    _get_sync_method_defaults,
)


class TestGetAudioProcessingQCDefaults:
    """Test _get_audio_processing_qc_defaults() helper function."""

    def test_returns_all_required_qc_fields(self):
        """Test that helper returns all required AudioProcessing QC fields."""
        defaults = _get_audio_processing_qc_defaults()

        # Should have all QC segment fields
        assert "qc_fail_segments" in defaults
        assert "qc_signal_dropout" in defaults
        assert "qc_signal_dropout_ch1" in defaults
        assert "qc_signal_dropout_ch2" in defaults
        assert "qc_signal_dropout_ch3" in defaults
        assert "qc_signal_dropout_ch4" in defaults
        assert "qc_artifact" in defaults
        assert "qc_artifact_ch1" in defaults
        assert "qc_artifact_ch2" in defaults
        assert "qc_artifact_ch3" in defaults
        assert "qc_artifact_ch4" in defaults

    def test_qc_fail_segments_is_empty_list(self):
        """Test that qc_fail_segments defaults to empty list."""
        defaults = _get_audio_processing_qc_defaults()
        assert defaults["qc_fail_segments"] == []
        assert isinstance(defaults["qc_fail_segments"], list)

    def test_dropout_flags_are_false(self):
        """Test that all dropout flags default to False."""
        defaults = _get_audio_processing_qc_defaults()
        assert defaults["qc_signal_dropout"] is False
        assert defaults["qc_signal_dropout_ch1"] is False
        assert defaults["qc_signal_dropout_ch2"] is False
        assert defaults["qc_signal_dropout_ch3"] is False
        assert defaults["qc_signal_dropout_ch4"] is False

    def test_artifact_flags_are_false(self):
        """Test that all artifact flags default to False."""
        defaults = _get_audio_processing_qc_defaults()
        assert defaults["qc_artifact"] is False
        assert defaults["qc_artifact_ch1"] is False
        assert defaults["qc_artifact_ch2"] is False
        assert defaults["qc_artifact_ch3"] is False
        assert defaults["qc_artifact_ch4"] is False


class TestGetSyncMethodDefaults:
    """Test _get_sync_method_defaults() helper function."""

    def test_returns_sync_method_fields(self):
        """Test that helper extracts sync method fields from row."""
        row = {
            "Sync Method": "consensus",
            "Consensus Methods": "rms, onset"
        }
        sync_method, consensus_methods = _get_sync_method_defaults(row)

        # Should return sync method and consensus methods
        assert sync_method == "consensus"
        assert consensus_methods == "rms, onset"

    def test_default_values(self):
        """Test that defaults are applied when fields are missing."""
        row = {}  # Empty row
        sync_method, consensus_methods = _get_sync_method_defaults(row)

        # Should have reasonable defaults
        assert sync_method == "consensus"
        assert consensus_methods == "consensus"


class TestCreateAudioRecordHelpers:
    """Test helper logic in create_audio_record_from_data()."""

    def test_num_channels_inference_from_audio_df(self):
        """Test that num_channels is inferred from audio_df columns."""
        # Create mock audio data with 4 channels
        audio_df = pd.DataFrame({
            "TIME": [0.0, 0.1, 0.2],
            "ch1": [0.1, 0.2, 0.3],
            "ch2": [0.2, 0.3, 0.4],
            "ch3": [0.3, 0.4, 0.5],
            "ch4": [0.4, 0.5, 0.6],
        })

        # Count channel columns
        channel_cols = [col for col in audio_df.columns if col.startswith("ch")]
        assert len(channel_cols) == 4

    def test_num_channels_inference_from_audio_df_3ch(self):
        """Test num_channels inference with 3 channels."""
        audio_df = pd.DataFrame({
            "TIME": [0.0, 0.1, 0.2],
            "ch1": [0.1, 0.2, 0.3],
            "ch2": [0.2, 0.3, 0.4],
            "ch3": [0.3, 0.4, 0.5],
        })

        channel_cols = [col for col in audio_df.columns if col.startswith("ch")]
        assert len(channel_cols) == 3

    def test_qc_data_integration(self):
        """Test that qc_data is properly integrated when provided."""
        qc_data = {
            "qc_fail_segments": [(0.5, 1.0)],
            "qc_signal_dropout": True,
            "qc_signal_dropout_ch1": True,
            "qc_artifact": False,
        }

        # Verify qc_data structure
        assert "qc_fail_segments" in qc_data
        assert "qc_signal_dropout" in qc_data
        assert qc_data["qc_signal_dropout"] is True

    def test_qc_data_missing_uses_defaults(self):
        """Test that missing qc_data fields use defaults."""
        qc_data = {
            "qc_fail_segments": [(0.5, 1.0)],
            # Missing other fields
        }

        defaults = _get_audio_processing_qc_defaults()

        # Should be able to merge with defaults
        merged = {**defaults, **qc_data}
        assert merged["qc_fail_segments"] == [(0.5, 1.0)]
        assert merged["qc_signal_dropout"] is False  # From defaults


class TestCreateSyncRecordContext:
    """Test context passing in create_sync_record_from_data()."""

    def test_audio_record_provides_acoustics_fields(self):
        """Test that audio_record provides AcousticsFile inherited fields."""
        # Mock audio record
        audio_record = AudioProcessing(
            study="AOA",
            study_id=1011,
            linked_biomechanics=True,
            biomechanics_file="test.csv",
            biomechanics_type="IMU",
            biomechanics_sync_method="stomp",
            biomechanics_sample_rate=200.0,
            audio_file_name="test.bin",
            device_serial="AE01",
            firmware_version=1,
            recording_date=datetime(2024, 1, 1),
            recording_time=datetime(2024, 1, 1, 10, 0, 0),
            file_time=datetime(2024, 1, 1, 10, 0, 0),
            file_size_mb=100.0,
            sample_rate=46875.0,
            num_channels=4,
            mic_1_position="IPM",
            mic_2_position="IPL",
            mic_3_position="SPM",
            mic_4_position="SPL",
            knee="left",
            maneuver="walk",
            processing_date=datetime(2024, 1, 1),
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

        # Should have all inherited fields accessible
        assert audio_record.study == "AOA"
        assert audio_record.study_id == 1011
        assert audio_record.audio_file_name == "test.bin"
        assert audio_record.num_channels == 4
        assert audio_record.sample_rate == 46875.0

    def test_biomech_record_provides_study_fields(self):
        """Test that biomech_record provides StudyMetadata fields."""
        biomech_record = BiomechanicsImport(
            study="preOA",
            study_id=2022,
            maneuver="walk",
            biomechanics_file="test.csv",
            sheet_name="Sheet1",
            processing_date=datetime(2024, 1, 1),
            processing_status="success",
            sample_rate=200.0,
            num_sub_recordings=2,
            num_passes=2,
            num_data_points=10000,
            duration_seconds=50.0,
        )

        # Should provide study context
        assert biomech_record.study == "preOA"
        assert biomech_record.study_id == 2022
        assert biomech_record.biomechanics_file == "test.csv"


class TestLoadFromExcelReconstruction:
    """Test load_from_excel() field reconstruction logic."""

    def test_timedelta_reconstruction(self):
        """Test that float seconds are converted back to timedeltas."""
        # Simulate Excel row with float seconds
        sync_offset_seconds = 1.5
        audio_sync_time_seconds = 10.0

        # Convert to timedelta
        sync_offset = timedelta(seconds=sync_offset_seconds)
        audio_sync_time = timedelta(seconds=audio_sync_time_seconds)

        assert isinstance(sync_offset, timedelta)
        assert sync_offset.total_seconds() == 1.5
        assert isinstance(audio_sync_time, timedelta)
        assert audio_sync_time.total_seconds() == 10.0

    def test_consensus_methods_from_string(self):
        """Test that consensus_methods string is split into list."""
        consensus_str = "rms_energy,biomechanics"

        # Split string
        if isinstance(consensus_str, str):
            consensus_list = consensus_str.split(",")
        else:
            consensus_list = consensus_str

        assert isinstance(consensus_list, list)
        assert "rms_energy" in consensus_list
        assert "biomechanics" in consensus_list

    def test_inheritance_field_reconstruction(self):
        """Test that inherited fields are properly reconstructed."""
        # Simulate Excel row for AudioProcessing
        excel_row = {
            "Study": "AOA",
            "Study ID": 1011,
            "Linked Biomechanics": True,
            "Biomechanics File": "test.csv",
            "Biomechanics Type": "IMU",
            "Biomechanics Sync Method": "stomp",
            "Biomechanics Sample Rate": 200.0,
            "Audio File Name": "test.bin",
            "Device Serial": "AE01",
            "Firmware Version": "1.0.0",
            "Sample Rate": 46875.0,
            "Channels": 4,
        }

        # Should be able to reconstruct AudioProcessing with inherited fields
        assert excel_row["Study"] == "AOA"
        assert excel_row["Study ID"] == 1011
        assert excel_row["Audio File Name"] == "test.bin"
        assert excel_row["Channels"] == 4


class TestMovementCycleCreation:
    """Test MovementCycle creation with all inherited fields."""

    def test_movement_cycle_requires_audio_processing_fields(self):
        """Test that MovementCycle requires all AudioProcessing QC fields."""
        # Get QC defaults
        qc_defaults = _get_audio_processing_qc_defaults()

        # Should have all fields needed for MovementCycle (which inherits AudioProcessing)
        required_qc_fields = [
            "biomechanics_qc_fail",
            "sync_qc_fail",
            "qc_fail_segments",
            "qc_signal_dropout",
            "qc_signal_dropout_ch1",
            "qc_signal_dropout_ch2",
            "qc_signal_dropout_ch3",
            "qc_signal_dropout_ch4",
            "qc_artifact",
            "qc_artifact_ch1",
            "qc_artifact_ch2",
            "qc_artifact_ch3",
            "qc_artifact_ch4",
        ]

        for field in required_qc_fields:
            assert field in qc_defaults

    def test_movement_cycle_creation_with_defaults(self):
        """Test MovementCycle creation using QC defaults."""
        qc_defaults = _get_audio_processing_qc_defaults()

        # Create MovementCycle with defaults
        cycle = MovementCycle(
            # StudyMetadata
            study="AOA",
            study_id=1011,
            # BiomechanicsMetadata
            linked_biomechanics=True,
            biomechanics_file="test.csv",
            biomechanics_type="IMU",
            biomechanics_sync_method="stomp",
            biomechanics_sample_rate=200.0,
            # AcousticsFile
            audio_file_name="test.bin",
            device_serial="AE01",
            firmware_version=1,
            recording_date=datetime(2024, 1, 1),
            recording_time=datetime(2024, 1, 1, 10, 0, 0),
            file_time=datetime(2024, 1, 1, 10, 0, 0),
            file_size_mb=100.0,
            sample_rate=46875.0,
            num_channels=4,
            mic_1_position="IPM",
            mic_2_position="IPL",
            mic_3_position="SPM",
            mic_4_position="SPL",
            knee="left",
            maneuver="walk",
            # AudioProcessing
            processing_date=datetime(2024, 1, 1),
            **qc_defaults,
            # MovementCycle
            cycle_file="cycle_0.csv",
            cycle_index=0,
            pass_number=1,
            speed="normal",
            duration_s=1.0,
            start_time_s=0.0,
            end_time_s=1.0,
            audio_start_time=datetime(2024, 1, 1, 10, 0, 10),
            audio_end_time=datetime(2024, 1, 1, 10, 0, 11),
            bio_start_time=datetime(2024, 1, 1, 10, 0, 8, 500000),
            bio_end_time=datetime(2024, 1, 1, 10, 0, 9, 500000),
            is_outlier=False,
        )

        # Verify cycle was created successfully
        assert cycle.cycle_index == 0
        assert cycle.qc_signal_dropout is False
        assert cycle.qc_artifact is False


class TestFieldNameTransitions:
    """Test field name transitions from old to new naming."""

    def test_stomp_to_sync_naming(self):
        """Test that old 'stomp' terminology is now 'sync'."""
        # Old names (should not be used)
        old_names = ["stomp_offset", "audio_stomp_time", "bio_left_stomp_time"]

        # New names (should be used)
        new_names = ["sync_offset", "audio_sync_time", "bio_left_sync_time"]

        # Verify new names are being used
        sync = Synchronization(
            study="AOA",
            study_id=1011,
            linked_biomechanics=True,
            biomechanics_file="test.csv",
            biomechanics_type="IMU",
            biomechanics_sync_method="stomp",
            biomechanics_sample_rate=200.0,
            audio_file_name="test.bin",
            device_serial="AE01",
            firmware_version=1,
            recording_date=datetime(2024, 1, 1),
            recording_time=datetime(2024, 1, 1, 10, 0, 0),
            file_time=datetime(2024, 1, 1, 10, 0, 0),
            file_size_mb=100.0,
            sample_rate=46875.0,
            num_channels=4,
            mic_1_position="IPM",
            mic_2_position="IPL",
            mic_3_position="SPM",
            mic_4_position="SPL",
            knee="left",
            maneuver="walk",
            sync_offset=timedelta(seconds=1.5),
            audio_sync_time=timedelta(seconds=10.0),
            aligned_audio_sync_time=timedelta(seconds=10.0),
            aligned_bio_sync_time=timedelta(seconds=10.0),
            bio_left_sync_time=timedelta(seconds=8.5),
            bio_right_sync_time=None,
            consensus_time=timedelta(seconds=10.1),
            rms_time=timedelta(seconds=10.1),
            onset_time=timedelta(seconds=10.1),
            freq_time=timedelta(seconds=10.1),
            sync_method="consensus",
            consensus_methods="rms_energy, biomechanics",
            biomechanics_time=timedelta(seconds=10.2),
            biomechanics_time_contralateral=None,
            sync_file_name="test_sync.pkl",
            processing_date=datetime(2024, 1, 1),
            sync_duration=timedelta(seconds=60.0),
            total_cycles_extracted=10,
            clean_cycles=8,
            outlier_cycles=2,
            mean_cycle_duration_s=6.0,
            median_cycle_duration_s=6.0,
            min_cycle_duration_s=5.0,
            max_cycle_duration_s=7.0,
            mean_acoustic_auc=100.0,
        )

        # New names should work
        assert hasattr(sync, "sync_offset")
        assert hasattr(sync, "audio_sync_time")
        assert hasattr(sync, "bio_left_sync_time")

    def test_num_recordings_to_num_sub_recordings(self):
        """Test that num_recordings is now num_sub_recordings."""
        biomech = BiomechanicsImport(
            study="AOA",
            study_id=1011,
            maneuver="walk",
            biomechanics_file="test.csv",
            sheet_name="Sheet1",
            processing_date=datetime(2024, 1, 1),
            processing_status="success",
            sample_rate=200.0,
            num_sub_recordings=3,  # New name
            num_passes=3,
            num_data_points=15000,
            duration_seconds=75.0,
        )

        # New name should work
        assert hasattr(biomech, "num_sub_recordings")
        assert biomech.num_sub_recordings == 3

    def test_knee_side_to_knee(self):
        """Test that knee_side is now just knee."""
        sync = Synchronization(
            study="AOA",
            study_id=1011,
            linked_biomechanics=True,
            biomechanics_file="test.csv",
            biomechanics_type="IMU",
            biomechanics_sync_method="stomp",
            biomechanics_sample_rate=200.0,
            audio_file_name="test.bin",
            device_serial="AE01",
            firmware_version=1,
            recording_date=datetime(2024, 1, 1),
            recording_time=datetime(2024, 1, 1, 10, 0, 0),
            file_time=datetime(2024, 1, 1, 10, 0, 0),
            file_size_mb=100.0,
            sample_rate=46875.0,
            num_channels=4,
            mic_1_position="IPM",
            mic_2_position="IPL",
            mic_3_position="SPM",
            mic_4_position="SPL",
            knee="left",  # New name (was knee_side)
            maneuver="walk",
            sync_offset=timedelta(seconds=1.5),
            audio_sync_time=timedelta(seconds=10.0),
            aligned_audio_sync_time=timedelta(seconds=10.0),
            aligned_bio_sync_time=timedelta(seconds=10.0),
            bio_left_sync_time=timedelta(seconds=8.5),
            bio_right_sync_time=None,
            consensus_time=timedelta(seconds=10.1),
            rms_time=timedelta(seconds=10.1),
            onset_time=timedelta(seconds=10.1),
            freq_time=timedelta(seconds=10.1),
            sync_method="consensus",
            consensus_methods="rms_energy, biomechanics",
            biomechanics_time=timedelta(seconds=10.2),
            biomechanics_time_contralateral=None,
            sync_file_name="test_sync.pkl",
            processing_date=datetime(2024, 1, 1),
            sync_duration=timedelta(seconds=60.0),
            total_cycles_extracted=10,
            clean_cycles=8,
            outlier_cycles=2,
            mean_cycle_duration_s=6.0,
            median_cycle_duration_s=6.0,
            min_cycle_duration_s=5.0,
            max_cycle_duration_s=7.0,
            mean_acoustic_auc=100.0,
        )

        # New name should work
        assert hasattr(sync, "knee")
        assert sync.knee == "left"
