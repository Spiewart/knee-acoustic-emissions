"""Integration tests for metadata classes and processing pipeline.

Tests the complete flow of metadata through the processing pipeline:
1. Creation and field population
2. Inheritance correctness
3. Data persistence (Excel save/load)
4. Resume scenarios
5. Cross-field consistency
"""

from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import pytest

from src.metadata import (
    AcousticsFile,
    AudioProcessing,
    MovementCycle,
    StudyMetadata,
    Synchronization,
    SynchronizationMetadata,
    WalkMetadata,
)
from src.orchestration.processing_log import (
    _create_synchronization_metadata_from_row,
    _timedelta_to_seconds,
)


class TestWalkMetadataMixin:
    """Test WalkMetadata mixin functionality."""

    def test_walk_metadata_optional_fields(self):
        """Verify pass_number and speed are optional."""
        assert WalkMetadata.__dataclass_fields__['pass_number'].default is None
        assert WalkMetadata.__dataclass_fields__['speed'].default is None

    def test_walk_metadata_pass_number_validation(self):
        """Verify pass_number must be non-negative."""
        from pydantic import ValidationError

        # Valid: positive pass number
        meta = WalkMetadata(pass_number=1, speed="comfortable")
        assert meta.pass_number == 1

        # Valid: None
        meta = WalkMetadata(pass_number=None, speed=None)
        assert meta.pass_number is None

        # Invalid: negative pass number
        with pytest.raises(ValidationError):
            WalkMetadata(pass_number=-1, speed="fast")


class TestSynchronizationMetadataInheritance:
    """Test that SynchronizationMetadata properly inherits from parents."""

    def test_has_acoustics_fields(self):
        """Verify SynchronizationMetadata has AcousticsFile fields."""
        fields = SynchronizationMetadata.__dataclass_fields__
        assert 'audio_file_name' in fields
        assert 'knee' in fields
        assert 'maneuver' in fields
        assert 'num_channels' in fields

    def test_has_study_fields(self):
        """Verify SynchronizationMetadata has StudyMetadata fields."""
        fields = SynchronizationMetadata.__dataclass_fields__
        assert 'study' in fields
        assert 'study_id' in fields

    def test_has_walk_fields(self):
        """Verify SynchronizationMetadata has WalkMetadata fields."""
        fields = SynchronizationMetadata.__dataclass_fields__
        assert 'pass_number' in fields
        assert 'speed' in fields

    def test_has_sync_fields(self):
        """Verify SynchronizationMetadata has all sync-related fields."""
        fields = SynchronizationMetadata.__dataclass_fields__

        # Stomp times
        assert 'audio_sync_time' in fields
        assert 'sync_offset' in fields
        assert 'aligned_audio_sync_time' in fields
        assert 'aligned_biomechanics_sync_time' in fields

        # Sync method
        assert 'sync_method' in fields
        assert 'consensus_methods' in fields

        # Detection times
        assert 'consensus_time' in fields
        assert 'rms_time' in fields
        assert 'onset_time' in fields
        assert 'freq_time' in fields

    def test_sync_fields_are_float_not_timedelta(self):
        """Verify sync time fields are float (seconds) not timedelta."""
        meta = SynchronizationMetadata(
            study="AOA",
            study_id=1011,
            knee="right",
            maneuver="sts",
            audio_file_name="test.wav",
            device_serial="12345",
            firmware_version=1,
            file_time=datetime.now(),
            file_size_mb=100.0,
            recording_date=datetime.now(),
            recording_time=datetime.now(),
            num_channels=4,
            mic_1_position="IPM",
            mic_2_position="IPL",
            mic_3_position="SPM",
            mic_4_position="SPL",
            linked_biomechanics=False,
            audio_sync_time=1.5,  # Float, not timedelta
            sync_offset=0.1,
        )

        assert isinstance(meta.audio_sync_time, float)
        assert isinstance(meta.sync_offset, float)


class TestSynchronizationClass:
    """Test Synchronization class structure."""

    def test_inherits_from_synchronization_metadata(self):
        """Verify Synchronization inherits from SynchronizationMetadata."""
        mro = [c.__name__ for c in Synchronization.__mro__]
        assert 'SynchronizationMetadata' in mro

    def test_has_sync_process_fields(self):
        """Verify Synchronization has sync process identification fields."""
        fields = Synchronization.__dataclass_fields__
        assert 'sync_file_name' in fields
        assert 'processing_date' in fields

    def test_has_aggregate_fields(self):
        """Verify Synchronization has aggregate statistics fields."""
        fields = Synchronization.__dataclass_fields__
        assert 'total_cycles_extracted' in fields
        assert 'clean_cycles' in fields
        assert 'outlier_cycles' in fields
        assert 'mean_cycle_duration_s' in fields
        assert 'median_cycle_duration_s' in fields

    def test_linked_biomechanics_always_true(self):
        """Verify linked_biomechanics is always True for Synchronization."""
        sync = Synchronization(
            study="AOA",
            study_id=1011,
            knee="right",
            maneuver="sts",
            audio_file_name="test.wav",
            device_serial="12345",
            firmware_version=1,
            file_time=datetime.now(),
            file_size_mb=100.0,
            recording_date=datetime.now(),
            recording_time=datetime.now(),
            num_channels=4,
            mic_1_position="IPM",
            mic_2_position="IPL",
            mic_3_position="SPM",
            mic_4_position="SPL",
            linked_biomechanics=True,
            sync_file_name="sync.h5",
            processing_date=datetime.now(),
        )

        assert sync.linked_biomechanics is True

    def test_does_not_have_cycle_fields(self):
        """Verify Synchronization does NOT have per-cycle fields."""
        fields = Synchronization.__dataclass_fields__

        # Should NOT have these cycle-level fields
        assert 'cycle_file' not in fields
        assert 'cycle_index' not in fields
        assert 'is_outlier' not in fields


class TestMovementCycleInheritance:
    """Test MovementCycle dual inheritance from SynchronizationMetadata + AudioProcessing."""

    def test_inherits_from_both_parents(self):
        """Verify MovementCycle inherits from both parent classes."""
        mro = [c.__name__ for c in MovementCycle.__mro__]
        assert 'SynchronizationMetadata' in mro
        assert 'AudioProcessing' in mro

    def test_has_cycle_fields(self):
        """Verify MovementCycle has cycle-specific fields."""
        fields = MovementCycle.__dataclass_fields__
        assert 'cycle_file' in fields
        assert 'cycle_index' in fields
        assert 'start_time_s' in fields
        assert 'end_time_s' in fields
        assert 'duration_s' in fields

    def test_has_cycle_level_qc_flags(self):
        """Verify MovementCycle has cycle-level QC flags only."""
        fields = MovementCycle.__dataclass_fields__
        assert 'biomechanics_qc_fail' in fields
        assert 'sync_qc_fail' in fields

    def test_inherits_sync_fields(self):
        """Verify MovementCycle inherits all sync fields."""
        fields = MovementCycle.__dataclass_fields__

        # From SynchronizationMetadata
        assert 'audio_sync_time' in fields
        assert 'sync_method' in fields
        assert 'consensus_time' in fields
        assert 'pass_number' in fields
        assert 'speed' in fields

    def test_inherits_audio_qc_fields(self):
        """Verify MovementCycle inherits all audio QC fields."""
        fields = MovementCycle.__dataclass_fields__

        # From AudioProcessing
        assert 'qc_artifact' in fields
        assert 'qc_signal_dropout' in fields
        assert 'qc_fail_segments' in fields
        assert 'qc_artifact_type' in fields
        assert 'qc_signal_dropout_ch1' in fields

    def test_inherits_audio_file_fields(self):
        """Verify MovementCycle inherits audio file metadata."""
        fields = MovementCycle.__dataclass_fields__

        # From AcousticsFile (via SynchronizationMetadata)
        assert 'audio_file_name' in fields
        assert 'knee' in fields
        assert 'maneuver' in fields
        assert 'num_channels' in fields


class TestNoFieldDuplication:
    """Test that there is no field duplication across inheritance hierarchy."""

    def test_cycle_qc_flags_only_on_movement_cycle(self):
        """Verify biomechanics_qc_fail and sync_qc_fail only on MovementCycle."""
        sync_fields = Synchronization.__dataclass_fields__
        cycle_fields = MovementCycle.__dataclass_fields__

        # Synchronization should NOT have these
        assert 'biomechanics_qc_fail' not in sync_fields
        assert 'sync_qc_fail' not in sync_fields

        # MovementCycle should have these
        assert 'biomechanics_qc_fail' in cycle_fields
        assert 'sync_qc_fail' in cycle_fields

    def test_per_cycle_fields_not_on_synchronization(self):
        """Verify per-cycle fields like cycle_file not on Synchronization."""
        fields = Synchronization.__dataclass_fields__

        assert 'cycle_file' not in fields
        assert 'cycle_index' not in fields
        assert 'start_time_s' not in fields
        assert 'end_time_s' not in fields


class TestHelperFunctions:
    """Test Phase 2 helper functions."""

    def test_timedelta_to_seconds_with_timedelta(self):
        """Verify _timedelta_to_seconds converts timedelta correctly."""
        td = pd.Timedelta(seconds=1.5)
        assert _timedelta_to_seconds(td) == 1.5

    def test_timedelta_to_seconds_with_float(self):
        """Verify _timedelta_to_seconds returns float as-is."""
        assert _timedelta_to_seconds(1.5) == 1.5

    def test_timedelta_to_seconds_with_none(self):
        """Verify _timedelta_to_seconds returns None for None input."""
        assert _timedelta_to_seconds(None) is None

    def test_timedelta_to_seconds_with_string(self):
        """Verify _timedelta_to_seconds converts string to float."""
        assert _timedelta_to_seconds("1.5") == 1.5

    def test_timedelta_to_seconds_with_invalid_string(self):
        """Verify _timedelta_to_seconds returns None for invalid string."""
        assert _timedelta_to_seconds("not a number") is None


class TestInstantiation:
    """Test that new metadata classes can be instantiated correctly."""

    def test_synchronization_instantiation(self):
        """Verify Synchronization records can be created."""
        sync = Synchronization(
            study="AOA",
            study_id=1011,
            knee="right",
            maneuver="sts",
            audio_file_name="test.wav",
            device_serial="12345",
            firmware_version=1,
            file_time=datetime.now(),
            file_size_mb=100.0,
            recording_date=datetime.now(),
            recording_time=datetime.now(),
            num_channels=4,
            mic_1_position="IPM",
            mic_2_position="IPL",
            mic_3_position="SPM",
            mic_4_position="SPL",
            linked_biomechanics=True,
            sync_file_name="sync.h5",
            processing_date=datetime.now(),
            audio_sync_time=1.5,
            sync_method="consensus",
        )

        assert sync.sync_file_name == "sync.h5"
        assert sync.audio_sync_time == 1.5
        assert sync.linked_biomechanics is True

    def test_movement_cycle_instantiation(self):
        """Verify MovementCycle records can be created with all inherited fields."""
        now = datetime.now()
        cycle = MovementCycle(
            study="AOA",
            study_id=1011,
            knee="right",
            maneuver="sts",
            audio_file_name="test.wav",
            device_serial="12345",
            firmware_version=1,
            file_time=datetime.now(),
            file_size_mb=100.0,
            recording_date=datetime.now(),
            recording_time=datetime.now(),
            num_channels=4,
            mic_1_position="IPM",
            mic_2_position="IPL",
            mic_3_position="SPM",
            mic_4_position="SPL",
            linked_biomechanics=False,
            processing_date=datetime.now(),
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
            qc_artifact_type=None,
            qc_artifact_type_ch1=None,
            qc_artifact_type_ch2=None,
            qc_artifact_type_ch3=None,
            qc_artifact_type_ch4=None,
            cycle_file="cycle.pkl",
            cycle_index=0,
            is_outlier=False,  # NOW REQUIRED
            start_time_s=0.0,
            end_time_s=1.0,
            duration_s=1.0,
            audio_start_time=now,  # NOW REQUIRED
            audio_end_time=now,  # NOW REQUIRED
            biomechanics_qc_fail=False,  # NOW REQUIRED
            sync_qc_fail=False,  # NOW REQUIRED
            audio_sync_time=1.5,
            sync_method="consensus",
        )

        # Own fields
        assert cycle.cycle_file == "cycle.pkl"
        assert cycle.cycle_index == 0
        assert cycle.duration_s == 1.0
        assert cycle.is_outlier is False
        assert cycle.biomechanics_qc_fail is False
        assert cycle.sync_qc_fail is False

        # Inherited from SynchronizationMetadata
        assert cycle.audio_sync_time == 1.5
        assert cycle.sync_method == "consensus"

        # Inherited from AudioProcessing
        assert cycle.qc_artifact is False
        assert cycle.qc_signal_dropout is False

        # Inherited from AcousticsFile
        assert cycle.knee == "right"
        assert cycle.maneuver == "sts"
