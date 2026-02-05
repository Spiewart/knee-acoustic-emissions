"""Integration tests for normalized metadata classes.

Validates the current (normalized) structure in src/metadata.py:
- No inheritance-based duplication
- FK-based references between entities
- Required fields and instantiation
"""

from datetime import datetime

import pandas as pd
import pytest
from pydantic import ValidationError

from src.metadata import BiomechanicsImport, MovementCycle, Synchronization
from src.orchestration.processing_log import _timedelta_to_seconds


class TestSynchronizationFields:
    """Test Synchronization model structure."""

    def test_has_fk_fields(self):
        fields = Synchronization.__dataclass_fields__
        assert "audio_processing_id" in fields
        assert "biomechanics_import_id" in fields

    def test_has_sync_fields(self):
        fields = Synchronization.__dataclass_fields__
        assert "sync_file_name" in fields
        assert "processing_date" in fields
        assert "sync_method" in fields
        assert "aligned_sync_time" in fields
        assert "bio_sync_offset" in fields

    def test_does_not_have_audio_file_fields(self):
        fields = Synchronization.__dataclass_fields__
        assert "audio_file_name" not in fields
        assert "device_serial" not in fields
        assert "num_channels" not in fields


class TestMovementCycleFields:
    """Test MovementCycle model structure."""

    def test_has_fk_fields(self):
        fields = MovementCycle.__dataclass_fields__
        assert "audio_processing_id" in fields
        assert "biomechanics_import_id" in fields
        assert "synchronization_id" in fields

    def test_has_cycle_fields(self):
        fields = MovementCycle.__dataclass_fields__
        assert "cycle_file" in fields
        assert "cycle_index" in fields
        assert "start_time_s" in fields
        assert "end_time_s" in fields
        assert "duration_s" in fields

    def test_does_not_have_audio_qc_fields(self):
        fields = MovementCycle.__dataclass_fields__
        assert "qc_fail_segments" not in fields
        assert "qc_signal_dropout" not in fields
        assert "qc_artifact" not in fields


class TestHelperFunctions:
    """Test helper functions used in processing log."""

    def test_timedelta_to_seconds_with_timedelta(self):
        td = pd.Timedelta(seconds=1.5)
        assert _timedelta_to_seconds(td) == 1.5

    def test_timedelta_to_seconds_with_float(self):
        assert _timedelta_to_seconds(1.5) == 1.5

    def test_timedelta_to_seconds_with_none(self):
        assert _timedelta_to_seconds(None) is None

    def test_timedelta_to_seconds_with_string(self):
        assert _timedelta_to_seconds("1.5") == 1.5

    def test_timedelta_to_seconds_with_invalid_string(self):
        assert _timedelta_to_seconds("not a number") is None


class TestInstantiation:
    """Test that normalized metadata classes can be instantiated correctly."""

    def test_synchronization_instantiation(self):
        sync = Synchronization(
            study="AOA",
            study_id=1011,
            audio_processing_id=1,
            biomechanics_import_id=2,
            sync_file_name="sync.h5",
            processing_date=datetime(2024, 1, 1, 12, 0, 0),
            processing_status="success",
            aligned_sync_time=1.5,
            sync_method="consensus",
        )

        assert sync.sync_file_name == "sync.h5"
        assert sync.aligned_sync_time == 1.5
        assert sync.audio_processing_id == 1
        assert sync.biomechanics_import_id == 2

    def test_movement_cycle_instantiation(self):
        now = datetime(2024, 1, 1, 10, 0, 0)
        cycle = MovementCycle(
            study="AOA",
            study_id=1011,
            audio_processing_id=1,
            biomechanics_import_id=None,
            synchronization_id=None,
            cycle_file="cycle.pkl",
            cycle_index=0,
            is_outlier=False,
            start_time_s=0.0,
            end_time_s=1.0,
            duration_s=1.0,
            audio_start_time=now,
            audio_end_time=now,
            biomechanics_qc_fail=False,
            sync_qc_fail=False,
        )

        assert cycle.cycle_file == "cycle.pkl"
        assert cycle.cycle_index == 0
        assert cycle.duration_s == 1.0
        assert cycle.is_outlier is False

    def test_movement_cycle_requires_bio_timestamps_when_biomech_id_set(self):
        now = datetime(2024, 1, 1, 10, 0, 0)
        with pytest.raises(ValidationError):
            MovementCycle(
                study="AOA",
                study_id=1011,
                audio_processing_id=1,
                biomechanics_import_id=2,
                synchronization_id=None,
                cycle_file="cycle.pkl",
                cycle_index=0,
                is_outlier=False,
                start_time_s=0.0,
                end_time_s=1.0,
                duration_s=1.0,
                audio_start_time=now,
                audio_end_time=now,
                bio_start_time=None,
                bio_end_time=None,
                biomechanics_qc_fail=False,
                sync_qc_fail=False,
            )


class TestBiomechanicsImportInstantiation:
    """Test BiomechanicsImport instantiation and walk validation."""

    def test_walk_requires_pass_and_speed(self):
        with pytest.raises(ValidationError):
            BiomechanicsImport(
                study="AOA",
                study_id=1011,
                biomechanics_file="test.xlsx",
                biomechanics_type="Motion Analysis",
                knee="left",
                maneuver="walk",
                pass_number=None,
                speed=None,
                biomechanics_sync_method="stomp",
                biomechanics_sample_rate=100.0,
                num_sub_recordings=1,
                duration_seconds=120.0,
                num_data_points=1000,
                processing_date=datetime(2024, 1, 1, 12, 0, 0),
            )

    def test_non_walk_requires_none_pass_and_speed(self):
        with pytest.raises(ValidationError):
            BiomechanicsImport(
                study="AOA",
                study_id=1011,
                biomechanics_file="test.xlsx",
                biomechanics_type="Motion Analysis",
                knee="left",
                maneuver="sts",
                pass_number=1,
                speed=None,
                biomechanics_sync_method="stomp",
                biomechanics_sample_rate=100.0,
                num_sub_recordings=1,
                duration_seconds=120.0,
                num_data_points=1000,
                processing_date=datetime(2024, 1, 1, 12, 0, 0),
            )
