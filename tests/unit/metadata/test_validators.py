"""Consolidated tests for metadata validators and exports.

Aligned with the normalized metadata models in src/metadata.py.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.metadata import (
    AudioProcessing,
    BiomechanicsImport,
    MovementCycle,
    StudyMetadata,
    Synchronization,
)


def _base_audio_kwargs():
    return {
        "study": "AOA",
        "study_id": 1011,
        "audio_file_name": "test.bin",
        "device_serial": "AE01",
        "firmware_version": 1,
        "file_time": datetime(2024, 1, 1, 10, 0, 0),
        "file_size_mb": 100.0,
        "recording_date": datetime(2024, 1, 1),
        "recording_time": datetime(2024, 1, 1, 10, 0, 0),
        "knee": "left",
        "maneuver": "walk",
        "num_channels": 4,
        "sample_rate": 46875.0,
        "mic_1_position": "IPM",
        "mic_2_position": "IPL",
        "mic_3_position": "SPM",
        "mic_4_position": "SPL",
        "processing_date": datetime(2024, 1, 1, 12, 0, 0),
        "processing_status": "success",
    }


def _base_biomech_kwargs():
    return {
        "study": "AOA",
        "study_id": 1011,
        "biomechanics_file": "test.xlsx",
        "biomechanics_type": "Motion Analysis",
        "knee": "left",
        "maneuver": "walk",
        "biomechanics_sync_method": "stomp",
        "biomechanics_sample_rate": 100.0,
        "num_sub_recordings": 1,
        "duration_seconds": 120.0,
        "num_data_points": 1000,
        "processing_date": datetime(2024, 1, 1, 12, 0, 0),
    }


def _base_sync_kwargs():
    return {
        "study": "AOA",
        "study_id": 1011,
        "audio_processing_id": 1,
        "biomechanics_import_id": 1,
        "sync_file_name": "sync.pkl",
        "processing_date": datetime(2024, 1, 1, 12, 0, 0),
    }


def _base_cycle_kwargs():
    return {
        "study": "AOA",
        "study_id": 1011,
        "audio_processing_id": 1,
        "cycle_file": "cycle_0001.pkl",
        "cycle_index": 0,
        "is_outlier": False,
        "start_time_s": 0.0,
        "end_time_s": 1.0,
        "duration_s": 1.0,
        "audio_start_time": datetime(2024, 1, 1, 10, 0, 0),
        "audio_end_time": datetime(2024, 1, 1, 10, 0, 1),
        "biomechanics_qc_fail": False,
        "sync_qc_fail": False,
    }


def _with_overrides(base: dict, **overrides) -> dict:
    data = dict(base)
    data.update(overrides)
    return data


class TestStudyMetadataValidators:
    def test_valid_study_values(self):
        valid_studies = ["AOA", "preOA", "SMoCK"]
        for study in valid_studies:
            metadata = StudyMetadata(study=study, study_id=1011)
            assert metadata.study == study

    def test_study_id_validation(self):
        metadata = StudyMetadata(study="AOA", study_id=1011)
        assert metadata.study_id == 1011

        metadata = StudyMetadata(study="AOA", study_id=0)
        assert metadata.study_id == 0


class TestAudioProcessingValidators:
    def test_num_channels_validation(self):
        for num_ch in [1, 2, 3, 4]:
            audio = AudioProcessing(**_with_overrides(_base_audio_kwargs(), num_channels=num_ch))
            assert audio.num_channels == num_ch

        with pytest.raises(ValidationError):
            AudioProcessing(**_with_overrides(_base_audio_kwargs(), num_channels=0))

        with pytest.raises(ValidationError):
            AudioProcessing(**_with_overrides(_base_audio_kwargs(), num_channels=5))


class TestBiomechanicsImportValidators:
    def test_biomechanics_import_basic(self):
        """Test basic BiomechanicsImport creation with required fields."""
        biomech = BiomechanicsImport(**_base_biomech_kwargs())
        assert biomech.biomechanics_file == "test.xlsx"
        assert biomech.knee == "left"

    def test_biomechanics_sample_rate_positive(self):
        with pytest.raises(ValidationError):
            BiomechanicsImport(
                **_with_overrides(
                    _base_biomech_kwargs(),
                    biomechanics_sample_rate=0.0,
                )
            )


class TestSynchronizationValidators:
    def test_positive_ids_required(self):
        with pytest.raises(ValidationError):
            Synchronization(**_with_overrides(_base_sync_kwargs(), audio_processing_id=0))

        with pytest.raises(ValidationError):
            Synchronization(**_with_overrides(_base_sync_kwargs(), biomechanics_import_id=-1))


class TestMovementCycleValidators:
    def test_requires_bio_timestamps_when_biomech_id_set(self):
        with pytest.raises(ValidationError):
            MovementCycle(
                **_base_cycle_kwargs(),
                biomechanics_import_id=1,
                bio_start_time=None,
                bio_end_time=None,
            )

    def test_requires_pass_speed_when_sync_id_set(self):
        with pytest.raises(ValidationError):
            MovementCycle(
                **_base_cycle_kwargs(),
                synchronization_id=1,
                pass_number=None,
                speed=None,
            )
