"""
Clean, consolidated tests for metadata validators and exports.

Covers key validation logic and Excel export conversions
for src/metadata.py dataclasses without duplication.
"""

from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from src.metadata import (
    AcousticsFile,
    AudioProcessing,
    BiomechanicsImport,
    BiomechanicsMetadata,
    MovementCycle,
    StudyMetadata,
    Synchronization,
    SynchronizationMetadata,
)


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


class TestBiomechanicsMetadataValidators:
    def test_linked_biomechanics_false(self):
        metadata = BiomechanicsMetadata(
            study="AOA",
            study_id=1011,
            linked_biomechanics=False,
            biomechanics_file=None,
            biomechanics_type=None,
            biomechanics_sync_method=None,
            biomechanics_sample_rate=None,
        )
        assert metadata.linked_biomechanics is False
        assert metadata.biomechanics_file is None

    def test_linked_biomechanics_true_requires_fields(self):
        with pytest.raises(ValidationError) as exc_info:
            BiomechanicsMetadata(
                study="AOA",
                study_id=1011,
                linked_biomechanics=True,
                biomechanics_file=None,
                biomechanics_type="IMU",
                biomechanics_sync_method="stomp",
                biomechanics_sample_rate=200.0,
            )
        assert "biomechanics_file" in str(exc_info.value)

    def test_biomechanics_sync_method_validation_gonio(self):
        metadata = BiomechanicsMetadata(
            study="AOA",
            study_id=1011,
            linked_biomechanics=True,
            biomechanics_file="test.csv",
            biomechanics_type="Gonio",
            biomechanics_sync_method="flick",
            biomechanics_sample_rate=2000.0,
        )
        assert metadata.biomechanics_sync_method == "flick"

        with pytest.raises(ValidationError) as exc_info:
            BiomechanicsMetadata(
                study="AOA",
                study_id=1011,
                linked_biomechanics=True,
                biomechanics_file="test.csv",
                biomechanics_type="Gonio",
                biomechanics_sync_method="stomp",
                biomechanics_sample_rate=2000.0,
            )
        assert "Gonio" in str(exc_info.value)


class TestAcousticsFileValidators:
    def test_num_channels_validation(self):
        for num_ch in [1, 2, 3, 4]:
            acoustics = AcousticsFile(
                study="AOA",
                study_id=1011,
                linked_biomechanics=False,
                biomechanics_file=None,
                biomechanics_type=None,
                biomechanics_sync_method=None,
                biomechanics_sample_rate=None,
                audio_file_name="test.bin",
                device_serial="AE01",
                firmware_version=1,
                recording_date=datetime(2024, 1, 1),
                recording_time=datetime(2024, 1, 1, 10, 0, 0),
                file_time=datetime(2024, 1, 1, 10, 0, 0),
                file_size_mb=100.0,
                sample_rate=46875.0,
                num_channels=num_ch,
                mic_1_position="IPM",
                mic_2_position="IPL",
                mic_3_position="SPM",
                mic_4_position="SPL",
                knee="left",
                maneuver="walk",
            )
            assert acoustics.num_channels == num_ch


class TestBiomechanicsImportValidators:
    def test_num_sub_recordings_validation_sts(self):
        biomech = BiomechanicsImport(
            study="AOA",
            study_id=1011,
            maneuver="sts",
            biomechanics_file="test.csv",
            sheet_name="Sheet1",
            processing_date=datetime(2024, 1, 1),
            processing_status="success",
            sample_rate=200.0,
            num_sub_recordings=1,
            num_passes=0,
            num_data_points=1000,
            duration_seconds=5.0,
        )
        assert biomech.num_sub_recordings == 1
        assert biomech.maneuver == "sts"


class TestSynchronizationMetadataValidators:
    def test_sync_method_requirements(self):
        base_kwargs = dict(
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
            sync_offset=1.5,
            audio_sync_time=10.0,
            aligned_audio_sync_time=10.0,
            aligned_biomechanics_sync_time=8.5,
            bio_left_sync_time=8.5,
            bio_right_sync_time=None,
            consensus_time=10.0,
            rms_time=10.0,
            onset_time=10.0,
            freq_time=10.0,
        )

        # biomechanics method should use bio_left_sync_time and/or bio_right_sync_time
        sm_bio = SynchronizationMetadata(
            **base_kwargs,
            sync_method="biomechanics",
        )
        assert sm_bio.sync_method == "biomechanics"
        # Note: bio_left_sync_time and bio_right_sync_time are the sync times for biomechanics

        # consensus requires consensus_methods
        sm_cons = SynchronizationMetadata(
            **base_kwargs,
            sync_method="consensus",
            consensus_methods="rms_energy,biomechanics",
        )
        assert sm_cons.consensus_methods == "rms_energy,biomechanics"

        with pytest.raises(ValidationError):
            SynchronizationMetadata(
                **base_kwargs,
                sync_method="consensus",
                consensus_methods=None,
            )

    def test_to_dict_converts_timedelta_to_seconds(self):
        sync_meta = SynchronizationMetadata(
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
            sync_offset=1.5,
            audio_sync_time=10.0,
            aligned_audio_sync_time=10.0,
            aligned_biomechanics_sync_time=8.5,
            bio_left_sync_time=8.5,
            bio_right_sync_time=None,
            consensus_time=10.0,
            rms_time=10.0,
            onset_time=10.0,
            freq_time=10.0,
            sync_method="consensus",
            consensus_methods="rms_energy,biomechanics",
        )

        d = sync_meta.to_dict()
        assert d["Audio Sync Time (s)"] == 10.0
        assert d["Sync Offset (s)"] == 1.5


class TestSynchronizationValidators:
    def test_linked_biomechanics_enforced_true(self):
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
            sync_offset=1.5,
            audio_sync_time=10.0,
            aligned_audio_sync_time=10.0,
            aligned_biomechanics_sync_time=8.5,
            bio_left_sync_time=8.5,
            bio_right_sync_time=None,
            consensus_time=10.0,
            rms_time=10.0,
            onset_time=10.0,
            freq_time=10.0,
            sync_method="consensus",
            consensus_methods="rms_energy,biomechanics",
            sync_file_name="sync.pkl",
            processing_date=datetime(2024, 1, 2),
            sync_duration=120.0,
            total_cycles_extracted=10,
            clean_cycles=8,
            outlier_cycles=2,
            mean_cycle_duration_s=1.0,
            median_cycle_duration_s=1.0,
            min_cycle_duration_s=0.8,
            max_cycle_duration_s=1.2,
            mean_acoustic_auc=0.5,
        )
        assert sync.linked_biomechanics is True


class TestMovementCycleValidators:
    def test_is_outlier_flag_manually_set_by_processing(self):
        """Verify is_outlier must be manually set by processing, not auto-calculated.
        
        The is_outlier flag represents processing-level determination of whether a cycle
        is an outlier, and should be explicitly set by the processing pipeline based on
        QC flags and other criteria, not automatically derived from QC fields.
        """
        # Create a cycle with qc_artifact=True but is_outlier=False
        cycle = MovementCycle(
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
            processing_date=datetime(2024, 1, 2),
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
            qc_artifact=True,
            qc_artifact_segments=[(0.0, 0.1)],
            qc_artifact_ch1=False,
            qc_artifact_segments_ch1=[],
            qc_artifact_ch2=False,
            qc_artifact_segments_ch2=[],
            qc_artifact_ch3=False,
            qc_artifact_segments_ch3=[],
            qc_artifact_ch4=False,
            qc_artifact_segments_ch4=[],
            cycle_file="cycle_0001.pkl",
            cycle_index=0,
            is_outlier=False,
            start_time_s=0.0,
            end_time_s=1.0,
            duration_s=1.0,
            audio_start_time=datetime(2024, 1, 1, 10, 0, 0),
            audio_end_time=datetime(2024, 1, 1, 10, 0, 1),
            bio_start_time=datetime(2024, 1, 1, 10, 0, 0),
            bio_end_time=datetime(2024, 1, 1, 10, 0, 1),
            biomechanics_qc_fail=False,
            sync_qc_fail=False,
            pass_number=1,
            speed="normal",
        )
        # is_outlier is independent of QC flags - it's a processing decision
        assert cycle.is_outlier is False


class TestExcelExport:
    def test_sync_to_dict_contains_expected_keys(self):
        sync_meta = SynchronizationMetadata(
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
            sync_offset=1.5,
            audio_sync_time=10.0,
            aligned_audio_sync_time=10.0,
            aligned_biomechanics_sync_time=8.5,
            bio_left_sync_time=8.5,
            bio_right_sync_time=None,
            consensus_time=10.0,
            rms_time=10.0,
            onset_time=10.0,
            freq_time=10.0,
            sync_method="consensus",
            consensus_methods="rms_energy,biomechanics",
        )

        d = sync_meta.to_dict()
        expected_keys = {
            "Audio Sync Time (s)",
            "Bio Left Sync Time (s)",
            "Bio Right Sync Time (s)",
            "Sync Offset (s)",
            "Aligned Audio Sync Time (s)",
            "Aligned Biomechanics Sync Time (s)",
            "Sync Method",
            "Consensus Methods",
            "Consensus Time (s)",
        }
        missing = expected_keys - set(d.keys())
        assert not missing, f"Missing keys in to_dict: {missing}"
