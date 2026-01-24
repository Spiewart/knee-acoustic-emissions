"""
Test module for metadata field validators added during metadata renovation.

This test file provides comprehensive coverage for:
1. New field validators in all metadata classes
2. Helper functions for metadata creation
3. Inheritance chain validation
4. Cross-field validation logic
5. Type conversion (timedelta, etc.)
"""

import pytest
from datetime import timedelta, datetime
from pathlib import Path
from pydantic import ValidationError

from src.metadata import (
    StudyMetadata,
    BiomechanicsMetadata,
    AcousticsFile,
    SynchronizationMetadata,
    AudioProcessing,
    BiomechanicsImport,
    Synchronization,
    MovementCycle,
)


class TestStudyMetadataValidators:
    """Test StudyMetadata field validators."""

    def test_valid_study_values(self):
        """Test that valid study values are accepted."""
        valid_studies = ["AOA", "preOA", "SMoCK"]
        for study in valid_studies:
            metadata = StudyMetadata(study=study, study_id=1011)
            assert metadata.study == study

    def test_study_id_validation(self):
        """Test that study_id is validated as positive integer."""
        # Valid study_id
        metadata = StudyMetadata(study="AOA", study_id=1011)
        assert metadata.study_id == 1011

        # Zero study_id should work (edge case)
        metadata = StudyMetadata(study="AOA", study_id=0)
        assert metadata.study_id == 0


class TestBiomechanicsMetadataValidators:
    """Test BiomechanicsMetadata field validators."""

    def test_linked_biomechanics_false(self):
        """Test that linked_biomechanics=False allows None values."""
        metadata = BiomechanicsMetadata(
            study="AOA",
            study_id=1011,
            linked_biomechanics=False,
            biomechanics_file=None,
            biomechanics_type=None,
            bio_sync_method=None,
            biomechanics_sample_rate=None,
        )
        assert metadata.linked_biomechanics is False
        assert metadata.biomechanics_file is None

    def test_linked_biomechanics_true_requires_fields(self):
        """Test that linked_biomechanics=True requires related fields."""
        with pytest.raises(ValidationError) as exc_info:
            BiomechanicsMetadata(
                study="AOA",
                study_id=1011,
                linked_biomechanics=True,
                biomechanics_file=None,  # Should not be None
                biomechanics_type="IMU",
                bio_sync_method="stomp",
                biomechanics_sample_rate=200.0,
            )
        assert "biomechanics_file" in str(exc_info.value)

    def test_bio_sync_method_validation_gonio(self):
        """Test bio_sync_method validation for Gonio type."""
        # Gonio must use flick
        metadata = BiomechanicsMetadata(
            study="AOA",
            study_id=1011,
            linked_biomechanics=True,
            biomechanics_file="test.csv",
            biomechanics_type="Gonio",
            bio_sync_method="flick",
            biomechanics_sample_rate=2000.0,
        )
        assert metadata.bio_sync_method == "flick"

        # Gonio with stomp should fail
        with pytest.raises(ValidationError) as exc_info:
            BiomechanicsMetadata(
                study="AOA",
                study_id=1011,
                linked_biomechanics=True,
                biomechanics_file="test.csv",
                biomechanics_type="Gonio",
                bio_sync_method="stomp",  # Invalid for Gonio
                biomechanics_sample_rate=2000.0,
            )
        assert "Gonio" in str(exc_info.value)
        assert "flick" in str(exc_info.value)

    def test_bio_sync_method_validation_imu(self):
        """Test bio_sync_method validation for IMU and Motion Analysis types."""
        for biomech_type in ["IMU", "Motion Analysis"]:
            # Must use stomp
            metadata = BiomechanicsMetadata(
                study="AOA",
                study_id=1011,
                linked_biomechanics=True,
                biomechanics_file="test.csv",
                biomechanics_type=biomech_type,
                bio_sync_method="stomp",
                biomechanics_sample_rate=200.0,
            )
            assert metadata.bio_sync_method == "stomp"

            # IMU/Motion Analysis with flick should fail
            with pytest.raises(ValidationError) as exc_info:
                BiomechanicsMetadata(
                    study="AOA",
                    study_id=1011,
                    linked_biomechanics=True,
                    biomechanics_file="test.csv",
                    biomechanics_type=biomech_type,
                    bio_sync_method="flick",  # Invalid
                    biomechanics_sample_rate=200.0,
                )
            assert biomech_type in str(exc_info.value)
            assert "stomp" in str(exc_info.value)

    def test_biomechanics_sample_rate_validation(self):
        """Test that biomechanics_sample_rate is validated as positive."""
        # Valid sample rate
        metadata = BiomechanicsMetadata(
            study="AOA",
            study_id=1011,
            linked_biomechanics=True,
            biomechanics_file="test.csv",
            biomechanics_type="IMU",
            bio_sync_method="stomp",
            biomechanics_sample_rate=200.0,
        )
        assert metadata.biomechanics_sample_rate == 200.0

        # Zero sample rate should fail
        with pytest.raises(ValidationError):
            BiomechanicsMetadata(
                study="AOA",
                study_id=1011,
                linked_biomechanics=True,
                biomechanics_file="test.csv",
                biomechanics_type="IMU",
                bio_sync_method="stomp",
                biomechanics_sample_rate=0.0,  # Invalid
            )


class TestAcousticsFileValidators:
    """Test AcousticsFile field validators."""

    def test_num_channels_validation(self):
        """Test num_channels validation."""
        # Valid num_channels
        for num_ch in [1, 2, 3, 4]:
            acoustics = AcousticsFile(
                study="AOA",
                study_id=1011,
                linked_biomechanics=False,
                biomechanics_file=None,
                biomechanics_type=None,
                bio_sync_method=None,
                biomechanics_sample_rate=None,
                audio_file_name="test.bin",
                device_serial="AE01",
                firmware_version="1.0.0",
                recording_time=datetime(2024, 1, 1, 10, 0, 0),
                file_time=datetime(2024, 1, 1, 10, 0, 0),
                file_size_mb=100.0,
                sample_rate=46875.0,
                num_channels=num_ch,
                mic_1_position="IPM",
                mic_2_position="IPL",
                mic_3_position="SPM",
                mic_4_position="SPL",
            )
            assert acoustics.num_channels == num_ch

        # Invalid num_channels
        with pytest.raises(ValidationError):
            AcousticsFile(
                study="AOA",
                study_id=1011,
                linked_biomechanics=False,
                biomechanics_file=None,
                biomechanics_type=None,
                bio_sync_method=None,
                biomechanics_sample_rate=None,
                audio_file_name="test.bin",
                device_serial="AE01",
                firmware_version="1.0.0",
                recording_time=datetime(2024, 1, 1, 10, 0, 0),
                file_time=datetime(2024, 1, 1, 10, 0, 0),
                file_size_mb=100.0,
                sample_rate=46875.0,
                num_channels=0,  # Invalid
                mic_1_position="IPM",
                mic_2_position="IPL",
                mic_3_position="SPM",
                mic_4_position="SPL",
            )

    def test_sample_rate_validation(self):
        """Test sample_rate validation."""
        # Valid sample rate
        acoustics = AcousticsFile(
            study="AOA",
            study_id=1011,
            linked_biomechanics=False,
            biomechanics_file=None,
            biomechanics_type=None,
            bio_sync_method=None,
            biomechanics_sample_rate=None,
            audio_file_name="test.bin",
            device_serial="AE01",
            firmware_version="1.0.0",
            recording_time=datetime(2024, 1, 1, 10, 0, 0),
            file_time=datetime(2024, 1, 1, 10, 0, 0),
            file_size_mb=100.0,
            sample_rate=46875.0,
            num_channels=4,
            mic_1_position="IPM",
            mic_2_position="IPL",
            mic_3_position="SPM",
            mic_4_position="SPL",
        )
        assert acoustics.sample_rate == 46875.0

        # Zero sample rate should fail
        with pytest.raises(ValidationError):
            AcousticsFile(
                study="AOA",
                study_id=1011,
                linked_biomechanics=False,
                biomechanics_file=None,
                biomechanics_type=None,
                bio_sync_method=None,
                biomechanics_sample_rate=None,
                audio_file_name="test.bin",
                device_serial="AE01",
                firmware_version="1.0.0",
                recording_time=datetime(2024, 1, 1, 10, 0, 0),
                file_time=datetime(2024, 1, 1, 10, 0, 0),
                file_size_mb=100.0,
                sample_rate=0.0,  # Invalid
                num_channels=4,
                mic_1_position="IPM",
                mic_2_position="IPL",
                mic_3_position="SPM",
                mic_4_position="SPL",
            )


class TestSynchronizationMetadataValidators:
    """Test SynchronizationMetadata field validators."""

    def test_timedelta_fields(self):
        """Test that timedelta fields are properly handled."""
        sync_meta = SynchronizationMetadata(
            study="AOA",
            study_id=1011,
            linked_biomechanics=True,
            biomechanics_file="test.csv",
            biomechanics_type="IMU",
            bio_sync_method="stomp",
            biomechanics_sample_rate=200.0,
            audio_file_name="test.bin",
            device_serial="AE01",
            firmware_version="1.0.0",
            recording_time=datetime(2024, 1, 1, 10, 0, 0),
            file_time=datetime(2024, 1, 1, 10, 0, 0),
            file_size_mb=100.0,
            sample_rate=46875.0,
            num_channels=4,
            mic_1_position="IPM",
            mic_2_position="IPL",
            mic_3_position="SPM",
            mic_4_position="SPL",
            sync_offset=timedelta(seconds=1.5),
            audio_sync_time=timedelta(seconds=10.0),
            bio_left_sync_time=timedelta(seconds=8.5),
            bio_right_sync_time=None,
            detection_left_time=timedelta(seconds=10.1),
            detection_right_time=None,
            knee="left",
            sync_method="consensus",
            consensus_methods=["rms_energy", "biomechanics"],
            biomechanics_time=timedelta(seconds=10.2),
            biomechanics_time_contralateral=None,
        )
        
        assert isinstance(sync_meta.sync_offset, timedelta)
        assert sync_meta.sync_offset.total_seconds() == 1.5
        assert isinstance(sync_meta.audio_sync_time, timedelta)
        assert sync_meta.audio_sync_time.total_seconds() == 10.0

    def test_knee_specific_validation(self):
        """Test that knee-specific fields are validated based on knee value."""
        # Left knee requires bio_left_sync_time
        sync_meta = SynchronizationMetadata(
            study="AOA",
            study_id=1011,
            linked_biomechanics=True,
            biomechanics_file="test.csv",
            biomechanics_type="IMU",
            bio_sync_method="stomp",
            biomechanics_sample_rate=200.0,
            audio_file_name="test.bin",
            device_serial="AE01",
            firmware_version="1.0.0",
            recording_time=datetime(2024, 1, 1, 10, 0, 0),
            file_time=datetime(2024, 1, 1, 10, 0, 0),
            file_size_mb=100.0,
            sample_rate=46875.0,
            num_channels=4,
            mic_1_position="IPM",
            mic_2_position="IPL",
            mic_3_position="SPM",
            mic_4_position="SPL",
            sync_offset=timedelta(seconds=1.5),
            audio_sync_time=timedelta(seconds=10.0),
            bio_left_sync_time=timedelta(seconds=8.5),
            bio_right_sync_time=None,
            detection_left_time=timedelta(seconds=10.1),
            detection_right_time=None,
            knee="left",
            sync_method="consensus",
            consensus_methods=["rms_energy", "biomechanics"],
            biomechanics_time=timedelta(seconds=10.2),
            biomechanics_time_contralateral=None,
        )
        assert sync_meta.knee == "left"
        assert sync_meta.bio_left_sync_time is not None

    def test_consensus_methods_validation(self):
        """Test that consensus_methods can be string or list."""
        # String input
        sync_meta = SynchronizationMetadata(
            study="AOA",
            study_id=1011,
            linked_biomechanics=True,
            biomechanics_file="test.csv",
            biomechanics_type="IMU",
            bio_sync_method="stomp",
            biomechanics_sample_rate=200.0,
            audio_file_name="test.bin",
            device_serial="AE01",
            firmware_version="1.0.0",
            recording_time=datetime(2024, 1, 1, 10, 0, 0),
            file_time=datetime(2024, 1, 1, 10, 0, 0),
            file_size_mb=100.0,
            sample_rate=46875.0,
            num_channels=4,
            mic_1_position="IPM",
            mic_2_position="IPL",
            mic_3_position="SPM",
            mic_4_position="SPL",
            sync_offset=timedelta(seconds=1.5),
            audio_sync_time=timedelta(seconds=10.0),
            bio_left_sync_time=timedelta(seconds=8.5),
            bio_right_sync_time=None,
            detection_left_time=timedelta(seconds=10.1),
            detection_right_time=None,
            knee="left",
            sync_method="consensus",
            consensus_methods="rms_energy,biomechanics",  # String
            biomechanics_time=timedelta(seconds=10.2),
            biomechanics_time_contralateral=None,
        )
        assert isinstance(sync_meta.consensus_methods, list)
        assert "rms_energy" in sync_meta.consensus_methods


class TestBiomechanicsImportValidators:
    """Test BiomechanicsImport field validators."""

    def test_num_sub_recordings_validation_sts(self):
        """Test num_sub_recordings validation for sit-to-stand."""
        # sts must have exactly 1 sub-recording
        biomech = BiomechanicsImport(
            study="AOA",
            study_id=1011,
            maneuver="sts",
            biomechanics_file="test.csv",
            sheet_name="Sheet1",
            processing_date=datetime(2024, 1, 1),
            sample_rate=200.0,
            num_sub_recordings=1,
            num_passes=0,
            num_data_points=1000,
            duration_seconds=5.0,
        )
        assert biomech.num_sub_recordings == 1

        # sts with num_sub_recordings != 1 should fail
        with pytest.raises(ValidationError) as exc_info:
            BiomechanicsImport(
                study="AOA",
                study_id=1011,
                maneuver="sts",
                biomechanics_file="test.csv",
                sheet_name="Sheet1",
                processing_date=datetime(2024, 1, 1),
                sample_rate=200.0,
                num_sub_recordings=2,  # Invalid for sts
                num_passes=0,
                num_data_points=1000,
                duration_seconds=5.0,
            )
        assert "sit to stand" in str(exc_info.value).lower()

    def test_num_sub_recordings_validation_fe(self):
        """Test num_sub_recordings validation for flexion-extension."""
        # fe must have exactly 1 sub-recording
        biomech = BiomechanicsImport(
            study="AOA",
            study_id=1011,
            maneuver="fe",
            biomechanics_file="test.csv",
            sheet_name="Sheet1",
            processing_date=datetime(2024, 1, 1),
            sample_rate=2000.0,
            num_sub_recordings=1,
            num_passes=0,
            num_data_points=10000,
            duration_seconds=5.0,
        )
        assert biomech.num_sub_recordings == 1

        # fe with num_sub_recordings != 1 should fail
        with pytest.raises(ValidationError) as exc_info:
            BiomechanicsImport(
                study="AOA",
                study_id=1011,
                maneuver="fe",
                biomechanics_file="test.csv",
                sheet_name="Sheet1",
                processing_date=datetime(2024, 1, 1),
                sample_rate=2000.0,
                num_sub_recordings=3,  # Invalid for fe
                num_passes=0,
                num_data_points=10000,
                duration_seconds=5.0,
            )
        assert "flexion" in str(exc_info.value).lower()

    def test_num_sub_recordings_validation_walk(self):
        """Test num_sub_recordings validation for walking."""
        # walk must have >= 1 sub-recording
        for num_sub in [1, 2, 3, 5]:
            biomech = BiomechanicsImport(
                study="AOA",
                study_id=1011,
                maneuver="walk",
                biomechanics_file="test.csv",
                sheet_name="Sheet1",
                processing_date=datetime(2024, 1, 1),
                sample_rate=200.0,
                num_sub_recordings=num_sub,
                num_passes=num_sub,
                num_data_points=10000,
                duration_seconds=50.0,
            )
            assert biomech.num_sub_recordings == num_sub

        # walk with num_sub_recordings == 0 should fail
        with pytest.raises(ValidationError) as exc_info:
            BiomechanicsImport(
                study="AOA",
                study_id=1011,
                maneuver="walk",
                biomechanics_file="test.csv",
                sheet_name="Sheet1",
                processing_date=datetime(2024, 1, 1),
                sample_rate=200.0,
                num_sub_recordings=0,  # Invalid for walk
                num_passes=0,
                num_data_points=10000,
                duration_seconds=50.0,
            )
        assert "walking" in str(exc_info.value).lower()


class TestSynchronizationValidators:
    """Test Synchronization field validators."""

    def test_linked_biomechanics_required_true(self):
        """Test that Synchronization requires linked_biomechanics=True."""
        # Synchronization should have linked_biomechanics as Literal[True]
        # This is enforced at the type level, so we just verify it works
        sync = Synchronization(
            study="AOA",
            study_id=1011,
            linked_biomechanics=True,
            biomechanics_file="test.csv",
            biomechanics_type="IMU",
            bio_sync_method="stomp",
            biomechanics_sample_rate=200.0,
            audio_file_name="test.bin",
            device_serial="AE01",
            firmware_version="1.0.0",
            recording_time=datetime(2024, 1, 1, 10, 0, 0),
            file_time=datetime(2024, 1, 1, 10, 0, 0),
            file_size_mb=100.0,
            sample_rate=46875.0,
            num_channels=4,
            mic_1_position="IPM",
            mic_2_position="IPL",
            mic_3_position="SPM",
            mic_4_position="SPL",
            sync_offset=timedelta(seconds=1.5),
            audio_sync_time=timedelta(seconds=10.0),
            bio_left_sync_time=timedelta(seconds=8.5),
            bio_right_sync_time=None,
            detection_left_time=timedelta(seconds=10.1),
            detection_right_time=None,
            knee="left",
            sync_method="consensus",
            consensus_methods=["rms_energy", "biomechanics"],
            biomechanics_time=timedelta(seconds=10.2),
            biomechanics_time_contralateral=None,
            total_cycles_extracted=10,
            clean_cycles=8,
            outlier_cycles=2,
        )
        assert sync.linked_biomechanics is True

    def test_cycle_counts_required(self):
        """Test that cycle extraction counts are required (no defaults)."""
        # Creating Synchronization without cycle counts should fail
        with pytest.raises(ValidationError) as exc_info:
            Synchronization(
                study="AOA",
                study_id=1011,
                linked_biomechanics=True,
                biomechanics_file="test.csv",
                biomechanics_type="IMU",
                bio_sync_method="stomp",
                biomechanics_sample_rate=200.0,
                audio_file_name="test.bin",
                device_serial="AE01",
                firmware_version="1.0.0",
                recording_time=datetime(2024, 1, 1, 10, 0, 0),
                file_time=datetime(2024, 1, 1, 10, 0, 0),
                file_size_mb=100.0,
                sample_rate=46875.0,
                num_channels=4,
                mic_1_position="IPM",
                mic_2_position="IPL",
                mic_3_position="SPM",
                mic_4_position="SPL",
                sync_offset=timedelta(seconds=1.5),
                audio_sync_time=timedelta(seconds=10.0),
                bio_left_sync_time=timedelta(seconds=8.5),
                bio_right_sync_time=None,
                detection_left_time=timedelta(seconds=10.1),
                detection_right_time=None,
                knee="left",
                sync_method="consensus",
                consensus_methods=["rms_energy", "biomechanics"],
                biomechanics_time=timedelta(seconds=10.2),
                biomechanics_time_contralateral=None,
                # Missing: total_cycles_extracted, clean_cycles, outlier_cycles
            )
        # Should fail due to missing required fields
        assert "total_cycles_extracted" in str(exc_info.value) or "Field required" in str(exc_info.value)


class TestMovementCycleValidators:
    """Test MovementCycle field validators."""

    def test_is_outlier_validation(self):
        """Test that is_outlier is set True if any QC fails."""
        # Create MovementCycle with QC fail
        cycle = MovementCycle(
            study="AOA",
            study_id=1011,
            linked_biomechanics=True,
            biomechanics_file="test.csv",
            biomechanics_type="IMU",
            bio_sync_method="stomp",
            biomechanics_sample_rate=200.0,
            audio_file_name="test.bin",
            device_serial="AE01",
            firmware_version="1.0.0",
            recording_time=datetime(2024, 1, 1, 10, 0, 0),
            file_time=datetime(2024, 1, 1, 10, 0, 0),
            file_size_mb=100.0,
            sample_rate=46875.0,
            num_channels=4,
            mic_1_position="IPM",
            mic_2_position="IPL",
            mic_3_position="SPM",
            mic_4_position="SPL",
            processing_date=datetime(2024, 1, 1),
            qc_fail_segments=[],
            qc_signal_dropout_overall=False,
            qc_signal_dropout_ch1=False,
            qc_signal_dropout_ch2=False,
            qc_signal_dropout_ch3=False,
            qc_signal_dropout_ch4=False,
            qc_artifact_overall=False,
            qc_artifact_ch1=False,
            qc_artifact_ch2=False,
            qc_artifact_ch3=False,
            qc_artifact_ch4=False,
            cycle_file="cycle_0.csv",
            cycle_index=0,
            pass_number=1,
            speed=1.2,
            duration_s=1.0,
            start_time_s=0.0,
            end_time_s=1.0,
            audio_start_time=timedelta(seconds=10.0),
            audio_end_time=timedelta(seconds=11.0),
            bio_start_time=timedelta(seconds=8.5),
            bio_end_time=timedelta(seconds=9.5),
            biomechanics_qc_fail=True,  # QC fail
            sync_qc_fail=False,
            is_outlier=False,  # Will be overridden to True
        )
        # is_outlier should be True due to biomechanics_qc_fail
        assert cycle.is_outlier is True


class TestInheritanceChain:
    """Test that inheritance chain works correctly."""

    def test_study_metadata_inheritance(self):
        """Test that all classes properly inherit from StudyMetadata."""
        # BiomechanicsMetadata inherits StudyMetadata
        biomech = BiomechanicsMetadata(
            study="AOA",
            study_id=1011,
            linked_biomechanics=False,
            biomechanics_file=None,
            biomechanics_type=None,
            bio_sync_method=None,
            biomechanics_sample_rate=None,
        )
        assert hasattr(biomech, "study")
        assert hasattr(biomech, "study_id")

        # BiomechanicsImport inherits StudyMetadata
        biomech_import = BiomechanicsImport(
            study="preOA",
            study_id=2022,
            maneuver="walk",
            biomechanics_file="test.csv",
            sheet_name="Sheet1",
            processing_date=datetime(2024, 1, 1),
            sample_rate=200.0,
            num_sub_recordings=2,
            num_passes=2,
            num_data_points=10000,
            duration_seconds=50.0,
        )
        assert hasattr(biomech_import, "study")
        assert hasattr(biomech_import, "study_id")

    def test_biomechanics_metadata_inheritance(self):
        """Test that AcousticsFile inherits BiomechanicsMetadata."""
        acoustics = AcousticsFile(
            study="SMoCK",
            study_id=3033,
            linked_biomechanics=True,
            biomechanics_file="test.csv",
            biomechanics_type="Gonio",
            bio_sync_method="flick",
            biomechanics_sample_rate=2000.0,
            audio_file_name="test.bin",
            device_serial="AE01",
            firmware_version="1.0.0",
            recording_time=datetime(2024, 1, 1, 10, 0, 0),
            file_time=datetime(2024, 1, 1, 10, 0, 0),
            file_size_mb=100.0,
            sample_rate=46875.0,
            num_channels=4,
            mic_1_position="IPM",
            mic_2_position="IPL",
            mic_3_position="SPM",
            mic_4_position="SPL",
        )
        # Should have StudyMetadata fields
        assert hasattr(acoustics, "study")
        assert hasattr(acoustics, "study_id")
        # Should have BiomechanicsMetadata fields
        assert hasattr(acoustics, "linked_biomechanics")
        assert hasattr(acoustics, "biomechanics_file")
        assert hasattr(acoustics, "bio_sync_method")

    def test_full_inheritance_chain(self):
        """Test full inheritance chain: StudyMetadata → BiomechanicsMetadata → AcousticsFile → SynchronizationMetadata → Synchronization."""
        sync = Synchronization(
            # StudyMetadata fields
            study="AOA",
            study_id=1011,
            # BiomechanicsMetadata fields
            linked_biomechanics=True,
            biomechanics_file="test.csv",
            biomechanics_type="IMU",
            bio_sync_method="stomp",
            biomechanics_sample_rate=200.0,
            # AcousticsFile fields
            audio_file_name="test.bin",
            device_serial="AE01",
            firmware_version="1.0.0",
            recording_time=datetime(2024, 1, 1, 10, 0, 0),
            file_time=datetime(2024, 1, 1, 10, 0, 0),
            file_size_mb=100.0,
            sample_rate=46875.0,
            num_channels=4,
            mic_1_position="IPM",
            mic_2_position="IPL",
            mic_3_position="SPM",
            mic_4_position="SPL",
            # SynchronizationMetadata fields
            sync_offset=timedelta(seconds=1.5),
            audio_sync_time=timedelta(seconds=10.0),
            bio_left_sync_time=timedelta(seconds=8.5),
            bio_right_sync_time=None,
            detection_left_time=timedelta(seconds=10.1),
            detection_right_time=None,
            knee="left",
            sync_method="consensus",
            consensus_methods=["rms_energy", "biomechanics"],
            biomechanics_time=timedelta(seconds=10.2),
            biomechanics_time_contralateral=None,
            # Synchronization fields
            total_cycles_extracted=10,
            clean_cycles=8,
            outlier_cycles=2,
        )
        
        # Verify all inherited fields are accessible
        assert sync.study == "AOA"
        assert sync.study_id == 1011
        assert sync.linked_biomechanics is True
        assert sync.biomechanics_file == "test.csv"
        assert sync.audio_file_name == "test.bin"
        assert sync.num_channels == 4
        assert sync.sync_offset.total_seconds() == 1.5
        assert sync.total_cycles_extracted == 10


class TestExcelExport:
    """Test to_dict() methods for Excel export."""

    def test_timedelta_converted_to_seconds(self):
        """Test that timedelta fields are converted to seconds for Excel."""
        sync = Synchronization(
            study="AOA",
            study_id=1011,
            linked_biomechanics=True,
            biomechanics_file="test.csv",
            biomechanics_type="IMU",
            bio_sync_method="stomp",
            biomechanics_sample_rate=200.0,
            audio_file_name="test.bin",
            device_serial="AE01",
            firmware_version="1.0.0",
            recording_time=datetime(2024, 1, 1, 10, 0, 0),
            file_time=datetime(2024, 1, 1, 10, 0, 0),
            file_size_mb=100.0,
            sample_rate=46875.0,
            num_channels=4,
            mic_1_position="IPM",
            mic_2_position="IPL",
            mic_3_position="SPM",
            mic_4_position="SPL",
            sync_offset=timedelta(seconds=1.5),
            audio_sync_time=timedelta(seconds=10.0),
            bio_left_sync_time=timedelta(seconds=8.5),
            bio_right_sync_time=None,
            detection_left_time=timedelta(seconds=10.1),
            detection_right_time=None,
            knee="left",
            sync_method="consensus",
            consensus_methods=["rms_energy", "biomechanics"],
            biomechanics_time=timedelta(seconds=10.2),
            biomechanics_time_contralateral=None,
            total_cycles_extracted=10,
            clean_cycles=8,
            outlier_cycles=2,
        )
        
        export_dict = sync.to_dict()
        
        # Timedeltas should be converted to floats (seconds)
        assert isinstance(export_dict["Sync Offset (s)"], float)
        assert export_dict["Sync Offset (s)"] == 1.5
        assert isinstance(export_dict["Audio Sync Time (s)"], float)
        assert export_dict["Audio Sync Time (s)"] == 10.0

    def test_num_sub_recordings_in_export(self):
        """Test that num_sub_recordings is exported with correct column name."""
        biomech = BiomechanicsImport(
            study="AOA",
            study_id=1011,
            maneuver="walk",
            biomechanics_file="test.csv",
            sheet_name="Sheet1",
            processing_date=datetime(2024, 1, 1),
            sample_rate=200.0,
            num_sub_recordings=3,
            num_passes=3,
            num_data_points=15000,
            duration_seconds=75.0,
        )
        
        export_dict = biomech.to_dict()
        
        # Should be exported as "Num Sub-Recordings"
        assert "Num Sub-Recordings" in export_dict
        assert export_dict["Num Sub-Recordings"] == 3
