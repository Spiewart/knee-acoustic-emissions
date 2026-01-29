"""
Tests for maneuver normalization and Speed/Pass validation.

Covers:
1. CLI maneuver abbreviation normalization (fe -> flexion_extension, sts -> sit_to_stand)
2. Maneuver being passed through audio records to sync records
3. Speed and Pass validation based on maneuver type
4. SyncData properly storing and passing speed/pass_number
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from src.metadata import Synchronization, SynchronizationMetadata
from src.orchestration.participant import _normalize_maneuver
from src.orchestration.participant_processor import SyncData


class TestManeuverNormalization:
    """Test CLI maneuver abbreviation normalization."""

    def test_normalize_maneuver_fe_to_flexion_extension(self):
        """Test that 'fe' abbreviation normalizes to 'flexion_extension'."""
        result = _normalize_maneuver("fe")
        assert result == "flexion_extension"

    def test_normalize_maneuver_sts_to_sit_to_stand(self):
        """Test that 'sts' abbreviation normalizes to 'sit_to_stand'."""
        result = _normalize_maneuver("sts")
        assert result == "sit_to_stand"

    def test_normalize_maneuver_walk_remains_walk(self):
        """Test that 'walk' remains 'walk'."""
        result = _normalize_maneuver("walk")
        assert result == "walk"

    def test_normalize_maneuver_full_names_pass_through(self):
        """Test that full maneuver names pass through unchanged."""
        assert _normalize_maneuver("flexion_extension") == "flexion_extension"
        assert _normalize_maneuver("sit_to_stand") == "sit_to_stand"

    def test_normalize_maneuver_case_insensitive(self):
        """Test that normalization is case insensitive."""
        assert _normalize_maneuver("FE") == "flexion_extension"
        assert _normalize_maneuver("Fe") == "flexion_extension"
        assert _normalize_maneuver("STS") == "sit_to_stand"
        assert _normalize_maneuver("Sts") == "sit_to_stand"
        assert _normalize_maneuver("WALK") == "walk"

    def test_normalize_maneuver_none_input(self):
        """Test that None input returns None."""
        result = _normalize_maneuver(None)
        assert result is None


class TestSpeedPassValidationForWalk:
    """Test Speed and Pass validation for walk maneuvers."""

    def test_walk_maneuver_requires_pass_number(self):
        """Test that walk maneuver with None pass_number raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            Synchronization(
                study="AOA",
                study_id=1011,
                linked_biomechanics=True,
                biomechanics_file="test.xlsx",
                biomechanics_type="Motion Analysis",
                biomechanics_sync_method="stomp",
                biomechanics_sample_rate=100.0,
                audio_file_name="test.bin",
                device_serial="TEST123",
                firmware_version=1,
                file_time=datetime(2024, 1, 1, 10, 0, 0),
                file_size_mb=100.0,
                recording_date=datetime(2024, 1, 1),
                recording_time=datetime(2024, 1, 1, 10, 0, 0),
                knee="left",
                maneuver="walk",
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
                rms_time=5.0,
                onset_time=5.0,
                freq_time=5.0,
                sync_file_name="test_sync.pkl",
                processing_date=datetime(2024, 1, 1, 12, 0, 0),
                sync_duration=120.0,
                pass_number=None,  # Should fail
                speed="normal",
            )
        assert "pass_number is required for walk maneuvers" in str(exc_info.value)

    def test_walk_maneuver_requires_speed(self):
        """Test that walk maneuver with None speed raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            Synchronization(
                study="AOA",
                study_id=1011,
                linked_biomechanics=True,
                biomechanics_file="test.xlsx",
                biomechanics_type="Motion Analysis",
                biomechanics_sync_method="stomp",
                biomechanics_sample_rate=100.0,
                audio_file_name="test.bin",
                device_serial="TEST123",
                firmware_version=1,
                file_time=datetime(2024, 1, 1, 10, 0, 0),
                file_size_mb=100.0,
                recording_date=datetime(2024, 1, 1),
                recording_time=datetime(2024, 1, 1, 10, 0, 0),
                knee="left",
                maneuver="walk",
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
                rms_time=5.0,
                onset_time=5.0,
                freq_time=5.0,
                sync_file_name="test_sync.pkl",
                processing_date=datetime(2024, 1, 1, 12, 0, 0),
                sync_duration=120.0,
                pass_number=1,
                speed=None,  # Should fail
            )
        assert "speed is required for walk maneuvers" in str(exc_info.value)

    def test_walk_maneuver_pass_number_must_be_positive(self):
        """Test that walk maneuver pass_number must be positive."""
        with pytest.raises(ValidationError) as exc_info:
            Synchronization(
                study="AOA",
                study_id=1011,
                linked_biomechanics=True,
                biomechanics_file="test.xlsx",
                biomechanics_type="Motion Analysis",
                biomechanics_sync_method="stomp",
                biomechanics_sample_rate=100.0,
                audio_file_name="test.bin",
                device_serial="TEST123",
                firmware_version=1,
                file_time=datetime(2024, 1, 1, 10, 0, 0),
                file_size_mb=100.0,
                recording_date=datetime(2024, 1, 1),
                recording_time=datetime(2024, 1, 1, 10, 0, 0),
                knee="left",
                maneuver="walk",
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
                rms_time=5.0,
                onset_time=5.0,
                freq_time=5.0,
                sync_file_name="test_sync.pkl",
                processing_date=datetime(2024, 1, 1, 12, 0, 0),
                sync_duration=120.0,
                pass_number=0,  # Should fail - not positive
                speed="normal",
            )
        assert "pass_number must be positive" in str(exc_info.value)

    def test_walk_maneuver_with_valid_speed_passes(self):
        """Test that walk maneuver with valid speed and pass_number passes validation."""
        for speed in ["slow", "normal", "fast", "medium", "comfortable"]:
            sync = Synchronization(
                study="AOA",
                study_id=1011,
                linked_biomechanics=True,
                biomechanics_file="test.xlsx",
                biomechanics_type="Motion Analysis",
                biomechanics_sync_method="stomp",
                biomechanics_sample_rate=100.0,
                audio_file_name="test.bin",
                device_serial="TEST123",
                firmware_version=1,
                file_time=datetime(2024, 1, 1, 10, 0, 0),
                file_size_mb=100.0,
                recording_date=datetime(2024, 1, 1),
                recording_time=datetime(2024, 1, 1, 10, 0, 0),
                knee="left",
                maneuver="walk",
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
                rms_time=5.0,
                onset_time=5.0,
                freq_time=5.0,
                sync_file_name="test_sync.pkl",
                processing_date=datetime(2024, 1, 1, 12, 0, 0),
                sync_duration=120.0,
                pass_number=1,
                speed=speed,
            )
            assert sync.speed == speed
            assert sync.pass_number == 1


class TestSpeedPassValidationForNonWalk:
    """Test Speed and Pass validation for non-walk maneuvers."""

    @pytest.mark.parametrize("maneuver", ["fe", "sts"])
    def test_non_walk_maneuver_requires_none_pass_number(self, maneuver):
        """Test that non-walk maneuvers with non-None pass_number fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            Synchronization(
                study="AOA",
                study_id=1011,
                linked_biomechanics=True,
                biomechanics_file="test.xlsx",
                biomechanics_type="Motion Analysis",
                biomechanics_sync_method="stomp",
                biomechanics_sample_rate=100.0,
                audio_file_name="test.bin",
                device_serial="TEST123",
                firmware_version=1,
                file_time=datetime(2024, 1, 1, 10, 0, 0),
                file_size_mb=100.0,
                recording_date=datetime(2024, 1, 1),
                recording_time=datetime(2024, 1, 1, 10, 0, 0),
                knee="left",
                maneuver=maneuver,
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
                rms_time=5.0,
                onset_time=5.0,
                freq_time=5.0,
                sync_file_name="test_sync.pkl",
                processing_date=datetime(2024, 1, 1, 12, 0, 0),
                sync_duration=120.0,
                pass_number=1,  # Should fail - must be None for non-walk
                speed=None,
            )
        assert "pass_number must be None for non-walk maneuvers" in str(exc_info.value)

    @pytest.mark.parametrize("maneuver", ["fe", "sts"])
    def test_non_walk_maneuver_requires_none_speed(self, maneuver):
        """Test that non-walk maneuvers with non-None speed fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            Synchronization(
                study="AOA",
                study_id=1011,
                linked_biomechanics=True,
                biomechanics_file="test.xlsx",
                biomechanics_type="Motion Analysis",
                biomechanics_sync_method="stomp",
                biomechanics_sample_rate=100.0,
                audio_file_name="test.bin",
                device_serial="TEST123",
                firmware_version=1,
                file_time=datetime(2024, 1, 1, 10, 0, 0),
                file_size_mb=100.0,
                recording_date=datetime(2024, 1, 1),
                recording_time=datetime(2024, 1, 1, 10, 0, 0),
                knee="left",
                maneuver=maneuver,
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
                rms_time=5.0,
                onset_time=5.0,
                freq_time=5.0,
                sync_file_name="test_sync.pkl",
                processing_date=datetime(2024, 1, 1, 12, 0, 0),
                sync_duration=120.0,
                pass_number=None,
                speed="normal",  # Should fail - must be None for non-walk
            )
        assert "speed must be None for non-walk maneuvers" in str(exc_info.value)

    @pytest.mark.parametrize("maneuver", ["fe", "sts"])
    def test_non_walk_maneuver_with_none_values_passes(self, maneuver):
        """Test that non-walk maneuvers with None speed and pass_number pass validation."""
        sync = Synchronization(
            study="AOA",
            study_id=1011,
            linked_biomechanics=True,
            biomechanics_file="test.xlsx",
            biomechanics_type="Motion Analysis",
            biomechanics_sync_method="stomp",
            biomechanics_sample_rate=100.0,
            audio_file_name="test.bin",
            device_serial="TEST123",
            firmware_version=1,
            file_time=datetime(2024, 1, 1, 10, 0, 0),
            file_size_mb=100.0,
            recording_date=datetime(2024, 1, 1),
            recording_time=datetime(2024, 1, 1, 10, 0, 0),
            knee="left",
            maneuver=maneuver,
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
            rms_time=5.0,
            onset_time=5.0,
            freq_time=5.0,
            sync_file_name="test_sync.pkl",
            processing_date=datetime(2024, 1, 1, 12, 0, 0),
            sync_duration=120.0,
            pass_number=None,
            speed=None,
        )
        assert sync.pass_number is None
        assert sync.speed is None


class TestSyncDataPassNumberAndSpeed:
    """Test SyncData properly stores pass_number and speed."""

    def test_syncdata_with_pass_number_and_speed(self):
        """Test SyncData stores pass_number and speed."""
        output_path = Path("/tmp/synced.pkl")
        df = pd.DataFrame({"col": [1, 2, 3]})
        stomp_times = (1.0, 2.0, 3.0, {})

        sync_data = SyncData(
            output_path=output_path,
            df=df,
            stomp_times=stomp_times,
            pass_number=1,
            speed="normal",
        )

        assert sync_data.pass_number == 1
        assert sync_data.speed == "normal"

    def test_syncdata_with_none_pass_number_and_speed(self):
        """Test SyncData can store None for pass_number and speed."""
        output_path = Path("/tmp/synced.pkl")
        df = pd.DataFrame({"col": [1, 2, 3]})
        stomp_times = (1.0, 2.0, 3.0, {})

        sync_data = SyncData(
            output_path=output_path,
            df=df,
            stomp_times=stomp_times,
            pass_number=None,
            speed=None,
        )

        assert sync_data.pass_number is None
        assert sync_data.speed is None

    def test_syncdata_defaults_to_none(self):
        """Test SyncData defaults to None for pass_number and speed."""
        output_path = Path("/tmp/synced.pkl")
        df = pd.DataFrame({"col": [1, 2, 3]})
        stomp_times = (1.0, 2.0, 3.0, {})

        sync_data = SyncData(
            output_path=output_path,
            df=df,
            stomp_times=stomp_times,
        )

        assert sync_data.pass_number is None
        assert sync_data.speed is None
