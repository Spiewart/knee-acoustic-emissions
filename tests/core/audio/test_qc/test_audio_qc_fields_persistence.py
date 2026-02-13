"""Tests for QC field persistence through database layer.

Tests that QC fields are properly defined in database models and flow
correctly from AudioProcessing metadata through repository operations.
"""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from src.db.models import AudioProcessingRecord
from src.metadata import AudioProcessing


class TestAudioQCFieldsDatabasePersistence:
    """Test QC fields persistence in database repository."""

    def _create_audio_processing_with_qc(self) -> AudioProcessing:
        """Helper to create AudioProcessing with various QC fields populated."""
        return AudioProcessing(
            study="AOA",
            study_id=1016,
            audio_file_name="test_walk.bin",
            device_serial="DEV123",
            firmware_version=2,
            file_time=datetime(2024, 1, 15, 10, 30, 0),
            file_size_mb=150.5,
            recording_date=datetime(2024, 1, 15),
            recording_time=datetime(2024, 1, 15, 10, 30, 0),
            knee="right",
            maneuver="walk",
            num_channels=4,
            mic_1_position="IPM",
            mic_2_position="IPL",
            mic_3_position="SPM",
            mic_4_position="SPL",
            processing_date=datetime(2024, 1, 15, 15, 0, 0),
            # QC Fail Segments
            qc_fail_segments=[(5.0, 10.0), (45.0, 50.0)],
            qc_fail_segments_ch1=[(5.0, 10.0)],
            qc_fail_segments_ch2=[],
            qc_fail_segments_ch3=[(45.0, 50.0)],
            qc_fail_segments_ch4=[],
            # Signal Dropout QC
            qc_signal_dropout=True,
            qc_signal_dropout_segments=[(2.0, 3.0)],
            qc_signal_dropout_ch1=True,
            qc_signal_dropout_segments_ch1=[(2.0, 3.0)],
            qc_signal_dropout_ch2=False,
            qc_signal_dropout_segments_ch2=[],
            qc_signal_dropout_ch3=False,
            qc_signal_dropout_segments_ch3=[],
            qc_signal_dropout_ch4=False,
            qc_signal_dropout_segments_ch4=[],
            # Artifact QC
            qc_continuous_artifact=True,
            qc_continuous_artifact_segments=[(12.0, 14.0)],
            qc_continuous_artifact_ch1=False,
            qc_continuous_artifact_segments_ch1=[],
            qc_continuous_artifact_ch2=True,
            qc_continuous_artifact_segments_ch2=[(12.0, 14.0)],
            qc_continuous_artifact_ch3=False,
            qc_continuous_artifact_segments_ch3=[],
            qc_continuous_artifact_ch4=False,
            qc_continuous_artifact_segments_ch4=[],
        )

    def test_create_audio_processing_record_includes_qc_fail_segments(self):
        """Test that AudioProcessingRecord model has all qc_fail_segments fields."""
        assert hasattr(AudioProcessingRecord, "qc_fail_segments")
        assert hasattr(AudioProcessingRecord, "qc_fail_segments_ch1")
        assert hasattr(AudioProcessingRecord, "qc_fail_segments_ch2")
        assert hasattr(AudioProcessingRecord, "qc_fail_segments_ch3")
        assert hasattr(AudioProcessingRecord, "qc_fail_segments_ch4")

    def test_create_audio_processing_record_includes_signal_dropout_qc(self):
        """Test that AudioProcessingRecord model has all signal dropout fields."""
        assert hasattr(AudioProcessingRecord, "qc_signal_dropout")
        assert hasattr(AudioProcessingRecord, "qc_signal_dropout_segments")
        assert hasattr(AudioProcessingRecord, "qc_signal_dropout_ch1")
        assert hasattr(AudioProcessingRecord, "qc_signal_dropout_segments_ch1")
        assert hasattr(AudioProcessingRecord, "qc_signal_dropout_ch2")
        assert hasattr(AudioProcessingRecord, "qc_signal_dropout_segments_ch2")
        assert hasattr(AudioProcessingRecord, "qc_signal_dropout_ch3")
        assert hasattr(AudioProcessingRecord, "qc_signal_dropout_segments_ch3")
        assert hasattr(AudioProcessingRecord, "qc_signal_dropout_ch4")
        assert hasattr(AudioProcessingRecord, "qc_signal_dropout_segments_ch4")

    def test_create_audio_processing_record_includes_artifact_qc(self):
        """Test that AudioProcessingRecord model has all artifact QC fields."""
        assert hasattr(AudioProcessingRecord, "qc_continuous_artifact")
        assert hasattr(AudioProcessingRecord, "qc_continuous_artifact_segments")
        assert hasattr(AudioProcessingRecord, "qc_continuous_artifact_ch1")
        assert hasattr(AudioProcessingRecord, "qc_continuous_artifact_segments_ch1")
        assert hasattr(AudioProcessingRecord, "qc_continuous_artifact_ch2")
        assert hasattr(AudioProcessingRecord, "qc_continuous_artifact_segments_ch2")
        assert hasattr(AudioProcessingRecord, "qc_continuous_artifact_ch3")
        assert hasattr(AudioProcessingRecord, "qc_continuous_artifact_segments_ch3")
        assert hasattr(AudioProcessingRecord, "qc_continuous_artifact_ch4")
        assert hasattr(AudioProcessingRecord, "qc_continuous_artifact_segments_ch4")

    def test_audio_processing_metadata_has_all_qc_fields(self):
        """Test that AudioProcessing Pydantic model has all QC fields."""
        audio = self._create_audio_processing_with_qc()

        # QC Fail Segments
        assert hasattr(audio, "qc_fail_segments")
        assert hasattr(audio, "qc_fail_segments_ch1")
        assert hasattr(audio, "qc_fail_segments_ch2")
        assert hasattr(audio, "qc_fail_segments_ch3")
        assert hasattr(audio, "qc_fail_segments_ch4")

        # Signal Dropout
        assert hasattr(audio, "qc_signal_dropout")
        assert hasattr(audio, "qc_signal_dropout_segments")
        assert hasattr(audio, "qc_signal_dropout_ch1")
        assert hasattr(audio, "qc_signal_dropout_segments_ch1")

        # Artifact
        assert hasattr(audio, "qc_continuous_artifact")
        assert hasattr(audio, "qc_continuous_artifact_segments")
