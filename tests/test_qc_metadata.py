"""Tests for QC metadata validation and Excel persistence.

Tests that QC_not_passed boolean fields are correctly auto-populated from
qc_fail_segments and persist through Excel save/load cycles.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.metadata import AudioProcessing


class TestQCNotPassedValidator:
    """Test the populate_qc_not_passed_fields validator."""

    def _create_audio_processing(
        self,
        qc_fail_segments=None,
        qc_fail_segments_ch1=None,
        qc_fail_segments_ch2=None,
        qc_fail_segments_ch3=None,
        qc_fail_segments_ch4=None,
    ) -> AudioProcessing:
        """Helper to create AudioProcessing with specified fail segments."""
        return AudioProcessing(
            study="AOA",
            study_id=1013,
            linked_biomechanics=False,
            audio_file_name="test.bin",
            device_serial="123",
            firmware_version=1,
            file_time=datetime.now(),
            file_size_mb=10.0,
            recording_date=datetime.now(),
            recording_time=datetime.now(),
            knee="left",
            maneuver="fe",
            num_channels=4,
            mic_1_position="IPM",
            mic_2_position="IPL",
            mic_3_position="SPM",
            mic_4_position="SPL",
            processing_date=datetime.now(),
            qc_fail_segments=qc_fail_segments or [],
            qc_fail_segments_ch1=qc_fail_segments_ch1 or [],
            qc_fail_segments_ch2=qc_fail_segments_ch2 or [],
            qc_fail_segments_ch3=qc_fail_segments_ch3 or [],
            qc_fail_segments_ch4=qc_fail_segments_ch4 or [],
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

    def test_qc_not_passed_false_when_no_fail_segments(self):
        """Test that qc_not_passed is False when all qc_fail_segments are empty."""
        audio = self._create_audio_processing()
        assert audio.qc_not_passed is False
        assert audio.qc_not_passed_mic_1 is False
        assert audio.qc_not_passed_mic_2 is False
        assert audio.qc_not_passed_mic_3 is False
        assert audio.qc_not_passed_mic_4 is False

    def test_qc_not_passed_true_when_overall_fail_segments_exist(self):
        """Test that qc_not_passed is True when qc_fail_segments has intervals."""
        audio = self._create_audio_processing(
            qc_fail_segments=[(1.0, 2.5), (5.0, 6.0)]
        )
        assert audio.qc_not_passed is True
        # Per-mic should still be False since they're empty
        assert audio.qc_not_passed_mic_1 is False
        assert audio.qc_not_passed_mic_2 is False
        assert audio.qc_not_passed_mic_3 is False
        assert audio.qc_not_passed_mic_4 is False

    def test_qc_not_passed_mic_1_true_when_ch1_fail_segments_exist(self):
        """Test that qc_not_passed_mic_1 is True when qc_fail_segments_ch1 has intervals."""
        audio = self._create_audio_processing(
            qc_fail_segments_ch1=[(1.0, 2.5)]
        )
        assert audio.qc_not_passed is False
        assert audio.qc_not_passed_mic_1 is True
        assert audio.qc_not_passed_mic_2 is False
        assert audio.qc_not_passed_mic_3 is False
        assert audio.qc_not_passed_mic_4 is False

    def test_qc_not_passed_mic_3_true_when_ch3_fail_segments_exist(self):
        """Test that qc_not_passed_mic_3 is True when qc_fail_segments_ch3 has intervals."""
        audio = self._create_audio_processing(
            qc_fail_segments_ch3=[(2.0, 3.0), (7.0, 8.5)]
        )
        assert audio.qc_not_passed is False
        assert audio.qc_not_passed_mic_1 is False
        assert audio.qc_not_passed_mic_2 is False
        assert audio.qc_not_passed_mic_3 is True
        assert audio.qc_not_passed_mic_4 is False

    def test_qc_not_passed_all_channels_with_fail_segments(self):
        """Test that all per-mic fields reflect their respective fail segments."""
        audio = self._create_audio_processing(
            qc_fail_segments=[(1.0, 2.5)],
            qc_fail_segments_ch1=[(1.0, 2.5)],
            qc_fail_segments_ch2=[(2.0, 3.0)],
            qc_fail_segments_ch3=[],
            qc_fail_segments_ch4=[(5.0, 6.0)],
        )
        assert audio.qc_not_passed is True
        assert audio.qc_not_passed_mic_1 is True
        assert audio.qc_not_passed_mic_2 is True
        assert audio.qc_not_passed_mic_3 is False
        assert audio.qc_not_passed_mic_4 is True

    def test_qc_not_passed_to_dict_includes_booleans(self):
        """Test that to_dict() exports qc_not_passed fields as booleans."""
        audio = self._create_audio_processing(
            qc_fail_segments=[(1.0, 2.5)],
            qc_fail_segments_ch1=[(1.0, 2.5)],
            qc_fail_segments_ch2=[],
            qc_fail_segments_ch3=[],
            qc_fail_segments_ch4=[],
        )
        result = audio.to_dict()

        assert "QC_not_passed" in result
        assert "QC_not_passed_mic_1" in result
        assert "QC_not_passed_mic_2" in result
        assert "QC_not_passed_mic_3" in result
        assert "QC_not_passed_mic_4" in result

        assert result["QC_not_passed"] is True
        assert result["QC_not_passed_mic_1"] is True
        assert result["QC_not_passed_mic_2"] is False
        assert result["QC_not_passed_mic_3"] is False
        assert result["QC_not_passed_mic_4"] is False


class TestQCMetadataExcelIntegration:
    """Test QC metadata persistence through Excel save/load cycles."""

    @pytest.fixture
    def temp_excel_path(self, tmp_path):
        """Create a temporary Excel file path."""
        return tmp_path / "test_qc_metadata.xlsx"

    def _create_audio_processing(
        self,
        qc_fail_segments=None,
        qc_fail_segments_ch1=None,
        qc_fail_segments_ch2=None,
        qc_fail_segments_ch3=None,
        qc_fail_segments_ch4=None,
    ) -> AudioProcessing:
        """Helper to create AudioProcessing with specified fail segments."""
        return AudioProcessing(
            study="AOA",
            study_id=1013,
            linked_biomechanics=False,
            audio_file_name="test.bin",
            device_serial="123",
            firmware_version=1,
            file_time=datetime.now(),
            file_size_mb=10.0,
            recording_date=datetime.now(),
            recording_time=datetime.now(),
            knee="left",
            maneuver="fe",
            num_channels=4,
            mic_1_position="IPM",
            mic_2_position="IPL",
            mic_3_position="SPM",
            mic_4_position="SPL",
            processing_date=datetime.now(),
            qc_fail_segments=qc_fail_segments or [],
            qc_fail_segments_ch1=qc_fail_segments_ch1 or [],
            qc_fail_segments_ch2=qc_fail_segments_ch2 or [],
            qc_fail_segments_ch3=qc_fail_segments_ch3 or [],
            qc_fail_segments_ch4=qc_fail_segments_ch4 or [],
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

    def test_qc_not_passed_persists_through_excel_save_load(self, temp_excel_path):
        """Test that qc_not_passed boolean values persist through Excel save/load."""
        # Create audio metadata with some fail segments
        audio = self._create_audio_processing(
            qc_fail_segments=[(1.0, 2.5)],
            qc_fail_segments_ch1=[(1.0, 2.5)],
            qc_fail_segments_ch2=[],
            qc_fail_segments_ch3=[(5.0, 6.0)],
            qc_fail_segments_ch4=[],
        )

        # Save to Excel
        with pd.ExcelWriter(temp_excel_path, engine="openpyxl") as writer:
            df = pd.DataFrame([audio.to_dict()])
            df.to_excel(writer, sheet_name="Audio", index=False)

        # Load from Excel
        df_loaded = pd.read_excel(temp_excel_path, sheet_name="Audio")

        # Verify boolean columns are loaded correctly (pandas returns np.bool_)
        assert df_loaded.loc[0, "QC_not_passed"] == True
        assert df_loaded.loc[0, "QC_not_passed_mic_1"] == True
        assert df_loaded.loc[0, "QC_not_passed_mic_2"] == False
        assert df_loaded.loc[0, "QC_not_passed_mic_3"] == True
        assert df_loaded.loc[0, "QC_not_passed_mic_4"] == False

    def test_qc_not_passed_false_values_persist_through_excel(self, temp_excel_path):
        """Test that False values for qc_not_passed persist through Excel save/load."""
        # Create audio metadata with no fail segments
        audio = self._create_audio_processing()

        # Save to Excel
        with pd.ExcelWriter(temp_excel_path, engine="openpyxl") as writer:
            df = pd.DataFrame([audio.to_dict()])
            df.to_excel(writer, sheet_name="Audio", index=False)

        # Load from Excel
        df_loaded = pd.read_excel(temp_excel_path, sheet_name="Audio")

        # Verify all boolean columns are False (pandas returns np.bool_)
        assert df_loaded.loc[0, "QC_not_passed"] == False
        assert df_loaded.loc[0, "QC_not_passed_mic_1"] == False
        assert df_loaded.loc[0, "QC_not_passed_mic_2"] == False
        assert df_loaded.loc[0, "QC_not_passed_mic_3"] == False
        assert df_loaded.loc[0, "QC_not_passed_mic_4"] == False

    def test_multiple_audio_records_with_different_qc_status(self, temp_excel_path):
        """Test multiple audio records with different QC statuses persist correctly."""
        # Create multiple audio metadata objects
        audio1 = self._create_audio_processing()  # All False
        audio2 = self._create_audio_processing(
            qc_fail_segments=[(1.0, 2.5)],
            qc_fail_segments_ch1=[(1.0, 2.5)],
        )  # Some True
        audio3 = self._create_audio_processing(
            qc_fail_segments=[(2.0, 3.0)],
            qc_fail_segments_ch2=[(2.0, 3.0)],
            qc_fail_segments_ch3=[(5.0, 6.0)],
            qc_fail_segments_ch4=[(7.0, 8.0)],
        )  # Most True

        # Save to Excel
        with pd.ExcelWriter(temp_excel_path, engine="openpyxl") as writer:
            df = pd.DataFrame([
                audio1.to_dict(),
                audio2.to_dict(),
                audio3.to_dict(),
            ])
            df.to_excel(writer, sheet_name="Audio", index=False)

        # Load from Excel
        df_loaded = pd.read_excel(temp_excel_path, sheet_name="Audio")

        # Verify first record (all False)
        assert df_loaded.loc[0, "QC_not_passed"] == False
        assert df_loaded.loc[0, "QC_not_passed_mic_1"] == False

        # Verify second record (some True)
        assert df_loaded.loc[1, "QC_not_passed"] == True
        assert df_loaded.loc[1, "QC_not_passed_mic_1"] == True
        assert df_loaded.loc[1, "QC_not_passed_mic_2"] == False

        # Verify third record (most True)
        assert df_loaded.loc[2, "QC_not_passed"] == True
        assert df_loaded.loc[2, "QC_not_passed_mic_2"] == True
        assert df_loaded.loc[2, "QC_not_passed_mic_3"] == True
        assert df_loaded.loc[2, "QC_not_passed_mic_4"] == True


class TestQCStatusIntegrationWithProcessing:
    """Integration tests for QC status with mocked processing pipeline."""

    def test_qc_not_passed_set_during_processing_mocked(self):
        """Test that qc_not_passed is correctly set during mocked processing."""
        # Mock the audio QC result with some fail segments
        mock_qc_result = {
            "qc_fail_segments": [(1.0, 2.5), (5.0, 6.0)],
            "qc_fail_segments_ch1": [(1.0, 2.5)],
            "qc_fail_segments_ch2": [],
            "qc_fail_segments_ch3": [(5.0, 6.0)],
            "qc_fail_segments_ch4": [],
            "qc_signal_dropout": False,
            "qc_signal_dropout_segments": [],
            "qc_signal_dropout_ch1": False,
            "qc_signal_dropout_segments_ch1": [],
            "qc_signal_dropout_ch2": False,
            "qc_signal_dropout_segments_ch2": [],
            "qc_signal_dropout_ch3": False,
            "qc_signal_dropout_segments_ch3": [],
            "qc_signal_dropout_ch4": False,
            "qc_signal_dropout_segments_ch4": [],
            "qc_artifact": False,
            "qc_artifact_segments": [],
            "qc_artifact_ch1": False,
            "qc_artifact_segments_ch1": [],
            "qc_artifact_ch2": False,
            "qc_artifact_segments_ch2": [],
            "qc_artifact_ch3": False,
            "qc_artifact_segments_ch3": [],
            "qc_artifact_ch4": False,
            "qc_artifact_segments_ch4": [],
        }

        # Create AudioProcessing object as if from processing
        audio = AudioProcessing(
            study="AOA",
            study_id=1013,
            linked_biomechanics=False,
            audio_file_name="test.bin",
            device_serial="123",
            firmware_version=1,
            file_time=datetime.now(),
            file_size_mb=10.0,
            recording_date=datetime.now(),
            recording_time=datetime.now(),
            knee="left",
            maneuver="fe",
            num_channels=4,
            mic_1_position="IPM",
            mic_2_position="IPL",
            mic_3_position="SPM",
            mic_4_position="SPL",
            processing_date=datetime.now(),
            **mock_qc_result,
        )

        # Verify QC status is correctly auto-populated
        assert audio.qc_not_passed is True  # Has overall fail segments
        assert audio.qc_not_passed_mic_1 is True  # Has ch1 fail segments
        assert audio.qc_not_passed_mic_2 is False  # No ch2 fail segments
        assert audio.qc_not_passed_mic_3 is True  # Has ch3 fail segments
        assert audio.qc_not_passed_mic_4 is False  # No ch4 fail segments

    def test_qc_not_passed_false_for_clean_audio(self):
        """Test that qc_not_passed is False for clean audio (no fail segments)."""
        # Mock clean audio QC result
        mock_qc_result = {
            "qc_fail_segments": [],
            "qc_fail_segments_ch1": [],
            "qc_fail_segments_ch2": [],
            "qc_fail_segments_ch3": [],
            "qc_fail_segments_ch4": [],
            "qc_signal_dropout": False,
            "qc_signal_dropout_segments": [],
            "qc_signal_dropout_ch1": False,
            "qc_signal_dropout_segments_ch1": [],
            "qc_signal_dropout_ch2": False,
            "qc_signal_dropout_segments_ch2": [],
            "qc_signal_dropout_ch3": False,
            "qc_signal_dropout_segments_ch3": [],
            "qc_signal_dropout_ch4": False,
            "qc_signal_dropout_segments_ch4": [],
            "qc_artifact": False,
            "qc_artifact_segments": [],
            "qc_artifact_ch1": False,
            "qc_artifact_segments_ch1": [],
            "qc_artifact_ch2": False,
            "qc_artifact_segments_ch2": [],
            "qc_artifact_ch3": False,
            "qc_artifact_segments_ch3": [],
            "qc_artifact_ch4": False,
            "qc_artifact_segments_ch4": [],
        }

        audio = AudioProcessing(
            study="AOA",
            study_id=1013,
            linked_biomechanics=False,
            audio_file_name="test.bin",
            device_serial="123",
            firmware_version=1,
            file_time=datetime.now(),
            file_size_mb=10.0,
            recording_date=datetime.now(),
            recording_time=datetime.now(),
            knee="left",
            maneuver="fe",
            num_channels=4,
            mic_1_position="IPM",
            mic_2_position="IPL",
            mic_3_position="SPM",
            mic_4_position="SPL",
            processing_date=datetime.now(),
            **mock_qc_result,
        )

        # Verify all QC status fields are False
        assert audio.qc_not_passed is False
        assert audio.qc_not_passed_mic_1 is False
        assert audio.qc_not_passed_mic_2 is False
        assert audio.qc_not_passed_mic_3 is False
        assert audio.qc_not_passed_mic_4 is False
