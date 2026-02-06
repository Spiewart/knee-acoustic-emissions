"""QC metadata reporting tests."""

from datetime import datetime

import pandas as pd
import pytest

from src.metadata import AudioProcessing
from src.reports.report_generator import ReportGenerator


def test_qc_fields_present_in_audio_sheet(
    db_session,
    repository,
    audio_processing_factory,
    tmp_path,
):
    audio = audio_processing_factory(
        study="AOA",
        study_id=7001,
        audio_file_name="AOA7001_audio",
        qc_signal_dropout=True,
        qc_continuous_artifact=True,
        qc_fail_segments=[(0.1, 0.2)],
    )
    audio_record = repository.save_audio_processing(audio)

    report = ReportGenerator(db_session)
    output_path = report.save_to_excel(
        tmp_path / "qc_audio.xlsx",
        participant_id=audio_record.participant_id,
        maneuver="walk",
        knee="left",
    )

    audio_sheet = pd.read_excel(output_path, sheet_name="Audio")
    assert "QC Signal Dropout" in audio_sheet.columns
    assert "QC Continuous Artifact" in audio_sheet.columns

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
            qc_continuous_artifact=False,
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
            qc_continuous_artifact=False,
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
            "qc_continuous_artifact": False,
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
            "qc_continuous_artifact": False,
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

class TestDetailedQCFieldsExport:
    """Test that detailed QC fields are exported and persist through Excel cycles."""

    def _create_audio_with_qc_fields(self) -> AudioProcessing:
        """Helper to create AudioProcessing with all QC fields populated."""
        return AudioProcessing(
            study="AOA",
            study_id=1016,
            audio_file_name="test.bin",
            device_serial="123",
            firmware_version=1,
            file_time=datetime.now(),
            file_size_mb=10.0,
            recording_date=datetime.now(),
            recording_time=datetime.now(),
            knee="right",
            maneuver="walk",
            num_channels=4,
            mic_1_position="IPM",
            mic_2_position="IPL",
            mic_3_position="SPM",
            mic_4_position="SPL",
            processing_date=datetime.now(),
            # QC Fail Segments
            qc_fail_segments=[(1.0, 2.5)],
            qc_fail_segments_ch1=[(1.0, 2.5)],
            qc_fail_segments_ch2=[],
            qc_fail_segments_ch3=[(3.0, 4.0)],
            qc_fail_segments_ch4=[],
            # Signal Dropout QC
            qc_signal_dropout=True,
            qc_signal_dropout_segments=[(0.5, 1.5), (8.0, 9.0)],
            qc_signal_dropout_ch1=True,
            qc_signal_dropout_segments_ch1=[(0.5, 1.5)],
            qc_signal_dropout_ch2=False,
            qc_signal_dropout_segments_ch2=[],
            qc_signal_dropout_ch3=True,
            qc_signal_dropout_segments_ch3=[(8.0, 9.0)],
            qc_signal_dropout_ch4=False,
            qc_signal_dropout_segments_ch4=[],
            # Artifact QC
            qc_continuous_artifact=True,
            qc_continuous_artifact_type=["Intermittent"],
            qc_continuous_artifact_segments=[(2.0, 3.0)],
            qc_continuous_artifact_ch1=True,
            qc_continuous_artifact_type_ch1=["Continuous"],
            qc_continuous_artifact_segments_ch1=[(2.0, 3.0)],
            qc_continuous_artifact_ch2=False,
            qc_continuous_artifact_type_ch2=None,
            qc_continuous_artifact_segments_ch2=[],
            qc_continuous_artifact_ch3=False,
            qc_continuous_artifact_type_ch3=None,
            qc_continuous_artifact_segments_ch3=[],
            qc_continuous_artifact_ch4=True,
            qc_continuous_artifact_type_ch4=["Intermittent"],
            qc_continuous_artifact_segments_ch4=[(5.0, 6.0)],
        )

    def test_qc_fields_in_to_dict(self):
        """Test that all QC fields are included in to_dict() export."""
        audio = self._create_audio_with_qc_fields()
        result = audio.to_dict()

        # QC Fail Segments
        assert "QC Fail Segments" in result
        assert "QC Fail Segments Ch1" in result
        assert "QC Fail Segments Ch2" in result
        assert "QC Fail Segments Ch3" in result
        assert "QC Fail Segments Ch4" in result

        # Signal Dropout QC
        assert "QC Signal Dropout" in result
        assert "QC Signal Dropout Segments" in result
        assert "QC Signal Dropout Ch1" in result
        assert "QC Signal Dropout Segments Ch1" in result
        assert "QC Signal Dropout Ch2" in result
        assert "QC Signal Dropout Segments Ch2" in result
        assert "QC Signal Dropout Ch3" in result
        assert "QC Signal Dropout Segments Ch3" in result
        assert "QC Signal Dropout Ch4" in result
        assert "QC Signal Dropout Segments Ch4" in result

        # Artifact QC
        assert "QC Continuous Artifact" in result
        assert "QC Continuous Artifact Type" in result
        assert "QC Continuous Artifact Segments" in result
        assert "QC Continuous Artifact Ch1" in result
        assert "QC Continuous Artifact Type Ch1" in result
        assert "QC Continuous Artifact Segments Ch1" in result
        assert "QC Continuous Artifact Ch2" in result
        assert "QC Continuous Artifact Type Ch2" in result
        assert "QC Continuous Artifact Segments Ch2" in result
        assert "QC Continuous Artifact Ch3" in result
        assert "QC Continuous Artifact Type Ch3" in result
        assert "QC Continuous Artifact Segments Ch3" in result
        assert "QC Continuous Artifact Ch4" in result
        assert "QC Continuous Artifact Type Ch4" in result
        assert "QC Continuous Artifact Segments Ch4" in result

    def test_qc_field_values_in_to_dict(self):
        """Test that QC field values are correctly exported."""
        audio = self._create_audio_with_qc_fields()
        result = audio.to_dict()

        # QC Fail Segments
        assert result["QC Fail Segments"] == [(1.0, 2.5)]
        assert result["QC Fail Segments Ch1"] == [(1.0, 2.5)]
        assert result["QC Fail Segments Ch2"] == []
        assert result["QC Fail Segments Ch3"] == [(3.0, 4.0)]
        assert result["QC Fail Segments Ch4"] == []

        # Signal Dropout QC
        assert result["QC Signal Dropout"] is True
        assert result["QC Signal Dropout Ch1"] is True
        assert result["QC Signal Dropout Ch2"] is False
        assert result["QC Signal Dropout Ch3"] is True
        assert result["QC Signal Dropout Ch4"] is False

        # Artifact QC
        assert result["QC Continuous Artifact"] is True
        assert result["QC Continuous Artifact Type"] == ["Intermittent"]
        assert result["QC Continuous Artifact Ch1"] is True
        assert result["QC Continuous Artifact Type Ch1"] == ["Continuous"]
        assert result["QC Continuous Artifact Ch2"] is False
        assert result["QC Continuous Artifact Type Ch2"] is None

    def test_qc_fields_persist_through_excel_roundtrip(self, tmp_path):
        """Test that QC fields survive Excel save/load cycle."""
        audio = self._create_audio_with_qc_fields()

        # Save to Excel
        excel_path = tmp_path / "test_qc_export.xlsx"
        df = pd.DataFrame([audio.to_dict()])
        df.to_excel(excel_path, sheet_name="Audio", index=False)

        # Load from Excel
        df_loaded = pd.read_excel(excel_path, sheet_name="Audio")

        # Verify QC fields are present in loaded data
        assert "QC Fail Segments" in df_loaded.columns
        assert "QC Signal Dropout" in df_loaded.columns
        assert "QC Continuous Artifact" in df_loaded.columns
        assert "QC Continuous Artifact Type" in df_loaded.columns

        # Verify boolean values
        assert df_loaded["QC Signal Dropout"].iloc[0] is True or df_loaded["QC Signal Dropout"].iloc[0] == 1
        assert df_loaded["QC Continuous Artifact"].iloc[0] is True or df_loaded["QC Continuous Artifact"].iloc[0] == 1

    def test_biomechanics_import_id_in_to_dict(self):
        """Test that biomechanics_import_id is exported in to_dict()."""
        # Test with biomechanics_import_id set
        audio = AudioProcessing(
            study="AOA",
            study_id=1016,
            audio_file_name="test.bin",
            device_serial="123",
            firmware_version=1,
            file_time=datetime.now(),
            file_size_mb=10.0,
            recording_date=datetime.now(),
            recording_time=datetime.now(),
            knee="right",
            maneuver="walk",
            num_channels=4,
            mic_1_position="IPM",
            mic_2_position="IPL",
            mic_3_position="SPM",
            mic_4_position="SPL",
            processing_date=datetime.now(),
            biomechanics_import_id=42,  # FK to biomechanics record
        )
        result = audio.to_dict()

        # Verify biomechanics_import_id is in the export
        assert "Biomechanics Import ID" in result
        assert result["Biomechanics Import ID"] == 42

    def test_biomechanics_import_id_none_in_to_dict(self):
        """Test that biomechanics_import_id=None is exported correctly."""
        audio = AudioProcessing(
            study="AOA",
            study_id=1016,
            audio_file_name="test.bin",
            device_serial="123",
            firmware_version=1,
            file_time=datetime.now(),
            file_size_mb=10.0,
            recording_date=datetime.now(),
            recording_time=datetime.now(),
            knee="right",
            maneuver="walk",
            num_channels=4,
            mic_1_position="IPM",
            mic_2_position="IPL",
            mic_3_position="SPM",
            mic_4_position="SPL",
            processing_date=datetime.now(),
            biomechanics_import_id=None,  # No biomechanics linked
        )
        result = audio.to_dict()

        # Verify biomechanics_import_id is in the export as None
        assert "Biomechanics Import ID" in result
        assert result["Biomechanics Import ID"] is None

    def test_biomechanics_import_id_persists_through_excel(self, tmp_path):
        """Test that biomechanics_import_id persists through Excel save/load."""
        audio = AudioProcessing(
            study="AOA",
            study_id=1016,
            audio_file_name="test_with_biomech.bin",
            device_serial="456",
            firmware_version=1,
            file_time=datetime.now(),
            file_size_mb=15.0,
            recording_date=datetime.now(),
            recording_time=datetime.now(),
            knee="left",
            maneuver="sts",
            num_channels=4,
            mic_1_position="IPM",
            mic_2_position="IPL",
            mic_3_position="SPM",
            mic_4_position="SPL",
            processing_date=datetime.now(),
            biomechanics_import_id=123,  # FK to biomechanics record
        )

        # Save to Excel
        excel_path = tmp_path / "test_biomech_id.xlsx"
        df = pd.DataFrame([audio.to_dict()])
        df.to_excel(excel_path, sheet_name="Audio", index=False)

        # Load from Excel
        df_loaded = pd.read_excel(excel_path, sheet_name="Audio")

        # Verify biomechanics_import_id is present and has correct value
        assert "Biomechanics Import ID" in df_loaded.columns
        assert df_loaded["Biomechanics Import ID"].iloc[0] == 123