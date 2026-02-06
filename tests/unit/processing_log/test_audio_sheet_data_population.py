"""Test Audio sheet data population from audio file names and QC data."""

from datetime import datetime

import pandas as pd
import pytest

from src.reports.report_generator import ReportGenerator


class TestAudioSheetDataPopulation:
    """Test that Audio sheet correctly populates all required fields."""

    def test_audio_sheet_device_serial_parsed_from_filename(
        self,
        db_session,
        repository,
        audio_processing_factory,
        biomechanics_import_factory,
        tmp_path,
    ):
        """Test that Device Serial is populated from audio filename, not defaulted."""
        # Create audio with filename that has device serial
        recording_date = datetime(2024, 3, 15)
        audio = audio_processing_factory(
            study="AOA",
            study_id=6001,
            knee="left",
            maneuver="walk",
            audio_file_name="HP_W11.2-42-20240315_143022.bin",  # Serial is 42
            device_serial="42",  # Should match filename
            recording_date=recording_date,
            recording_time=datetime(2024, 3, 15, 14, 30, 22),  # Must match date
        )
        audio_record = repository.save_audio_processing(audio)
        db_session.commit()

        # Generate report
        report = ReportGenerator(db_session)
        output_path = report.save_to_excel(
            tmp_path / "test_audio.xlsx",
            participant_id=audio_record.participant_id,
            maneuver="walk",
            knee="left",
        )

        # Read Audio sheet
        audio_sheet = pd.read_excel(output_path, sheet_name="Audio")

        # Device Serial should NOT be "UNKNOWN"
        device_serial = audio_sheet["Device Serial"].iloc[0]
        assert device_serial != "UNKNOWN", (
            f"Device Serial should be parsed from filename, got: {device_serial}"
        )
        assert str(device_serial) == "42", (
            f"Device Serial should be '42', got: {device_serial}"
        )

    def test_audio_sheet_recording_date_not_datetime_now(
        self,
        db_session,
        repository,
        audio_processing_factory,
        tmp_path,
    ):
        """Test that Recording Datetime is from filename, not defaulted to datetime.now()."""
        # Create audio with specific date
        recording_date = datetime(2024, 3, 15)
        audio = audio_processing_factory(
            study="AOA",
            study_id=6002,
            knee="right",
            maneuver="sts",
            audio_file_name="HP_W12.1-5-20240315_143022.bin",
            device_serial="5",
            recording_date=recording_date,
            recording_time=datetime(2024, 3, 15, 14, 30, 22),  # Must match date
        )
        audio_record = repository.save_audio_processing(audio)
        db_session.commit()

        # Generate report
        report = ReportGenerator(db_session)
        output_path = report.save_to_excel(
            tmp_path / "test_date.xlsx",
            participant_id=audio_record.participant_id,
            maneuver="sts",
            knee="right",
        )

        # Read Audio sheet
        audio_sheet = pd.read_excel(output_path, sheet_name="Audio")

        # Recording Datetime should match the expected date
        stored_date = pd.to_datetime(audio_sheet["Recording Datetime"].iloc[0])
        assert stored_date.date() == recording_date.date(), (
            f"Recording Datetime should be {recording_date.date()}, got: {stored_date.date()}"
        )
        assert audio_sheet["Recording Timezone"].iloc[0] == "UTC"

    def test_audio_sheet_has_qc_artifact_segments(
        self,
        db_session,
        repository,
        audio_processing_factory,
        tmp_path,
    ):
        """Test that QC Artifact Segments column exists and is populated."""
        # Create audio with QC artifact data
        audio = audio_processing_factory(
            study="AOA",
            study_id=6003,
            knee="left",
            maneuver="fe",
            audio_file_name="HP_W11.2-3-20240315_143022.bin",
            device_serial="3",
            qc_continuous_artifact=True,
            qc_artifact_segments=[(0.5, 1.5), (3.0, 4.2)],  # Two artifact segments as tuples
        )
        audio_record = repository.save_audio_processing(audio)
        db_session.commit()

        # Generate report
        report = ReportGenerator(db_session)
        output_path = report.save_to_excel(
            tmp_path / "test_artifact.xlsx",
            participant_id=audio_record.participant_id,
            maneuver="fe",
            knee="left",
        )

        # Read Audio sheet
        audio_sheet = pd.read_excel(output_path, sheet_name="Audio")

        # Should have QC Artifact Segments column
        assert "QC Artifact Segments" in audio_sheet.columns, (
            "Audio sheet missing 'QC Artifact Segments' column"
        )

        # Should have segment data
        segments = audio_sheet["QC Artifact Segments"].iloc[0]
        assert segments is not None and len(segments) > 0, (
            "QC Artifact Segments should be populated"
        )

    def test_audio_sheet_has_qc_signal_dropout_segments(
        self,
        db_session,
        repository,
        audio_processing_factory,
        tmp_path,
    ):
        """Test that QC Signal Dropout Segments column exists and is populated."""
        # Create audio with signal dropout data
        audio = audio_processing_factory(
            study="AOA",
            study_id=6004,
            knee="right",
            maneuver="walk",
            audio_file_name="HP_W11.2-7-20240315_143022.bin",
            device_serial="7",
            qc_signal_dropout=True,
            qc_signal_dropout_segments=[(2.1, 2.8), (5.5, 6.0)],  # Two dropout segments as tuples
        )
        audio_record = repository.save_audio_processing(audio)
        db_session.commit()

        # Generate report
        report = ReportGenerator(db_session)
        output_path = report.save_to_excel(
            tmp_path / "test_dropout.xlsx",
            participant_id=audio_record.participant_id,
            maneuver="walk",
            knee="right",
        )

        # Read Audio sheet
        audio_sheet = pd.read_excel(output_path, sheet_name="Audio")

        # Should have QC Signal Dropout Segments column
        assert "QC Signal Dropout Segments" in audio_sheet.columns, (
            "Audio sheet missing 'QC Signal Dropout Segments' column"
        )

        # Should have segment data
        segments = audio_sheet["QC Signal Dropout Segments"].iloc[0]
        assert segments is not None and len(segments) > 0, (
            "QC Signal Dropout Segments should be populated"
        )

    def test_audio_sheet_has_per_channel_segments(
        self,
        db_session,
        repository,
        audio_processing_factory,
        tmp_path,
    ):
        """Test that per-channel artifact and dropout segment columns exist."""
        # Create audio with per-channel QC data
        audio = audio_processing_factory(
            study="AOA",
            study_id=6005,
            knee="left",
            maneuver="walk",
            audio_file_name="HP_W11.2-9-20240315_143022.bin",
            device_serial="9",
            qc_artifact_ch1=True,
            qc_artifact_segments_ch1=[(0.1, 0.5)],
            qc_signal_dropout_ch2=True,
            qc_signal_dropout_segments_ch2=[(1.0, 1.8)],
        )
        audio_record = repository.save_audio_processing(audio)
        db_session.commit()

        # Generate report
        report = ReportGenerator(db_session)
        output_path = report.save_to_excel(
            tmp_path / "test_per_channel.xlsx",
            participant_id=audio_record.participant_id,
            maneuver="walk",
            knee="left",
        )

        # Read Audio sheet
        audio_sheet = pd.read_excel(output_path, sheet_name="Audio")

        # Should have per-channel segment columns
        expected_columns = [
            "QC Artifact Segments Ch1",
            "QC Artifact Segments Ch2",
            "QC Artifact Segments Ch3",
            "QC Artifact Segments Ch4",
            "QC Signal Dropout Segments Ch1",
            "QC Signal Dropout Segments Ch2",
            "QC Signal Dropout Segments Ch3",
            "QC Signal Dropout Segments Ch4",
        ]

        for col in expected_columns:
            assert col in audio_sheet.columns, (
                f"Audio sheet missing '{col}' column"
            )

    def test_audio_sheet_recording_length_and_filename(
        self,
        db_session,
        repository,
        audio_processing_factory,
        tmp_path,
    ):
        """Test recording length and original audio file name handling."""
        audio = audio_processing_factory(
            study="AOA",
            study_id=6006,
            knee="left",
            maneuver="walk",
            audio_file_name="HP_W11.2-5-20240312_124055_with_freq",
            duration_seconds=12.5,
        )
        audio_record = repository.save_audio_processing(audio)
        db_session.commit()

        report = ReportGenerator(db_session)
        output_path = report.save_to_excel(
            tmp_path / "test_length.xlsx",
            participant_id=audio_record.participant_id,
            maneuver="walk",
            knee="left",
        )

        audio_sheet = pd.read_excel(output_path, sheet_name="Audio")
        assert audio_sheet["Audio File Name"].iloc[0] == "HP_W11.2-5-20240312_124055"
        assert audio_sheet["Recording Length (s)"].iloc[0] == 12.5
