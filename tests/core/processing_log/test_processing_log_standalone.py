"""Standalone-style DB report test."""

import pandas as pd

from src.reports.report_generator import ReportGenerator


def test_report_generator_basic(db_session, repository, audio_processing_factory, tmp_path):
    """Test that ReportGenerator can save Excel report."""
    # Create audio record
    audio = audio_processing_factory(study="AOA", study_id=5555, knee="left", maneuver="walk")
    audio_record = repository.save_audio_processing(audio)
    db_session.commit()

    # Generate and save report
    report = ReportGenerator(db_session)
    output_path = report.save_to_excel(
        tmp_path / "standalone_report.xlsx",
        participant_id=audio_record.study_id,
        maneuver="walk",
        knee="left",
    )

    # Verify output file exists
    assert output_path.exists()

    # Verify summary sheet can be read
    summary = pd.read_excel(output_path, sheet_name="Summary")
    assert "Metric" in summary.columns or len(summary) > 0


def test_report_with_empty_database(db_session, tmp_path):
    """Test that ReportGenerator handles empty database gracefully."""
    report = ReportGenerator(db_session)
    output_path = report.save_to_excel(
        tmp_path / "empty_report.xlsx",
        participant_id=99999,  # Non-existent participant
        maneuver="walk",
        knee="left",
    )

    # Should still create file with empty or summary-only content
    assert output_path.exists()
