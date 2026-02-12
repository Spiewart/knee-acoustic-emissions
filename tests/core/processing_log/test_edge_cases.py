"""DB-backed edge case tests for report generation."""

import pandas as pd

from src.reports.report_generator import ReportGenerator


def test_report_generator_summary_with_no_records(db_session, tmp_path):
    report = ReportGenerator(db_session)
    output_path = report.save_to_excel(
        tmp_path / "empty_report.xlsx",
        participant_id=9999,
        maneuver="walk",
        knee="left",
    )

    summary = pd.read_excel(output_path, sheet_name="Summary")
    assert summary.loc[summary["Metric"] == "Audio Records", "Value"].iloc[0] == 0
