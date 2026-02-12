import pandas as pd

from src.reports.report_generator import ReportGenerator


def test_summary_sheet_contains_metrics(db_session, tmp_path):
    report = ReportGenerator(db_session)
    output_path = report.save_to_excel(
        tmp_path / "summary.xlsx",
        participant_id=999,
        maneuver="walk",
        knee="left",
    )
    summary = pd.read_excel(output_path, sheet_name="Summary")
    assert {"Audio Records", "Biomechanics Records", "Synchronization Records"}.issubset(
        set(summary["Metric"])
    )
