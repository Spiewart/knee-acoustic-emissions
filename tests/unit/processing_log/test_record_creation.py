"""Helper tests for DB-backed reporting and sync record creation."""

import pandas as pd

from src.orchestration.processing_log import create_sync_record_from_data
from src.reports.report_generator import ReportGenerator


def test_create_sync_record_normalizes_speed():
    synced_df = pd.DataFrame({
        "tt": pd.to_timedelta([0, 1, 2], unit="s"),
        "ch1": [0.1, 0.2, 0.3],
    })
    record = create_sync_record_from_data(
        sync_file_name="test_sync.pkl",
        synced_df=synced_df,
        pass_number=1,
        speed="normal",
        audio_stomp_time=2.0,
        detection_results={"consensus_time": 2.0, "consensus_methods": ["rms", "onset"]},
    )

    assert record.speed == "medium"
    assert record.consensus_methods == "rms, onset"


def test_report_generator_summary_counts(db_session, repository, audio_processing_factory, tmp_path):
    audio = audio_processing_factory(study="AOA", study_id=2001, audio_file_name="AOA2001_audio")
    audio_record = repository.save_audio_processing(audio)

    report = ReportGenerator(db_session)
    output_path = report.save_to_excel(
        tmp_path / "summary.xlsx",
        participant_id=audio_record.participant_id,
        maneuver="walk",
        knee="left",
    )

    summary = pd.read_excel(output_path, sheet_name="Summary")
    assert summary.loc[summary["Metric"] == "Audio Records", "Value"].iloc[0] == 1
