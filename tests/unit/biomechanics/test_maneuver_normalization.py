from pathlib import Path

from src.orchestration.processing_log import create_biomechanics_record_from_data


def test_biomechanics_record_normalizes_maneuver():
    record = create_biomechanics_record_from_data(
        biomechanics_file=Path("bio.xlsx"),
        recordings=[],
        sheet_name="Sheet1",
        maneuver="sit_to_stand",
        biomechanics_type="Motion Analysis",
        knee="left",
        biomechanics_sync_method="stomp",
        biomechanics_sample_rate=100.0,
        study_id=1001,
    )

    assert record.maneuver == "sts"