from pathlib import Path

from src.orchestration.processing_log import create_biomechanics_record_from_data


def test_biomechanics_record_normalizes_maneuver():
    record = create_biomechanics_record_from_data(
        biomechanics_file=Path("bio.xlsx"),
        recordings=[],
        sheet_name="Sheet1",
        maneuver="sit_to_stand",
        error=Exception("fail"),
    )

    assert record.maneuver == "sts"