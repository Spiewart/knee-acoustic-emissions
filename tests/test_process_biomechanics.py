import pandas as pd

from models import BiomechanicsCycle
from process_biomechanics import (
    import_biomechanics_recordings,
)


def test_import_biomechanics(fake_biomechanics_excel) -> None:
    """Test importing biomechanics recordings from an Excel file.
    Every Excel file imported this way should return a list of DataFrames with the Time (sec)
    column updated to reflect the start time of the event as indicated in the event metadata sheet."""

    biomechanics_recordings = import_biomechanics_recordings(
        biomechanics_file=str(fake_biomechanics_excel["excel_path"]),
        data_sheet_name=fake_biomechanics_excel["data_sheet"],
        event_data_sheet_name=fake_biomechanics_excel["events_sheet"],
    )

    assert len(biomechanics_recordings) == 2

    for recording in biomechanics_recordings:
        assert isinstance(recording, BiomechanicsCycle)
        assert recording.maneuver
        maneuvers = ["walk", "sit_to_stand", "flexion_extension"]
        assert recording.maneuver in maneuvers
        speeds = ["slow", "normal", "fast", None]
        assert recording.speed in speeds
        assert recording.pass_number
        assert isinstance(recording.pass_number, int)
        assert isinstance(recording.data, pd.DataFrame)
        assert not recording.data.empty
        assert isinstance(recording.data, pd.DataFrame)
        assert "TIME" in recording.data.columns
        assert recording.data["TIME"].dtype == "m8[ns]"
        time_values = recording.data["TIME"].values
        # Start time should be >= 0 seconds
        assert time_values[0] > pd.to_timedelta(0, unit="s")
