import pandas as pd

from acoustic_emissions_processing.models import BiomechanicsCycle
from acoustic_emissions_processing.process_biomechanics import (
    import_biomechanics_recordings,
)


def test_import_biomechanics() -> None:
    """Test importing biomechanics recordings from an Excel file.
    Every Excel file imported this way should return a list of DataFrames with the Time (sec)
    column updated to reflect the start time of the event as indicated in the event metadata sheet."""

    sample_file_path = "/Users/spiewart/kae_signal_processing_ml/sample_files/AOA1011_Biomechanics_Full_Set.xlsx"
    sample_sheet_name = "AOA1011_Slow_Walking"
    metadata_sheet_name = "AOA1011_Walk0001"

    biomechanics_recordings = import_biomechanics_recordings(
        biomechanics_file=sample_file_path,
        data_sheet_name=sample_sheet_name,
        event_data_sheet_name=metadata_sheet_name,
    )

    assert len(biomechanics_recordings) == 3

    for recording in biomechanics_recordings:
        assert isinstance(recording, BiomechanicsCycle)
        assert recording.maneuver
        assert recording.maneuver in ["walk", "sit_to_stand", "flexion_extension"]
        assert recording.speed in ["slow", "normal", "fast", None]
        assert recording.pass_number
        assert isinstance(recording.pass_number, int)
        assert isinstance(recording.data, pd.DataFrame)
        assert not recording.data.empty
        assert isinstance(recording.data, pd.DataFrame)
        assert "TIME" in recording.data.columns
        assert recording.data["TIME"].dtype == "m8[ns]"
        time_values = recording.data["TIME"].values
        assert time_values[0] > pd.to_timedelta(0, unit="s")  # Start time should be >= 0 seconds
