import pandas as pd
import pytest

from src.biomechanics.importers import (
    _construct_biomechanics_sheet_names,
    _extract_maneuver_from_uid,
    _extract_walking_pass_info,
    _get_event_sheet_name,
    clean_uid,
    create_composite_column_names,
    extract_recording_data,
    extract_unique_ids_from_columns,
    get_biomechanics_metadata,
    get_non_walk_start_time,
    get_walking_start_time,
    import_biomechanics_recordings,
    normalize_recording_dataframe,
)
from src.models import BiomechanicsCycle


def test_import_biomechanics(fake_participant_directory) -> None:
    """Test importing biomechanics recordings from an Excel file.

    Every Excel file imported this way should return a list of DataFrames
    with the Time (sec) column updated to reflect the start time of the
    event as indicated in the event metadata sheet.
    """
    from pathlib import Path

    excel_file = Path(fake_participant_directory["biomechanics"]["excel_path"])

    biomechanics_recordings = import_biomechanics_recordings(
        biomechanics_file=excel_file,
        maneuver="walk",
        speed="slow",
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
        # Start time should be > 0 seconds
        assert time_values[0] > pd.to_timedelta(0, unit="s")


def test_extract_unique_ids_from_columns() -> None:
    """Test extracting unique column identifiers from DataFrame columns."""
    # Create a test DataFrame with duplicate column names that have suffixes
    data = {
        "V3D\\Study1_Walk_SP1": [1, 2, 3],
        "V3D\\Study1_Walk_SP1.1": [4, 5, 6],
        "V3D\\Study1_Walk_SP2": [7, 8, 9],
        "V3D\\Study1_Walk_SP2.1": [10, 11, 12],
    }
    df = pd.DataFrame(data)

    unique_ids = extract_unique_ids_from_columns(df)

    # Should have removed the ".1" suffixes
    assert len(unique_ids) == 2
    assert "V3D\\Study1_Walk_SP1" in unique_ids
    assert "V3D\\Study1_Walk_SP2" in unique_ids


def test_clean_uid() -> None:
    """Test cleaning unique identifiers."""
    uid = "V3D\\Study123_Walk0001_NSP1_Filt.c3d"
    cleaned = clean_uid(uid)

    assert cleaned == "Study123_Walk0001_NSP1_Filt"
    assert "V3D" not in cleaned
    assert ".c3d" not in cleaned


def test_extract_recording_data() -> None:
    """Test extracting recording data for a specific unique ID."""
    data = {
        "Study1_Walk_SP1": [1, 2, 3],
        "Study1_Walk_SP1_angle": [4, 5, 6],
        "Study1_Walk_SP2": [7, 8, 9],
    }
    df = pd.DataFrame(data)

    # Extract data for Study1_Walk_SP1
    result = extract_recording_data(df, "Study1_Walk_SP1")

    assert isinstance(result, pd.DataFrame)
    assert len(result.columns) == 2
    assert "Study1_Walk_SP1" in result.columns
    assert "Study1_Walk_SP1_angle" in result.columns
    assert "Study1_Walk_SP2" not in result.columns


def test_extract_recording_data_not_found() -> None:
    """Test that extracting non-existent data raises ValueError."""
    data = {"col1": [1, 2, 3]}
    df = pd.DataFrame(data)

    with pytest.raises(ValueError, match="No data found for unique ID"):
        extract_recording_data(df, "nonexistent")


def test_create_composite_column_names() -> None:
    """Test creating composite column names from two rows."""
    first_row = pd.Series(["LAnkleAngles", "RAnkleAngles", "LHipAngles"])
    second_row = pd.Series(["X", "Y", "Z"])

    result = create_composite_column_names(first_row, second_row)

    assert result == ["LAnkleAngles_X", "RAnkleAngles_Y", "LHipAngles_Z"]


def test_normalize_recording_dataframe() -> None:
    """Test normalizing a recording DataFrame."""
    # Create a test DataFrame with header rows
    data = {
        "col1": ["Frame", "", 0.0, 1.0, 2.0],
        "col2": ["LAnkleAngles", "X", 10.0, 11.0, 12.0],
        "col3": ["LAnkleAngles", "Y", 20.0, 21.0, 22.0],
    }
    df = pd.DataFrame(data)

    start_time = pd.to_timedelta(5.0, "s")
    result = normalize_recording_dataframe(df, start_time)

    # Check that header rows are removed
    assert len(result) == 3

    # Check that column names are composite
    assert "TIME" in result.columns
    assert "LAnkleAngles_X" in result.columns
    assert "LAnkleAngles_Y" in result.columns

    # Check that TIME column is properly adjusted
    assert result["TIME"].dtype == "timedelta64[ns]"
    # First TIME value should be the start_time
    assert result["TIME"].iloc[0] == start_time


def test_get_biomechanics_metadata() -> None:
    """Test extracting metadata from unique identifier."""
    # Test with Slow Speed, Pass 1
    uid = "Study123_Walk0001_SSP1_Filt"
    metadata = get_biomechanics_metadata(uid)

    assert metadata.maneuver == "walk"
    assert metadata.speed == "slow"
    assert metadata.pass_number == 1

    # Test with Normal Speed, Pass 2
    uid = "Study456_Walk0002_NSP2_Filt"
    metadata = get_biomechanics_metadata(uid)

    assert metadata.maneuver == "walk"
    assert metadata.speed == "normal"
    assert metadata.pass_number == 2

    # Test with Fast Speed, Pass 1
    uid = "Study789_Walk0001_FSP1_Filt"
    metadata = get_biomechanics_metadata(uid)

    assert metadata.maneuver == "walk"
    assert metadata.speed == "fast"
    assert metadata.pass_number == 1


def test_get_walking_start_time(fake_participant_directory) -> None:
    """Test retrieving walking start time from event data."""
    events_df = pd.read_excel(
        fake_participant_directory["biomechanics"]["excel_path"],
        sheet_name=fake_participant_directory["biomechanics"]["events_sheets"][
            "walk_pass"
        ],
    )

    # Get the start time for Slow Speed Pass 1
    start_time = get_walking_start_time(
        event_data_df=events_df,
        pass_number=1,
        pass_speed="slow",
    )

    # Should be 19.28 seconds (from fixture)
    assert start_time == pd.to_timedelta(19.28, "s")

    # Get the start time for Normal Speed Pass 2
    start_time = get_walking_start_time(
        event_data_df=events_df,
        pass_number=2,
        pass_speed="normal",
    )

    # Should be 144.13 seconds (from fixture)
    assert start_time == pd.to_timedelta(144.13, "s")


def test_get_walking_start_time_not_found(fake_participant_directory) -> None:
    """Test that requesting non-existent start time raises ValueError."""
    events_df = pd.read_excel(
        fake_participant_directory["biomechanics"]["excel_path"],
        sheet_name=fake_participant_directory["biomechanics"]["events_sheets"][
            "walk_pass"
        ],
    )

    # Try to get a start time that doesn't exist
    with pytest.raises(ValueError, match="No start time found"):
        get_walking_start_time(
            event_data_df=events_df,
            pass_number=99,
            pass_speed="slow",
        )


def test_get_event_sheet_name_walk() -> None:
    """Test event sheet name generation for walk maneuver."""
    sheet_name = _get_event_sheet_name(
        study_id="AOA1011",
        maneuver="walk",
        pass_number=1,
    )
    assert sheet_name == "AOA1011_Walk0001"

    sheet_name = _get_event_sheet_name(
        study_id="AOA1011",
        maneuver="walk",
        pass_number=5,
    )
    assert sheet_name == "AOA1011_Walk0001"


def test_get_event_sheet_name_sit_to_stand() -> None:
    """Test event sheet name generation for sit-to-stand maneuver."""
    sheet_name = _get_event_sheet_name(
        study_id="AOA1011",
        maneuver="sit_to_stand",
    )
    assert sheet_name == "AOA1011_StoS_Events"


def test_get_event_sheet_name_flexion_extension() -> None:
    """Test event sheet name generation for flexion-extension maneuver."""
    sheet_name = _get_event_sheet_name(
        study_id="AOA1011",
        maneuver="flexion_extension",
    )
    assert sheet_name == "AOA1011_FE_Events"


def test_get_event_sheet_name_walk_missing_pass_number() -> None:
    """Walk maneuver defaults to Walk0001 regardless of pass_number."""
    sheet_name = _get_event_sheet_name(
        study_id="AOA1011",
        maneuver="walk",
        pass_number=None,
    )
    assert sheet_name == "AOA1011_Walk0001"


def test_construct_biomechanics_sheet_names_walk() -> None:
    """Test sheet name construction for walk maneuver."""
    sheet_names = _construct_biomechanics_sheet_names(
        study_id="AOA1011",
        maneuver="walk",
        speed="slow",
        pass_number=1,
    )
    assert sheet_names["data"] == "AOA1011_Slow_Walking"
    assert sheet_names["events"] == "AOA1011_Walk0001"

    sheet_names = _construct_biomechanics_sheet_names(
        study_id="AOA1011",
        maneuver="walk",
        speed="medium",
        pass_number=2,
    )
    assert sheet_names["data"] == "AOA1011_Medium_Walking"
    assert sheet_names["events"] == "AOA1011_Walk0001"

    sheet_names = _construct_biomechanics_sheet_names(
        study_id="AOA1011",
        maneuver="walk",
        speed="fast",
        pass_number=1,
    )
    assert sheet_names["data"] == "AOA1011_Fast_Walking"
    assert sheet_names["events"] == "AOA1011_Walk0001"


def test_construct_biomechanics_sheet_names_sit_to_stand() -> None:
    """Test sheet name construction for sit-to-stand maneuver."""
    sheet_names = _construct_biomechanics_sheet_names(
        study_id="AOA1011",
        maneuver="sit_to_stand",
        speed=None,
    )
    assert sheet_names["data"] == "AOA1011_SitToStand"
    assert sheet_names["events"] == "AOA1011_StoS_Events"


def test_construct_biomechanics_sheet_names_flexion_extension() -> None:
    """Test sheet name construction for flexion-extension maneuver."""
    sheet_names = _construct_biomechanics_sheet_names(
        study_id="AOA1011",
        maneuver="flexion_extension",
        speed=None,
    )
    assert sheet_names["data"] == "AOA1011_FlexExt"
    assert sheet_names["events"] == "AOA1011_FE_Events"


def test_construct_biomechanics_sheet_names_walk_missing_speed() -> None:
    """Test that walk maneuver requires speed."""
    with pytest.raises(ValueError, match="speed is required"):
        _construct_biomechanics_sheet_names(
            study_id="AOA1011",
            maneuver="walk",
            speed=None,
        )


def test_extract_walking_pass_info_slow() -> None:
    """Test extracting pass number and speed from UID."""
    pass_number, speed = _extract_walking_pass_info("Study123_Walk0001_SSP1_Filt")
    assert pass_number == 1
    assert speed == "slow"


def test_extract_walking_pass_info_normal() -> None:
    """Test extracting normal speed pass info."""
    pass_number, speed = _extract_walking_pass_info("Study123_Walk0001_NSP2_Filt")
    assert pass_number == 2
    assert speed == "normal"


def test_extract_walking_pass_info_fast() -> None:
    """Test extracting fast speed pass info."""
    pass_number, speed = _extract_walking_pass_info("Study123_Walk0001_FSP3_Filt")
    assert pass_number == 3
    assert speed == "fast"


def test_extract_walking_pass_info_invalid_speed() -> None:
    """Test that invalid speed code raises ValueError."""
    with pytest.raises(ValueError, match="Unknown speed code"):
        _extract_walking_pass_info("Study123_Walk0001_XSP1_Filt")


def test_extract_maneuver_from_uid_walk() -> None:
    """Test extracting walk maneuver from UID."""
    maneuver = _extract_maneuver_from_uid("Study123_Walk0001_NSP1_Filt")
    assert maneuver == "walk"


def test_extract_maneuver_from_uid_sit_to_stand() -> None:
    """Test extracting sit-to-stand maneuver from UID."""
    maneuver = _extract_maneuver_from_uid("Study123_SitToStand0001_Filt")
    assert maneuver == "sit_to_stand"


def test_extract_maneuver_from_uid_flexion_extension() -> None:
    """Test extracting flexion-extension maneuver from UID."""
    maneuver = _extract_maneuver_from_uid("Study123_FlexExt0001_Filt")
    assert maneuver == "flexion_extension"


def test_extract_maneuver_from_uid_invalid() -> None:
    """Test that invalid maneuver raises ValueError."""
    with pytest.raises(ValueError, match="Unknown maneuver"):
        _extract_maneuver_from_uid("Study123_Invalid0001_Filt")


def test_get_non_walk_start_time_sit_to_stand(
    fake_participant_directory,
) -> None:
    """Test retrieving sit-to-stand start time from event data."""
    events_df = pd.read_excel(
        fake_participant_directory["biomechanics"]["excel_path"],
        sheet_name=fake_participant_directory["biomechanics"]["events_sheets"][
            "sit_to_stand"
        ],
    )

    start_time = get_non_walk_start_time(
        event_data_df=events_df,
        maneuver="sit_to_stand",
    )

    # Should be 5.0 seconds (from fixture)
    assert start_time == pd.to_timedelta(5.0, "s")


def test_get_non_walk_start_time_flexion_extension(
    fake_participant_directory,
) -> None:
    """Test retrieving flexion-extension start time from event data."""
    events_df = pd.read_excel(
        fake_participant_directory["biomechanics"]["excel_path"],
        sheet_name=fake_participant_directory["biomechanics"]["events_sheets"][
            "flexion_extension"
        ],
    )

    start_time = get_non_walk_start_time(
        event_data_df=events_df,
        maneuver="flexion_extension",
    )

    # Should be 2.0 seconds (from fixture)
    assert start_time == pd.to_timedelta(2.0, "s")


def test_import_biomechanics_sit_to_stand(fake_participant_directory) -> None:
    """Test importing sit-to-stand biomechanics recordings."""
    from pathlib import Path

    excel_file = Path(fake_participant_directory["biomechanics"]["excel_path"])

    biomechanics_recordings = import_biomechanics_recordings(
        biomechanics_file=excel_file,
        maneuver="sit_to_stand",
        speed=None,
    )

    # Should have exactly one recording for sit-to-stand
    assert len(biomechanics_recordings) == 1

    recording = biomechanics_recordings[0]
    assert isinstance(recording, BiomechanicsCycle)
    assert recording.maneuver == "sit_to_stand"
    assert recording.speed is None
    assert recording.pass_number is None
    assert isinstance(recording.data, pd.DataFrame)
    assert not recording.data.empty
    assert "TIME" in recording.data.columns


def test_import_biomechanics_flexion_extension(
    fake_participant_directory,
) -> None:
    """Test importing flexion-extension biomechanics recordings."""
    from pathlib import Path

    excel_file = Path(fake_participant_directory["biomechanics"]["excel_path"])

    biomechanics_recordings = import_biomechanics_recordings(
        biomechanics_file=excel_file,
        maneuver="flexion_extension",
        speed=None,
    )

    # Should have exactly one recording for flexion-extension
    assert len(biomechanics_recordings) == 1

    recording = biomechanics_recordings[0]
    assert isinstance(recording, BiomechanicsCycle)
    assert recording.maneuver == "flexion_extension"
    assert recording.speed is None
    assert recording.pass_number is None
    assert isinstance(recording.data, pd.DataFrame)
    assert not recording.data.empty
    assert "TIME" in recording.data.columns
