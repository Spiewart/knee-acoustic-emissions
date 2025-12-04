"""Collection of methods to process and update biomechanics files generated
by QTM and Visual3D software."""

from pathlib import Path
from typing import Literal

import pandas as pd

from models import BiomechanicsCycle, BiomechanicsMetadata


def extract_unique_ids_from_columns(bio_df: pd.DataFrame) -> list[str]:
    """Extract unique column identifiers from biomechanics DataFrame.

    Takes column names and removes duplicate suffixes (e.g., ".1", ".2")
    added by Pandas for duplicate names.

    Args:
        bio_df: DataFrame with column names containing duplicate identifiers

    Returns:
        List of unique column identifier strings
    """
    all_cols = bio_df.columns
    # Drop the terminal ".#" from each column name
    # (these are added by Pandas for duplicate names)
    all_cols = all_cols.str.replace(r"\.\d+$", "", regex=True)
    unique_ids = all_cols.unique().tolist()
    return unique_ids


def clean_uid(uid: str) -> str:
    """Clean unique identifier by removing V3D path and file extension.

    Extracts just the study ID, maneuver, and pass info from a full
    V3D path. E.g., "V3D\\Study123_Walk0001_NSP1_Filt.c3d" ->
    "Study123_Walk0001_NSP1_Filt"

    Args:
        uid: The unique identifier string to clean

    Returns:
        Cleaned unique identifier without V3D path and .c3d extension
    """
    # Get everything after the final folder in the path "V3D"
    uid = uid.split("V3D\\")[-1]
    # Remove the terminal file extension (".c3d")
    uid = uid.replace(".c3d", "")
    return uid


def extract_recording_data(bio_df: pd.DataFrame, uid: str) -> pd.DataFrame:
    """Extract recording data for a specific unique identifier.

    Gets all columns from the DataFrame that match the given unique ID.

    Args:
        bio_df: The full biomechanics DataFrame
        uid: The unique identifier to extract data for

    Returns:
        DataFrame containing only the columns for this recording

    Raises:
        ValueError: If no data found for the given unique ID
    """
    recording_df = bio_df.loc[:, bio_df.columns.str.contains(uid, na=False)]
    if recording_df.empty:
        raise ValueError(f"No data found for unique ID: {uid}")
    return recording_df


def create_composite_column_names(
    first_row: pd.Series,
    second_row: pd.Series
) -> list[str]:
    """Create composite column names from two rows of data.

    Combines values from row 0 and row 1 of each column with an underscore.
    E.g., ("LAnkleAngles", "X") -> "LAnkleAngles_X"

    Args:
        first_row: Values from first row (typically row 0)
        second_row: Values from second row (typically row 1)

    Returns:
        List of composite column names
    """
    new_column_names = [
        f"{first}_{second}"
        for first, second in zip(first_row, second_row)
    ]
    return new_column_names


def normalize_recording_dataframe(
    recording_df: pd.DataFrame,
    start_time: pd.Timedelta,
) -> pd.DataFrame:
    """Normalize a recording DataFrame by processing its structure.

    This function:
    1. Creates composite column names from rows 0 and 1
    2. Drops those header rows
    3. Renames the first column to "TIME"
    4. Converts TIME to timedelta relative to the start_time

    Args:
        recording_df: The recording DataFrame to normalize
        start_time: Start time for this recording (for TIME adjustment)

    Returns:
        Normalized DataFrame with proper column names and TIME column
    """
    # Create an array of the value in row 0 and row 1 for each column
    first_row = recording_df.iloc[0]
    second_row = recording_df.iloc[1]
    # Create new column names by combining first and second row values
    # with an underscore
    new_column_names = create_composite_column_names(first_row, second_row)
    # Set the new column names
    recording_df.columns = new_column_names
    # Drop rows 0 and 1
    recording_df = recording_df.drop(index=[0, 1]).reset_index(drop=True)
    # Rename the first column to "TIME"
    first_col = recording_df.columns[0]
    recording_df = recording_df.rename(columns={first_col: "TIME"})
    # Convert TIME column to timedelta and adjust to be relative to start_time
    recording_df["TIME"] = pd.to_timedelta(recording_df["TIME"], unit='s')
    recording_df["TIME"] = (
        recording_df["TIME"]
        - recording_df["TIME"].iloc[0]
        + start_time
    )

    return recording_df


def import_biomechanics_recordings(
    biomechanics_file: str,
    data_sheet_name: str,
    event_data_sheet_name: str,
) -> list[BiomechanicsCycle]:
    """Import biomechanics recordings from an Excel file, separate into
    separate DataFrames based on the unique identifiers in the first row,
    adjust the TIME column of each to be relative to the start_time, then
    return a list of dicts of DataFrames with metadata: maneuver (walk,
    sit-to-stand, flexion-extension), speed (slow, normal, fast, None),
    pass (int) and the DataFrame itself."""

    bio_file = Path(biomechanics_file)

    bio_df = pd.read_excel(
        bio_file,
        sheet_name=data_sheet_name,
    )

    event_data_df = pd.read_excel(
        bio_file,
        sheet_name=event_data_sheet_name,
    )

    # Get the unique identifiers in the first row, which is the column names
    unique_ids = extract_unique_ids_from_columns(bio_df)

    # Separate the whole DataFrame into a list of DataFrames
    # based on the unique IDs
    recordings = []
    for uid in unique_ids:
        # Clean the unique ID
        uid_clean = clean_uid(uid)

        # This will be the study ID, the maneuver performed, and
        # the pass number followed by a "Filt" suffix
        # e.g., "Study123_Walk0001_NSP1_Filt"

        metadata = get_biomechanics_metadata(uid_clean)

        if not metadata.speed:
            # TODO: implement for non-walking maneuvers
            continue

        # Get the start time for this walking pass from the event data
        start_time = get_walking_start_time(
            event_data_df=event_data_df,
            pass_number=metadata.pass_number,
            pass_speed=metadata.speed,
        )

        # Set the recording_df to all the columns in the bio_df
        # that contain the uid
        recording_df = extract_recording_data(bio_df, uid_clean)

        # Normalize the recording DataFrame
        recording_df = normalize_recording_dataframe(recording_df, start_time)

        recordings.append(BiomechanicsCycle(
            maneuver=metadata.maneuver,
            speed=metadata.speed,
            pass_number=metadata.pass_number,
            data=recording_df,
        ))
    return recordings


def get_biomechanics_metadata(
    uid: str,
) -> BiomechanicsMetadata:

    """Extract maneuver, pass number, and speed from biomechanics unique
    identifier string."""
    maneuver = uid.split("_")[1]
    # Get any non-numeric characters from the maneuver string
    maneuver = ''.join(filter(str.isalpha, maneuver)).lower()
    pass_info = uid.split("_")[-2]

    # Get any integers, as these represent the pass number
    pass_number = int(''.join(filter(str.isdigit, pass_info)))
    # Get any non-numeric characters from the pass string
    pass_speed = ''.join(filter(str.isalpha, pass_info))
    # Remove any "P" characters from the speed, as these indicate "Pass"
    pass_speed = pass_speed.replace("P", "").upper()
    # Map abbreviations to full speed names
    speed_map = {
        "SS": "slow",
        "NS": "normal",
        "FS": "fast",
    }
    pass_speed = speed_map.get(pass_speed, None)

    return BiomechanicsMetadata(
        maneuver=maneuver,
        speed=pass_speed,
        pass_number=pass_number
    )


def get_walking_start_time(
    event_data_df: pd.DataFrame,
    pass_number: int,
    pass_speed: Literal["slow", "normal", "fast"],
) -> pd.Timedelta:
    """Get the start time for a walking pass from the event data DataFrame."""

    # Convert the pass_speed back to the prefix used in
    # the "Event Info" column
    pass_speed_prefix = {
        "slow": "SS",
        "normal": "NS",
        "fast": "FS",
    }
    pass_speed_prefix = pass_speed_prefix[pass_speed]
    # Get all the "Time (sec)" entries from the event data for this speed
    # by matching the speed to the starting chars of the "Event Info" column
    sub_event_data_df = event_data_df.loc[
        event_data_df["Event Info"].str.startswith(
            pass_speed_prefix,
            na=False,
        )
    ].copy()
    # Remove the speed prefix and the space following it from
    # the "Event Info" column
    prefix_len = len(pass_speed_prefix) + 1
    sub_event_data_df["Event Info"] = (
        sub_event_data_df["Event Info"].str[prefix_len:].str.strip()
    )
    # Get the start time by searching the "Event Info" column
    # for the start event name
    start_event_name = f"Pass {pass_number} Start"

    start_time_entries = sub_event_data_df.loc[
        sub_event_data_df["Event Info"] == start_event_name,
        "Time (sec)",
    ].dropna().tolist()
    if not start_time_entries:
        raise ValueError(f"No start time found for {start_event_name}")
    start_time = pd.to_timedelta(start_time_entries[0], "s")
    return start_time
