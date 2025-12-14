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


def _get_event_sheet_name(
    study_id: str,
    maneuver: Literal["walk", "sit_to_stand", "flexion_extension"],
    speed: Literal["slow", "medium", "fast"] | None = None,
    pass_number: int | None = None,
) -> str:
    """Get the event sheet name based on maneuver type.

    Sheet naming convention:
    - Walk: AOAXXXX_Walk0001 (pass metadata with sync events, not speed-specific)
    - Sit-to-stand: AOAXXXX_StoS_Events
    - Flexion-extension: AOAXXXX_FE_Events

    Args:
        study_id: Study identifier (e.g., "AOA1011")
        maneuver: Type of maneuver
        speed: Speed level (unused for walk, ignored)
        pass_number: Pass number (unused, kept for compatibility)

    Returns:
        Event sheet name

    Raises:
        ValueError: If walk maneuver cannot resolve sheet name
    """
    if maneuver == "walk":
        # Walk0001 is the shared pass-metadata/sync sheet for all speeds
        return f"{study_id}_Walk0001"
    elif maneuver == "sit_to_stand":
        return f"{study_id}_StoS_Events"
    elif maneuver == "flexion_extension":
        return f"{study_id}_FE_Events"
    else:
        raise ValueError(f"Unknown maneuver: {maneuver}")


def _construct_biomechanics_sheet_names(
    study_id: str,
    maneuver: Literal["walk", "sit_to_stand", "flexion_extension"],
    speed: Literal["slow", "medium", "fast"] | None,
    pass_number: int | None = None,
) -> dict[str, str]:
    """Construct Excel sheet names based on study ID, maneuver, speed.

    Sheet naming convention:
    - Walk: AOAXXXX_Speed_Walking (Speed is Slow, Medium, or Fast)
    - Sit-to-stand: AOAXXXX_SitToStand
    - Flexion-extension: AOAXXXX_FlexExt
    - Events sheet: Depends on maneuver (see _get_event_sheet_name)

    Args:
        study_id: Study identifier (e.g., "AOA1011")
        maneuver: Type of maneuver
        speed: Speed level (required for walk, ignored for others)
        pass_number: Pass number (kept for compatibility, unused)

    Returns:
        Dictionary with "data" and "events" sheet names

    Raises:
        ValueError: If speed is None for walk maneuver
    """
    if maneuver == "walk":
        if speed is None:
            raise ValueError("speed is required for walk maneuver")

        speed_map = {
            "slow": "Slow",
            "medium": "Medium",
            "fast": "Fast",
        }
        speed_capitalized = speed_map[speed]
        data_sheet = f"{study_id}_{speed_capitalized}_Walking"
    elif maneuver == "sit_to_stand":
        data_sheet = f"{study_id}_SitToStand"
    elif maneuver == "flexion_extension":
        data_sheet = f"{study_id}_FlexExt"
    else:
        raise ValueError(f"Unknown maneuver: {maneuver}")

    # Get the appropriate events sheet name based on maneuver
    events_sheet = _get_event_sheet_name(
        study_id, maneuver, speed, pass_number
    )

    return {
        "data": data_sheet,
        "events": events_sheet,
    }


def _extract_stomp_time(
    event_df: pd.DataFrame,
    event_name: str,
) -> float:
    """Extract stomp time for a specific sync event.

    Args:
        event_df: DataFrame with "Event Info" and "Time (sec)" columns
        event_name: Name of the sync event (e.g., "Sync Left", "Sync Right")

    Returns:
        Time in seconds for the sync event

    Raises:
        ValueError: If event not found in DataFrame
    """
    event_row = event_df[event_df["Event Info"] == event_name]
    if event_row.empty:
        raise ValueError(f"Event '{event_name}' not found in event data")

    return float(event_row["Time (sec)"].iloc[0])


def _process_biomechanics_recordings(
    bio_df: pd.DataFrame,
    event_data_df: pd.DataFrame,
    maneuver: Literal["walk", "sit_to_stand", "flexion_extension"],
) -> list[BiomechanicsCycle]:
    """Process biomechanics data into BiomechanicsCycle objects.

    This is a shared helper that processes the data/event DataFrames
    and creates BiomechanicsCycle objects for each unique recording.
    Handles both walk and non-walk maneuvers.

    Args:
        bio_df: DataFrame containing biomechanics data
        event_data_df: DataFrame containing event metadata
        maneuver: Type of maneuver

    Returns:
        List of BiomechanicsCycle objects
    """
    # Get the unique identifiers in the column names
    unique_ids = extract_unique_ids_from_columns(bio_df)

    # Separate the DataFrame into individual recordings
    recordings = []
    for uid in unique_ids:
        uid_clean = clean_uid(uid)
        metadata = get_biomechanics_metadata(uid_clean)

        # Get the start time based on maneuver type
        if maneuver == "walk":
            # For walk maneuvers, speed and pass_number are guaranteed
            # non-None
            assert metadata.pass_number is not None, (
                "pass_number must not be None for walk maneuver"
            )
            assert metadata.speed is not None, (
                "speed must not be None for walk maneuver"
            )
            start_time = get_walking_start_time(
                event_data_df=event_data_df,
                pass_number=metadata.pass_number,
                pass_speed=metadata.speed,
            )
        else:
            # Non-walk maneuvers already have TIME relative to their own
            # recording start; do not frameshift by Movement Start.
            start_time = pd.Timedelta(seconds=0)

        # Extract and normalize recording data
        recording_df = extract_recording_data(bio_df, uid_clean)
        recording_df = normalize_recording_dataframe(recording_df, start_time)

        # Extract sync times for walk maneuvers
        sync_left_time = sync_right_time = None
        if maneuver == "walk":
            sync_left_time = _extract_stomp_time(event_data_df, "Sync Left")
            sync_right_time = _extract_stomp_time(
                event_data_df, "Sync Right"
            )

        recordings.append(
            BiomechanicsCycle(
                maneuver=metadata.maneuver,
                speed=metadata.speed,
                pass_number=metadata.pass_number,
                sync_left_time=sync_left_time,
                sync_right_time=sync_right_time,
                pass_metadata=event_data_df if maneuver == "walk" else None,
                data=recording_df,  # type: ignore[arg-type]
            )
        )

    return recordings


def import_walk_biomechanics(
    biomechanics_file: Path,
    speed: Literal["slow", "normal", "fast"],
) -> list[BiomechanicsCycle]:
    """Import walking biomechanics recordings from an Excel file.

    Handles walk-specific logic: extracting pass_number from speed sheet,
    calculating start times from pass metadata, and extracting sync events.

    Args:
        biomechanics_file: Path to the Excel file
        speed: Walking speed ("slow", "normal", or "fast")

    Returns:
        List of BiomechanicsCycle objects for the walking maneuver

    Raises:
        ValueError: If sheet not found or data extraction fails
        FileNotFoundError: If biomechanics file doesn't exist
    """
    bio_file = Path(biomechanics_file)
    if not bio_file.exists():
        raise FileNotFoundError(f"Biomechanics file not found: {bio_file}")

    # Extract study ID from file name (first 7 characters)
    study_id = bio_file.stem[:7]

    # Read the speed-specific sheet to extract pass_number from UID columns
    speed_capitalized = speed.capitalize()
    speed_sheet_name = f"{study_id}_{speed_capitalized}_Walking"
    try:
        speed_df = pd.read_excel(bio_file, sheet_name=speed_sheet_name)
        unique_ids_raw = extract_unique_ids_from_columns(speed_df)
        if not unique_ids_raw:
            raise ValueError(
                f"No valid recording columns found in sheet "
                f"'{speed_sheet_name}' for {speed} walk maneuver"
            )
        pass_number, _ = _extract_walking_pass_info(
            clean_uid(unique_ids_raw[0])
        )
    except ValueError as e:
        raise ValueError(
            f"Failed to extract pass_number for {speed} walking: {e}"
        )

    # Construct sheet names
    sheet_names = _construct_biomechanics_sheet_names(
        study_id, "walk", speed, pass_number
    )
    data_sheet_name = sheet_names["data"]
    event_sheet_name = sheet_names["events"]

    # Read data and event sheets
    bio_df = pd.read_excel(bio_file, sheet_name=data_sheet_name)
    event_data_df = pd.read_excel(bio_file, sheet_name=event_sheet_name)

    # Process recordings using shared helper
    recordings = _process_biomechanics_recordings(
        bio_df=bio_df,
        event_data_df=event_data_df,
        maneuver="walk",
    )

    # Validate: walking should have one or more recordings
    if len(recordings) < 1:
        raise ValueError(
            f"Expected at least one recording for maneuver 'walk', "
            f"but found {len(recordings)}"
        )

    return recordings


def import_fe_sts_biomechanics(
    biomechanics_file: Path,
    maneuver: Literal["sit_to_stand", "flexion_extension"],
) -> list[BiomechanicsCycle]:
    """Import flexion-extension or sit-to-stand biomechanics recordings.

    Handles non-walk maneuver logic: sets start_time to 0 (no frameshift),
    no sync time extraction, and validates exactly one recording.

    Args:
        biomechanics_file: Path to the Excel file
        maneuver: Maneuver type ("sit_to_stand" or "flexion_extension")

    Returns:
        List with single BiomechanicsCycle object for the maneuver

    Raises:
        ValueError: If not exactly one recording found or sheet not found
        FileNotFoundError: If biomechanics file doesn't exist
    """
    bio_file = Path(biomechanics_file)
    if not bio_file.exists():
        raise FileNotFoundError(f"Biomechanics file not found: {bio_file}")

    # Extract study ID from file name
    study_id = bio_file.stem[:7]

    # Construct sheet names
    sheet_names = _construct_biomechanics_sheet_names(
        study_id, maneuver, speed=None, pass_number=None
    )
    data_sheet_name = sheet_names["data"]
    event_sheet_name = sheet_names["events"]

    # Read data and event sheets
    bio_df = pd.read_excel(bio_file, sheet_name=data_sheet_name)
    event_data_df = pd.read_excel(bio_file, sheet_name=event_sheet_name)

    # Process recordings using shared helper
    recordings = _process_biomechanics_recordings(
        bio_df=bio_df,
        event_data_df=event_data_df,
        maneuver=maneuver,
    )

    # Validate: non-walk maneuvers should have exactly one recording
    if len(recordings) != 1:
        raise ValueError(
            f"Expected exactly one recording for maneuver '{maneuver}', "
            f"but found {len(recordings)}"
        )

    return recordings


def import_biomechanics_recordings(
    biomechanics_file: Path,
    maneuver: Literal["walk", "sit_to_stand", "flexion_extension"],
    speed: Literal["slow", "normal", "fast"] | None = None,
) -> list[BiomechanicsCycle]:
    """Import biomechanics recordings from an Excel file.

    Extracts the study ID from the file name, constructs the appropriate sheet
    name based on maneuver type, and returns a list of BiomechanicsCycle objects.

    Dispatches to specialized functions based on maneuver type:
    - Walking: import_walk_biomechanics
    - Sit-to-stand/Flexion-extension: import_fe_sts_biomechanics

    Args:
        biomechanics_file: Path to the Excel file
        maneuver: Type of maneuver ("walk", "sit_to_stand", or
            "flexion_extension")
        speed: Speed level. Required for "walk", must be None for other
            maneuvers.

    Returns:
        List of BiomechanicsCycle objects with data and metadata

    Raises:
        ValueError: If speed/maneuver combination is invalid or sheet not found
        FileNotFoundError: If biomechanics file doesn't exist
    """
    # Validate speed/maneuver combination
    if maneuver == "walk" and speed is None:
        raise ValueError("speed is required when maneuver is 'walk'")
    if maneuver != "walk" and speed is not None:
        raise ValueError(f"speed must be None when maneuver is '{maneuver}'")

    # Dispatch to specialized function based on maneuver type
    if maneuver == "walk":
        if speed is None:
            raise ValueError("speed is required for walk maneuvers")
        return import_walk_biomechanics(
            biomechanics_file=biomechanics_file,
            speed=speed,
        )
    else:
        return import_fe_sts_biomechanics(
            biomechanics_file=biomechanics_file,
            maneuver=maneuver,
        )



def _extract_maneuver_from_uid(
    uid: str,
) -> Literal["walk", "sit_to_stand", "flexion_extension"]:
    """Extract and normalize maneuver type from unique identifier.

    Converts from UID format (CamelCase) to Pydantic format (snake_case).
    E.g., "Walk" -> "walk", "SitToStand" -> "sit_to_stand"

    Args:
        uid: The unique identifier string

    Returns:
        Normalized maneuver name

    Raises:
        ValueError: If maneuver is not recognized
    """
    maneuver_raw = ''.join(filter(str.isalpha, uid.split("_")[1])).lower()

    # Map raw extracted maneuver to valid Pydantic Literal values
    maneuver_map: dict[
        str,
        Literal["walk", "sit_to_stand", "flexion_extension"],
    ] = {
        "walk": "walk",
        "sittostand": "sit_to_stand",
        "sitstand": "sit_to_stand",  # Common abbreviation
        "flexext": "flexion_extension",
    }

    maneuver = maneuver_map.get(maneuver_raw)
    if maneuver is None:
        raise ValueError(f"Unknown maneuver '{maneuver_raw}' in UID")

    return maneuver


def _extract_walking_pass_info(
    uid: str,
) -> tuple[int, Literal["slow", "normal", "fast"]]:
    """Extract pass number and speed from walking maneuver UID.

    Args:
        uid: The unique identifier string (e.g., "Study123_Walk0001_NSP1_Filt")

    Returns:
        Tuple of (pass_number, speed)

    Raises:
        ValueError: If pass info cannot be parsed or speed not recognized
    """
    speed_map: dict[str, Literal["slow", "normal", "fast"]] = {
        "SS": "slow",
        "NS": "normal",
        "FS": "fast",
    }

    pass_info = uid.split("_")[-2]
    pass_number = int(''.join(filter(str.isdigit, pass_info)))

    pass_speed_code = ''.join(filter(str.isalpha, pass_info))
    pass_speed_code = pass_speed_code.replace("P", "").upper()

    speed = speed_map.get(pass_speed_code)
    if speed is None:
        raise ValueError(
            (
                f"Unknown speed code '{pass_speed_code}' in pass info "
                f"'{pass_info}'"
            )
        )

    return pass_number, speed


def get_biomechanics_metadata(uid: str) -> BiomechanicsMetadata:
    """Extract maneuver, pass number, and speed from biomechanics unique
    identifier.

    UID Format: Study{StudyID}_{Maneuver}{StudyNum}_{SpeedPass}_{Filt}
    Example: Study123_Walk0001_NSP1_Filt

    Args:
        uid: The unique identifier string

    Returns:
        BiomechanicsMetadata with maneuver, speed, and pass_number

    Raises:
        ValueError: If UID format is invalid or speed code not recognized
    """
    maneuver = _extract_maneuver_from_uid(uid)

    # Non-walk maneuvers have no speed or pass_number
    if maneuver != "walk":
        return BiomechanicsMetadata(
            maneuver=maneuver,
            speed=None,
            pass_number=None,
        )

    # Walk maneuvers require speed and pass_number extraction
    pass_number, speed = _extract_walking_pass_info(uid)

    return BiomechanicsMetadata(
        maneuver="walk",
        speed=speed,
        pass_number=pass_number,
    )


def get_non_walk_start_time(
    event_data_df: pd.DataFrame,
    maneuver: Literal["sit_to_stand", "flexion_extension"],
) -> pd.Timedelta:
    """Get the start time for a non-walking maneuver from the event data
    DataFrame."""

    maneuver_map: dict[
        Literal["sit_to_stand", "flexion_extension"],
        str
    ] = {
        "sit_to_stand": "Movement Start",
        "flexion_extension": "Movement Start",
    }

    start_event_name = maneuver_map[maneuver]

    start_time_entries = event_data_df.loc[
        event_data_df["Event Info"] == start_event_name,
        "Time (sec)",
    ].dropna().tolist()
    if not start_time_entries:
        raise ValueError(f"No start time found for {start_event_name}")
    start_time = pd.to_timedelta(start_time_entries[0], "s")
    return start_time


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
