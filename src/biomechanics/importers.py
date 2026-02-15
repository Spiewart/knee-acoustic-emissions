"""Collection of methods to process and update biomechanics files generated
by QTM and Visual3D software."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import pandas as pd

from src.models import BiomechanicsFileMetadata, BiomechanicsRecording


def extract_unique_ids_from_columns(bio_df: pd.DataFrame) -> list[str]:
    """Extract unique column identifiers from biomechanics DataFrame.

    Removes numeric suffixes (e.g., ".1", ".2") added by Pandas when reading
    duplicate column names, leaving clean identifiers.

    Args:
        bio_df: DataFrame with column names potentially containing duplicate suffixes.

    Returns:
        List of unique column identifier strings (cleaned of Pandas suffixes).
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
    Visual3D (V3D) file path. For example:
    "V3D\\Study123_Walk0001_NSP1_Filt.c3d" -> "Study123_Walk0001_NSP1_Filt"

    Args:
        uid: The unique identifier string to clean (may include V3D path and .c3d extension).

    Returns:
        Cleaned unique identifier with path and extension removed.
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


def create_composite_column_names(first_row: pd.Series, second_row: pd.Series) -> list[str]:
    """Create composite column names from two rows of data.

    Combines values from row 0 and row 1 of each column with an underscore.
    E.g., ("LAnkleAngles", "X") -> "LAnkleAngles_X"

    Args:
        first_row: Values from first row (typically row 0)
        second_row: Values from second row (typically row 1)

    Returns:
        List of composite column names
    """
    new_column_names = [f"{first}_{second}" for first, second in zip(first_row, second_row)]
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
    recording_df["TIME"] = pd.to_timedelta(recording_df["TIME"], unit="s")
    recording_df["TIME"] = recording_df["TIME"] - recording_df["TIME"].iloc[0] + start_time

    return recording_df


def _construct_biomechanics_sheet_names(
    study_id: str,
    maneuver: Literal["walk", "sit_to_stand", "flexion_extension"],
    speed: Literal["slow", "medium", "fast"] | None,
    study_name: str = "AOA",
) -> dict[str, str]:
    """Construct Excel sheet names via study config dispatch.

    Delegates to ``StudyConfig.construct_biomechanics_sheet_names()`` for
    the given study so that sheet naming conventions are study-specific.

    Args:
        study_id: Full study-prefixed ID (e.g., "AOA1011")
        maneuver: Type of maneuver
        speed: Speed level (required for walk, ignored for others)
        study_name: Study identifier for config dispatch

    Returns:
        Dictionary with "data_sheet" and "event_sheet" keys
    """
    from src.studies import get_study_config

    study_config = get_study_config(study_name)
    return study_config.construct_biomechanics_sheet_names(
        study_id,
        maneuver,
        speed=speed,
    )


def _extract_stomp_time(
    event_df: pd.DataFrame,
    event_name: str,
    study_name: str = "AOA",
) -> float:
    """Extract stomp time for a specific sync event.

    Trims leading and trailing whitespace from event names to handle
    erroneous keystrokes in the data.

    Args:
        event_df: DataFrame with event and time columns.
        event_name: Name of the sync event (e.g., "Sync Left").
        study_name: Study identifier for config dispatch.

    Returns:
        Time in seconds for the sync event.

    Raises:
        ValueError: If event not found in DataFrame.
    """
    from src.studies import get_study_config

    study_config = get_study_config(study_name)
    event_col = study_config.get_biomechanics_event_column()
    time_col = study_config.get_biomechanics_time_column()

    event_name_stripped = event_name.strip()
    event_row = event_df[event_df[event_col].str.strip() == event_name_stripped]
    if event_row.empty:
        raise ValueError(f"Event '{event_name_stripped}' not found in event data")

    return float(event_row[time_col].iloc[0])


def _process_biomechanics_recordings(
    bio_df: pd.DataFrame,
    event_data_df: pd.DataFrame,
    maneuver: Literal["walk", "sit_to_stand", "flexion_extension"],
    biomechanics_file: Path,
    biomechanics_type: str | None = None,
    study_name: str | None = None,
) -> list[BiomechanicsRecording]:
    """Process biomechanics data into BiomechanicsRecording objects.

    This is a shared helper that processes the data/event DataFrames
    and creates BiomechanicsRecording objects for each unique recording.
    Handles both walk and non-walk maneuvers.

    Args:
        bio_df: DataFrame containing biomechanics data.
        event_data_df: DataFrame containing event metadata.
        maneuver: Type of maneuver.
        biomechanics_file: Path to the source Excel file.
        biomechanics_type: Optional biomechanics system/type.
        study_name: Study identifier for config dispatch.

    Returns:
        List of BiomechanicsRecording objects.
    """
    from src.studies import get_study_config
    from src.synchronization.sync import get_bio_start_time

    resolved_study_name = study_name or "AOA"
    study_config = get_study_config(resolved_study_name)

    unique_ids = extract_unique_ids_from_columns(bio_df)

    recordings = []
    for uid in unique_ids:
        uid_clean = clean_uid(uid)
        metadata = get_biomechanics_metadata(
            uid_clean,
            study_name=resolved_study_name,
        )

        # Get the start time based on maneuver type
        if maneuver == "walk":
            assert metadata.pass_number is not None, "pass_number must not be None for walk maneuver"
            assert metadata.speed is not None, "speed must not be None for walk maneuver"
            start_time = pd.Timedelta(
                get_bio_start_time(
                    event_metadata=event_data_df,
                    maneuver="walk",
                    speed=cast(Literal["slow", "normal", "fast"], metadata.speed),
                    pass_number=metadata.pass_number,
                    study_name=resolved_study_name,
                ),
            )
        else:
            # Non-walk maneuvers already have TIME relative to their own
            # recording start; do not frameshift by Movement Start.
            start_time = pd.Timedelta(seconds=0)

        # Extract and normalize recording data
        recording_df = extract_recording_data(bio_df, uid_clean)
        recording_df = normalize_recording_dataframe(
            recording_df,
            start_time,
        )

        # Extract sync times for walk maneuvers
        biomech_sync_left_time = biomech_sync_right_time = None
        if maneuver == "walk":
            sync_left_event = study_config.get_stomp_event_name(
                "left",
            )
            sync_right_event = study_config.get_stomp_event_name(
                "right",
            )
            biomech_sync_left_time = _extract_stomp_time(
                event_data_df,
                sync_left_event,
                study_name=resolved_study_name,
            )
            biomech_sync_right_time = _extract_stomp_time(
                event_data_df,
                sync_right_event,
                study_name=resolved_study_name,
            )

        base_kwargs = metadata.model_dump()
        base_kwargs.update(
            {
                "biomech_file_name": biomechanics_file.name,
                "biomech_sync_left_time": biomech_sync_left_time,
                "biomech_sync_right_time": biomech_sync_right_time,
                "pass_data": (event_data_df if maneuver == "walk" else None),
                "data": recording_df,
            },
        )

        recordings.append(BiomechanicsRecording(**base_kwargs))

    return recordings


def import_walk_biomechanics(
    biomechanics_file: Path,
    speed: Literal["slow", "normal", "fast"],
    biomechanics_type: str | None = None,
    study_name: str | None = None,
) -> list[BiomechanicsRecording]:
    """Import walking biomechanics recordings from an Excel file.

    Handles walk-specific logic: extracting pass_number from speed sheet,
    calculating start times from pass metadata, and extracting sync events.

    Args:
        biomechanics_file: Path to the Excel file
        speed: Walking speed ("slow", "normal", or "fast")

    Returns:
        List of BiomechanicsRecording objects for the walking maneuver

    Raises:
        ValueError: If sheet not found or data extraction fails
        FileNotFoundError: If biomechanics file doesn't exist
    """
    bio_file = Path(biomechanics_file)
    if not bio_file.exists():
        raise FileNotFoundError(f"Biomechanics file not found: {bio_file}")

    # Extract study prefix from file name using study config
    from src.studies import get_study_config

    resolved_study_name = study_name or "AOA"
    study_config = get_study_config(resolved_study_name)
    raw_id = bio_file.stem.split("_")[0]
    _, numeric_id = study_config.parse_participant_id(raw_id)
    study_id = study_config.format_study_prefix(numeric_id)

    # Construct sheet names (data + events) for this speed
    sheet_names = _construct_biomechanics_sheet_names(
        study_id,
        "walk",
        cast(Literal["slow", "medium", "fast"], speed),
        study_name=resolved_study_name,
    )

    # Read the speed-specific data sheet to extract pass_number from UID columns
    speed_sheet_name = sheet_names["data_sheet"]
    try:
        speed_df = pd.read_excel(bio_file, sheet_name=speed_sheet_name)
        unique_ids_raw = extract_unique_ids_from_columns(speed_df)
        if not unique_ids_raw:
            raise ValueError(
                f"No valid recording columns found in sheet '{speed_sheet_name}' for {speed} walk maneuver"
            )
        get_biomechanics_metadata(
            clean_uid(unique_ids_raw[0]),
            study_name=resolved_study_name,
        )
    except ValueError as e:
        raise ValueError(f"Failed to extract pass_number for {speed} walking: {e}") from e
    data_sheet_name = sheet_names["data_sheet"]
    event_sheet_name = sheet_names["event_sheet"]

    # Read data and event sheets
    bio_df = pd.read_excel(bio_file, sheet_name=data_sheet_name)
    event_data_df = pd.read_excel(bio_file, sheet_name=event_sheet_name)

    # Process recordings using shared helper
    recordings = _process_biomechanics_recordings(
        bio_df=bio_df,
        event_data_df=event_data_df,
        maneuver="walk",
        biomechanics_file=bio_file,
        biomechanics_type=biomechanics_type,
        study_name=study_name,
    )

    # Validate: walking should have one or more recordings
    if len(recordings) < 1:
        raise ValueError(f"Expected at least one recording for maneuver 'walk', but found {len(recordings)}")

    return recordings


def import_fe_sts_biomechanics(
    biomechanics_file: Path,
    maneuver: Literal["sit_to_stand", "flexion_extension"],
    biomechanics_type: str | None = None,
    study_name: str | None = None,
) -> list[BiomechanicsRecording]:
    """Import flexion-extension or sit-to-stand biomechanics recordings.

    Handles non-walk maneuver logic: sets start_time to 0 (no frameshift),
    no sync time extraction, and validates exactly one recording.

    Args:
        biomechanics_file: Path to the Excel file
        maneuver: Maneuver type ("sit_to_stand" or "flexion_extension")

    Returns:
        List with single BiomechanicsRecording object for the maneuver

    Raises:
        ValueError: If not exactly one recording found or sheet not found
        FileNotFoundError: If biomechanics file doesn't exist
    """
    bio_file = Path(biomechanics_file)
    if not bio_file.exists():
        raise FileNotFoundError(f"Biomechanics file not found: {bio_file}")

    # Extract study prefix from file name using study config
    from src.studies import get_study_config

    resolved_study_name = study_name or "AOA"
    study_config = get_study_config(resolved_study_name)
    raw_id = bio_file.stem.split("_")[0]
    _, numeric_id = study_config.parse_participant_id(raw_id)
    study_id = study_config.format_study_prefix(numeric_id)

    # Construct sheet names
    sheet_names = _construct_biomechanics_sheet_names(
        study_id,
        maneuver,
        speed=None,
        study_name=resolved_study_name,
    )
    data_sheet_name = sheet_names["data_sheet"]
    event_sheet_name = sheet_names["event_sheet"]

    # Read data and event sheets
    bio_df = pd.read_excel(bio_file, sheet_name=data_sheet_name)
    event_data_df = pd.read_excel(bio_file, sheet_name=event_sheet_name)

    # Process recordings using shared helper
    recordings = _process_biomechanics_recordings(
        bio_df=bio_df,
        event_data_df=event_data_df,
        maneuver=maneuver,
        biomechanics_file=bio_file,
        biomechanics_type=biomechanics_type,
        study_name=study_name,
    )

    # Validate: non-walk maneuvers should have exactly one recording
    if len(recordings) != 1:
        raise ValueError(f"Expected exactly one recording for maneuver '{maneuver}', but found {len(recordings)}")

    return recordings


def import_biomechanics_recordings(
    biomechanics_file: Path,
    maneuver: Literal["walk", "sit_to_stand", "flexion_extension"],
    speed: Literal["slow", "normal", "fast"] | None = None,
    biomechanics_type: str | None = None,
    study_name: str | None = None,
) -> list[BiomechanicsRecording]:
    """Import biomechanics recordings from an Excel file.

    Extracts the study ID from the file name, constructs the appropriate sheet
    name based on maneuver type, and returns a list of BiomechanicsRecording objects.

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
        List of BiomechanicsRecording objects with data and metadata

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
            biomechanics_type=biomechanics_type,
            study_name=study_name,
        )
    else:
        return import_fe_sts_biomechanics(
            biomechanics_file=biomechanics_file,
            maneuver=maneuver,
            biomechanics_type=biomechanics_type,
            study_name=study_name,
        )


def get_biomechanics_metadata(
    uid: str,
    study_name: str = "AOA",
) -> BiomechanicsFileMetadata:
    """Extract maneuver, pass number, speed, and study info from a UID.

    Delegates to the study-specific config for all UID parsing logic.

    Args:
        uid: Cleaned UID (no V3D path prefix or .c3d extension).
        study_name: Study identifier for config dispatch.

    Returns:
        BiomechanicsFileMetadata with all parsed fields.
    """
    from src.studies import get_study_config

    return get_study_config(study_name).parse_biomechanics_uid(uid)
