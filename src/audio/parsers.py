from typing import Any, Literal, Optional

import pandas as pd

from src.models import AcousticsFileMetadata, MicrophonePosition


def find_knee_table_start(
    metadata_df: pd.DataFrame,
    knee: Literal["left", "right"],
) -> int:
    """Find the starting row index for a specific knee's data table.

    Args:
        metadata_df: DataFrame loaded from the metadata file
        knee: The knee laterality to search for ("left" or "right")

    Returns:
        The row index where the knee table starts

    Raises:
        ValueError: If the knee data is not found in the metadata file
    """
    knee_label: str = f"{knee[0].capitalize()} Knee"
    table_header_index: list[Any] = metadata_df.index[
        metadata_df.iloc[:, 0].str.contains(knee_label, na=False)
    ].tolist()
    if not table_header_index:
        raise ValueError(f"No data found for {knee_label} in metadata file.")
    # Data starts after the table header row
    return table_header_index[0] + 1


def extract_knee_metadata_table(
    metadata_df: pd.DataFrame,
    table_start_row: int,
) -> pd.DataFrame:
    """Extract and prepare knee metadata table from full metadata DataFrame.

    Args:
        metadata_df: Full metadata DataFrame
        table_start_row: Starting row index for the knee table

    Returns:
        Cleaned DataFrame with proper column names and data rows
    """
    # Get the sub-DataFrame for the desired knee (13 rows of data)
    knee_metadata_df: pd.DataFrame = metadata_df.iloc[
        table_start_row: table_start_row + 13, :
    ].reset_index(drop=True)
    # Rename the columns based on the values in the first row
    knee_metadata_df.columns = knee_metadata_df.iloc[0]
    knee_metadata_df: pd.DataFrame = knee_metadata_df.drop(
        knee_metadata_df.index[0]
    ).reset_index(drop=True)
    return knee_metadata_df


def normalize_maneuver_column(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize the Maneuvers column by cleaning and forward-filling data.

    This function:
    1. Removes "-" characters and normalizes whitespace in maneuver names.
    2. Forward-fills the Maneuvers column to associate each maneuver
       with its corresponding 3 microphone rows (mics 2-4 inherit mic 1's maneuver).

    This is necessary because metadata files only label the first microphone
    for each maneuver, leaving mics 2-4 blank. Forward-fill associates
    those rows with their parent maneuver.

    Args:
        df: DataFrame with a "Maneuvers" column to normalize.

    Returns:
        DataFrame with normalized and forward-filled Maneuvers column.
    """
    # Strip any "-" characters from the "Maneuvers" column
    df["Maneuvers"] = df["Maneuvers"].str.replace(
        "-", " ", regex=False,
    )
    # Eliminate anything more than a single space
    df["Maneuvers"] = df["Maneuvers"].str.replace(
        r"\s+", " ", regex=True,
    ).str.strip()
    # Fill in the Maneuvers column downwards to associate each maneuver
    # with the subsequent three empty rows for microphones 2-4
    df["Maneuvers"] = df["Maneuvers"].ffill()
    return df


def filter_by_maneuver(
    df: pd.DataFrame,
    scripted_maneuver: Literal["walk", "sit_to_stand", "flexion_extension"],
) -> pd.DataFrame:
    """Filter metadata DataFrame to rows matching the scripted maneuver.

    Searches the "Maneuvers" column for text matching the requested maneuver,
    with underscores replaced by spaces (e.g., "sit_to_stand" -> "sit to stand").

    Args:
        df: Metadata DataFrame with normalized Maneuvers column.
        scripted_maneuver: The maneuver to filter by: "walk", "sit_to_stand", or "flexion_extension".

    Returns:
        DataFrame containing only rows for the specified maneuver.
        Empty if no rows match.
    """
    maneuver_search_str: str = scripted_maneuver.replace("_", " ")
    return df.loc[
        df["Maneuvers"].str.contains(
            maneuver_search_str, case=False, na=False
        )
    ]


def extract_microphone_positions(
    maneuver_df: pd.DataFrame,
) -> tuple[dict, dict]:
    """Extract microphone positions and notes from maneuver metadata.

    Parses rows in the maneuver DataFrame to build dictionaries mapping
    microphone numbers (1-4) to their anatomical positions (patellar/laterality)
    and any associated notes.

    Args:
        maneuver_df: DataFrame containing microphone data for a maneuver.
                    Expected to have columns: "Microphone", "Patellar Position", "Laterality", "Notes".

    Returns:
        Tuple of (microphones dict, microphone_notes dict):
        - microphones: Maps mic_number (int) -> MicrophonePosition object.
        - microphone_notes: Maps mic_number (int) -> note string for mics with notes.
    """
    microphones = {}
    microphone_notes = {}

    for _, row in maneuver_df.iterrows():
        mic_number = int(row["Microphone"])
        microphones[mic_number] = MicrophonePosition(
            patellar_position=row["Patellar Position"],
            laterality=row["Laterality"],
        )
        if pd.notna(row["Notes"]):
            microphone_notes[mic_number] = row["Notes"]

    return microphones, microphone_notes


def extract_file_name_and_notes(
    maneuver_df: pd.DataFrame,
) -> tuple[str, Optional[str]]:
    """Extract file name and notes from maneuver metadata.

    Args:
        maneuver_df: DataFrame containing metadata for a maneuver

    Returns:
        Tuple of (file_name, notes)
    """
    raw_file_name = maneuver_df["File Name"].values[0]
    file_name = "" if pd.isna(raw_file_name) else str(raw_file_name)
    notes = None

    if pd.notna(maneuver_df["Notes"].values[0]):
        notes = maneuver_df["Notes"].values[0]

    return file_name, notes


def get_acoustics_metadata(
    metadata_file_path: str,
    scripted_maneuver: Literal["walk", "sit_to_stand", "flexion_extension"],
    knee: Literal["left", "right"],
    acoustics_sheet_name: str = "Acoustic Notes",
) -> AcousticsFileMetadata:
    """Gets acoustics metadata for a given acoustics file legend.

    Extracts the relevant metadata based on the scripted maneuver and knee
    laterality and returns a Pydantic Model instance that validates the
    metadata.

    File legend is two Excel tables on top of one another, separated by a
    blank row, where the first row in each table is either "R Knee" or
    "L Knee", and the second row is divided into columns:
        "Maneuvers": Literal["Walk (slow,medium, fast)",
                              "Flexion - Extension", "Sit - to - Stand"]
        "File Name": str
        "Microphone": Literal[1, 2, 3, 4]
        "Patellar Position": Literal["Infrapatellar", "Suprapatellar"]
        "Laterality": Literal["Medial", "Lateral"]
        "Notes": str

    Args:
        metadata_file_path: Path to the Excel file containing the acoustics
                           file legend.
        scripted_maneuver: The scripted maneuver to filter by.
        knee: The knee laterality to filter by.

    Returns:
        AcousticsFileMetadata: A Pydantic model instance containing the
                          validated acoustics metadata.
    """
    metadata_df: pd.DataFrame = pd.read_excel(
        metadata_file_path, sheet_name=acoustics_sheet_name, header=None
    )

    # Find the starting row for the desired knee's data
    table_start_row: int = find_knee_table_start(metadata_df, knee)

    # Extract and prepare the knee metadata table
    knee_metadata_df: pd.DataFrame = extract_knee_metadata_table(
        metadata_df, table_start_row
    )

    # Normalize the Maneuvers column
    knee_metadata_df: pd.DataFrame = normalize_maneuver_column(knee_metadata_df)

    # Filter to the specified maneuver
    maneuver_metadata_df: pd.DataFrame = filter_by_maneuver(
        knee_metadata_df, scripted_maneuver
    )

    # Extract file name and notes
    file_name, notes = extract_file_name_and_notes(maneuver_metadata_df)

    # Extract microphone positions and notes
    microphones, microphone_notes = extract_microphone_positions(
        maneuver_metadata_df
    )

    acoustics_metadata = AcousticsFileMetadata(
        scripted_maneuver=scripted_maneuver,
        knee=knee,
        study="AOA",
        study_id=0,
        file_name=file_name,
        microphones=microphones,
        audio_notes=("; ".join([microphone_note for microphone_note in microphone_notes.values()]) if microphone_notes else notes),
    )

    return acoustics_metadata
