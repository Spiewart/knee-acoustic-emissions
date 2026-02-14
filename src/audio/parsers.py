import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional

import pandas as pd

from src.models import AcousticsFileMetadata, MicrophonePosition

logger = logging.getLogger(__name__)


@dataclass
class MicSetupData:
    """Parsed data from the Mic Setup sheet for a single knee + maneuver."""

    file_name: str
    serial_number: str
    file_size_mb: Optional[float]
    timestamp: Optional[str]
    date_of_recording: Optional[datetime]
    microphones: dict[int, MicrophonePosition]
    notes: Optional[str]


@dataclass
class LegendMismatch:
    """A mismatch between Acoustic Notes and Mic Setup sheets."""

    knee: str
    maneuver: str
    field: str
    acoustic_notes_value: Any
    mic_setup_value: Any


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


def _cross_validate_sheets(
    acoustic_notes_file_name: str,
    acoustic_notes_mics: dict[int, MicrophonePosition],
    mic_setup: MicSetupData,
    knee: str,
    maneuver: str,
) -> list[LegendMismatch]:
    """Compare overlapping fields between Acoustic Notes and Mic Setup.

    Only compares when both sheets have non-empty values for a field.

    Returns:
        List of LegendMismatch instances for each detected disagreement.
    """
    mismatches: list[LegendMismatch] = []

    # Compare file names (both non-empty)
    if acoustic_notes_file_name and mic_setup.file_name:
        if acoustic_notes_file_name != mic_setup.file_name:
            mismatches.append(LegendMismatch(
                knee=knee,
                maneuver=maneuver,
                field="file_name",
                acoustic_notes_value=acoustic_notes_file_name,
                mic_setup_value=mic_setup.file_name,
            ))

    # Compare microphone positions
    for mic_num in [1, 2, 3, 4]:
        an_mic = acoustic_notes_mics.get(mic_num)
        ms_mic = mic_setup.microphones.get(mic_num)
        if an_mic and ms_mic:
            if an_mic.patellar_position != ms_mic.patellar_position:
                mismatches.append(LegendMismatch(
                    knee=knee,
                    maneuver=maneuver,
                    field=f"mic_{mic_num}_patellar_position",
                    acoustic_notes_value=an_mic.patellar_position,
                    mic_setup_value=ms_mic.patellar_position,
                ))
            if an_mic.laterality != ms_mic.laterality:
                mismatches.append(LegendMismatch(
                    knee=knee,
                    maneuver=maneuver,
                    field=f"mic_{mic_num}_laterality",
                    acoustic_notes_value=an_mic.laterality,
                    mic_setup_value=ms_mic.laterality,
                ))

    return mismatches


def get_acoustics_metadata(
    metadata_file_path: str,
    scripted_maneuver: Literal["walk", "sit_to_stand", "flexion_extension"],
    knee: Literal["left", "right"],
    acoustics_sheet_name: str = "Acoustic Notes",
    study_name: str = "AOA",
) -> tuple[AcousticsFileMetadata, list[LegendMismatch]]:
    """Gets acoustics metadata from legend file using dual-source fallback.

    Primary source is the Acoustic Notes sheet. When fields are missing
    (NaN/empty), values are filled from the study-specific fallback sheet
    (e.g. Mic Setup for AOA). When both sheets have values for the same
    field, cross-validation detects mismatches.

    Args:
        metadata_file_path: Path to the Excel file containing the acoustics
                           file legend.
        scripted_maneuver: The scripted maneuver to filter by.
        knee: The knee laterality to filter by.
        acoustics_sheet_name: Name of the Acoustic Notes sheet.
        study_name: Study identifier for dispatch to study-specific parsing.

    Returns:
        Tuple of (AcousticsFileMetadata, list[LegendMismatch]).
        The mismatch list is empty when sheets agree or only one source
        has data.
    """
    from src.studies import get_study_config

    study_config = get_study_config(study_name)

    # --- Source 1: Acoustic Notes (existing logic) ---
    metadata_df: pd.DataFrame = pd.read_excel(
        metadata_file_path, sheet_name=acoustics_sheet_name, header=None
    )

    table_start_row: int = find_knee_table_start(metadata_df, knee)
    knee_metadata_df: pd.DataFrame = extract_knee_metadata_table(
        metadata_df, table_start_row
    )
    knee_metadata_df: pd.DataFrame = normalize_maneuver_column(knee_metadata_df)
    maneuver_metadata_df: pd.DataFrame = filter_by_maneuver(
        knee_metadata_df, scripted_maneuver
    )
    file_name, notes = extract_file_name_and_notes(maneuver_metadata_df)
    microphones, microphone_notes = extract_microphone_positions(
        maneuver_metadata_df
    )

    # --- Source 2: Study-specific fallback (e.g. Mic Setup for AOA) ---
    mic_setup: Optional[MicSetupData] = None
    mismatches: list[LegendMismatch] = []

    try:
        mic_setup = study_config.parse_legend_fallback(
            metadata_file_path, scripted_maneuver, knee,
        )
    except Exception as exc:
        logger.warning(
            f"Could not parse fallback legend from {metadata_file_path}: {exc}"
        )

    # --- Cross-validation ---
    if mic_setup is not None:
        mismatches = _cross_validate_sheets(
            file_name, microphones, mic_setup, knee, scripted_maneuver,
        )
        for mm in mismatches:
            logger.warning(
                f"Legend mismatch ({mm.knee} {mm.maneuver}): "
                f"{mm.field} — Acoustic Notes='{mm.acoustic_notes_value}', "
                f"Mic Setup='{mm.mic_setup_value}'"
            )

    # --- Fallback: fill missing Acoustic Notes fields from fallback ---
    if mic_setup is not None:
        if not file_name and mic_setup.file_name:
            logger.info(
                f"Fallback: using file_name from fallback sheet: "
                f"'{mic_setup.file_name}'"
            )
            file_name = mic_setup.file_name

        if not microphones and mic_setup.microphones:
            logger.info("Fallback: using microphone positions from fallback sheet.")
            microphones = mic_setup.microphones

    # --- Build extra fields from fallback ---
    serial_number = "unknown"
    date_of_recording = None
    if mic_setup is not None:
        serial_number = mic_setup.serial_number
        date_of_recording = mic_setup.date_of_recording

    acoustics_metadata = AcousticsFileMetadata(
        scripted_maneuver=scripted_maneuver,
        knee=knee,
        study=study_name,
        study_id=0,
        file_name=file_name,
        microphones=microphones,
        audio_serial_number=serial_number,
        date_of_recording=date_of_recording if date_of_recording else datetime.min,
        audio_notes=(
            "; ".join(microphone_notes.values())
            if microphone_notes
            else notes
        ),
    )

    return acoustics_metadata, mismatches
