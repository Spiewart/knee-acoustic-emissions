"""AOA-specific Mic Setup sheet parser.

The AOA study uses a fixed-layout "Mic Setup" sheet in the acoustics legend
Excel file. This module handles parsing that sheet's specific structure:
row/column positions, landmark strings, and field locations.

Other studies may have different legend formats or no Mic Setup sheet at all.
"""

import logging
import re
from datetime import datetime
from typing import Any, Literal, Optional

import pandas as pd

from src.audio.parsers import MicSetupData
from src.models import MicrophonePosition

logger = logging.getLogger(__name__)


def _find_mic_setup_knee_data_row(
    df: pd.DataFrame,
    knee: Literal["left", "right"],
) -> int:
    """Find the first data row for file info in the Mic Setup sheet.

    The sheet has fixed landmarks: "Left knee" and "Right Knee" mark the
    start of each knee's file table. The header row is immediately after
    the landmark, and data rows start one row after that.

    Args:
        df: Full Mic Setup DataFrame (header=None).
        knee: Which knee to find.

    Returns:
        Row index of the first data row (Walk) for the specified knee.

    Raises:
        ValueError: If the knee landmark is not found.
    """
    if knee == "left":
        landmark = "Left knee"
    else:
        landmark = "Right Knee"

    matches = df.index[
        df.iloc[:, 0].astype(str).str.strip().str.lower()
        == landmark.lower()
    ].tolist()
    if not matches:
        raise ValueError(
            f"Mic Setup sheet: no '{landmark}' landmark found."
        )
    # Header row is landmark + 1, first data row is landmark + 2
    return matches[0] + 2


def _match_maneuver_row(
    df: pd.DataFrame,
    data_start_row: int,
    scripted_maneuver: Literal["walk", "sit_to_stand", "flexion_extension"],
) -> int:
    """Find the row matching a maneuver in the Mic Setup file table.

    Searches 3 consecutive rows (Walk, FE, STS) starting from
    data_start_row. Uses the same fuzzy match as Acoustic Notes:
    underscores replaced by spaces, case-insensitive contains.

    Args:
        df: Full Mic Setup DataFrame.
        data_start_row: First data row for this knee.
        scripted_maneuver: Maneuver to search for.

    Returns:
        The row index matching the maneuver.

    Raises:
        ValueError: If no matching maneuver is found.
    """
    search_str = scripted_maneuver.replace("_", " ")
    # The Maneuver column is col 4 in the Mic Setup sheet
    maneuver_col = 4
    for row_idx in range(data_start_row, data_start_row + 3):
        if row_idx >= len(df):
            break
        cell = df.iloc[row_idx, maneuver_col]
        if pd.notna(cell):
            # Normalize cell text: strip dashes and collapse whitespace
            # (same logic as Acoustic Notes normalize_maneuver_column)
            cell_normalized = str(cell).lower().replace("-", " ")
            cell_normalized = re.sub(r"\s+", " ", cell_normalized).strip()
            if search_str in cell_normalized:
                return row_idx

    raise ValueError(
        f"Mic Setup sheet: no row matching maneuver '{scripted_maneuver}' "
        f"in rows {data_start_row}-{data_start_row + 2}."
    )


def _parse_mic_setup_date(header_value: Any) -> Optional[datetime]:
    """Parse date of recording from the Mic Setup header row.

    The header cell contains either "Date of Recording: MM/DD/YYYY" or
    just "Date of Recording" (no value). Returns None if no date is found.
    """
    if pd.isna(header_value):
        return None
    text = str(header_value).strip()
    if ":" in text:
        date_str = text.split(":", 1)[1].strip()
        if date_str:
            try:
                return datetime.strptime(date_str, "%m/%d/%Y")
            except ValueError:
                logger.warning(
                    f"Mic Setup: could not parse date '{date_str}'."
                )
    return None


def _parse_mic_setup_study_id(header_value: Any) -> Optional[int]:
    """Parse study ID from the Mic Setup header row.

    The header cell contains either "Study ID: 1013" or just "Study ID".
    """
    if pd.isna(header_value):
        return None
    text = str(header_value).strip()
    if ":" in text:
        id_str = text.split(":", 1)[1].strip()
        if id_str:
            try:
                return int(float(id_str))
            except (ValueError, TypeError):
                return None
    return None


def parse_aoa_mic_setup_sheet(
    metadata_file_path: str,
    scripted_maneuver: Literal["walk", "sit_to_stand", "flexion_extension"],
    knee: Literal["left", "right"],
    sheet_name: str = "Mic Setup",
) -> MicSetupData:
    """Parse the AOA Mic Setup sheet for a specific knee and maneuver.

    The Mic Setup sheet has a fixed layout:
    - Row 0: Study ID and Date of Recording (cols 0 and 2)
    - Row 1: Left/Right knee board IDs (cols 0 and 3)
    - Rows 2-6: Mic positions — left knee cols 0-2, right knee cols 3-5
    - Row 7: "Left knee" landmark
    - Row 8: File data header (File Name, File Size, Serial, Timestamp, Maneuver, Notes)
    - Rows 9-11: Left knee data (Walk, FE, STS)
    - Row 14: "Right Knee" landmark
    - Row 15: File data header
    - Rows 16-18: Right knee data (Walk, FE, STS)

    Args:
        metadata_file_path: Path to the legend Excel file.
        scripted_maneuver: Maneuver to extract.
        knee: Knee laterality.
        sheet_name: Name of the Mic Setup sheet.

    Returns:
        MicSetupData with all available fields populated.

    Raises:
        ValueError: If the sheet structure doesn't match expectations.
    """
    df = pd.read_excel(metadata_file_path, sheet_name=sheet_name, header=None)

    # --- Header fields (row 0) ---
    date_of_recording = _parse_mic_setup_date(df.iloc[0, 2])
    study_id = _parse_mic_setup_study_id(df.iloc[0, 0])

    # --- Microphone positions (rows 3-6, fixed) ---
    if knee == "left":
        mic_col_offset = 0  # cols 0, 1, 2
    else:
        mic_col_offset = 3  # cols 3, 4, 5

    microphones: dict[int, MicrophonePosition] = {}
    for i in range(4):  # Mics 1-4 at rows 3-6
        row_idx = 3 + i
        mic_num = int(df.iloc[row_idx, mic_col_offset])
        patellar = str(df.iloc[row_idx, mic_col_offset + 1]).strip()
        # Col header is "Medial / Lateral" but values are "Medial" or "Lateral"
        laterality = str(df.iloc[row_idx, mic_col_offset + 2]).strip()
        microphones[mic_num] = MicrophonePosition(
            patellar_position=patellar,
            laterality=laterality,
        )

    # --- File data row ---
    data_start_row = _find_mic_setup_knee_data_row(df, knee)
    maneuver_row = _match_maneuver_row(df, data_start_row, scripted_maneuver)

    raw_file_name = df.iloc[maneuver_row, 0]
    file_name = "" if pd.isna(raw_file_name) else str(raw_file_name).strip()

    raw_file_size = df.iloc[maneuver_row, 1]
    file_size_mb = float(raw_file_size) if pd.notna(raw_file_size) else None

    raw_serial = df.iloc[maneuver_row, 2]
    serial_number = (
        str(raw_serial).strip() if pd.notna(raw_serial) else "unknown"
    )

    raw_timestamp = df.iloc[maneuver_row, 3]
    timestamp = str(raw_timestamp).strip() if pd.notna(raw_timestamp) else None

    raw_notes = df.iloc[maneuver_row, 5]
    notes = str(raw_notes).strip() if pd.notna(raw_notes) else None

    return MicSetupData(
        file_name=file_name,
        serial_number=serial_number,
        file_size_mb=file_size_mb,
        timestamp=timestamp,
        date_of_recording=date_of_recording,
        microphones=microphones,
        notes=notes,
    )
