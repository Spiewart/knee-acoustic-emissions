import pandas as pd
import pytest

from models import MicrophonePosition
from parse_acoustic_file_legend import (
    extract_file_name_and_notes,
    extract_knee_metadata_table,
    extract_microphone_positions,
    filter_by_maneuver,
    find_knee_table_start,
    get_acoustics_metadata,
    normalize_maneuver_column,
)


def test_get_acoustics_metadata(sample_acoustic_legend_file) -> None:
    """Test parsing acoustics metadata from a sample Excel file legend."""

    metadata = get_acoustics_metadata(
        metadata_file_path=str(sample_acoustic_legend_file),
        scripted_maneuver="walk",
        knee="left",
    )

    assert metadata.scripted_maneuver == "walk"
    assert metadata.knee == "left"
    assert metadata.file_name == "HP_W11.2-5-20240126_135702"
    assert len(metadata.microphones) == 4
    assert metadata.microphones[1].patellar_position == "Infrapatellar"
    assert metadata.microphones[1].laterality == "Lateral"
    assert metadata.microphones[2].patellar_position == "Infrapatellar"
    assert metadata.microphones[2].laterality == "Medial"
    assert metadata.microphones[3].patellar_position == "Suprapatellar"
    assert metadata.microphones[3].laterality == "Medial"
    assert metadata.microphones[4].patellar_position == "Suprapatellar"
    assert metadata.microphones[4].laterality == "Lateral"
    assert metadata.notes is None
    assert metadata.microphone_notes is None
    assert metadata.speed is None


def test_get_acoustics_metadata_right_knee(
    sample_acoustic_legend_file,
) -> None:
    """Test parsing acoustics metadata for right knee."""

    metadata = get_acoustics_metadata(
        metadata_file_path=str(sample_acoustic_legend_file),
        scripted_maneuver="walk",
        knee="right",
    )

    assert metadata.scripted_maneuver == "walk"
    assert metadata.knee == "right"
    assert metadata.file_name == "HP_W12.2-5-20240126_135802"
    assert len(metadata.microphones) == 4


def test_get_acoustics_metadata_flexion_extension(
    sample_acoustic_legend_file,
) -> None:
    """Test parsing acoustics metadata for flexion-extension maneuver."""

    metadata = get_acoustics_metadata(
        metadata_file_path=str(sample_acoustic_legend_file),
        scripted_maneuver="flexion_extension",
        knee="left",
    )

    assert metadata.scripted_maneuver == "flexion_extension"
    assert metadata.knee == "left"
    assert metadata.file_name == "HP_W11.2-1-20240126_135704"
    assert len(metadata.microphones) == 4


def test_get_acoustics_metadata_sit_to_stand(
    sample_acoustic_legend_file,
) -> None:
    """Test parsing acoustics metadata for sit-to-stand maneuver."""

    metadata = get_acoustics_metadata(
        metadata_file_path=str(sample_acoustic_legend_file),
        scripted_maneuver="sit_to_stand",
        knee="left",
    )

    assert metadata.scripted_maneuver == "sit_to_stand"
    assert metadata.knee == "left"
    assert metadata.file_name == "HP_W11.2-3-20240126_135706"
    assert len(metadata.microphones) == 4


def test_find_knee_table_start_left_knee(
    sample_acoustic_legend_file,
) -> None:
    """Test finding the start row for left knee data."""
    metadata_df = pd.read_excel(
        sample_acoustic_legend_file,
        sheet_name="Acoustic Notes",
        header=None,
    )

    start_row = find_knee_table_start(metadata_df, "left")

    # Verify it returns an integer and is reasonable
    assert isinstance(start_row, int)
    assert start_row > 0


def test_find_knee_table_start_right_knee(
    sample_acoustic_legend_file,
) -> None:
    """Test finding the start row for right knee data."""
    metadata_df = pd.read_excel(
        sample_acoustic_legend_file,
        sheet_name="Acoustic Notes",
        header=None,
    )

    start_row = find_knee_table_start(metadata_df, "right")

    # Verify it returns an integer
    assert isinstance(start_row, int)
    assert start_row > 0


def test_find_knee_table_start_not_found(sample_acoustic_legend_file) -> None:
    """Test that finding non-existent knee raises ValueError."""
    metadata_df = pd.read_excel(
        sample_acoustic_legend_file,
        sheet_name="Acoustic Notes",
        header=None,
    )

    with pytest.raises(ValueError, match="No data found"):
        find_knee_table_start(metadata_df, "center")


def test_extract_knee_metadata_table(
    sample_acoustic_legend_file,
) -> None:
    """Test extracting knee metadata table from full DataFrame."""
    metadata_df = pd.read_excel(
        sample_acoustic_legend_file,
        sheet_name="Acoustic Notes",
        header=None,
    )
    start_row = find_knee_table_start(metadata_df, "left")

    knee_metadata_df = extract_knee_metadata_table(metadata_df, start_row)

    # Verify the shape
    assert len(knee_metadata_df) == 12  # 13 rows minus header
    assert "Maneuvers" in knee_metadata_df.columns
    assert "File Name" in knee_metadata_df.columns
    assert "Microphone" in knee_metadata_df.columns


def test_normalize_maneuver_column() -> None:
    """Test normalizing the Maneuvers column."""
    data = {
        "Maneuvers": [
            "Walk (slow,medium, fast)",
            None,
            None,
            None,
            "Flexion - Extension",
            None,
            None,
            None,
        ],
        "Other": [1, 2, 3, 4, 5, 6, 7, 8],
    }
    df = pd.DataFrame(data)

    result = normalize_maneuver_column(df)

    # Check that "-" was replaced with space in "Flexion - Extension"
    assert "Flexion - Extension" not in result["Maneuvers"].values
    assert "Flexion   Extension" in result["Maneuvers"].values or \
           "Flexion Extension" in result["Maneuvers"].values
    # Check that ffill worked
    assert result["Maneuvers"].iloc[1] == result["Maneuvers"].iloc[0]
    assert result["Maneuvers"].iloc[2] == result["Maneuvers"].iloc[0]


def test_filter_by_maneuver() -> None:
    """Test filtering metadata by maneuver."""
    data = {
        "Maneuvers": [
            "Walk (slow, medium, fast)",
            "Walk (slow, medium, fast)",
            "Walk (slow, medium, fast)",
            "Flexion Extension",
            "Flexion Extension",
        ],
        "Microphone": [1, 2, 3, 1, 2],
        "File Name": ["file1"] * 5,
    }
    df = pd.DataFrame(data)

    result = filter_by_maneuver(df, "walk")

    # Should only have walk maneuver rows
    assert len(result) == 3
    assert all("walk" in m.lower() for m in result["Maneuvers"].values)


def test_filter_by_maneuver_sit_to_stand() -> None:
    """Test filtering by sit_to_stand maneuver."""
    data = {
        "Maneuvers": [
            "Walk (slow, medium, fast)",
            "Sit to Stand",
            "Sit to Stand",
            "Flexion Extension",
        ],
        "Microphone": [1, 1, 2, 1],
        "File Name": ["file1"] * 4,
    }
    df = pd.DataFrame(data)

    result = filter_by_maneuver(df, "sit_to_stand")

    # Should only have sit to stand rows
    assert len(result) == 2


def test_extract_microphone_positions() -> None:
    """Test extracting microphone positions from maneuver data."""
    data = {
        "Microphone": [1, 2, 3, 4],
        "Patellar Position": [
            "Infrapatellar",
            "Infrapatellar",
            "Suprapatellar",
            "Suprapatellar",
        ],
        "Laterality": ["Lateral", "Medial", "Medial", "Lateral"],
        "Notes": ["note1", None, None, "note4"],
    }
    df = pd.DataFrame(data)

    microphones, microphone_notes = extract_microphone_positions(df)

    # Verify all microphones extracted
    assert len(microphones) == 4
    assert all(isinstance(m, MicrophonePosition) for m in microphones.values())

    # Verify microphone positions
    assert microphones[1].patellar_position == "Infrapatellar"
    assert microphones[1].laterality == "Lateral"
    assert microphones[4].laterality == "Lateral"

    # Verify notes extraction
    assert len(microphone_notes) == 2
    assert microphone_notes[1] == "note1"
    assert microphone_notes[4] == "note4"


def test_extract_file_name_and_notes() -> None:
    """Test extracting file name and notes."""
    data = {
        "File Name": ["test_file.wav", "other.wav"],
        "Notes": ["test note", None],
    }
    df = pd.DataFrame(data)

    file_name, notes = extract_file_name_and_notes(df)

    assert file_name == "test_file.wav"
    assert notes == "test note"


def test_extract_file_name_and_notes_no_notes() -> None:
    """Test extracting file name with no notes."""
    data = {
        "File Name": ["test_file.wav", "other.wav"],
        "Notes": [None, None],
    }
    df = pd.DataFrame(data)

    file_name, notes = extract_file_name_and_notes(df)

    assert file_name == "test_file.wav"
    assert notes is None
