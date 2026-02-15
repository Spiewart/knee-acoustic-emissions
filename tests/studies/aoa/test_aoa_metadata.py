import pandas as pd
import pytest

from src.audio.parsers import (
    LegendMismatch,
    _acoustic_notes_is_filled,
    extract_file_name_and_notes,
    extract_knee_metadata_table,
    extract_microphone_positions,
    filter_by_maneuver,
    find_knee_table_start,
    get_acoustics_metadata,
    normalize_maneuver_column,
)
from src.models import MicrophonePosition
from src.studies.aoa.legend import parse_aoa_mic_setup_sheet


def test_get_acoustics_metadata(fake_participant_directory) -> None:
    """Test parsing acoustics metadata from a sample Excel file legend."""

    legend_path = fake_participant_directory["legend_file"]

    metadata, mismatches = get_acoustics_metadata(
        metadata_file_path=str(legend_path),
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
    assert mismatches == []


def test_get_acoustics_metadata_right_knee(
    fake_participant_directory,
) -> None:
    """Test parsing acoustics metadata for right knee."""

    legend_path = fake_participant_directory["legend_file"]

    metadata, mismatches = get_acoustics_metadata(
        metadata_file_path=str(legend_path),
        scripted_maneuver="walk",
        knee="right",
    )

    assert metadata.scripted_maneuver == "walk"
    assert metadata.knee == "right"
    assert metadata.file_name == "HP_W12.2-5-20240126_135802"
    assert len(metadata.microphones) == 4
    assert mismatches == []


def test_get_acoustics_metadata_flexion_extension(
    fake_participant_directory,
) -> None:
    """Test parsing acoustics metadata for flexion-extension maneuver."""

    legend_path = fake_participant_directory["legend_file"]

    metadata, mismatches = get_acoustics_metadata(
        metadata_file_path=str(legend_path),
        scripted_maneuver="flexion_extension",
        knee="left",
    )

    assert metadata.scripted_maneuver == "flexion_extension"
    assert metadata.knee == "left"
    assert metadata.file_name == "HP_W11.2-1-20240126_135704"
    assert len(metadata.microphones) == 4
    assert mismatches == []


def test_get_acoustics_metadata_sit_to_stand(
    fake_participant_directory,
) -> None:
    """Test parsing acoustics metadata for sit-to-stand maneuver."""

    legend_path = fake_participant_directory["legend_file"]

    metadata, mismatches = get_acoustics_metadata(
        metadata_file_path=str(legend_path),
        scripted_maneuver="sit_to_stand",
        knee="left",
    )

    assert metadata.scripted_maneuver == "sit_to_stand"
    assert metadata.knee == "left"
    assert metadata.file_name == "HP_W11.2-3-20240126_135706"
    assert len(metadata.microphones) == 4
    assert mismatches == []


def test_find_knee_table_start_left_knee(
    fake_participant_directory,
) -> None:
    """Test finding the start row for left knee data."""
    legend_path = fake_participant_directory["legend_file"]
    metadata_df = pd.read_excel(
        legend_path,
        sheet_name="Acoustic Notes",
        header=None,
    )

    start_row = find_knee_table_start(metadata_df, "left")

    # Verify it returns an integer and is reasonable
    assert isinstance(start_row, int)
    assert start_row > 0


def test_find_knee_table_start_right_knee(
    fake_participant_directory,
) -> None:
    """Test finding the start row for right knee data."""
    legend_path = fake_participant_directory["legend_file"]
    metadata_df = pd.read_excel(
        legend_path,
        sheet_name="Acoustic Notes",
        header=None,
    )

    start_row = find_knee_table_start(metadata_df, "right")

    # Verify it returns an integer
    assert isinstance(start_row, int)
    assert start_row > 0


def test_find_knee_table_start_not_found(fake_participant_directory) -> None:
    """Test that finding non-existent knee raises ValueError."""
    legend_path = fake_participant_directory["legend_file"]
    metadata_df = pd.read_excel(
        legend_path,
        sheet_name="Acoustic Notes",
        header=None,
    )

    with pytest.raises(ValueError, match="No data found"):
        find_knee_table_start(metadata_df, "center")


def test_extract_knee_metadata_table(
    fake_participant_directory,
) -> None:
    """Test extracting knee metadata table from full DataFrame."""
    legend_path = fake_participant_directory["legend_file"]
    metadata_df = pd.read_excel(
        legend_path,
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
    assert "Flexion   Extension" in result["Maneuvers"].values or "Flexion Extension" in result["Maneuvers"].values
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


# ── Mic Setup parser tests ──────────────────────────────────────────


class TestParseMicSetupSheet:
    """Tests for parse_aoa_mic_setup_sheet() directly."""

    def test_parse_left_knee_walk(self, fake_participant_directory):
        """Parse Mic Setup for left knee Walk — basic happy path."""
        legend_path = str(fake_participant_directory["legend_file"])

        result = parse_aoa_mic_setup_sheet(legend_path, "walk", "left")

        assert result.file_name == "HP_W11.2-5-20240126_135702"
        assert result.serial_number == "HP_W11.2-5"
        assert result.file_size_mb == pytest.approx(100.5)
        assert result.timestamp == "13:57:02"
        assert len(result.microphones) == 4
        assert result.microphones[1].patellar_position == "Infrapatellar"
        assert result.microphones[1].laterality == "Lateral"

    def test_parse_right_knee_walk(self, fake_participant_directory):
        """Parse Mic Setup for right knee Walk."""
        legend_path = str(fake_participant_directory["legend_file"])

        result = parse_aoa_mic_setup_sheet(legend_path, "walk", "right")

        assert result.file_name == "HP_W12.2-5-20240126_135802"
        assert result.serial_number == "HP_W12.2-5"
        assert result.file_size_mb == pytest.approx(98.3)
        assert len(result.microphones) == 4

    def test_parse_sit_to_stand(self, fake_participant_directory):
        """Parse Mic Setup for STS maneuver."""
        legend_path = str(fake_participant_directory["legend_file"])

        result = parse_aoa_mic_setup_sheet(legend_path, "sit_to_stand", "left")

        assert result.file_name == "HP_W11.2-3-20240126_135706"
        assert result.serial_number == "HP_W11.2-5"

    def test_parse_flexion_extension(self, fake_participant_directory):
        """Parse Mic Setup for FE maneuver."""
        legend_path = str(fake_participant_directory["legend_file"])

        result = parse_aoa_mic_setup_sheet(
            legend_path,
            "flexion_extension",
            "left",
        )

        assert result.file_name == "HP_W11.2-1-20240126_135704"

    def test_date_of_recording_parsed(self, fake_participant_directory):
        """Date of recording should be parsed from header."""
        legend_path = str(fake_participant_directory["legend_file"])
        from datetime import datetime

        result = parse_aoa_mic_setup_sheet(legend_path, "walk", "left")

        assert result.date_of_recording == datetime(2024, 1, 26)

    def test_all_mic_positions_extracted(self, fake_participant_directory):
        """All 4 microphone positions should be populated."""
        legend_path = str(fake_participant_directory["legend_file"])

        result = parse_aoa_mic_setup_sheet(legend_path, "walk", "left")

        assert set(result.microphones.keys()) == {1, 2, 3, 4}
        # Verify specific positions
        assert result.microphones[2].patellar_position == "Infrapatellar"
        assert result.microphones[2].laterality == "Medial"
        assert result.microphones[3].patellar_position == "Suprapatellar"
        assert result.microphones[3].laterality == "Medial"
        assert result.microphones[4].patellar_position == "Suprapatellar"
        assert result.microphones[4].laterality == "Lateral"


# ── Mic Setup extra fields in get_acoustics_metadata() ──────────────


class TestMicSetupExtraFields:
    """Tests that Mic Setup-exclusive fields populate the metadata."""

    def test_serial_number_populated(self, fake_participant_directory):
        """audio_serial_number should come from Mic Setup."""
        legend_path = str(fake_participant_directory["legend_file"])

        metadata, _ = get_acoustics_metadata(
            legend_path,
            "walk",
            "left",
        )

        assert metadata.audio_serial_number == "HP_W11.2-5"

    def test_date_of_recording_populated(self, fake_participant_directory):
        """date_of_recording should come from Mic Setup header."""
        from datetime import datetime

        legend_path = str(fake_participant_directory["legend_file"])

        metadata, _ = get_acoustics_metadata(
            legend_path,
            "walk",
            "left",
        )

        assert metadata.date_of_recording == datetime(2024, 1, 26)


# ── Fallback tests ──────────────────────────────────────────────────


def _create_legend_with_blank_acoustic_notes(tmp_path):
    """Create a legend where Acoustic Notes has blank file names but Mic Setup is populated."""
    excel_path = tmp_path / "legend_blank_acoustic.xlsx"

    # Acoustic Notes with blank file names
    acoustic_data = [
        ["L Knee", None, None, None, None, None],
        ["Maneuvers", "File Name", "Microphone", "Patellar Position", "Laterality", "Notes"],
        ["Walk (slow,medium, fast)", None, 1, "Infrapatellar", "Lateral", None],
        [None, None, 2, "Infrapatellar", "Medial", None],
        [None, None, 3, "Suprapatellar", "Medial", None],
        [None, None, 4, "Suprapatellar", "Lateral", None],
        ["Flexion - Extension", None, 1, "Infrapatellar", "Lateral", None],
        [None, None, 2, "Infrapatellar", "Medial", None],
        [None, None, 3, "Suprapatellar", "Medial", None],
        [None, None, 4, "Suprapatellar", "Lateral", None],
        ["Sit - to - Stand", None, 1, "Infrapatellar", "Lateral", None],
        [None, None, 2, "Infrapatellar", "Medial", None],
        [None, None, 3, "Suprapatellar", "Medial", None],
        [None, None, 4, "Suprapatellar", "Lateral", None],
        [None, None, None, None, None, None],
        ["R Knee", None, None, None, None, None],
        ["Maneuvers", "File Name", "Microphone", "Patellar Position", "Laterality", "Notes"],
        ["Walk (slow,medium, fast)", None, 1, "Infrapatellar", "Lateral", None],
        [None, None, 2, "Infrapatellar", "Medial", None],
        [None, None, 3, "Suprapatellar", "Medial", None],
        [None, None, 4, "Suprapatellar", "Lateral", None],
        ["Flexion - Extension", None, 1, "Infrapatellar", "Lateral", None],
        [None, None, 2, "Infrapatellar", "Medial", None],
        [None, None, 3, "Suprapatellar", "Medial", None],
        [None, None, 4, "Suprapatellar", "Lateral", None],
        ["Sit - to - Stand", None, 1, "Infrapatellar", "Lateral", None],
        [None, None, 2, "Infrapatellar", "Medial", None],
        [None, None, 3, "Suprapatellar", "Medial", None],
        [None, None, 4, "Suprapatellar", "Lateral", None],
    ]

    # Mic Setup with populated file names
    mic_setup_data = [
        ["Study ID: 1013", None, "Date of Recording: 02/01/2024", None, None, None, None, None],
        ["Left Knee HP_W11.2-5", None, None, "Right Knee: HP_W11.2-1", None, None, None, None],
        [
            "Microphones",
            "Patellar Position",
            "Medial / Lateral",
            "Microphones",
            "Patellar Position",
            "Medial / Lateral",
            None,
            None,
        ],
        [1, "Infrapatellar", "Lateral", 1, "Infrapatellar", "Lateral", None, None],
        [2, "Infrapatellar", "Medial", 2, "Infrapatellar", "Medial", None, None],
        [3, "Suprapatellar", "Medial", 3, "Suprapatellar", "Medial", None, None],
        [4, "Suprapatellar", "Lateral", 4, "Suprapatellar", "Lateral", None, None],
        ["Left knee", None, None, None, None, None, None, None],
        ["File Name", "File Size (mb)", "Audio Board Serial Number", "Timestamp", "Maneuver", "Notes", None, None],
        [
            "HP_W11.2-5-20240201_112407",
            123.7,
            "HP_W11.2-5",
            "11:24:07",
            "Walk (slow,medium, fast) 80 seconds each speed",
            None,
            None,
            None,
        ],
        ["HP_W11.2-5-20240201_113516", 33.6, "HP_W11.2-5", "11:35:16", "Flexion - Extension", None, None, None],
        ["HP_W11.2-5-20240201_114038", 32.3, "HP_W11.2-5", "11:40:38", "Sit - to - Stand", None, None, None],
        [None] * 8,
        [None] * 8,
        ["Right Knee", None, None, None, None, None, None, None],
        ["File Name", "File Size (mb)", "Audio Board Serial Number", "Timestamp", "Maneuver", "Notes", None, None],
        [
            "HP_W11.2-1-20240201_112411",
            122.2,
            "HP_W11.2-1",
            "11:24:11",
            "Walk (slow,medium, fast) 80 seconds each speed",
            None,
            None,
            None,
        ],
        ["HP_W11.2-1-20240201_113517", 32.2, "HP_W11.2-1", "11:35:17", "Flexion - Extension", None, None, None],
        ["HP_W11.2-1-20240201_114035", 34, "HP_W11.2-1", "11:40:35", "Sit - to - Stand", None, None, None],
        *[[None] * 8 for _ in range(7)],
        [None, None, None, None, None, None, "Red Device", "HP_W11.2-5"],
        [None, None, None, None, None, None, "White Device", "HP_W11.2-1"],
    ]

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        pd.DataFrame(acoustic_data).to_excel(
            writer,
            sheet_name="Acoustic Notes",
            index=False,
            header=False,
        )
        pd.DataFrame(mic_setup_data).to_excel(
            writer,
            sheet_name="Mic Setup",
            index=False,
            header=False,
        )

    return excel_path


class TestMicSetupFallback:
    """Tests that Mic Setup fills in missing Acoustic Notes data."""

    def test_file_name_fallback_when_acoustic_notes_blank(self, tmp_path):
        """When Acoustic Notes has blank file names, Mic Setup fills them."""
        legend_path = _create_legend_with_blank_acoustic_notes(tmp_path)

        metadata, mismatches = get_acoustics_metadata(
            str(legend_path),
            "walk",
            "left",
        )

        # File name came from Mic Setup
        assert metadata.file_name == "HP_W11.2-5-20240201_112407"
        # No mismatches (AN was blank, so no comparison)
        assert mismatches == []

    def test_file_name_fallback_right_knee(self, tmp_path):
        """Fallback works for right knee too."""
        legend_path = _create_legend_with_blank_acoustic_notes(tmp_path)

        metadata, _ = get_acoustics_metadata(
            str(legend_path),
            "walk",
            "right",
        )

        assert metadata.file_name == "HP_W11.2-1-20240201_112411"

    def test_file_name_fallback_sit_to_stand(self, tmp_path):
        """Fallback works for STS maneuver."""
        legend_path = _create_legend_with_blank_acoustic_notes(tmp_path)

        metadata, _ = get_acoustics_metadata(
            str(legend_path),
            "sit_to_stand",
            "left",
        )

        assert metadata.file_name == "HP_W11.2-5-20240201_114038"

    def test_serial_number_from_mic_setup(self, tmp_path):
        """Serial number should be populated from Mic Setup even when AN has data."""
        legend_path = _create_legend_with_blank_acoustic_notes(tmp_path)

        metadata, _ = get_acoustics_metadata(
            str(legend_path),
            "walk",
            "left",
        )

        assert metadata.audio_serial_number == "HP_W11.2-5"

    def test_date_of_recording_from_mic_setup(self, tmp_path):
        """Date of recording should come from Mic Setup header."""
        from datetime import datetime

        legend_path = _create_legend_with_blank_acoustic_notes(tmp_path)

        metadata, _ = get_acoustics_metadata(
            str(legend_path),
            "walk",
            "left",
        )

        assert metadata.date_of_recording == datetime(2024, 2, 1)

    def test_mic_positions_from_mic_setup_when_acoustic_notes_unfilled(self, tmp_path):
        """When Acoustic Notes has template defaults (no file name), mic
        positions should come from Mic Setup, even if the templates differ."""
        excel_path = tmp_path / "legend_unfilled_an.xlsx"

        # Acoustic Notes: blank file names, template mic positions that
        # DIFFER from Mic Setup (mic 3 & 4 swapped laterality)
        acoustic_data = [
            ["L Knee", None, None, None, None, None],
            ["Maneuvers", "File Name", "Microphone", "Patellar Position", "Laterality", "Notes"],
            ["Walk (slow,medium, fast)", None, 1, "Infrapatellar", "Lateral", None],
            [None, None, 2, "Infrapatellar", "Medial", None],
            [None, None, 3, "Suprapatellar", "Lateral", None],  # template default
            [None, None, 4, "Suprapatellar", "Medial", None],  # template default
            ["Flexion - Extension", None, 1, "Infrapatellar", "Lateral", None],
            [None, None, 2, "Infrapatellar", "Medial", None],
            [None, None, 3, "Suprapatellar", "Lateral", None],
            [None, None, 4, "Suprapatellar", "Medial", None],
            ["Sit - to - Stand", None, 1, "Infrapatellar", "Lateral", None],
            [None, None, 2, "Infrapatellar", "Medial", None],
            [None, None, 3, "Suprapatellar", "Lateral", None],
            [None, None, 4, "Suprapatellar", "Medial", None],
            [None, None, None, None, None, None],
            ["R Knee", None, None, None, None, None],
            ["Maneuvers", "File Name", "Microphone", "Patellar Position", "Laterality", "Notes"],
            ["Walk (slow,medium, fast)", None, 1, "Infrapatellar", "Lateral", None],
            [None, None, 2, "Infrapatellar", "Medial", None],
            [None, None, 3, "Suprapatellar", "Lateral", None],
            [None, None, 4, "Suprapatellar", "Medial", None],
            ["Flexion - Extension", None, 1, "Infrapatellar", "Lateral", None],
            [None, None, 2, "Infrapatellar", "Medial", None],
            [None, None, 3, "Suprapatellar", "Lateral", None],
            [None, None, 4, "Suprapatellar", "Medial", None],
            ["Sit - to - Stand", None, 1, "Infrapatellar", "Lateral", None],
            [None, None, 2, "Infrapatellar", "Medial", None],
            [None, None, 3, "Suprapatellar", "Lateral", None],
            [None, None, 4, "Suprapatellar", "Medial", None],
        ]

        # Mic Setup: mic 3 = Medial, mic 4 = Lateral (opposite of AN template)
        mic_setup_data = [
            ["Study ID: 1013", None, "Date of Recording: 02/01/2024", None, None, None, None, None],
            ["Left Knee HP_W11.2-5", None, None, "Right Knee: HP_W11.2-1", None, None, None, None],
            [
                "Microphones",
                "Patellar Position",
                "Medial / Lateral",
                "Microphones",
                "Patellar Position",
                "Medial / Lateral",
                None,
                None,
            ],
            [1, "Infrapatellar", "Lateral", 1, "Infrapatellar", "Lateral", None, None],
            [2, "Infrapatellar", "Medial", 2, "Infrapatellar", "Medial", None, None],
            [3, "Suprapatellar", "Medial", 3, "Suprapatellar", "Medial", None, None],
            [4, "Suprapatellar", "Lateral", 4, "Suprapatellar", "Lateral", None, None],
            ["Left knee", None, None, None, None, None, None, None],
            ["File Name", "File Size (mb)", "Audio Board Serial Number", "Timestamp", "Maneuver", "Notes", None, None],
            [
                "HP_W11.2-5-20240201_112407",
                123.7,
                "HP_W11.2-5",
                "11:24:07",
                "Walk (slow,medium, fast) 80 seconds each speed",
                None,
                None,
                None,
            ],
            ["HP_W11.2-5-20240201_113516", 33.6, "HP_W11.2-5", "11:35:16", "Flexion - Extension", None, None, None],
            ["HP_W11.2-5-20240201_114038", 32.3, "HP_W11.2-5", "11:40:38", "Sit - to - Stand", None, None, None],
            [None] * 8,
            [None] * 8,
            ["Right Knee", None, None, None, None, None, None, None],
            ["File Name", "File Size (mb)", "Audio Board Serial Number", "Timestamp", "Maneuver", "Notes", None, None],
            [
                "HP_W11.2-1-20240201_112411",
                122.2,
                "HP_W11.2-1",
                "11:24:11",
                "Walk (slow,medium, fast) 80 seconds each speed",
                None,
                None,
                None,
            ],
            ["HP_W11.2-1-20240201_113517", 32.2, "HP_W11.2-1", "11:35:17", "Flexion - Extension", None, None, None],
            ["HP_W11.2-1-20240201_114035", 34, "HP_W11.2-1", "11:40:35", "Sit - to - Stand", None, None, None],
            *[[None] * 8 for _ in range(7)],
            [None, None, None, None, None, None, "Red Device", "HP_W11.2-5"],
            [None, None, None, None, None, None, "White Device", "HP_W11.2-1"],
        ]

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            pd.DataFrame(acoustic_data).to_excel(
                writer,
                sheet_name="Acoustic Notes",
                index=False,
                header=False,
            )
            pd.DataFrame(mic_setup_data).to_excel(
                writer,
                sheet_name="Mic Setup",
                index=False,
                header=False,
            )

        metadata, mismatches = get_acoustics_metadata(
            str(excel_path),
            "walk",
            "left",
        )

        # No mismatches — AN is unfilled so template defaults are ignored
        assert mismatches == []

        # Mic positions should come from Mic Setup, NOT the AN template
        assert metadata.microphones[3].laterality == "Medial"  # Mic Setup value
        assert metadata.microphones[4].laterality == "Lateral"  # Mic Setup value


class TestAcousticNotesIsFilled:
    """Unit tests for the _acoustic_notes_is_filled() helper."""

    def test_empty_string_is_unfilled(self):
        assert _acoustic_notes_is_filled("") is False

    def test_whitespace_only_is_unfilled(self):
        assert _acoustic_notes_is_filled("   ") is False

    def test_real_filename_is_filled(self):
        assert _acoustic_notes_is_filled("HP_W11.2-5-20240201_112407") is True

    def test_none_like_is_unfilled(self):
        # In practice file_name is always str, but edge-case defense
        assert _acoustic_notes_is_filled("") is False


# ── Cross-sheet mismatch tests ──────────────────────────────────────


def _create_legend_with_mismatch(tmp_path):
    """Create a legend where Acoustic Notes and Mic Setup disagree on mic positions."""
    excel_path = tmp_path / "legend_mismatch.xlsx"

    # Acoustic Notes: mic 1 = Infrapatellar Lateral
    acoustic_data = [
        ["L Knee", None, None, None, None, None],
        ["Maneuvers", "File Name", "Microphone", "Patellar Position", "Laterality", "Notes"],
        ["Walk (slow,medium, fast)", "SAME_FILE_NAME", 1, "Infrapatellar", "Lateral", None],
        [None, None, 2, "Infrapatellar", "Medial", None],
        [None, None, 3, "Suprapatellar", "Medial", None],
        [None, None, 4, "Suprapatellar", "Lateral", None],
        ["Flexion - Extension", "FE_FILE", 1, "Infrapatellar", "Lateral", None],
        [None, None, 2, "Infrapatellar", "Medial", None],
        [None, None, 3, "Suprapatellar", "Medial", None],
        [None, None, 4, "Suprapatellar", "Lateral", None],
        ["Sit - to - Stand", "STS_FILE", 1, "Infrapatellar", "Lateral", None],
        [None, None, 2, "Infrapatellar", "Medial", None],
        [None, None, 3, "Suprapatellar", "Medial", None],
        [None, None, 4, "Suprapatellar", "Lateral", None],
        [None, None, None, None, None, None],
        ["R Knee", None, None, None, None, None],
        ["Maneuvers", "File Name", "Microphone", "Patellar Position", "Laterality", "Notes"],
        ["Walk (slow,medium, fast)", "R_FILE", 1, "Infrapatellar", "Lateral", None],
        [None, None, 2, "Infrapatellar", "Medial", None],
        [None, None, 3, "Suprapatellar", "Medial", None],
        [None, None, 4, "Suprapatellar", "Lateral", None],
        ["Flexion - Extension", "R_FE_FILE", 1, "Infrapatellar", "Lateral", None],
        [None, None, 2, "Infrapatellar", "Medial", None],
        [None, None, 3, "Suprapatellar", "Medial", None],
        [None, None, 4, "Suprapatellar", "Lateral", None],
        ["Sit - to - Stand", "R_STS_FILE", 1, "Infrapatellar", "Lateral", None],
        [None, None, 2, "Infrapatellar", "Medial", None],
        [None, None, 3, "Suprapatellar", "Medial", None],
        [None, None, 4, "Suprapatellar", "Lateral", None],
    ]

    # Mic Setup: mic 1 = Infrapatellar MEDIAL (mismatch!)
    mic_setup_data = [
        ["Study ID: 9999", None, "Date of Recording: 01/26/2024", None, None, None, None, None],
        ["Left Knee HP_W11.2-5", None, None, "Right Knee: HP_W12.2-5", None, None, None, None],
        [
            "Microphones",
            "Patellar Position",
            "Medial / Lateral",
            "Microphones",
            "Patellar Position",
            "Medial / Lateral",
            None,
            None,
        ],
        # Mic 1: MEDIAL instead of LATERAL (mismatch!)
        [1, "Infrapatellar", "Medial", 1, "Infrapatellar", "Lateral", None, None],
        [2, "Infrapatellar", "Medial", 2, "Infrapatellar", "Medial", None, None],
        [3, "Suprapatellar", "Medial", 3, "Suprapatellar", "Medial", None, None],
        [4, "Suprapatellar", "Lateral", 4, "Suprapatellar", "Lateral", None, None],
        ["Left knee", None, None, None, None, None, None, None],
        ["File Name", "File Size (mb)", "Audio Board Serial Number", "Timestamp", "Maneuver", "Notes", None, None],
        [
            "SAME_FILE_NAME",
            100.0,
            "HP_W11.2-5",
            "12:00:00",
            "Walk (slow,medium, fast) 80 seconds each speed",
            None,
            None,
            None,
        ],
        ["FE_FILE", 30.0, "HP_W11.2-5", "12:10:00", "Flexion - Extension", None, None, None],
        ["STS_FILE", 28.0, "HP_W11.2-5", "12:20:00", "Sit - to - Stand", None, None, None],
        [None] * 8,
        [None] * 8,
        ["Right Knee", None, None, None, None, None, None, None],
        ["File Name", "File Size (mb)", "Audio Board Serial Number", "Timestamp", "Maneuver", "Notes", None, None],
        ["R_FILE", 98.0, "HP_W12.2-5", "12:00:00", "Walk (slow,medium, fast) 80 seconds each speed", None, None, None],
        ["R_FE_FILE", 29.0, "HP_W12.2-5", "12:10:00", "Flexion - Extension", None, None, None],
        ["R_STS_FILE", 27.0, "HP_W12.2-5", "12:20:00", "Sit - to - Stand", None, None, None],
        *[[None] * 8 for _ in range(7)],
        [None, None, None, None, None, None, "Red Device", "HP_W11.2-5"],
        [None, None, None, None, None, None, "White Device", "HP_W12.2-5"],
    ]

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        pd.DataFrame(acoustic_data).to_excel(
            writer,
            sheet_name="Acoustic Notes",
            index=False,
            header=False,
        )
        pd.DataFrame(mic_setup_data).to_excel(
            writer,
            sheet_name="Mic Setup",
            index=False,
            header=False,
        )

    return excel_path


class TestCrossSheetMismatch:
    """Tests for cross-sheet validation between Acoustic Notes and Mic Setup."""

    def test_mismatch_detected_for_mic_laterality(self, tmp_path):
        """Mic laterality mismatch between sheets should be flagged."""
        legend_path = _create_legend_with_mismatch(tmp_path)

        metadata, mismatches = get_acoustics_metadata(
            str(legend_path),
            "walk",
            "left",
        )

        # Should have detected the mic 1 laterality mismatch
        laterality_mismatches = [m for m in mismatches if m.field == "mic_1_laterality"]
        assert len(laterality_mismatches) == 1
        assert laterality_mismatches[0].acoustic_notes_value == "Lateral"
        assert laterality_mismatches[0].mic_setup_value == "Medial"

    def test_no_file_name_mismatch_when_names_match(self, tmp_path):
        """When file names match, no file_name mismatch should be reported."""
        legend_path = _create_legend_with_mismatch(tmp_path)

        _, mismatches = get_acoustics_metadata(
            str(legend_path),
            "walk",
            "left",
        )

        file_name_mismatches = [m for m in mismatches if m.field == "file_name"]
        assert len(file_name_mismatches) == 0

    def test_no_mismatch_when_sheets_agree(self, fake_participant_directory):
        """Happy path: no mismatches when both sheets agree."""
        legend_path = fake_participant_directory["legend_file"]

        _, mismatches = get_acoustics_metadata(
            str(legend_path),
            "walk",
            "left",
        )

        assert mismatches == []

    def test_mismatch_includes_knee_and_maneuver(self, tmp_path):
        """Mismatches should be tagged with knee and maneuver context."""
        legend_path = _create_legend_with_mismatch(tmp_path)

        _, mismatches = get_acoustics_metadata(
            str(legend_path),
            "walk",
            "left",
        )

        assert len(mismatches) > 0
        for mm in mismatches:
            assert mm.knee == "left"
            assert mm.maneuver == "walk"


# ── Legend Validation sheet in report ────────────────────────────────


class TestLegendValidationSheet:
    """Tests for the Legend Validation Excel sheet."""

    def test_legend_validation_sheet_generated(self):
        """Legend Validation sheet should contain mismatch data."""
        from src.reports.report_generator import ReportGenerator

        mismatches = [
            LegendMismatch(
                knee="left",
                maneuver="walk",
                field="mic_1_laterality",
                acoustic_notes_value="Lateral",
                mic_setup_value="Medial",
            ),
        ]

        df = ReportGenerator.generate_legend_validation_sheet(mismatches)

        assert len(df) == 1
        assert df.iloc[0]["Field"] == "mic_1_laterality"
        assert df.iloc[0]["Acoustic Notes Value"] == "Lateral"
        assert df.iloc[0]["Mic Setup Value"] == "Medial"

    def test_empty_sheet_when_no_mismatches(self):
        """No mismatches should produce an empty DataFrame."""
        from src.reports.report_generator import ReportGenerator

        df = ReportGenerator.generate_legend_validation_sheet([])

        assert df.empty
