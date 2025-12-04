import pandas as pd
import pytest

from acoustic_emissions_processing.parse_acoustic_file_legend import (
    get_acoustics_metadata,
)


@pytest.fixture
def sample_acoustic_legend_file(tmp_path):
    """Create a sample acoustic file legend Excel file for testing."""
    excel_path = tmp_path / "test_acoustic_file_legend.xlsx"

    # Create the structure as described in the docstring:
    # Two tables (L Knee and R Knee) separated by a blank row
    data = [
        ["L Knee", None, None, None, None, None],
        [
            "Maneuvers",
            "File Name",
            "Microphone",
            "Patellar Position",
            "Laterality",
            "Notes",
        ],
        [
            "Walk (slow,medium, fast)",
            "HP_W11.2-5-20240126_135702",
            1,
            "Infrapatellar",
            "Lateral",
            None,
        ],
        [None, None, 2, "Infrapatellar", "Medial", None],
        [None, None, 3, "Suprapatellar", "Medial", None],
        [None, None, 4, "Suprapatellar", "Lateral", None],
        [
            "Flexion - Extension",
            "HP_W11.2-1-20240126_135704",
            1,
            "Infrapatellar",
            "Lateral",
            None,
        ],
        [None, None, 2, "Infrapatellar", "Medial", None],
        [None, None, 3, "Suprapatellar", "Medial", None],
        [None, None, 4, "Suprapatellar", "Lateral", None],
        [
            "Sit - to - Stand",
            "HP_W11.2-3-20240126_135706",
            1,
            "Infrapatellar",
            "Lateral",
            None,
        ],
        [None, None, 2, "Infrapatellar", "Medial", None],
        [None, None, 3, "Suprapatellar", "Medial", None],
        [None, None, 4, "Suprapatellar", "Lateral", None],
        [None, None, None, None, None, None],  # Blank row separator
        ["R Knee", None, None, None, None, None],
        [
            "Maneuvers",
            "File Name",
            "Microphone",
            "Patellar Position",
            "Laterality",
            "Notes",
        ],
        [
            "Walk (slow,medium, fast)",
            "HP_W12.2-5-20240126_135802",
            1,
            "Infrapatellar",
            "Lateral",
            None,
        ],
        [None, None, 2, "Infrapatellar", "Medial", None],
        [None, None, 3, "Suprapatellar", "Medial", None],
        [None, None, 4, "Suprapatellar", "Lateral", None],
        [
            "Flexion - Extension",
            "HP_W12.2-1-20240126_135804",
            1,
            "Infrapatellar",
            "Lateral",
            None,
        ],
        [None, None, 2, "Infrapatellar", "Medial", None],
        [None, None, 3, "Suprapatellar", "Medial", None],
        [None, None, 4, "Suprapatellar", "Lateral", None],
        [
            "Sit - to - Stand",
            "HP_W12.2-3-20240126_135806",
            1,
            "Infrapatellar",
            "Lateral",
            None,
        ],
        [None, None, 2, "Infrapatellar", "Medial", None],
        [None, None, 3, "Suprapatellar", "Medial", None],
        [None, None, 4, "Suprapatellar", "Lateral", None],
    ]

    df = pd.DataFrame(data)

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(
            writer, sheet_name="Acoustic Notes", index=False, header=False
        )

    return excel_path


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
