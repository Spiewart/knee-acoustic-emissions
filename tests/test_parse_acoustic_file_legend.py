from parse_acoustic_file_legend import (
    get_acoustics_metadata,
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
