
from acoustic_emissions_processing.parse_acoustic_file_legend import get_acoustics_metadata


def test_get_acoustics_metadata() -> None:
    """Test parsing acoustics metadata from a sample Excel file legend."""

    sample_file_path = (
        "/Users/spiewart/kae_signal_processing_ml/sample_files/1011_acoustic_file_legend_templatee.xlsx"
    )

    metadata = get_acoustics_metadata(
        metadata_file_path=sample_file_path,
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