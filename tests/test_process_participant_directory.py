import shutil
from pathlib import Path

import pandas as pd
import pytest

from acoustic_emissions_processing.process_participant_directory import (
    dir_has_acoustic_file_legend,
    get_audio_file_name,
    get_study_id_from_directory,
    knee_folder_has_subfolder_each_maneuver,
    knee_subfolder_has_acoustic_files,
    motion_capture_folder_has_required_data,
    parse_participant_directory,
)


@pytest.fixture
def sample_participant_dir(tmp_path_factory):
    """Create a complete sample participant directory structure for testing."""
    project_dir = tmp_path_factory.mktemp("project")
    participant_dir = project_dir / "#1011"
    participant_dir.mkdir()

    # Create acoustic file legend
    legend_file = participant_dir / "acoustic_file_legend.xlsx"
    legend_file.touch()

    # Create top-level folders
    left_knee_dir = participant_dir / "Left Knee"
    right_knee_dir = participant_dir / "Right Knee"
    motion_capture_dir = participant_dir / "Motion Capture"
    motion_capture_dir.mkdir()

    # Create maneuver folders and files for both knees
    for knee_dir in [left_knee_dir, right_knee_dir]:
        for maneuver in ["Flexion-Extension", "Sit-Stand", "Walking"]:
            maneuver_dir = knee_dir / maneuver
            maneuver_dir.mkdir(parents=True)

            # Create .bin file
            bin_file = maneuver_dir / "test_audio.bin"
            bin_file.touch()

            # Create outputs directory and .pkl file
            audio_file_path = str(bin_file.with_suffix(""))
            outputs_dir = Path(audio_file_path + "_outputs")
            outputs_dir.mkdir()
            pkl_file = outputs_dir / (audio_file_path + ".pkl")
            pkl_file.parent.mkdir(parents=True, exist_ok=True)
            pkl_file.touch()

    # Create motion capture Excel file with required sheets
    study_id = "1011"
    excel_filename = f"AOA{study_id}_Biomechanics_Full_Set.xlsx"
    excel_file_path = motion_capture_dir / excel_filename

    # Create Excel file with all required sheets
    with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
        # Create empty DataFrames for each required sheet
        empty_df = pd.DataFrame()

        # Walking sheets (with speeds)
        empty_df.to_excel(
            writer, sheet_name=f"AOA{study_id}_Walking_Events", index=False
        )
        empty_df.to_excel(
            writer, sheet_name=f"AOA{study_id}_Slow_Walking", index=False
        )
        empty_df.to_excel(
            writer, sheet_name=f"AOA{study_id}_Medium_Walking", index=False
        )
        empty_df.to_excel(
            writer, sheet_name=f"AOA{study_id}_Fast_Walking", index=False
        )

        # Sit-to-Stand sheets
        empty_df.to_excel(
            writer, sheet_name=f"AOA{study_id}_StoS_Events", index=False
        )
        empty_df.to_excel(
            writer, sheet_name=f"AOA{study_id}_StoS", index=False
        )

        # Flexion-Extension sheets
        empty_df.to_excel(
            writer, sheet_name=f"AOA{study_id}_FE_Events", index=False
        )
        empty_df.to_excel(
            writer, sheet_name=f"AOA{study_id}_FE", index=False
        )

    return project_dir


def test_parse_participant_directory(sample_participant_dir):
    """Test parse_participant_directory with valid structure."""
    parse_participant_directory(str(sample_participant_dir))


def test_get_study_id_from_directory(sample_participant_dir):
    participant_dir = sample_participant_dir / "#1011"

    study_id = get_study_id_from_directory(participant_dir)

    assert study_id == "1011"


def test_dir_has_acoustic_file_legend(sample_participant_dir, tmp_path):
    participant_dir = sample_participant_dir / "#1011"

    dir_has_acoustic_file_legend(participant_dir)

    # Create an empty participant directory without legend
    empty_participant_dir = tmp_path / "#2022"
    empty_participant_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError):
        dir_has_acoustic_file_legend(empty_participant_dir)


def test_knee_folder_has_subfolder_each_maneuver(sample_participant_dir):
    """Test knee_folder_has_subfolder_each_maneuver validation."""
    participant_dir = sample_participant_dir / "#1011"
    left_knee_dir = participant_dir / "Left Knee"
    right_knee_dir = participant_dir / "Right Knee"

    knee_folder_has_subfolder_each_maneuver(left_knee_dir)
    knee_folder_has_subfolder_each_maneuver(right_knee_dir)

    # Now test with a missing subfolder
    shutil.rmtree(left_knee_dir / "Walking")

    with pytest.raises(FileNotFoundError):
        knee_folder_has_subfolder_each_maneuver(left_knee_dir)


def test_get_audio_file_name_success(sample_participant_dir):
    """Test get_audio_file_name with exactly one .bin file."""
    participant_dir = sample_participant_dir / "#1011"
    maneuver_dir = participant_dir / "Left Knee" / "Flexion-Extension"

    result = get_audio_file_name(maneuver_dir)

    assert result == str(maneuver_dir / "test_audio")


def test_get_audio_file_name_no_bin_files(tmp_path):
    """Test get_audio_file_name raises error when no .bin files exist."""
    maneuver_dir = tmp_path

    with pytest.raises(FileNotFoundError) as exc_info:
        get_audio_file_name(maneuver_dir)

    assert "No .bin acoustic files found" in str(exc_info.value)


def test_get_audio_file_name_multiple_bin_files(tmp_path):
    """Test get_audio_file_name raises error with multiple .bin files."""
    maneuver_dir = tmp_path

    # Create multiple .bin files
    (maneuver_dir / "test_audio1.bin").touch()
    (maneuver_dir / "test_audio2.bin").touch()

    with pytest.raises(AssertionError) as exc_info:
        get_audio_file_name(maneuver_dir)

    assert "Expected exactly one .bin file" in str(exc_info.value)


def test_knee_subfolder_has_acoustic_files_success(sample_participant_dir):
    """Test knee_subfolder_has_acoustic_files with valid structure."""
    participant_dir = sample_participant_dir / "#1011"
    maneuver_dir = participant_dir / "Left Knee" / "Walking"

    # Should not raise any exception
    knee_subfolder_has_acoustic_files(maneuver_dir)


def test_knee_subfolder_has_acoustic_files_no_bin(tmp_path):
    """Test knee_subfolder_has_acoustic_files with no .bin file."""
    maneuver_dir = tmp_path

    with pytest.raises(FileNotFoundError) as exc_info:
        knee_subfolder_has_acoustic_files(maneuver_dir)

    assert "No .bin acoustic files found" in str(exc_info.value)


def test_knee_subfolder_has_acoustic_files_no_pkl(tmp_path):
    """Test knee_subfolder_has_acoustic_files with no .pkl file."""
    maneuver_dir = tmp_path

    # Create a .bin file
    bin_file = maneuver_dir / "test_audio.bin"
    bin_file.touch()

    # Create the outputs directory but no .pkl file
    outputs_dir = maneuver_dir / "test_audio_outputs"
    outputs_dir.mkdir()

    with pytest.raises(FileNotFoundError) as exc_info:
        knee_subfolder_has_acoustic_files(maneuver_dir)

    assert "Processed audio .pkl file" in str(exc_info.value)


def test_knee_subfolder_has_acoustic_files_no_outputs_dir(tmp_path):
    """Test knee_subfolder_has_acoustic_files with no outputs directory."""
    maneuver_dir = tmp_path

    # Create a .bin file but no outputs directory
    bin_file = maneuver_dir / "test_audio.bin"
    bin_file.touch()

    with pytest.raises(FileNotFoundError) as exc_info:
        knee_subfolder_has_acoustic_files(maneuver_dir)

    assert "Processed audio .pkl file" in str(exc_info.value)


def test_motion_capture_folder_has_required_data_success(
    sample_participant_dir
):
    """Test motion_capture_folder_has_required_data with valid Excel file."""
    participant_dir = sample_participant_dir / "#1011"
    motion_capture_dir = participant_dir / "Motion Capture"

    # Should not raise any exception
    motion_capture_folder_has_required_data(motion_capture_dir)


def test_motion_capture_folder_has_required_data_no_file(tmp_path):
    """Test motion_capture_folder_has_required_data with missing file."""
    participant_dir = tmp_path / "#1011"
    participant_dir.mkdir()
    motion_capture_dir = participant_dir / "Motion Capture"
    motion_capture_dir.mkdir()

    with pytest.raises(FileNotFoundError) as exc_info:
        motion_capture_folder_has_required_data(motion_capture_dir)

    assert "AOA1011_Biomechanics_Full_Set.xlsx" in str(exc_info.value)


def test_motion_capture_folder_missing_walking_events_sheet(tmp_path):
    """Test motion_capture_folder with missing Walking_Events sheet."""
    participant_dir = tmp_path / "#1011"
    participant_dir.mkdir()
    motion_capture_dir = participant_dir / "Motion Capture"
    motion_capture_dir.mkdir()

    study_id = "1011"
    excel_filename = f"AOA{study_id}_Biomechanics_Full_Set.xlsx"
    excel_file_path = motion_capture_dir / excel_filename

    # Create Excel file missing Walking_Events sheet
    with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
        empty_df = pd.DataFrame()
        # Only create some sheets, not all required ones
        empty_df.to_excel(
            writer, sheet_name=f"AOA{study_id}_StoS", index=False
        )

    with pytest.raises(ValueError) as exc_info:
        motion_capture_folder_has_required_data(motion_capture_dir)

    assert "AOA1011_Walking_Events" in str(exc_info.value)


def test_motion_capture_folder_missing_speed_sheet(tmp_path):
    """Test motion_capture_folder with missing walking speed sheet."""
    participant_dir = tmp_path / "#1011"
    participant_dir.mkdir()
    motion_capture_dir = participant_dir / "Motion Capture"
    motion_capture_dir.mkdir()

    study_id = "1011"
    excel_filename = f"AOA{study_id}_Biomechanics_Full_Set.xlsx"
    excel_file_path = motion_capture_dir / excel_filename

    # Create Excel file with events but missing speed sheets
    with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
        empty_df = pd.DataFrame()
        empty_df.to_excel(
            writer, sheet_name=f"AOA{study_id}_Walking_Events", index=False
        )
        # Missing speed sheets

    with pytest.raises(ValueError) as exc_info:
        motion_capture_folder_has_required_data(motion_capture_dir)

    assert "AOA1011_Slow_Walking" in str(exc_info.value)


def test_motion_capture_folder_missing_maneuver_sheet(tmp_path):
    """Test motion_capture_folder with missing StoS sheet."""
    participant_dir = tmp_path / "#1011"
    participant_dir.mkdir()
    motion_capture_dir = participant_dir / "Motion Capture"
    motion_capture_dir.mkdir()

    study_id = "1011"
    excel_filename = f"AOA{study_id}_Biomechanics_Full_Set.xlsx"
    excel_file_path = motion_capture_dir / excel_filename

    # Create Excel file with all walking sheets but missing StoS
    with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
        empty_df = pd.DataFrame()
        # All walking sheets
        empty_df.to_excel(
            writer, sheet_name=f"AOA{study_id}_Walking_Events", index=False
        )
        empty_df.to_excel(
            writer, sheet_name=f"AOA{study_id}_Slow_Walking", index=False
        )
        empty_df.to_excel(
            writer, sheet_name=f"AOA{study_id}_Medium_Walking", index=False
        )
        empty_df.to_excel(
            writer, sheet_name=f"AOA{study_id}_Fast_Walking", index=False
        )
        # StoS events but missing StoS data sheet
        empty_df.to_excel(
            writer, sheet_name=f"AOA{study_id}_StoS_Events", index=False
        )

    with pytest.raises(ValueError) as exc_info:
        motion_capture_folder_has_required_data(motion_capture_dir)

    assert "AOA1011_StoS" in str(exc_info.value)
