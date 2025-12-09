import logging
import shutil
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from process_participant_directory import (
    _generate_synced_filename,
    _load_event_data,
    dir_has_acoustic_file_legend,
    find_participant_directories,
    get_audio_file_name,
    get_study_id_from_directory,
    knee_folder_has_subfolder_each_maneuver,
    knee_subfolder_has_acoustic_files,
    motion_capture_folder_has_required_data,
    process_participant,
    setup_logging,
)


def test_get_study_id_from_directory(fake_participant_directory):
    participant_dir = fake_participant_directory["participant_dir"]

    study_id = get_study_id_from_directory(participant_dir)

    assert study_id == "1011"


def test_dir_has_acoustic_file_legend(fake_participant_directory, tmp_path):
    participant_dir = fake_participant_directory["participant_dir"]

    dir_has_acoustic_file_legend(participant_dir)

    # Create an empty participant directory without legend
    empty_participant_dir = tmp_path / "#2022"
    empty_participant_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError):
        dir_has_acoustic_file_legend(empty_participant_dir)


def test_knee_folder_has_subfolder_each_maneuver(fake_participant_directory):
    """Test knee_folder_has_subfolder_each_maneuver validation."""
    participant_dir = fake_participant_directory["participant_dir"]
    left_knee_dir = participant_dir / "Left Knee"
    right_knee_dir = participant_dir / "Right Knee"

    knee_folder_has_subfolder_each_maneuver(left_knee_dir)
    knee_folder_has_subfolder_each_maneuver(right_knee_dir)

    # Now test with a missing subfolder
    shutil.rmtree(left_knee_dir / "Walking")

    with pytest.raises(FileNotFoundError):
        knee_folder_has_subfolder_each_maneuver(left_knee_dir)


def test_get_audio_file_name_success(fake_participant_directory):
    """Test get_audio_file_name with exactly one .bin file."""
    participant_dir = fake_participant_directory["participant_dir"]
    maneuver_dir = participant_dir / "Left Knee" / "Flexion-Extension"

    result = get_audio_file_name(maneuver_dir)

    assert result == str(maneuver_dir / "test_audio")


def test_get_audio_file_name_with_freq(fake_participant_directory):
    """get_audio_file_name returns suffixed name when with_freq is True."""
    participant_dir = fake_participant_directory["participant_dir"]
    maneuver_dir = participant_dir / "Left Knee" / "Flexion-Extension"

    result = get_audio_file_name(maneuver_dir, with_freq=True)

    assert result == str(maneuver_dir / "test_audio_with_freq")


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


def test_knee_subfolder_has_acoustic_files_success(fake_participant_directory):
    """Test knee_subfolder_has_acoustic_files with valid structure."""
    participant_dir = fake_participant_directory["participant_dir"]
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
    fake_participant_directory,
):
    """Test motion_capture_folder_has_required_data with valid Excel file."""
    participant_dir = fake_participant_directory["participant_dir"]
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


def test_motion_capture_folder_missing_walking_pass_metadata(tmp_path):
    """Test motion_capture_folder with missing Walk0001 pass metadata sheet."""
    participant_dir = tmp_path / "#1011"
    participant_dir.mkdir()
    motion_capture_dir = participant_dir / "Motion Capture"
    motion_capture_dir.mkdir()

    study_id = "1011"
    excel_filename = f"AOA{study_id}_Biomechanics_Full_Set.xlsx"
    excel_file_path = motion_capture_dir / excel_filename

    # Create Excel file missing Walk0001 pass metadata sheet
    with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
        empty_df = pd.DataFrame()
        # Only create some sheets, not all required ones
        empty_df.to_excel(
            writer, sheet_name=f"AOA{study_id}_StoS", index=False
        )

    with pytest.raises(ValueError) as exc_info:
        motion_capture_folder_has_required_data(motion_capture_dir)

    assert "AOA1011_Walk0001" in str(exc_info.value)


def test_motion_capture_folder_missing_speed_sheet(tmp_path):
    """Test motion_capture_folder with missing walking speed sheet."""
    participant_dir = tmp_path / "#1011"
    participant_dir.mkdir()
    motion_capture_dir = participant_dir / "Motion Capture"
    motion_capture_dir.mkdir()

    study_id = "1011"
    excel_filename = f"AOA{study_id}_Biomechanics_Full_Set.xlsx"
    excel_file_path = motion_capture_dir / excel_filename

    # Create Excel file with pass metadata but missing speed sheets
    with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
        empty_df = pd.DataFrame()
        empty_df.to_excel(
            writer, sheet_name=f"AOA{study_id}_Walk0001", index=False
        )
        # Missing speed sheets

    with pytest.raises(ValueError) as exc_info:
        motion_capture_folder_has_required_data(motion_capture_dir)

    assert "AOA1011_Slow_Walking" in str(exc_info.value)


def test_motion_capture_folder_missing_maneuver_sheet(tmp_path):
    """Test motion_capture_folder with missing SitToStand sheet."""
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
        # All walking sheets (including pass metadata)
        empty_df.to_excel(
            writer, sheet_name=f"AOA{study_id}_Walk0001", index=False
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

    assert "AOA1011_SitToStand" in str(exc_info.value)


def test_generate_synced_filename_walk():
    result = _generate_synced_filename(
        knee_side="Left",
        maneuver_key="walk",
        pass_number=2,
        speed="slow",
    )

    assert result == "Left_walk_Pass0002_slow"


def test_generate_synced_filename_non_walk():
    result = _generate_synced_filename(
        knee_side="Right",
        maneuver_key="sit_to_stand",
        pass_number=None,
        speed=None,
    )

    assert result == "Right_sit_to_stand"


def test_load_event_data_walk(fake_participant_directory):
    biomechanics_file = fake_participant_directory["biomechanics"][
        "excel_path"
    ]

    events_df = _load_event_data(
        biomechanics_file=biomechanics_file,
        maneuver_key="walk",
    )

    assert "Event Info" in events_df.columns
    assert events_df.iloc[0]["Event Info"] == "Sync Left"


def test_load_event_data_sit_to_stand(fake_participant_directory):
    biomechanics_file = fake_participant_directory["biomechanics"][
        "excel_path"
    ]

    events_df = _load_event_data(
        biomechanics_file=biomechanics_file,
        maneuver_key="sit_to_stand",
    )

    assert not events_df.empty
    assert "Movement Start" in events_df["Event Info"].values


# ============================================================================
# Tests for CLI functions (batch processing)
# ============================================================================


def test_find_participant_directories_single(fake_participant_directory):
    """Test finding participant directories in a project folder."""
    project_dir = fake_participant_directory["project_dir"]

    result = find_participant_directories(project_dir)

    assert len(result) == 1
    assert result[0].name == "#1011"


def test_find_participant_directories_multiple(tmp_path):
    """Test finding multiple participant directories."""
    # Create multiple participant directories
    p1 = tmp_path / "#1011"
    p2 = tmp_path / "#2024"
    p3 = tmp_path / "#3001"
    p1.mkdir()
    p2.mkdir()
    p3.mkdir()

    # Create a non-participant directory (should be ignored)
    other = tmp_path / "other_folder"
    other.mkdir()

    result = find_participant_directories(tmp_path)

    assert len(result) == 3
    assert result[0].name == "#1011"
    assert result[1].name == "#2024"
    assert result[2].name == "#3001"


def test_find_participant_directories_empty(tmp_path):
    """Test with no participant directories."""
    result = find_participant_directories(tmp_path)

    assert result == []


def test_find_participant_directories_nonexistent_path():
    """Test with nonexistent path."""
    nonexistent = Path("/nonexistent/path/to/studies")

    result = find_participant_directories(nonexistent)

    assert result == []


def test_find_participant_directories_not_directory(tmp_path):
    """Test with a file instead of directory."""
    file_path = tmp_path / "file.txt"
    file_path.write_text("test")

    result = find_participant_directories(file_path)

    assert result == []


def test_find_participant_directories_sorting(tmp_path):
    """Test that participant directories are sorted."""
    # Create in non-sorted order
    p3 = tmp_path / "#3001"
    p1 = tmp_path / "#1011"
    p2 = tmp_path / "#2024"
    p3.mkdir()
    p1.mkdir()
    p2.mkdir()

    result = find_participant_directories(tmp_path)

    assert [d.name for d in result] == ["#1011", "#2024", "#3001"]


def test_setup_logging_console_only():
    """Test logging setup with console only."""
    # Should not raise an exception
    setup_logging(log_file=None)
    assert True  # Setup completed successfully


def test_setup_logging_with_file(tmp_path):
    """Test logging setup with file output."""
    log_file = tmp_path / "test.log"

    setup_logging(log_file=log_file)

    # Should not raise an exception
    assert True  # Setup completed successfully


def test_process_participant_success(
    fake_participant_directory, caplog, monkeypatch
):
    """Test processing a valid participant directory succeeds."""
    participant_dir = fake_participant_directory["participant_dir"]

    # Mock parse_participant_directory to avoid processing all data
    def mock_parse(*args, **kwargs):
        pass

    monkeypatch.setattr(
        "process_participant_directory.parse_participant_directory",
        mock_parse,
    )

    # Capture INFO level logs
    caplog.set_level(logging.INFO)
    result = process_participant(participant_dir)

    assert result is True
    assert "Successfully completed processing" in caplog.text


def test_process_participant_validation_failure(tmp_path, caplog):
    """Test processing with directory validation failure."""
    # Create a directory without required structure
    invalid_dir = tmp_path / "#invalid"
    invalid_dir.mkdir()

    result = process_participant(invalid_dir)

    assert result is False
    assert "Validation error" in caplog.text or "Left Knee" in caplog.text


def test_process_participant_returns_false_on_error(
    tmp_path, caplog, monkeypatch
):
    """Test that process_participant returns False on any exception."""
    participant_dir = tmp_path / "#1011"
    participant_dir.mkdir()

    # Mock parse_participant_directory to raise an exception
    def mock_parse(*args, **kwargs):
        raise RuntimeError("Unexpected error")

    monkeypatch.setattr(
        "process_participant_directory.parse_participant_directory",
        mock_parse,
    )

    # Mock check_participant_dir_for_required_files to not raise
    def mock_check(*args, **kwargs):
        pass

    monkeypatch.setattr(
        (
            "process_participant_directory."
            "check_participant_dir_for_required_files"
        ),
        mock_check,
    )

    result = process_participant(participant_dir)

    assert result is False
    assert "Unexpected error" in caplog.text


def test_process_participant_extracts_study_id(
    fake_participant_directory, caplog
):
    """Test that process_participant correctly extracts study ID."""
    participant_dir = fake_participant_directory["participant_dir"]

    caplog.set_level(logging.INFO)

    process_participant(participant_dir)

    # Check that the study ID "1011" appears in logs
    assert "#1011" in caplog.text or "1011" in caplog.text


def test_process_participant_validation_passed_message(
    fake_participant_directory, caplog, monkeypatch
):
    """Test that validation passed message is logged."""
    participant_dir = fake_participant_directory["participant_dir"]

    # Mock parse_participant_directory to avoid processing all data
    def mock_parse(*args, **kwargs):
        pass

    monkeypatch.setattr(
        "process_participant_directory.parse_participant_directory",
        mock_parse,
    )

    # Capture INFO level logs
    caplog.set_level(logging.INFO)
    process_participant(participant_dir)

    assert "Directory validation passed" in caplog.text
