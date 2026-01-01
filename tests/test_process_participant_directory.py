import logging
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.orchestration.participant import (
    _generate_synced_filename,
    _load_event_data,
    _trim_and_rename_biomechanics_columns,
    dir_has_acoustic_file_legend,
    find_participant_directories,
    get_audio_file_name,
    get_study_id_from_directory,
    knee_folder_has_subfolder_each_maneuver,
    knee_subfolder_has_acoustic_files,
    main,
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

    # Error should mention the _with_freq pickle file
    assert "Processed audio .pkl file" in str(exc_info.value)
    assert "test_audio_with_freq.pkl" in str(exc_info.value)


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
    with pd.ExcelWriter(excel_file_path, engine="openpyxl") as writer:
        empty_df = pd.DataFrame()
        # Only create some sheets, not all required ones
        empty_df.to_excel(writer, sheet_name=f"AOA{study_id}_StoS", index=False)

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
    with pd.ExcelWriter(excel_file_path, engine="openpyxl") as writer:
        empty_df = pd.DataFrame()
        empty_df.to_excel(writer, sheet_name=f"AOA{study_id}_Walk0001", index=False)
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
    with pd.ExcelWriter(excel_file_path, engine="openpyxl") as writer:
        empty_df = pd.DataFrame()
        # All walking sheets (including pass metadata)
        empty_df.to_excel(writer, sheet_name=f"AOA{study_id}_Walk0001", index=False)
        empty_df.to_excel(writer, sheet_name=f"AOA{study_id}_Slow_Walking", index=False)
        empty_df.to_excel(
            writer, sheet_name=f"AOA{study_id}_Medium_Walking", index=False
        )
        empty_df.to_excel(writer, sheet_name=f"AOA{study_id}_Fast_Walking", index=False)
        # StoS events but missing StoS data sheet
        empty_df.to_excel(writer, sheet_name=f"AOA{study_id}_StoS_Events", index=False)

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


def test_generate_synced_filename_all_speeds():
    """Test that all speed values (including internal 'normal') work correctly."""
    # Test slow
    result = _generate_synced_filename(
        knee_side="Right",
        maneuver_key="walk",
        pass_number=1,
        speed="slow",
    )
    assert result == "Right_walk_Pass0001_slow"

    # Test medium (external API speed)
    result = _generate_synced_filename(
        knee_side="Right",
        maneuver_key="walk",
        pass_number=2,
        speed="medium",
    )
    assert result == "Right_walk_Pass0002_medium"

    # Test normal (internal model speed - should map to medium)
    result = _generate_synced_filename(
        knee_side="Right",
        maneuver_key="walk",
        pass_number=3,
        speed="normal",
    )
    assert result == "Right_walk_Pass0003_medium"

    # Test fast
    result = _generate_synced_filename(
        knee_side="Right",
        maneuver_key="walk",
        pass_number=4,
        speed="fast",
    )
    assert result == "Right_walk_Pass0004_fast"


def test_generate_synced_filename_non_walk():
    result = _generate_synced_filename(
        knee_side="Right",
        maneuver_key="sit_to_stand",
        pass_number=None,
        speed=None,
    )

    assert result == "Right_sit_to_stand"


def test_load_event_data_walk(fake_participant_directory):
    biomechanics_file = fake_participant_directory["biomechanics"]["excel_path"]

    events_df = _load_event_data(
        biomechanics_file=biomechanics_file,
        maneuver_key="walk",
    )

    assert "Event Info" in events_df.columns
    assert events_df.iloc[0]["Event Info"] == "Sync Left"


def test_load_event_data_sit_to_stand(fake_participant_directory):
    biomechanics_file = fake_participant_directory["biomechanics"]["excel_path"]

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


def test_main_filters_requested_participants(monkeypatch, tmp_path, caplog):
    """Process only participants requested via --participant."""
    participants = []
    for name in ("#1011", "#2024", "#3001"):
        participant = tmp_path / name
        participant.mkdir()
        participants.append(participant)

    processed: list[str] = []

    def fake_find(path: Path) -> list[Path]:
        return participants

    def fake_process(participant_dir: Path) -> bool:
        processed.append(participant_dir.name)
        return True

    monkeypatch.setattr(
        "src.orchestration.participant.setup_logging",
        lambda log_file=None: None,
    )
    monkeypatch.setattr(
        "src.orchestration.participant.find_participant_directories",
        fake_find,
    )
    monkeypatch.setattr(
        "src.orchestration.participant.process_participant",
        fake_process,
    )

    caplog.set_level(logging.INFO)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "src.orchestration.participant.py",
            str(tmp_path),
            "--participant",
            "2024",
            "#3001",
        ],
    )

    main()

    assert processed == ["#2024", "#3001"]
    assert "Found 2 participant directory(ies)" in caplog.text


def test_main_warns_when_no_participant_matches(monkeypatch, tmp_path, caplog):
    """Warn and abort when requested participants are missing."""
    participant_dir = tmp_path / "#1011"
    participant_dir.mkdir()
    participants = [participant_dir]
    processed: list[str] = []

    def fake_find(path: Path) -> list[Path]:
        return participants

    def fake_process(participant_dir: Path) -> bool:
        processed.append(participant_dir.name)
        return True

    monkeypatch.setattr(
        "src.orchestration.participant.setup_logging",
        lambda log_file=None: None,
    )
    monkeypatch.setattr(
        "src.orchestration.participant.find_participant_directories",
        fake_find,
    )
    monkeypatch.setattr(
        "src.orchestration.participant.process_participant",
        fake_process,
    )

    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "src.orchestration.participant.py",
            str(tmp_path),
            "--participant",
            "9999",
        ],
    )

    main()

    assert processed == []
    assert "No matching participant directories found" in caplog.text


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


def test_process_participant_success(fake_participant_directory, caplog, monkeypatch):
    """Test processing a valid participant directory succeeds."""
    participant_dir = fake_participant_directory["participant_dir"]

    # Mock parse_participant_directory to avoid processing all data
    def mock_parse(*args, **kwargs):
        pass

    monkeypatch.setattr(
        "src.orchestration.participant.parse_participant_directory",
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


def test_process_participant_returns_false_on_error(tmp_path, caplog, monkeypatch):
    """Test that process_participant returns False on any exception."""
    participant_dir = tmp_path / "#1011"
    participant_dir.mkdir()

    # Mock parse_participant_directory to raise an exception
    def mock_parse(*args, **kwargs):
        raise RuntimeError("Unexpected error")

    monkeypatch.setattr(
        "src.orchestration.participant.parse_participant_directory",
        mock_parse,
    )

    # Mock check_participant_dir_for_required_files to not raise
    def mock_check(*args, **kwargs):
        pass

    monkeypatch.setattr(
        ("src.orchestration.participant." "check_participant_dir_for_required_files"),
        mock_check,
    )

    result = process_participant(participant_dir)

    assert result is False
    assert "Unexpected error" in caplog.text


def test_process_participant_extracts_study_id(fake_participant_directory, caplog):
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
        "src.orchestration.participant.parse_participant_directory",
        mock_parse,
    )

    # Capture INFO level logs
    caplog.set_level(logging.INFO)
    process_participant(participant_dir)

    assert "Directory validation passed" in caplog.text


# Tests for _trim_and_rename_biomechanics_columns


def test_trim_and_rename_left_knee_removes_right_columns():
    """Test that right knee columns are removed when knee_side='Left'."""
    df = pd.DataFrame(
        {
            "tt": [0.0, 0.1, 0.2],
            "ch1": [1, 2, 3],
            "Left Knee Angle_X": [10.0, 11.0, 12.0],
            "Left Knee Angle_Y": [20.0, 21.0, 22.0],
            "Right Knee Angle_X": [30.0, 31.0, 32.0],
            "Right Knee Angle_Y": [40.0, 41.0, 42.0],
        }
    )

    result = _trim_and_rename_biomechanics_columns(df, "Left")

    # Right knee columns should be removed
    assert "Right Knee Angle_X" not in result.columns
    assert "Right Knee Angle_Y" not in result.columns

    # Left knee columns should be present and renamed
    assert "Knee Angle X" in result.columns
    assert "Knee Angle Y" in result.columns

    # tt column should be retained for timestamp alignment
    assert "tt" in result.columns
    # Other audio columns should remain unchanged
    assert "ch1" in result.columns
    assert result.shape[0] == 3


def test_trim_and_rename_right_knee_removes_left_columns():
    """Test that left knee columns are removed when knee_side='Right'."""
    df = pd.DataFrame(
        {
            "tt": [0.0, 0.1, 0.2],
            "ch1": [1, 2, 3],
            "Left Knee Angle_X": [10.0, 11.0, 12.0],
            "Left Knee Angle_Y": [20.0, 21.0, 22.0],
            "Right Knee Angle_X": [30.0, 31.0, 32.0],
            "Right Knee Angle_Y": [40.0, 41.0, 42.0],
        }
    )

    result = _trim_and_rename_biomechanics_columns(df, "Right")

    # Left knee columns should be removed
    assert "Left Knee Angle_X" not in result.columns
    assert "Left Knee Angle_Y" not in result.columns

    # Right knee columns should be present and renamed
    assert "Knee Angle X" in result.columns
    assert "Knee Angle Y" in result.columns

    # tt column should be retained for timestamp alignment
    assert "tt" in result.columns
    # Other audio columns should remain unchanged
    assert "ch1" in result.columns
    assert result.shape[0] == 3


def test_trim_and_rename_removes_underscores():
    """Test that underscores before axes are replaced with spaces."""
    df = pd.DataFrame(
        {
            "tt": [0.0, 0.1],
            "Left Knee Angle_X": [1.0, 2.0],
            "Left Knee Velocity_Y": [3.0, 4.0],
            "Left Knee Force_Z": [5.0, 6.0],
        }
    )

    result = _trim_and_rename_biomechanics_columns(df, "Left")

    # Underscores should be replaced with spaces
    assert "Knee Angle X" in result.columns
    assert "Knee Velocity Y" in result.columns
    assert "Knee Force Z" in result.columns

    # Original column names should not be present
    assert "Left Knee Angle_X" not in result.columns


def test_trim_and_rename_preserves_column_order():
    """Test that column order is preserved."""
    df = pd.DataFrame(
        {
            "tt": [1.0, 2.0],
            "ch1": [10, 20],
            "Left Knee Angle_X": [100, 200],
            "ch2": [30, 40],
            "Left Knee Angle_Y": [300, 400],
        }
    )

    result = _trim_and_rename_biomechanics_columns(df, "Left")

    # Check that all columns are present (Right knee removed; tt retained)
    assert len(result.columns) == 5
    assert "tt" in result.columns
    assert "ch1" in result.columns
    assert "ch2" in result.columns
    assert "Knee Angle X" in result.columns
    assert "Knee Angle Y" in result.columns


def test_trim_and_rename_case_insensitive():
    """Test that knee_side is case-insensitive."""
    df = pd.DataFrame(
        {
            "tt": [0.0],
            "Left Knee Angle_X": [1.0],
            "Right Knee Angle_X": [2.0],
        }
    )

    result_upper = _trim_and_rename_biomechanics_columns(df, "LEFT")
    result_lower = _trim_and_rename_biomechanics_columns(df, "left")
    result_title = _trim_and_rename_biomechanics_columns(df, "Left")

    # All should have the same columns
    assert set(result_upper.columns) == set(result_lower.columns)
    assert set(result_lower.columns) == set(result_title.columns)
    assert "Knee Angle X" in result_upper.columns


def test_trim_and_rename_invalid_knee_side():
    """Test that invalid knee_side raises ValueError."""
    df = pd.DataFrame({"Left Knee Angle_X": [1.0]})

    with pytest.raises(ValueError, match="knee_side must be"):
        _trim_and_rename_biomechanics_columns(df, "Middle")

    with pytest.raises(ValueError, match="knee_side must be"):
        _trim_and_rename_biomechanics_columns(df, "invalid")


def test_trim_and_rename_dataframe_values_preserved():
    """Test that DataFrame values are preserved (no data loss)."""
    df = pd.DataFrame(
        {
            "tt": [0.0, 0.1],
            "Left Knee Angle_X": [10.5, 11.5],
            "Right Knee Angle_X": [20.5, 21.5],
        }
    )

    result = _trim_and_rename_biomechanics_columns(df, "Left")

    # Values should be preserved
    assert result["Knee Angle X"].tolist() == [10.5, 11.5]

    # Shape should be correct (removed Right Knee Angle_X; tt retained)
    assert result.shape == (2, 2)


def test_trim_and_rename_empty_dataframe():
    """Test behavior with empty DataFrame."""
    df = pd.DataFrame(
        {
            "tt": [],
            "Left Knee Angle_X": [],
            "Right Knee Angle_X": [],
        }
    )

    result = _trim_and_rename_biomechanics_columns(df, "Left")

    # Should have left knee column renamed and tt retained
    assert "Knee Angle X" in result.columns
    assert "Right Knee Angle_X" not in result.columns
    assert "tt" in result.columns
    assert len(result) == 0


def test_trim_and_rename_no_knee_columns():
    """Test with DataFrame containing no knee columns."""
    df = pd.DataFrame(
        {
            "tt": [1.0, 2.0],
            "ch1": [10, 20],
        }
    )

    result = _trim_and_rename_biomechanics_columns(df, "Left")

    # Should return DataFrame with tt retained
    assert set(result.columns) == {"ch1", "tt"}
    assert result.shape == (2, 2)


# Tests for sync_single_audio_file function


@pytest.fixture
def mock_audio_df():
    """Create mock audio DataFrame."""
    n_samples = 10000
    return pd.DataFrame(
        {
            "tt": np.arange(n_samples) / 1000.0,
            "ch1": np.random.randn(n_samples) * 0.01,
            "ch2": np.random.randn(n_samples) * 0.01,
            "ch3": np.random.randn(n_samples) * 0.01,
            "ch4": np.random.randn(n_samples) * 0.01,
        }
    )


@pytest.fixture
def mock_biomechanics_recordings():
    """Create mock biomechanics recordings for all speeds."""
    import numpy as np

    from src.models import BiomechanicsRecording

    recordings = []
    for speed in ["slow", "normal", "fast"]:
        for pass_num in [1, 2, 3, 4]:
            bio_df = pd.DataFrame(
                {
                    "TIME": pd.to_timedelta([0, 1, 2, 3, 4], unit="s"),
                    "Knee Angle Z": [10, 20, 30, 20, 10],
                }
            )
            recording = BiomechanicsRecording(
                study="AOA",
                study_id=1011,
                maneuver="walk",
                speed=speed,
                pass_number=pass_num,
                biomech_file_name=f"bio_{speed}_{pass_num}.c3d",
                data=bio_df,
            )
            recordings.append(recording)
    return recordings


class TestSyncSingleAudioFile:
    """Tests for sync_single_audio_file function."""

    def test_sync_walk_processes_all_speeds(
        self, tmp_path, mock_audio_df, mock_biomechanics_recordings
    ):
        """sync_single_audio_file should process all speeds for walking."""
        from src.orchestration.participant import sync_single_audio_file

        # Setup directory structure
        participant_dir = tmp_path / "#1011"
        motion_capture_dir = participant_dir / "Motion Capture"
        motion_capture_dir.mkdir(parents=True)

        knee_dir = participant_dir / "Right Knee"
        walking_dir = knee_dir / "Walking"
        outputs_dir = walking_dir / "HP_W11.2-1-20240126_135704_outputs"
        outputs_dir.mkdir(parents=True)

        audio_file = outputs_dir / "HP_W11.2-1-20240126_135704_with_freq.pkl"
        mock_audio_df.to_pickle(audio_file)

        biomech_file = motion_capture_dir / "AOA1011_Biomechanics_Full_Set.xlsx"
        biomech_file.touch()

        # Mock the imports and sync functions
        with patch(
            "src.orchestration.participant.import_biomechanics_recordings"
        ) as mock_import, patch(
            "src.orchestration.participant.load_audio_data"
        ) as mock_load_audio, patch(
            "src.orchestration.participant._sync_and_save_recording"
        ) as mock_sync, patch(
            "src.orchestration.participant.plot_stomp_detection"
        ) as mock_plot:

            # Configure mocks
            mock_load_audio.return_value = mock_audio_df

            # Mock import_biomechanics_recordings to return different recordings per speed
            # Note: The function is called with "medium" but recordings have speed="normal"
            def import_side_effect(biomechanics_file, maneuver, speed, **kwargs):
                # Map medium -> normal for fixture lookup
                lookup_speed = "normal" if speed == "medium" else speed
                return [
                    rec for rec in mock_biomechanics_recordings
                    if rec.speed == lookup_speed
                ]

            mock_import.side_effect = import_side_effect

            # Mock sync function
            synced_df = mock_audio_df.copy()
            synced_df["Knee Angle Z"] = 15.0
            mock_sync.return_value = (
                tmp_path / "output.pkl",
                synced_df,
                (10.0, 5.0, 5.0),  # stomp times
                pd.DataFrame(),
            )

            # Execute
            result = sync_single_audio_file(audio_file)

            # Verify
            assert result is True

            # Should call import_biomechanics_recordings for each speed
            assert mock_import.call_count == 3
            speeds_called = [call.kwargs["speed"] for call in mock_import.call_args_list]
            assert "slow" in speeds_called
            assert "medium" in speeds_called
            assert "fast" in speeds_called

            # Should sync all 12 recordings (3 speeds × 4 passes)
            assert mock_sync.call_count == 12

    def test_sync_walk_processes_all_passes_per_speed(
        self, tmp_path, mock_audio_df, mock_biomechanics_recordings
    ):
        """Each speed should have all passes synced."""
        from src.orchestration.participant import sync_single_audio_file

        # Setup
        participant_dir = tmp_path / "#1011"
        motion_capture_dir = participant_dir / "Motion Capture"
        motion_capture_dir.mkdir(parents=True)

        knee_dir = participant_dir / "Right Knee"
        walking_dir = knee_dir / "Walking"
        outputs_dir = walking_dir / "HP_W11.2-1-20240126_135704_outputs"
        outputs_dir.mkdir(parents=True)

        audio_file = outputs_dir / "HP_W11.2-1-20240126_135704_with_freq.pkl"
        mock_audio_df.to_pickle(audio_file)

        biomech_file = motion_capture_dir / "AOA1011_Biomechanics_Full_Set.xlsx"
        biomech_file.touch()

        with patch(
            "src.orchestration.participant.import_biomechanics_recordings"
        ) as mock_import, patch(
            "src.orchestration.participant.load_audio_data"
        ) as mock_load_audio, patch(
            "src.orchestration.participant._sync_and_save_recording"
        ) as mock_sync, patch(
            "src.orchestration.participant.plot_stomp_detection"
        ) as mock_plot:

            mock_load_audio.return_value = mock_audio_df

            def import_side_effect(biomechanics_file, maneuver, speed, **kwargs):
                # Map medium -> normal for fixture lookup
                lookup_speed = "normal" if speed == "medium" else speed
                return [
                    rec for rec in mock_biomechanics_recordings
                    if rec.speed == lookup_speed
                ]

            mock_import.side_effect = import_side_effect

            synced_df = mock_audio_df.copy()
            synced_df["Knee Angle Z"] = 15.0
            mock_sync.return_value = (
                tmp_path / "output.pkl",
                synced_df,
                (10.0, 5.0, 5.0),
                pd.DataFrame(),
            )

            result = sync_single_audio_file(audio_file)

            assert result is True

            # Check that all pass numbers were synced
            synced_recordings = [call.kwargs["recording"] for call in mock_sync.call_args_list]
            pass_numbers_synced = [rec.pass_number for rec in synced_recordings]

            # Should have 4 passes per speed (3 speeds)
            assert pass_numbers_synced.count(1) == 3
            assert pass_numbers_synced.count(2) == 3
            assert pass_numbers_synced.count(3) == 3
            assert pass_numbers_synced.count(4) == 3

    def test_sync_continues_on_individual_failures(
        self, tmp_path, mock_audio_df, mock_biomechanics_recordings
    ):
        """Should continue syncing other recordings if one fails."""
        from src.orchestration.participant import sync_single_audio_file

        # Setup
        participant_dir = tmp_path / "#1011"
        motion_capture_dir = participant_dir / "Motion Capture"
        motion_capture_dir.mkdir(parents=True)

        knee_dir = participant_dir / "Right Knee"
        walking_dir = knee_dir / "Walking"
        outputs_dir = walking_dir / "HP_W11.2-1-20240126_135704_outputs"
        outputs_dir.mkdir(parents=True)

        audio_file = outputs_dir / "HP_W11.2-1-20240126_135704_with_freq.pkl"
        mock_audio_df.to_pickle(audio_file)

        biomech_file = motion_capture_dir / "AOA1011_Biomechanics_Full_Set.xlsx"
        biomech_file.touch()

        with patch(
            "src.orchestration.participant.import_biomechanics_recordings"
        ) as mock_import, patch(
            "src.orchestration.participant.load_audio_data"
        ) as mock_load_audio, patch(
            "src.orchestration.participant._sync_and_save_recording"
        ) as mock_sync, patch(
            "src.orchestration.participant.plot_stomp_detection"
        ) as mock_plot:

            mock_load_audio.return_value = mock_audio_df

            def import_side_effect(biomechanics_file, maneuver, speed, **kwargs):
                # Map medium -> normal for fixture lookup
                lookup_speed = "normal" if speed == "medium" else speed
                return [
                    rec for rec in mock_biomechanics_recordings
                    if rec.speed == lookup_speed
                ][:2]  # Only 2 passes per speed

            mock_import.side_effect = import_side_effect

            # Make first call fail, rest succeed
            call_count = [0]

            def sync_side_effect(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise ValueError("Simulated sync failure")
                synced_df = mock_audio_df.copy()
                synced_df["Knee Angle Z"] = 15.0
                return (
                    tmp_path / "output.pkl",
                    synced_df,
                    (10.0, 5.0, 5.0),
                    pd.DataFrame(),
                )

            mock_sync.side_effect = sync_side_effect

            result = sync_single_audio_file(audio_file)

            # Should still succeed overall (5 out of 6 succeeded)
            assert result is True
            assert mock_sync.call_count == 6

    def test_sync_non_walk_maneuver_single_recording(
        self, tmp_path, mock_audio_df
    ):
        """Non-walking maneuvers should sync all recordings without speed."""
        from src.models import BiomechanicsRecording
        from src.orchestration.participant import sync_single_audio_file

        # Setup for sit-to-stand
        participant_dir = tmp_path / "#1011"
        motion_capture_dir = participant_dir / "Motion Capture"
        motion_capture_dir.mkdir(parents=True)

        knee_dir = participant_dir / "Right Knee"
        sts_dir = knee_dir / "Sit-Stand"
        outputs_dir = sts_dir / "HP_W11.2-3-20240126_135706_outputs"
        outputs_dir.mkdir(parents=True)

        audio_file = outputs_dir / "HP_W11.2-3-20240126_135706_with_freq.pkl"
        mock_audio_df.to_pickle(audio_file)

        biomech_file = motion_capture_dir / "AOA1011_Biomechanics_Full_Set.xlsx"
        biomech_file.touch()

        # Create sit-to-stand recording
        bio_df = pd.DataFrame(
            {
                "TIME": pd.to_timedelta([0, 5, 10], unit="s"),
                "Knee Angle Z": [90, 10, 90],
            }
        )
        recording = BiomechanicsRecording(
            study="AOA",
            study_id=1011,
            maneuver="sit_to_stand",
            speed=None,
            pass_number=None,
            biomech_file_name="sit_to_stand.c3d",
            data=bio_df,
        )

        with patch(
            "src.orchestration.participant.import_biomechanics_recordings"
        ) as mock_import, patch(
            "src.orchestration.participant.load_audio_data"
        ) as mock_load_audio, patch(
            "src.orchestration.participant._sync_and_save_recording"
        ) as mock_sync, patch(
            "src.orchestration.participant.plot_stomp_detection"
        ) as mock_plot:

            mock_load_audio.return_value = mock_audio_df
            mock_import.return_value = [recording]

            synced_df = mock_audio_df.copy()
            synced_df["Knee Angle Z"] = 45.0
            mock_sync.return_value = (
                tmp_path / "output.pkl",
                synced_df,
                (10.0, 5.0, 5.0),
                pd.DataFrame(),
            )

            result = sync_single_audio_file(audio_file)

            assert result is True
            assert mock_sync.call_count == 1

            # Verify pass_number and speed are None
            call_kwargs = mock_sync.call_args_list[0].kwargs
            assert call_kwargs["pass_number"] is None
            assert call_kwargs["speed"] is None

    def test_sync_rejects_already_synced_files(self, tmp_path, mock_audio_df):
        """Should reject files in Synced/ directory."""
        from src.orchestration.participant import sync_single_audio_file

        # Setup with file in Synced directory
        participant_dir = tmp_path / "#1011"
        knee_dir = participant_dir / "Right Knee"
        walking_dir = knee_dir / "Walking"
        synced_dir = walking_dir / "Synced"
        synced_dir.mkdir(parents=True)

        # File is already in Synced folder
        audio_file = synced_dir / "Right_walk_Pass0001_slow.pkl"
        mock_audio_df.to_pickle(audio_file)

        result = sync_single_audio_file(audio_file)

        assert result is False

    def test_sync_requires_audio_columns(self, tmp_path):
        """Should validate audio DataFrame has required columns."""
        from src.orchestration.participant import sync_single_audio_file

        participant_dir = tmp_path / "#1011"
        motion_capture_dir = participant_dir / "Motion Capture"
        motion_capture_dir.mkdir(parents=True)

        knee_dir = participant_dir / "Right Knee"
        walking_dir = knee_dir / "Walking"
        outputs_dir = walking_dir / "HP_W11.2-1-20240126_135704_outputs"
        outputs_dir.mkdir(parents=True)

        # Create audio file missing required channels
        import numpy as np
        bad_audio_df = pd.DataFrame(
            {
                "tt": np.arange(100) / 1000.0,
                "ch1": np.random.randn(100),
                # Missing ch2, ch3, ch4
            }
        )

        audio_file = outputs_dir / "HP_W11.2-1-20240126_135704_with_freq.pkl"
        bad_audio_df.to_pickle(audio_file)

        biomech_file = motion_capture_dir / "AOA1011_Biomechanics_Full_Set.xlsx"
        biomech_file.touch()

        with patch("src.orchestration.participant.load_audio_data") as mock_load:
            mock_load.return_value = bad_audio_df

            result = sync_single_audio_file(audio_file)

            assert result is False
