"""Parses a study participant directory and returns a list of
SynchronizedCycle Pydantic models representing synchronized
acoustics and biomechanics data with associated metadata."""

import logging
from pathlib import Path

import pandas as pd


def parse_participant_directories(dir_path: str) -> None:

    # Iterate over all the folders in the project directory
    project_dir = Path(dir_path)
    participant_directories = [
        d for d in project_dir.iterdir() if d.is_dir()
    ]
    logging.info(
        "Found %s participant directories to process.",
        len(participant_directories),
    )

    for participant_dir in participant_directories:
        parse_participant_directory(participant_dir)


def parse_participant_directory(participant_dir: Path) -> None:
    """Parses a single participant directory to validate its structure
    and contents."""

    study_id = get_study_id_from_directory(participant_dir)
    logging.info("Processing participant with Study ID: %s", study_id)

    check_participant_dir_for_required_files(participant_dir)
    logging.info(
        "All required files found for participant %s", study_id
    )


def get_study_id_from_directory(path: Path) -> str:
    """Extracts the study ID from the participant directory path.
    Need to remove the "#" prefix from the folder name."""
    return path.name.lstrip("#")


def check_participant_dir_for_required_files(
    participant_dir: Path,
) -> None:
    """Checks that the participant directory contains all required files."""

    dir_has_acoustic_file_legend(participant_dir)
    participant_dir_has_top_level_folders(participant_dir)
    knee_folder_has_subfolder_each_maneuver(participant_dir/"Left Knee")
    knee_folder_has_subfolder_each_maneuver(participant_dir/"Right Knee")
    motion_capture_folder_has_required_data(participant_dir/"Motion Capture")


def dir_has_acoustic_file_legend(participant_dir: Path):

    # Check that the directory contains and Excel file that
    # contains "acoustic_file_legend" in the filename
    try:
        excel_files = list(
            participant_dir.glob("*acoustic_file_legend*.xlsx")
        )
        if not excel_files:
            raise FileNotFoundError(
                f"No Excel file with 'acoustic_file_legend' found in {participant_dir}"
            )
    except Exception as e:
        logging.error(
            "Error checking for Excel file in %s: %s",
            participant_dir,
            str(e),
        )
        raise e


def participant_dir_has_top_level_folders(
    participant_dir: Path,
) -> None:
    """Checks that the participant directory contains the required
    top-level folders."""

    required_folders = [
        "Left Knee",
        "Right Knee",
        "Motion Capture",
    ]

    for folder in required_folders:
        folder_path = participant_dir / folder
        if not folder_path.exists() or not folder_path.is_dir():
            raise FileNotFoundError(
                f"Required folder '{folder}' not found in {participant_dir}"
            )


def knee_folder_has_subfolder_each_maneuver(knee_dir: Path) -> None:
    """Checks that the knee directory contains subfolders for each maneuver."""

    required_maneuvers = [
        "Flexion-Extension",
        "Sit-Stand",
        "Walking",
    ]

    for maneuver in required_maneuvers:
        maneuver_path = knee_dir / maneuver
        if not maneuver_path.exists() or not maneuver_path.is_dir():
            raise FileNotFoundError(
                f"Required maneuver folder '{maneuver}' not found in {knee_dir}"
            )
        knee_subfolder_has_acoustic_files(maneuver_path)


def knee_subfolder_has_acoustic_files(
    maneuver_dir: Path,
) -> None:
    """Checks that the maneuver directory contains acoustic files."""

    try:
        audio_file_name = get_audio_file_name(maneuver_dir)
    except FileNotFoundError as e:
        logging.error(
            "Error checking for acoustic files in %s: %s",
            maneuver_dir,
            str(e),
        )
        raise e

    processed_audio_outputs = Path(
        maneuver_dir / (audio_file_name + "_outputs")
    )

    # Assert that there is a pickled DataFrame with the same name as the audio
    # file but with a .pkl extension in the processed_audio_outputs directory
    pkl_file = processed_audio_outputs / (audio_file_name + ".pkl")
    if not pkl_file.exists():
        raise FileNotFoundError(
            f"Processed audio .pkl file '{pkl_file}' "
            f"not found in {processed_audio_outputs}"
        )
    # TODO: Add checks for the correct columns in the pickled DataFrame


def get_audio_file_name(maneuver_dir: Path) -> str:
    # Check that there is a raw .bin acoustics file in the maneuver directory
    bin_files = list(maneuver_dir.glob("*.bin"))
    if not bin_files:
        raise FileNotFoundError(
            f"No .bin acoustic files found in {maneuver_dir}"
        )

    assert len(bin_files) == 1, (
        f"Expected exactly one .bin file in {maneuver_dir}, found {len(bin_files)}"
    )

    # Set audio_file to the path of the .bin file minus the file extension
    return str(bin_files[0].with_suffix(""))


def motion_capture_folder_has_required_data(
    motion_capture_dir: Path,
) -> None:
    """Checks that the motion capture directory contains the single
    required Excel file, but that the required Excel file has all the
    required elements in it."""

    study_id = get_study_id_from_directory(motion_capture_dir.parent)

    expected_filename = f"AOA{study_id}_Biomechanics_Full_Set.xlsx"
    excel_file_path = motion_capture_dir / expected_filename
    if not excel_file_path.exists():
        raise FileNotFoundError(
            f"Required motion capture Excel file '{expected_filename}' "
            f"not found in {motion_capture_dir}"
        )

    # Get all the sheet names in the Excel file
    xls = pd.ExcelFile(excel_file_path)
    sheet_names = xls.sheet_names

    maneuvers = {
        "Walking": "Walking",
        "SitToStand": "StoS",
        "FlexExt": "FE",
    }

    speeds = {
        "Slow": "SS",
        "Medium": "NS",
        "Fast": "FS",
    }

    for maneuver_key, maneuver_value in maneuvers.items():
        events_sheet_name = f"AOA{study_id}_{maneuver_value}_Events"
        if events_sheet_name not in sheet_names:
            raise ValueError(
                f"Required sheet '{events_sheet_name}' not found "
                f"in motion capture Excel file '{expected_filename}'"
            )
        if maneuver_key == "Walking":
            for speed_key, _ in speeds.items():
                speed_sheet_name = f"AOA{study_id}_{speed_key}_{maneuver_value}"
                if speed_sheet_name not in sheet_names:
                    raise ValueError(
                        f"Required sheet '{speed_sheet_name}' not found "
                        f"in motion capture Excel file '{expected_filename}'"
                    )
        else:
            maneuver_sheet_name = f"AOA{study_id}_{maneuver_value}"
            if maneuver_sheet_name not in sheet_names:
                raise ValueError(
                    f"Required sheet '{maneuver_sheet_name}' not found "
                    f"in motion capture Excel file '{expected_filename}'"
                )
