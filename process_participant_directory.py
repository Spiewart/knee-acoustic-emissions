"""Process and parse study participant directories.

Parses individual study participant directories to validate structure,
extract metadata, and synchronize acoustics with biomechanics data.
Can be used as a module (import functions) or called from command line
to process batch directories.

Usage as module:
    from process_participant_directory import parse_participant_directory
    parse_participant_directory(Path("/path/to/participant/#1011"))

Usage from command line:
    python process_participant_directory.py /path/to/studies
    python process_participant_directory.py /path/to/studies --limit 5
    python process_participant_directory.py /path/to/studies --log output.log
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import pandas as pd

from process_biomechanics import import_biomechanics_recordings
from sync_audio_with_biomechanics import (
    get_audio_stomp_time,
    get_bio_end_time,
    get_bio_start_time,
    get_left_stomp_time,
    get_right_stomp_time,
    get_stomp_time,
    load_audio_data,
    plot_stomp_detection,
    sync_audio_with_biomechanics,
)

if TYPE_CHECKING:
    from models import BiomechanicsCycle


def parse_participant_directory(participant_dir: Path) -> None:
    """Parses a single participant directory to validate its structure
    and contents.

    This function processes all maneuvers for both knees in a transactional
    manner. If ANY maneuver fails to process, NO files will be written.
    """

    study_id = get_study_id_from_directory(participant_dir)
    logging.info("Processing participant with Study ID: %s", study_id)

    check_participant_dir_for_required_files(participant_dir)
    logging.info(
        "All required files found for participant %s", study_id
    )

    # Get the biomechanics file path
    biomechanics_file = participant_dir / "Motion Capture" / (
        f"AOA{study_id}_Biomechanics_Full_Set.xlsx"
    )

    # Collect all synchronized data before writing anything (transactional)
    all_synced_data: list[tuple[Path, pd.DataFrame]] = []
    # Also collect stomp visualization metadata
    stomp_viz_data: list[tuple[Path, pd.DataFrame, tuple]] = []

    # Process each knee (Left and Right)
    for knee_side in ["Left", "Right"]:
        knee_dir = participant_dir / f"{knee_side} Knee"
        logging.info(
            "Processing %s Knee for participant %s",
            knee_side,
            study_id
        )
        knee_synced_data, knee_viz_data = _process_knee_maneuvers(
            knee_dir, biomechanics_file, knee_side
        )
        all_synced_data.extend(knee_synced_data)
        stomp_viz_data.extend(knee_viz_data)

    # If we got here, all processing succeeded - now write all files
    logging.info(
        "All maneuvers processed successfully for participant %s. "
        "Writing %d output files...",
        study_id,
        len(all_synced_data),
    )
    for output_path, synced_df in all_synced_data:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        synced_df.to_pickle(output_path)
        logging.info("Saved synchronized data to %s", output_path)

    # Save stomp visualizations
    for output_path, audio_df, bio_df, synced_df, (audio_stomp, bio_left, bio_right) in stomp_viz_data:
        plot_stomp_detection(audio_df, bio_df, synced_df, audio_stomp, bio_left, bio_right, output_path)

    logging.info("Completed processing participant %s", study_id)


def _process_knee_maneuvers(
    knee_dir: Path,
    biomechanics_file: Path,
    knee_side: str,
) -> tuple[list[tuple[Path, pd.DataFrame]], list[tuple[Path, pd.DataFrame, tuple]]]:
    """Process all maneuvers for a specific knee (Left or Right).

    Synchronizes audio and biomechanics data for all maneuvers but does NOT
    write any files. Returns lists of data tuples for later writing and visualization.

    Args:
        knee_dir: Path to the knee directory (Left Knee or Right Knee)
        biomechanics_file: Path to the biomechanics Excel file
        knee_side: "Left" or "Right"

    Returns:
        Tuple of:
        - List of (output_path, synchronized_dataframe) tuples
        - List of (output_path, audio_df, (audio_stomp, bio_left, bio_right)) tuples for visualization

    Raises:
        Exception: If any maneuver fails to process
    """
    maneuver_mapping: dict[str, Literal[
        "walk",
        "sit_to_stand",
        "flexion_extension",
    ]] = {
        "Walking": "walk",
        "Sit-Stand": "sit_to_stand",
        "Flexion-Extension": "flexion_extension",
    }

    all_synced_data: list[tuple[Path, pd.DataFrame]] = []
    all_viz_data: list[tuple[Path, pd.DataFrame, tuple]] = []

    for maneuver_folder, maneuver_key in maneuver_mapping.items():
        maneuver_dir = knee_dir / maneuver_folder
        if not maneuver_dir.exists():
            logging.warning(
                "Maneuver folder %s not found in %s",
                maneuver_folder,
                knee_dir,
            )
            continue

        # Create Synced folder path (don't create yet)
        synced_dir = maneuver_dir / "Synced"

        maneuver_synced_data, maneuver_viz_data = _sync_maneuver_data(
            maneuver_dir=maneuver_dir,
            synced_dir=synced_dir,
            biomechanics_file=biomechanics_file,
            maneuver_key=cast(
                Literal["walk", "sit_to_stand", "flexion_extension"],
                maneuver_key,
            ),
            knee_side=knee_side,
        )
        all_synced_data.extend(maneuver_synced_data)
        all_viz_data.extend(maneuver_viz_data)

    return all_synced_data, all_viz_data


def _sync_maneuver_data(
    maneuver_dir: Path,
    synced_dir: Path,
    biomechanics_file: Path,
    maneuver_key: Literal["walk", "sit_to_stand", "flexion_extension"],
    knee_side: str,
    with_freq: bool = True,
) -> tuple[list[tuple[Path, pd.DataFrame]], list[tuple[Path, pd.DataFrame, tuple]]]:
    """Synchronize audio and biomechanics data for a specific maneuver.

    Processes and synchronizes data but does NOT write files.

    Args:
        maneuver_dir: Path to the maneuver directory
        synced_dir: Path where synchronized data will be saved (used for path)
        biomechanics_file: Path to the biomechanics Excel file
        maneuver_key: Type of maneuver
        knee_side: "Left" or "Right"

    Returns:
        Tuple of:
        - List of (output_path, synchronized_dataframe) tuples
        - List of (output_path, audio_df, (audio_stomp, bio_left, bio_right)) tuples for visualization

    Raises:
        Exception: If processing fails
    """
    synced_data: list[tuple[Path, pd.DataFrame]] = []
    viz_data: list[tuple[Path, pd.DataFrame, tuple]] = []

    # Get audio file name
    audio_file_name = get_audio_file_name(maneuver_dir, with_freq=False)
    audio_base = Path(audio_file_name).name
    pickle_file_name = get_audio_file_name(maneuver_dir, with_freq=with_freq)
    pickle_base = Path(pickle_file_name).name
    audio_pkl_file = (
        maneuver_dir / f"{audio_base}_outputs" /
        f"{pickle_base}.pkl"
    )

    if not audio_pkl_file.exists():
        raise FileNotFoundError(
            f"Audio pickle file not found: {audio_pkl_file}"
        )

    # Load audio data
    audio_df = load_audio_data(audio_pkl_file)

    # Import biomechanics recordings
    if maneuver_key == "walk":
        # For walking, process each speed
        walk_speeds: tuple[Literal["slow", "medium", "fast"], ...] = (
            "slow",
            "medium",
            "fast",
        )
        for speed in walk_speeds:
            speed_synced_data, speed_viz_data = _process_walk_speed(
                speed=speed,
                biomechanics_file=biomechanics_file,
                audio_df=audio_df,
                maneuver_dir=maneuver_dir,
                synced_dir=synced_dir,
                knee_side=knee_side,
            )
            synced_data.extend(speed_synced_data)
            viz_data.extend(speed_viz_data)
    else:
        # For sit-to-stand and flexion-extension, single recording
        recordings = import_biomechanics_recordings(
            biomechanics_file=biomechanics_file,
            maneuver=maneuver_key,
            speed=None,
        )

        if not recordings:
            raise ValueError(
                f"No biomechanics recordings found for {maneuver_key}"
            )

        # Process first (and only) recording
        recording = recordings[0]
        output_path, synced_df, stomp_times, bio_df = _sync_and_save_recording(
            recording=recording,
            audio_df=audio_df,
            synced_dir=synced_dir,
            biomechanics_file=biomechanics_file,
            maneuver_key=maneuver_key,
            knee_side=knee_side,
            pass_number=None,
            speed=None,
        )
        synced_data.append((output_path, synced_df))
        viz_data.append((output_path, audio_df, bio_df, synced_df, stomp_times))

    return synced_data, viz_data


def _process_walk_speed(
    speed: Literal["slow", "medium", "fast"],
    biomechanics_file: Path,
    audio_df: pd.DataFrame,
    maneuver_dir: Path,
    synced_dir: Path,
    knee_side: str,
) -> tuple[list[tuple[Path, pd.DataFrame]], list[tuple[Path, pd.DataFrame, tuple]]]:
    """Process walking data for a specific speed.

    Processes and synchronizes data but does NOT write files.

    Args:
        speed: One of "slow", "medium", or "fast". Note: "medium" is
            translated to "normal" for event metadata lookups.
        biomechanics_file: Path to the biomechanics Excel file
        audio_df: Audio data DataFrame
        maneuver_dir: Path to the maneuver directory
        synced_dir: Path where synchronized data will be saved (used for path)
        knee_side: "Left" or "Right"

    Returns:
        Tuple of:
        - List of (output_path, synchronized_dataframe) tuples
        - List of (output_path, audio_df, (audio_stomp, bio_left, bio_right)) tuples for visualization
    """
    synced_data: list[tuple[Path, pd.DataFrame]] = []
    viz_data: list[tuple[Path, pd.DataFrame, tuple]] = []

    try:
        recordings = import_biomechanics_recordings(
            biomechanics_file=biomechanics_file,
            maneuver="walk",
            speed=speed,  # type: ignore[arg-type]
        )

        if not recordings:
            logging.info(
                "No biomechanics recordings found for %s at speed %s",
                "walk",
                speed,
            )
            return synced_data, viz_data

        # Process each pass/recording at this speed
        for recording in recordings:
            output_path, synced_df, stomp_times, bio_df = _sync_and_save_recording(
                recording=recording,
                audio_df=audio_df,
                synced_dir=synced_dir,
                biomechanics_file=biomechanics_file,
                maneuver_key="walk",
                knee_side=knee_side,
                pass_number=recording.pass_number,
                # Keep original speed for filename; events normalized later
                speed=speed,
            )
            synced_data.append((output_path, synced_df))
            viz_data.append((output_path, audio_df, bio_df, synced_df, stomp_times))

    except Exception as e:
        logging.error(
            "Error processing walking at %s speed in %s: %s",
            speed,
            maneuver_dir,
            str(e),
        )
        raise

    return synced_data, viz_data


def _sync_and_save_recording(
    recording: "BiomechanicsCycle",  # type: ignore[name-defined]
    audio_df: pd.DataFrame,
    synced_dir: Path,
    biomechanics_file: Path,
    maneuver_key: str,
    knee_side: str,
    pass_number: int | None,
    speed: str | None,
) -> tuple[Path, pd.DataFrame, tuple]:
    """Synchronize a single biomechanics recording with audio.

    Processes and synchronizes data but does NOT write files.

    Args:
        recording: BiomechanicsCycle object with biomechanics data
        audio_df: Audio data DataFrame
        synced_dir: Path where synchronized data will be saved (used for path)
        biomechanics_file: Path to biomechanics Excel file (for event data)
        maneuver_key: Type of maneuver
        knee_side: "Left" or "Right"
        pass_number: Pass number (for walking), None otherwise
        speed: Speed level (for walking), None otherwise

    Returns:
        Tuple of (output_path, synchronized_dataframe, (audio_stomp, bio_left, bio_right))

    Raises:
        Exception: If synchronization fails
    """
    bio_df = recording.data

    # Read event data from biomechanics file
    event_meta_data = _load_event_data(
        biomechanics_file=biomechanics_file,
        maneuver_key=maneuver_key,
    )

    # Get both stomp times from biomechanics for dual-knee disambiguation
    right_stomp_time = get_right_stomp_time(event_meta_data)
    left_stomp_time = get_left_stomp_time(event_meta_data)

    # Get recorded knee from knee_side ("Left" or "Right" -> "left"/"right")
    recorded_knee: Literal["left", "right"] = (
        "left" if knee_side == "Left" else "right"
    )

    # Get audio stomp time with dual-knee disambiguation
    audio_stomp_time = get_audio_stomp_time(
        audio_df,
        recorded_knee=recorded_knee,
        right_stomp_time=right_stomp_time,
        left_stomp_time=left_stomp_time,
    )

    # Get biomechanics stomp time for the recorded knee
    bio_stomp_time = get_stomp_time(
        bio_meta=event_meta_data,
        foot=_get_foot_from_knee_side(knee_side),
    )

    # Normalize speed for event metadata lookups (medium -> normal)
    normalized_speed: Literal["slow", "normal", "fast"] | None = None
    if speed is not None:
        event_speed_map: dict[str, Literal["slow", "normal", "fast"]] = {
            "slow": "slow",
            "medium": "normal",
            "fast": "fast",
        }
        normalized_speed = event_speed_map[speed]

    # Get maneuver timing for pre-merge clipping
    # For walking, use Movement Start/End to clip to specific passes
    # For flexion-extension and sit-to-stand, Movement Start/End often
    # excludes the actual periodic cycles, so sync the full recording
    if maneuver_key == "walk":
        bio_start_time = get_bio_start_time(
            event_metadata=event_meta_data,
            maneuver=cast(
                Literal["walk", "sit_to_stand", "flexion_extension"],
                maneuver_key,
            ),
            speed=normalized_speed,
            pass_number=pass_number,
        )
        bio_end_time = get_bio_end_time(
            event_metadata=event_meta_data,
            maneuver=cast(
                Literal["walk", "sit_to_stand", "flexion_extension"],
                maneuver_key,
            ),
            speed=normalized_speed,
            pass_number=pass_number,
        )
    else:
        # For non-walk maneuvers, clip to Movement Start/End to
        # truncate the synchronized output to the maneuver window.
        bio_start_time = get_bio_start_time(
            event_metadata=event_meta_data,
            maneuver=cast(
                Literal["walk", "sit_to_stand", "flexion_extension"],
                maneuver_key,
            ),
            speed=None,
            pass_number=None,
        )
        bio_end_time = get_bio_end_time(
            event_metadata=event_meta_data,
            maneuver=cast(
                Literal["walk", "sit_to_stand", "flexion_extension"],
                maneuver_key,
            ),
            speed=None,
            pass_number=None,
        )

    # Sync audio with biomechanics, clipping bio to maneuver window
    # ±0.5s before merge to reduce file size (only for walk)
    clipped_df = sync_audio_with_biomechanics(
        audio_stomp_time=audio_stomp_time,
        bio_stomp_time=bio_stomp_time,
        audio_df=audio_df.copy(),
        bio_df=bio_df,
        bio_start_time=bio_start_time,
        bio_end_time=bio_end_time,
        maneuver_key=maneuver_key,
        knee_side=knee_side,
        pass_number=pass_number,
        speed=speed,
    )

    # Trim biomechanics columns to only those for the knee laterality
    # Update their names to remove laterality and "_" delimiter between the
    # biomechanics variable and the axis if present
    trimmed_df = _trim_and_rename_biomechanics_columns(
        clipped_df, knee_side
    )

    # Generate filename (but don't save yet)
    filename = _generate_synced_filename(
        knee_side=knee_side,
        maneuver_key=maneuver_key,
        pass_number=pass_number,
        speed=speed,
    )

    output_path = synced_dir / f"{filename}.pkl"

    # Return path, dataframe, stomp times, and original bio_df for visualization
    stomp_times = (audio_stomp_time, left_stomp_time, right_stomp_time)

    return output_path, trimmed_df, stomp_times, bio_df


def _trim_and_rename_biomechanics_columns(
    df: pd.DataFrame,
    knee_side: str,
) -> pd.DataFrame:
    """Trim biomechanics columns to only those for the desired knee side.

    Removes columns for the opposite knee and renames columns to remove
    the laterality prefix and the "_" delimiter between the biomechanics
    variable and axis.

    Examples:
        "Left Knee Angle_X" -> "Knee Angle X"
        "Right Knee Angle_Z" -> removed (if knee_side is "Left")

    Args:
        df: DataFrame containing synchronized audio and biomechanics data
        knee_side: "Left" or "Right" (case-insensitive)

    Returns:
        DataFrame with only desired knee columns, renamed
    """
    # Normalize knee_side input
    knee_side_lower = knee_side.lower()
    if knee_side_lower not in ("left", "right"):
        raise ValueError(
            f"knee_side must be 'Left' or 'Right', got '{knee_side}'"
        )

    # Determine which prefix to keep and which to remove
    # Capitalize for proper prefix matching
    keep_side = "Left" if knee_side_lower == "left" else "Right"
    remove_side = "Right" if knee_side_lower == "left" else "Left"

    keep_prefix = f"{keep_side} "
    remove_prefix = f"{remove_side} "

    # Create a copy to avoid modifying original
    result_df = df.copy()

    # Remove columns with the opposite knee prefix
    cols_to_remove = [
        col for col in result_df.columns
        if col.startswith(remove_prefix)
    ]
    result_df = result_df.drop(columns=cols_to_remove)

    # Rename columns: remove knee_side prefix and underscore before axis
    rename_mapping = {}
    for col in result_df.columns:
        if col.startswith(keep_prefix):
            # Remove the knee_side prefix (e.g., "Left ")
            without_prefix = col[len(keep_prefix):]
            # Replace underscore before axis with space
            # (e.g., "Angle_X" -> "Angle X")
            renamed = without_prefix.replace("_", " ")
            rename_mapping[col] = renamed

    result_df = result_df.rename(columns=rename_mapping)

    return result_df


def _load_event_data(
    biomechanics_file: Path,
    maneuver_key: str,
) -> pd.DataFrame:
    """Load event metadata from biomechanics Excel file.

    Args:
        biomechanics_file: Path to the biomechanics Excel file
        maneuver_key: Type of maneuver ("walk", "sit_to_stand",
            "flexion_extension")
        pass_number: Pass number (unused for walk, for compatibility)
        speed: Speed (unused for walk, for compatibility)

    Returns:
        DataFrame containing event metadata

    Raises:
        ValueError: If event sheet cannot be found
    """
    raw_study_id = biomechanics_file.stem.split("_")[0]
    study_id = raw_study_id.replace("AOA", "")
    study_prefix = f"AOA{study_id}"

    if maneuver_key == "walk":
        # For all walk speeds, use Walk0001 which has sync events
        event_sheet_name = f"{study_prefix}_Walk0001"
    elif maneuver_key == "sit_to_stand":
        event_sheet_name = f"{study_prefix}_StoS_Events"
    elif maneuver_key == "flexion_extension":
        event_sheet_name = f"{study_prefix}_FE_Events"
    else:
        raise ValueError(f"Unknown maneuver type: {maneuver_key}")

    try:
        event_meta_data = pd.read_excel(
            biomechanics_file,
            sheet_name=event_sheet_name,
        )
        return event_meta_data
    except ValueError as e:
        logging.error(
            "Event sheet '%s' not found in %s: %s",
            event_sheet_name,
            biomechanics_file,
            str(e),
        )
        raise


def _get_foot_from_knee_side(knee_side: str) -> str:
    """Map knee side to foot for stomp time extraction.

    Args:
        knee_side: "Left" or "Right"

    Returns:
        "left" or "right"
    """
    return "left" if knee_side.lower() == "left" else "right"


def _generate_synced_filename(
    knee_side: str,
    maneuver_key: str,
    pass_number: int | None,
    speed: str | None,
) -> str:
    """Generate standardized filename for synchronized data.

    Args:
        knee_side: "Left" or "Right"
        maneuver_key: "walk", "sit_to_stand", or "flexion_extension"
        pass_number: Pass number for walking, None otherwise
        speed: Speed for walking, None otherwise

    Returns:
        Standardized filename without extension
    """
    base = f"{knee_side}_{maneuver_key}"

    if maneuver_key == "walk":
        return f"{base}_Pass{pass_number:04d}_{speed}"
    else:
        return base


def get_study_id_from_directory(path: Path) -> str:
    """Extract the study ID from participant directory path.

    Remove the "#" prefix from the folder name.
    """
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
                f"No Excel file with 'acoustic_file_legend' "
                f"found in {participant_dir}"
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
                f"Required maneuver folder '{maneuver}' "
                f"not found in {knee_dir}"
            )
        knee_subfolder_has_acoustic_files(maneuver_path)


def knee_subfolder_has_acoustic_files(
    maneuver_dir: Path,
) -> None:
    """Checks that the maneuver directory contains acoustic files."""

    try:
        # Validate that a .bin file exists (raises FileNotFoundError if not)
        audio_file_name = get_audio_file_name(maneuver_dir, with_freq=False)
    except FileNotFoundError as e:
        logging.error(
            "Error checking for acoustic files in %s: %s",
            maneuver_dir,
            str(e),
        )
        raise e

    # Get the pickle file name with "with_freq" appended
    pickle_file_name = get_audio_file_name(maneuver_dir, with_freq=True)

    audio_base = Path(audio_file_name).name
    pickle_base = Path(pickle_file_name).name

    # Outputs directory based on audio file name (without _with_freq)
    processed_audio_outputs = Path(
        maneuver_dir / f"{audio_base}_outputs"
    )

    # Assert that there is a pickled DataFrame with the pickle file name
    # (which has "with_freq" appended) and a .pkl extension in the
    # processed_audio_outputs directory
    pkl_file = processed_audio_outputs / f"{pickle_base}.pkl"
    if not pkl_file.exists():
        raise FileNotFoundError(
            f"Processed audio .pkl file '{pkl_file}' "
            f"not found in {processed_audio_outputs}"
        )


def get_audio_file_name(maneuver_dir: Path, with_freq: bool = False) -> str:
    """Get the audio file name from the maneuver directory by looking for the
    original .bin file and pulling the name written in the file.
    Args:
        maneuver_dir: Path to the maneuver directory
        with_freq: If True, looks for file with '_with_freq' suffix
    Returns:
        The audio file name as a string
    Raises:
        FileNotFoundError: If no file is found in the maneuver directory
    """
    # Check that there is a raw .bin acoustics file in the maneuver directory
    bin_files = list(maneuver_dir.glob("*.bin"))
    if not bin_files:
        raise FileNotFoundError(
            f"No .bin acoustic files found in {maneuver_dir}"
        )

    assert len(bin_files) == 1, (
        f"Expected exactly one .bin file in {maneuver_dir}, "
        f"found {len(bin_files)}"
    )

    # Set audio_file to the path of the .bin file minus the file extension
    audio_file_name = str(bin_files[0].with_suffix(""))
    # If with_freq is True, append '_with_freq' to the file name
    if with_freq:
        audio_file_name += "_with_freq"
    return audio_file_name


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
        "Walking": {
            "data": "Walking",
            "pass_metadata": "Walk0001",
            "speeds": ["Slow", "Medium", "Fast"],
        },
        "SitToStand": {
            "data": "SitToStand",
            "events": "StoS_Events",
        },
        "FlexExt": {
            "data": "FlexExt",
            "events": "FE_Events",
        },
    }

    for maneuver_key, maneuver_config in maneuvers.items():
        if maneuver_key == "Walking":
            # Check for pass metadata sheet (single sheet shared across speeds)
            pass_metadata_sheet_name = (
                f"AOA{study_id}_{maneuver_config['pass_metadata']}"
            )
            if pass_metadata_sheet_name not in sheet_names:
                raise ValueError(
                    f"Required sheet '{pass_metadata_sheet_name}' not found "
                    f"in motion capture Excel file '{expected_filename}'"
                )
            # Check for speed data sheets
            for speed_key in maneuver_config["speeds"]:
                speed_sheet_name = (
                    f"AOA{study_id}_{speed_key}_{maneuver_config['data']}"
                )
                if speed_sheet_name not in sheet_names:
                    raise ValueError(
                        f"Required sheet '{speed_sheet_name}' not found "
                        f"in motion capture Excel file '{expected_filename}'"
                    )
        else:
            maneuver_sheet_name = (
                f"AOA{study_id}_{maneuver_config['data']}"
            )

            if maneuver_sheet_name not in sheet_names:
                raise ValueError(
                    f"Required sheet '{maneuver_sheet_name}' not found "
                    f"in motion capture Excel file '{expected_filename}'"
                )
            events_sheet_name = f"AOA{study_id}_{maneuver_config['events']}"
            if events_sheet_name not in sheet_names:
                raise ValueError(
                    f"Required sheet '{events_sheet_name}' not found "
                    f"in motion capture Excel file '{expected_filename}'"
                )


# ============================================================================
# Command-line interface functions for batch processing
# ============================================================================


def find_participant_directories(path: Path) -> list[Path]:
    """Return a sorted list of participant directories.

    A valid participant directory is a subdirectory whose name starts with "#"
    (e.g., "#1011", "#2024").

    Args:
        path: Directory to search (recursively searches subdirectories)

    Returns:
        Sorted list of participant directory paths
    """
    if not path.is_dir():
        return []

    # Find all subdirectories that start with "#"
    participant_dirs = [
        d for d in path.iterdir()
        if d.is_dir() and d.name.startswith("#")
    ]

    return sorted(participant_dirs)


def setup_logging(log_file: Path | None = None) -> None:
    """Configure logging to both console and optional file.

    Args:
        log_file: Optional path to write log file
    """
    log_level = logging.INFO
    log_format = "%(asctime)s %(levelname)s: %(message)s"

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[logging.StreamHandler()],
    )

    # Add file handler if specified
    if log_file:
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setLevel(log_level)
        formatter = logging.Formatter(log_format)
        fh.setFormatter(formatter)
        logging.getLogger().addHandler(fh)


def process_participant(participant_dir: Path) -> bool:
    """Process a single participant directory.

    Args:
        participant_dir: Path to the participant directory

    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        study_id = participant_dir.name.lstrip("#")
        logging.info("Processing participant #%s", study_id)

        # Validate directory structure
        check_participant_dir_for_required_files(participant_dir)
        logging.info(
            "Directory validation passed for participant #%s", study_id
        )

        # Parse and process all maneuvers
        parse_participant_directory(participant_dir)
        logging.info(
            "Successfully completed processing participant #%s", study_id
        )
        return True

    except FileNotFoundError as e:
        logging.error(
            "Validation error for %s: %s", participant_dir.name, str(e)
        )
        return False
    except Exception as e:  # pylint: disable=broad-except
        logging.error(
            "Unexpected error processing %s: %s",
            participant_dir.name,
            str(e),
        )
        return False


def _extract_metadata_from_audio_path(
    audio_pkl_path: Path,
) -> dict:
    """Extract participant, knee, and maneuver info from audio file path.

    Walks backwards from the audio file path to find the participant
    directory and extract metadata.

    Args:
        audio_pkl_path: Path to the audio pickle file
            (e.g., /path/to/#1011/Left Knee/Walking/audio_outputs/audio.pkl)

    Returns:
        Dictionary with keys: participant_dir, knee_side, maneuver_key

    Raises:
        ValueError: If path structure is invalid
    """
    audio_pkl_path = Path(audio_pkl_path)

    if not audio_pkl_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_pkl_path}")

    # Walk backwards to find participant directory (one with "#" prefix)
    current = audio_pkl_path.parent
    participant_dir = None

    while current != current.parent:  # Stop at root
        if current.name.startswith("#"):
            participant_dir = current
            break
        current = current.parent

    if not participant_dir:
        raise ValueError(
            f"Could not find participant directory "
            f"(with '#' prefix) in path: {audio_pkl_path}"
        )

    # Extract knee_side from path (should be "Left Knee" or "Right Knee")
    knee_side = None
    for part in audio_pkl_path.parts:
        if part in ("Left Knee", "Right Knee"):
            knee_side = part.split()[0]  # "Left" or "Right"
            break

    if not knee_side:
        raise ValueError(
            f"Could not determine knee side from path: {audio_pkl_path}"
        )

    # Extract maneuver from path
    maneuver_folder_map: dict[str, Literal[
        "walk",
        "sit_to_stand",
        "flexion_extension",
    ]] = {
        "Walking": "walk",
        "Sit-Stand": "sit_to_stand",
        "Flexion-Extension": "flexion_extension",
    }

    maneuver_key = None
    for folder_name, maneuver in maneuver_folder_map.items():
        if folder_name in audio_pkl_path.parts:
            maneuver_key = maneuver
            break

    if not maneuver_key:
        raise ValueError(
            f"Could not determine maneuver type from path: {audio_pkl_path}"
        )

    return {
        "participant_dir": participant_dir,
        "knee_side": knee_side,
        "maneuver_key": maneuver_key,
    }


def sync_single_audio_file(
    audio_pkl_path: str | Path,
) -> bool:
    """Synchronize a single audio file with biomechanics.

    Works with unsynced audio pickle files from the _outputs/ folder.
    Do NOT use with files already in the Synced/ folder.

    Args:
        audio_pkl_path: Path to the unsynced audio pickle file
            (should be in {maneuver}/_outputs/, not Synced/)

    Returns:
        True if successful, False otherwise
    """
    try:
        audio_pkl_path = Path(audio_pkl_path)
        logging.info("Synchronizing audio file: %s", audio_pkl_path)

        # Check if file is in Synced folder (already synchronized)
        if "Synced" in audio_pkl_path.parts:
            logging.error(
                "File is already synchronized (in Synced/ folder). "
                "Use --sync-single with unsynced audio files from _outputs/"
            )
            return False

        # Extract metadata from path
        metadata = _extract_metadata_from_audio_path(audio_pkl_path)
        participant_dir = metadata["participant_dir"]
        knee_side = metadata["knee_side"]
        maneuver_key = metadata["maneuver_key"]

        logging.info(
            "Extracted metadata: knee=%s, maneuver=%s, participant=%s",
            knee_side,
            maneuver_key,
            participant_dir.name,
        )

        # Get biomechanics file
        study_id = participant_dir.name.lstrip("#")
        biomechanics_file = participant_dir / "Motion Capture" / (
            f"AOA{study_id}_Biomechanics_Full_Set.xlsx"
        )

        if not biomechanics_file.exists():
            logging.error(
                "Biomechanics file not found: %s", biomechanics_file
            )
            return False

        # Load audio data
        audio_df = load_audio_data(audio_pkl_path)

        # Validate audio data has required columns
        required_audio_cols = {"tt", "ch1", "ch2", "ch3", "ch4"}
        if not required_audio_cols.issubset(audio_df.columns):
            missing = required_audio_cols - set(audio_df.columns)
            logging.error(
                "Audio file missing required columns: %s. "
                "Are you sure this is an unsynced audio file?",
                missing,
            )
            return False

        # Import biomechanics recordings
        if maneuver_key == "walk":
            # Infer speed from the path or use all speeds
            # For now, try to infer from parent directory naming
            recordings = import_biomechanics_recordings(
                biomechanics_file=biomechanics_file,
                maneuver="walk",
                speed="slow",  # type: ignore[arg-type]
            )
            if not recordings:
                logging.error(
                    "No biomechanics recordings found for walk/slow"
                )
                return False
            recording = recordings[0]
            pass_number = recording.pass_number
            speed = "slow"
        else:
            recordings = import_biomechanics_recordings(
                biomechanics_file=biomechanics_file,
                maneuver=cast(
                    Literal["sit_to_stand", "flexion_extension"],
                    maneuver_key,
                ),
                speed=None,
            )
            if not recordings:
                logging.error(
                    "No biomechanics recordings found for %s", maneuver_key
                )
                return False
            recording = recordings[0]
            pass_number = None
            speed = None

        # Determine output directory
        maneuver_dir = None
        maneuver_folder_map_rev = {
            "walk": "Walking",
            "sit_to_stand": "Sit-Stand",
            "flexion_extension": "Flexion-Extension",
        }
        maneuver_folder = maneuver_folder_map_rev[maneuver_key]
        knee_dir = participant_dir / f"{knee_side} Knee"
        maneuver_dir = knee_dir / maneuver_folder
        synced_dir = maneuver_dir / "Synced"

        # Sync the recording
        output_path, synced_df, stomp_times, bio_df = _sync_and_save_recording(
            recording=recording,
            audio_df=audio_df,
            synced_dir=synced_dir,
            biomechanics_file=biomechanics_file,
            maneuver_key=maneuver_key,
            knee_side=knee_side,
            pass_number=pass_number,
            speed=speed,
        )

        # Write the file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        synced_df.to_pickle(output_path)
        logging.info("Saved synchronized data to %s", output_path)

        # Generate stomp visualization
        audio_stomp, bio_left, bio_right = stomp_times
        plot_stomp_detection(audio_df, bio_df, synced_df, audio_stomp, bio_left, bio_right, output_path)
        logging.info("Saved stomp detection visualization")
        return True

    except Exception as e:  # pylint: disable=broad-except
        logging.error("Error syncing audio file: %s", str(e))
        return False


def main() -> None:
    """Main entry point for command-line script."""
    parser = argparse.ArgumentParser(
        description=(
            "Process all participant directories in a folder. "
            "Each directory should be named with format '#<study_id>'."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python process_participant_directory.py /path/to/studies\n"
            "  python process_participant_directory.py /path/to/studies "
            "--limit 5\n"
            "  python process_participant_directory.py --sync-single "
            "/path/to/audio.pkl"
        ),
    )

    parser.add_argument(
        "path",
        nargs="?",
        help=(
            "Path to directory containing participant folders "
            "(e.g., #1011, #2024), or to audio file with --sync-single"
        ),
    )
    parser.add_argument(
        "--sync-single",
        action="store_true",
        help="Sync a single audio file (PATH should be audio file path)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N participant directories (0 = all, default: 0)",
    )
    parser.add_argument(
        "--participant",
        nargs="+",
        help=(
            "One or more participant folder names to process within PATH "
            "(with or without leading '#', e.g., 1011 #2024)"
        ),
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Optional path to write detailed log file",
    )

    args = parser.parse_args()

    # Set up logging
    log_file = Path(args.log) if args.log else None
    setup_logging(log_file)

    # Handle single-file sync
    if args.sync_single:
        if not args.path:
            logging.error(
                "--sync-single requires PATH argument (audio file path)"
            )
            return
        success = sync_single_audio_file(args.path)
        if not success:
            logging.error("Failed to sync audio file: %s", args.path)
        return

    # Validate input path
    if not args.path:
        parser.print_help()
        return

    path = Path(args.path)
    if not path.exists():
        logging.error("Path does not exist: %s", path)
        return
    if not path.is_dir():
        logging.error("Path is not a directory: %s", path)
        return

    # Find participant directories
    participants = find_participant_directories(path)
    if not participants:
        logging.warning(
            "No participant directories found in %s "
            "(looking for folders named #<study_id>)",
            path,
        )
        return

    # Filter to specific participants if requested
    if args.participant:
        requested = {p.lstrip("#") for p in args.participant}
        participants = [d for d in participants if d.name.lstrip("#") in requested]
        if not participants:
            logging.warning(
                "No matching participant directories found for %s in %s",
                sorted(requested),
                path,
            )
            return

    # Apply limit if specified
    if args.limit > 0:
        participants = participants[: args.limit]

    logging.info(
        "Found %d participant directory(ies) to process", len(participants)
    )

    # Process each participant
    success_count = 0
    failure_count = 0

    for participant_dir in participants:
        if process_participant(participant_dir):
            success_count += 1
        else:
            failure_count += 1

    # Summary
    logging.info(
        "Processing complete: %d succeeded, %d failed",
        success_count,
        failure_count,
    )

    if failure_count > 0:
        logging.warning(
            "Some participants failed processing; check logs for details"
        )


if __name__ == "__main__":
    main()
