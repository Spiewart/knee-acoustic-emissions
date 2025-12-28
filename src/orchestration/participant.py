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
import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Literal, Optional, cast

import pandas as pd

from src.biomechanics.importers import import_biomechanics_recordings
from src.orchestration.processing_log import (
    KneeProcessingLog,
    ManeuverProcessingLog,
    create_audio_record_from_data,
    create_biomechanics_record_from_data,
    create_cycles_record_from_data,
    create_sync_record_from_data,
)
from src.synchronization.quality_control import find_synced_files, perform_sync_qc
from src.synchronization.sync import (
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
    from src.models import BiomechanicsCycle


def parse_participant_directory(participant_dir: Path) -> None:
    """Parse and process a complete participant study directory.

    Orchestrates the full processing pipeline for a single participant:
    1. Validates directory structure and required files.
    2. Processes all maneuvers (walk, sit-to-stand, flexion-extension) for both knees.
    3. Synchronizes audio with biomechanics using stomp detection.
    4. Applies quality control filtering to identify clean vs. outlier cycles.
    5. Writes synchronized data and visualization plots (transactional: all or nothing).

    **Transactional behavior**: If ANY maneuver fails to process, NO output files
    are written. This ensures datasets are either fully processed or fully skipped.

    Args:
        participant_dir: Path to the participant directory (e.g., "studies/#1011").
                        Expected to contain "Motion Capture" and "Left Knee"/"Right Knee" subdirectories.

    Raises:
        FileNotFoundError: If required files or directory structure is missing.
        ValueError: If file contents are invalid or synchronization fails.
    """

    study_id = get_study_id_from_directory(participant_dir)
    logging.info("Processing participant with Study ID: %s", study_id)

    check_participant_dir_for_required_files(participant_dir)
    logging.info(
        "All required files found for participant %s", study_id
    )

    # Get the biomechanics file path
    motion_capture_dir = participant_dir / "Motion Capture"
    biomechanics_file = _find_excel_file(
        motion_capture_dir,
        f"AOA{study_id}_Biomechanics_Full_Set",
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


def _save_or_update_processing_log(
    study_id: str,
    knee_side: Literal["Left", "Right"],
    maneuver_key: Literal["walk", "sit_to_stand", "flexion_extension"],
    maneuver_dir: Path,
    audio_pkl_file: Optional[Path] = None,
    audio_df: Optional[pd.DataFrame] = None,
    audio_metadata: Optional[Dict] = None,
    biomechanics_file: Optional[Path] = None,
    biomechanics_recordings: Optional[list] = None,
    synced_data: Optional[list[tuple[Path, pd.DataFrame, tuple]]] = None,
) -> None:
    """Save or update the processing log for a maneuver.

    Args:
        study_id: Study participant ID
        knee_side: "Left" or "Right"
        maneuver_key: Maneuver type
        maneuver_dir: Path to maneuver directory
        audio_pkl_file: Path to audio pickle file
        audio_df: Audio DataFrame
        audio_metadata: Audio metadata dictionary
        biomechanics_file: Path to biomechanics file
        biomechanics_recordings: List of biomechanics recordings
        synced_data: List of (output_path, synced_df, stomp_times) tuples
    """
    try:
        # Get or create processing log
        log = ManeuverProcessingLog.get_or_create(
            study_id=study_id,
            knee_side=knee_side,
            maneuver=maneuver_key,
            maneuver_directory=maneuver_dir,
        )

        # Update audio record if data provided
        if audio_df is not None and audio_pkl_file is not None:
            audio_bin_path = audio_pkl_file.parent.parent / f"{audio_pkl_file.stem.replace('_with_freq', '')}.bin"
            if not audio_bin_path.exists():
                # Try to find any .bin file
                bin_files = list(maneuver_dir.glob("*.bin"))
                audio_bin_path = bin_files[0] if bin_files else None

            audio_record = create_audio_record_from_data(
                audio_file_name=audio_pkl_file.stem,
                audio_df=audio_df,
                audio_bin_path=audio_bin_path,
                audio_pkl_path=audio_pkl_file,
                metadata=audio_metadata,
            )
            log.update_audio_record(audio_record)

        # Update biomechanics record if data provided
        if biomechanics_recordings is not None and biomechanics_file is not None:
            bio_record = create_biomechanics_record_from_data(
                biomechanics_file=biomechanics_file,
                recordings=biomechanics_recordings,
                sheet_name=f"{maneuver_key}_data",
            )
            log.update_biomechanics_record(bio_record)

        # Update synchronization records if data provided
        if synced_data is not None:
            for output_path, synced_df, (audio_stomp, bio_left, bio_right) in synced_data:
                # Extract pass number and speed from filename
                filename = output_path.stem
                pass_number = None
                speed = None
                if "pass" in filename.lower():
                    import re
                    match = re.search(r'pass(\d+)', filename.lower())
                    if match:
                        pass_number = int(match.group(1))
                if "slow" in filename.lower():
                    speed = "slow"
                elif "medium" in filename.lower() or "normal" in filename.lower():
                    speed = "medium"
                elif "fast" in filename.lower():
                    speed = "fast"

                sync_record = create_sync_record_from_data(
                    sync_file_name=filename,
                    synced_df=synced_df,
                    audio_stomp_time=audio_stomp,
                    bio_left_stomp_time=bio_left,
                    bio_right_stomp_time=bio_right,
                    knee_side=knee_side.lower(),
                    pass_number=pass_number,
                    speed=speed,
                )
                log.add_synchronization_record(sync_record)

        # Save the log
        log.save_to_excel()

        # Update knee-level log
        try:
            study_id_for_knee = study_id
            knee_log = KneeProcessingLog.get_or_create(
                study_id=study_id_for_knee,
                knee_side=knee_side,
                knee_directory=maneuver_dir.parent,
            )
            knee_log.update_maneuver_summary(maneuver_key, log)
            knee_log.save_to_excel()
        except Exception as e:
            logging.warning(f"Failed to update knee-level log: {e}")

    except Exception as e:
        logging.warning(f"Failed to save processing log for {knee_side} {maneuver_key}: {e}")


def _process_knee_maneuvers(
    knee_dir: Path,
    biomechanics_file: Path,
    knee_side: str,
) -> tuple[list[tuple[Path, pd.DataFrame]], list[tuple[Path, pd.DataFrame, tuple]]]:
    """Process all maneuvers for a specific knee (Left or Right).

    Iterates through all maneuver types (walk, sit-to-stand, flexion-extension),
    synchronizes audio and biomechanics data for each, but does NOT write files.
    Data is collected for later transactional write.

    Args:
        knee_dir: Path to the knee directory (Left Knee or Right Knee).
        biomechanics_file: Path to the biomechanics Excel file.
        knee_side: "Left" or "Right" (used for logging and lookups).

    Returns:
        Tuple of:
        - List of (output_path, synchronized_dataframe) tuples for file writing.
        - List of (output_path, audio_df, bio_df, synced_df, stomp_times) tuples for visualization.

    Raises:
        Exception: If any maneuver fails to process (transactional semantics).
    """
    maneuver_keys: tuple[Literal[
        "walk",
        "sit_to_stand",
        "flexion_extension",
    ], ...] = (
        "walk",
        "sit_to_stand",
        "flexion_extension",
    )

    all_synced_data: list[tuple[Path, pd.DataFrame]] = []
    all_viz_data: list[tuple[Path, pd.DataFrame, tuple]] = []

    for maneuver_key in maneuver_keys:
        maneuver_dir = _find_maneuver_dir(knee_dir, maneuver_key)
        if maneuver_dir is None:
            logging.warning(
                "Maneuver folder for %s not found in %s",
                maneuver_key,
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

    # Load audio metadata if available
    audio_metadata = None
    meta_json_path = audio_pkl_file.parent / f"{audio_base}_meta.json"
    if meta_json_path.exists():
        try:
            with open(meta_json_path, 'r') as f:
                audio_metadata = json.load(f)
        except Exception:
            pass

    # Import biomechanics recordings
    all_recordings = []
    if maneuver_key == "walk":
        # For walking, process each speed
        walk_speeds: tuple[Literal["slow", "medium", "fast"], ...] = (
            "slow",
            "medium",
            "fast",
        )
        for speed in walk_speeds:
            speed_synced_data, speed_viz_data, speed_recordings = _process_walk_speed(
                speed=speed,
                biomechanics_file=biomechanics_file,
                audio_df=audio_df,
                maneuver_dir=maneuver_dir,
                synced_dir=synced_dir,
                knee_side=knee_side,
            )
            synced_data.extend(speed_synced_data)
            viz_data.extend(speed_viz_data)
            all_recordings.extend(speed_recordings)
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

        all_recordings = recordings
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

    # Save processing log for this maneuver
    try:
        # Get study ID from parent directory structure
        participant_dir = maneuver_dir.parent.parent
        study_id = participant_dir.name.lstrip("#")

        # Prepare synced data for logging (convert viz_data to appropriate format)
        synced_log_data = [
            (output_path, synced_df, stomp_times)
            for output_path, _, _, synced_df, stomp_times in viz_data
        ]

        _save_or_update_processing_log(
            study_id=study_id,
            knee_side=cast(Literal["Left", "Right"], knee_side),
            maneuver_key=maneuver_key,
            maneuver_dir=maneuver_dir,
            audio_pkl_file=audio_pkl_file,
            audio_df=audio_df,
            audio_metadata=audio_metadata,
            biomechanics_file=biomechanics_file,
            biomechanics_recordings=all_recordings,
            synced_data=synced_log_data,
        )
    except Exception as e:
        logging.warning(f"Failed to update processing log: {e}")

    return synced_data, viz_data


def _process_walk_speed(
    speed: Literal["slow", "medium", "fast"],
    biomechanics_file: Path,
    audio_df: pd.DataFrame,
    maneuver_dir: Path,
    synced_dir: Path,
    knee_side: str,
) -> tuple[list[tuple[Path, pd.DataFrame]], list[tuple[Path, pd.DataFrame, tuple]], list]:
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
        - List of (output_path, audio_df, bio_df, synced_df, (audio_stomp, bio_left, bio_right)) tuples for visualization
        - List of biomechanics recordings
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
            return synced_data, viz_data, []

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

    return synced_data, viz_data, recordings


def _sync_and_save_recording(
    recording: "BiomechanicsCycle",  # type: ignore[name-defined]
    audio_df: pd.DataFrame,
    synced_dir: Path,
    biomechanics_file: Path,
    maneuver_key: str,
    knee_side: str,
    pass_number: Optional[int],
    speed: Optional[str],
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
    # Normalize speed for event metadata lookups
    # Handle both API speeds (slow/medium/fast) and internal model speeds (slow/normal/fast)
    normalized_speed: Optional[Literal["slow", "normal", "fast"]] = None
    if speed is not None:
        event_speed_map: dict[str, Literal["slow", "normal", "fast"]] = {
            "slow": "slow",
            "medium": "normal",  # API speed -> model speed
            "normal": "normal",  # Model speed -> model speed (passthrough)
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
        return _normalize_event_metadata_columns(event_meta_data)
    except ValueError as e:
        logging.error(
            "Event sheet '%s' not found in %s: %s",
            event_sheet_name,
            biomechanics_file,
            str(e),
        )
        raise


def _normalize_event_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize biomechanics event metadata columns.

    Ensures required columns are named "Event Info" and "Time (sec)" even when
    the sheet uses slight variants such as "Event", "EventInfo", "Time(sec)", or
    "Time". Does not drop any columns; only renames when a clear match exists.
    """

    def _norm(col: str) -> str:
        return re.sub(r"[^a-z0-9]", "", col.lower())

    norm_map = {_norm(col): col for col in df.columns}

    event_col = None
    for key in ("eventinfo", "event", "eventname"):
        if key in norm_map:
            event_col = norm_map[key]
            break

    time_col = None
    for key in ("timesec", "time", "timesecs", "timesecond", "timeseconds"):
        if key in norm_map:
            time_col = norm_map[key]
            break

    rename_map: dict[str, str] = {}
    if event_col and event_col != "Event Info":
        rename_map[event_col] = "Event Info"
    if time_col and time_col != "Time (sec)":
        rename_map[time_col] = "Time (sec)"

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


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
    pass_number: Optional[int],
    speed: Optional[str],
) -> str:
    """Generate standardized filename for synchronized data.

    Args:
        knee_side: "Left" or "Right"
        maneuver_key: "walk", "sit_to_stand", or "flexion_extension"
        pass_number: Pass number for walking, None otherwise
        speed: Speed for walking ("slow", "normal", "fast"), None otherwise

    Returns:
        Standardized filename without extension
    """
    base = f"{knee_side}_{maneuver_key}"

    if maneuver_key == "walk":
        # Map "normal" to "medium" for filename consistency
        # (model uses "normal" internally, but files use "medium")
        filename_speed_map = {
            "slow": "slow",
            "normal": "medium",
            "fast": "fast",
        }
        filename_speed = filename_speed_map.get(speed, speed) if speed else speed
        return f"{base}_Pass{pass_number:04d}_{filename_speed}"
    else:
        return base


def _find_excel_file(
    directory: Path,
    filename_pattern: str,
) -> Optional[Path]:
    """Find an Excel file (.xlsx or .xlsm) matching the given pattern.

    Args:
        directory: Directory to search in.
        filename_pattern: Glob pattern for the filename (e.g., "*acoustic_file_legend*").

    Returns:
        Path to the Excel file, or None if not found.
    """
    # Try .xlsx first, then .xlsm
    for extension in [".xlsx", ".xlsm"]:
        files = list(directory.glob(f"{filename_pattern}{extension}"))
        if files:
            return files[0]
    return None


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


def check_participant_dir_for_bin_stage(participant_dir: Path) -> None:
    """Bin-stage validation: ensure raw .bin exists but do not require processed outputs."""

    dir_has_acoustic_file_legend(participant_dir)
    participant_dir_has_top_level_folders(participant_dir)
    knee_folder_has_subfolder_each_maneuver(participant_dir/"Left Knee", require_processed=False)
    knee_folder_has_subfolder_each_maneuver(participant_dir/"Right Knee", require_processed=False)


def dir_has_acoustic_file_legend(participant_dir: Path) -> None:
    """Verify that participant directory contains acoustic_file_legend (.xlsx or .xlsm)."""
    # Check that the directory contains an Excel file that
    # contains "acoustic_file_legend" in the filename
    try:
        excel_file = _find_excel_file(participant_dir, "*acoustic_file_legend*")
        if excel_file is None:
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


def knee_folder_has_subfolder_each_maneuver(knee_dir: Path, require_processed: bool = True) -> None:
    """Checks that the knee directory contains subfolders for each maneuver."""
    for maneuver_key in ("walk", "sit_to_stand", "flexion_extension"):
        maneuver_path = _find_maneuver_dir(knee_dir, maneuver_key)
        if maneuver_path is None:
            raise FileNotFoundError(
                f"Required maneuver folder for '{maneuver_key}' "
                f"not found in {knee_dir}"
            )
        knee_subfolder_has_acoustic_files(maneuver_path, require_processed=require_processed)


def knee_subfolder_has_acoustic_files(
    maneuver_dir: Path,
    *,
    require_processed: bool = True,
) -> None:
    """Checks that the maneuver directory contains acoustic files.

    When require_processed=False, only requires the raw .bin file; this is used by
    the bin-stage to allow generation of processed outputs.
    """

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

    if not require_processed:
        return

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


def _normalize_folder_name(name: str) -> str:
    """Normalize maneuver folder names to compare variants."""
    name = name.lower()
    name = name.replace("_", "-").replace(" ", "-")
    name = re.sub(r"-+", "-", name)
    return name


_MANEUVER_ALIAS_MAP: dict[str, set[str]] = {
    "walk": {"walking", "walk"},
    "sit_to_stand": {
        "sit-stand",
        "sit-to-stand",
        "sit_to_stand",
        "sitstand",
        "sit to stand",
        "sittostand",
    },
    "flexion_extension": {
        "flexion-extension",
        "flexion_extension",
        "flexion extension",
        "flexionextension",
    },
}


def _find_maneuver_dir(knee_dir: Path, maneuver_key: str) -> Optional[Path]:
    """Find a maneuver directory using alias matching."""
    aliases = _MANEUVER_ALIAS_MAP.get(maneuver_key, set())
    for child in knee_dir.iterdir():
        if not child.is_dir():
            continue
        norm = _normalize_folder_name(child.name)
        if norm in aliases:
            return child
    return None


def _load_acoustics_file_names(participant_dir: Path) -> dict[tuple[str, str], str]:
    """Load acoustics file names from the legend for each knee/maneuver.

    Returns mapping: (knee, maneuver_key) -> file base name (without .bin extension).
    Best-effort; missing entries are skipped.
    """
    legend_path = _find_excel_file(participant_dir, "*acoustic_file_legend*")
    if legend_path is None:
        return {}
    mapping: dict[tuple[str, str], str] = {}
    from src.audio.parsers import get_acoustics_metadata  # Local import to avoid cycle

    for knee in ("left", "right"):
        for maneuver_key in ("walk", "sit_to_stand", "flexion_extension"):
            try:
                meta = get_acoustics_metadata(
                    metadata_file_path=str(legend_path),
                    scripted_maneuver=maneuver_key,
                    knee=knee,
                )
                # The legend stores base name without extension
                mapping[(knee, maneuver_key)] = meta.file_name
            except Exception as exc:  # pylint: disable=broad-except
                logging.debug(
                    "Legend lookup failed for %s/%s: %s", knee, maneuver_key, exc
                )
                continue

    return mapping


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

    excel_file_path = _find_excel_file(
        motion_capture_dir,
        f"AOA{study_id}_Biomechanics_Full_Set",
    )
    if excel_file_path is None:
        raise FileNotFoundError(
            f"Required motion capture Excel file 'AOA{study_id}_Biomechanics_Full_Set.xlsx' "
            f"or 'AOA{study_id}_Biomechanics_Full_Set.xlsm' "
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
                    f"in motion capture Excel file '{excel_file_path.name}'"
                )
            # Check for speed data sheets
            for speed_key in maneuver_config["speeds"]:
                speed_sheet_name = (
                    f"AOA{study_id}_{speed_key}_{maneuver_config['data']}"
                )
                if speed_sheet_name not in sheet_names:
                    raise ValueError(
                        f"Required sheet '{speed_sheet_name}' not found "
                        f"in motion capture Excel file '{excel_file_path.name}'"
                    )
        else:
            maneuver_sheet_name = (
                f"AOA{study_id}_{maneuver_config['data']}"
            )

            if maneuver_sheet_name not in sheet_names:
                raise ValueError(
                    f"Required sheet '{maneuver_sheet_name}' not found "
                    f"in motion capture Excel file '{excel_file_path.name}'"
                )
            events_sheet_name = f"AOA{study_id}_{maneuver_config['events']}"
            if events_sheet_name not in sheet_names:
                raise ValueError(
                    f"Required sheet '{events_sheet_name}' not found "
                    f"in motion capture Excel file '{excel_file_path.name}'"
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


def setup_logging(log_file: Optional[Path] = None) -> None:
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


def _determine_fs_from_df_or_meta(df: pd.DataFrame, meta_json_path: Path) -> float:
    """Determine sampling frequency from tt or meta file.

    Prefers computing from `tt`; falls back to meta JSON `fs` if present.
    """
    try:
        if "tt" in df.columns and df["tt"].notna().any():
            tt = pd.to_numeric(df["tt"], errors="coerce").to_numpy()
            diffs = pd.Series(tt).diff().dropna()
            if not diffs.empty and float(diffs.median()) > 0:
                return 1.0 / float(diffs.median())
        if meta_json_path.exists():
            with open(meta_json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            fs_val = float(meta.get("fs", float("nan")))
            if fs_val and fs_val > 0:
                return fs_val
    except Exception as e:  # pylint: disable=broad-except
        logging.warning("Failed to determine fs from df/meta: %s", e)
    raise RuntimeError("Cannot determine sampling frequency for audio")


def _process_bin_stage(participant_dir: Path) -> list[Path]:
    """Process all .bin files to frequency-augmented pickles.

    For each knee and maneuver, reads the .bin, writes base pickle + meta
    into `<audio_base>_outputs`, then adds instantaneous frequency and saves
    `<audio_base>_with_freq.pkl`. Removes base `.pkl` after frequency save
    to keep only the resultant dataframe, as requested.

    Returns:
        List of paths to `_with_freq.pkl` files produced.
    """
    file_name_map = _load_acoustics_file_names(participant_dir)
    if file_name_map:
        logging.info("Legend-derived audio base names: %s", {k: v for k, v in file_name_map.items()})
    produced: list[Path] = []
    for knee_side in ["Left", "Right"]:
        knee_dir = participant_dir / f"{knee_side} Knee"
        if not knee_dir.exists():
            continue
        for maneuver_key in ["walk", "sit_to_stand", "flexion_extension"]:
            maneuver_dir = _find_maneuver_dir(knee_dir, maneuver_key)
            if maneuver_dir is None:
                continue

            # Prefer legend-derived filename; fall back to filesystem glob if missing
            legend_key = (knee_side.lower(), maneuver_key)
            legend_base = file_name_map.get(legend_key)
            audio_base_path: Path
            if legend_base:
                audio_base_path = maneuver_dir / legend_base
            else:
                try:
                    audio_base_path = Path(get_audio_file_name(maneuver_dir, with_freq=False))
                except Exception as e:  # pylint: disable=broad-except
                    logging.warning("Skipping %s/%s: %s", knee_side, maneuver_key, e)
                    continue

            bin_path = audio_base_path.with_suffix(".bin")
            if not bin_path.exists():
                # Fallback: glob any .bin in the maneuver directory
                alt_bin_files = list(maneuver_dir.glob("*.bin"))
                if len(alt_bin_files) == 1:
                    bin_path = alt_bin_files[0]
                    audio_base_path = bin_path.with_suffix("")
                    if legend_base:
                        logging.info(
                            "Using fallback .bin %s for %s/%s (legend expected %s.bin)",
                            bin_path.name,
                            knee_side,
                            maneuver_key,
                            legend_base,
                        )
                    else:
                        logging.info(
                            "Using .bin file %s for %s/%s (no legend entry found)",
                            bin_path.name,
                            knee_side,
                            maneuver_key,
                        )
                elif len(alt_bin_files) > 1:
                    logging.error(
                        "Multiple .bin files found in %s; please keep only one: %s",
                        maneuver_dir,
                        [f.name for f in alt_bin_files],
                    )
                    continue
                else:
                    logging.error(
                        "No .bin found in %s (expected %s.bin)",
                        maneuver_dir,
                        audio_base_path.name,
                    )
                    continue

            outputs_dir = maneuver_dir / f"{audio_base_path.name}_outputs"
            outputs_dir.mkdir(parents=True, exist_ok=True)

            # Read raw .bin -> base pkl + meta (import locally to avoid circular imports)
            try:
                from src.audio.readers import read_audio_board_file
                read_audio_board_file(str(bin_path), str(outputs_dir))
            except Exception as e:  # pylint: disable=broad-except
                logging.error("Failed reading audio board file %s: %s", bin_path, e)
                continue

            base_pkl = outputs_dir / f"{audio_base_path.name}.pkl"
            meta_json = outputs_dir / f"{audio_base_path.name}_meta.json"
            if not base_pkl.exists():
                logging.error("Base pickle not found after read: %s", base_pkl)
                continue

            try:
                df = pd.read_pickle(base_pkl)
                fs = _determine_fs_from_df_or_meta(df, meta_json)
                from src.audio.instantaneous_frequency import (
                    add_instantaneous_frequency,
                )
                df_with_freq = add_instantaneous_frequency(df, fs)
                out_with_freq = outputs_dir / f"{audio_base_path.name}_with_freq.pkl"
                df_with_freq.to_pickle(out_with_freq)
                produced.append(out_with_freq)
                # Remove base pkl to keep only resultant dataframe
                try:
                    base_pkl.unlink(missing_ok=True)
                except Exception:
                    pass
            except Exception as e:  # pylint: disable=broad-except
                logging.error("Failed frequency augmentation for %s: %s", base_pkl, e)
                continue
    return produced


def _run_audio_qc_for_maneuver(df: pd.DataFrame, maneuver_key: str, biomech_file: Optional[Path]) -> None:
    """Run maneuver-specific audio QC with lenient defaults and log outcome."""
    time_col = "tt"
    channels = [ch for ch in ["ch1", "ch2", "ch3", "ch4"] if ch in df.columns]
    if not channels or time_col not in df.columns:
        logging.info("Audio QC skipped: missing time or channels")
        return

    try:
        # Local imports to avoid circular dependency at module import time
        from src.audio.quality_control import (
            qc_audio_flexion_extension,
            qc_audio_sit_to_stand,
            qc_audio_walk,
        )
        if maneuver_key == "walk":
            results = qc_audio_walk(
                df=df,
                time_col=time_col,
                audio_channels=channels,
                biomech_file=biomech_file,
            )
            passed_count = sum(1 for r in results if bool(r.get("passed", False)))
            logging.info("Walk audio QC: %d pass(es) valid", passed_count)
        elif maneuver_key == "flexion_extension":
            passed, coverage = qc_audio_flexion_extension(
                df=df,
                time_col=time_col,
                audio_channels=channels,
                target_freq_hz=0.25,
                tail_length_s=5.0,
            )
            logging.info("Flexion-Extension audio QC: passed=%s coverage=%.2f", passed, coverage)
        elif maneuver_key == "sit_to_stand":
            passed, coverage = qc_audio_sit_to_stand(
                df=df,
                time_col=time_col,
                audio_channels=channels,
                target_freq_hz=0.25,
                tail_length_s=5.0,
            )
            logging.info("Sit-to-Stand audio QC: passed=%s coverage=%.2f", passed, coverage)
    except Exception as e:  # pylint: disable=broad-except
        logging.warning("Audio QC failed (%s): %s", maneuver_key, e)


def process_participant(participant_dir: Path, entrypoint: Literal["bin", "sync", "cycles"] = "sync") -> bool:
    """Process a single participant directory.

    Args:
        participant_dir: Path to the participant directory
        entrypoint: Stage to start from: 'bin' | 'sync' | 'cycles'

    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        study_id = participant_dir.name.lstrip("#")
        logging.info("Processing participant #%s", study_id)

        # Validate directory structure
        if entrypoint == "bin":
            check_participant_dir_for_bin_stage(participant_dir)
        else:
            check_participant_dir_for_required_files(participant_dir)
        logging.info(
            "Directory validation passed for participant #%s", study_id
        )

        # Biomechanics validation (best-effort) — skip for bin-only stage
        motion_capture_dir = participant_dir / "Motion Capture"
        bio_valid = False
        biomechanics_file = _find_excel_file(
            motion_capture_dir,
            f"AOA{study_id}_Biomechanics_Full_Set",
        )
        if entrypoint != "bin":
            try:
                motion_capture_folder_has_required_data(motion_capture_dir)
                bio_valid = True
            except Exception as e:
                logging.error("Biomechanics validation failed: %s", e)

        # BIN stage: process raw audio to frequency DF + audio QC
        if entrypoint == "bin":
            produced = _process_bin_stage(participant_dir)
            logging.info("Bin stage produced %d frequency pickle(s)", len(produced))
            # Run audio QC per maneuver using produced files
            for pkl in produced:
                try:
                    df = pd.read_pickle(pkl)
                except Exception:
                    continue
                # Infer maneuver from path components
                parts = set(Path(pkl).parts)
                if "Walking" in parts:
                    _run_audio_qc_for_maneuver(
                        df,
                        "walk",
                        biomechanics_file if (bio_valid and biomechanics_file.exists()) else None,
                    )
                elif "Flexion-Extension" in parts:
                    _run_audio_qc_for_maneuver(
                        df,
                        "flexion_extension",
                        biomechanics_file if (bio_valid and biomechanics_file.exists()) else None,
                    )
                elif "Sit-Stand" in parts:
                    _run_audio_qc_for_maneuver(
                        df,
                        "sit_to_stand",
                        biomechanics_file if (bio_valid and biomechanics_file.exists()) else None,
                    )

            # If nothing was produced, stop early so validation doesn't fail with a
            # missing processed-file error that hides the missing .bin root cause.
            if not produced:
                logging.error(
                    "Bin stage produced no outputs; verify raw .bin files exist in each maneuver folder."
                )
                return False

        # SYNC stage: synchronize audio with biomechanics and save Synced files
        if entrypoint in ("bin", "sync"):
            parse_participant_directory(participant_dir)
            logging.info(
                "Synchronization completed for participant #%s", study_id
            )

        # CYCLES stage: run movement cycle QC on all Synced files and save outputs
        synced_files = find_synced_files(participant_dir)
        if entrypoint == "cycles" and not synced_files:
            logging.warning("No synced files found to run cycle QC")
        for synced_file in synced_files:
            try:
                clean_cycles, outlier_cycles, output_dir = perform_sync_qc(synced_file, create_plots=True)

                # Update processing log with movement cycles info
                try:
                    # Extract metadata from path
                    parts = synced_file.parts
                    knee_idx = -1
                    for i, part in enumerate(parts):
                        if "Knee" in part:
                            knee_idx = i
                            break

                    if knee_idx >= 0:
                        knee_side = parts[knee_idx].split()[0]  # "Left" or "Right"
                        maneuver_dir = synced_file.parent.parent  # Go up from Synced to maneuver dir

                        # Determine maneuver type from path
                        maneuver_key = None
                        for part in parts:
                            if "walk" in part.lower():
                                maneuver_key = "walk"
                            elif "sit" in part.lower() and "stand" in part.lower():
                                maneuver_key = "sit_to_stand"
                            elif "flexion" in part.lower():
                                maneuver_key = "flexion_extension"

                        if maneuver_key:
                            log = ManeuverProcessingLog.get_or_create(
                                study_id=study_id,
                                knee_side=cast(Literal["Left", "Right"], knee_side),
                                maneuver=cast(Literal["walk", "sit_to_stand", "flexion_extension"], maneuver_key),
                                maneuver_directory=maneuver_dir,
                            )

                            cycles_record = create_cycles_record_from_data(
                                sync_file_name=synced_file.stem,
                                clean_cycles=clean_cycles,
                                outlier_cycles=outlier_cycles,
                                output_dir=output_dir,
                                acoustic_threshold=100.0,  # Default threshold
                                plots_created=True,
                            )
                            log.add_movement_cycles_record(cycles_record)

                            # Mark synchronization record QC status
                            try:
                                for i, rec in enumerate(log.synchronization_records):
                                    if rec.sync_file_name == synced_file.stem:
                                        rec.sync_qc_performed = True
                                        # Simple pass criterion: at least one clean cycle
                                        rec.sync_qc_passed = bool(len(clean_cycles) > 0)
                                        log.synchronization_records[i] = rec
                                        break
                            except Exception:
                                pass
                            log.save_to_excel()

                            # Update knee-level log
                            try:
                                knee_log = KneeProcessingLog.get_or_create(
                                    study_id=study_id,
                                    knee_side=cast(Literal["Left", "Right"], knee_side),
                                    knee_directory=maneuver_dir.parent,
                                )
                                knee_log.update_maneuver_summary(
                                    cast(Literal["walk", "sit_to_stand", "flexion_extension"], maneuver_key),
                                    log
                                )
                                knee_log.save_to_excel()
                            except Exception as e:
                                logging.warning(f"Failed to update knee-level log: {e}")
                except Exception as e:
                    logging.warning(f"Failed to update movement cycles log for {synced_file}: {e}")

            except Exception as e:  # pylint: disable=broad-except
                logging.warning("Cycle QC failed for %s: %s", synced_file, e)

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
        motion_capture_dir = participant_dir / "Motion Capture"
        biomechanics_file = _find_excel_file(
            motion_capture_dir,
            f"AOA{study_id}_Biomechanics_Full_Set",
        )

        if biomechanics_file is None:
            logging.error(
                "Biomechanics file not found: %s", motion_capture_dir
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
        all_recordings = []
        if maneuver_key == "walk":
            # Process all speeds for walking
            for speed in ["slow", "medium", "fast"]:
                recordings = import_biomechanics_recordings(
                    biomechanics_file=biomechanics_file,
                    maneuver="walk",
                    speed=speed,  # type: ignore[arg-type]
                )
                all_recordings.extend(recordings)

            if not all_recordings:
                logging.error(
                    "No biomechanics recordings found for walk (any speed)"
                )
                return False
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
            all_recordings = recordings

        # Determine output directory
        maneuver_folder_map_rev = {
            "walk": "Walking",
            "sit_to_stand": "Sit-Stand",
            "flexion_extension": "Flexion-Extension",
        }
        maneuver_folder = maneuver_folder_map_rev[maneuver_key]
        knee_dir = participant_dir / f"{knee_side} Knee"
        maneuver_dir = knee_dir / maneuver_folder
        synced_dir = maneuver_dir / "Synced"

        # Sync all recordings
        success_count = 0
        for recording in all_recordings:
            try:
                if maneuver_key == "walk":
                    pass_number = recording.pass_number
                    speed = recording.speed
                else:
                    pass_number = None
                    speed = None

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
                success_count += 1
            except Exception as e:  # pylint: disable=broad-except
                logging.error(
                    "Error syncing recording (pass=%s, speed=%s): %s",
                    pass_number, speed, str(e)
                )
                continue

        if success_count == 0:
            logging.error("Failed to sync any recordings")
            return False

        logging.info("Successfully synced %d/%d recordings", success_count, len(all_recordings))
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
