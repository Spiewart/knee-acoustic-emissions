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

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Literal, Optional, cast

import pandas as pd

from src.biomechanics.importers import import_biomechanics_recordings
from src.config import get_data_root, load_env_file
from src.orchestration.processing_log import (
    KneeProcessingLog,
    ManeuverProcessingLog,
    create_audio_record_from_data,
    create_biomechanics_record_from_data,
    create_sync_record_from_data,
)
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
    from src.models import BiomechanicsRecording


def _normalize_maneuver(maneuver: Optional[str]) -> Optional[str]:
    """Normalize maneuver shorthand to internal format.

    Converts CLI shorthand (walk, fe, sts) to internal format (walk, flexion_extension, sit_to_stand).
    Returns None if input is None.

    Args:
        maneuver: CLI maneuver shorthand or internal format

    Returns:
        Internal maneuver format or None
    """
    if maneuver is None:
        return None
    maneuver_lower = maneuver.lower()
    if maneuver_lower == "fe":
        return "flexion_extension"
    if maneuver_lower == "sts":
        return "sit_to_stand"
    if maneuver_lower == "walk":
        return "walk"
    # If it's already in internal format, return as-is
    if maneuver_lower in ("flexion_extension", "sit_to_stand"):
        return maneuver_lower
    # Unrecognized format, return as-is for error handling upstream
    return maneuver_lower


def _filter_synced_files(synced_files: list[Path], knee: Optional[str] = None, maneuver: Optional[str] = None) -> list[Path]:
    """Filter synced files by knee and/or maneuver.

    Args:
        synced_files: List of synced file paths
        knee: Optional knee filter ('left' or 'right')
        maneuver: Optional maneuver filter (internal format: 'walk', 'flexion_extension', 'sit_to_stand')

    Returns:
        Filtered list of synced files
    """
    filtered = synced_files

    if knee:
        knee_lower = knee.lower()
        filtered = [f for f in filtered if knee_lower in str(f).lower()]

    if maneuver:
        # Map internal maneuver names to directory names
        maneuver_map = {
            "walk": "walking",
            "flexion_extension": "flexion-extension",
            "sit_to_stand": "sit-stand",
        }
        maneuver_dir = maneuver_map.get(maneuver, maneuver)
        filtered = [f for f in filtered if maneuver_dir.lower() in str(f).lower()]

    return filtered


def parse_participant_directory(participant_dir: Path, knee: Optional[str] = None, maneuver: Optional[str] = None, biomechanics_type: Optional[str] = None) -> None:
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
        knee: Optional knee filter ('left' or 'right')
        maneuver: Optional maneuver filter (internal format: 'walk', 'flexion_extension', 'sit_to_stand')
        biomechanics_type: Optional biomechanics type (e.g., 'Motion Analysis', 'Gonio', 'IMU')

    Raises:
        FileNotFoundError: If required files or directory structure is missing.
        ValueError: If file contents are invalid or synchronization fails.
    """

    study_id = get_study_id_from_directory(participant_dir)
    logging.info("Processing participant %s", study_id)

    check_participant_dir_for_required_files(participant_dir, knee=knee, maneuver=maneuver)

    # Get the biomechanics file path
    motion_capture_dir = participant_dir / "Motion Capture"
    biomechanics_file = _find_excel_file(
        motion_capture_dir,
        f"AOA{study_id}_Biomechanics_Full_Set",
    )

    # Collect all synchronized data before writing anything (transactional)
    all_synced_data: list[tuple[Path, pd.DataFrame]] = []

    # Process each knee (Left and Right)
    # Apply knee filter
    knees_to_process = []
    if knee is None:
        knees_to_process = ["Left", "Right"]
    elif knee.lower() == "left":
        knees_to_process = ["Left"]
    elif knee.lower() == "right":
        knees_to_process = ["Right"]

    for knee_side in knees_to_process:
        knee_dir = participant_dir / f"{knee_side} Knee"
        logging.info("Processing %s knee", knee_side)
        knee_synced_data, _ = _process_knee_maneuvers(
            knee_dir, biomechanics_file, knee_side, maneuver=maneuver, biomechanics_type=biomechanics_type
        )
        all_synced_data.extend(knee_synced_data)

    # If we got here, all processing succeeded - now write all files
    logging.info("Writing %d synchronized files", len(all_synced_data))
    for output_path, synced_df, _ in all_synced_data:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        synced_df.to_pickle(output_path)

    logging.info("Completed participant %s", study_id)


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
    qc_not_passed: Optional[str] = None,
    qc_not_passed_mic_1: Optional[str] = None,
    qc_not_passed_mic_2: Optional[str] = None,
    qc_not_passed_mic_3: Optional[str] = None,
    qc_not_passed_mic_4: Optional[str] = None,
    biomechanics_type: Optional[str] = None,
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
        qc_not_passed: String representation of bad intervals list (any mic)
        qc_not_passed_mic_1: String representation of bad intervals for mic 1
        qc_not_passed_mic_2: String representation of bad intervals for mic 2
        qc_not_passed_mic_3: String representation of bad intervals for mic 3
        qc_not_passed_mic_4: String representation of bad intervals for mic 4
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

            if audio_metadata is None:
                audio_metadata = {}
            audio_metadata["study_id"] = int(study_id)
            audio_metadata["recording_timezone"] = "UTC"
            audio_metadata["mic_positions"] = _load_mic_positions_from_legend(
                participant_dir=maneuver_dir.parents[1],
                knee=knee_side.lower(),
                maneuver_key=maneuver_key,
            )

            audio_file_base = None
            if audio_bin_path is not None:
                audio_file_base = Path(audio_bin_path).stem
            if audio_file_base is None:
                audio_file_base = audio_pkl_file.stem.replace("_with_freq", "")

            audio_record = create_audio_record_from_data(
                audio_file_name=audio_file_base,
                audio_df=audio_df,
                audio_bin_path=audio_bin_path,
                audio_pkl_path=audio_pkl_file,
                metadata=audio_metadata,
                biomechanics_type=biomechanics_type,
                knee=knee_side.lower(),
                maneuver=maneuver_key,
            )

            # Preserve existing biomechanics metadata from log if present
            # This ensures we don't overwrite biomechanics info when re-running bin stage
            if log.audio_record is not None and log.audio_record.linked_biomechanics:
                audio_record.linked_biomechanics = True
                audio_record.biomechanics_file = log.audio_record.biomechanics_file
                audio_record.biomechanics_type = log.audio_record.biomechanics_type
                audio_record.biomechanics_sync_method = log.audio_record.biomechanics_sync_method
                audio_record.biomechanics_sample_rate = log.audio_record.biomechanics_sample_rate
                audio_record.biomechanics_notes = log.audio_record.biomechanics_notes
            # If biomechanics were processed in this call, propagate linkage and context
            elif biomechanics_recordings is not None and len(biomechanics_recordings) > 0 and biomechanics_file is not None:
                audio_record.linked_biomechanics = True
                audio_record.biomechanics_file = str(biomechanics_file)
                audio_record.biomechanics_type = biomechanics_type
                # Use known biomech type/sample rate when available
                if biomechanics_recordings:
                    first_rec = biomechanics_recordings[0]
                    # Try to infer sample rate from the biomechanics data
                    try:
                        # Check if data is a BiomechanicsData object with a TIME column
                        if hasattr(first_rec, "data") and first_rec.data is not None:
                            data_obj = first_rec.data
                            # BiomechanicsData is a dataclass with a 'data' DataFrame attribute
                            if hasattr(data_obj, "data"):
                                time_col = data_obj.data.get("TIME", None)
                            else:
                                # data_obj is a DataFrame
                                time_col = data_obj.get("TIME", None) if hasattr(data_obj, "get") else None

                            if time_col is not None and len(time_col) > 1:
                                time_diffs = time_col.diff().dropna()
                                if len(time_diffs) > 0:
                                    avg_diff_sec = time_diffs.mean().total_seconds()
                                    if avg_diff_sec > 0:
                                        audio_record.biomechanics_sample_rate = 1.0 / avg_diff_sec
                        elif hasattr(first_rec, "sample_rate"):
                            audio_record.biomechanics_sample_rate = float(first_rec.sample_rate)
                    except Exception:
                        pass
                # Set sync method based on biomechanics type
                if biomechanics_type == "Gonio":
                    audio_record.biomechanics_sync_method = "flick"
                else:
                    # Default sync method for IMU, Motion Analysis, or unknown types is stomp
                    audio_record.biomechanics_sync_method = "stomp"
            else:
                pass
            # Maneuver should mirror the current processing step
            audio_record.maneuver = maneuver_key
            # Set QC_not_passed fields if provided
            if qc_not_passed is not None:
                audio_record.qc_not_passed = qc_not_passed
            if qc_not_passed_mic_1 is not None:
                audio_record.qc_not_passed_mic_1 = qc_not_passed_mic_1
            if qc_not_passed_mic_2 is not None:
                audio_record.qc_not_passed_mic_2 = qc_not_passed_mic_2
            if qc_not_passed_mic_3 is not None:
                audio_record.qc_not_passed_mic_3 = qc_not_passed_mic_3
            if qc_not_passed_mic_4 is not None:
                audio_record.qc_not_passed_mic_4 = qc_not_passed_mic_4
            # Set log update timestamp
            audio_record.log_updated = datetime.now()
            log.update_audio_record(audio_record)

        # Update biomechanics record if data provided
        if biomechanics_recordings is not None and biomechanics_file is not None:
            bio_record = create_biomechanics_record_from_data(
                biomechanics_file=biomechanics_file,
                recordings=biomechanics_recordings,
                sheet_name=f"{maneuver_key}_data",
                maneuver=maneuver_key,
                biomechanics_type=biomechanics_type,
                knee=knee_side.lower(),
                biomechanics_sync_method=("flick" if biomechanics_type == "Gonio" else "stomp"),
                biomechanics_sample_rate=audio_record.biomechanics_sample_rate if audio_record else None,
                study_id=int(study_id),
            )
            log.update_biomechanics_record(bio_record)

            # Back-populate biomechanics metadata to audio record if it exists
            # This handles the case where audio was processed in bin stage (without biomechanics)
            # and now we're in sync stage (with biomechanics)
            if log.audio_record is not None:
                log.audio_record.linked_biomechanics = True
                log.audio_record.biomechanics_file = str(biomechanics_file)
                log.audio_record.biomechanics_type = biomechanics_type
                # Set sync method based on biomechanics type
                if biomechanics_type == "Gonio":
                    log.audio_record.biomechanics_sync_method = "flick"
                else:
                    # Default sync method for IMU, Motion Analysis, or unknown types is stomp
                    log.audio_record.biomechanics_sync_method = "stomp"
                if biomechanics_recordings:
                    first_rec = biomechanics_recordings[0]
                    # Try to infer sample rate from the biomechanics data
                    try:
                        # Check if data is a BiomechanicsData object with a TIME column
                        if hasattr(first_rec, "data") and first_rec.data is not None:
                            data_obj = first_rec.data
                            # BiomechanicsData is a dataclass with a 'data' DataFrame attribute
                            if hasattr(data_obj, "data"):
                                time_col = data_obj.data.get("TIME", None)
                            else:
                                # data_obj is a DataFrame
                                time_col = data_obj.get("TIME", None) if hasattr(data_obj, "get") else None

                            if time_col is not None and len(time_col) > 1:
                                time_diffs = time_col.diff().dropna()
                                if len(time_diffs) > 0:
                                    avg_diff_sec = time_diffs.mean().total_seconds()
                                    if avg_diff_sec > 0:
                                        log.audio_record.biomechanics_sample_rate = 1.0 / avg_diff_sec
                        elif hasattr(first_rec, "sample_rate"):
                            log.audio_record.biomechanics_sample_rate = float(first_rec.sample_rate)
                    except Exception:
                        pass
                # Update timestamp on audio record as well
                log.audio_record.log_updated = datetime.now()
                # Mark log as updated so it gets saved
                log.log_updated = datetime.now()

        # Update synchronization records if data provided
        if synced_data is not None:
            # Parse study name and numeric ID from study_id
            study_id_str = str(study_id).lstrip("#")
            study_name = "AOA"  # default
            numeric_id = 1
            if study_id_str.startswith("AOA"):
                study_name = "AOA"
                numeric_id = int(study_id_str[3:]) if len(study_id_str) > 3 else 1
            elif study_id_str.startswith("preOA"):
                study_name = "preOA"
                numeric_id = int(study_id_str[5:]) if len(study_id_str) > 5 else 1
            elif study_id_str.startswith("SMoCK"):
                study_name = "SMoCK"
                numeric_id = int(study_id_str[5:]) if len(study_id_str) > 5 else 1
            else:
                # Try to parse as just a number
                try:
                    numeric_id = int(study_id_str)
                except ValueError:
                    numeric_id = 1

            for item in synced_data:
                # Support both legacy 3-tuple and new 4-tuple with detection_results
                output_path, synced_df, stomp_tuple = item
                detection_results = None
                try:
                    audio_stomp, bio_left, bio_right, detection_results = stomp_tuple  # type: ignore[misc]
                except Exception:
                    audio_stomp, bio_left, bio_right = stomp_tuple  # type: ignore[misc]
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
                    detection_results=detection_results,
                    # Additional context for required fields
                    audio_record=audio_record if audio_df is not None else None,
                    biomech_record=bio_record if biomechanics_recordings is not None else None,
                    metadata=audio_metadata,
                    study=study_name,
                    study_id=numeric_id,
                    biomechanics_type=biomechanics_type,
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
    maneuver: Optional[str] = None,
    biomechanics_type: Optional[str] = None,
) -> tuple[list[tuple[Path, pd.DataFrame]], list[tuple[Path, pd.DataFrame, tuple]]]:
    """Process all maneuvers for a specific knee (Left or Right).

    Iterates through all maneuver types (walk, sit-to-stand, flexion-extension),
    synchronizes audio and biomechanics data for each, but does NOT write files.
    Data is collected for later transactional write.

    Args:
        knee_dir: Path to the knee directory (Left Knee or Right Knee).
        biomechanics_file: Path to the biomechanics Excel file.
        knee_side: "Left" or "Right" (used for logging and lookups).
        maneuver: Optional maneuver filter (internal format: 'walk', 'flexion_extension', 'sit_to_stand')

    Returns:
        Tuple of:
        - List of (output_path, synchronized_dataframe) tuples for file writing.
        - List of (output_path, audio_df, bio_df, synced_df, stomp_times) tuples for visualization.

    Raises:
        Exception: If any maneuver fails to process (transactional semantics).
    """
    all_maneuver_keys: tuple[Literal[
        "walk",
        "sit_to_stand",
        "flexion_extension",
    ], ...] = (
        "walk",
        "sit_to_stand",
        "flexion_extension",
    )

    # Apply maneuver filter
    if maneuver is not None:
        maneuver_keys = (maneuver,)
    else:
        maneuver_keys = all_maneuver_keys

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
            biomechanics_type=biomechanics_type,
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
    biomechanics_type: Optional[str] = None,
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
                biomechanics_type=biomechanics_type,
            )
            synced_data.extend(speed_synced_data)
            viz_data.extend(speed_viz_data)
            all_recordings.extend(speed_recordings)
    else:
        # For sit-to-stand and flexion-extension, single recording
        study_name = biomechanics_file.stem.split("_")[0]
        recordings = import_biomechanics_recordings(
            biomechanics_file=biomechanics_file,
            maneuver=maneuver_key,
            speed=None,
            biomechanics_type=biomechanics_type,
            study_name=study_name,
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
            biomechanics_type=biomechanics_type,
        )
        synced_data.append((output_path, synced_df, stomp_times))
        # Generate visualization immediately to avoid keeping large DataFrames in memory
        audio_stomp, bio_left, bio_right, detection_results = stomp_times
        plot_stomp_detection(audio_df, bio_df, synced_df, audio_stomp, bio_left, bio_right, output_path, detection_results)
        viz_data.append((output_path, audio_df, bio_df, synced_df, stomp_times))
        del bio_df, synced_df

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
            biomechanics_type=biomechanics_type,
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
    biomechanics_type: Optional[str] = None,
) -> tuple[list[tuple[Path, pd.DataFrame]], list[tuple[Path, pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple]], list]:
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
        - List of (output_path, audio_df, bio_df, synced_df, stomp_times) tuples for visualization
        - List of biomechanics recordings
    """
    synced_data: list[tuple[Path, pd.DataFrame]] = []
    viz_data: list[tuple[Path, pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple]] = []

    try:
        study_name = biomechanics_file.stem.split("_")[0]
        recordings = import_biomechanics_recordings(
            biomechanics_file=biomechanics_file,
            maneuver="walk",
            speed=speed,  # type: ignore[arg-type]
            biomechanics_type=biomechanics_type,
            study_name=study_name,
        )

        if not recordings:
            # No recordings found - this is expected in some cases
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
                biomechanics_type=biomechanics_type,
            )
            synced_data.append((output_path, synced_df, stomp_times))

            # Generate visualization immediately to avoid accumulating DataFrames in memory
            audio_stomp, bio_left, bio_right, detection_results = stomp_times
            plot_stomp_detection(audio_df, bio_df, synced_df, audio_stomp, bio_left, bio_right, output_path, detection_results)
            viz_data.append((output_path, audio_df, bio_df, synced_df, stomp_times))
            # Explicitly delete large intermediate DataFrames to free memory
            del bio_df, synced_df

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
    recording: "BiomechanicsRecording",  # type: ignore[name-defined]
    audio_df: pd.DataFrame,
    synced_dir: Path,
    biomechanics_file: Path,
    maneuver_key: str,
    knee_side: str,
    pass_number: Optional[int],
    speed: Optional[str],
    biomechanics_type: Optional[str] = None,
) -> tuple[Path, pd.DataFrame, tuple]:
    """Synchronize a single biomechanics recording with audio.

    Processes and synchronizes data but does NOT write files.

    Args:
        recording: BiomechanicsRecording object with biomechanics data
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

    # Get audio stomp time with dual-knee disambiguation and biomechanics context
    try:
        study_name = biomechanics_file.stem.split("_")[0] if biomechanics_file is not None else None
    except Exception:
        study_name = None
    audio_stomp_time, detection_results = get_audio_stomp_time(
        audio_df,
        recorded_knee=recorded_knee,
        right_stomp_time=right_stomp_time,
        left_stomp_time=left_stomp_time,
        return_details=True,
        biomechanics_type=biomechanics_type,
        study_name=study_name,
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
    # Note: sync_audio_with_biomechanics handles copying internally as needed
    clipped_df = sync_audio_with_biomechanics(
        audio_stomp_time=audio_stomp_time,
        bio_stomp_time=bio_stomp_time,
        audio_df=audio_df,
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

    # Return path, dataframe, stomp times (with detection results), and original bio_df for visualization
    stomp_times = (audio_stomp_time, left_stomp_time, right_stomp_time, detection_results)

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
    knee: Optional[str] = None,
    maneuver: Optional[str] = None,
) -> None:
    """Checks that the participant directory contains all required files.

    Args:
        participant_dir: Path to participant directory
        knee: Optional knee filter ('left' or 'right')
        maneuver: Optional maneuver filter (internal format)
    """

    dir_has_acoustic_file_legend(participant_dir)
    participant_dir_has_top_level_folders(participant_dir, knee=knee)

    # Apply knee filter
    knees_to_check = []
    if knee is None:
        knees_to_check = ["Left Knee", "Right Knee"]
    elif knee.lower() == "left":
        knees_to_check = ["Left Knee"]
    elif knee.lower() == "right":
        knees_to_check = ["Right Knee"]

    for knee_folder in knees_to_check:
        knee_folder_has_subfolder_each_maneuver(
            participant_dir / knee_folder,
            maneuver=maneuver
        )

    motion_capture_folder_has_required_data(participant_dir/"Motion Capture")


def check_participant_dir_for_bin_stage(
    participant_dir: Path,
    knee: Optional[str] = None,
    maneuver: Optional[str] = None,
) -> None:
    """Bin-stage validation: ensure raw .bin exists but do not require processed outputs.

    Args:
        participant_dir: Path to participant directory
        knee: Optional knee filter ('left' or 'right')
        maneuver: Optional maneuver filter (internal format)
    """

    dir_has_acoustic_file_legend(participant_dir)
    participant_dir_has_top_level_folders(participant_dir, knee=knee)

    # Apply knee filter
    knees_to_check = []
    if knee is None:
        knees_to_check = ["Left Knee", "Right Knee"]
    elif knee.lower() == "left":
        knees_to_check = ["Left Knee"]
    elif knee.lower() == "right":
        knees_to_check = ["Right Knee"]

    for knee_folder in knees_to_check:
        knee_folder_has_subfolder_each_maneuver(
            participant_dir / knee_folder,
            require_processed=False,
            maneuver=maneuver
        )


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
    knee: Optional[str] = None,
) -> None:
    """Checks that the participant directory contains the required
    top-level folders.

    Args:
        participant_dir: Path to participant directory
        knee: Optional knee filter ('left' or 'right')
    """

    # Always require Motion Capture folder
    required_folders = ["Motion Capture"]

    # Add knee folders based on filter
    if knee is None:
        required_folders.extend(["Left Knee", "Right Knee"])
    elif knee.lower() == "left":
        required_folders.append("Left Knee")
    elif knee.lower() == "right":
        required_folders.append("Right Knee")

    for folder in required_folders:
        folder_path = participant_dir / folder
        if not folder_path.exists() or not folder_path.is_dir():
            raise FileNotFoundError(
                f"Required folder '{folder}' not found in {participant_dir}"
            )


def knee_folder_has_subfolder_each_maneuver(
    knee_dir: Path,
    require_processed: bool = True,
    maneuver: Optional[str] = None,
) -> None:
    """Checks that the knee directory contains subfolders for each maneuver.

    Args:
        knee_dir: Path to knee directory
        require_processed: Whether to require processed files
        maneuver: Optional maneuver filter (internal format: 'walk', 'sit_to_stand', 'flexion_extension')
    """
    # Apply maneuver filter
    maneuvers_to_check = ("walk", "sit_to_stand", "flexion_extension")
    if maneuver is not None:
        maneuvers_to_check = (maneuver,)

    for maneuver_key in maneuvers_to_check:
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
            except Exception:  # pylint: disable=broad-except
                # Silently skip missing entries - legend is best-effort
                continue

    return mapping


def _load_mic_positions_from_legend(
    participant_dir: Path,
    knee: str,
    maneuver_key: str,
) -> dict:
    """Load microphone positions from the acoustics file legend."""
    legend_path = _find_excel_file(participant_dir, "*acoustic_file_legend*")
    if legend_path is None:
        raise FileNotFoundError(
            f"No acoustic file legend found in {participant_dir}"
        )

    from src.audio.parsers import get_acoustics_metadata  # Local import to avoid cycle
    meta = get_acoustics_metadata(
        metadata_file_path=str(legend_path),
        scripted_maneuver=maneuver_key,
        knee=knee,
    )

    def _to_code(pos) -> str:
        patellar = "I" if pos.patellar_position == "Infrapatellar" else "S"
        lateral = "M" if pos.laterality == "Medial" else "L"
        return f"{patellar}P{lateral}"

    mic_positions = {}
    for mic_num, pos in meta.microphones.items():
        mic_positions[f"mic_{mic_num}_position"] = _to_code(pos)

    if len(mic_positions) != 4:
        raise ValueError(
            f"Incomplete microphone positions in legend: {legend_path}"
        )

    return mic_positions


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
            "pass_data": "Walk0001",
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
                f"AOA{study_id}_{maneuver_config['pass_data']}"
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


def setup_logging(log_file: Optional[Path] = None, log_level: int = logging.INFO) -> None:
    """Configure logging to both console and optional file.

    Args:
        log_file: Optional path to write log file
        log_level: Logging level (e.g., logging.DEBUG)
    """
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


def _process_bin_stage(participant_dir: Path, knee: Optional[str] = None, maneuver: Optional[str] = None, biomechanics_type: Optional[str] = None) -> tuple[list[Path], list[pd.DataFrame]]:
    """Process all .bin files to frequency-augmented pickles with filtering options.

    Args:
        participant_dir: Path to the participant directory
        knee: Specify which knee to process ('left' or 'right')
        maneuver: Specify which maneuver to process ('walk', 'fe', or 'sts')

    Returns:
        Tuple of (pickle_paths, dataframes) where pickle_paths is a list of paths to
        `_with_freq.pkl` files and dataframes is a list of corresponding DataFrames
        already loaded in memory to avoid redundant disk reads.
    """
    # Normalize maneuver shorthand to internal format
    maneuver = _normalize_maneuver(maneuver)
    file_name_map = _load_acoustics_file_names(participant_dir)
    produced: list[Path] = []
    produced_dfs: list[pd.DataFrame] = []
    for knee_side in ["Left", "Right"]:
        if knee and knee_side.lower() != knee.lower():
            continue
        knee_dir = participant_dir / f"{knee_side} Knee"
        if not knee_dir.exists():
            continue
        for maneuver_key in ["walk", "sit_to_stand", "flexion_extension"]:
            if maneuver and maneuver_key != maneuver:
                continue
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
            # Returns the DataFrame directly to avoid re-loading from disk
            try:
                from src.audio.raw_qc import (
                    merge_bad_intervals,
                    run_raw_audio_qc,
                    run_raw_audio_qc_per_mic,
                )
                from src.audio.readers import read_audio_board_file

                df = read_audio_board_file(str(bin_path), str(outputs_dir))
            except Exception as e:  # pylint: disable=broad-except
                logging.error("Failed reading audio board file %s: %s", bin_path, e)
                continue

            base_pkl = outputs_dir / f"{audio_base_path.name}.pkl"
            meta_json = outputs_dir / f"{audio_base_path.name}_meta.json"
            if not base_pkl.exists():
                logging.error("Base pickle not found after read: %s", base_pkl)
                continue

            try:
                # Run raw audio QC before frequency augmentation (per-microphone)
                # Use the already-loaded DataFrame from read_audio_board_file

                # Run overall QC (any mic fails)
                dropout_intervals, artifact_intervals = run_raw_audio_qc(df)
                bad_intervals = merge_bad_intervals(dropout_intervals, artifact_intervals)

                # Run per-microphone QC
                per_mic_bad_intervals = run_raw_audio_qc_per_mic(df)

                                # Store QC results for logging
                qc_not_passed = str(bad_intervals) if bad_intervals else None

                # Store per-mic QC results (map channel names to mic numbers)
                qc_not_passed_mic_1 = str(per_mic_bad_intervals.get("ch1", [])) if per_mic_bad_intervals.get("ch1") else None
                qc_not_passed_mic_2 = str(per_mic_bad_intervals.get("ch2", [])) if per_mic_bad_intervals.get("ch2") else None
                qc_not_passed_mic_3 = str(per_mic_bad_intervals.get("ch3", [])) if per_mic_bad_intervals.get("ch3") else None
                qc_not_passed_mic_4 = str(per_mic_bad_intervals.get("ch4", [])) if per_mic_bad_intervals.get("ch4") else None

                # Load metadata for processing log
                metadata = {}
                if meta_json.exists():
                    try:
                        with open(meta_json, "r") as f:
                            metadata = json.load(f)
                    except Exception as exc:
                        # Proceed with empty metadata but log the failure for visibility
                        logging.warning("Failed to load metadata from %s: %s", meta_json, exc)

                fs = _determine_fs_from_df_or_meta(df, meta_json)
                from src.audio.instantaneous_frequency import (
                    add_instantaneous_frequency,
                )
                df_with_freq = add_instantaneous_frequency(df, fs)
                out_with_freq = outputs_dir / f"{audio_base_path.name}_with_freq.pkl"
                df_with_freq.to_pickle(out_with_freq)
                produced.append(out_with_freq)
                produced_dfs.append(df_with_freq)  # Store DF to avoid reloading

                # Update processing log with QC results
                _save_or_update_processing_log(
                    study_id=participant_dir.name.lstrip("#"),
                    knee_side=cast("Literal['Left', 'Right']", knee_side),
                    maneuver_key=cast("Literal['walk', 'sit_to_stand', 'flexion_extension']", maneuver_key),
                    maneuver_dir=maneuver_dir,
                    audio_pkl_file=out_with_freq,
                    audio_df=df_with_freq,
                    audio_metadata=metadata,
                    qc_not_passed=qc_not_passed,
                    qc_not_passed_mic_1=qc_not_passed_mic_1,
                    qc_not_passed_mic_2=qc_not_passed_mic_2,
                    qc_not_passed_mic_3=qc_not_passed_mic_3,
                    qc_not_passed_mic_4=qc_not_passed_mic_4,
                    biomechanics_type=biomechanics_type,
                )

                # Remove base pkl to keep only resultant dataframe
                try:
                    base_pkl.unlink(missing_ok=True)
                except Exception:
                    pass
            except Exception as e:  # pylint: disable=broad-except
                logging.error("Failed frequency augmentation for %s: %s", base_pkl, e)
                continue
    return produced, produced_dfs


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


def process_participant(participant_dir: Path, entrypoint: Literal["bin", "sync", "cycles"] = "sync", knee: Optional[str] = None, maneuver: Optional[str] = None, biomechanics_type: Optional[str] = None) -> bool:
    """Process a single participant directory (wrapper for backward compatibility).

    Delegates to the new ParticipantProcessor class-based architecture while
    maintaining the original function signature for CLI compatibility.

    Runs all stages from the specified entrypoint onwards:
    - 'bin': runs bin -> sync -> cycles
    - 'sync': runs sync -> cycles
    - 'cycles': runs cycles only

    Args:
        participant_dir: Path to the participant directory
        entrypoint: Stage to start from: 'bin' | 'sync' | 'cycles'
        knee: Specify which knee to process ('left' or 'right')
        maneuver: Specify which maneuver to process ('walk', 'fe', or 'sts')
        biomechanics_type: Optional biomechanics type (e.g., 'Motion Analysis', 'Gonio', 'IMU')

    Returns:
        True if processing succeeded, False otherwise
    """
    from src.orchestration.participant_processor import ParticipantProcessor

    study_id = participant_dir.name.lstrip("#")
    try:
        # Normalize maneuver shorthand to internal format
        maneuver = _normalize_maneuver(maneuver)

        # Create processor and run
        processor = ParticipantProcessor(
            participant_dir=participant_dir,
            biomechanics_type=biomechanics_type,
        )
        success = processor.process(
            entrypoint=entrypoint,
            knee=knee,
            maneuver=maneuver,
        )

        if success:
            logging.info("Successfully completed processing participant #%s", study_id)

        return success
    except FileNotFoundError as e:
        logging.error("Validation error for %s: %s", participant_dir.name, str(e))
        return False
    except Exception as e:
        logging.error("Error processing participant #%s: %s", study_id, e)
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

        # Default biomechanics_type to None here; it may be inferred later
        biomechanics_type = None

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
            study_name = biomechanics_file.stem.split("_")[0]
            for speed in ["slow", "medium", "fast"]:
                recordings = import_biomechanics_recordings(
                    biomechanics_file=biomechanics_file,
                    maneuver="walk",
                    speed=speed,  # type: ignore[arg-type]
                    biomechanics_type=biomechanics_type,
                    study_name=study_name,
                )
                all_recordings.extend(recordings)

            if not all_recordings:
                logging.error(
                    "No biomechanics recordings found for walk (any speed)"
                )
                return False
        else:
            study_name = biomechanics_file.stem.split("_")[0]
            recordings = import_biomechanics_recordings(
                biomechanics_file=biomechanics_file,
                maneuver=cast(
                    Literal["sit_to_stand", "flexion_extension"],
                    maneuver_key,
                ),
                speed=None,
                biomechanics_type=biomechanics_type,
                study_name=study_name,
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
                    biomechanics_type=biomechanics_type,
                )

                # Write the file
                output_path.parent.mkdir(parents=True, exist_ok=True)
                synced_df.to_pickle(output_path)
                logging.info("Saved synchronized data to %s", output_path)

                # Generate stomp visualization
                audio_stomp, bio_left, bio_right, detection_results = stomp_times
                plot_stomp_detection(audio_df, bio_df, synced_df, audio_stomp, bio_left, bio_right, output_path, detection_results)
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
    parser.add_argument(
        "--knee",
        type=str,
        choices=["left", "right"],
        help="Specify which knee to process (for filtering synced files)",
    )
    parser.add_argument(
        "--maneuver",
        type=str,
 choices=["walk", "fe", "sts"],

        help="Specify which maneuver to process (for filtering synced files)",
    )

    args = parser.parse_args()

    # Load environment defaults (e.g., AE_DATA_ROOT, AE_DATABASE_URL)
    load_env_file()

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

    # Resolve input path from CLI or environment
    resolved_path = Path(args.path) if args.path else get_data_root()
    if resolved_path is None:
        parser.print_help()
        logging.error("No PATH provided. Set AE_DATA_ROOT or pass PATH.")
        return

    path = resolved_path
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
    failed_participants: list[str] = []

    for participant_dir in participants:
        if process_participant(participant_dir, knee=args.knee, maneuver=args.maneuver):
            success_count += 1
        else:
            failure_count += 1
            failed_participants.append(participant_dir.name)

    # Summary
    logging.info(
        "Processing complete: %d succeeded, %d failed",
        success_count,
        failure_count,
    )

    if failure_count > 0:
        logging.warning(
            "Some participants failed processing: %s",
            ", ".join(failed_participants),
        )


if __name__ == "__main__":
    main()
