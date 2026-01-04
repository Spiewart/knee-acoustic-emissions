"""Data loading utilities for ML analysis."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def load_demographics(
    filepath: Path,
    outcome_column: str = "Knee Pain",
) -> pd.DataFrame:
    """Load participant demographics data.

    Args:
        filepath: Path to demographics Excel file
        outcome_column: Name of outcome column (default: "Knee Pain")

    Returns:
        DataFrame with demographics and outcome variable

    Raises:
        FileNotFoundError: If demographics file doesn't exist
        ValueError: If outcome column is missing
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Demographics file not found: {filepath}")

    df = pd.read_excel(filepath)

    if outcome_column not in df.columns:
        raise ValueError(
            f"Outcome column '{outcome_column}' not found in demographics. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Clean up whitespace in string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip() if hasattr(df[col], 'str') else df[col]

    logger.info(f"Loaded demographics: {len(df)} participants")
    return df


def load_movement_cycles(
    participant_dir: Path,
    cycle_type: Literal["clean", "outliers"] = "clean",
    maneuvers: Optional[List[str]] = None,
) -> List[Tuple[int, Path, Dict[str, Any], pd.DataFrame]]:
    """Load movement cycle data for a participant.

    Args:
        participant_dir: Path to participant directory (e.g., #1020)
        cycle_type: Load "clean" or "outliers" cycles
        maneuvers: List of maneuver names to load, or None for all
            (e.g., ["Walking", "Flexion-Extension", "Sit-to-Stand"])

    Returns:
        List of tuples: (study_id, pkl_path, metadata_dict, cycle_dataframe)

    Raises:
        FileNotFoundError: If participant directory doesn't exist
    """
    if not participant_dir.exists():
        raise FileNotFoundError(f"Participant directory not found: {participant_dir}")

    # Extract study ID from folder name
    study_id = int(participant_dir.name.lstrip("#"))

    # Find all movement cycle directories
    cycles: List[Tuple[int, Path, Dict[str, Any], pd.DataFrame]] = []
    maneuver_counts = {"flexion-extension": 0, "sit-to-stand": 0, "walking": 0}

    # Search through knees and maneuvers
    for knee_dir in participant_dir.glob("*Knee"):
        def _normalize_key(name: str) -> str:
            return name.lower().replace(" ", "_").replace("-", "_")

        def _candidate_folders(key: str) -> List[str]:
            if key in {"walk", "walking"}:
                return ["Walking", "Walk"]
            if key in {"sit_to_stand", "sit_stand"}:
                return ["Sit-Stand", "Sit_to_Stand", "Sit-to-Stand", "Sit Stand"]
            if key in {"flexion_extension", "flexionextension"}:
                return ["Flexion-Extension", "Flexion_Extension", "Flexion Extension"]
            return [key]

        if maneuvers is not None:
            maneuver_dirs = []
            maneuver_labels: List[str] = []
            for m in maneuvers:
                key = _normalize_key(m)
                found = None
                for cand in _candidate_folders(key):
                    path = knee_dir / cand
                    if path.exists():
                        found = path
                        break
                if found:
                    maneuver_dirs.append(found)
                    maneuver_labels.append(key)
        else:
            maneuver_dirs = [d for d in knee_dir.iterdir() if d.is_dir()]
            maneuver_labels = [d.name.lower().replace(" ", "_").replace("-", "_") for d in maneuver_dirs]

        for maneuver_dir, maneuver_key in zip(maneuver_dirs, maneuver_labels):
            cycle_dir = maneuver_dir / "Synced" / "MovementCycles" / cycle_type

            if not cycle_dir.exists():
                continue

            # Load all pickle files with their metadata
            for pkl_file in sorted(cycle_dir.glob("*.pkl")):
                json_file = pkl_file.with_suffix(".json")

                # Load metadata
                metadata = {}
                if json_file.exists():
                    try:
                        with open(json_file, 'r') as f:
                            metadata = json.load(f)
                    except Exception as e:
                        logger.warning(f"Failed to load metadata for {pkl_file}: {e}")

                # Load cycle dataframe
                try:
                    cycle_df = pd.read_pickle(pkl_file)
                    cycles.append((study_id, pkl_file, metadata, cycle_df))
                    if "flexion" in maneuver_key:
                        maneuver_counts["flexion-extension"] += 1
                    elif "sit" in maneuver_key:
                        maneuver_counts["sit-to-stand"] += 1
                    elif "walk" in maneuver_key:
                        maneuver_counts["walking"] += 1
                except Exception as e:
                    logger.warning(f"Failed to load cycle {pkl_file}: {e}")

    if len(cycles) == 0:
        logger.warning(f"Loaded 0 {cycle_type} cycles for participant {study_id}")
    else:
        logger.info(
            "Loaded %d %s cycles for participant %s (flexion-extension: %d, sit-to-stand: %d, walking: %d)",
            len(cycles),
            cycle_type,
            study_id,
            maneuver_counts["flexion-extension"],
            maneuver_counts["sit-to-stand"],
            maneuver_counts["walking"],
        )
    return cycles


def load_all_participant_cycles(
    project_dir: Path,
    cycle_type: Literal["clean", "outliers"] = "clean",
    maneuvers: Optional[List[str]] = None,
    participant_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load movement cycles from all participants in a project.

    Args:
        project_dir: Path to project directory containing participant folders
        cycle_type: Load "clean" or "outliers" cycles
        maneuvers: List of maneuver names to load, or None for all

    Returns:
        DataFrame with columns: study_id, cycle_path, knee, maneuver, etc.
    """
    all_cycles = []

    allowed_ids = {str(pid) for pid in participant_ids} if participant_ids else None

    for participant_dir in sorted(project_dir.glob("#*")):
        if not participant_dir.is_dir():
            continue

        study_id = participant_dir.name.lstrip("#")
        if allowed_ids is not None and study_id not in allowed_ids:
            continue

        try:
            cycles = load_movement_cycles(
                participant_dir,
                cycle_type=cycle_type,
                maneuvers=maneuvers,
            )
            all_cycles.extend(cycles)
        except Exception as e:
            logger.warning(f"Failed to load cycles for {participant_dir.name}: {e}")

    if not all_cycles:
        logger.warning("No cycles found")
        return pd.DataFrame()

    # Convert to DataFrame for easier manipulation
    records = []
    for study_id, pkl_path, metadata, cycle_df in all_cycles:
        records.append({
            "study_id": study_id,
            "cycle_path": str(pkl_path),
            "knee": metadata.get("knee"),
            "maneuver": metadata.get("scripted_maneuver"),
            "speed": metadata.get("speed"),
            "pass_number": metadata.get("pass_number"),
            "cycle_index": metadata.get("cycle_index"),
            "cycle_acoustic_energy": metadata.get("cycle_acoustic_energy"),
            "cycle_qc_pass": metadata.get("cycle_qc_pass"),
            "cycle_df": cycle_df,
            "metadata": metadata,
        })

    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} total cycles from {df['study_id'].nunique()} participants")
    return df


def prepare_ml_dataset(
    demographics_path: Path,
    project_dir: Path,
    outcome_column: str = "Knee Pain",
    cycle_type: Literal["clean", "outliers"] = "clean",
    maneuvers: Optional[List[str]] = None,
    participant_ids: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare combined dataset for ML analysis.

    Args:
        demographics_path: Path to demographics Excel file
        project_dir: Path to project directory with participant data
        outcome_column: Name of outcome column in demographics
        cycle_type: Load "clean" or "outliers" cycles
        maneuvers: List of maneuver names to load, or None for all
        participant_ids: List of participant IDs to include (e.g., [1011, 1013, 1020]).
            If None, uses all available participants.

    Returns:
        Tuple of (demographics_df, cycles_df) with aligned study IDs
    """
    # Load demographics
    demographics = load_demographics(demographics_path, outcome_column)

    # Filter to specified participant IDs if provided
    if participant_ids is not None:
        demographics = demographics[demographics["Study ID"].isin(participant_ids)]
        logger.info(f"Filtered to {len(demographics)} specified participants")

    # Load all cycles (optionally filtered to participants)
    participant_ids_str = [str(pid) for pid in participant_ids] if participant_ids is not None else None
    cycles = load_all_participant_cycles(
        project_dir,
        cycle_type,
        maneuvers,
        participant_ids=participant_ids_str,
    )

    if cycles.empty:
        logger.warning("No cycles loaded - returning empty DataFrames")
        return demographics, pd.DataFrame()

    # Filter cycles to only participants in demographics
    available_ids = set(demographics["Study ID"].values)
    cycles = cycles[cycles["study_id"].isin(available_ids)]

    logger.info(
        f"Prepared ML dataset: {len(demographics)} participants, "
        f"{len(cycles)} cycles"
    )

    return demographics, cycles
