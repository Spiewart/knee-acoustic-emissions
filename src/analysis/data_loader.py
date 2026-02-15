"""Data loading utilities for ML analysis."""

import logging
from pathlib import Path
from typing import Any, Literal

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
            f"Outcome column '{outcome_column}' not found in demographics. Available columns: {df.columns.tolist()}"
        )

    # Clean up whitespace in string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip() if hasattr(df[col], "str") else df[col]

    logger.info(f"Loaded demographics: {len(df)} participants")
    return df


def load_movement_cycles(
    participant_dir: Path,
    cycle_type: Literal["clean", "outliers"] = "clean",
    maneuvers: list[str] | None = None,
) -> list[tuple[int, Path, dict[str, Any], pd.DataFrame]]:
    """Load movement cycle data for a participant.

    TODO: Reimplement as a DB query via the ORM (MovementCycleRecord).
    The previous implementation read per-cycle JSON metadata files from disk.
    The new pipeline passes cycle metadata in-memory via CycleQCResult dataclasses
    and persists to PostgreSQL. This function should query the DB instead.

    Args:
        participant_dir: Path to participant directory (e.g., #1020)
        cycle_type: Load "clean" or "outliers" cycles
        maneuvers: List of maneuver names to load, or None for all

    Returns:
        List of tuples: (study_id, pkl_path, metadata_dict, cycle_dataframe)
    """
    raise NotImplementedError(
        "load_movement_cycles must be reimplemented as a DB query. "
        "JSON cycle metadata has been removed from the pipeline."
    )


def load_all_participant_cycles(
    project_dir: Path,
    cycle_type: Literal["clean", "outliers"] = "clean",
    maneuvers: list[str] | None = None,
    participant_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Load movement cycles from all participants in a project.

    TODO: Reimplement as a DB query via the ORM (MovementCycleRecord).
    The previous implementation aggregated per-cycle JSON metadata from disk.
    The new pipeline persists cycle metadata to PostgreSQL via CycleQCResult
    dataclasses. This function should query the DB instead.

    Args:
        project_dir: Path to project directory containing participant folders
        cycle_type: Load "clean" or "outliers" cycles
        maneuvers: List of maneuver names to load, or None for all

    Returns:
        DataFrame with columns: study_id, cycle_path, knee, maneuver, etc.
    """
    raise NotImplementedError(
        "load_all_participant_cycles must be reimplemented as a DB query. "
        "JSON cycle metadata has been removed from the pipeline."
    )


def prepare_ml_dataset(
    demographics_path: Path,
    project_dir: Path,
    outcome_column: str = "Knee Pain",
    cycle_type: Literal["clean", "outliers"] = "clean",
    maneuvers: list[str] | None = None,
    participant_ids: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    logger.info(f"Prepared ML dataset: {len(demographics)} participants, {len(cycles)} cycles")

    return demographics, cycles
