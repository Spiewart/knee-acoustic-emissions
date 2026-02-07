"""Shared helpers for ML CLI commands."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.preprocessing import StandardScaler

from src.analysis.data_loader import load_all_participant_cycles
from src.analysis.feature_extraction import extract_features_from_cycles
from src.analysis.models import (
    prepare_features_and_labels,
    prepare_knee_level_features_and_labels,
)
from src.orchestration.participant import (
    find_participant_directories,
    get_study_id_from_directory,
)

logger = logging.getLogger(__name__)

SCRIPTED_MANEUVERS: List[str] = ["walk", "sit_to_stand", "flexion_extension"]


def _normalize_knee_label(val: Any) -> str:
    sval = str(val).strip()
    if sval.lower() in {"right", "r"}:
        return "R"
    if sval.lower() in {"left", "l"}:
        return "L"
    return sval


def is_knee_fully_processed(
    study_id: str,
    knee_dir: Path,
    knee_side: Literal["Left", "Right"],
    maneuvers: Sequence[str] = SCRIPTED_MANEUVERS,
) -> bool:
    """Return True if the knee has processing logs.

    Checks for the existence of knee-level Excel log file as a proxy for processing completion.
    In a full DB implementation, this would query the database directly for sync records.
    """
    log_path = knee_dir / f"knee_processing_log_{study_id}_{knee_side}.xlsx"
    return log_path.exists()


def find_processed_participant_dirs(
    project_dir: Path,
    maneuvers: Sequence[str] = SCRIPTED_MANEUVERS,
    require_both_knees: bool = True,
) -> List[Tuple[str, Path]]:
    """Return (study_id, participant_dir) for participants with required processed knees."""
    processed: List[Tuple[str, Path]] = []
    for participant_dir in find_participant_directories(project_dir):
        study_id = get_study_id_from_directory(participant_dir)
        left_ok = is_knee_fully_processed(study_id, participant_dir / "Left Knee", "Left", maneuvers)
        right_ok = is_knee_fully_processed(study_id, participant_dir / "Right Knee", "Right", maneuvers)

        if require_both_knees:
            if left_ok and right_ok:
                processed.append((study_id, participant_dir))
        else:
            if left_ok or right_ok:
                processed.append((study_id, participant_dir))
    return processed


def _detect_outcome_format(df: pd.DataFrame) -> str:
    """Detect the format of knee outcome data.

    Returns:
        "wide": Format with separate "Right Knee" and "Left Knee" columns
        "long": Format with a side column (e.g., "Knee") and separate outcome column
    """
    if "Right Knee" in df.columns and "Left Knee" in df.columns:
        return "wide"
    return "long"


def _load_wide_format_outcomes(
    df: pd.DataFrame,
    outcome_column: str,
) -> pd.DataFrame:
    """Load outcomes from wide format (separate columns per knee).

    Expected columns: ["Study ID", "Right Knee", "Left Knee"]
    Output columns: ["Study ID", "knee", outcome_column]
    """
    frames = []

    # Process Right Knee
    right_df = df[["Study ID", "Right Knee"]].copy()
    right_df["knee"] = "R"
    right_df = right_df.rename(columns={"Right Knee": outcome_column})
    frames.append(right_df)

    # Process Left Knee
    left_df = df[["Study ID", "Left Knee"]].copy()
    left_df["knee"] = "L"
    left_df = left_df.rename(columns={"Left Knee": outcome_column})
    frames.append(left_df)

    return pd.concat(frames, ignore_index=True)


def _load_long_format_outcomes(
    df: pd.DataFrame,
    outcome_column: str,
    side_column: str = "Knee",
) -> pd.DataFrame:
    """Load outcomes from long format (single side column and outcome column).

    Expected columns: ["Study ID", side_column, outcome_column]
    Output columns: ["Study ID", "knee", outcome_column]
    """
    result = df.rename(columns={side_column: "knee"})
    return result[["Study ID", "knee", outcome_column]]


def _normalize_and_clean_outcomes(
    df: pd.DataFrame,
    outcome_column: str,
    knee_label_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Normalize knee labels and convert binary outcomes to numeric.

    Args:
        df: DataFrame with columns ["Study ID", "knee", outcome_column]
        outcome_column: Name of the outcome column
        knee_label_map: Optional mapping of knee labels (e.g., {"Right": "R", "Left": "L"})

    Returns:
        Cleaned DataFrame with standardized column names and data types
    """
    result = df.copy()
    result = result.rename(columns={"Study ID": "study_id"})

    def _normalize_study_id(val: Any) -> str:
        """Normalize study ID by removing common prefixes (e.g., AOA1001 -> 1001)."""
        sval = str(val).strip()
        # Remove common prefixes
        for prefix in ["AOA", "AO", "KAE"]:
            if sval.startswith(prefix):
                sval = sval[len(prefix):]
        return sval

    def _normalize(val: Any) -> str:
        sval = str(val).strip()
        if knee_label_map and sval in knee_label_map:
            sval = str(knee_label_map[sval]).strip()
        return _normalize_knee_label(sval)

    def _convert_binary_outcome(val: Any) -> int | float:
        """Convert binary outcomes (y/n/yes/no) to numeric (1/0)."""
        sval = str(val).strip().lower()
        if sval in {"y", "yes", "true", "1"}:
            return 1
        elif sval in {"n", "no", "false", "0"}:
            return 0
        else:
            try:
                return float(val)
            except (ValueError, TypeError):
                return val

    # Normalize study ID (remove prefixes like AOA)
    result["study_id"] = result["study_id"].apply(_normalize_study_id)
    # Convert to int if possible (to match cycles DataFrame which uses int study_id)
    try:
        result["study_id"] = result["study_id"].astype(int)
    except (ValueError, TypeError):
        # If conversion fails, keep as string
        pass

    result["knee"] = result["knee"].apply(_normalize)
    result[outcome_column] = result[outcome_column].apply(_convert_binary_outcome)

    # Verify required columns
    cols = ["study_id", "knee", outcome_column]
    missing_cols = [c for c in cols if c not in result.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns in outcome data: {missing_cols}")

    return result[cols]


def load_knee_outcomes_from_excel(
    excel_path: Path,
    outcome_column: str,
    side_column: str = "Knee",
    sheet: Optional[str] = None,
    left_sheet: Optional[str] = None,
    right_sheet: Optional[str] = None,
    knee_label_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Load knee-level outcomes from Excel.

    Supports three formats:
    1. Wide format with separate knee columns: ["Study ID", "Right Knee", "Left Knee"]
    2. Long format with side column: ["Study ID", "Knee", outcome_column]
    3. Separate sheets for left/right knees (left_sheet/right_sheet parameters)

    Returns a DataFrame with columns ["study_id", "knee", outcome_column].
    """
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Outcome Excel not found: {excel_path}")

    if left_sheet or right_sheet:
        # Separate sheets mode
        frames = []
        if left_sheet:
            left_df = pd.read_excel(excel_path, sheet_name=left_sheet)
            left_df[side_column] = "L"
            frames.append(left_df)
        if right_sheet:
            right_df = pd.read_excel(excel_path, sheet_name=right_sheet)
            right_df[side_column] = "R"
            frames.append(right_df)
        combined = pd.concat(frames, ignore_index=True)
    else:
        combined = pd.read_excel(excel_path, sheet_name=sheet)

    # Detect and load based on format
    format_type = _detect_outcome_format(combined)

    if format_type == "wide":
        combined = _load_wide_format_outcomes(combined, outcome_column)
    else:
        combined = _load_long_format_outcomes(combined, outcome_column, side_column)

    # Normalize and clean the data
    result = _normalize_and_clean_outcomes(combined, outcome_column, knee_label_map)

    return result


def train_with_fallback(
    X: pd.DataFrame, y: pd.Series, exclude_contralateral_knees: bool = True
) -> Dict[str, Any]:
    """Train logistic regression with LOOCV fallback for tiny datasets.

    When exclude_contralateral_knees is True and X has a MultiIndex with (study_id, knee),
    any knees from the same participant that differ only in side are excluded from training
    if their contralateral knee is in the test set. This prevents data leakage.
    """
    if y.nunique() < 2:
        raise ValueError("Outcome has fewer than 2 classes")

    # Check if we can do contralateral knee exclusion
    has_knee_index = (
        isinstance(X.index, pd.MultiIndex)
        and X.index.nlevels >= 2
        and X.index.names[0] == "study_id"
        and X.index.names[1] == "knee"
    )

    if len(y) <= 5:
        loo = LeaveOneOut()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        correct = 0
        for train_idx, test_idx in loo.split(X_scaled):
            train_X = X_scaled[train_idx]
            test_X = X_scaled[test_idx]
            train_y = y.iloc[train_idx]
            test_y = y.iloc[test_idx]

            # Exclude contralateral knees from training if applicable
            if exclude_contralateral_knees and has_knee_index:
                test_sample = X.iloc[test_idx]
                test_study_id = test_sample.index[0][0]
                test_knee = test_sample.index[0][1]
                contralateral_knee = "L" if test_knee == "R" else "R"

                # Find indices to exclude (contralateral knee of test sample's participant)
                exclude_mask = np.array(
                    [
                        idx[0] == test_study_id and idx[1] == contralateral_knee
                        for idx in X.index[train_idx]
                    ]
                )

                if exclude_mask.any():
                    keep_train_idx = np.where(~exclude_mask)[0]
                    train_X = train_X[keep_train_idx]
                    train_y = train_y.iloc[keep_train_idx]

            if len(train_y) == 0:
                # No training data left after exclusion
                continue

            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(train_X, train_y)
            pred = model.predict(test_X)[0]
            if pred == test_y.values[0]:
                correct += 1

        accuracy = correct / len(y)
        return {"mode": "loocv", "accuracy": accuracy, "samples": len(y)}

    # For larger datasets, use train-test split with contralateral exclusion
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Exclude contralateral knees from training
    if exclude_contralateral_knees and has_knee_index:
        test_study_ids = set(X_test.index.get_level_values("study_id").unique())
        test_knees_by_study = {}
        for study_id in test_study_ids:
            knees = X_test.loc[study_id].index.get_level_values("knee").unique()
            test_knees_by_study[study_id] = set(knees)

        # Filter training data to exclude contralateral knees
        exclude_mask = []
        for idx in X_train.index:
            study_id, knee = idx
            if study_id in test_knees_by_study:
                test_knees = test_knees_by_study[study_id]
                contralateral_knee = "L" if knee == "R" else "R"
                # Exclude if this is the contralateral of a knee in test
                if contralateral_knee in test_knees:
                    exclude_mask.append(False)
                else:
                    exclude_mask.append(True)
            else:
                exclude_mask.append(True)

        exclude_mask = np.array(exclude_mask)
        X_train = X_train[exclude_mask]
        y_train = y_train[exclude_mask]

        if len(y_train) == 0:
            raise ValueError(
                "No training samples remain after excluding contralateral knees. "
                "Consider disabling contralateral exclusion."
            )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    acc = model.score(X_test_scaled, y_test)
    return {
        "mode": "train_test_split",
        "accuracy": acc,
        "samples": len(y),
        "training_samples": len(y_train),
        "excluded_contralateral": exclude_mask.sum() - len(y_train) if exclude_contralateral_knees and has_knee_index else 0,
    }


def run_participant_level_pipeline(
    project_dir: Path,
    demographics_path: Path,
    outcome_column: str,
    participant_ids: List[str],
    cycle_type: Literal["clean", "outliers"] = "clean",
    maneuvers: Optional[List[str]] = None,
    aggregation: str = "mean",
) -> Dict[str, Any]:
    """Load cycles, extract features, and train participant-level model."""
    from src.analysis.data_loader import prepare_ml_dataset

    demographics, cycles = prepare_ml_dataset(
        demographics_path=demographics_path,
        project_dir=project_dir,
        outcome_column=outcome_column,
        cycle_type=cycle_type,
        maneuvers=maneuvers,
        participant_ids=[int(pid) for pid in participant_ids],
    )

    if cycles.empty:
        raise ValueError("No cycles found for selected participants")

    features_df = extract_features_from_cycles(cycles)
    X, y = prepare_features_and_labels(
        features_df=features_df,
        outcome_df=demographics,
        outcome_column=outcome_column,
        aggregation=aggregation,
    )

    return train_with_fallback(X, y)


def run_knee_level_pipeline(
    project_dir: Path,
    outcome_df: pd.DataFrame,
    outcome_column: str,
    participant_ids: List[str],
    cycle_type: Literal["clean", "outliers"] = "clean",
    maneuvers: Optional[List[str]] = None,
    exclude_contralateral_knees: bool = True,
) -> Dict[str, Any]:
    """Load cycles, extract features, and train knee-level model."""
    cycles = load_all_participant_cycles(
        project_dir=project_dir,
        cycle_type=cycle_type,
        maneuvers=maneuvers,
        participant_ids=participant_ids,
    )
    if cycles.empty:
        raise ValueError("No cycles found for selected participants")

    features_df = extract_features_from_cycles(cycles)

    X, y = prepare_knee_level_features_and_labels(
        features_df=features_df,
        outcome_column=outcome_column,
        outcome_df=outcome_df,
    )

    return train_with_fallback(X, y, exclude_contralateral_knees=exclude_contralateral_knees)
