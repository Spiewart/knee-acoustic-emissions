"""Shared helpers for ML CLI commands."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

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
from src.orchestration.processing_log import KneeProcessingLog

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
    """Return True if the knee processing log exists and all maneuvers have >=1 synced files."""
    log_path = knee_dir / f"knee_processing_log_{study_id}_{knee_side}.xlsx"
    log = KneeProcessingLog.load_from_excel(log_path)
    if log is None:
        logger.debug("Missing knee log: %s", log_path)
        return False

    summaries = {str(s.get("Maneuver")): s for s in log.maneuver_summaries}
    for maneuver in maneuvers:
        summary = summaries.get(maneuver)
        if not summary:
            logger.debug("Missing maneuver %s in log %s", maneuver, log_path)
            return False
        num_synced = summary.get("Num Synced Files", 0) or 0
        try:
            if float(num_synced) < 1:
                logger.debug("Maneuver %s has insufficient synced files in %s", maneuver, log_path)
                return False
        except Exception:
            logger.debug("Invalid Num Synced Files for %s in %s", maneuver, log_path)
            return False
    return True


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

    Supports either a single sheet with a side column, or separate sheets for left/right knees.
    Returns a DataFrame with columns ["Study ID", "knee", outcome_column].
    """
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Outcome Excel not found: {excel_path}")

    def _normalize(val: Any) -> str:
        sval = str(val).strip()
        if knee_label_map and sval in knee_label_map:
            sval = str(knee_label_map[sval]).strip()
        return _normalize_knee_label(sval)

    if left_sheet or right_sheet:
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

    combined = combined.rename(columns={"Study ID": "study_id", side_column: "knee"})
    combined["knee"] = combined["knee"].apply(_normalize)
    # Keep only necessary columns
    cols = ["study_id", "knee", outcome_column]
    missing_cols = [c for c in cols if c not in combined.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns in outcome sheet: {missing_cols}")

    return combined[cols]


def train_with_fallback(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """Train logistic regression with LOOCV fallback for tiny datasets."""
    if y.nunique() < 2:
        raise ValueError("Outcome has fewer than 2 classes")

    if len(y) <= 5:
        loo = LeaveOneOut()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        correct = 0
        for train_idx, test_idx in loo.split(X_scaled):
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_scaled[train_idx], y.iloc[train_idx])
            pred = model.predict(X_scaled[test_idx])[0]
            if pred == y.iloc[test_idx].values[0]:
                correct += 1
        accuracy = correct / len(y)
        return {"mode": "loocv", "accuracy": accuracy, "samples": len(y)}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
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
        aggregation="per_cycle",
        outcome_df=outcome_df,
    )

    return train_with_fallback(X, y)
