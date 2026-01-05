"""Machine learning model training and evaluation."""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def prepare_features_and_labels(
    features_df: pd.DataFrame,
    outcome_df: pd.DataFrame,
    outcome_column: str = "Knee Pain",
    aggregation: str = "mean",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix and labels for ML training.

    Args:
        features_df: DataFrame with extracted features (one row per cycle)
        outcome_df: DataFrame with outcome variable (e.g., demographics, clinical data).
            Must have a "Study ID" column matching the study_id in features_df.
        outcome_column: Name of outcome column in outcome_df
        aggregation: How to aggregate features per participant
            ("mean", "median", "max", "min")

    Returns:
        Tuple of (X, y) where:
            X: Feature matrix with one row per participant
            y: Binary outcome labels (0/1)
    """
    # Aggregate features by participant
    feature_cols = [
        col for col in features_df.columns
        if col not in ["study_id", "knee", "maneuver", "speed", "cycle_index"]
    ]

    if aggregation == "mean":
        agg_features = features_df.groupby("study_id")[feature_cols].mean()
    elif aggregation == "median":
        agg_features = features_df.groupby("study_id")[feature_cols].median()
    elif aggregation == "max":
        agg_features = features_df.groupby("study_id")[feature_cols].max()
    elif aggregation == "min":
        agg_features = features_df.groupby("study_id")[feature_cols].min()
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")

    # Merge with outcome data to get labels
    outcome_indexed = outcome_df.set_index("Study ID")
    merged = agg_features.join(outcome_indexed[outcome_column], how="inner")

    # Convert outcome to binary (0/1)
    y = merged[outcome_column].map(lambda x: 1 if str(x).lower().strip() == "yes" else 0)
    X = merged[feature_cols]

    logger.info(
        f"Prepared {len(X)} samples with {len(feature_cols)} features. "
        f"Outcome distribution: {y.value_counts().to_dict()}"
    )

    return X, y


def prepare_knee_level_features_and_labels(
    features_df: pd.DataFrame,
    outcome_column: str,
    outcome_df: Optional[pd.DataFrame] = None,
    outcome_df_per_knee: Optional[Dict[str, pd.DataFrame]] = None,
    side_column: str = "knee",
    knee_label_map: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix and labels at the knee level using per-cycle features.

    Supports two patterns for outcome data:
    1) Single sheet with a knee side column (e.g., Varus Thrust sheet).
    2) Separate sheets for each knee (e.g., "KOOS R Knee" and "KOOS L Knee").

    Each cycle becomes a separate training sample with its study_id and knee side.

    Args:
        features_df: DataFrame with extracted features (one row per cycle) containing
            "study_id" and a knee side column (default: "knee").
        outcome_column: Name of the outcome column in the outcome data.
        outcome_df: Outcome DataFrame when both knees are in one sheet and
            differentiated by ``side_column``.
        outcome_df_per_knee: Mapping of knee label to outcome DataFrame when each
            knee is in a separate sheet. Keys should correspond to knee labels
            (e.g., {"R": df_right, "L": df_left}).
        side_column: Column in outcome data indicating knee side (ignored when
            using ``outcome_df_per_knee`` because the side is inferred from the
            mapping key).
        knee_label_map: Optional mapping to normalize knee labels between outcome
            data and features (e.g., {"Right": "R", "Left": "L"}).

    Returns:
        Tuple of (X, y) where:
            X: Feature matrix with one row per cycle (indexed by study_id, knee)
            y: Binary outcome labels (0/1) corresponding to the study_id/knee
    """
    if outcome_df is None and outcome_df_per_knee is None:
        raise ValueError("Provide either outcome_df or outcome_df_per_knee")

    # Extract feature columns (all columns except metadata)
    feature_cols = [
        col for col in features_df.columns
        if col not in ["study_id", "knee", "maneuver", "speed", "cycle_index"]
    ]

    # Normalize knee labels in features
    agg_features = features_df.copy()

    def _normalize_knee(val: Any) -> str:
        sval = str(val).strip()
        if knee_label_map and sval in knee_label_map:
            sval = str(knee_label_map[sval]).strip()
        # Default to upper-case short labels if not mapped
        if sval.lower() in {"right", "r"}:
            return "R"
        if sval.lower() in {"left", "l"}:
            return "L"
        return sval

    agg_features["knee"] = agg_features["knee"].apply(_normalize_knee)
    agg_features = agg_features.set_index(["study_id", "knee"]).sort_index()

    # Build a combined outcome DataFrame with standardized knee labels
    if outcome_df_per_knee is not None:
        outcome_frames = []
        for knee_label, df in outcome_df_per_knee.items():
            df_copy = df.copy()
            df_copy[side_column] = knee_label
            outcome_frames.append(df_copy)
        combined_outcomes = pd.concat(outcome_frames, ignore_index=True)
    else:
        combined_outcomes = outcome_df.copy()

    combined_outcomes[side_column] = combined_outcomes[side_column].apply(_normalize_knee)

    # Standardize outcome column names to align indexes
    combined_outcomes = combined_outcomes.rename(columns={"Study ID": "study_id", side_column: "knee"})
    combined_outcomes["study_id"] = combined_outcomes["study_id"]
    combined_outcomes = combined_outcomes.set_index(["study_id", "knee"])

    # Validate uniqueness to avoid silent drops
    dup_outcomes = combined_outcomes.index[combined_outcomes.index.duplicated()].unique()
    if len(dup_outcomes) > 0:
        raise ValueError(
            "Duplicate outcome entries found for study/knee combinations: "
            f"{dup_outcomes.tolist()}"
        )

    merged = agg_features.join(combined_outcomes[outcome_column], how="inner")

    # Derive y: prefer numeric labels; fall back to boolean yes/no mapping; otherwise keep raw
    outcome_series = merged[outcome_column]
    y: pd.Series
    # Try numeric conversion first
    y_numeric = pd.to_numeric(outcome_series, errors="coerce")
    if not y_numeric.isna().any():
        y = y_numeric
    else:
        normalized = outcome_series.astype(str).str.lower().str.strip()
        yes_vals = {"yes", "y", "true", "1"}
        no_vals = {"no", "n", "false", "0"}
        unique_vals = set(normalized.unique())
        if unique_vals <= yes_vals | no_vals:
            y = normalized.map(lambda v: 1 if v in yes_vals else 0)
        else:
            y = outcome_series

    X = merged[feature_cols]

    logger.info(
        f"Prepared {len(X)} knee-level samples with {len(feature_cols)} features. "
        f"Outcome distribution: {y.value_counts().to_dict()}"
    )

    return X, y


def train_logistic_regression(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    scale_features: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """Train a logistic regression model.

    Args:
        X: Feature matrix
        y: Binary outcome labels
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        scale_features: Whether to standardize features
        **kwargs: Additional arguments passed to LogisticRegression

    Returns:
        Dictionary containing:
            - model: Trained LogisticRegression model
            - scaler: StandardScaler (if scale_features=True) or None
            - X_train, X_test, y_train, y_test: Train/test splits
            - metrics: Dictionary of evaluation metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train.values
        X_test_scaled = X_test.values

    # Train model
    model = LogisticRegression(random_state=random_state, max_iter=1000, **kwargs)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    y_prob_test = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "train_accuracy": accuracy_score(y_train, y_pred_train),
        "test_accuracy": accuracy_score(y_test, y_pred_test),
        "test_auc": roc_auc_score(y_test, y_prob_test),
        "confusion_matrix": confusion_matrix(y_test, y_pred_test),
        "classification_report": classification_report(y_test, y_pred_test),
    }

    logger.info(f"Model trained: Train acc={metrics['train_accuracy']:.3f}, "
                f"Test acc={metrics['test_accuracy']:.3f}, "
                f"Test AUC={metrics['test_auc']:.3f}")

    return {
        "model": model,
        "scaler": scaler,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "metrics": metrics,
    }


def evaluate_model(
    result: Dict[str, Any],
    top_n_features: int = 10,
) -> None:
    """Print detailed model evaluation.

    Args:
        result: Dictionary returned by train_logistic_regression
        top_n_features: Number of top features to display by coefficient magnitude
    """
    metrics = result["metrics"]

    print("\n" + "="*60)
    print("LOGISTIC REGRESSION MODEL EVALUATION")
    print("="*60)

    print(f"\nTrain Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Test Accuracy:  {metrics['test_accuracy']:.4f}")
    print(f"Test AUC:       {metrics['test_auc']:.4f}")

    print("\nConfusion Matrix (Test Set):")
    print(metrics['confusion_matrix'])

    print("\nClassification Report (Test Set):")
    print(metrics['classification_report'])

    # Feature importance (coefficient magnitudes)
    model = result["model"]
    X_train = result["X_train"]

    coef_df = pd.DataFrame({
        "feature": X_train.columns,
        "coefficient": model.coef_[0],
        "abs_coefficient": np.abs(model.coef_[0]),
    }).sort_values("abs_coefficient", ascending=False)

    print(f"\nTop {top_n_features} Most Important Features:")
    print(coef_df.head(top_n_features).to_string(index=False))

    print("\n" + "="*60)


def cross_validate_model(
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 5,
    random_state: int = 42,
    scale_features: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """Perform k-fold cross-validation.

    Args:
        X: Feature matrix
        y: Binary outcome labels
        n_folds: Number of cross-validation folds
        random_state: Random seed for reproducibility
        scale_features: Whether to standardize features
        **kwargs: Additional arguments passed to LogisticRegression

    Returns:
        Dictionary with cross-validation results
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    accuracies = []
    aucs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train_fold = X.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_train_fold = y.iloc[train_idx]
        y_test_fold = y.iloc[test_idx]

        # Scale
        if scale_features:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_test_scaled = scaler.transform(X_test_fold)
        else:
            X_train_scaled = X_train_fold.values
            X_test_scaled = X_test_fold.values

        # Train and evaluate
        model = LogisticRegression(random_state=random_state, max_iter=1000, **kwargs)
        model.fit(X_train_scaled, y_train_fold)

        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        accuracies.append(accuracy_score(y_test_fold, y_pred))
        aucs.append(roc_auc_score(y_test_fold, y_prob))

        logger.info(f"Fold {fold+1}/{n_folds}: Acc={accuracies[-1]:.3f}, AUC={aucs[-1]:.3f}")

    results = {
        "accuracies": accuracies,
        "aucs": aucs,
        "mean_accuracy": np.mean(accuracies),
        "std_accuracy": np.std(accuracies),
        "mean_auc": np.mean(aucs),
        "std_auc": np.std(aucs),
    }

    logger.info(
        f"Cross-validation complete: "
        f"Acc={results['mean_accuracy']:.3f}±{results['std_accuracy']:.3f}, "
        f"AUC={results['mean_auc']:.3f}±{results['std_auc']:.3f}"
    )

    return results
