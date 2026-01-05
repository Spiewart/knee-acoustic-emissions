"""Demo: Knee Pain Classification using Movement Cycle Features.

This script demonstrates using movement cycle acoustic features to predict
knee pain status using logistic regression.
"""

import logging
from pathlib import Path

import pandas as pd

from src.analysis.data_loader import prepare_ml_dataset
from src.analysis.feature_extraction import extract_features_from_cycles
from src.analysis.models import (
    cross_validate_model,
    evaluate_model,
    prepare_features_and_labels,
    train_logistic_regression,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Run knee pain classification demo."""
    # Paths
    project_dir = Path("/Users/spiewart/kae_signal_processing_ml/sample_project_directory")
    demographics_path = project_dir / "Winter 2023 Pilot Participant Demographics.xlsx"

    # Participants with complete processed data
    participant_ids = [1011, 1013, 1016, 1019, 1020]

    logger.info("="*70)
    logger.info("KNEE PAIN CLASSIFICATION DEMO")
    logger.info("="*70)

    # Step 1: Load data
    logger.info("\n1. Loading demographics and movement cycles...")
    demographics, cycles = prepare_ml_dataset(
        demographics_path=demographics_path,
        project_dir=project_dir,
        outcome_column="Knee Pain",
        cycle_type="clean",
        maneuvers=None,  # Load all maneuvers
        participant_ids=participant_ids,
    )

    if cycles.empty:
        logger.error("No cycles found. Exiting.")
        return

    logger.info(f"   Loaded {len(demographics)} participants")
    logger.info(f"   Loaded {len(cycles)} clean movement cycles")
    logger.info(f"   Maneuvers: {cycles['maneuver'].value_counts().to_dict()}")

    # Step 2: Extract features
    logger.info("\n2. Extracting features from movement cycles...")
    features_df = extract_features_from_cycles(cycles)
    logger.info(f"   Extracted {len(features_df.columns)} features")

    # Step 3: Prepare for ML
    logger.info("\n3. Preparing feature matrix and labels...")
    X, y = prepare_features_and_labels(
        features_df=features_df,
        outcome_df=demographics,
        outcome_column="Knee Pain",
        aggregation="mean",  # Average features across all cycles per participant
    )

    logger.info(f"   Feature matrix shape: {X.shape}")
    logger.info(f"   Outcome distribution: {y.value_counts().to_dict()}")

    # Step 4: Train and evaluate model (with small dataset, use leave-one-out CV)
    logger.info("\n4. Training logistic regression model with Leave-One-Out cross-validation...")
    logger.info("   (Using LOOCV due to small dataset size: 5 participants)")

    # For small datasets, use leave-one-out cross-validation
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    predictions = []
    accuracies = []

    for train_idx, test_idx in loo.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]
        predictions.append(pred)
        accuracies.append(pred == y_test.values[0])

    # Final model on all data
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_scaled, y)

    print("\n" + "="*60)
    print("LEAVE-ONE-OUT CROSS-VALIDATION RESULTS")
    print("="*60)
    loo_accuracy = np.mean(accuracies)
    print(f"Accuracy: {loo_accuracy:.4f}")
    print(f"Correct predictions: {sum(accuracies)} / {len(accuracies)}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)

    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    print("="*60 + "\n")

    logger.info("Demo complete!")


if __name__ == "__main__":
    main()
