import pandas as pd

from src.analysis.models import (
    prepare_features_and_labels,
    prepare_knee_level_features_and_labels,
)


def test_prepare_features_and_labels_basic_mean():
    features_df = pd.DataFrame(
        [
            {
                "study_id": 101,
                "knee": "R",
                "maneuver": "walk",
                "speed": "slow",
                "cycle_index": 0,
                "feat_a": 1.0,
                "feat_b": 2.0,
            },
            {
                "study_id": 101,
                "knee": "L",
                "maneuver": "walk",
                "speed": "slow",
                "cycle_index": 1,
                "feat_a": 3.0,
                "feat_b": 4.0,
            },
            {
                "study_id": 102,
                "knee": "R",
                "maneuver": "walk",
                "speed": "fast",
                "cycle_index": 0,
                "feat_a": 5.0,
                "feat_b": 6.0,
            },
        ]
    )

    outcome_df = pd.DataFrame(
        {
            "Study ID": [101, 102],
            "Knee Pain": ["Yes", "No"],
        }
    )

    X, y = prepare_features_and_labels(
        features_df=features_df,
        outcome_df=outcome_df,
        outcome_column="Knee Pain",
        aggregation="mean",
    )

    assert list(y.values) == [1, 0]
    assert X.shape == (2, 2)
    assert X.loc[101, "feat_a"] == 2.0  # mean of 1 and 3
    assert X.loc[101, "feat_b"] == 3.0  # mean of 2 and 4
    assert X.loc[102, "feat_a"] == 5.0
    assert X.loc[102, "feat_b"] == 6.0


def test_prepare_knee_level_features_single_sheet():
    features_df = pd.DataFrame(
        [
            {"study_id": 101, "knee": "R", "maneuver": "walk", "speed": "slow", "cycle_index": 0, "feat": 1.0},
            {"study_id": 101, "knee": "R", "maneuver": "walk", "speed": "slow", "cycle_index": 1, "feat": 3.0},
            {"study_id": 101, "knee": "L", "maneuver": "walk", "speed": "slow", "cycle_index": 0, "feat": 5.0},
            {"study_id": 102, "knee": "R", "maneuver": "walk", "speed": "fast", "cycle_index": 0, "feat": 7.0},
            {"study_id": 102, "knee": "L", "maneuver": "walk", "speed": "fast", "cycle_index": 0, "feat": 9.0},
        ]
    )

    outcome_df = pd.DataFrame(
        {
            "Study ID": [101, 101, 102, 102],
            "Knee": ["Right", "Left", "Right", "Left"],
            "Varus Thrust": ["Yes", "No", "No", "Yes"],
        }
    )

    X, y = prepare_knee_level_features_and_labels(
        features_df=features_df,
        outcome_column="Varus Thrust",
        outcome_df=outcome_df,
        side_column="Knee",
        knee_label_map={"Right": "R", "Left": "L"},
    )

    # Expect 5 per-cycle rows (participant 101 R has 2 cycles, others have 1 each)
    assert X.shape == (5, 1)
    # Check that we have both cycles for participant 101 R
    idx_list = X.index.tolist()
    assert idx_list.count((101, "R")) == 2
    assert idx_list.count((101, "L")) == 1
    assert idx_list.count((102, "R")) == 1
    assert idx_list.count((102, "L")) == 1
    # Per-cycle feature values
    assert X.loc[(101, "R"), "feat"].tolist() == [1.0, 3.0]
    assert X.loc[(101, "L"), "feat"].iloc[0] == 5.0
    assert X.loc[(102, "R"), "feat"].iloc[0] == 7.0
    assert X.loc[(102, "L"), "feat"].iloc[0] == 9.0


def test_prepare_knee_level_features_per_knee_sheets():
    features_df = pd.DataFrame(
        [
            {"study_id": 201, "knee": "R", "maneuver": "walk", "speed": "slow", "cycle_index": 0, "feat": 2.0},
            {"study_id": 201, "knee": "L", "maneuver": "walk", "speed": "slow", "cycle_index": 0, "feat": 4.0},
            {"study_id": 202, "knee": "R", "maneuver": "walk", "speed": "fast", "cycle_index": 0, "feat": 6.0},
            {"study_id": 202, "knee": "L", "maneuver": "walk", "speed": "fast", "cycle_index": 0, "feat": 8.0},
        ]
    )

    outcome_df_per_knee = {
        "R": pd.DataFrame({"Study ID": [201, 202], "KOOS": ["Yes", "No"]}),
        "L": pd.DataFrame({"Study ID": [201, 202], "KOOS": ["No", "Yes"]}),
    }

    X, y = prepare_knee_level_features_and_labels(
        features_df=features_df,
        outcome_column="KOOS",
        outcome_df_per_knee=outcome_df_per_knee,
    )

    assert X.shape == (4, 1)
    assert set(X.index) == {(201, "R"), (201, "L"), (202, "R"), (202, "L")}
    assert y.loc[(201, "R")] == 1
    assert y.loc[(201, "L")] == 0
    assert y.loc[(202, "R")] == 0
    assert y.loc[(202, "L")] == 1
    assert X.loc[(201, "R"), "feat"] == 2.0
    assert X.loc[(201, "L"), "feat"] == 4.0
    assert X.loc[(202, "R"), "feat"] == 6.0
    assert X.loc[(202, "L"), "feat"] == 8.0
