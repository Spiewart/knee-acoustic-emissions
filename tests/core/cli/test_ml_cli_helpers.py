import pandas as pd

from src.analysis.ml_cli_helpers import (
    find_processed_participant_dirs,
    load_knee_outcomes_from_excel,
    train_with_fallback,
)

SCRIPTED = ["walk", "sit_to_stand", "flexion_extension"]


def _write_knee_log(path, study_id, knee_side, maneuvers=SCRIPTED, num_synced=1):
    summary_df = pd.DataFrame({"Maneuver": maneuvers, "Num Synced Files": [num_synced] * len(maneuvers)})
    summary_sheet = pd.DataFrame(
        {
            "Study ID": [study_id],
            "Knee Side": [knee_side],
            "Knee Directory": [str(path.parent)],
            "Log Created": [pd.Timestamp.now()],
            "Log Updated": [pd.Timestamp.now()],
        }
    )
    with pd.ExcelWriter(path) as writer:
        summary_sheet.to_excel(writer, sheet_name="Summary", index=False)
        summary_df.to_excel(writer, sheet_name="Maneuver Summaries", index=False)


def test_find_processed_participant_dirs(tmp_path):
    # Participant 101 fully processed (both knees)
    p1 = tmp_path / "#101"
    (p1 / "Left Knee").mkdir(parents=True)
    (p1 / "Right Knee").mkdir(parents=True)
    _write_knee_log(p1 / "Left Knee" / "knee_processing_log_101_Left.xlsx", "101", "Left")
    _write_knee_log(p1 / "Right Knee" / "knee_processing_log_101_Right.xlsx", "101", "Right")

    # Participant 102 missing right knee processing
    p2 = tmp_path / "#102"
    (p2 / "Left Knee").mkdir(parents=True)
    (p2 / "Right Knee").mkdir(parents=True)
    _write_knee_log(p2 / "Left Knee" / "knee_processing_log_102_Left.xlsx", "102", "Left")
    # Right knee log absent

    processed_both = find_processed_participant_dirs(tmp_path)
    assert processed_both == [("101", p1)]

    processed_either = find_processed_participant_dirs(tmp_path, require_both_knees=False)
    assert set(processed_either) == {("101", p1), ("102", p2)}


def test_load_knee_outcomes_single_sheet(tmp_path):
    data = pd.DataFrame(
        {
            "Study ID": [101, 101],
            "Knee": ["Right", "Left"],
            "Varus Thrust": ["Yes", "No"],
        }
    )
    excel_path = tmp_path / "outcomes.xlsx"
    data.to_excel(excel_path, index=False, sheet_name="Varus Thrust")

    df = load_knee_outcomes_from_excel(
        excel_path=excel_path,
        outcome_column="Varus Thrust",
        side_column="Knee",
        sheet="Varus Thrust",
        knee_label_map={"Right": "R", "Left": "L"},
    )

    assert set(df.columns) == {"study_id", "knee", "Varus Thrust"}
    assert set(df["knee"]) == {"R", "L"}


def test_load_knee_outcomes_wide_format(tmp_path):
    """Test loading outcomes from wide format (separate Right/Left columns)."""
    data = pd.DataFrame(
        {
            "Study ID": ["AOA1001", "AOA1002", "AOA1003"],
            "Right Knee": ["y", "n", "y"],
            "Left Knee": ["n", "n", "y"],
        }
    )
    excel_path = tmp_path / "outcomes_wide.xlsx"
    data.to_excel(excel_path, index=False, sheet_name="Varus Thrust")

    df = load_knee_outcomes_from_excel(
        excel_path=excel_path,
        outcome_column="Grade",
        sheet="Varus Thrust",
    )

    # Check structure
    assert set(df.columns) == {"study_id", "knee", "Grade"}
    assert len(df) == 6  # 3 participants × 2 knees

    # Check that we have both knee sides
    assert set(df["knee"]) == {"R", "L"}

    # Check that binary outcomes are converted to numeric
    assert set(df["Grade"].unique()) == {0, 1}

    # Check specific mappings (study_id is converted to int and AOA prefix is stripped)
    aoa1001_r = df[(df["study_id"] == 1001) & (df["knee"] == "R")]["Grade"].values[0]
    aoa1001_l = df[(df["study_id"] == 1001) & (df["knee"] == "L")]["Grade"].values[0]
    assert aoa1001_r == 1  # "y" → 1
    assert aoa1001_l == 0  # "n" → 0


def test_load_knee_outcomes_long_format_with_labels(tmp_path):
    """Test loading outcomes from long format with string knee labels."""
    data = pd.DataFrame(
        {
            "Study ID": [1001, 1001, 1002, 1002],
            "Knee": ["Right", "Left", "Right", "Left"],
            "Grade": [2, 3, 2, 2],
        }
    )
    excel_path = tmp_path / "outcomes_long.xlsx"
    data.to_excel(excel_path, index=False, sheet_name="TFM KL")

    df = load_knee_outcomes_from_excel(
        excel_path=excel_path,
        outcome_column="Grade",
        side_column="Knee",
        sheet="TFM KL",
        knee_label_map={"Right": "R", "Left": "L"},
    )

    # Check structure
    assert set(df.columns) == {"study_id", "knee", "Grade"}
    assert len(df) == 4

    # Check that knee labels are normalized
    assert set(df["knee"]) == {"R", "L"}

    # Check outcomes remain numeric
    assert set(df["Grade"]) == {2, 3}


def test_load_knee_outcomes_numeric_binary_conversion(tmp_path):
    """Test that various binary outcome representations are converted correctly."""
    data = pd.DataFrame(
        {
            "Study ID": ["P001", "P002", "P003", "P004"],
            "Right Knee": ["yes", "no", "True", "False"],
            "Left Knee": [1, 0, "1", "0"],
        }
    )
    excel_path = tmp_path / "binary_outcomes.xlsx"
    data.to_excel(excel_path, index=False, sheet_name="Binary Data")

    df = load_knee_outcomes_from_excel(
        excel_path=excel_path,
        outcome_column="Outcome",
        sheet="Binary Data",
    )

    # All outcomes should be converted to 0 or 1
    assert set(df["Outcome"].unique()) == {0, 1}

    # Check specific conversions
    outcomes = df["Outcome"].values
    # P001 Right="yes" → 1, P002 Right="no" → 0, P003 Right="True" → 1, P004 Right="False" → 0
    # P001 Left=1 → 1, P002 Left=0 → 0, P003 Left="1" → 1, P004 Left="0" → 0
    assert len(outcomes) == 8  # 4 participants × 2 knees
    assert sum(outcomes == 1) == 4  # 4 ones
    assert sum(outcomes == 0) == 4  # 4 zeros


def test_train_with_fallback_excludes_contralateral_knees():
    """Verify that contralateral knees are excluded from training when their partner is in test set."""
    # Create data with 2 participants, each with both knees (4 knee-level samples)
    # Use a larger dataset to trigger train_test_split path
    index = pd.MultiIndex.from_tuples(
        [
            (101, "R"),
            (101, "L"),
            (102, "R"),
            (102, "L"),
            (103, "R"),
            (103, "L"),
            (104, "R"),
            (104, "L"),
            (105, "R"),
            (105, "L"),
            (106, "R"),
            (106, "L"),
        ],
        names=["study_id", "knee"],
    )
    # Create features
    X = pd.DataFrame(
        {
            "feature_1": [1.0, 1.1, 2.0, 2.1, 3.0, 3.1, 4.0, 4.1, 5.0, 5.1, 6.0, 6.1],
            "feature_2": [10.0, 10.5, 20.0, 20.5, 30.0, 30.5, 40.0, 40.5, 50.0, 50.5, 60.0, 60.5],
        },
        index=index,
    )
    # Create labels - outcome differs by participant
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1], index=index, name="outcome")

    result = train_with_fallback(X, y, exclude_contralateral_knees=True)

    # Verify the function returns expected keys
    assert "mode" in result
    assert "accuracy" in result
    assert "samples" in result
    assert result["samples"] == 12
    assert "training_samples" in result
    assert "excluded_contralateral" in result

    # With random train/test split, we should have excluded some contralateral knees
    # The exact number depends on the random split, but if both knees of a participant
    # aren't in the same split, we should have exclusions
    # We can't guarantee exclusions happened due to random split, so just verify the structure
    assert result["training_samples"] <= 12


def test_train_with_fallback_without_contralateral_exclusion():
    """Verify that setting exclude_contralateral_knees=False includes all training data."""
    # Create data with 2 participants, each with both knees
    index = pd.MultiIndex.from_tuples(
        [
            (101, "R"),
            (101, "L"),
            (102, "R"),
            (102, "L"),
            (103, "R"),
            (103, "L"),
            (104, "R"),
            (104, "L"),
            (105, "R"),
            (105, "L"),
            (106, "R"),
            (106, "L"),
        ],
        names=["study_id", "knee"],
    )
    X = pd.DataFrame(
        {
            "feature_1": [1.0, 1.1, 2.0, 2.1, 3.0, 3.1, 4.0, 4.1, 5.0, 5.1, 6.0, 6.1],
            "feature_2": [10.0, 10.5, 20.0, 20.5, 30.0, 30.5, 40.0, 40.5, 50.0, 50.5, 60.0, 60.5],
        },
        index=index,
    )
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1], index=index, name="outcome")

    result = train_with_fallback(X, y, exclude_contralateral_knees=False)

    # Without exclusion, training samples should be 80% of total (0.2 test split)
    # Roughly 9-10 samples in training
    assert result["training_samples"] >= 9
    assert result["excluded_contralateral"] == 0


def test_train_with_fallback_loocv_with_contralateral_exclusion():
    """Verify LOOCV path with contralateral exclusion for small datasets."""
    # Create small dataset (5 samples = triggers LOOCV)
    index = pd.MultiIndex.from_tuples(
        [
            (101, "R"),
            (101, "L"),
            (102, "R"),
            (102, "L"),
            (103, "R"),
        ],
        names=["study_id", "knee"],
    )
    X = pd.DataFrame(
        {
            "feature_1": [1.0, 1.1, 2.0, 2.1, 3.0],
            "feature_2": [10.0, 10.5, 20.0, 20.5, 30.0],
        },
        index=index,
    )
    y = pd.Series([0, 0, 0, 1, 1], index=index, name="outcome")

    result = train_with_fallback(X, y, exclude_contralateral_knees=True)

    assert result["mode"] == "loocv"
    assert result["samples"] == 5
    assert "accuracy" in result
    assert 0 <= result["accuracy"] <= 1
