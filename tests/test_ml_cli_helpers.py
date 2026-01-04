import pandas as pd

from src.analysis.ml_cli_helpers import (
    find_processed_participant_dirs,
    load_knee_outcomes_from_excel,
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
