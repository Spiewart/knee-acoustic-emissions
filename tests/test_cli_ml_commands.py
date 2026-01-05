from pathlib import Path

import pandas as pd

from cli.ml_kneepain import main as kneepain_main
from cli.ml_varusthrust import main as varusthrust_main

SCRIPTED = ["walk", "sit_to_stand", "flexion_extension"]


def _write_knee_log(path: Path, study_id: str, knee_side: str, maneuvers=SCRIPTED, num_synced=1):
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


def test_kneepain_cli_uses_processed_participants(monkeypatch, tmp_path, capsys):
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    p1 = project_dir / "#101"
    (p1 / "Left Knee").mkdir(parents=True)
    (p1 / "Right Knee").mkdir(parents=True)
    _write_knee_log(p1 / "Left Knee" / "knee_processing_log_101_Left.xlsx", "101", "Left")
    _write_knee_log(p1 / "Right Knee" / "knee_processing_log_101_Right.xlsx", "101", "Right")

    # Stub pipeline to avoid heavy processing
    captured_ids = {}

    def fake_pipeline(**kwargs):
        captured_ids["ids"] = kwargs.get("participant_ids")
        return {"mode": "loocv", "accuracy": 1.0, "samples": 1}

    monkeypatch.setattr("cli.ml_kneepain.run_participant_level_pipeline", fake_pipeline)

    kneepain_main([str(project_dir), "--demographics", str(project_dir / "demo.xlsx")])
    out = capsys.readouterr().out
    assert "Accuracy" in out
    assert captured_ids.get("ids") == ["101"]


def test_varusthrust_cli_loads_outcomes(monkeypatch, tmp_path, capsys):
    project_dir = tmp_path / "proj2"
    project_dir.mkdir()
    p1 = project_dir / "#201"
    (p1 / "Left Knee").mkdir(parents=True)
    (p1 / "Right Knee").mkdir(parents=True)
    _write_knee_log(p1 / "Left Knee" / "knee_processing_log_201_Left.xlsx", "201", "Left")
    _write_knee_log(p1 / "Right Knee" / "knee_processing_log_201_Right.xlsx", "201", "Right")

    fake_outcomes = pd.DataFrame({"study_id": [201], "knee": ["L"], "Varus Thrust": ["Yes"]})

    def fake_outcome_loader(**kwargs):
        return fake_outcomes

    def fake_pipeline(**kwargs):
        return {"mode": "loocv", "accuracy": 0.5, "samples": 1}

    monkeypatch.setattr("cli.ml_knee_outcome_runner.load_knee_outcomes_from_excel", fake_outcome_loader)
    monkeypatch.setattr("cli.ml_knee_outcome_runner.run_knee_level_pipeline", fake_pipeline)

    varusthrust_main([str(project_dir), "--outcome-file", str(project_dir / "outcomes.xlsx")])
    out = capsys.readouterr().out
    assert "Accuracy" in out
