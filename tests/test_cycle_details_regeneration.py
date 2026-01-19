from pathlib import Path

import numpy as np
import pandas as pd

from src.metadata import MovementCycles, Synchronization
from src.orchestration.processing_log import (
    ManeuverProcessingLog,
    create_cycles_record_from_data,
)


def _make_cycle_df(n=100, with_tt=True, use_filtered=False):
    cols = {}
    if with_tt:
        cols['tt'] = np.linspace(0.0, 1.0, n)
    # create either filtered or raw channel columns
    ch_prefix = 'f_ch' if use_filtered else 'ch'
    for i in range(1, 5):
        cols[f"{ch_prefix}{i}"] = np.abs(np.sin(np.linspace(0, np.pi, n))) * (10 * i)
    return pd.DataFrame(cols)


def test_cycle_details_regenerates_across_passes_flat_dir(tmp_path):
    # Prepare flat MovementCycles directory
    out_dir = tmp_path / "MovementCycles" / "clean"
    out_dir.mkdir(parents=True)

    # Two different sync stems for two passes
    stem1 = "left_walk_Pass001_normal"
    stem2 = "left_walk_Pass012_fast"

    # Write cycle PKLs for each stem
    df1_clean = _make_cycle_df(n=120, with_tt=True, use_filtered=False)
    df1_outlier = _make_cycle_df(n=80, with_tt=True, use_filtered=False)
    df1_clean.to_pickle(out_dir / f"{stem1}_cycle_001.pkl")
    df1_outlier.to_pickle(out_dir / f"{stem1}_outlier_001.pkl")

    df2_c1 = _make_cycle_df(n=150, with_tt=True, use_filtered=True)
    df2_c2 = _make_cycle_df(n=160, with_tt=True, use_filtered=True)
    df2_c1.to_pickle(out_dir / f"{stem2}_cycle_001.pkl")
    df2_c2.to_pickle(out_dir / f"{stem2}_cycle_002.pkl")

    # Add an unrelated pkl to ensure it's ignored
    other_df = _make_cycle_df(n=90, with_tt=True)
    other_df.to_pickle(out_dir / "other_sync_cycle_001.pkl")

    # Build Maneuver log with two MovementCycles records pointing to same flat dir
    log = ManeuverProcessingLog(
        study_id="1013",
        knee_side="Left",
        maneuver="walk",
        maneuver_directory=tmp_path,
    )

    rec1 = MovementCycles(
        sync_file_name=stem1,
        pass_number=1,
        speed="normal",
        output_directory=str(out_dir),
        total_cycles_extracted=2,
        clean_cycles=1,
        outlier_cycles=1,
        processing_status="success",
    )
    rec2 = MovementCycles(
        sync_file_name=stem2,
        pass_number=12,
        speed="fast",
        output_directory=str(out_dir),
        total_cycles_extracted=2,
        clean_cycles=2,
        outlier_cycles=0,
        processing_status="success",
    )
    log.add_movement_cycles_record(rec1)
    log.add_movement_cycles_record(rec2)

    excel_path = tmp_path / "processing_log.xlsx"
    log.save_to_excel(excel_path)

    # Read Cycle Details and validate rows per pass
    details = pd.read_excel(excel_path, sheet_name="Cycle Details")
    assert len(details) == 4  # 2 for stem1 (clean+outlier), 2 for stem2

    by_sync = details.groupby("Sync File").size().to_dict()
    assert by_sync.get(stem1) == 2
    assert by_sync.get(stem2) == 2

    # Validate context columns present and sensible values
    assert set(["Study ID", "Knee Side", "Maneuver", "Pass Number", "Speed"]).issubset(details.columns)
    assert details[details["Sync File"] == stem1]["Pass Number"].iloc[0] == 1
    assert details[details["Sync File"] == stem2]["Pass Number"].iloc[0] == 12

    # Acoustic metrics present
    assert details["Acoustic AUC"].notna().all()
    # RMS columns may be NaN if computed against missing channels; ensure columns exist
    for col in ["Ch1 RMS", "Ch2 RMS", "Ch3 RMS", "Ch4 RMS"]:
        assert col in details.columns


def test_cycle_details_fallback_without_output_dir(tmp_path):
    # Build in-memory cycles using helper without output_dir
    clean_cycles = [_make_cycle_df(n=50, with_tt=True), _make_cycle_df(n=60, with_tt=True)]
    outliers = [_make_cycle_df(n=40, with_tt=True)]

    sync = Synchronization(sync_file_name="left_walk_Pass002_normal", pass_number=2, speed="normal")
    rec = create_cycles_record_from_data(
        sync_file_name="left_walk_Pass002_normal",
        clean_cycles=clean_cycles,
        outlier_cycles=outliers,
        output_dir=None,
        sync_record=sync,
    )

    log = ManeuverProcessingLog(
        study_id="1013",
        knee_side="Left",
        maneuver="walk",
        maneuver_directory=tmp_path,
    )
    log.add_movement_cycles_record(rec)

    excel_path = tmp_path / "processing_log_no_outdir.xlsx"
    log.save_to_excel(excel_path)

    details = pd.read_excel(excel_path, sheet_name="Cycle Details")
    # Expect 3 rows (2 clean + 1 outlier) from in-memory per_cycle_details
    assert len(details) == 3
    assert set(["Cycle Index", "Duration (s)", "Acoustic AUC", "Sync File"]).issubset(details.columns)


def test_cycle_details_handles_missing_tt(tmp_path):
    out_dir = tmp_path / "MovementCycles" / "clean"
    out_dir.mkdir(parents=True)

    stem = "left_walk_Pass003_normal"
    df_no_tt = _make_cycle_df(n=30, with_tt=False)
    df_no_tt.to_pickle(out_dir / f"{stem}_cycle_001.pkl")

    log = ManeuverProcessingLog(
        study_id="1013",
        knee_side="Left",
        maneuver="walk",
        maneuver_directory=tmp_path,
    )
    rec = MovementCycles(
        sync_file_name=stem,
        pass_number=3,
        speed="normal",
        output_directory=str(out_dir),
        processing_status="success",
        total_cycles_extracted=1,
        clean_cycles=1,
        outlier_cycles=0,
    )
    log.add_movement_cycles_record(rec)

    excel_path = tmp_path / "processing_log_missing_tt.xlsx"
    log.save_to_excel(excel_path)

    details = pd.read_excel(excel_path, sheet_name="Cycle Details")
    assert len(details) == 1
    # Duration should still be present (0.0 fallback when tt missing)
    assert float(details["Duration (s)"].iloc[0]) >= 0.0


def test_cycle_details_roundtrip_then_regenerate(tmp_path):
    out_dir = tmp_path / "MovementCycles" / "clean"
    out_dir.mkdir(parents=True)

    stem1 = "left_walk_Pass004_normal"
    stem2 = "left_walk_Pass005_fast"
    _make_cycle_df(50).to_pickle(out_dir / f"{stem1}_cycle_001.pkl")
    _make_cycle_df(55).to_pickle(out_dir / f"{stem2}_cycle_001.pkl")

    log = ManeuverProcessingLog(
        study_id="1013",
        knee_side="Left",
        maneuver="walk",
        maneuver_directory=tmp_path,
    )
    log.add_movement_cycles_record(MovementCycles(sync_file_name=stem1, pass_number=4, speed="normal", output_directory=str(out_dir), processing_status="success"))
    log.add_movement_cycles_record(MovementCycles(sync_file_name=stem2, pass_number=5, speed="fast", output_directory=str(out_dir), processing_status="success"))

    excel1 = tmp_path / "processing_log_roundtrip.xlsx"
    log.save_to_excel(excel1)
    # Load back (Cycle Details is intentionally not loaded)
    loaded = ManeuverProcessingLog.load_from_excel(excel1)
    # Re-save to a new file; Cycle Details should regenerate with both passes
    excel2 = tmp_path / "processing_log_roundtrip_resave.xlsx"
    loaded.save_to_excel(excel2)

    details = pd.read_excel(excel2, sheet_name="Cycle Details")
    assert len(details) == 2
    assert set(details["Sync File"]) == {stem1, stem2}
