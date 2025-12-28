#!/usr/bin/env python3
"""Summarize processing logs for a participant.

Usage:
  python scripts/summarize_processing_logs.py /path/to/participant/#1020

Prints a concise summary of Synchronization QC, Movement Cycles aggregates,
per-cycle count, and Biomechanics stats across all knee/maneuver logs.
"""

import glob
import os
import sys
from pathlib import Path

import pandas as pd


def summarize_log(log_path: Path) -> None:
    print(f"\n=== {log_path} ===")
    try:
        # Summary sheet (optional for context)
        try:
            summary = pd.read_excel(log_path, sheet_name="Summary")
            if not summary.empty:
                row = summary.iloc[0]
                print(
                    f"Knee={row.get('Knee Side')}, Maneuver={row.get('Maneuver')}, "
                    f"Synced={row.get('Num Synced Files')}, CyclesAnalyses={row.get('Num Movement Cycle Analyses')}"
                )
        except Exception:
            pass

        # Synchronization sheet
        try:
            sync_df = pd.read_excel(log_path, sheet_name="Synchronization")
            if not sync_df.empty and "Sync File" in sync_df.columns:
                total = len(sync_df)
                qc_done = int(sync_df.get("Sync QC Done", pd.Series(dtype=bool)).fillna(False).sum())
                qc_pass = int(sync_df.get("Sync QC Passed", pd.Series(dtype=bool)).fillna(False).sum())
                print(f"Synchronization: rows={total}, qc_done={qc_done}, qc_pass={qc_pass}")
        except Exception as e:
            print(f"Synchronization: ERROR reading sheet: {e}")

        # Movement Cycles sheet
        try:
            mc_df = pd.read_excel(log_path, sheet_name="Movement Cycles")
            if not mc_df.empty and "Source Sync File" in mc_df.columns:
                total_cycles = int(mc_df.get("Total Cycles", pd.Series(dtype=int)).fillna(0).sum())
                clean_cycles = int(mc_df.get("Clean Cycles", pd.Series(dtype=int)).fillna(0).sum())
                outlier_cycles = int(mc_df.get("Outlier Cycles", pd.Series(dtype=int)).fillna(0).sum())
                mean_dur = mc_df.get("Mean Duration (s)", pd.Series(dtype=float)).dropna().mean()
                mean_auc = mc_df.get("Mean Acoustic AUC", pd.Series(dtype=float)).dropna().mean()
                mean_dur_str = f"{mean_dur:.3f}" if pd.notna(mean_dur) else "n/a"
                mean_auc_str = f"{mean_auc:.3f}" if pd.notna(mean_auc) else "n/a"
                print(
                    "Movement Cycles: "
                    f"total={total_cycles}, clean={clean_cycles}, outliers={outlier_cycles}, "
                    f"mean_duration_s={mean_dur_str}, "
                    f"mean_auc={mean_auc_str}"
                )
        except Exception as e:
            print(f"Movement Cycles: ERROR reading sheet: {e}")

        # Cycle Details sheet
        try:
            details_df = pd.read_excel(log_path, sheet_name="Cycle Details")
            if not details_df.empty:
                print(f"Cycle Details: rows={len(details_df)}")
        except Exception:
            pass

        # Biomechanics sheet
        try:
            bio_df = pd.read_excel(log_path, sheet_name="Biomechanics")
            if not bio_df.empty and "Biomechanics File" in bio_df.columns:
                row = bio_df.iloc[0]
                sr = row.get("Sample Rate (Hz)")
                start_s = row.get("Start Time (s)")
                end_s = row.get("End Time (s)")
                dur_s = row.get("Duration (s)")
                print(
                    f"Biomechanics: sample_rate_hz={sr}, start_s={start_s}, end_s={end_s}, duration_s={dur_s}"
                )
        except Exception as e:
            print(f"Biomechanics: ERROR reading sheet: {e}")

    except Exception as e:
        print(f"ERROR summarizing {log_path}: {e}")


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/summarize_processing_logs.py /path/to/participant/#ID")
        return 1

    participant_dir = Path(sys.argv[1])
    if not participant_dir.exists():
        print(f"Participant directory not found: {participant_dir}")
        return 1

    # Find all processing logs under this participant
    pattern = os.path.join(str(participant_dir), "**", "processing_log_*.xlsx")
    files = glob.glob(pattern, recursive=True)
    if not files:
        print("No processing logs found.")
        return 0

    for f in sorted(files):
        summarize_log(Path(f))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
