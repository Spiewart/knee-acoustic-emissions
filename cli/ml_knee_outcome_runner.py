#!/usr/bin/env python3
"""Shared runner for knee-level ML CLI commands."""

import argparse
import logging
from pathlib import Path
from typing import Optional

from src.analysis.ml_cli_helpers import (
    SCRIPTED_MANEUVERS,
    find_processed_participant_dirs,
    load_knee_outcomes_from_excel,
    run_knee_level_pipeline,
)

DEFAULT_PROJECT_DATA = "/Users/spiewart/kae_signal_processing_ml/sample_project_directory"
DEFAULT_OUTCOME_FILE = "cohort_chars_PRELIM_12_22_2025.xlsx"


def build_parser(command_help: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=command_help)
    parser.add_argument("project_data", type=Path, help="Path to project directory containing participant folders")
    parser.add_argument("--outcome-file", type=Path, default=Path(DEFAULT_PROJECT_DATA) / DEFAULT_OUTCOME_FILE, help=f"Path to Excel with knee-level outcomes (default: {DEFAULT_PROJECT_DATA}/{DEFAULT_OUTCOME_FILE})")
    parser.add_argument("--sheet", help="Sheet name when both knees are in one sheet")
    parser.add_argument("--left-sheet", help="Sheet name for left knee outcomes (separate sheets mode)")
    parser.add_argument("--right-sheet", help="Sheet name for right knee outcomes (separate sheets mode)")
    parser.add_argument("--side-column", default="Knee", help="Column indicating knee side (default: Knee)")
    parser.add_argument("--maneuvers", nargs="+", help="Optional maneuver filter (e.g., walk sit_to_stand)")
    parser.add_argument("--cycle-type", choices=["clean", "outliers"], default="clean")
    # Aggregation is fixed to per_cycle; option removed to simplify usage
    parser.add_argument("--allow-partial-knees", action="store_true", help="Include participants with only one processed knee")
    return parser


def run_knee_outcome_cli(
    outcome_column: str,
    default_sheet: Optional[str],
    argv: Optional[list[str]] = None,
) -> None:
    parser = build_parser(f"Run knee-level ML for {outcome_column}")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    project_dir = Path(args.project_data)
    maneuvers = args.maneuvers if args.maneuvers else SCRIPTED_MANEUVERS

    processed = find_processed_participant_dirs(
        project_dir,
        maneuvers=maneuvers,
        require_both_knees=not args.allow_partial_knees,
    )
    participant_ids = [study_id for study_id, _ in processed]
    logging.info("Found %d processed participants", len(participant_ids))
    if not participant_ids:
        logging.error("No processed participants found matching criteria")
        return

    outcome_file = Path(args.outcome_file) if args.outcome_file else project_dir / DEFAULT_OUTCOME_FILE
    outcome_df = load_knee_outcomes_from_excel(
        excel_path=outcome_file,
        outcome_column=outcome_column,
        side_column=args.side_column,
        sheet=args.sheet or default_sheet,
        left_sheet=args.left_sheet,
        right_sheet=args.right_sheet,
        knee_label_map={"Right": "R", "Left": "L"},
    )

    result = run_knee_level_pipeline(
        project_dir=project_dir,
        outcome_df=outcome_df,
        outcome_column=outcome_column,
        participant_ids=participant_ids,
        cycle_type=args.cycle_type,
        maneuvers=maneuvers,
    )

    print("\nML run complete")
    print(f"Mode: {result['mode']}")
    print(f"Samples: {result['samples']}")
    print(f"Accuracy: {result['accuracy']:.4f}")


__all__ = ["run_knee_outcome_cli"]
