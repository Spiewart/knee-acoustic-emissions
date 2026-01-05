#!/usr/bin/env python3
"""CLI for participant-level knee pain classification."""

import argparse
import logging
from pathlib import Path
from typing import Optional

from src.analysis.ml_cli_helpers import (
    SCRIPTED_MANEUVERS,
    find_processed_participant_dirs,
    run_participant_level_pipeline,
)

DEFAULT_PROJECT_DATA = "/Users/spiewart/kae_signal_processing_ml/sample_project_directory"
DEFAULT_DEMOGRAPHICS = "Winter 2023 Pilot Participant Demographics.xlsx"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ML model for knee pain (participant-level)")
    parser.add_argument("project_data", type=Path, help="Path to project directory containing participant folders")
    parser.add_argument("--demographics", type=Path, default=Path(DEFAULT_PROJECT_DATA) / DEFAULT_DEMOGRAPHICS, help=f"Path to demographics Excel (default: {DEFAULT_PROJECT_DATA}/{DEFAULT_DEMOGRAPHICS})")
    parser.add_argument("--maneuvers", nargs="+", help="Optional maneuver filter (e.g., walk sit_to_stand)")
    parser.add_argument("--cycle-type", choices=["clean", "outliers"], default="clean")
    parser.add_argument("--aggregation", choices=["mean", "median", "max", "min"], default="mean")
    parser.add_argument("--allow-partial-knees", action="store_true", help="Include participants with only one processed knee")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    project_dir = Path(args.project_data)
    demographics_path = Path(args.demographics)

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

    result = run_participant_level_pipeline(
        project_dir=project_dir,
        demographics_path=demographics_path,
        outcome_column="Knee Pain",
        participant_ids=participant_ids,
        cycle_type=args.cycle_type,
        maneuvers=maneuvers,
        aggregation=args.aggregation,
    )

    print("\nML run complete")
    print(f"Mode: {result['mode']}")
    print(f"Samples: {result['samples']}")
    print(f"Accuracy: {result['accuracy']:.4f}")


if __name__ == "__main__":
    main()
