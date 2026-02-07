#!/usr/bin/env python3
"""
Command-line interface for processing participant directories.

This script processes all participant directories in a folder, where each
directory should be named with format '#<study_id>'.
"""

import argparse
import logging
import os
from pathlib import Path

from src.config import get_data_root, load_env_file
from src.orchestration.participant import (
    find_participant_directories,
    process_participant,
    setup_logging,
    sync_single_audio_file,
)
from src.orchestration.persistent_processor import create_persistent_processor
from src.orchestration.processing_log import _infer_biomechanics_type_from_study


def main() -> None:
    """Process participant folders or sync a single audio file.

    Behavior:
    - With `--sync-single`, treats `path` as an audio pickle and runs sync.
    - Otherwise, `path` must be a directory containing participant folders
        named like `#<study_id>`. Filters via `--participant`, limits with
        `--limit`, and selects stage via `--entrypoint`.

    Logs progress and summaries; exits by returning from `main()` without
    raising to keep CLI-friendly behavior.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Process all participant directories in a folder. "
            "Each directory should be named with format '#<study_id>'."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python process_participant_directory.py /path/to/studies\n"
            "  python process_participant_directory.py /path/to/studies "
            "--limit 5\n"
            "  python process_participant_directory.py --sync-single "
            "/path/to/audio.pkl"
        ),
    )

    parser.add_argument(
        "path",
        nargs="?",
        help=(
            "Path to directory containing participant folders "
            "(e.g., #1011, #2024), or to audio file with --sync-single"
        ),
    )
    parser.add_argument(
        "--sync-single",
        action="store_true",
        help="Sync a single audio file (PATH should be audio file path)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N participant directories (0 = all, default: 0)",
    )
    parser.add_argument(
        "--participant",
        nargs="+",
        help=(
            "One or more participant folder names to process within PATH "
            "(with or without leading '#', e.g., 1011 #2024)"
        ),
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Optional path to write detailed log file",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--entrypoint",
        type=str,
        choices=["bin", "sync", "cycles"],
        default="sync",
        help=(
            "Stage to start processing from: 'bin' to (re)process raw audio & QC, "
            "'sync' to synchronize audio with biomechanics & QC, 'cycles' to run movement cycle QC only."
        ),
    )
    parser.add_argument(
        "--knee",
        type=str,
        choices=["left", "right"],
        help="Specify which knee to process: 'left' or 'right'.",
    )
    parser.add_argument(
        "--maneuver",
        type=str,
        choices=["walk", "fe", "sts"],
        help="Specify which maneuver to process: 'walk', 'fe', or 'sts'.",
    )
    parser.add_argument(
        "--persist-to-db",
        action="store_true",
        help=(
            "Enable optional database persistence. Requires AE_DATABASE_URL to be set. "
            "If database is unavailable, processing continues without saving to database."
        ),
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=None,
        help=(
            "PostgreSQL database URL for persistence (e.g., postgresql://user@localhost/db). "
            "Overrides AE_DATABASE_URL environment variable if provided."
        ),
    )

    args = parser.parse_args()

    # Load environment defaults (e.g., AE_DATA_ROOT, AE_DATABASE_URL)
    load_env_file()

    # Set up logging
    log_file = Path(args.log) if args.log else None
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(log_file, log_level)

    # Handle single-file sync
    if args.sync_single:
        if not args.path:
            logging.error(
                "--sync-single requires PATH argument (audio file path)"
            )
            return
        success = sync_single_audio_file(args.path)
        if not success:
            logging.error("Failed to sync audio file: %s", args.path)
        return

    # Resolve input path from CLI or environment
    resolved_path = Path(args.path) if args.path else get_data_root()
    if resolved_path is None:
        parser.print_help()
        logging.error("No PATH provided. Set AE_DATA_ROOT or pass PATH.")
        return

    path = resolved_path
    if not path.exists():
        logging.error("Path does not exist: %s", path)
        return
    if not path.is_dir():
        logging.error("Path is not a directory: %s", path)
        return

    # Infer study token from folder path (e.g., path contains 'AOA').
    # If not found, default to 'AOA' to preserve previous behavior.
    study_token = None
    for part in path.parts:
        if str(part).strip().lower() == "aoa":
            study_token = "AOA"
            break

    if study_token is None:
        study_token = "AOA"
        logging.warning(
            "Could not find explicit study folder in path %s; defaulting study to '%s'",
            path,
            study_token,
        )

    biomechanics_type = _infer_biomechanics_type_from_study(study_token)

    # Determine database URL for optional persistence
    db_url = None
    if args.persist_to_db or args.db_url:
        # Use explicit --db-url if provided, otherwise fall back to environment
        db_url = args.db_url or os.getenv("AE_DATABASE_URL")
        if db_url:
            logging.info("Database persistence enabled: %s", db_url)
        else:
            logging.warning(
                "Database persistence requested but no database URL available. "
                "Set AE_DATABASE_URL or use --db-url."
            )

    # Find participant directories
    participants = find_participant_directories(path)
    if not participants:
        logging.warning(
            "No participant directories found in %s "
            "(looking for folders named #<study_id>)",
            path,
        )
        return

    # Filter to specific participants if requested
    if args.participant:
        requested = {p.lstrip("#") for p in args.participant}
        participants = [d for d in participants if d.name.lstrip("#") in requested]
        if not participants:
            logging.warning(
                "No matching participant directories found for %s in %s",
                sorted(requested),
                path,
            )
            return

    # Apply limit if specified
    if args.limit > 0:
        participants = participants[: args.limit]

    logging.info(
        "Found %d participant directory(ies) to process", len(participants)
    )

    # Process each participant with optional database persistence
    success_count = 0
    failure_count = 0
    failed_participants: list[str] = []

    for participant_dir in participants:
        if args.persist_to_db or args.db_url:
            # Use persistent processor with optional database integration
            processor = create_persistent_processor(
                participant_dir=participant_dir,
                biomechanics_type=biomechanics_type,
                db_url=db_url,
            )
            success = processor.process(
                entrypoint=args.entrypoint,
                knee=args.knee,
                maneuver=args.maneuver,
            )
        else:
            # Use standard processor without database persistence
            success = process_participant(
                participant_dir,
                entrypoint=args.entrypoint,
                knee=args.knee,
                maneuver=args.maneuver,
                biomechanics_type=biomechanics_type,
            )

        if success:
            success_count += 1
        else:
            failure_count += 1
            failed_participants.append(participant_dir.name)

    # Summary
    logging.info(
        "Processing complete: %d succeeded, %d failed",
        success_count,
        failure_count,
    )

    if failure_count > 0:
        logging.warning(
            "Some participants failed processing: %s",
            ", ".join(failed_participants),
        )


if __name__ == "__main__":
    main()
