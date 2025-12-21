#!/usr/bin/env python3
"""
Command-line interface for processing participant directories.

This script processes all participant directories in a folder, where each
directory should be named with format '#<study_id>'.
"""

import argparse
import logging
from pathlib import Path

from src.orchestration.participant import (
    find_participant_directories,
    process_participant,
    setup_logging,
    sync_single_audio_file,
)


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
        "--entrypoint",
        type=str,
        choices=["bin", "sync", "cycles"],
        default="sync",
        help=(
            "Stage to start processing from: 'bin' to (re)process raw audio & QC, "
            "'sync' to synchronize audio with biomechanics & QC, 'cycles' to run movement cycle QC only."
        ),
    )

    args = parser.parse_args()

    # Set up logging
    log_file = Path(args.log) if args.log else None
    setup_logging(log_file)

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

    # Validate input path
    if not args.path:
        parser.print_help()
        return

    path = Path(args.path)
    if not path.exists():
        logging.error("Path does not exist: %s", path)
        return
    if not path.is_dir():
        logging.error("Path is not a directory: %s", path)
        return

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

    # Process each participant
    success_count = 0
    failure_count = 0

    for participant_dir in participants:
        if process_participant(participant_dir, entrypoint=args.entrypoint):
            success_count += 1
        else:
            failure_count += 1

    # Summary
    logging.info(
        "Processing complete: %d succeeded, %d failed",
        success_count,
        failure_count,
    )

    if failure_count > 0:
        logging.warning(
            "Some participants failed processing; check logs for details"
        )


if __name__ == "__main__":
    main()
