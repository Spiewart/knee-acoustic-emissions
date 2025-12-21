#!/usr/bin/env python3
"""Command-line interface for synchronized data QC."""

import logging
import sys
from pathlib import Path

from src.synchronization.quality_control import (
    find_synced_files,
    perform_sync_qc,
    setup_logging,
)


def main() -> int:
    """Run QC over synced files at `path` and summarize.

        - Accepts a single synced pickle, a `Synced` directory, or a participant
            directory; finds all synced `.pkl` files under `Synced` folders.
        - Controls acoustic energy threshold with `--threshold` and plotting
            via `--no-plots`.
        - Optional `--maneuver`/`--speed` override inference.

        Returns 0 on success, non-zero if path missing or no synced files found.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Perform QC analysis on synchronized audio and biomechanics data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "path",
        type=Path,
        help="Path to synced pickle file, Synced directory, or participant directory",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=100.0,
        help="Minimum acoustic energy threshold per cycle (default: 100.0)",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip creation of visualization plots",
    )

    parser.add_argument(
        "--maneuver",
        type=str,
        choices=["walk", "sit_to_stand", "flexion_extension"],
        help="Override maneuver type inference",
    )

    parser.add_argument(
        "--speed",
        type=str,
        choices=["slow", "medium", "fast"],
        help="Override speed inference (for walking)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)

    # Validate path
    if not args.path.exists():
        logger.error(f"Path does not exist: {args.path}")
        return 1

    # Find files to process
    files_to_process = find_synced_files(args.path)

    if not files_to_process:
        logger.warning(f"No synced files found in {args.path}")
        return 1

    logger.info(f"Found {len(files_to_process)} synced file(s) to process")

    # Process each file
    total_clean = 0
    total_outliers = 0

    for synced_file in files_to_process:
        logger.info(f"\nProcessing: {synced_file}")

        try:
            clean_cycles, outlier_cycles, output_dir = perform_sync_qc(
                synced_file,
                maneuver=args.maneuver,
                speed=args.speed,
                acoustic_threshold=args.threshold,
                create_plots=not args.no_plots,
            )

            total_clean += len(clean_cycles)
            total_outliers += len(outlier_cycles)

            logger.info(
                f"✓ Complete: {len(clean_cycles)} clean, {len(outlier_cycles)} outliers "
                f"→ {output_dir}"
            )

        except Exception as e:
            logger.error(f"✗ Failed to process {synced_file}: {e}", exc_info=args.verbose)
            continue

    logger.info(f"\n{'='*60}")
    logger.info(f"Total: {total_clean} clean cycles, {total_outliers} outliers")
    logger.info(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
