#!/usr/bin/env python3
"""
Command-line interface for reading audio board files.

This script reads binary audio board files and converts them to CSV format.
"""

import argparse

from src.audio.readers import read_audio_board_file


def main() -> None:
    """Main entry point for audio board file reader CLI."""
    parser = argparse.ArgumentParser(description="Read binary audio board file and convert to CSV")
    parser.add_argument(
        "fname",
        help="Path to binary audio board file",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output CSV file path (defaults to input filename with .csv extension)",
    )

    args = parser.parse_args()
    read_audio_board_file(args.fname, args.out)


if __name__ == "__main__":
    main()
