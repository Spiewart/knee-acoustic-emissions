#!/usr/bin/env python3
"""
CLI for exporting time and channel columns (ch1..ch4) from a pickled
DataFrame to CSV. Uses `src.audio.exporters.export_channels_to_csv`.
"""

import argparse
import logging
from pathlib import Path

from src.audio.exporters import export_channels_to_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Export time and channels from audio pickle to CSV")
    parser.add_argument("pkl", help="Path to pickled DataFrame")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    pkl_path = Path(args.pkl)
    if not pkl_path.exists():
        logging.error("Pickle file not found: %s", pkl_path)
        return 2

    try:
        csv_out = export_channels_to_csv(pkl_path)
    except Exception as exc:
        logging.exception("Failed to export channels: %s", exc)
        return 3

    logging.info("Wrote CSV: %s", csv_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
