"""CLI for exporting channel data from a pickle file to CSV."""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.audio.exporters import dump_channels_to_csv


def main():
    """CLI for exporting channel data from a pickle file to CSV."""
    p = argparse.ArgumentParser()
    p.add_argument("pkl", help="Path to pickled DataFrame")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )

    pkl_path = Path(args.pkl)
    if not pkl_path.exists():
        logging.error("Pickle file not found: %s", pkl_path)
        raise SystemExit(2)

    base = pkl_path.stem
    csv_out = pkl_path.with_name(base + "_channels.csv")
    meta_json = pkl_path.with_name(base + "_meta.json")

    try:
        df = pd.read_pickle(pkl_path)
    except (OSError, EOFError, ValueError) as exc:
        logging.exception("Failed to read pickle: %s", exc)
        raise SystemExit(3)

    dump_channels_to_csv(df, csv_out, meta_json)
    print(f"Wrote CSV: {csv_out}")


if __name__ == "__main__":
    main()
