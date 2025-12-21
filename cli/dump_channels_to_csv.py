"""CLI for exporting channel data from a pickle file to CSV."""

import argparse
import logging
import pickle
from pathlib import Path

import pandas as pd

from src.audio.exporters import dump_channels_to_csv


def main() -> None:
    """Export time and audio channel data from a pickled DataFrame to CSV.

    Reads a pickled pandas DataFrame containing audio data and writes a CSV file
    with columns: tt (time), ch1, ch2, ch3, ch4. Missing channels are filled with NaN.
    If a sibling _meta.json file exists, attempts to derive timestamps from it if
    the 'tt' column is missing.

    Exit codes:
    - 0: Success
    - 2: Pickle file not found
    - 3: Unreadable pickle (corrupted or invalid format)
    """
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
    except (OSError, EOFError, ValueError, pickle.UnpicklingError) as exc:
        logging.exception("Failed to read pickle: %s", exc)
        raise SystemExit(3)

    dump_channels_to_csv(df, csv_out, meta_json)
    print(f"Wrote CSV: {csv_out}")


if __name__ == "__main__":
    main()
