"""CLI for adding instantaneous frequency to a pickle file."""

import argparse
import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.audio.instantaneous_frequency import add_instantaneous_frequency


def determine_fs_and_dt(df: pd.DataFrame, meta_json: Path) -> Tuple[float, float]:
    """Determine sampling frequency and time step from DataFrame or metadata.

    Attempts to derive sampling rate and time delta in this order:
    1. From `df['tt']` column (timestamp array) if present and contains valid data.
    2. From `meta_json` 'fs' field if the JSON file exists and contains an 'fs' key.

    Raises `RuntimeError` if neither source provides valid sampling frequency.

    Args:
        df: DataFrame containing audio data, optionally with 'tt' time column.
        meta_json: Path to metadata JSON file that may contain 'fs' field.

    Returns:
        Tuple of (sampling_frequency_Hz, time_delta_seconds).

    Raises:
        RuntimeError: If sampling frequency cannot be determined.
    """
    if "tt" in df.columns and df["tt"].notna().any():
        tt = df["tt"].astype(float).to_numpy()
        dt = float(np.median(np.diff(tt)))
        return 1.0 / dt, dt
    if meta_json.exists():
        try:
            with open(meta_json, "r", encoding="utf-8") as f:
                meta = json.load(f)
            fs = float(meta.get("fs", np.nan))
            if not np.isnan(fs) and fs > 0:
                return fs, 1.0 / fs
        except (OSError, json.JSONDecodeError, ValueError) as e:
            logging.exception("Failed to read meta json for fs: %s", e)
    raise RuntimeError("Cannot determine sampling frequency: missing tt and meta.fs")


def main() -> None:
    """Add instantaneous frequency columns to a pickled audio DataFrame.

    Processes each audio channel (ch1-ch4) using Hilbert transform to compute
    instantaneous frequency. Applies bandpass filtering before transform to improve
    signal quality. Outputs new columns f_ch1, f_ch2, f_ch3, f_ch4 containing
    frequency estimates in Hz.

    Exit codes:
    - 0: Success
    - 1: Input pickle not found or failed to read
    - 2: Cannot determine sampling frequency (missing tt and meta.fs)
    """
    p = argparse.ArgumentParser(
        description="Add instantaneous frequency to a pickle file."
    )
    p.add_argument("input_file", type=Path, help="Input pickle file.")
    p.add_argument("--lowcut", type=float, default=10.0, help="Low cutoff frequency (Hz)")
    p.add_argument(
        "--highcut", type=float, default=5000.0, help="High cutoff frequency (Hz)"
    )
    p.add_argument("--order", type=int, default=4, help="Butterworth order")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )

    pkl_path = args.input_file
    if not pkl_path.exists():
        logging.error("Pickle not found: %s", pkl_path)
        raise SystemExit(1)

    base = pkl_path.stem
    dirpath = pkl_path.parent
    meta_json = dirpath / (base + "_meta.json")

    df = pd.read_pickle(pkl_path)
    df.columns = [c.lower() for c in df.columns]

    fs, dt = determine_fs_and_dt(df, meta_json)
    logging.info("Using sampling frequency: %.6f Hz (dt=%.9f s)", fs, dt)

    df_with_freq = add_instantaneous_frequency(
        df, fs, lowcut=args.lowcut, highcut=args.highcut, order=args.order
    )

    out_pkl = dirpath / (base + "_with_freq.pkl")
    out_csv = dirpath / (base + "_with_freq_channels.csv")
    df_with_freq.to_pickle(out_pkl)
    cols = ["tt", "ch1", "ch2", "ch3", "ch4", "f_ch1", "f_ch2", "f_ch3", "f_ch4"]
    available = [c for c in cols if c in df_with_freq.columns]
    df_with_freq[available].to_csv(out_csv, index=False, float_format="%.6f")
    print(f"Saving updated pickle to {out_pkl}")


if __name__ == "__main__":
    main()
