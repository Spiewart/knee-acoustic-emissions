"""Export utilities for audio DataFrames.

Provides functions to export time (`tt`) and audio channels (`ch1..ch4`)
from a pickled DataFrame to a CSV file. If `tt` is missing, attempts to
reconstruct timestamps from a sibling `_meta.json` file, falling back to
integer sample indices.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def export_channels_to_csv(pkl_path: str | Path) -> Path:
    """Export time and channel columns from a pickle to CSV.

    Args:
        pkl_path: Path to pickled pandas DataFrame containing audio data.

    Returns:
        Path to the written CSV file (`<stem>_channels.csv`).

    Raises:
        FileNotFoundError: If the pickle path does not exist.
        ValueError: If the pickle cannot be read.
    """
    pkl = Path(pkl_path)
    if not pkl.exists():
        raise FileNotFoundError(f"Pickle file not found: {pkl}")

    base = pkl.stem
    csv_out = pkl.with_name(base + "_channels.csv")
    meta_json = pkl.with_name(base + "_meta.json")

    try:
        df = pd.read_pickle(pkl)
    except (OSError, EOFError, ValueError) as exc:
        raise ValueError(f"Failed to read pickle: {exc}")

    # Normalize column names to lower-case
    df.columns = [str(c).lower() for c in df.columns]

    # Determine timestamps
    ts: Optional[pd.Series]
    if "tt" in df.columns:
        ts = pd.to_numeric(df["tt"], errors="coerce")
    else:
        ts = None
        if meta_json.exists():
            try:
                with open(meta_json, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                st = meta.get("startTime", None)
                sp = meta.get("stopTime", None)
                if st is not None and sp is not None:
                    chcol = next((c for c in ["ch1", "ch2", "ch3", "ch4"] if c in df.columns), None)
                    n = int(len(df[chcol])) if chcol is not None else int(len(df))
                    try:
                        ts = pd.Series(np.linspace(float(st), float(sp), n))
                    except (TypeError, ValueError):
                        ts = pd.Series(np.arange(len(df)))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                ts = pd.Series(np.arange(len(df)))
        if ts is None:
            ts = pd.Series(np.arange(len(df)))

    out_df = pd.DataFrame({"tt": ts})
    for c in ["ch1", "ch2", "ch3", "ch4"]:
        out_df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan

    out_df.to_csv(csv_out, index=False, float_format="%.6f")
    return csv_out


def dump_channels_to_csv(df: pd.DataFrame, csv_out: Path, meta_json: Optional[Path] = None):
    """Export time and channel columns from a DataFrame to CSV."""
    df.columns = [c.lower() for c in df.columns]

    # Determine timestamps
    if "tt" in df.columns:
        ts = df["tt"].astype(float)
    else:
        ts = None
        if meta_json and meta_json.exists():
            try:
                with open(meta_json, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                st = meta.get("startTime", None)
                sp = meta.get("stopTime", None)
                if st is not None and sp is not None:
                    # find first available channel to get length
                    chcol = next(
                        (
                            c
                            for c in ["ch1", "ch2", "ch3", "ch4"]
                            if c in df.columns
                        ),
                        None,
                    )
                    if chcol is not None:
                        n = len(df[chcol])
                        try:
                            ts = pd.Series(np.linspace(float(st), float(sp), n))
                        except (TypeError, ValueError) as e:
                            logging.debug(
                                "Failed to create linspace from meta start/stop: %s", e
                            )
                            ts = pd.Series(np.arange(len(df)))
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
                logging.debug("Failed to read/parse meta json: %s", e)
                ts = pd.Series(np.arange(len(df)))
        if ts is None:
            ts = pd.Series(np.arange(len(df)))

    out_df = pd.DataFrame({"tt": ts})
    for c in ["ch1", "ch2", "ch3", "ch4"]:
        out_df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan

    out_df.to_csv(csv_out, index=False, float_format="%.6f")
    logging.info("Wrote CSV: %s", csv_out)
    logging.info("Rows,Cols: %s", out_df.shape)
    logging.info("\nHead:\n%s", out_df.head().to_string(index=False))
    logging.info("\nDtypes:\n%s", out_df.dtypes)
    for c in ["ch1", "ch2", "ch3", "ch4"]:
        if out_df[c].notna().any():
            logging.info(
                "%s: min=%f, max=%f, mean=%f",
                c,
                out_df[c].min(),
                out_df[c].max(),
                out_df[c].mean(),
            )
        else:
            logging.info("%s: no data", c)


if __name__ == "__main__":
    # This block is for direct script execution, which is now handled by CLI.
    # The `export_channels_to_csv` function remains for library use.
    pass
