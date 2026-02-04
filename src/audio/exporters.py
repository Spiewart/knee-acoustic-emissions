"""Export utilities for audio DataFrames.

Provides functions to export time (`tt`) and audio channels (`ch1..ch4`)
from a pickled DataFrame to a CSV file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

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

    if ts is None:
        ts = pd.Series(np.arange(len(df)))

    out_df = pd.DataFrame({"tt": ts})
    for c in ["ch1", "ch2", "ch3", "ch4"]:
        out_df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan

    out_df.to_csv(csv_out, index=False, float_format="%.6f")
    return csv_out


def _load_meta_json(meta_json: Optional[Path | str]) -> Optional[dict[str, Any]]:
    if not meta_json:
        return None

    meta_path = Path(meta_json)
    if not meta_path.exists():
        return None

    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logging.warning("Failed to read meta JSON %s: %s", meta_path, exc)
        return None


def dump_channels_to_csv(
    df: pd.DataFrame,
    csv_out: Path,
    meta_json: Optional[Path | str] = None,
) -> None:
    """Export time and channel columns from a DataFrame to CSV.

    Args:
        df: DataFrame with optional `tt` and any of `ch1..ch4` columns.
        csv_out: Destination CSV path.

    Returns:
        None. Writes `csv_out` with columns `tt, ch1..ch4` (missing channels filled with NaN).
    """
    df.columns = [c.lower() for c in df.columns]

    # Determine timestamps
    if "tt" in df.columns:
        ts = pd.to_numeric(df["tt"], errors="coerce")
    else:
        ts = None

    if ts is None:
        meta = _load_meta_json(meta_json)
        ts_values: Optional[np.ndarray] = None

        if meta:
            try:
                if "startTime" in meta and "stopTime" in meta:
                    start = float(meta["startTime"])
                    stop = float(meta["stopTime"])
                    if len(df) <= 1:
                        ts_values = np.array([start] if len(df) == 1 else [])
                    else:
                        ts_values = np.linspace(start, stop, len(df))
                elif "fs" in meta:
                    fs = float(meta["fs"])
                    if fs > 0:
                        start = float(meta.get("startTime", 0.0))
                        ts_values = start + (np.arange(len(df)) / fs)
            except (TypeError, ValueError):
                ts_values = None

        if ts_values is None:
            ts_values = np.arange(len(df))

        ts = pd.Series(ts_values)

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
