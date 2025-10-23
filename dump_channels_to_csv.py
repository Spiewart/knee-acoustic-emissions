"""Export time and channels ch1..ch4 from a pickled DataFrame to CSV.

The script prefers a `tt` column for timestamps. If missing it will try
to reconstruct timestamps from the corresponding `_meta.json` file
(`startTime`/`stopTime`) and fall back to integer sample indices.

Usage: python dump_channels_to_csv.py <pkl_path>
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pkl", help="Path to pickled DataFrame")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

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

    # Normalize column names
    df.columns = [c.lower() for c in df.columns]

    # Determine timestamps
    if "tt" in df.columns:
        ts = df["tt"].astype(float)
    else:
        ts = None
        if meta_json.exists():
            try:
                with open(meta_json, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                st = meta.get("startTime", None)
                sp = meta.get("stopTime", None)
                if st is not None and sp is not None:
                    # find first available channel to get length
                    chcol = next((c for c in ["ch1", "ch2", "ch3", "ch4"] if c in df.columns), None)
                    if chcol is not None:
                        n = len(df[chcol])
                        try:
                            ts = pd.Series(np.linspace(float(st), float(sp), n))
                        except (TypeError, ValueError) as e:
                            logging.debug("Failed to create linspace from meta start/stop: %s", e)
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
            logging.info("%s: min=%f, max=%f, mean=%f", c, out_df[c].min(), out_df[c].max(), out_df[c].mean())
        else:
            logging.info("%s: no data", c)


if __name__ == "__main__":
    main()
