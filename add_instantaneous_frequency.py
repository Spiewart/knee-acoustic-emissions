"""Add instantaneous frequency columns (f_ch1..f_ch4) to a dataframe pickle.

Method: use analytic signal (Hilbert transform) and derivative of unwrapped
phase: f_inst = (1 / (2*pi)) * d/dt unwrap(angle(hilbert(x))). A band-pass
filter is applied before Hilbert to stabilise phase estimates. Defaults are
10 Hz to 5000 Hz; these are configurable via CLI flags.

Usage: python add_instantaneous_frequency.py <pkl_path> [--lowcut --highcut]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def apply_bandpass(
    y: np.ndarray, fs: float, lowcut: float = 10.0, highcut: float = 5000.0, order: int = 4
) -> np.ndarray:
    try:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y_filt = filtfilt(b, a, y)
        return y_filt
    except (ValueError, TypeError) as e:
        logging.exception("Bandpass filtering failed; returning raw signal: %s", e)
        return y


def determine_fs_and_dt(df: pd.DataFrame, meta_json: Path) -> Tuple[float, float]:
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pkl", help="Pickled DataFrame with channel columns")
    p.add_argument("--lowcut", type=float, default=10.0, help="Low cutoff frequency (Hz)")
    p.add_argument("--highcut", type=float, default=5000.0, help="High cutoff frequency (Hz)")
    p.add_argument("--order", type=int, default=4, help="Butterworth order")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    pkl_path = Path(args.pkl)
    if not pkl_path.exists():
        logging.error("Pickle not found: %s", pkl_path)
        raise SystemExit(1)

    base = pkl_path.stem
    dirpath = pkl_path.parent
    meta_json = dirpath / (base + "_meta.json")

    logging.info("Loading pickle: %s", pkl_path)
    df = pd.read_pickle(pkl_path)
    df.columns = [c.lower() for c in df.columns]

    fs, dt = determine_fs_and_dt(df, meta_json)
    logging.info("Using sampling frequency: %.6f Hz (dt=%.9f s)", fs, dt)

    channels = ["ch1", "ch2", "ch3", "ch4"]

    for ch in channels:
        col_freq = "f_" + ch
        if ch not in df.columns or not df[ch].notna().any():
            logging.info("%s not present or empty; filling %s with NaN", ch, col_freq)
            df[col_freq] = np.nan
            continue

        y = pd.to_numeric(df[ch], errors="coerce").to_numpy()
        y_centered = y - np.nanmean(y)
        y_filtered = apply_bandpass(y_centered, fs, lowcut=args.lowcut, highcut=args.highcut, order=args.order)

        try:
            analytic = hilbert(y_filtered)
        except (ValueError, TypeError) as e:
            logging.exception("Hilbert transform failed for %s; filling with NaN: %s", ch, e)
            df[col_freq] = np.nan
            continue

        phase = np.unwrap(np.angle(analytic))
        dphase_dt = np.gradient(phase, dt)
        inst_freq = dphase_dt / (2.0 * np.pi)
        df[col_freq] = inst_freq

        finite = inst_freq[np.isfinite(inst_freq)]
        if finite.size > 0:
            logging.info(
                "%s: min=%.3f Hz, max=%.3f Hz, mean=%.3f Hz", col_freq, finite.min(), finite.max(), finite.mean()
            )
        else:
            logging.info("%s: no finite values", col_freq)

    out_pkl = dirpath / (base + "_with_freq.pkl")
    out_csv = dirpath / (base + "_with_freq_channels.csv")
    logging.info("Saving updated pickle to %s", out_pkl)
    df.to_pickle(out_pkl)
    cols = ["tt", "ch1", "ch2", "ch3", "ch4", "f_ch1", "f_ch2", "f_ch3", "f_ch4"]
    available = [c for c in cols if c in df.columns]
    logging.info("Saving CSV to %s", out_csv)
    df[available].to_csv(out_csv, index=False, float_format="%.6f")
    logging.info("Done")


if __name__ == "__main__":
    main()
