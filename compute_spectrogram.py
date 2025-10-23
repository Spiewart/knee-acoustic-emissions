"""Compute STFT spectrograms for channels in a pickled DataFrame.

Saves per-channel PNG spectrograms and a combined .npz containing
frequencies, times and magnitude spectrograms.

Usage:
  python compute_spectrogram.py <pkl_path> [--nperseg N] [--noverlap M] [--fmax F]

Defaults: nperseg=2048, noverlap=1536 (75%), fmax=5000 Hz
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import stft


def get_fs_from_df_or_meta(df: pd.DataFrame, meta_json: Path | None = None) -> float:
    # Try tt
    if "tt" in df.columns and df["tt"].notna().any():
        tt = df["tt"].astype(float).to_numpy()
        dt = np.median(np.diff(tt))
        return 1.0 / dt
    # Try meta
    if meta_json and meta_json.exists():
        try:
            with open(meta_json, "r", encoding="utf-8") as f:
                meta = json.load(f)
            fs = float(meta.get("fs", np.nan))
            if not np.isnan(fs):
                return fs
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
            logging.exception("Failed to read/parse meta json: %s", e)
    raise RuntimeError("Cannot determine sampling frequency (no tt column or valid meta fs)")


def save_spectrogram_png(
    f: np.ndarray, t: np.ndarray, Sxx_db: np.ndarray, out_png: Path, fmax: float | None = None
) -> None:
    plt.figure(figsize=(10, 4))
    if fmax is not None:
        mask = f <= fmax
        im = plt.pcolormesh(t, f[mask], Sxx_db[mask, :], shading="auto", cmap="magma")
        plt.ylim(0, fmax)
    else:
        im = plt.pcolormesh(t, f, Sxx_db, shading="auto", cmap="magma")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.colorbar(im, label="Magnitude (dB)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("pkl")
    p.add_argument("--nperseg", type=int, default=2048)
    p.add_argument("--noverlap", type=int, default=1536)
    p.add_argument("--fmax", type=float, default=5000.0)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    pkl_path = Path(args.pkl)
    if not pkl_path.exists():
        logging.error("Pickle not found: %s", pkl_path)
        raise SystemExit(1)

    base = pkl_path.stem
    out_dir = pkl_path.parent
    meta_json = out_dir / (base + "_meta.json")

    logging.info("Loading: %s", pkl_path)
    df = pd.read_pickle(pkl_path)
    df.columns = [c.lower() for c in df.columns]

    fs = get_fs_from_df_or_meta(df, meta_json)
    logging.info("Using sampling frequency: %.6f Hz", fs)

    channels = ["ch1", "ch2", "ch3", "ch4"]
    nperseg = args.nperseg
    noverlap = args.noverlap
    fmax = args.fmax

    spec_dict: dict[str, np.ndarray | None] = {"f": None, "t": None}

    for ch in channels:
        if ch not in df.columns or not df[ch].notna().any():
            logging.info("Skipping %s (missing)", ch)
            continue
        y = pd.to_numeric(df[ch], errors="coerce").to_numpy()
        y = y - np.nanmean(y)

        logging.info("Computing STFT for %s (nperseg=%d, noverlap=%d)", ch, nperseg, noverlap)
        f, t, Zxx = stft(y, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, boundary=None)
        Sxx = np.abs(Zxx)
        eps = 1e-12
        Sxx_db = 20.0 * np.log10(Sxx + eps)

        png_out = out_dir / f"{base}_spec_{ch}.png"
        save_spectrogram_png(f, t, Sxx_db, png_out, fmax=fmax)
        logging.info("Saved %s", png_out)

        if spec_dict["f"] is None:
            spec_dict["f"] = f
            spec_dict["t"] = t
        spec_dict[f"spec_{ch}"] = Sxx_db

    npz_out = out_dir / f"{base}_spectrogram.npz"
    logging.info("Saving spectrogram arrays to %s", npz_out)
    np.savez_compressed(
        npz_out,
        **{
            k: (v.astype(np.float32) if isinstance(v, np.ndarray) else v) for k, v in spec_dict.items() if v is not None
        },
    )

    logging.info("Done")


if __name__ == "__main__":
    main()
