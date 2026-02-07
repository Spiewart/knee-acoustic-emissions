"""Spectrogram computation utilities for acoustic channels.

Provides functions to compute STFT spectrograms for `ch1..ch4` from a
DataFrame or a pickle file, save per-channel PNGs, and persist arrays
to a compressed NPZ file.

This module mirrors and formalizes the behavior of the legacy
`compute_spectrogram.py` script with explicit type hints, descriptive
docstrings, and focused helpers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import stft


def get_fs_from_df_or_meta(df: pd.DataFrame, meta_json: Optional[Path] = None) -> float:
    """Infer sampling frequency from DataFrame `tt` or a meta JSON.

    Args:
        df: DataFrame potentially containing a `tt` column with seconds.
        meta_json: Optional path to a JSON file with an `fs` field.

    Returns:
        The sampling frequency in Hz.

    Raises:
        RuntimeError: If neither `tt` nor a valid `fs` in the meta JSON is available.
    """
    if "tt" in df.columns and df["tt"].notna().any():
        tt = df["tt"].astype(float).to_numpy()
        dt = np.median(np.diff(tt))
        if dt <= 0 or not np.isfinite(dt):
            raise RuntimeError("Invalid tt column for sampling frequency inference")
        return 1.0 / dt
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


def _save_spectrogram_png(
    f: np.ndarray,
    t: np.ndarray,
    Sxx_db: np.ndarray,
    out_png: Path,
    fmax: Optional[float] = None,
) -> None:
    """Save a spectrogram PNG.

    Args:
        f: Frequencies in Hz.
        t: Times in seconds.
        Sxx_db: Magnitude in dB (shape: [freq, time]).
        out_png: Output PNG path.
        fmax: Optional maximum frequency for display.
    """
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


# Type alias for spectrogram array mapping
SpecsMap = Dict[str, np.ndarray]


def compute_spectrogram_arrays(
    df: pd.DataFrame,
    fs: float,
    nperseg: int,
    noverlap: int,
) -> Tuple[np.ndarray, np.ndarray, SpecsMap]:
    """Compute STFT spectrogram arrays for available channels.

    Args:
        df: Input DataFrame containing `ch1..ch4` (some may be missing or NaN).
        fs: Sampling frequency in Hz.
        nperseg: STFT window length.
        noverlap: STFT overlap between windows.

    Returns:
        A tuple `(f, t, specs)` where `f` and `t` are 1D arrays of frequencies and
        times, and `specs` is a mapping with keys `spec_ch1..spec_ch4` present for
        channels that exist, each an array of shape `[len(f), len(t)]` in dB.
        If no channels are available, returns empty arrays for `f` and `t` and an
        empty mapping for `specs`.
    """
    channels = ["ch1", "ch2", "ch3", "ch4"]
    specs: Dict[str, np.ndarray] = {}
    f_ref: Optional[np.ndarray] = None
    t_ref: Optional[np.ndarray] = None

    for ch in channels:
        if ch not in df.columns or not df[ch].notna().any():
            continue
        y = pd.to_numeric(df[ch], errors="coerce").to_numpy()
        y = y - np.nanmean(y)
        f, t, Zxx = stft(y, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, boundary=None)
        Sxx = np.abs(Zxx)
        eps = 1e-12
        Sxx_db = 20.0 * np.log10(Sxx + eps)
        specs[f"spec_{ch}"] = Sxx_db
        if f_ref is None:
            f_ref = f
            t_ref = t

    if f_ref is None or t_ref is None:
        # No channels available
        return np.array([]), np.array([]), {}

    return f_ref, t_ref, specs


def compute_spectrogram_from_pickle(
    pkl_path: Path,
    nperseg: int = 2048,
    noverlap: int = 1536,
    fmax: float = 5000.0,
) -> Tuple[Path, List[Path]]:
    """Load a DataFrame from a pickle and compute spectrograms.

    Args:
        pkl_path: Path to the pickled DataFrame.
        nperseg: STFT window length (default: 2048).
        noverlap: STFT overlap (default: 1536; 75%).
        fmax: Optional max frequency for PNGs (default: 5000 Hz).

    Returns:
        A tuple `(npz_path, png_paths)` where `npz_path` is the compressed arrays
        output and `png_paths` lists the per-channel spectrogram images saved.

    Raises:
        FileNotFoundError: If `pkl_path` does not exist.
        RuntimeError: If sampling frequency cannot be determined.
    """
    if not pkl_path.exists():
        raise FileNotFoundError(f"Pickle not found: {pkl_path}")

    base = pkl_path.stem
    dirpath = pkl_path.parent

    logging.info("Loading pickle: %s", pkl_path)
    df = pd.read_pickle(pkl_path)

    fs = get_fs_from_df_or_meta(df)
    logging.info("Using sampling frequency: %.3f Hz", fs)
    f, t, specs = compute_spectrogram_arrays(df, fs=fs, nperseg=nperseg, noverlap=noverlap)

    png_paths: List[Path] = []
    for key, Sxx_db in specs.items():
        ch = key.replace("spec_", "")
        png_out = dirpath / f"{base}_spec_{ch}.png"
        _save_spectrogram_png(f, t, Sxx_db, png_out, fmax=fmax)
        png_paths.append(png_out)

    npz_out = dirpath / f"{base}_spectrogram.npz"
    # Persist as float32 to reduce size
    save_dict = {"f": f.astype(np.float32), "t": t.astype(np.float32)}
    for key, arr in specs.items():
        save_dict[key] = arr.astype(np.float32)
    np.savez_compressed(npz_out, **save_dict)

    return npz_out, png_paths
