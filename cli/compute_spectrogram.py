"""CLI for computing spectrograms from a pickled DataFrame.

Wraps `src.audio.spectrogram.compute_spectrogram_from_pickle`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.audio.spectrogram import compute_spectrogram_from_pickle


def main() -> int:
    """Compute STFT spectrograms from a pickled audio DataFrame.

    Loads a pickled DataFrame containing audio data and computes Short-Time
    Fourier Transform (STFT) spectrograms for each channel (ch1-ch4).
    Saves per-channel PNG images and compressed NPZ archive of arrays.

    Returns:
        0 on success, 1 if processing fails (missing file, sampling frequency error, etc.).
    """
    p = argparse.ArgumentParser(description="Compute STFT spectrograms for acoustic channels")
    p.add_argument("pkl", help="Path to pickled DataFrame")
    p.add_argument("--nperseg", type=int, default=2048, help="STFT window length")
    p.add_argument("--noverlap", type=int, default=1536, help="STFT overlap")
    p.add_argument("--fmax", type=float, default=5000.0, help="Max frequency for PNG display")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    try:
        npz_path, png_paths = compute_spectrogram_from_pickle(
            pkl_path=Path(args.pkl), nperseg=args.nperseg, noverlap=args.noverlap, fmax=args.fmax
        )
        print(f"Saved spectrogram arrays: {npz_path}")
        for pth in png_paths:
            print(f"Saved PNG: {pth}")
        return 0
    except FileNotFoundError as e:
        logging.error(str(e))
        return 1
    except Exception as e:  # noqa: BLE001 - report unexpected errors
        logging.exception("Failed to compute spectrogram: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
