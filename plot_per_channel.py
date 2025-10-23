"""Plot per-channel subplots from the pickled DataFrame and save PNG.

Usage: python plot_per_channel.py <pkl_path>
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl", help="Path to pickled DataFrame")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    pkl_path = Path(args.pkl)
    if not pkl_path.exists():
        logging.error("Pickle not found: %s", pkl_path)
        raise SystemExit(1)

    base = pkl_path.stem
    out_png = pkl_path.with_name(base + "_waveform_per_channel.png")

    df = pd.read_pickle(pkl_path)
    df.columns = [c.lower() for c in df.columns]

    if "tt" in df.columns and df["tt"].notna().any():
        tt = df["tt"].astype(float).to_numpy()
    else:
        tt = np.arange(len(df))

    channels = ["ch1", "ch2", "ch3", "ch4"]

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 8))
    for i, ch in enumerate(channels):
        ax = axes[i]
        if ch in df.columns and df[ch].notna().any():
            y = pd.to_numeric(df[ch], errors="coerce").to_numpy()
            low, high = np.nanpercentile(y, [1, 99])
            if np.isfinite(low) and np.isfinite(high) and high > low:
                pad = 0.05 * (high - low)
                ax.set_ylim(low - pad, high + pad)
            else:
                ymin = np.nanmin(y)
                ymax = np.nanmax(y)
                ax.set_ylim(ymin, ymax)
            ax.plot(tt[: len(y)], y, lw=0.5)
            ax.set_ylabel(ch)
        else:
            ax.text(0.5, 0.5, f"No data for {ch}", ha="center", va="center")
            ax.set_ylabel(ch)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    logging.info("Saved per-channel waveform: %s", out_png)


if __name__ == "__main__":
    main()
