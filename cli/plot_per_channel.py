"""CLI for plotting per-channel waveforms from a pickle file."""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.visualization.plots import plot_per_channel


def main():
    """CLI for plotting per-channel waveforms from a pickle file."""
    parser = argparse.ArgumentParser(
        description="Plot per-channel waveforms from a pickle file."
    )
    parser.add_argument("input_file", type=Path, help="Input pickle file.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )

    pkl_path = args.input_file
    if not pkl_path.exists():
        logging.error("Pickle not found: %s", pkl_path)
        raise SystemExit(1)

    base = pkl_path.stem
    out_png = pkl_path.with_name(base + "_waveform_per_channel.png")

    df = pd.read_pickle(pkl_path)

    plot_per_channel(df, out_png)
    print(f"Saved per-channel waveform: {out_png}")


if __name__ == "__main__":
    main()
