"""Process .bin audio files (single file or directory) and run full pipeline.

For each .bin file found, creates an output folder named
<basename>_outputs and runs the following steps:
    - read_audio_board_file.py  (creates .pkl, _meta.json, _waveform.png)
    - dump_channels_to_csv.py  (creates _channels.csv)
    - plot_per_channel.py      (creates _waveform_per_channel.png)
    - add_instantaneous_frequency.py
        (creates _with_freq.pkl/_with_freq_channels.csv)
    - compute_spectrogram.py
        (creates per-channel spectrogram PNGs and _spectrogram.npz)

Usage:
  python process_bin_files.py <file_or_directory>

This script calls the existing scripts using the same Python interpreter.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# Directory where these helper scripts live (same folder as this script)
SCRIPT_DIR = Path(__file__).resolve().parent

SCRIPTS = {
    "convert": "read_audio_board_file.py",
    "csv": "dump_channels_to_csv.py",
    "plot": "plot_per_channel.py",
    "freq": "add_instantaneous_frequency.py",
    "spec": "compute_spectrogram.py",
}


def find_bin_files(path: Path):
    """Return a sorted list of .bin files.

    - If given a .bin file, return it.
    - If given a directory, search it recursively for .bin files in any
      subdirectory as well.
    """
    if path.is_file() and path.suffix.lower() == ".bin":
        return [path]
    if path.is_dir():
        # Recursive, case-insensitive search for all .bin files under dir
        # Use suffix check to match .BIN, .Bin, etc.
        return sorted(
            p
            for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() == ".bin"
        )
    return []


def run_script(script, args, cwd=None):
    script_path = SCRIPT_DIR / script
    cmd = [sys.executable, str(script_path)] + args
    logging.info("Running: %s", " ".join(cmd))
    # Capture stdout/stderr and returncode
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        text=True,
    )
    out_lines = []
    # proc.stdout is Optional[IO[str]]; assert for type-checkers
    assert proc.stdout is not None
    for line in proc.stdout:
        out_lines.append(line)
        logging.info(line.rstrip())
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, output="".join(out_lines)
        )
    return "".join(out_lines)


def process_file(bin_path: Path):
    base = bin_path.stem
    outdir = bin_path.parent / (base + "_outputs")
    outdir.mkdir(exist_ok=True)

    # Set up per-run log file
    log_file = outdir / (base + ".log")
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logging.getLogger().addHandler(fh)
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Starting processing of %s", bin_path)

    # 1) conversion: write outputs into outdir (use --out flag)
    run_script(SCRIPTS["convert"], [str(bin_path), "--out", str(outdir)])

    # The converter writes <base>.pkl into outdir
    pkl_path = outdir / (base + ".pkl")
    if not pkl_path.exists():
        # Some older runs used .mat; check for alternatives
        possible = list(outdir.glob(base + "*pkl"))
        if possible:
            pkl_path = possible[0]

    # 2) CSV dump
    run_script(SCRIPTS["csv"], [str(pkl_path)])

    # 3) per-channel plot
    run_script(SCRIPTS["plot"], [str(pkl_path)])

    # 4) instantaneous frequency (bandpass inside script)
    run_script(SCRIPTS["freq"], [str(pkl_path)])

    # 5) spectrograms
    run_script(SCRIPTS["spec"], [str(pkl_path)])

    logging.info(
        "Finished processing %s; outputs in %s", bin_path.name, outdir
    )
    # Remove handler to avoid duplicate logs in subsequent runs
    logging.getLogger().removeHandler(fh)
    fh.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "path",
        help=(
            ".bin file or directory (searched recursively) "
            "containing .bin files"
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N files (0 = all)",
    )
    args = p.parse_args()

    path = Path(args.path)
    bins = find_bin_files(path)
    if not bins:
        logging.error("No .bin files found at %s", path)
        return

    if args.limit > 0:
        bins = bins[: args.limit]

    for b in bins:
        try:
            process_file(b)
        except subprocess.CalledProcessError as e:
            logging.exception("Error processing %s: %s", b, e)
        except Exception:  # pylint: disable=broad-except
            # Keep batch running even if one file fails
            logging.exception("Unexpected error processing %s", b)


if __name__ == "__main__":
    main()
