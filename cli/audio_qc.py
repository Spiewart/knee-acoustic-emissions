#!/usr/bin/env python3
"""
Command-line interface for audio quality control.

This script provides QC checks for audio recordings of different maneuvers
(walking, sit-to-stand, flexion-extension).
"""

import argparse
from pathlib import Path

from src.audio.quality_control import (
    DEFAULT_CHANNELS,
    _run_qc_on_file,
    qc_audio_directory,
)


def _build_cli_parser() -> argparse.ArgumentParser:
    """Build argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="Audio QC utilities for maneuver-specific checks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--time",
        default="tt",
        help="Time column name in the audio pickle",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=DEFAULT_CHANNELS,
        help="Audio channel columns to average for QC",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    file_parser = subparsers.add_parser(
        "file", help="Run QC on a single audio pickle"
    )
    file_parser.add_argument("pkl", help="Path to audio pickle file")
    file_parser.add_argument(
        "--maneuver",
        required=True,
        choices=["flexion_extension", "sit_to_stand", "walk"],
        help="Maneuver type to QC",
    )
    file_parser.add_argument(
        "--freq",
        type=float,
        default=0.25,
        help="Target frequency (Hz) for flexion-extension or sit-to-stand",
    )
    file_parser.add_argument(
        "--tail",
        type=float,
        default=5.0,
        help="Tail length (s) to exclude for periodic maneuvers",
    )
    file_parser.add_argument(
        "--bandpower-min-ratio",
        type=float,
        help="Optional minimum bandpower ratio around target/detected freq",
    )
    file_parser.add_argument(
        "--resample-walk",
        type=float,
        default=100.0,
        help="Resample rate (Hz) for walking heel-strike detection",
    )
    file_parser.add_argument(
        "--min-pass-peaks",
        type=int,
        default=6,
        help="Minimum heel strikes required in a walking pass",
    )
    file_parser.add_argument(
        "--min-gap-s",
        type=float,
        default=2.0,
        help="Gap (s) that splits walking passes",
    )
    file_parser.add_argument(
        "--use-frequency-segmentation",
        action="store_true",
        help="Use step frequency clustering instead of gap-based segmentation for walking",
    )
    file_parser.add_argument(
        "--frequency-tolerance-frac",
        type=float,
        default=0.06,
        help="Fractional tolerance for grouping similar frequencies (0.06 = 6%%)",
    )

    dir_parser = subparsers.add_parser(
        "dir", help="Run QC across a participant directory"
    )
    dir_parser.add_argument(
        "participant_dir",
        help="Path to participant directory (contains Left/Right Knee)",
    )
    dir_parser.add_argument(
        "--maneuver",
        choices=["walk", "sit_to_stand", "flexion_extension", "all"],
        default="all",
        help="Restrict QC to a single maneuver or run all",
    )
    dir_parser.add_argument(
        "--freq",
        type=float,
        default=0.25,
        help="Target frequency (Hz) for non-walk maneuvers",
    )
    dir_parser.add_argument(
        "--tail",
        type=float,
        default=5.0,
        help="Tail length (s) to exclude for non-walk maneuvers",
    )
    dir_parser.add_argument(
        "--bandpower-min-ratio",
        type=float,
        help="Optional minimum bandpower ratio around target/detected freq",
    )
    dir_parser.add_argument(
        "--resample-walk",
        type=float,
        default=100.0,
        help="Resample rate (Hz) for walking heel-strike detection",
    )
    dir_parser.add_argument(
        "--min-pass-peaks",
        type=int,
        default=6,
        help="Minimum heel strikes required in a walking pass",
    )
    dir_parser.add_argument(
        "--min-gap-s",
        type=float,
        default=2.0,
        help="Gap (s) that splits walking passes",
    )
    dir_parser.add_argument(
        "--use-frequency-segmentation",
        action="store_true",
        help="Use step frequency clustering instead of gap-based segmentation for walking",
    )
    dir_parser.add_argument(
        "--frequency-tolerance-frac",
        type=float,
        default=0.06,
        help="Fractional tolerance for grouping similar frequencies (0.06 = 6%%)",
    )

    return parser


def _print_file_result(result: dict[str, object]) -> None:
    """Print QC results for a single file."""
    from src.audio.quality_control import _group_passes_by_speed

    maneuver = result["maneuver"]
    print(f"QC result for {maneuver} on {result['path']}")
    if maneuver == "walk":
        passes = result.get("passes", [])
        has_numbering = any(("speed" in p) or ("pass_num" in p) for p in passes)
        if passes and not has_numbering:
            grouped = _group_passes_by_speed(passes)
            if not grouped:
                print("  No walking passes detected")
            else:
                print("  Grouped by speed (no pass numbers):")
                for g in grouped:
                    print(
                        f"  {g['speed_label']}: freq={g['gait_cycle_hz']:.2f} Hz, "
                        f"coverage={g['coverage_frac']:.2%}, "
                        f"bandpower={g['bandpower_ratio']:.3f}, "
                        f"passed={g['passed']} (passes={g['pass_count']}, pass_rate={g['pass_rate']:.0%})"
                    )
        else:
            for idx, p in enumerate(passes, start=1):
                print(
                    f"  Pass {idx}: freq={p['gait_cycle_hz']:.2f} Hz, "
                    f"coverage={p['coverage_frac']:.2%}, "
                    f"bandpower={p['bandpower_ratio']:.3f}, "
                    f"passed={p['passed']}"
                )
            if not passes:
                print("  No walking passes detected")
    else:
        print(
            f"  passed={result['passed']}, "
            f"coverage={result['coverage']:.2%}"
        )


def _print_dir_results(results: list[dict[str, object]]) -> None:
    """Print QC results for all files in a directory."""
    from src.audio.quality_control import _group_passes_by_speed

    if not results:
        print("No QC results found.")
        return
    for res in results:
        header = (
            f"{res['knee']} | {res['maneuver']} | {res['path']}"
        )
        print(header)
        if res["maneuver"] == "walk":
            passes = res.get("passes", [])
            has_numbering = any(("speed" in p) or ("pass_num" in p) for p in passes)
            if passes and not has_numbering:
                grouped = _group_passes_by_speed(passes)
                if not grouped:
                    print("  No walking passes detected")
                    continue
                print("  Grouped by speed (no pass numbers):")
                for g in grouped:
                    print(
                        f"  {g['speed_label']}: freq={g['gait_cycle_hz']:.2f} Hz, "
                        f"coverage={g['coverage_frac']:.2%}, "
                        f"bandpower={g['bandpower_ratio']:.3f}, "
                        f"passed={g['passed']} (passes={g['pass_count']}, pass_rate={g['pass_rate']:.0%})"
                    )
                continue
            if not passes:
                print("  No walking passes detected")
                continue
            for idx, p in enumerate(passes, start=1):
                print(
                    f"  Pass {idx}: freq={p['gait_cycle_hz']:.2f} Hz, "
                    f"coverage={p['coverage_frac']:.2%}, "
                    f"bandpower={p['bandpower_ratio']:.3f}, "
                    f"passed={p['passed']}"
                )
        else:
            print(
                f"  passed={res['passed']}, "
                f"coverage={res['coverage']:.2%}"
            )


def main() -> None:
    """Main entry point for audio QC CLI."""
    parser = _build_cli_parser()
    args = parser.parse_args()

    channels = list(args.channels)
    bandpower_min_ratio = args.bandpower_min_ratio

    if args.command == "file":
        pkl_path = Path(args.pkl)
        if not pkl_path.exists():
            print(f"Error: File does not exist: {pkl_path}")
            return
        result = _run_qc_on_file(
            pkl_path=pkl_path,
            maneuver=args.maneuver,
            time_col=args.time,
            audio_channels=channels,
            target_freq_hz=args.freq,
            tail_length_s=args.tail,
            bandpower_min_ratio=bandpower_min_ratio,
            resample_hz_walk=args.resample_walk,
            min_pass_peaks=args.min_pass_peaks,
            min_gap_s=args.min_gap_s,
            use_frequency_segmentation=args.use_frequency_segmentation,
            frequency_tolerance_frac=args.frequency_tolerance_frac,
        )
        _print_file_result(result)
    else:
        participant_dir = Path(args.participant_dir)
        if not participant_dir.exists():
            print(f"Error: Directory does not exist: {participant_dir}")
            return
        results = qc_audio_directory(
            participant_dir=participant_dir,
            time_col=args.time,
            audio_channels=channels,
            maneuver=args.maneuver,
            target_freq_hz=args.freq,
            tail_length_s=args.tail,
            bandpower_min_ratio=bandpower_min_ratio,
            resample_hz_walk=args.resample_walk,
            min_pass_peaks=args.min_pass_peaks,
            min_gap_s=args.min_gap_s,
            use_frequency_segmentation=args.use_frequency_segmentation,
            frequency_tolerance_frac=args.frequency_tolerance_frac,
        )
        _print_dir_results(results)


def _cli_main() -> None:
    """Compatibility shim for legacy tests expecting _cli_main."""
    main()


if __name__ == "__main__":
    main()
