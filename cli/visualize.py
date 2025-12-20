#!/usr/bin/env python3
"""
Command-line interface for visualization utilities.

This script plots synchronized audio and biomechanics data.
"""

import argparse

from src.visualization.plots import plot_syncd_data


def main() -> None:
    """Main entry point for visualization CLI."""
    parser = argparse.ArgumentParser(
        description="Plot synchronized audio and biomechanics data"
    )
    parser.add_argument(
        "syncd_data_path",
        type=str,
        help="Path to pickled synchronized DataFrame",
    )
    parser.add_argument(
        "--joint-angle-col",
        type=str,
        default="Knee Angle Z",
        help="Base name for joint angle columns (default: Knee Angle Z)",
    )
    parser.add_argument(
        "--figsize",
        type=int,
        nargs=2,
        default=[14, 8],
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches (default: 14 8)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save the figure (if not provided, displays only)",
    )

    args = parser.parse_args()

    plot_syncd_data(
        syncd_data_path=args.syncd_data_path,
        joint_angle_col=args.joint_angle_col,
        figsize=tuple(args.figsize),
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
