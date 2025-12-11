"""Visualization functions for acoustic emissions and biomechanics data."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


def plot_syncd_data(
    syncd_data_path: str,
    joint_angle_col: str = "Knee Angle Z",
    figsize: tuple[int, int] = (14, 8),
    save_path: str | None = None,
) -> Figure:
    """Plot synchronized audio and biomechanics data as time series.

    Creates a dual-axis plot with:
    - Audio channels (ch1-ch4) on the left y-axis
    - Normalized left and right knee angles on the right y-axis
    - Time on the x-axis

    Args:
        syncd_data_path: Path to pickled synchronized DataFrame.
        joint_angle_col: Base name for joint angle columns. Will look for
            "Left {joint_angle_col}" and "Right {joint_angle_col}".
            Default: "Knee Angle Z"
        figsize: Figure size as (width, height) in inches.
        save_path: Optional path to save the figure. If None, figure is
            displayed but not saved.

    Returns:
        matplotlib Figure object.

    Raises:
        FileNotFoundError: If syncd_data_path does not exist.
        ValueError: If required columns are missing from the DataFrame.
    """
    # Load synchronized data
    syncd_path = Path(syncd_data_path)
    if not syncd_path.exists():
        raise FileNotFoundError(
            f"Synchronized data file not found: {syncd_data_path}"
        )

    syncd_df = pd.read_pickle(syncd_path)

    # Validate audio channels
    audio_channels = ["ch1", "ch2", "ch3", "ch4"]
    missing_channels = [
        ch for ch in audio_channels if ch not in syncd_df.columns
    ]
    if missing_channels:
        raise ValueError(
            f"Missing audio channels in DataFrame: {missing_channels}"
        )

    # Auto-detect joint angle column if not provided
    left_angle_col = f"Left {joint_angle_col}"
    right_angle_col = f"Right {joint_angle_col}"

    # Check if both columns exist
    missing_cols = []
    if left_angle_col not in syncd_df.columns:
        missing_cols.append(left_angle_col)
    if right_angle_col not in syncd_df.columns:
        missing_cols.append(right_angle_col)

    if missing_cols:
        raise ValueError(
            f"Joint angle columns not found: {missing_cols}. "
            f"Available columns: {list(syncd_df.columns)}"
        )

    # Determine time column ('tt' or 'TIME')
    time_col = "tt" if "tt" in syncd_df.columns else "TIME"
    if time_col not in syncd_df.columns:
        raise ValueError(
            "No time column found. Expected 'tt' or 'TIME' in DataFrame."
        )

    # Convert time to seconds if it's timedelta
    if pd.api.types.is_timedelta64_dtype(syncd_df[time_col]):
        time_data = syncd_df[time_col].dt.total_seconds()
        time_label = "Time (seconds)"
    else:
        time_data = syncd_df[time_col]
        time_label = "Time"

    # Normalize joint angles to [0, 1] range
    # Combine both angles to get overall min/max for consistent scaling
    left_angle_data = syncd_df[left_angle_col].dropna()
    right_angle_data = syncd_df[right_angle_col].dropna()
    all_angle_data = pd.concat([left_angle_data, right_angle_data])

    if len(all_angle_data) > 0:
        angle_min = all_angle_data.min()
        angle_max = all_angle_data.max()
        angle_range = angle_max - angle_min
        if angle_range > 0:
            normalized_left = (
                syncd_df[left_angle_col] - angle_min
            ) / angle_range
            normalized_right = (
                syncd_df[right_angle_col] - angle_min
            ) / angle_range
        else:
            normalized_left = syncd_df[left_angle_col]
            normalized_right = syncd_df[right_angle_col]
    else:
        normalized_left = syncd_df[left_angle_col]
        normalized_right = syncd_df[right_angle_col]

    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=figsize)

    # Plot audio channels on left y-axis
    # Blue, orange, green, red
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for channel, color in zip(audio_channels, colors):
        ax1.plot(
            time_data,
            syncd_df[channel],
            label=channel.upper(),
            color=color,
            alpha=0.7,
            linewidth=0.8,
        )

    ax1.set_xlabel(time_label, fontsize=12)
    ax1.set_ylabel("Audio Amplitude", fontsize=12, color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Create second y-axis for normalized joint angles
    ax2 = ax1.twinx()
    ax2.plot(
        time_data,
        normalized_left,
        label=f"Left {joint_angle_col} (normalized)",
        color="purple",
        linewidth=1.5,
        linestyle="--",
        alpha=0.8,
    )
    ax2.plot(
        time_data,
        normalized_right,
        label=f"Right {joint_angle_col} (normalized)",
        color="magenta",
        linewidth=1.5,
        linestyle="-.",
        alpha=0.8,
    )

    ax2.set_ylabel("Normalized Joint Angle", fontsize=12, color="purple")
    ax2.tick_params(axis="y", labelcolor="purple")
    ax2.legend(loc="upper right", fontsize=10)
    ax2.set_ylim(-0.1, 1.1)  # Slightly expand to show full range

    # Set title
    filename = syncd_path.stem
    plt.title(
        f"Synchronized Audio & Biomechanics Data\n{filename}",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    # Save or display
    if save_path:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()

    return fig


if __name__ == "__main__":
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
