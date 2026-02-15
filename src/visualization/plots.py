"""Visualization functions for acoustic emissions and biomechanics data."""

import argparse
from pathlib import Path

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def plot_syncd_data(
    syncd_data_path: str,
    joint_angle_col: str = "Knee Angle Z",
    figsize: tuple[int, int] = (14, 8),
    save_path: str | None = None,
) -> Figure:
    """Plot synchronized audio and biomechanics data as time series.

    Creates a dual-axis plot with:
    - Audio channels (ch1-ch4) on the left y-axis
    - Normalized joint angle on the right y-axis
    - Time on the x-axis

    Note: Assumes biomechanics columns no longer have laterality prefixes
    (e.g., "Knee Angle Z" instead of "Left Knee Angle Z").

    Args:
        syncd_data_path: Path to pickled synchronized DataFrame.
        joint_angle_col: Name for joint angle column.
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
        raise FileNotFoundError(f"Synchronized data file not found: {syncd_data_path}")

    syncd_df = pd.read_pickle(syncd_path)

    # Validate audio channels
    audio_channels = ["ch1", "ch2", "ch3", "ch4"]
    missing_channels = [ch for ch in audio_channels if ch not in syncd_df.columns]
    if missing_channels:
        raise ValueError(f"Missing audio channels in DataFrame: {missing_channels}")

    # Check if joint angle column exists (no laterality prefix)
    if joint_angle_col not in syncd_df.columns:
        raise ValueError(
            f"Joint angle column '{joint_angle_col}' not found. Available columns: {list(syncd_df.columns)}"
        )

    # Determine time column ('tt' or 'TIME')
    time_col = "tt" if "tt" in syncd_df.columns else "TIME"
    if time_col not in syncd_df.columns:
        raise ValueError("No time column found. Expected 'tt' or 'TIME' in DataFrame.")

    # Convert time to seconds if it's timedelta
    if pd.api.types.is_timedelta64_dtype(syncd_df[time_col]):
        time_data = syncd_df[time_col].dt.total_seconds()
        time_label = "Time (seconds)"
    else:
        time_data = syncd_df[time_col]
        time_label = "Time"

    # Normalize joint angle to [0, 1] range and convert to numeric
    # Handle potential object dtype by converting to numeric first
    angle_series = pd.to_numeric(syncd_df[joint_angle_col], errors="coerce")
    angle_data = angle_series.dropna()

    if len(angle_data) > 0:
        angle_min = angle_data.min()
        angle_max = angle_data.max()
        angle_range = angle_max - angle_min
        if angle_range > 0:
            # Normalize to [0, 1] then scale to 50% of range [0.25, 0.75]
            normalized_angle = (angle_series - angle_min) / angle_range
            normalized_angle = 0.25 + (normalized_angle * 0.5)

            # Apply Savitzky-Goyal smoothing directly to sparse data
            # Biomechanics data is sparse (390-780 sample gaps at 52 kHz).
            # Smooth the sparse points directly without interpolation.
            valid_mask = ~normalized_angle.isna()
            valid_indices = valid_mask[valid_mask].index

            if len(valid_indices) > 5:
                # Extract only the valid (sparse) data points
                valid_data = normalized_angle[valid_indices]

                # Small window for joint motion (~2 Hz at 120 Hz = 60 pts)
                # Base window on number of valid points, not indices
                window_length = min(21, max(len(valid_indices) // 10, 5))
                if window_length % 2 == 0:
                    window_length += 1

                # Apply filter directly to valid data values
                smoothed_valid = savgol_filter(valid_data.values, window_length=window_length, polyorder=3)

                # Replace values with smoothed version at sparse locations
                normalized_angle[valid_indices] = smoothed_valid
        else:
            normalized_angle = angle_series * 0 + 0.5  # Constant at midpoint
    else:
        normalized_angle = angle_series

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

    # Create second y-axis for normalized joint angle
    ax2 = ax1.twinx()

    # Plot as smooth continuous waveform
    # Filter to only plot valid (non-NaN) data points
    valid_mask = ~normalized_angle.isna()
    valid_time = time_data[valid_mask]
    valid_angle = normalized_angle[valid_mask]

    ax2.plot(
        valid_time,
        valid_angle,
        label=f"{joint_angle_col} (normalized)",
        color="purple",
        linewidth=1.5,
        alpha=0.9,
        linestyle="-",
    )

    ax2.set_ylabel(f"{joint_angle_col} (degrees)", fontsize=12, color="purple")
    ax2.tick_params(axis="y", labelcolor="purple")

    # Create custom y-axis labels showing actual angle values
    if len(angle_data) > 0:
        angle_min = angle_data.min()
        angle_max = angle_data.max()
        # Show 5 tick marks with actual angle values
        tick_positions = [0.25, 0.375, 0.5, 0.625, 0.75]
        tick_labels = [f"{angle_min + (pos - 0.25) / 0.5 * (angle_max - angle_min):.1f}" for pos in tick_positions]
        ax2.set_yticks(tick_positions)
        ax2.set_yticklabels(tick_labels)

    ax2.legend(loc="upper right", fontsize=10)
    ax2.set_ylim(0, 1)  # Full range, with angle occupying middle 50%

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


def plot_per_channel(df: pd.DataFrame, out_png: Path) -> None:
    """Plot per-channel subplots from a DataFrame and save to PNG."""
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot synchronized audio and biomechanics data")
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
