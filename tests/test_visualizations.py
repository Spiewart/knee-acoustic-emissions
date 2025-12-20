"""Tests for visualization functions."""

from unittest.mock import patch

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.visualization.plots import plot_syncd_data


def test_plot_syncd_data_basic(syncd_data):
    """Test basic plotting with valid synchronized DataFrame."""
    pkl_path, syncd_df = syncd_data

    # Plot with save_path
    output_path = pkl_path.parent / "output.png"
    fig = plot_syncd_data(
        str(pkl_path),
        joint_angle_col="Knee Angle Z",
        save_path=str(output_path),
    )

    assert fig is not None
    assert output_path.exists()
    plt.close(fig)


def test_plot_syncd_data_display_only(syncd_data):
    """Test plotting without saving (display only)."""
    pkl_path, syncd_df = syncd_data

    # Mock plt.show to avoid actually displaying
    with patch("matplotlib.pyplot.show"):
        fig = plot_syncd_data(
            str(pkl_path),
            joint_angle_col="Knee Angle Z",
            save_path=None,
        )

    assert fig is not None
    plt.close(fig)


def test_plot_syncd_data_custom_figsize(syncd_data):
    """Test plotting with custom figure size."""
    pkl_path, syncd_df = syncd_data

    custom_size = (12, 6)
    with patch("matplotlib.pyplot.show"):
        fig = plot_syncd_data(
            str(pkl_path),
            joint_angle_col="Knee Angle Z",
            figsize=custom_size,
        )

    assert fig.get_size_inches().tolist() == list(custom_size)
    plt.close(fig)


def test_plot_syncd_data_with_timedelta_time(tmp_path):
    """Test plotting with timedelta time column."""
    # Create DataFrame with timedelta time column
    syncd_df = pd.DataFrame(
        {
            "TIME": pd.to_timedelta([0, 100, 200, 300, 400], unit="ms"),
            "ch1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "ch2": [2.0, 3.0, 4.0, 5.0, 6.0],
            "ch3": [3.0, 4.0, 5.0, 6.0, 7.0],
            "ch4": [4.0, 5.0, 6.0, 7.0, 8.0],
            "Knee Angle Z": [10.0, 12.0, 14.0, 16.0, 18.0],
        }
    )

    pkl_path = tmp_path / "syncd_timedelta.pkl"
    syncd_df.to_pickle(pkl_path)

    with patch("matplotlib.pyplot.show"):
        fig = plot_syncd_data(str(pkl_path))

    assert fig is not None
    plt.close(fig)


def test_plot_syncd_data_with_nan_values(tmp_path):
    """Test plotting with NaN values in joint angle column."""
    syncd_df = pd.DataFrame(
        {
            "tt": [0.0, 0.1, 0.2, 0.3, 0.4],
            "ch1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "ch2": [2.0, 3.0, 4.0, 5.0, 6.0],
            "ch3": [3.0, 4.0, 5.0, 6.0, 7.0],
            "ch4": [4.0, 5.0, 6.0, 7.0, 8.0],
            "Knee Angle Z": [10.0, float("nan"), 14.0, float("nan"), 18.0],
        }
    )

    pkl_path = tmp_path / "syncd_nan.pkl"
    syncd_df.to_pickle(pkl_path)

    with patch("matplotlib.pyplot.show"):
        fig = plot_syncd_data(str(pkl_path))

    assert fig is not None
    plt.close(fig)


def test_plot_syncd_data_alternative_joint_angle(syncd_data, tmp_path):
    """Test plotting with different joint angle column."""
    # Create DataFrame with different angle column
    pkl_path, syncd_df = syncd_data
    syncd_df["Knee Angle X"] = syncd_df["Knee Angle Z"] * 0.5

    new_pkl_path = tmp_path / "syncd_alt_angle.pkl"
    syncd_df.to_pickle(new_pkl_path)

    with patch("matplotlib.pyplot.show"):
        fig = plot_syncd_data(
            str(new_pkl_path),
            joint_angle_col="Knee Angle X",
        )

    assert fig is not None
    plt.close(fig)


def test_plot_syncd_data_file_not_found():
    """Test error handling when file does not exist."""
    with pytest.raises(FileNotFoundError):
        plot_syncd_data("/nonexistent/path.pkl")


def test_plot_syncd_data_missing_audio_channels(syncd_data, tmp_path):
    """Test error handling when audio channels are missing."""
    # Create DataFrame without all audio channels
    pkl_path, syncd_df = syncd_data
    syncd_df_missing = syncd_df.drop(columns=["ch4"])

    new_pkl_path = tmp_path / "syncd_missing_ch.pkl"
    syncd_df_missing.to_pickle(new_pkl_path)

    with pytest.raises(ValueError, match="Missing audio channels"):
        plot_syncd_data(str(new_pkl_path))


def test_plot_syncd_data_missing_joint_angle_column(syncd_data):
    """Test error handling when joint angle column is missing."""
    pkl_path, syncd_df = syncd_data

    with pytest.raises(ValueError, match="Joint angle column"):
        plot_syncd_data(str(pkl_path), joint_angle_col="Nonexistent Column")


def test_plot_syncd_data_missing_time_column(syncd_data, tmp_path):
    """Test error handling when time column is missing."""
    # Create DataFrame without time columns
    pkl_path, syncd_df = syncd_data
    syncd_df_missing = syncd_df.drop(columns=["tt"])

    new_pkl_path = tmp_path / "syncd_missing_time.pkl"
    syncd_df_missing.to_pickle(new_pkl_path)

    with pytest.raises(ValueError, match="No time column found"):
        plot_syncd_data(str(new_pkl_path))


def test_plot_syncd_data_normalization(syncd_data):
    """Test that normalization works correctly with fixture data."""
    pkl_path, syncd_df = syncd_data

    with patch("matplotlib.pyplot.show"):
        fig = plot_syncd_data(
            str(pkl_path),
            joint_angle_col="Knee Angle Z",
        )

    # The function should normalize the angle data
    # Just verify it completes without error
    assert fig is not None
    plt.close(fig)


def test_plot_syncd_data_constant_joint_angle(tmp_path):
    """Test plotting when joint angle has constant value (no range)."""
    # Create DataFrame with constant joint angle
    syncd_df = pd.DataFrame(
        {
            "tt": [0.0, 0.1, 0.2, 0.3, 0.4],
            "ch1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "ch2": [2.0, 3.0, 4.0, 5.0, 6.0],
            "ch3": [3.0, 4.0, 5.0, 6.0, 7.0],
            "ch4": [4.0, 5.0, 6.0, 7.0, 8.0],
            "Knee Angle Z": [10.0, 10.0, 10.0, 10.0, 10.0],
        }
    )

    pkl_path = tmp_path / "syncd_constant.pkl"
    syncd_df.to_pickle(pkl_path)

    with patch("matplotlib.pyplot.show"):
        fig = plot_syncd_data(str(pkl_path))

    # Should handle constant values gracefully
    assert fig is not None
    plt.close(fig)


def test_plot_syncd_data_empty_dataframe(tmp_path):
    """Test error handling with empty DataFrame."""
    df = pd.DataFrame(
        {
            "tt": [],
            "ch1": [],
            "ch2": [],
            "ch3": [],
            "ch4": [],
            "Knee Angle Z": [],
        }
    )
    pkl_path = tmp_path / "syncd_empty.pkl"
    df.to_pickle(pkl_path)

    # Should handle empty DataFrame (audio channels present)
    # but joint angle will have no values to normalize
    with patch("matplotlib.pyplot.show"):
        fig = plot_syncd_data(str(pkl_path))

    assert fig is not None
    plt.close(fig)


def test_plot_syncd_data_output_directory_creation(syncd_data, tmp_path):
    """Test that output directory is created if it doesn't exist."""
    pkl_path, syncd_df = syncd_data

    # Use nested directory that doesn't exist yet
    output_path = tmp_path / "output" / "subdir" / "plot.png"

    fig = plot_syncd_data(
        str(pkl_path),
        save_path=str(output_path),
    )

    assert output_path.exists()
    assert output_path.parent.exists()
    plt.close(fig)
