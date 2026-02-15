"""Tests for synchronized DataFrame validation."""

import numpy as np
import pandas as pd
import pytest

from src.synchronization.sync import _validate_synchronized_dataframe


def test_empty_dataframe_raises_error():
    """Empty DataFrame should raise ValueError."""
    df = pd.DataFrame()
    with pytest.raises(ValueError, match=r"empty|Empty"):
        _validate_synchronized_dataframe(df)


def test_missing_audio_channels_raises_error():
    """DataFrame without audio channels should raise ValueError."""
    df = pd.DataFrame({"tt": [0.0, 0.1, 0.2], "bio_col1": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match=r"audio|ch"):
        _validate_synchronized_dataframe(df)


def test_audio_all_nan_raises_error():
    """DataFrame with all NaN audio should raise ValueError."""
    df = pd.DataFrame(
        {
            "tt": [0.0, 0.1, 0.2],
            "ch1": [np.nan, np.nan, np.nan],
            "ch2": [np.nan, np.nan, np.nan],
            "bio_col1": [1.0, 2.0, 3.0],
        }
    )
    with pytest.raises(ValueError, match=r"audio|valid"):
        _validate_synchronized_dataframe(df)


def test_missing_biomechanics_raises_error():
    """DataFrame without biomechanics columns should raise ValueError."""
    df = pd.DataFrame({"tt": [0.0, 0.1, 0.2], "ch1": [0.1, 0.2, 0.3], "ch2": [0.1, 0.2, 0.3]})
    with pytest.raises(ValueError, match=r"biomechanics|bio"):
        _validate_synchronized_dataframe(df)


def test_biomechanics_all_nan_raises_error():
    """DataFrame with all NaN biomechanics should raise ValueError."""
    df = pd.DataFrame(
        {
            "tt": [0.0, 0.1, 0.2],
            "ch1": [0.1, 0.2, 0.3],
            "ch2": [0.1, 0.2, 0.3],
            "bio_col1": [np.nan, np.nan, np.nan],
            "bio_col2": [np.nan, np.nan, np.nan],
        }
    )
    with pytest.raises(ValueError, match=r"biomechanics|valid"):
        _validate_synchronized_dataframe(df)


def test_valid_dataframe_passes():
    """Valid DataFrame should pass validation without errors."""
    df = pd.DataFrame(
        {
            "tt": [0.0, 0.1, 0.2],
            "ch1": [0.1, 0.2, 0.3],
            "ch2": [0.1, 0.2, 0.3],
            "ch3": [0.1, 0.2, 0.3],
            "ch4": [0.1, 0.2, 0.3],
            "bio_col1": [1.0, 2.0, 3.0],
            "bio_col2": [4.0, 5.0, 6.0],
        }
    )
    # Should not raise any exception
    _validate_synchronized_dataframe(df)


def test_low_audio_coverage_warns(capsys):
    """Low audio coverage should trigger a warning but not fail."""
    df = pd.DataFrame(
        {
            "tt": [0.0, 0.1, 0.2, 0.3, 0.4],
            "ch1": [0.1, np.nan, np.nan, np.nan, np.nan],
            "ch2": [0.1, np.nan, np.nan, np.nan, np.nan],
            "bio_col1": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    # Should not raise, but should print warning
    _validate_synchronized_dataframe(df)

    # Check that a warning was printed about coverage
    captured = capsys.readouterr()
    assert "coverage" in captured.out.lower() or "audio" in captured.out.lower()
    assert "20.0%" in captured.out or "1/5" in captured.out


def test_partial_audio_channels_passes():
    """DataFrame with only some audio channels should still pass."""
    df = pd.DataFrame(
        {
            "tt": [0.0, 0.1, 0.2],
            "ch1": [0.1, 0.2, 0.3],
            "ch2": [0.1, 0.2, 0.3],
            # Missing ch3 and ch4 is ok
            "Knee Angle Z": [45.0, 50.0, 55.0],
        }
    )
    # Should not raise
    _validate_synchronized_dataframe(df)


def test_mixed_nan_values_passes():
    """DataFrame with some NaN values (but not all) should pass."""
    df = pd.DataFrame(
        {
            "tt": [0.0, 0.1, 0.2, 0.3],
            "ch1": [0.1, 0.2, np.nan, 0.4],
            "ch2": [0.1, np.nan, 0.3, 0.4],
            "ch3": [0.1, 0.2, 0.3, 0.4],
            "Knee Angle Z": [45.0, 50.0, 55.0, 60.0],
        }
    )
    # Should not raise - some data is valid
    _validate_synchronized_dataframe(df)
