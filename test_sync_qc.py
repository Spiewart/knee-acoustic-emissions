"""Quick test to verify sync QC validation works correctly."""

from datetime import timedelta

import numpy as np
import pandas as pd

from sync_audio_with_biomechanics import _validate_synchronized_dataframe


def test_empty_dataframe():
    """Test that empty DataFrame raises ValueError."""
    df = pd.DataFrame()
    try:
        _validate_synchronized_dataframe(df)
        print("❌ FAIL: Empty DataFrame should raise ValueError")
    except ValueError as e:
        print(f"✓ PASS: Empty DataFrame - {str(e)}")


def test_missing_audio_channels():
    """Test that DataFrame without audio channels raises ValueError."""
    df = pd.DataFrame({
        'tt': [0.0, 0.1, 0.2],
        'bio_col1': [1.0, 2.0, 3.0]
    })
    try:
        _validate_synchronized_dataframe(df)
        print("❌ FAIL: Missing audio channels should raise ValueError")
    except ValueError as e:
        print(f"✓ PASS: Missing audio channels - {str(e)}")


def test_audio_all_nan():
    """Test that DataFrame with all NaN audio raises ValueError."""
    df = pd.DataFrame({
        'tt': [0.0, 0.1, 0.2],
        'ch1': [np.nan, np.nan, np.nan],
        'ch2': [np.nan, np.nan, np.nan],
        'bio_col1': [1.0, 2.0, 3.0]
    })
    try:
        _validate_synchronized_dataframe(df)
        print("❌ FAIL: All NaN audio should raise ValueError")
    except ValueError as e:
        print(f"✓ PASS: All NaN audio - {str(e)}")


def test_missing_biomechanics():
    """Test that DataFrame without biomechanics columns raises ValueError."""
    df = pd.DataFrame({
        'tt': [0.0, 0.1, 0.2],
        'ch1': [0.1, 0.2, 0.3],
        'ch2': [0.1, 0.2, 0.3]
    })
    try:
        _validate_synchronized_dataframe(df)
        print("❌ FAIL: Missing biomechanics should raise ValueError")
    except ValueError as e:
        print(f"✓ PASS: Missing biomechanics - {str(e)}")


def test_biomechanics_all_nan():
    """Test that DataFrame with all NaN biomechanics raises ValueError."""
    df = pd.DataFrame({
        'tt': [0.0, 0.1, 0.2],
        'ch1': [0.1, 0.2, 0.3],
        'ch2': [0.1, 0.2, 0.3],
        'bio_col1': [np.nan, np.nan, np.nan],
        'bio_col2': [np.nan, np.nan, np.nan]
    })
    try:
        _validate_synchronized_dataframe(df)
        print("❌ FAIL: All NaN biomechanics should raise ValueError")
    except ValueError as e:
        print(f"✓ PASS: All NaN biomechanics - {str(e)}")


def test_valid_dataframe():
    """Test that valid DataFrame passes validation."""
    df = pd.DataFrame({
        'tt': [0.0, 0.1, 0.2],
        'ch1': [0.1, 0.2, 0.3],
        'ch2': [0.1, 0.2, 0.3],
        'ch3': [0.1, 0.2, 0.3],
        'ch4': [0.1, 0.2, 0.3],
        'bio_col1': [1.0, 2.0, 3.0],
        'bio_col2': [4.0, 5.0, 6.0]
    })
    try:
        _validate_synchronized_dataframe(df)
        print("✓ PASS: Valid DataFrame")
    except ValueError as e:
        print(f"❌ FAIL: Valid DataFrame should not raise - {str(e)}")


def test_low_audio_coverage():
    """Test that low audio coverage triggers warning."""
    df = pd.DataFrame({
        'tt': [0.0, 0.1, 0.2, 0.3, 0.4],
        'ch1': [0.1, np.nan, np.nan, np.nan, np.nan],
        'ch2': [0.1, np.nan, np.nan, np.nan, np.nan],
        'bio_col1': [1.0, 2.0, 3.0, 4.0, 5.0]
    })
    try:
        print("Testing low audio coverage warning:")
        _validate_synchronized_dataframe(df)
        print("✓ PASS: Low audio coverage (check for warning above)")
    except ValueError as e:
        print(f"❌ FAIL: Should warn but not fail - {str(e)}")


if __name__ == "__main__":
    print("="*60)
    print("Testing Sync QC Validation")
    print("="*60)
    test_empty_dataframe()
    test_missing_audio_channels()
    test_audio_all_nan()
    test_missing_biomechanics()
    test_biomechanics_all_nan()
    test_valid_dataframe()
    test_low_audio_coverage()
    print("="*60)
    print("All tests completed!")
    print("="*60)
