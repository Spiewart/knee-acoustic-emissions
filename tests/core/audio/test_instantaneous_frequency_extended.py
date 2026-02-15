"""Extended tests for instantaneous frequency: positive paths and edge cases."""

import numpy as np
import pandas as pd

from src.audio.instantaneous_frequency import add_instantaneous_frequency


def test_add_instantaneous_frequency_single_tone():
    """Test inst. freq. on a single sine tone; should recover the tone frequency."""
    fs = 10000.0
    duration = 2.0
    freq = 100.0  # 100 Hz sine
    N = int(fs * duration)
    t = np.arange(N) / fs
    signal = np.sin(2 * np.pi * freq * t)

    df = pd.DataFrame({"ch1": signal})
    out = add_instantaneous_frequency(df.copy(), fs=fs, lowcut=50.0, highcut=200.0, channels=["ch1"])

    # Instantaneous frequency should cluster around 100 Hz (with some transient effects)
    f_ch1 = out["f_ch1"].to_numpy()
    # Ignore first/last 10% due to filter transients
    mid_idx = slice(int(0.1 * N), int(0.9 * N))
    mid_freq = f_ch1[mid_idx]
    mid_freq_finite = mid_freq[np.isfinite(mid_freq)]
    assert len(mid_freq_finite) > 0
    assert np.abs(np.median(mid_freq_finite) - freq) < 5.0  # Within 5 Hz


def test_add_instantaneous_frequency_mixed_channels():
    """Test processing a DataFrame with multiple channels present and absent."""
    fs = 5000.0
    N = 512
    t = np.arange(N) / fs

    df = pd.DataFrame(
        {
            "ch1": np.sin(2 * np.pi * 50.0 * t),  # Present
            "ch2": [np.nan] * N,  # Present but all NaN
            "ch3": np.cos(2 * np.pi * 75.0 * t),  # Present
            # ch4 absent
        }
    )

    out = add_instantaneous_frequency(df.copy(), fs=fs, channels=None)  # Auto-detect

    assert "f_ch1" in out.columns
    assert "f_ch2" in out.columns
    assert "f_ch3" in out.columns
    assert "f_ch4" not in out.columns

    # ch2 should be all NaN (empty input)
    assert np.isnan(out["f_ch2"]).all()

    # ch1 and ch3 should have data
    assert out["f_ch1"].notna().any()
    assert out["f_ch3"].notna().any()


def test_add_instantaneous_frequency_partial_nan_channel():
    """Test channel with some NaN: Hilbert transform handles gracefully."""
    # When NaN appears in input, the Hilbert transform will produce all NaN.
    # This is expected behavior; test that the function doesn't crash.
    fs = 8000.0
    N = 256
    t = np.arange(N) / fs
    signal = np.sin(2 * np.pi * 80.0 * t)

    # Add some NaN in the middle
    mixed = signal.copy()
    mixed[100:120] = np.nan

    df = pd.DataFrame({"ch1": mixed})
    out = add_instantaneous_frequency(df.copy(), fs=fs, lowcut=20.0, highcut=2000.0, channels=["ch1"])

    assert "f_ch1" in out.columns
    # Hilbert fails on NaN, so the whole output is NaN
    assert np.isnan(out["f_ch1"]).all()
