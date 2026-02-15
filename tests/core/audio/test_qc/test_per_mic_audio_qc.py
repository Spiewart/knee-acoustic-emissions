"""Tests for per-microphone audio QC functionality."""

import numpy as np
import pandas as pd
import pytest

from src.audio.raw_qc import (
    detect_artifactual_noise_per_mic,
    detect_signal_dropout_per_mic,
    run_raw_audio_qc_per_mic,
)


def test_detect_signal_dropout_per_mic():
    """Test per-microphone signal dropout detection."""
    # Create test data with dropout in ch1 only
    time_s = np.linspace(0, 10, 10000)
    df = pd.DataFrame(
        {
            "tt": time_s,
            "ch1": np.sin(2 * np.pi * 10 * time_s) + 0.1 * np.random.randn(10000),
            "ch2": np.sin(2 * np.pi * 10 * time_s) + 0.1 * np.random.randn(10000),
            "ch3": np.sin(2 * np.pi * 10 * time_s) + 0.1 * np.random.randn(10000),
            "ch4": np.sin(2 * np.pi * 10 * time_s) + 0.1 * np.random.randn(10000),
        }
    )

    # Add dropout to ch1 between 2-3 seconds
    df.loc[(df["tt"] >= 2.0) & (df["tt"] <= 3.0), "ch1"] = 0.0

    # Run per-mic dropout detection with thresholds appropriate for sine wave data
    # (sine wave centered at 0 with amplitude ~1, unlike the AE sensor data with DC offset at 1.5)
    per_mic_dropout = detect_signal_dropout_per_mic(
        df,
        silence_threshold=0.1,  # For sine wave, 0.1 is appropriate
        flatline_threshold=0.001,  # For sine wave noise, 0.001 variance threshold
    )

    # Check that ch1 has dropout intervals
    assert "ch1" in per_mic_dropout
    assert len(per_mic_dropout["ch1"]) > 0, "ch1 should have dropout intervals"

    # Check that first interval is around 2-3 seconds
    if per_mic_dropout["ch1"]:
        start, end = per_mic_dropout["ch1"][0]
        assert 1.8 < start < 2.5, f"Dropout should start around 2s, got {start}"
        assert 2.7 < end < 3.3, f"Dropout should end around 3s, got {end}"

    # Check that other channels have no or minimal dropout
    # (may have some due to random noise, but should be less than ch1)
    for ch in ["ch2", "ch3", "ch4"]:
        if ch in per_mic_dropout:
            assert len(per_mic_dropout[ch]) < len(per_mic_dropout["ch1"]), (
                f"{ch} should have fewer dropout intervals than ch1"
            )


def test_detect_artifactual_noise_per_mic():
    """Test per-microphone artifact detection."""
    # Create test data with spikes in ch2 only
    time_s = np.linspace(0, 10, 10000)
    df = pd.DataFrame(
        {
            "tt": time_s,
            "ch1": np.sin(2 * np.pi * 10 * time_s) + 0.1 * np.random.randn(10000),
            "ch2": np.sin(2 * np.pi * 10 * time_s) + 0.1 * np.random.randn(10000),
            "ch3": np.sin(2 * np.pi * 10 * time_s) + 0.1 * np.random.randn(10000),
            "ch4": np.sin(2 * np.pi * 10 * time_s) + 0.1 * np.random.randn(10000),
        }
    )

    # Add large spikes to ch2 between 5-6 seconds
    spike_mask = (df["tt"] >= 5.0) & (df["tt"] <= 6.0)
    df.loc[spike_mask, "ch2"] += 10.0  # Add large spikes

    # Run per-mic artifact detection (disable periodic noise detection for this test)
    per_mic_artifacts, artifact_types = detect_artifactual_noise_per_mic(df, detect_periodic_noise=False)

    # Check that artifact is around 5-6 seconds
    if per_mic_artifacts["ch2"]:
        # May detect multiple intervals or one large one, check if any overlap with expected range
        found_expected_artifact = False
        for start, end in per_mic_artifacts["ch2"]:
            if (4.0 < start < 6.0) or (4.0 < end < 6.5):
                found_expected_artifact = True
                break
        assert found_expected_artifact, f"Should find artifact around 5-6s, got intervals: {per_mic_artifacts['ch2']}"

    # Check that artifact types are provided and valid
    assert isinstance(artifact_types, dict)
    assert "ch2" in artifact_types
    for artifact_type in artifact_types["ch2"]:
        assert artifact_type in ["Intermittent", "Continuous"], (
            f"Artifact type should be Intermittent or Continuous, got {artifact_type}"
        )


def test_run_raw_audio_qc_per_mic_combined():
    """Test combined per-mic QC (dropout + artifacts)."""
    # Create test data with different issues in different channels
    time_s = np.linspace(0, 10, 10000)
    df = pd.DataFrame(
        {
            "tt": time_s,
            "ch1": np.sin(2 * np.pi * 10 * time_s) + 0.1 * np.random.randn(10000),
            "ch2": np.sin(2 * np.pi * 10 * time_s) + 0.1 * np.random.randn(10000),
            "ch3": np.sin(2 * np.pi * 10 * time_s) + 0.1 * np.random.randn(10000),
            "ch4": np.sin(2 * np.pi * 10 * time_s) + 0.1 * np.random.randn(10000),
        }
    )

    # Add dropout to ch1
    df.loc[(df["tt"] >= 2.0) & (df["tt"] <= 3.0), "ch1"] = 0.0

    # Add spikes to ch3
    spike_mask = (df["tt"] >= 7.0) & (df["tt"] <= 8.0)
    df.loc[spike_mask, "ch3"] += 10.0

    # Run combined per-mic QC
    per_mic_bad = run_raw_audio_qc_per_mic(df)

    # Check that we got results for all channels
    assert isinstance(per_mic_bad, dict)
    assert len(per_mic_bad) == 4, "Should have results for all 4 channels"

    # Check that ch1 has bad intervals (dropout)
    assert "ch1" in per_mic_bad
    assert len(per_mic_bad["ch1"]) > 0, "ch1 should have bad intervals from dropout"

    # Check that ch3 has bad intervals (artifacts)
    assert "ch3" in per_mic_bad
    assert len(per_mic_bad["ch3"]) > 0, "ch3 should have bad intervals from artifacts"

    # ch2 and ch4 should be mostly clean (may have some random noise detections)
    # but should have significantly fewer issues than ch1 and ch3
    for ch in ["ch2", "ch4"]:
        if ch in per_mic_bad:
            assert len(per_mic_bad[ch]) <= max(len(per_mic_bad["ch1"]), len(per_mic_bad["ch3"])), (
                f"{ch} should have fewer or equal bad intervals than problematic channels"
            )


def test_per_mic_qc_with_missing_channels():
    """Test per-mic QC handles missing channels gracefully."""
    # Create test data with only 2 channels
    time_s = np.linspace(0, 10, 10000)
    df = pd.DataFrame(
        {
            "tt": time_s,
            "ch1": np.sin(2 * np.pi * 10 * time_s) + 0.1 * np.random.randn(10000),
            "ch2": np.sin(2 * np.pi * 10 * time_s) + 0.1 * np.random.randn(10000),
        }
    )

    # Run per-mic QC
    per_mic_bad = run_raw_audio_qc_per_mic(df)

    # Should only have results for available channels
    assert "ch1" in per_mic_bad
    assert "ch2" in per_mic_bad
    assert "ch3" not in per_mic_bad or len(per_mic_bad["ch3"]) == 0
    assert "ch4" not in per_mic_bad or len(per_mic_bad["ch4"]) == 0


def test_per_mic_qc_clean_signal():
    """Test per-mic QC with clean signal (no issues)."""
    # Create clean test data
    time_s = np.linspace(0, 10, 10000)
    df = pd.DataFrame(
        {
            "tt": time_s,
            "ch1": np.sin(2 * np.pi * 10 * time_s) + 0.1 * np.random.randn(10000),
            "ch2": np.sin(2 * np.pi * 10 * time_s) + 0.1 * np.random.randn(10000),
            "ch3": np.sin(2 * np.pi * 10 * time_s) + 0.1 * np.random.randn(10000),
            "ch4": np.sin(2 * np.pi * 10 * time_s) + 0.1 * np.random.randn(10000),
        }
    )

    # Run per-mic QC
    per_mic_bad = run_raw_audio_qc_per_mic(df)

    # All channels should be mostly clean (empty or very few intervals)
    for ch in ["ch1", "ch2", "ch3", "ch4"]:
        if ch in per_mic_bad:
            # Allow for occasional false positives from random noise,
            # but should be minimal
            assert len(per_mic_bad[ch]) <= 2, (
                f"{ch} should have at most 2 bad intervals for clean signal, got {len(per_mic_bad[ch])}"
            )


def test_per_mic_qc_all_channels_bad():
    """Test per-mic QC when all channels have issues."""
    # Create test data with issues in all channels
    time_s = np.linspace(0, 10, 10000)
    df = pd.DataFrame(
        {
            "tt": time_s,
            "ch1": np.sin(2 * np.pi * 10 * time_s) + 0.1 * np.random.randn(10000),
            "ch2": np.sin(2 * np.pi * 10 * time_s) + 0.1 * np.random.randn(10000),
            "ch3": np.sin(2 * np.pi * 10 * time_s) + 0.1 * np.random.randn(10000),
            "ch4": np.sin(2 * np.pi * 10 * time_s) + 0.1 * np.random.randn(10000),
        }
    )

    # Add dropout to all channels at different times
    df.loc[(df["tt"] >= 2.0) & (df["tt"] <= 3.0), "ch1"] = 0.0
    df.loc[(df["tt"] >= 3.5) & (df["tt"] <= 4.5), "ch2"] = 0.0
    df.loc[(df["tt"] >= 5.0) & (df["tt"] <= 6.0), "ch3"] = 0.0
    df.loc[(df["tt"] >= 6.5) & (df["tt"] <= 7.5), "ch4"] = 0.0

    # Run per-mic QC
    per_mic_bad = run_raw_audio_qc_per_mic(df)

    # All channels should have bad intervals
    for ch in ["ch1", "ch2", "ch3", "ch4"]:
        assert ch in per_mic_bad, f"{ch} should be in results"
        assert len(per_mic_bad[ch]) > 0, f"{ch} should have bad intervals"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
