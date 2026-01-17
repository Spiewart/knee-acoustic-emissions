"""Tests for biomechanics-guided stomp detection with expected time delta."""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from src.synchronization.sync import get_audio_stomp_time


def create_synthetic_audio_with_two_stomps(
    stomp1_time: float,
    stomp2_time: float,
    stomp1_amplitude: float = 1000.0,
    stomp2_amplitude: float = 800.0,
    duration: float = 20.0,
    sr: float = 52000.0,
) -> pd.DataFrame:
    """Create synthetic audio data with two distinct stomp events.

    Args:
        stomp1_time: Time of first stomp in seconds
        stomp2_time: Time of second stomp in seconds
        stomp1_amplitude: Amplitude of first stomp (louder = recorded knee)
        stomp2_amplitude: Amplitude of second stomp (quieter = opposite knee)
        duration: Total duration of audio in seconds
        sr: Sampling rate in Hz

    Returns:
        DataFrame with tt and ch1-ch4 columns
    """
    n_samples = int(duration * sr)
    tt = np.linspace(0, duration, n_samples)

    # Create base noise
    audio = np.random.randn(n_samples, 4) * 10

    # Add first stomp (Gaussian pulse)
    stomp1_width = 0.05  # 50ms wide
    stomp1_pulse = stomp1_amplitude * np.exp(-((tt - stomp1_time) ** 2) / (2 * stomp1_width ** 2))

    # Add second stomp
    stomp2_width = 0.05
    stomp2_pulse = stomp2_amplitude * np.exp(-((tt - stomp2_time) ** 2) / (2 * stomp2_width ** 2))

    # Add stomps to all channels
    for ch in range(4):
        audio[:, ch] += stomp1_pulse + stomp2_pulse

    return pd.DataFrame({
        'tt': tt,
        'ch1': audio[:, 0],
        'ch2': audio[:, 1],
        'ch3': audio[:, 2],
        'ch4': audio[:, 3],
    })


def test_biomech_guided_detection_right_then_left():
    """Test detection when right stomp occurs before left stomp."""
    # Create audio with right stomp at 5s, left stomp at 8s
    right_time = 5.0
    left_time = 8.0
    expected_delta = left_time - right_time  # 3.0 seconds

    audio_df = create_synthetic_audio_with_two_stomps(
        stomp1_time=right_time,
        stomp2_time=left_time,
        stomp1_amplitude=1000.0,  # Right is louder (recorded knee)
        stomp2_amplitude=700.0,   # Left is quieter
    )

    # Test detection for right knee recording
    detected_time, details = get_audio_stomp_time(
        audio_df,
        recorded_knee="right",
        right_stomp_time=timedelta(seconds=right_time),
        left_stomp_time=timedelta(seconds=left_time),
        return_details=True,
    )

    detected_seconds = detected_time.total_seconds()

    # Should detect right stomp (first one)
    assert abs(detected_seconds - right_time) < 0.3, \
        f"Expected ~{right_time}s, got {detected_seconds}s"

    # Verify details are returned
    assert 'consensus_time' in details
    assert 'rms_time' in details


def test_biomech_guided_detection_left_then_right():
    """Test detection when left stomp occurs before right stomp."""
    # Create audio with left stomp at 4s, right stomp at 7s
    left_time = 4.0
    right_time = 7.0
    expected_delta = right_time - left_time  # 3.0 seconds

    audio_df = create_synthetic_audio_with_two_stomps(
        stomp1_time=left_time,
        stomp2_time=right_time,
        stomp1_amplitude=1000.0,  # Left is louder (recorded knee)
        stomp2_amplitude=700.0,   # Right is quieter
    )

    # Test detection for left knee recording
    detected_time, details = get_audio_stomp_time(
        audio_df,
        recorded_knee="left",
        right_stomp_time=timedelta(seconds=right_time),
        left_stomp_time=timedelta(seconds=left_time),
        return_details=True,
    )

    detected_seconds = detected_time.total_seconds()

    # Should detect left stomp (first one)
    assert abs(detected_seconds - left_time) < 0.3, \
        f"Expected ~{left_time}s, got {detected_seconds}s"


def test_biomech_guided_detection_close_stomps():
    """Test detection when stomps are very close together (< 1 second)."""
    # Create audio with stomps 0.8 seconds apart
    stomp1_time = 6.0
    stomp2_time = 6.8
    expected_delta = 0.8

    audio_df = create_synthetic_audio_with_two_stomps(
        stomp1_time=stomp1_time,
        stomp2_time=stomp2_time,
        stomp1_amplitude=1000.0,
        stomp2_amplitude=800.0,
    )

    # Test detection for right knee (first stomp)
    detected_time, details = get_audio_stomp_time(
        audio_df,
        recorded_knee="right",
        right_stomp_time=timedelta(seconds=stomp1_time),
        left_stomp_time=timedelta(seconds=stomp2_time),
        return_details=True,
    )

    detected_seconds = detected_time.total_seconds()

    # Should still find the pair and select the right one
    assert abs(detected_seconds - stomp1_time) < 0.3, \
        f"Expected ~{stomp1_time}s, got {detected_seconds}s"


def test_biomech_guided_detection_far_apart_stomps():
    """Test detection when stomps are far apart (> 5 seconds)."""
    # Create audio with stomps 7 seconds apart
    stomp1_time = 3.0
    stomp2_time = 10.0
    expected_delta = 7.0

    audio_df = create_synthetic_audio_with_two_stomps(
        stomp1_time=stomp1_time,
        stomp2_time=stomp2_time,
        stomp1_amplitude=1000.0,
        stomp2_amplitude=700.0,
    )

    # Test detection for right knee (first stomp)
    detected_time, details = get_audio_stomp_time(
        audio_df,
        recorded_knee="right",
        right_stomp_time=timedelta(seconds=stomp1_time),
        left_stomp_time=timedelta(seconds=stomp2_time),
        return_details=True,
    )

    detected_seconds = detected_time.total_seconds()

    # Should find the pair even when far apart
    assert abs(detected_seconds - stomp1_time) < 0.5, \
        f"Expected ~{stomp1_time}s, got {detected_seconds}s"


def test_biomech_guided_detection_with_extra_peaks():
    """Test detection when there are more than 2 peaks (noise/artifacts)."""
    # Create audio with 2 main stomps plus noise peaks
    right_time = 5.0
    left_time = 8.5

    audio_df = create_synthetic_audio_with_two_stomps(
        stomp1_time=right_time,
        stomp2_time=left_time,
        stomp1_amplitude=1000.0,
        stomp2_amplitude=750.0,
    )

    # Add some smaller noise peaks
    n_samples = len(audio_df)
    noise_peak_times = [2.0, 12.0, 15.0]
    for noise_time in noise_peak_times:
        noise_idx = int(noise_time * 52000)
        if 0 <= noise_idx < n_samples:
            # Add small peaks (below 30% threshold hopefully)
            audio_df.loc[noise_idx:noise_idx+100, ['ch1', 'ch2', 'ch3', 'ch4']] += 200

    # Test detection for right knee
    detected_time, details = get_audio_stomp_time(
        audio_df,
        recorded_knee="right",
        right_stomp_time=timedelta(seconds=right_time),
        left_stomp_time=timedelta(seconds=left_time),
        return_details=True,
    )

    detected_seconds = detected_time.total_seconds()

    # Should still find the correct pair based on expected delta
    assert abs(detected_seconds - right_time) < 0.4, \
        f"Expected ~{right_time}s, got {detected_seconds}s"


def test_biomech_guided_fallback_to_consensus():
    """Test fallback to consensus when no valid peak pair found."""
    # Create audio with only one clear stomp
    single_stomp_time = 6.0

    audio_df = create_synthetic_audio_with_two_stomps(
        stomp1_time=single_stomp_time,
        stomp2_time=single_stomp_time,  # Same time = only one peak
        stomp1_amplitude=1500.0,
        stomp2_amplitude=0.0,  # No second stomp
    )

    # Test detection with biomechanics expecting two stomps
    detected_time, details = get_audio_stomp_time(
        audio_df,
        recorded_knee="right",
        right_stomp_time=timedelta(seconds=5.0),
        left_stomp_time=timedelta(seconds=10.0),  # Expects 5s separation
        return_details=True,
    )

    detected_seconds = detected_time.total_seconds()
    consensus_time = details['consensus_time']

    # Should fall back to consensus (which should be near the single stomp)
    assert abs(detected_seconds - single_stomp_time) < 1.0, \
        f"Expected fallback near {single_stomp_time}s, got {detected_seconds}s"

    # Verify consensus was used as fallback
    assert abs(detected_seconds - consensus_time) < 0.1, \
        "Should have fallen back to consensus time"


def test_biomech_guided_energy_prioritization():
    """Test that higher energy peak pair is selected when multiple pairs exist."""
    # Create audio with multiple potential pairs at different energies
    # Main pair: high energy peaks at 5s and 8s (delta = 3s)
    # Decoy pair: low energy peaks at 6s and 9s (delta = 3s)

    audio_df = create_synthetic_audio_with_two_stomps(
        stomp1_time=5.0,
        stomp2_time=8.0,
        stomp1_amplitude=1200.0,  # High energy
        stomp2_amplitude=900.0,   # High energy
    )

    # Add decoy pair with same delta but lower energy
    n_samples = len(audio_df)
    tt = audio_df['tt'].values
    for decoy_time, decoy_amp in [(6.0, 400.0), (9.0, 400.0)]:
        decoy_pulse = decoy_amp * np.exp(-((tt - decoy_time) ** 2) / (2 * 0.05 ** 2))
        audio_df[['ch1', 'ch2', 'ch3', 'ch4']] += decoy_pulse[:, np.newaxis]

    # Test detection
    detected_time, details = get_audio_stomp_time(
        audio_df,
        recorded_knee="right",
        right_stomp_time=timedelta(seconds=5.0),
        left_stomp_time=timedelta(seconds=8.0),
        return_details=True,
    )

    detected_seconds = detected_time.total_seconds()

    # Should select the higher energy pair (5s and 8s), not the decoys
    assert abs(detected_seconds - 5.0) < 0.4, \
        f"Expected high-energy peak at 5.0s, got {detected_seconds}s"


def test_backward_compatibility_without_biomechanics():
    """Test that function still works without biomechanics guidance."""
    # Create audio with single stomp
    stomp_time = 6.0
    audio_df = create_synthetic_audio_with_two_stomps(
        stomp1_time=stomp_time,
        stomp2_time=stomp_time,
        stomp1_amplitude=1000.0,
        stomp2_amplitude=0.0,
    )

    # Call without biomechanics parameters
    detected_time = get_audio_stomp_time(audio_df)

    detected_seconds = detected_time.total_seconds()

    # Should still detect the stomp via consensus
    assert abs(detected_seconds - stomp_time) < 0.5, \
        f"Expected ~{stomp_time}s, got {detected_seconds}s"

    # Test with return_details but no biomechanics
    detected_time, details = get_audio_stomp_time(audio_df, return_details=True)

    assert 'consensus_time' in details
    assert 'rms_time' in details
    assert 'onset_time' in details
    assert 'freq_time' in details


def test_biomech_guided_tolerances():
    """Test that tolerances work correctly (0.20s then 0.30s)."""
    # Create stomps exactly 3.25s apart
    stomp1_time = 5.0
    stomp2_time = 8.25
    actual_delta = 3.25

    audio_df = create_synthetic_audio_with_two_stomps(
        stomp1_time=stomp1_time,
        stomp2_time=stomp2_time,
        stomp1_amplitude=1000.0,
        stomp2_amplitude=800.0,
    )

    # Provide biomechanics with expected delta = 3.0s
    # Actual delta (3.25) is within 0.30s tolerance but not 0.20s
    detected_time, details = get_audio_stomp_time(
        audio_df,
        recorded_knee="right",
        right_stomp_time=timedelta(seconds=5.0),
        left_stomp_time=timedelta(seconds=8.0),  # Expected delta = 3.0s
        return_details=True,
    )

    detected_seconds = detected_time.total_seconds()

    # Should still find the pair using the relaxed tolerance
    assert abs(detected_seconds - stomp1_time) < 0.4, \
        f"Expected ~{stomp1_time}s with relaxed tolerance, got {detected_seconds}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
