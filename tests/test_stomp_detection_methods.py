"""Test stomp detection methods with synthetic audio."""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from src.synchronization.sync import (
    _detect_stomp_by_frequency_content,
    _detect_stomp_by_impact_onset,
    _detect_stomp_by_rms_energy,
    get_audio_stomp_time,
)


def create_synthetic_stomp_audio(sr=10000, duration=3.0, stomp_time=1.5):
    """Create synthetic audio with a stomp event at a specific time.

    Args:
        sr: Sampling rate in Hz (reduced to 10kHz for fast testing)
        duration: Total duration in seconds (reduced for fast testing)
        stomp_time: Time of the stomp event in seconds

    Returns:
        DataFrame with synthetic audio data
    """
    n_samples = int(sr * duration)
    t = np.arange(n_samples) / sr

    # Start with quiet background noise
    audio = np.random.normal(0, 0.01, (n_samples, 4))

    # Add a stomp event: sudden burst of high-energy vibration around stomp_time
    # Contact mic stomp typically has broadband energy
    stomp_start_idx = int(stomp_time * sr)
    stomp_duration_idx = int(0.15 * sr)  # 150ms stomp duration

    # Gaussian envelope for stomp (realistic onset and decay)
    stomp_envelope = np.exp(-((t - stomp_time)**2) / (2 * 0.075**2))

    # Add multiple frequency components (100-1000 Hz range for contact mics)
    for ch in range(4):
        # Fundamental frequency around 200 Hz with harmonics
        freq1 = 200 + ch * 50  # Slightly different per channel
        freq2 = 400 + ch * 100
        freq3 = 800 + ch * 150

        stomp_signal = (
            0.5 * np.sin(2 * np.pi * freq1 * t) +
            0.3 * np.sin(2 * np.pi * freq2 * t) +
            0.2 * np.sin(2 * np.pi * freq3 * t)
        )

        # Apply stomp envelope
        audio[:, ch] += stomp_signal * stomp_envelope * 0.8

    # Create DataFrame
    df = pd.DataFrame({
        'tt': t,
        'ch1': audio[:, 0],
        'ch2': audio[:, 1],
        'ch3': audio[:, 2],
        'ch4': audio[:, 3],
    })

    return df


def test_rms_energy_stomp_detection():
    """Test RMS energy method detects stomp within 200ms."""
    audio_df = create_synthetic_stomp_audio(stomp_time=1.5)
    sr = 10000

    audio_channels = audio_df[['ch1', 'ch2', 'ch3', 'ch4']].values
    tt_seconds = audio_df['tt'].values

    rms_time, rms_energy = _detect_stomp_by_rms_energy(audio_channels, tt_seconds, sr)

    assert rms_time is not None, "RMS method should detect stomp"
    assert abs(rms_time - 1.5) < 0.2, f"RMS method should detect stomp within 200ms, got error of {abs(rms_time - 1.5):.3f}s"
    assert rms_energy > 0, "RMS energy should be positive"


def test_impact_onset_stomp_detection():
    """Test impact onset method detects stomp within 200ms."""
    audio_df = create_synthetic_stomp_audio(stomp_time=1.5)
    sr = 10000

    audio_channels = audio_df[['ch1', 'ch2', 'ch3', 'ch4']].values
    tt_seconds = audio_df['tt'].values

    onset_time, onset_mag = _detect_stomp_by_impact_onset(audio_channels, tt_seconds, sr)

    assert onset_time is not None, "Onset method should detect stomp"
    assert abs(onset_time - 1.5) < 0.2, f"Onset method should detect stomp within 200ms, got error of {abs(onset_time - 1.5):.3f}s"
    assert onset_mag > 0, "Onset magnitude should be positive"


def test_frequency_content_stomp_detection():
    """Test frequency content method detects stomp within 200ms."""
    audio_df = create_synthetic_stomp_audio(stomp_time=1.5)
    sr = 10000

    audio_channels = audio_df[['ch1', 'ch2', 'ch3', 'ch4']].values
    tt_seconds = audio_df['tt'].values

    freq_time, freq_energy = _detect_stomp_by_frequency_content(audio_channels, tt_seconds, sr)

    assert freq_time is not None, "Frequency method should detect stomp"
    assert abs(freq_time - 1.5) < 0.2, f"Frequency method should detect stomp within 200ms, got error of {abs(freq_time - 1.5):.3f}s"
    assert freq_energy > 0, "Frequency energy should be positive"


def test_consensus_stomp_detection():
    """Test that consensus of methods produces accurate results."""
    audio_df = create_synthetic_stomp_audio(stomp_time=1.5)
    sr = 10000

    audio_channels = audio_df[['ch1', 'ch2', 'ch3', 'ch4']].values
    tt_seconds = audio_df['tt'].values

    # Test each method
    rms_time, _ = _detect_stomp_by_rms_energy(audio_channels, tt_seconds, sr)
    onset_time, _ = _detect_stomp_by_impact_onset(audio_channels, tt_seconds, sr)
    freq_time, _ = _detect_stomp_by_frequency_content(audio_channels, tt_seconds, sr)

    # Check consensus
    times = [rms_time, onset_time, freq_time]
    consensus = np.median(times)

    assert abs(consensus - 1.5) < 0.2, f"Consensus should detect stomp within 200ms, got error of {abs(consensus - 1.5):.3f}s"


def test_full_stomp_detection_without_biomechanics():
    """Test get_audio_stomp_time function without biomechanics guidance."""
    audio_df = create_synthetic_stomp_audio(stomp_time=2.3)

    detected_stomp_td = get_audio_stomp_time(audio_df)
    detected_stomp_s = detected_stomp_td.total_seconds()

    assert abs(detected_stomp_s - 2.3) < 0.2, (
        f"Stomp should be detected within 200ms of expected time, "
        f"got error of {abs(detected_stomp_s - 2.3):.3f}s"
    )


def create_dual_stomp_audio(sr=10000, duration=5.0, stomp_time_1=1.0, stomp_time_2=3.0):
    """Create synthetic audio with two distinct stomp events.

    Args:
        sr: Sampling rate in Hz (reduced to 10kHz for fast testing)
        duration: Total duration in seconds
        stomp_time_1: Time of the first stomp event in seconds
        stomp_time_2: Time of the second stomp event in seconds

    Returns:
        DataFrame with synthetic audio containing two stomps
    """
    n_samples = int(sr * duration)
    t = np.arange(n_samples) / sr

    # Start with quiet background noise
    audio = np.random.normal(0, 0.01, (n_samples, 4))

    # Add first stomp event
    stomp_envelope_1 = np.exp(-((t - stomp_time_1)**2) / (2 * 0.075**2))

    # Add second stomp event
    stomp_envelope_2 = np.exp(-((t - stomp_time_2)**2) / (2 * 0.075**2))

    # Add multiple frequency components for both stomps
    for ch in range(4):
        freq1 = 200 + ch * 50
        freq2 = 400 + ch * 100
        freq3 = 800 + ch * 150

        stomp_signal = (
            0.5 * np.sin(2 * np.pi * freq1 * t) +
            0.3 * np.sin(2 * np.pi * freq2 * t) +
            0.2 * np.sin(2 * np.pi * freq3 * t)
        )

        # Apply both stomp envelopes
        audio[:, ch] += stomp_signal * stomp_envelope_1 * 0.8
        audio[:, ch] += stomp_signal * stomp_envelope_2 * 0.8

    # Create DataFrame
    df = pd.DataFrame({
        'tt': t,
        'ch1': audio[:, 0],
        'ch2': audio[:, 1],
        'ch3': audio[:, 2],
        'ch4': audio[:, 3],
    })

    return df


def test_stomp_detection_with_left_knee_biomechanics():
    """Test stomp detection with left knee biomechanics guidance."""
    # Create synthetic audio with two stomps (left at 1.0s, right at 3.0s)
    audio_df = create_dual_stomp_audio(stomp_time_1=1.0, stomp_time_2=3.0)

    # Simulate biomechanics stomp times
    left_stomp_bio = timedelta(seconds=1.0)
    right_stomp_bio = timedelta(seconds=3.0)

    # Test recording left knee
    detected_left = get_audio_stomp_time(
        audio_df,
        recorded_knee="left",
        left_stomp_time=left_stomp_bio,
        right_stomp_time=right_stomp_bio,
    )

    assert abs(detected_left.total_seconds() - 1.0) < 0.5, (
        f"Left knee stomp should be detected near first stomp (1.0s), "
        f"got {detected_left.total_seconds():.3f}s"
    )


def test_stomp_detection_with_right_knee_biomechanics():
    """Test stomp detection with right knee biomechanics guidance."""
    # Create synthetic audio with two stomps (left at 1.0s, right at 3.0s)
    audio_df = create_dual_stomp_audio(stomp_time_1=1.0, stomp_time_2=3.0)

    # Simulate biomechanics stomp times
    left_stomp_bio = timedelta(seconds=1.0)
    right_stomp_bio = timedelta(seconds=3.0)

    # Test recording right knee
    detected_right = get_audio_stomp_time(
        audio_df,
        recorded_knee="right",
        left_stomp_time=left_stomp_bio,
        right_stomp_time=right_stomp_bio,
    )

    assert abs(detected_right.total_seconds() - 3.0) < 0.5, (
        f"Right knee stomp should be detected near second stomp (3.0s), "
        f"got {detected_right.total_seconds():.3f}s"
    )
