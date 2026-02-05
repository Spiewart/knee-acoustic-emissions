"""Tests for detection_results fields: selected_stomp_method, bio_selected_time, contra_bio_selected_time."""

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
    """Create synthetic audio data with two distinct stomp events."""
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

    df = pd.DataFrame({
        'tt': tt,
        'ch1': audio[:, 0],
        'ch2': audio[:, 1],
        'ch3': audio[:, 2],
        'ch4': audio[:, 3],
    })
    return df


def test_consensus_method_fields():
    """Test that consensus method populates fields correctly without biomechanics."""
    # Create synthetic audio with single clear stomp
    audio_df = create_synthetic_audio_with_two_stomps(
        stomp1_time=2.0,
        stomp2_time=5.0,
        stomp1_amplitude=1000.0,
        stomp2_amplitude=500.0,
    )

    # Call without biomechanics parameters (should use consensus)
    result_td, detection_results = get_audio_stomp_time(
        audio_df,
        return_details=True
    )

    # Verify fields exist and have expected values
    assert 'selected_stomp_method' in detection_results
    assert detection_results['selected_stomp_method'] == 'consensus'
    assert 'bio_selected_time' in detection_results
    assert detection_results['bio_selected_time'] is None  # Not populated in consensus mode
    assert 'contra_bio_selected_time' in detection_results
    assert detection_results['contra_bio_selected_time'] is None  # Not populated in consensus mode


def test_biomechanics_guided_method_fields():
    """Test that biomechanics-guided method populates all fields correctly.

    Note: Recorded knee's stomp must be >= 20% louder than contralateral to pass
    energy ratio validation (ratio >= 1.2).
    """
    # Create synthetic audio: right stomp at 2.0s, left stomp at 3.5s (1.5s apart)
    # Right stomp (recorded knee) must be louder: 1200/900 = 1.33 > 1.2
    audio_df = create_synthetic_audio_with_two_stomps(
        stomp1_time=2.0,  # Right stomp (earlier, louder - recorded leg)
        stomp2_time=3.5,  # Left stomp (later, quieter - contralateral)
        stomp1_amplitude=1200.0,  # Recorded knee must be >= 20% louder
        stomp2_amplitude=900.0,
    )

    # Biomechanics times matching the synthetic audio
    right_stomp_td = timedelta(seconds=2.0)
    left_stomp_td = timedelta(seconds=3.5)

    # Test for right knee (should select first stomp, contra is second)
    result_td, detection_results = get_audio_stomp_time(
        audio_df,
        recorded_knee="right",
        right_stomp_time=right_stomp_td,
        left_stomp_time=left_stomp_td,
        return_details=True
    )

    # Verify method is biomechanics (energy ratio should pass)
    assert detection_results['selected_stomp_method'] == 'biomechanics', \
        f"Expected biomechanics but got {detection_results['selected_stomp_method']}. " \
        f"Energy ratio validation may have failed."

    # Verify bio_selected_time is populated and close to right stomp
    assert detection_results['bio_selected_time'] is not None
    assert abs(detection_results['bio_selected_time'] - 2.0) < 0.3, \
        f"Selected time {detection_results['bio_selected_time']} not close to 2.0s"

    # Verify contra_bio_selected_time is populated and close to left stomp
    assert detection_results['contra_bio_selected_time'] is not None
    assert abs(detection_results['contra_bio_selected_time'] - 3.5) < 0.3, \
        f"Contra time {detection_results['contra_bio_selected_time']} not close to 3.5s"

    # Verify energy_ratio is populated and passes threshold
    assert detection_results['energy_ratio'] is not None
    assert detection_results['energy_ratio'] >= 1.2, \
        f"Energy ratio {detection_results['energy_ratio']} should be >= 1.2"

    # Test for left knee (should select second stomp, contra is first)
    # Left stomp (recorded knee) must be louder: 1200/900 = 1.33 > 1.2
    audio_df_left = create_synthetic_audio_with_two_stomps(
        stomp1_time=2.0,  # Right stomp (earlier, quieter - contralateral)
        stomp2_time=3.5,  # Left stomp (later, louder - recorded leg)
        stomp1_amplitude=900.0,
        stomp2_amplitude=1200.0,  # Recorded knee (left) is louder
    )

    result_td, detection_results = get_audio_stomp_time(
        audio_df_left,
        recorded_knee="left",
        right_stomp_time=right_stomp_td,
        left_stomp_time=left_stomp_td,
        return_details=True
    )

    assert detection_results['selected_stomp_method'] == 'biomechanics'
    assert detection_results['bio_selected_time'] is not None
    assert abs(detection_results['bio_selected_time'] - 3.5) < 0.3
    assert detection_results['contra_bio_selected_time'] is not None
    assert abs(detection_results['contra_bio_selected_time'] - 2.0) < 0.3


def test_fallback_to_consensus_fields():
    """Test that when biomechanics-guided fails, method stays 'consensus' with None times."""
    # Create audio with stomps that don't match expected biomechanics delta
    audio_df = create_synthetic_audio_with_two_stomps(
        stomp1_time=2.0,
        stomp2_time=2.3,  # Only 0.3s apart, won't match expected 5.0s delta
        stomp1_amplitude=1000.0,
        stomp2_amplitude=900.0,
    )

    # Biomechanics suggests 5 second separation (much larger than actual)
    right_stomp_td = timedelta(seconds=1.0)
    left_stomp_td = timedelta(seconds=6.0)  # 5 seconds apart

    result_td, detection_results = get_audio_stomp_time(
        audio_df,
        recorded_knee="right",
        right_stomp_time=right_stomp_td,
        left_stomp_time=left_stomp_td,
        return_details=True
    )

    # Should fallback to consensus when no valid pair found
    assert detection_results['selected_stomp_method'] == 'consensus'
    assert detection_results['bio_selected_time'] is None
    assert detection_results['contra_bio_selected_time'] is None


def test_all_standard_fields_still_present():
    """Verify that existing detection_results fields are still populated."""
    audio_df = create_synthetic_audio_with_two_stomps(
        stomp1_time=2.0,
        stomp2_time=3.5,
        stomp1_amplitude=1000.0,
    )

    right_stomp_td = timedelta(seconds=2.0)
    left_stomp_td = timedelta(seconds=3.5)

    result_td, detection_results = get_audio_stomp_time(
        audio_df,
        recorded_knee="right",
        right_stomp_time=right_stomp_td,
        left_stomp_time=left_stomp_td,
        return_details=True
    )

    # Verify all expected fields exist
    expected_fields = [
        'consensus_time', 'rms_time', 'rms_energy',
        'onset_time', 'onset_magnitude',
        'freq_time', 'freq_energy',
        'selected_stomp_method', 'bio_selected_time', 'contra_bio_selected_time'
    ]

    for field in expected_fields:
        assert field in detection_results, f"Missing field: {field}"

    # Verify numeric fields have reasonable values
    assert detection_results['consensus_time'] > 0
    assert detection_results['rms_time'] > 0
    assert detection_results['rms_energy'] > 0
    assert detection_results['onset_time'] > 0
    assert detection_results['onset_magnitude'] > 0
    assert detection_results['freq_time'] > 0
    assert detection_results['freq_energy'] > 0


def test_backward_compatibility_without_return_details():
    """Test that function still works without return_details flag (backward compatibility)."""
    audio_df = create_synthetic_audio_with_two_stomps(
        stomp1_time=2.0,
        stomp2_time=3.5,
    )

    # Call without return_details (default behavior)
    result_td = get_audio_stomp_time(audio_df)

    # Should return just timedelta, not tuple
    assert isinstance(result_td, timedelta)
    assert not isinstance(result_td, tuple)
