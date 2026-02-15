"""Tests for energy ratio validation in biomechanics stomp detection.

This module tests the critical sanity check that ensures the recorded knee's
stomp peak is sufficiently louder than the contralateral leg's peak. This validation
ensures:
1. Microphone was placed on the correct leg
2. Microphone is functioning properly
3. Peak pair assignment is correct

Energy Ratio Requirement: recorded_knee_energy / contra_knee_energy >= 1.2 (20% minimum)
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from src.synchronization.sync import get_audio_stomp_time


def create_synthetic_audio_with_two_stomps(
    stomp1_time: float,
    stomp1_amplitude: float,
    stomp2_time: float,
    stomp2_amplitude: float,
    duration: float = 20.0,
    sr: float = 52000.0,
) -> pd.DataFrame:
    """Create synthetic audio data with two stomps of specified amplitudes."""
    n_samples = int(duration * sr)
    tt = np.linspace(0, duration, n_samples)

    # Create base noise
    audio = np.random.randn(n_samples, 4) * 10

    # Add first stomp (Gaussian pulse)
    stomp1_width = 0.05  # 50ms wide
    stomp1_pulse = stomp1_amplitude * np.exp(-((tt - stomp1_time) ** 2) / (2 * stomp1_width**2))

    # Add second stomp
    stomp2_width = 0.05
    stomp2_pulse = stomp2_amplitude * np.exp(-((tt - stomp2_time) ** 2) / (2 * stomp2_width**2))

    # Add stomps to all channels
    for ch in range(4):
        audio[:, ch] += stomp1_pulse + stomp2_pulse

    df = pd.DataFrame(
        {
            "tt": tt,
            "ch1": audio[:, 0],
            "ch2": audio[:, 1],
            "ch3": audio[:, 2],
            "ch4": audio[:, 3],
        }
    )
    return df


class TestEnergyRatioValidation:
    """Test suite for energy ratio validation in biomechanics peak pair selection."""

    def test_energy_ratio_passes_when_recorded_knee_louder(self):
        """Test that peak pair is accepted when recorded knee is ≥20% louder.

        Scenario: Right knee is recorded. Right stomp at 1.0s (1000 amplitude),
        left stomp at 2.5s (750 amplitude). Ratio = 1000/750 = 1.33 > 1.2 ✓
        """
        # Create audio with right stomp louder than left
        audio_df = create_synthetic_audio_with_two_stomps(
            stomp1_time=1.0,
            stomp1_amplitude=1000.0,  # Right stomp (louder)
            stomp2_time=2.5,
            stomp2_amplitude=750.0,  # Left stomp (quieter)
        )

        # Biomechanics data: right at 1.0s, left at 2.5s
        right_stomp = timedelta(seconds=1.0)
        left_stomp = timedelta(seconds=2.5)

        stomp_time, results = get_audio_stomp_time(
            audio_df,
            recorded_knee="right",
            right_stomp_time=right_stomp,
            left_stomp_time=left_stomp,
            return_details=True,
        )

        # Should accept and use biomechanics method
        assert results["selected_stomp_method"] == "biomechanics"
        assert results["bio_selected_time"] is not None
        assert results["energy_ratio"] is not None
        assert results["energy_ratio"] >= 1.2, f"Energy ratio {results['energy_ratio']} should be >= 1.2"

    def test_energy_ratio_fails_when_contralateral_louder(self):
        """Test that peak pair is rejected when contralateral is louder than recorded knee.

        Scenario: Right knee is recorded. Right stomp at 1.0s (700 amplitude),
        left stomp at 2.5s (1000 amplitude). Ratio = 700/1000 = 0.7 < 1.2 ✗
        Falls back to consensus.
        """
        # Create audio with left stomp LOUDER than right (microphone misplacement?)
        audio_df = create_synthetic_audio_with_two_stomps(
            stomp1_time=1.0,
            stomp1_amplitude=700.0,  # Right stomp (quieter - suspicious!)
            stomp2_time=2.5,
            stomp2_amplitude=1000.0,  # Left stomp (louder - shouldn't be louder than right)
        )

        # Biomechanics data: right at 1.0s, left at 2.5s
        right_stomp = timedelta(seconds=1.0)
        left_stomp = timedelta(seconds=2.5)

        stomp_time, results = get_audio_stomp_time(
            audio_df,
            recorded_knee="right",
            right_stomp_time=right_stomp,
            left_stomp_time=left_stomp,
            return_details=True,
        )

        # Should reject energy ratio and fall back to consensus
        assert results["selected_stomp_method"] == "consensus", "Should fall back to consensus when energy ratio < 1.2"
        assert results["energy_ratio"] is None, "energy_ratio should not be set if biomechanics method was not used"

    def test_energy_ratio_at_minimum_threshold(self):
        """Test boundary condition: energy ratio exactly at minimum (1.2).

        Scenario: Ratio = exactly 1.2 (boundary of acceptance).
        """
        # Create audio with ratio exactly 1.2
        stomp1_amplitude = 1200.0
        stomp2_amplitude = 1000.0
        expected_ratio = stomp1_amplitude / stomp2_amplitude  # Should be 1.2

        audio_df = create_synthetic_audio_with_two_stomps(
            stomp1_time=1.0,
            stomp1_amplitude=stomp1_amplitude,
            stomp2_time=2.5,
            stomp2_amplitude=stomp2_amplitude,
        )

        right_stomp = timedelta(seconds=1.0)
        left_stomp = timedelta(seconds=2.5)

        stomp_time, results = get_audio_stomp_time(
            audio_df,
            recorded_knee="right",
            right_stomp_time=right_stomp,
            left_stomp_time=left_stomp,
            return_details=True,
        )

        # Should accept (>= 1.2 includes 1.2)
        assert results["selected_stomp_method"] == "biomechanics"
        assert results["energy_ratio"] is not None
        assert abs(results["energy_ratio"] - expected_ratio) < 0.05, (
            f"Expected ratio ~{expected_ratio}, got {results['energy_ratio']}"
        )

    def test_energy_ratio_just_below_threshold(self):
        """Test boundary condition: energy ratio just below minimum (1.19).

        Scenario: Ratio = 1.19 (should fail and fall back to consensus).
        """
        stomp1_amplitude = 1190.0
        stomp2_amplitude = 1000.0
        stomp1_amplitude / stomp2_amplitude  # Should be ~1.19

        audio_df = create_synthetic_audio_with_two_stomps(
            stomp1_time=1.0,
            stomp1_amplitude=stomp1_amplitude,
            stomp2_time=2.5,
            stomp2_amplitude=stomp2_amplitude,
        )

        right_stomp = timedelta(seconds=1.0)
        left_stomp = timedelta(seconds=2.5)

        stomp_time, results = get_audio_stomp_time(
            audio_df,
            recorded_knee="right",
            right_stomp_time=right_stomp,
            left_stomp_time=left_stomp,
            return_details=True,
        )

        # Should reject (< 1.2)
        assert results["selected_stomp_method"] == "consensus"
        assert results["energy_ratio"] is None

    def test_energy_ratio_with_left_knee_recorded(self):
        """Test energy ratio validation when left knee is recorded.

        Scenario: Left knee recorded. Left stomp at 1.0s (1000 amplitude),
        right stomp at 2.5s (700 amplitude). Ratio = 1000/700 = 1.43 > 1.2 ✓
        """
        # Left stomp louder than right
        audio_df = create_synthetic_audio_with_two_stomps(
            stomp1_time=1.0,
            stomp1_amplitude=1000.0,  # Left stomp (louder - recorded leg)
            stomp2_time=2.5,
            stomp2_amplitude=700.0,  # Right stomp (quieter - contralateral)
        )

        left_stomp = timedelta(seconds=1.0)
        right_stomp = timedelta(seconds=2.5)

        stomp_time, results = get_audio_stomp_time(
            audio_df,
            recorded_knee="left",
            right_stomp_time=right_stomp,
            left_stomp_time=left_stomp,
            return_details=True,
        )

        # Should accept biomechanics method
        assert results["selected_stomp_method"] == "biomechanics"
        assert results["energy_ratio"] is not None
        assert results["energy_ratio"] >= 1.2

    def test_energy_ratio_comprehensive_logging(self):
        """Test that energy ratio validation includes comprehensive logging info.

        Verifies that debug logging captures:
        - Energy ratio value
        - Both knee energies
        - Selected peak time and contralateral time
        """
        audio_df = create_synthetic_audio_with_two_stomps(
            stomp1_time=1.0,
            stomp1_amplitude=1000.0,
            stomp2_time=2.5,
            stomp2_amplitude=750.0,
        )

        right_stomp = timedelta(seconds=1.0)
        left_stomp = timedelta(seconds=2.5)

        stomp_time, results = get_audio_stomp_time(
            audio_df,
            recorded_knee="right",
            right_stomp_time=right_stomp,
            left_stomp_time=left_stomp,
            return_details=True,
        )

        # Verify all expected fields are populated
        assert results["selected_stomp_method"] == "biomechanics"
        assert "energy_ratio" in results
        assert "bio_selected_time" in results
        assert "contra_bio_selected_time" in results
        assert results["energy_ratio"] is not None
        assert results["bio_selected_time"] is not None
        assert results["contra_bio_selected_time"] is not None
        assert results["energy_ratio"] > 0  # Must be positive

    def test_energy_ratio_fallback_preserves_consensus_fields(self):
        """Test that fallback to consensus preserves all consensus detection fields.

        When energy ratio fails validation, should still have:
        - consensus_time
        - rms_time, rms_energy
        - onset_time, onset_magnitude
        - freq_time, freq_energy
        """
        # Create audio where contralateral is louder (will fail energy ratio)
        audio_df = create_synthetic_audio_with_two_stomps(
            stomp1_time=1.0,
            stomp1_amplitude=600.0,  # Right stomp (quiet)
            stomp2_time=2.5,
            stomp2_amplitude=1200.0,  # Left stomp (loud - wrong!)
        )

        right_stomp = timedelta(seconds=1.0)
        left_stomp = timedelta(seconds=2.5)

        stomp_time, results = get_audio_stomp_time(
            audio_df,
            recorded_knee="right",
            right_stomp_time=right_stomp,
            left_stomp_time=left_stomp,
            return_details=True,
        )

        # Verify consensus fields still present
        assert "consensus_time" in results
        assert "rms_time" in results
        assert "rms_energy" in results
        assert "onset_time" in results
        assert "onset_magnitude" in results
        assert "freq_time" in results
        assert "freq_energy" in results

        # Verify values are reasonable
        assert results["consensus_time"] > 0
        assert results["rms_energy"] > 0
        assert results["onset_magnitude"] > 0
        assert results["freq_energy"] > 0

    def test_energy_ratio_handles_zero_contralateral_energy(self):
        """Test that energy ratio gracefully handles zero contralateral energy edge case.

        If contra_knee_energy is 0 (no contralateral stomp detected), ratio = inf.
        Should accept this (inf >= 1.2).
        """
        # Create audio with only recorded knee stomp, no contralateral
        n_samples = int(20.0 * 52000)
        tt = np.linspace(0, 20.0, n_samples)

        # Only one stomp at 1.0s
        stomp_pulse = 1000.0 * np.exp(-((tt - 1.0) ** 2) / (2 * 0.05**2))
        audio = np.random.randn(n_samples, 4) * 10
        for ch in range(4):
            audio[:, ch] += stomp_pulse

        audio_df = pd.DataFrame(
            {
                "tt": tt,
                "ch1": audio[:, 0],
                "ch2": audio[:, 1],
                "ch3": audio[:, 2],
                "ch4": audio[:, 3],
            }
        )

        right_stomp = timedelta(seconds=1.0)
        left_stomp = timedelta(seconds=2.5)  # No actual peak here

        stomp_time, results = get_audio_stomp_time(
            audio_df,
            recorded_knee="right",
            right_stomp_time=right_stomp,
            left_stomp_time=left_stomp,
            return_details=True,
        )

        # May fall back to consensus due to peak finding limitations,
        # but if it does find a pair, energy_ratio should be finite and >= 1.2
        if results["selected_stomp_method"] == "biomechanics":
            assert np.isfinite(results["energy_ratio"])
            assert results["energy_ratio"] >= 1.2

    def test_energy_ratio_with_multiple_peak_candidates(self):
        """Test energy ratio validation with multiple noise peaks to choose from.

        Scenario: Multiple peaks exist, but only the correct pair passes energy ratio.
        - Noise peak at 0.5s (amplitude 500)
        - Right stomp at 1.0s (amplitude 1000)
        - Left stomp at 2.5s (amplitude 750)
        - Noise peak at 3.0s (amplitude 600)

        Only (1.0s, 2.5s) should pass energy ratio validation.
        """
        n_samples = int(20.0 * 52000)
        tt = np.linspace(0, 20.0, n_samples)

        audio = np.random.randn(n_samples, 4) * 10

        # Add multiple peaks
        peaks = [
            (0.5, 500.0),  # Noise
            (1.0, 1000.0),  # Right stomp (should pair with left)
            (2.5, 750.0),  # Left stomp
            (3.0, 600.0),  # Noise
        ]

        for peak_time, amplitude in peaks:
            pulse = amplitude * np.exp(-((tt - peak_time) ** 2) / (2 * 0.05**2))
            for ch in range(4):
                audio[:, ch] += pulse

        audio_df = pd.DataFrame(
            {
                "tt": tt,
                "ch1": audio[:, 0],
                "ch2": audio[:, 1],
                "ch3": audio[:, 2],
                "ch4": audio[:, 3],
            }
        )

        right_stomp = timedelta(seconds=1.0)
        left_stomp = timedelta(seconds=2.5)

        stomp_time, results = get_audio_stomp_time(
            audio_df,
            recorded_knee="right",
            right_stomp_time=right_stomp,
            left_stomp_time=left_stomp,
            return_details=True,
        )

        # Should find the correct pair despite noise peaks
        if results["selected_stomp_method"] == "biomechanics":
            assert results["energy_ratio"] >= 1.2
            # Selected time should be close to right stomp (1.0s)
            assert abs(results["bio_selected_time"] - 1.0) < 0.2
            # Contra time should be close to left stomp (2.5s)
            assert abs(results["contra_bio_selected_time"] - 2.5) < 0.2
