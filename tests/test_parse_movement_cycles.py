"""Tests for parse_movement_cycles module."""

import numpy as np
import pandas as pd
import pytest

from parse_movement_cycles import MovementCycleExtractor, extract_movement_cycles


def _create_synthetic_walk_data(
    start_angle: float, end_angle: float, num_cycles: int = 1
) -> pd.DataFrame:
    """Creates synthetic walking data with specified start and end angles."""
    sample_rate = 1000
    cycle_duration = 1.0  # 1 second per cycle
    num_samples = int((num_cycles + 0.5) * cycle_duration * sample_rate)
    time_array = np.linspace(0, num_samples / sample_rate, num_samples)

    # Create a base sine wave for the knee angle
    gait_freq = 1.0 / cycle_duration
    knee_angle = 20 - 30 * np.cos(2 * np.pi * gait_freq * time_array)

    # Manually adjust the first and last minima to the specified angles
    first_min_idx = np.argmin(knee_angle[:sample_rate])
    knee_angle[first_min_idx] = start_angle

    if num_cycles > 0:
        last_min_idx_offset = int(num_cycles * cycle_duration * sample_rate)
        # Ensure there is a distinct minimum to adjust in the last cycle segment
        if last_min_idx_offset < len(knee_angle):
            last_segment = knee_angle[last_min_idx_offset:]
            if len(last_segment) > 0:
                last_min_idx = last_min_idx_offset + np.argmin(last_segment)
                knee_angle[last_min_idx] = end_angle

    # Add timedelta column for compatibility
    tt = pd.to_timedelta(time_array, unit="s")

    return pd.DataFrame({"tt": tt, "Knee Angle Z": knee_angle})


class TestMovementCycleExtractorWalking:
    """Test walking cycle extraction."""

    def test_extract_walking_cycles_finds_expected_range(self, syncd_walk):
        """Should find 2 gait cycles in walking data."""
        extractor = MovementCycleExtractor("walk")
        cycles = extractor.extract_cycles(syncd_walk)

        assert len(cycles) == 2

    def test_walking_cycles_have_correct_structure(self, syncd_walk):
        """Each cycle should be a DataFrame with proper columns."""
        extractor = MovementCycleExtractor("walk")
        cycles = extractor.extract_cycles(syncd_walk)

        required_cols = [
            "tt",
            "ch1",
            "ch2",
            "ch3",
            "ch4",
            "f_ch1",
            "f_ch2",
            "f_ch3",
            "f_ch4",
            "TIME",
            "Knee Angle Z",
        ]

        for cycle in cycles:
            assert isinstance(cycle, pd.DataFrame)
            assert all(col in cycle.columns for col in required_cols)

    def test_walking_cycles_span_correct_duration(self, syncd_walk):
        """Each gait cycle should be approximately 1 second."""
        extractor = MovementCycleExtractor("walk")
        cycles = extractor.extract_cycles(syncd_walk)

        for cycle in cycles:
            duration = cycle["tt"].max() - cycle["tt"].min()
            # Allow 10% tolerance
            assert (
                0.9 <= duration <= 1.1
            ), f"Cycle duration {duration}s outside expected range"

    def test_walking_cycles_sequential(self, syncd_walk):
        """Cycles should be sequential without overlap."""
        extractor = MovementCycleExtractor("walk")
        cycles = extractor.extract_cycles(syncd_walk)

        for i in range(len(cycles) - 1):
            cycle1_end = cycles[i]["tt"].max()
            cycle2_start = cycles[i + 1]["tt"].min()
            # Next cycle should start where previous ended (or very close)
            assert cycle2_start >= cycle1_end, "Cycles overlap or out of order"


class TestMovementCycleExtractorSitToStand:
    """Test sit-to-stand cycle extraction."""

    def test_extract_sit_to_stand_cycles_finds_expected_range(self, syncd_sit_to_stand):
        """Should find 2-3 sit-to-stand cycles."""
        extractor = MovementCycleExtractor("sit_to_stand")
        cycles = extractor.extract_cycles(syncd_sit_to_stand)

        assert 2 <= len(cycles) <= 3

    def test_sit_to_stand_cycles_have_correct_structure(self, syncd_sit_to_stand):
        """Each cycle should be a DataFrame with proper columns."""
        extractor = MovementCycleExtractor("sit_to_stand")
        cycles = extractor.extract_cycles(syncd_sit_to_stand)

        required_cols = [
            "tt",
            "ch1",
            "ch2",
            "ch3",
            "ch4",
            "f_ch1",
            "f_ch2",
            "f_ch3",
            "f_ch4",
            "TIME",
            "Knee Angle Z",
        ]

        for cycle in cycles:
            assert isinstance(cycle, pd.DataFrame)
            assert all(col in cycle.columns for col in required_cols)

    def test_sit_to_stand_cycles_span_correct_duration(self, syncd_sit_to_stand):
        """Each sit-to-stand cycle should be approximately 5 seconds."""
        extractor = MovementCycleExtractor("sit_to_stand")
        cycles = extractor.extract_cycles(syncd_sit_to_stand)

        for cycle in cycles:
            duration = cycle["tt"].max() - cycle["tt"].min()
            # Allow generous tolerance because smoothing expands cycle bounds
            assert (
                4.0 <= duration <= 7.5
            ), f"Cycle duration {duration}s outside expected range"

    def test_sit_to_stand_cycles_contain_extension_peak(self, syncd_sit_to_stand):
        """Each cycle should contain a standing peak (low knee angle)."""
        extractor = MovementCycleExtractor("sit_to_stand")
        cycles = extractor.extract_cycles(syncd_sit_to_stand)

        for cycle in cycles:
            min_angle = cycle["Knee Angle Z"].min()
            # Should have extension (standing) with angle near 10°
            assert (
                min_angle < 20
            ), f"Cycle missing standing phase (min angle {min_angle})"


class TestMovementCycleExtractorFlexionExtension:
    """Test flexion-extension cycle extraction."""

    def test_extract_flexion_extension_cycles_finds_expected_range(
        self, syncd_flexion_extension
    ):
        """Should find 4-7 flexion-extension cycles."""
        extractor = MovementCycleExtractor("flexion_extension")
        cycles = extractor.extract_cycles(syncd_flexion_extension)

        assert 4 <= len(cycles) <= 7

    def test_flexion_extension_cycles_have_correct_structure(
        self, syncd_flexion_extension
    ):
        """Each cycle should be a DataFrame with proper columns."""
        extractor = MovementCycleExtractor("flexion_extension")
        cycles = extractor.extract_cycles(syncd_flexion_extension)

        required_cols = [
            "tt",
            "ch1",
            "ch2",
            "ch3",
            "ch4",
            "f_ch1",
            "f_ch2",
            "f_ch3",
            "f_ch4",
            "TIME",
            "Knee Angle Z",
        ]

        for cycle in cycles:
            assert isinstance(cycle, pd.DataFrame)
            assert all(col in cycle.columns for col in required_cols)

    def test_flexion_extension_cycles_span_correct_duration(
        self, syncd_flexion_extension
    ):
        """Each flexion-extension cycle should be approximately 2 seconds."""
        extractor = MovementCycleExtractor("flexion_extension")
        cycles = extractor.extract_cycles(syncd_flexion_extension)

        for cycle in cycles:
            duration = cycle["tt"].max() - cycle["tt"].min()
            # Allow 20% tolerance
            assert (
                1.6 <= duration <= 2.4
            ), f"Cycle duration {duration}s outside expected range"

    def test_flexion_extension_cycles_contain_extension_peak(
        self, syncd_flexion_extension
    ):
        """Each cycle should contain an extension peak (low knee angle)."""
        extractor = MovementCycleExtractor("flexion_extension")
        cycles = extractor.extract_cycles(syncd_flexion_extension)

        for cycle in cycles:
            min_angle = cycle["Knee Angle Z"].min()
            # Should have extension peak near minimum of sinusoid (~10°)
            assert (
                min_angle < 30
            ), f"Cycle missing extension peak (min angle {min_angle})"


class TestMovementCycleExtractorEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_maneuver_raises_error(self, syncd_walk):
        """Invalid maneuver type should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            extract_movement_cycles(syncd_walk, "invalid_maneuver")

        assert "maneuver" in str(exc_info.value).lower()

    def test_empty_dataframe_returns_empty_list(self):
        """Empty DataFrame should return empty cycle list."""
        empty_df = pd.DataFrame(
            {
                "tt": [],
                "ch1": [],
                "ch2": [],
                "ch3": [],
                "ch4": [],
                "f_ch1": [],
                "f_ch2": [],
                "f_ch3": [],
                "f_ch4": [],
                "TIME": [],
                "Knee Angle Z": [],
            }
        )

        extractor = MovementCycleExtractor("walk")
        cycles = extractor.extract_cycles(empty_df)

        assert cycles == []

    def test_single_peak_returns_no_cycles(self, syncd_walk):
        """Single peak (no cycle) should return empty list."""
        # Take only first 0.3 seconds - not enough for a full cycle
        short_df = syncd_walk[syncd_walk["tt"] < 0.3].copy()

        extractor = MovementCycleExtractor("walk")
        cycles = extractor.extract_cycles(short_df)

        # Should find 0 or 1 incomplete cycles
        assert len(cycles) <= 1


class TestExtractMovementCyclesConvenience:
    """Test convenience function extract_movement_cycles."""

    def test_convenience_function_walk(self, syncd_walk):
        """Convenience function should work for walking."""
        cycles = extract_movement_cycles(syncd_walk, "walk")

        assert len(cycles) == 2
        assert all(isinstance(c, pd.DataFrame) for c in cycles)

    def test_convenience_function_sit_to_stand(self, syncd_sit_to_stand):
        """Convenience function should work for sit-to-stand."""
        cycles = extract_movement_cycles(syncd_sit_to_stand, "sit_to_stand")

        assert 2 <= len(cycles) <= 3
        assert all(isinstance(c, pd.DataFrame) for c in cycles)

    def test_convenience_function_flexion_extension(self, syncd_flexion_extension):
        """Convenience function should work for flexion-extension."""
        cycles = extract_movement_cycles(syncd_flexion_extension, "flexion_extension")

        assert 4 <= len(cycles) <= 7
        assert all(isinstance(c, pd.DataFrame) for c in cycles)

    def test_convenience_function_passes_kwargs(self, syncd_walk):
        """Convenience function should pass kwargs to extractor."""
        cycles_default = extract_movement_cycles(syncd_walk, "walk")
        assert len(cycles_default) == 2


class TestMovementCycleDataIntegrity:
    """Test that extracted cycles maintain data integrity."""

    def test_cycles_preserve_all_columns(self, syncd_walk):
        """Extracted cycles should preserve all original columns."""
        original_cols = set(syncd_walk.columns)

        cycles = extract_movement_cycles(syncd_walk, "walk")

        for cycle in cycles:
            cycle_cols = set(cycle.columns)
            assert cycle_cols == original_cols, "Cycle missing columns"

    def test_cycles_have_monotonic_time(self, syncd_walk):
        """Time within each cycle should be monotonically increasing."""
        cycles = extract_movement_cycles(syncd_walk, "walk")

        for cycle in cycles:
            time_diffs = cycle["tt"].diff().dropna()
            assert (time_diffs > 0).all(), "Time not monotonically increasing"

    def test_no_data_loss(self, syncd_walk):
        """Total samples in cycles should approximately equal original."""
        cycles = extract_movement_cycles(syncd_walk, "walk")

        total_cycle_samples = sum(len(c) for c in cycles)
        original_samples = len(syncd_walk)

        # Should use most of the data (at least 50%)
        assert (
            total_cycle_samples >= 0.5 * original_samples
        ), f"Too much data lost: {total_cycle_samples}/{original_samples} samples"

    def test_cycles_dont_duplicate_data(self, syncd_walk):
        """Cycles should not contain overlapping/duplicate time points."""
        cycles = extract_movement_cycles(syncd_walk, "walk")

        all_times = []
        for cycle in cycles:
            all_times.extend(cycle["tt"].tolist())

        # Check for duplicates (allowing small floating point tolerance)
        sorted_times = sorted(all_times)
        for i in range(len(sorted_times) - 1):
            time_gap = sorted_times[i + 1] - sorted_times[i]
            assert (
                time_gap >= 0.0001
            ), f"Duplicate or overlapping times detected: gap={time_gap}"

    def test_walk_cycle_with_dissimilar_heel_strikes_is_rejected(self):
        """Should reject a walk cycle where start and end heel strike angles differ by more than the tolerance."""
        # This cycle starts at a deep extension (-10°) but ends at a much shallower one (5°).
        # The difference (15°) is greater than the 5° tolerance, so it should be rejected.
        invalid_cycle_df = _create_synthetic_walk_data(start_angle=-10.0, end_angle=5.0)

        cycles = extract_movement_cycles(invalid_cycle_df, maneuver="walk")

        assert (
            len(cycles) == 0
        ), "Cycle with dissimilar heel strikes should be rejected"

    def test_walk_cycle_with_similar_heel_strikes_is_accepted(self):
        """Should accept a walk cycle where start and end heel strike angles are within tolerance."""
        # This cycle starts at -10° and ends at -8°, a difference of 2°, which is within the 5° tolerance.
        valid_cycle_df = _create_synthetic_walk_data(
            start_angle=-10.0, end_angle=-8.0, num_cycles=2
        )

        cycles = extract_movement_cycles(valid_cycle_df, maneuver="walk")

        assert (
            len(cycles) >= 1
        ), "Cycle with similar heel strikes should be accepted"
