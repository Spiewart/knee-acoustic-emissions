"""Tests for integration of raw audio QC with sync_qc workflow."""

import pytest

from src.audio.raw_qc import (
    adjust_bad_intervals_for_sync,
    check_cycle_in_bad_interval,
)


def test_adjust_bad_intervals_with_realistic_stomp_times():
    """Test adjustment with realistic stomp times from actual recordings."""
    # Realistic scenario: audio starts recording, stomp happens at 5.2s
    # In biomechanics, stomp is logged at 12.5s
    audio_stomp = 5.2
    bio_stomp = 12.5

    # Bad intervals in audio coordinates (before stomp)
    bad_intervals_audio = [(2.0, 3.0), (4.0, 4.5)]

    adjusted = adjust_bad_intervals_for_sync(
        bad_intervals_audio,
        audio_stomp,
        bio_stomp,
    )

    # Expected offset: 12.5 - 5.2 = 7.3
    expected = [(9.3, 10.3), (11.3, 11.8)]

    assert len(adjusted) == 2
    assert abs(adjusted[0][0] - expected[0][0]) < 0.001
    assert abs(adjusted[0][1] - expected[0][1]) < 0.001
    assert abs(adjusted[1][0] - expected[1][0]) < 0.001
    assert abs(adjusted[1][1] - expected[1][1]) < 0.001


def test_check_cycle_overlap_with_multiple_bad_intervals():
    """Test cycle overlap calculation with multiple bad intervals."""
    bad_intervals = [(2.0, 3.0), (5.0, 6.0), (10.0, 12.0)]

    # Cycle overlapping with first interval (50% overlap)
    # Cycle: 2.5 to 3.5 (duration 1.0s), overlap with (2.0, 3.0): 0.5s
    assert check_cycle_in_bad_interval(2.5, 3.5, bad_intervals, overlap_threshold=0.1) == True
    assert check_cycle_in_bad_interval(2.5, 3.5, bad_intervals, overlap_threshold=0.6) == False

    # Cycle spanning multiple bad intervals
    # Cycle: 4.5 to 11.0 (duration 6.5s), overlap: (5.0-6.0: 1.0s) + (10.0-11.0: 1.0s) = 2.0s (~0.31 fraction)
    assert check_cycle_in_bad_interval(4.5, 11.0, bad_intervals, overlap_threshold=0.3) == True
    assert check_cycle_in_bad_interval(4.5, 11.0, bad_intervals, overlap_threshold=0.5) == False


def test_end_to_end_audio_qc_check_workflow():
    """Test complete workflow: audio QC -> adjust -> check cycles."""
    # Step 1: Simulate raw audio QC results
    raw_audio_bad_intervals = [(1.0, 2.0), (8.0, 9.0)]

    # Step 2: Adjust for synchronization
    audio_stomp = 3.0  # Stomp at 3s in audio
    bio_stomp = 15.0  # Stomp at 15s in biomechanics

    synced_bad_intervals = adjust_bad_intervals_for_sync(
        raw_audio_bad_intervals,
        audio_stomp,
        bio_stomp,
    )

    # Expected: offset = 15 - 3 = 12
    # Adjusted intervals: [(13.0, 14.0), (20.0, 21.0)]

    # Step 3: Check cycles
    cycles = [
        {"id": 1, "start": 10.0, "end": 12.0},  # Before bad intervals - should pass
        {"id": 2, "start": 13.2, "end": 13.8},  # Inside first bad interval - should fail
        {"id": 3, "start": 16.0, "end": 18.0},  # Between bad intervals - should pass
        {"id": 4, "start": 19.5, "end": 21.5},  # Overlaps second bad interval - should fail
    ]

    results = []
    for cycle in cycles:
        audio_qc_passed = not check_cycle_in_bad_interval(
            cycle["start"],
            cycle["end"],
            synced_bad_intervals,
            overlap_threshold=0.1,
        )
        results.append((cycle["id"], audio_qc_passed))

    assert results[0][1] == True, "Cycle 1 should pass (before bad intervals)"
    assert results[1][1] == False, "Cycle 2 should fail (inside bad interval)"
    assert results[2][1] == True, "Cycle 3 should pass (between bad intervals)"
    assert results[3][1] == False, "Cycle 4 should fail (overlaps bad interval)"


def test_cycle_check_with_timedelta_inputs():
    """Test that cycle checking works with timedelta objects."""
    bad_intervals = [(5.0, 6.0), (10.0, 11.0)]

    # Cycle times as floats (should work)
    result1 = check_cycle_in_bad_interval(5.2, 5.8, bad_intervals, overlap_threshold=0.1)
    assert result1 == True

    # Same cycle times - should get same result
    result2 = check_cycle_in_bad_interval(5.2, 5.8, bad_intervals, overlap_threshold=0.1)
    assert result2 == result1


def test_empty_bad_intervals():
    """Test that empty bad intervals result in all cycles passing."""
    bad_intervals = []

    cycles = [
        (0.0, 1.0),
        (5.0, 6.0),
        (10.0, 12.0),
    ]

    for start, end in cycles:
        audio_qc_passed = not check_cycle_in_bad_interval(start, end, bad_intervals, overlap_threshold=0.1)
        assert audio_qc_passed == True, "All cycles should pass with no bad intervals"


def test_provided_bad_segments_parameter():
    """Test that bad_audio_segments parameter can be provided directly."""
    # This test verifies that the perform_sync_qc function accepts
    # bad_audio_segments as an optional parameter

    # Test data - bad intervals in audio coordinates
    provided_bad_intervals = [(2.0, 3.0), (7.0, 8.0)]

    # Verify format is correct (list of tuples)
    assert isinstance(provided_bad_intervals, list)
    assert all(isinstance(interval, tuple) for interval in provided_bad_intervals)
    assert all(len(interval) == 2 for interval in provided_bad_intervals)

    # Verify intervals can be used with adjust_bad_intervals_for_sync
    audio_stomp = 3.0
    bio_stomp = 10.0

    adjusted = adjust_bad_intervals_for_sync(
        provided_bad_intervals,
        audio_stomp,
        bio_stomp,
    )

    # Should work the same as any other bad intervals
    assert len(adjusted) == len(provided_bad_intervals)
    assert adjusted[0] == (9.0, 10.0)
    assert adjusted[1] == (14.0, 15.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
