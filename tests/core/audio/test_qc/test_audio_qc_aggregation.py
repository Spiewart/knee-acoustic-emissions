"""Tests for audio_qc_fail and audio_qc_failures aggregate computation.

Validates the aggregation logic used in _persist_cycle_records() to compute
audio_qc_fail (bool) and audio_qc_failures (list[str]) from audio-stage
and cycle-stage QC sources:
  - Audio-stage: dropout, continuous (trimmed to cycle boundaries)
  - Cycle-stage: intermittent, periodic (from CycleQCResult)
"""

import pytest

from src.audio.raw_qc import trim_intervals_to_cycle
from src.orchestration.participant_processor import ManeuverProcessor


class TestTrimIntervalsToKnownCycleBoundaries:
    """Verify trim_intervals_to_cycle clips correctly for known scenarios."""

    def test_no_overlap(self):
        """Interval entirely outside cycle should be discarded."""
        result = trim_intervals_to_cycle([(5.0, 10.0)], 20.0, 25.0)
        assert result == []

    def test_full_overlap(self):
        """Interval entirely inside cycle should pass through."""
        result = trim_intervals_to_cycle([(21.0, 23.0)], 20.0, 25.0)
        assert result == [(21.0, 23.0)]

    def test_partial_overlap_start(self):
        """Interval starting before cycle should be clipped to cycle start."""
        result = trim_intervals_to_cycle([(18.0, 22.0)], 20.0, 25.0)
        assert result == [(20.0, 22.0)]

    def test_partial_overlap_end(self):
        """Interval ending after cycle should be clipped to cycle end."""
        result = trim_intervals_to_cycle([(23.0, 28.0)], 20.0, 25.0)
        assert result == [(23.0, 25.0)]

    def test_interval_spanning_entire_cycle(self):
        """Interval fully containing cycle should be clipped to cycle bounds."""
        result = trim_intervals_to_cycle([(10.0, 40.0)], 20.0, 25.0)
        assert result == [(20.0, 25.0)]

    def test_multiple_intervals_mixed(self):
        """Multiple intervals with varying overlap should be filtered/clipped."""
        intervals = [
            (1.0, 3.0),    # no overlap
            (18.0, 22.0),  # partial start
            (21.0, 23.0),  # full overlap
            (23.0, 28.0),  # partial end
            (30.0, 35.0),  # no overlap
        ]
        result = trim_intervals_to_cycle(intervals, 20.0, 25.0)
        assert result == [(20.0, 22.0), (21.0, 23.0), (23.0, 25.0)]

    def test_empty_intervals(self):
        result = trim_intervals_to_cycle([], 20.0, 25.0)
        assert result == []


class TestAudioQCFailAggregation:
    """Test the aggregation pattern used in _persist_cycle_records.

    Simulates the logic that checks four sources:
      1. Signal dropout (audio-stage, trimmed to cycle)
      2. Continuous artifact (audio-stage, trimmed to cycle)
      3. Intermittent artifact (cycle-stage, from CycleQCResult)
      4. Periodic artifact (cycle-stage, from CycleQCResult)
    """

    def _compute_audio_qc_failures(
        self,
        dropout_segments_per_ch: dict[str, list[float] | None],
        continuous_segments_per_ch: dict[str, list[float] | None],
        intermittent_intervals_per_ch: dict[str, list[tuple[float, float]]],
        periodic_noise_detected: bool,
        cycle_start: float,
        cycle_end: float,
    ) -> tuple[bool, list[str]]:
        """Reproduce the aggregation logic from _persist_cycle_records."""
        audio_qc_failures: list[str] = []

        # Audio-stage: dropout
        has_dropout = False
        for ch_num in range(1, 5):
            ch_key = f"ch{ch_num}"
            flat = dropout_segments_per_ch.get(ch_key)
            trimmed = trim_intervals_to_cycle(
                ManeuverProcessor._unflatten_intervals(flat),
                cycle_start, cycle_end,
            )
            if trimmed:
                has_dropout = True
        if has_dropout:
            audio_qc_failures.append("dropout")

        # Audio-stage: continuous
        has_continuous = False
        for ch_num in range(1, 5):
            ch_key = f"ch{ch_num}"
            flat = continuous_segments_per_ch.get(ch_key)
            trimmed = trim_intervals_to_cycle(
                ManeuverProcessor._unflatten_intervals(flat),
                cycle_start, cycle_end,
            )
            if trimmed:
                has_continuous = True
        if has_continuous:
            audio_qc_failures.append("continuous")

        # Cycle-stage: intermittent
        has_intermittent = any(
            bool(intermittent_intervals_per_ch.get(f"ch{ch}", []))
            for ch in range(1, 5)
        )
        if has_intermittent:
            audio_qc_failures.append("intermittent")

        # Cycle-stage: periodic
        if periodic_noise_detected:
            audio_qc_failures.append("periodic")

        audio_qc_fail = len(audio_qc_failures) > 0
        return audio_qc_fail, audio_qc_failures

    def test_no_failures(self):
        """Clean cycle should have audio_qc_fail=False, empty failures list."""
        fail, failures = self._compute_audio_qc_failures(
            dropout_segments_per_ch={"ch1": None, "ch2": None, "ch3": None, "ch4": None},
            continuous_segments_per_ch={"ch1": None, "ch2": None, "ch3": None, "ch4": None},
            intermittent_intervals_per_ch={"ch1": [], "ch2": [], "ch3": [], "ch4": []},
            periodic_noise_detected=False,
            cycle_start=10.0,
            cycle_end=11.2,
        )
        assert fail is False
        assert failures == []

    def test_dropout_only(self):
        """Cycle overlapping a dropout segment should show only 'dropout'."""
        fail, failures = self._compute_audio_qc_failures(
            dropout_segments_per_ch={
                "ch1": [9.0, 10.5],  # overlaps [10.0, 11.2]
                "ch2": None, "ch3": None, "ch4": None,
            },
            continuous_segments_per_ch={"ch1": None, "ch2": None, "ch3": None, "ch4": None},
            intermittent_intervals_per_ch={"ch1": [], "ch2": [], "ch3": [], "ch4": []},
            periodic_noise_detected=False,
            cycle_start=10.0,
            cycle_end=11.2,
        )
        assert fail is True
        assert failures == ["dropout"]

    def test_continuous_only(self):
        """Cycle overlapping continuous artifact should show only 'continuous'."""
        fail, failures = self._compute_audio_qc_failures(
            dropout_segments_per_ch={"ch1": None, "ch2": None, "ch3": None, "ch4": None},
            continuous_segments_per_ch={
                "ch1": None, "ch2": None,
                "ch3": [10.5, 11.0],  # overlaps cycle
                "ch4": None,
            },
            intermittent_intervals_per_ch={"ch1": [], "ch2": [], "ch3": [], "ch4": []},
            periodic_noise_detected=False,
            cycle_start=10.0,
            cycle_end=11.2,
        )
        assert fail is True
        assert failures == ["continuous"]

    def test_intermittent_only(self):
        """Cycle with intermittent artifacts should show only 'intermittent'."""
        fail, failures = self._compute_audio_qc_failures(
            dropout_segments_per_ch={"ch1": None, "ch2": None, "ch3": None, "ch4": None},
            continuous_segments_per_ch={"ch1": None, "ch2": None, "ch3": None, "ch4": None},
            intermittent_intervals_per_ch={
                "ch1": [], "ch2": [(10.3, 10.6)], "ch3": [], "ch4": [],
            },
            periodic_noise_detected=False,
            cycle_start=10.0,
            cycle_end=11.2,
        )
        assert fail is True
        assert failures == ["intermittent"]

    def test_periodic_only(self):
        """Cycle with periodic noise should show only 'periodic'."""
        fail, failures = self._compute_audio_qc_failures(
            dropout_segments_per_ch={"ch1": None, "ch2": None, "ch3": None, "ch4": None},
            continuous_segments_per_ch={"ch1": None, "ch2": None, "ch3": None, "ch4": None},
            intermittent_intervals_per_ch={"ch1": [], "ch2": [], "ch3": [], "ch4": []},
            periodic_noise_detected=True,
            cycle_start=10.0,
            cycle_end=11.2,
        )
        assert fail is True
        assert failures == ["periodic"]

    def test_all_four_failure_types(self):
        """Cycle with all four failure types should list all four."""
        fail, failures = self._compute_audio_qc_failures(
            dropout_segments_per_ch={
                "ch1": [10.0, 10.2], "ch2": None, "ch3": None, "ch4": None,
            },
            continuous_segments_per_ch={
                "ch1": None, "ch2": [10.5, 11.0], "ch3": None, "ch4": None,
            },
            intermittent_intervals_per_ch={
                "ch1": [], "ch2": [], "ch3": [(10.3, 10.5)], "ch4": [],
            },
            periodic_noise_detected=True,
            cycle_start=10.0,
            cycle_end=11.2,
        )
        assert fail is True
        assert failures == ["dropout", "continuous", "intermittent", "periodic"]

    def test_dropout_outside_cycle_not_counted(self):
        """Dropout segment outside cycle boundaries should not contribute."""
        fail, failures = self._compute_audio_qc_failures(
            dropout_segments_per_ch={
                "ch1": [5.0, 8.0],  # entirely before cycle [10, 11.2]
                "ch2": None, "ch3": None, "ch4": None,
            },
            continuous_segments_per_ch={"ch1": None, "ch2": None, "ch3": None, "ch4": None},
            intermittent_intervals_per_ch={"ch1": [], "ch2": [], "ch3": [], "ch4": []},
            periodic_noise_detected=False,
            cycle_start=10.0,
            cycle_end=11.2,
        )
        assert fail is False
        assert failures == []

    def test_continuous_on_one_channel_sufficient(self):
        """Continuous artifact on any single channel is sufficient to flag."""
        fail, failures = self._compute_audio_qc_failures(
            dropout_segments_per_ch={"ch1": None, "ch2": None, "ch3": None, "ch4": None},
            continuous_segments_per_ch={
                "ch1": None, "ch2": None, "ch3": None,
                "ch4": [10.0, 12.0],  # overlaps cycle
            },
            intermittent_intervals_per_ch={"ch1": [], "ch2": [], "ch3": [], "ch4": []},
            periodic_noise_detected=False,
            cycle_start=10.0,
            cycle_end=11.2,
        )
        assert fail is True
        assert failures == ["continuous"]

    def test_failure_order_is_deterministic(self):
        """Failure types should appear in order: dropout, continuous, intermittent, periodic."""
        _, failures = self._compute_audio_qc_failures(
            dropout_segments_per_ch={
                "ch1": [10.0, 10.2], "ch2": None, "ch3": None, "ch4": None,
            },
            continuous_segments_per_ch={
                "ch1": None, "ch2": [10.5, 11.0], "ch3": None, "ch4": None,
            },
            intermittent_intervals_per_ch={
                "ch1": [], "ch2": [], "ch3": [(10.3, 10.5)], "ch4": [],
            },
            periodic_noise_detected=True,
            cycle_start=10.0,
            cycle_end=11.2,
        )
        assert failures == ["dropout", "continuous", "intermittent", "periodic"]


class TestIntermittentFailFlagComputation:
    """Test intermittent fail flags per channel mirror interval presence."""

    def test_ch_has_intervals_means_fail_true(self):
        intervals_ch2 = [(10.3, 10.6)]
        assert bool(intervals_ch2) is True

    def test_ch_has_no_intervals_means_fail_false(self):
        intervals_ch1: list[tuple[float, float]] = []
        assert bool(intervals_ch1) is False

    def test_overall_intermittent_from_any_channel(self):
        """Overall intermittent fail is True if any channel has intervals."""
        per_ch = {
            "ch1": [],
            "ch2": [(10.3, 10.6)],
            "ch3": [],
            "ch4": [],
        }
        has_intermittent = any(bool(per_ch[f"ch{ch}"]) for ch in range(1, 5))
        assert has_intermittent is True


class TestFlattenForDBPersistence:
    """Test the flatten pattern used to store interval timestamps in DB."""

    def test_flatten_intermittent_timestamps(self):
        """Intermittent intervals from all channels flattened for 'audio_artifact_timestamps'."""
        ch1 = [(0.5, 0.8)]
        ch2: list[tuple[float, float]] = []
        ch3 = [(1.0, 1.5), (2.0, 2.3)]
        ch4: list[tuple[float, float]] = []
        all_intervals = ch1 + ch2 + ch3 + ch4
        flat = ManeuverProcessor._flatten_intervals(all_intervals)
        assert flat == [0.5, 0.8, 1.0, 1.5, 2.0, 2.3]

    def test_flatten_empty_yields_empty(self):
        """No intervals on any channel produces empty flat list."""
        all_intervals: list[tuple[float, float]] = [] + [] + [] + []
        flat = ManeuverProcessor._flatten_intervals(all_intervals)
        assert flat == []

    def test_flatten_or_none_pattern(self):
        """The `... or None` pattern converts empty list to None for DB storage."""
        flat = ManeuverProcessor._flatten_intervals([])
        result = flat or None
        assert result is None

        flat2 = ManeuverProcessor._flatten_intervals([(1.0, 2.0)])
        result2 = flat2 or None
        assert result2 == [1.0, 2.0]
