"""Tests for per-channel QC fail segments merge logic.

Validates that qc_fail_segments (overall and per-channel) are correctly
merged from signal dropout + continuous artifact + intermittent artifact
sources in participant_processor.py's audio QC data assembly.
"""

from src.audio.raw_qc import merge_bad_intervals


class TestMergeBadIntervalsPerChannel:
    """Test that merge_bad_intervals correctly combines interval sources."""

    def test_empty_sources_produce_empty_result(self):
        result = merge_bad_intervals([], [])
        assert result == []

    def test_single_source_passes_through(self):
        result = merge_bad_intervals([(1.0, 2.0)], [])
        assert result == [(1.0, 2.0)]

    def test_two_sources_merged(self):
        dropout = [(1.0, 3.0)]
        artifact = [(5.0, 7.0)]
        result = merge_bad_intervals(dropout, artifact)
        assert result == [(1.0, 3.0), (5.0, 7.0)]

    def test_overlapping_intervals_merged(self):
        dropout = [(1.0, 4.0)]
        artifact = [(3.0, 6.0)]
        result = merge_bad_intervals(dropout, artifact)
        assert result == [(1.0, 6.0)]

    def test_adjacent_intervals_within_gap_merged(self):
        """Intervals within merge_gap_s (default 0.5s) should merge."""
        dropout = [(1.0, 2.0)]
        artifact = [(2.3, 4.0)]  # 0.3s gap < 0.5s default
        result = merge_bad_intervals(dropout, artifact)
        assert result == [(1.0, 4.0)]

    def test_three_sources_via_concatenation(self):
        """Simulates the per-channel merge pattern from participant_processor."""
        dropout = [(1.0, 2.0)]
        continuous = [(5.0, 7.0)]
        intermittent = [(10.0, 11.0)]
        all_sources = dropout + continuous + intermittent
        result = merge_bad_intervals(all_sources, [])
        assert result == [(1.0, 2.0), (5.0, 7.0), (10.0, 11.0)]


class TestPerChannelQCDataAssembly:
    """Test that qc_data dict is correctly assembled for per-channel merge.

    Simulates the logic from participant_processor.py process_bin_stage().
    """

    def test_per_channel_merge_with_dropout_only(self):
        """Channel with only dropout gets qc_fail_segments = dropout segments."""
        dropout_per_mic = {"ch1": [(2.0, 3.0)], "ch2": [], "ch3": [], "ch4": []}
        continuous_per_mic = {}
        artifact_per_mic = {}

        for ch_num in range(1, 5):
            ch_name = f"ch{ch_num}"
            ch_fail_sources = []
            if dropout_per_mic.get(ch_name):
                ch_fail_sources.extend(dropout_per_mic[ch_name])
            if continuous_per_mic.get(ch_name):
                ch_fail_sources.extend(continuous_per_mic[ch_name])
            if artifact_per_mic.get(ch_name):
                ch_fail_sources.extend(artifact_per_mic[ch_name])

            merged = merge_bad_intervals(ch_fail_sources, []) if ch_fail_sources else []

            if ch_name == "ch1":
                assert merged == [(2.0, 3.0)]
            else:
                assert merged == []

    def test_per_channel_merge_with_all_three_sources(self):
        """Channel with all three sources gets merged qc_fail_segments."""
        dropout_per_mic = {"ch2": [(1.0, 2.0)]}
        continuous_per_mic = {"ch2": [(3.0, 5.0)]}
        artifact_per_mic = {"ch2": [(8.0, 9.0)]}

        ch_fail_sources = (
            dropout_per_mic.get("ch2", []) + continuous_per_mic.get("ch2", []) + artifact_per_mic.get("ch2", [])
        )
        merged = merge_bad_intervals(ch_fail_sources, [])
        assert merged == [(1.0, 2.0), (3.0, 5.0), (8.0, 9.0)]

    def test_overall_merge_includes_continuous_artifacts(self):
        """Overall qc_fail_segments must include all three failure types."""
        dropout_intervals = [(1.0, 2.0)]
        artifact_intervals = [(5.0, 6.0)]
        continuous_intervals = [(10.0, 12.0)]

        all_fail = dropout_intervals + artifact_intervals + continuous_intervals
        result = merge_bad_intervals(all_fail, [])
        assert len(result) == 3
        assert (10.0, 12.0) in result  # continuous must be present

    def test_overall_merge_without_continuous_still_works(self):
        """Overall merge works when continuous is empty."""
        dropout_intervals = [(1.0, 2.0)]
        artifact_intervals = [(5.0, 6.0)]
        continuous_intervals = []

        all_fail = dropout_intervals + artifact_intervals + continuous_intervals
        result = merge_bad_intervals(all_fail, []) if all_fail else []
        assert len(result) == 2


class TestQCFailSegmentsNoDefaults:
    """Test that AudioProcessing requires explicit QC field values."""

    def test_missing_qc_fields_raises_validation_error(self):
        """AudioProcessing without QC fields raises ValidationError."""
        from datetime import datetime

        import pytest

        from src.metadata import AudioProcessing

        with pytest.raises(Exception) as exc_info:
            AudioProcessing(
                study="AOA",
                study_id=1001,
                audio_file_name="test.bin",
                device_serial="DEV001",
                firmware_version=1,
                file_time=datetime(2024, 1, 1),
                file_size_mb=100.0,
                recording_date=datetime(2024, 1, 1),
                recording_time=datetime(2024, 1, 1),
                knee="left",
                maneuver="walk",
                num_channels=4,
                sample_rate=46875.0,
                mic_1_position="IPM",
                mic_2_position="IPL",
                mic_3_position="SPM",
                mic_4_position="SPL",
                processing_date=datetime(2024, 1, 1),
                processing_status="success",
                # Intentionally omitting QC fields
            )
        assert "Field required" in str(exc_info.value)

    def test_missing_cycle_qc_fields_raises_validation_error(self):
        """MovementCycle without audio QC fields raises ValidationError."""
        from datetime import datetime

        import pytest

        from src.metadata import MovementCycle

        with pytest.raises(Exception) as exc_info:
            MovementCycle(
                study="AOA",
                study_id=1001,
                audio_processing_id=1,
                cycle_file="test_cycle.pkl",
                cycle_index=0,
                is_outlier=False,
                start_time_s=0.0,
                end_time_s=1.0,
                duration_s=1.0,
                start_time=datetime(2024, 1, 1),
                end_time=datetime(2024, 1, 1),
                biomechanics_qc_fail=False,
                sync_qc_fail=False,
                # Intentionally omitting audio_qc_fail and artifact fields
            )
        assert "Field required" in str(exc_info.value)
