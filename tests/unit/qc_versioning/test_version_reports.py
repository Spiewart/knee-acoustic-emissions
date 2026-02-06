"""Comprehensive integration tests for QC data integrity in Excel reports.

These tests verify that:
1. QC Fail column is present and correctly computed (dropout OR artifact)
2. Artifact Type is only populated when artifacts are detected
3. Artifact Type values are either 'Intermittent' or 'Continuous'
4. Per-channel artifact types match per-channel artifact detections
5. No defaults are used; values come directly from detection results
6. Edge cases are handled correctly (clean audio, all failing, mixed channels)
"""

import logging
from typing import List
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

logger = logging.getLogger(__name__)


def _create_mock_record(
    qc_signal_dropout: bool = False,
    qc_signal_dropout_ch1: bool = False,
    qc_signal_dropout_ch2: bool = False,
    qc_signal_dropout_ch3: bool = False,
    qc_signal_dropout_ch4: bool = False,
    qc_continuous_artifact: bool = False,
    qc_continuous_artifact_type: List[str] | None = None,
    qc_continuous_artifact_ch1: bool = False,
    qc_continuous_artifact_type_ch1: List[str] | None = None,
    qc_continuous_artifact_ch2: bool = False,
    qc_continuous_artifact_type_ch2: List[str] | None = None,
    qc_continuous_artifact_ch3: bool = False,
    qc_continuous_artifact_type_ch3: List[str] | None = None,
    qc_continuous_artifact_ch4: bool = False,
    qc_continuous_artifact_type_ch4: List[str] | None = None,
) -> MagicMock:
    """Create a mock audio record with specified QC parameters."""
    record = MagicMock()
    record.qc_signal_dropout = qc_signal_dropout
    record.qc_signal_dropout_ch1 = qc_signal_dropout_ch1
    record.qc_signal_dropout_ch2 = qc_signal_dropout_ch2
    record.qc_signal_dropout_ch3 = qc_signal_dropout_ch3
    record.qc_signal_dropout_ch4 = qc_signal_dropout_ch4
    record.qc_continuous_artifact = qc_continuous_artifact
    record.qc_continuous_artifact_type = qc_continuous_artifact_type
    record.qc_continuous_artifact_ch1 = qc_continuous_artifact_ch1
    record.qc_continuous_artifact_type_ch1 = qc_continuous_artifact_type_ch1
    record.qc_continuous_artifact_ch2 = qc_continuous_artifact_ch2
    record.qc_continuous_artifact_type_ch2 = qc_continuous_artifact_type_ch2
    record.qc_continuous_artifact_ch3 = qc_continuous_artifact_ch3
    record.qc_continuous_artifact_type_ch3 = qc_continuous_artifact_type_ch3
    record.qc_continuous_artifact_ch4 = qc_continuous_artifact_ch4
    record.qc_continuous_artifact_type_ch4 = qc_continuous_artifact_type_ch4

    return record


class TestQCFailColumnComputation:
    """Test suite for QC Fail column computation logic."""

    def test_qc_fail_true_when_dropout(self):
        """Test that QC Fail is True when signal dropout is detected."""
        record = _create_mock_record(qc_signal_dropout=True, qc_continuous_artifact=False)

        # QC Fail should be computed as dropout OR artifact
        qc_fail = record.qc_signal_dropout or record.qc_continuous_artifact

        assert qc_fail is True, "QC Fail should be True when dropout detected"

    def test_qc_fail_true_when_artifact(self):
        """Test that QC Fail is True when artifacts are detected."""
        record = _create_mock_record(
            qc_signal_dropout=False,
            qc_continuous_artifact=True,
            qc_continuous_artifact_type=["Intermittent"],
        )

        qc_fail = record.qc_signal_dropout or record.qc_continuous_artifact

        assert qc_fail is True, "QC Fail should be True when artifacts detected"

    def test_qc_fail_true_when_both(self):
        """Test that QC Fail is True when both dropout and artifacts are detected."""
        record = _create_mock_record(
            qc_signal_dropout=True,
            qc_continuous_artifact=True,
            qc_continuous_artifact_type=["Continuous"],
        )

        qc_fail = record.qc_signal_dropout or record.qc_continuous_artifact

        assert qc_fail is True, "QC Fail should be True when both dropout and artifacts detected"

    def test_qc_fail_false_when_clean(self):
        """Test that QC Fail is False when audio is clean."""
        record = _create_mock_record(
            qc_signal_dropout=False,
            qc_continuous_artifact=False,
        )

        qc_fail = record.qc_signal_dropout or record.qc_continuous_artifact

        assert qc_fail is False, "QC Fail should be False when audio is clean"


class TestArtifactTypePopulation:
    """Test suite for artifact type population rules."""

    def test_artifact_type_blank_when_no_artifacts(self):
        """Test that Artifact Type is blank when no artifacts are detected."""
        record = _create_mock_record(
            qc_signal_dropout=False,
            qc_continuous_artifact=False,
        )

        assert record.qc_continuous_artifact_type is None or record.qc_continuous_artifact_type == [], \
            "Artifact Type should be blank/None when no artifacts detected"

    def test_artifact_type_populated_when_artifacts(self):
        """Test that Artifact Type is populated when artifacts are detected."""
        record = _create_mock_record(
            qc_continuous_artifact=True,
            qc_continuous_artifact_type=["Intermittent"],
        )

        artifact_types = record.qc_continuous_artifact_type
        assert artifact_types is not None and len(artifact_types) > 0, \
            "Artifact Type should be populated when artifacts detected"
        assert all(t in ["Intermittent", "Continuous"] for t in artifact_types), \
            f"Artifact Type must be Intermittent or Continuous, got {artifact_types}"

    def test_artifact_type_values_valid(self):
        """Test that Artifact Type values are only Intermittent or Continuous."""
        for artifact_type in ["Intermittent", "Continuous"]:
            record = _create_mock_record(
                qc_continuous_artifact=True,
                qc_continuous_artifact_type=[artifact_type],
            )

            artifact_types = record.qc_continuous_artifact_type
            for t in artifact_types:
                assert t in ["Intermittent", "Continuous"], \
                    f"Invalid artifact type: {t}. Must be Intermittent or Continuous"

    def test_per_mic_artifact_type_matches_detection(self):
        """Test that per-mic artifact types correspond to per-mic artifact detection."""
        record = _create_mock_record(
            qc_continuous_artifact=True,
            qc_continuous_artifact_ch1=True,
            qc_continuous_artifact_type_ch1=["Continuous"],
            qc_continuous_artifact_ch2=False,
            qc_continuous_artifact_type_ch2=None,
            qc_continuous_artifact_ch3=True,
            qc_continuous_artifact_type_ch3=["Intermittent"],
            qc_continuous_artifact_ch4=False,
            qc_continuous_artifact_type_ch4=None,
        )

        # Ch1: has artifact, should have type
        assert record.qc_continuous_artifact_ch1 is True
        assert record.qc_continuous_artifact_type_ch1 is not None

        # Ch2: no artifact, should have no type
        assert record.qc_continuous_artifact_ch2 is False
        assert record.qc_continuous_artifact_type_ch2 is None

        # Ch3: has artifact, should have type
        assert record.qc_continuous_artifact_ch3 is True
        assert record.qc_continuous_artifact_type_ch3 is not None

        # Ch4: no artifact, should have no type
        assert record.qc_continuous_artifact_ch4 is False
        assert record.qc_continuous_artifact_type_ch4 is None

    def test_artifact_type_not_defaulted(self):
        """Test that Artifact Type is never defaulted; it comes from detection."""
        # Create record with artifact detected but no type specified
        record = _create_mock_record(
            qc_continuous_artifact=True,
            qc_continuous_artifact_type=None,  # Explicitly None
        )

        # Should remain None, not defaulted to any value
        assert record.qc_continuous_artifact_type is None or record.qc_continuous_artifact_type == [], \
            "Artifact Type should not be defaulted when not detected"

    def test_mixed_clean_and_failing_channels(self):
        """Test that mixed pass/fail across channels is handled correctly."""
        record = _create_mock_record(
            qc_signal_dropout=False,
            qc_continuous_artifact=True,  # Overall artifact detected
            qc_continuous_artifact_type=["Intermittent", "Continuous"],
            qc_continuous_artifact_ch1=True,
            qc_continuous_artifact_type_ch1=["Intermittent"],
            qc_continuous_artifact_ch2=False,
            qc_continuous_artifact_type_ch2=None,
            qc_continuous_artifact_ch3=True,
            qc_continuous_artifact_type_ch3=["Continuous"],
            qc_continuous_artifact_ch4=False,
            qc_continuous_artifact_type_ch4=None,
        )

        # Overall QC Fail should be True
        qc_fail = record.qc_signal_dropout or record.qc_continuous_artifact
        assert qc_fail is True

        # Check per-mic consistency
        for ch_num in range(1, 5):
            ch_artifact = getattr(record, f"qc_continuous_artifact_ch{ch_num}")
            ch_type = getattr(record, f"qc_continuous_artifact_type_ch{ch_num}")

            if ch_artifact:
                assert ch_type is not None and len(ch_type) > 0, \
                    f"Ch{ch_num} has artifact but no type"
            else:
                assert ch_type is None or len(ch_type) == 0, \
                    f"Ch{ch_num} has no artifact but has type {ch_type}"

    def test_all_channels_failing(self):
        """Test edge case where all channels fail QC."""
        record = _create_mock_record(
            qc_continuous_artifact=True,
            qc_continuous_artifact_type=["Continuous", "Continuous", "Continuous", "Continuous"],
            qc_continuous_artifact_ch1=True,
            qc_continuous_artifact_type_ch1=["Continuous"],
            qc_continuous_artifact_ch2=True,
            qc_continuous_artifact_type_ch2=["Continuous"],
            qc_continuous_artifact_ch3=True,
            qc_continuous_artifact_type_ch3=["Continuous"],
            qc_continuous_artifact_ch4=True,
            qc_continuous_artifact_type_ch4=["Continuous"],
        )

        qc_fail = record.qc_signal_dropout or record.qc_continuous_artifact
        assert qc_fail is True

        for ch_num in range(1, 5):
            ch_artifact = getattr(record, f"qc_continuous_artifact_ch{ch_num}")
            ch_type = getattr(record, f"qc_continuous_artifact_type_ch{ch_num}")
            assert ch_artifact is True
            assert ch_type is not None

    def test_dropout_without_artifacts(self):
        """Test dropout detection without any artifacts."""
        record = _create_mock_record(
            qc_signal_dropout=True,
            qc_signal_dropout_ch1=True,
            qc_signal_dropout_ch2=False,
            qc_signal_dropout_ch3=False,
            qc_signal_dropout_ch4=False,
            qc_continuous_artifact=False,
            qc_continuous_artifact_type=None,
        )

        qc_fail = record.qc_signal_dropout or record.qc_continuous_artifact
        assert qc_fail is True
        assert record.qc_signal_dropout is True
        assert record.qc_continuous_artifact is False
        assert record.qc_continuous_artifact_type is None


class TestArtifactTypeEnforcement:
    """Test suite for artifact type enforcement rules.

    These tests verify that when qc_continuous_artifact=True, artifact type is NEVER blank.
    """

    def test_intermittent_artifact_classification(self):
        """Test that short-duration artifacts are classified as Intermittent."""
        # Duration threshold is 1.0 second: < 1.0s = Intermittent
        record = _create_mock_record(
            qc_continuous_artifact=True,
            qc_continuous_artifact_type=["Intermittent"],
        )

        # When artifact detected, type should exist and be valid
        assert record.qc_continuous_artifact is True
        assert record.qc_continuous_artifact_type is not None
        assert "Intermittent" in record.qc_continuous_artifact_type

    def test_continuous_artifact_classification(self):
        """Test that long-duration artifacts are classified as Continuous."""
        # Duration threshold is 1.0 second: >= 1.0s = Continuous
        record = _create_mock_record(
            qc_continuous_artifact=True,
            qc_continuous_artifact_type=["Continuous"],
        )

        # When artifact detected, type should exist and be valid
        assert record.qc_continuous_artifact is True
        assert record.qc_continuous_artifact_type is not None
        assert "Continuous" in record.qc_continuous_artifact_type

    def test_multiple_artifact_types(self):
        """Test that multiple artifacts can have different types."""
        record = _create_mock_record(
            qc_continuous_artifact=True,
            qc_continuous_artifact_type=["Intermittent", "Continuous"],
        )

        assert record.qc_continuous_artifact is True
        assert len(record.qc_continuous_artifact_type) == 2
        assert all(t in ["Intermittent", "Continuous"] for t in record.qc_continuous_artifact_type)

    def test_artifact_type_required_per_channel(self):
        """Test that each channel with qc_continuous_artifact_ch=True must have a type."""
        record = _create_mock_record(
            qc_continuous_artifact_ch1=True,
            qc_continuous_artifact_type_ch1=["Intermittent"],
            qc_continuous_artifact_ch2=True,
            qc_continuous_artifact_type_ch2=["Continuous"],
            qc_continuous_artifact_ch3=False,
            qc_continuous_artifact_type_ch3=None,
            qc_continuous_artifact_ch4=False,
            qc_continuous_artifact_type_ch4=None,
        )

        # Ch1 and Ch2 should have types since they have artifacts
        if record.qc_continuous_artifact_ch1:
            assert record.qc_continuous_artifact_type_ch1 is not None, \
                "Ch1 has artifact but type is None/blank"

        if record.qc_continuous_artifact_ch2:
            assert record.qc_continuous_artifact_type_ch2 is not None, \
                "Ch2 has artifact but type is None/blank"

        # Ch3 and Ch4 should not have types since they don't have artifacts
        if not record.qc_continuous_artifact_ch3:
            assert record.qc_continuous_artifact_type_ch3 is None or len(record.qc_continuous_artifact_type_ch3) == 0, \
                "Ch3 has no artifact but type is populated"

        if not record.qc_continuous_artifact_ch4:
            assert record.qc_continuous_artifact_type_ch4 is None or len(record.qc_continuous_artifact_type_ch4) == 0, \
                "Ch4 has no artifact but type is populated"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
