"""Tests for SynchronizationRecord new fields."""

from datetime import datetime

import pandas as pd
import pytest

from src.orchestration.processing_log import (
    SynchronizationRecord,
    create_sync_record_from_data,
)


def test_synchronization_record_new_fields():
    """Test that SynchronizationRecord has new fields for biomechanics-guided detection."""
    record = SynchronizationRecord(
        sync_file_name="test_sync.pkl",
        audio_stomp_method="biomechanics-guided",
        selected_time=2.5,
        contra_selected_time=3.8,
    )

    assert record.audio_stomp_method == "biomechanics-guided"
    assert record.selected_time == 2.5
    assert record.contra_selected_time == 3.8


def test_synchronization_record_to_dict_includes_new_fields():
    """Test that to_dict includes new fields."""
    record = SynchronizationRecord(
        sync_file_name="test_sync.pkl",
        audio_stomp_method="biomechanics-guided",
        selected_time=2.5,
        contra_selected_time=3.8,
    )

    record_dict = record.to_dict()

    assert "Detection Method" in record_dict
    assert record_dict["Detection Method"] == "biomechanics-guided"
    assert "Selected Time (s)" in record_dict
    assert record_dict["Selected Time (s)"] == 2.5
    assert "Contra Selected Time (s)" in record_dict
    assert record_dict["Contra Selected Time (s)"] == 3.8


def test_create_sync_record_populates_new_fields_from_detection_results():
    """Test that create_sync_record_from_data extracts new fields from detection_results."""
    # Create minimal synced dataframe
    synced_df = pd.DataFrame({
        'tt': pd.to_timedelta([0, 1, 2], unit='s'),
        'ch1': [0.1, 0.2, 0.3],
        'ch2': [0.1, 0.2, 0.3],
        'ch3': [0.1, 0.2, 0.3],
        'ch4': [0.1, 0.2, 0.3],
    })

    # Detection results with new fields
    detection_results = {
        'consensus_time': 2.0,
        'rms_time': 2.1,
        'rms_energy': 500.0,
        'onset_time': 1.9,
        'onset_magnitude': 100.0,
        'freq_time': 2.0,
        'freq_energy': 300.0,
        'audio_stomp_method': 'biomechanics-guided',
        'selected_time': 2.5,
        'contra_selected_time': 3.8,
    }

    record = create_sync_record_from_data(
        sync_file_name="test_sync.pkl",
        synced_df=synced_df,
        audio_stomp_time=2.5,
        bio_left_stomp_time=3.8,
        bio_right_stomp_time=2.5,
        knee_side="right",
        detection_results=detection_results,
    )

    # Verify new fields are populated
    assert record.audio_stomp_method == "biomechanics-guided"
    assert record.selected_time == 2.5
    assert record.contra_selected_time == 3.8

    # Verify standard fields still work
    assert record.consensus_time == 2.0
    assert record.rms_time == 2.1
    assert record.onset_time == 1.9
    assert record.freq_time == 2.0


def test_create_sync_record_handles_consensus_method():
    """Test that create_sync_record_from_data handles consensus method correctly."""
    synced_df = pd.DataFrame({
        'tt': pd.to_timedelta([0, 1, 2], unit='s'),
        'ch1': [0.1, 0.2, 0.3],
    })

    # Detection results with consensus method
    detection_results = {
        'consensus_time': 2.0,
        'rms_time': 2.1,
        'rms_energy': 500.0,
        'onset_time': 1.9,
        'onset_magnitude': 100.0,
        'freq_time': 2.0,
        'freq_energy': 300.0,
        'audio_stomp_method': 'consensus',
        'selected_time': None,
        'contra_selected_time': None,
    }

    record = create_sync_record_from_data(
        sync_file_name="test_sync.pkl",
        synced_df=synced_df,
        audio_stomp_time=2.0,
        bio_left_stomp_time=3.5,
        bio_right_stomp_time=2.0,
        knee_side="right",
        detection_results=detection_results,
    )

    # Verify method is consensus
    assert record.audio_stomp_method == "consensus"
    # Verify times are None for consensus
    assert record.selected_time is None
    assert record.contra_selected_time is None


def test_create_sync_record_without_detection_results():
    """Test backward compatibility when detection_results not provided."""
    synced_df = pd.DataFrame({
        'tt': pd.to_timedelta([0, 1, 2], unit='s'),
        'ch1': [0.1, 0.2, 0.3],
    })

    record = create_sync_record_from_data(
        sync_file_name="test_sync.pkl",
        synced_df=synced_df,
        audio_stomp_time=2.0,
        bio_left_stomp_time=3.5,
        bio_right_stomp_time=2.0,
        knee_side="right",
        detection_results=None,  # No detection results
    )

    # New fields should be None
    assert record.audio_stomp_method is None
    assert record.selected_time is None
    assert record.contra_selected_time is None

    # Standard fields should still work
    assert record.audio_stomp_time == 2.0
    assert record.bio_left_stomp_time == 3.5
    assert record.bio_right_stomp_time == 2.0


def test_synchronization_record_fields_optional():
    """Test that new fields are optional (default to None)."""
    record = SynchronizationRecord(
        sync_file_name="test_sync.pkl",
    )

    # New fields should default to None
    assert record.audio_stomp_method is None
    assert record.selected_time is None
    assert record.contra_selected_time is None
