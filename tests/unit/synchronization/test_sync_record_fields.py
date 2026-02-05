"""Tests for Synchronization class new fields."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.metadata import Synchronization
from src.orchestration.processing_log import create_sync_record_from_data


def _create_test_record(**kwargs):
    """Helper to create a test Synchronization record with required fields."""
    defaults = {
        "study": "AOA",
        "study_id": 1001,
        "audio_processing_id": 1,
        "biomechanics_import_id": 1,
        "pass_number": 1,
        "speed": "comfortable",  # Changed from "normal"
        "audio_sync_time": 0.0,
        "bio_left_sync_time": 0.0,
        "sync_offset": 0.0,
        "aligned_audio_sync_time": 0.0,
        "aligned_biomechanics_sync_time": 0.0,
        "sync_method": "consensus",
        "consensus_time": 0.0,
        "rms_time": 0.0,
        "onset_time": 0.0,
        "freq_time": 0.0,
        "sync_file_name": "test_sync.pkl",
        "processing_date": datetime.now(),
        "processing_status": "success",
        "sync_duration": 3.0,
        "total_cycles_extracted": 0,
        "clean_cycles": 0,
        "outlier_cycles": 0,
        "mean_cycle_duration_s": 0.0,
        "median_cycle_duration_s": 0.0,
        "min_cycle_duration_s": 0.0,
        "max_cycle_duration_s": 0.0,
    }
    defaults.update(kwargs)
    return Synchronization(**defaults)


def test_synchronization_has_biomech_guided_fields():
    """Test that Synchronization has new fields for biomechanics-guided detection."""
    record = _create_test_record(
        audio_stomp_method="biomechanics-guided",
        selected_audio_sync_time=2.5,
        contra_selected_audio_sync_time=3.8,
    )

    assert record.audio_stomp_method == "biomechanics-guided"
    assert record.selected_audio_sync_time == 2.5
    assert record.contra_selected_audio_sync_time == 3.8

def test_synchronization_fields_include_new_values():
    """Test that new fields are populated on the model."""
    record = _create_test_record(
        audio_stomp_method="biomechanics-guided",
        selected_audio_sync_time=2.5,
        contra_selected_audio_sync_time=3.8,
    )

    assert record.audio_stomp_method == "biomechanics-guided"
    assert record.selected_audio_sync_time == 2.5
    assert record.contra_selected_audio_sync_time == 3.8


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
        pass_number=1,
        speed="normal",
        audio_stomp_time=2.0,
        detection_results=detection_results,
    )

    # Verify new fields are populated
    assert record.audio_stomp_method == "biomechanics-guided"
    assert record.selected_audio_sync_time == 2.5
    assert record.contra_selected_audio_sync_time == 3.8



def test_synchronization_biomech_guided_can_be_none():
    """Test that biomechanics-guided fields can be None (for consensus detection)."""
    record = _create_test_record()

    # These should default to None
    assert record.audio_stomp_method is None
    assert record.selected_audio_sync_time is None
    assert record.contra_selected_audio_sync_time is None



def test_create_sync_record_consensus_without_biomech_guided():
    """Test creating a sync record with consensus detection (no biomech-guided fields)."""
    synced_df = pd.DataFrame({
        'tt': pd.to_timedelta([0, 1, 2], unit='s'),
        'ch1': [0.1, 0.2, 0.3],
        'ch2': [0.1, 0.2, 0.3],
        'ch3': [0.1, 0.2, 0.3],
        'ch4': [0.1, 0.2, 0.3],
    })

    # Detection results without biomech-guided fields (pure consensus)
    detection_results = {
        'consensus_time': 2.0,
        'consensus_methods': ['rms', 'onset'],
        'rms_time': 2.1,
        'onset_time': 1.9,
    }

    record = create_sync_record_from_data(
        sync_file_name="test_sync.pkl",
        synced_df=synced_df,
        pass_number=1,
        speed="normal",
        audio_stomp_time=2.0,
        detection_results=detection_results,
    )

    # Biomech-guided fields should be None
    assert record.audio_stomp_method is None
    assert record.selected_audio_sync_time is None
    assert record.contra_selected_audio_sync_time is None

    # But consensus fields should be populated
    assert record.consensus_time == 2.0
    assert record.rms_time == 2.1



def test_synchronization_consensus_method_agreement_span_calculation():
    """Test calculation of method_agreement_span in create_sync_record_from_data."""
    synced_df = pd.DataFrame({
        'tt': pd.to_timedelta([0, 1, 2, 3], unit='s'),
        'ch1': [0.1, 0.2, 0.3, 0.4],
        'ch2': [0.1, 0.2, 0.3, 0.4],
        'ch3': [0.1, 0.2, 0.3, 0.4],
        'ch4': [0.1, 0.2, 0.3, 0.4],
    })

    # All three methods agree within 0.2s
    detection_results = {
        'consensus_time': 2.0,
        'consensus_methods': ['rms', 'onset', 'freq'],
        'rms_time': 1.9,
        'onset_time': 2.0,
        'freq_time': 2.1,
    }

    record = create_sync_record_from_data(
        sync_file_name="test_sync.pkl",
        synced_df=synced_df,
        pass_number=1,
        speed="normal",
        audio_stomp_time=2.0,
        detection_results=detection_results,
    )

    # Agreement span should be max - min of the three times
    assert record.method_agreement_span == pytest.approx(0.2, abs=0.01)
    # Verify consensus_methods is stored
    assert record.consensus_methods == "rms, onset, freq"



def test_synchronization_agreement_span_with_partial_consensus():
    """Test agreement span when only some methods agree (not all 3)."""
    synced_df = pd.DataFrame({
        'tt': pd.to_timedelta([0, 1, 2], unit='s'),
        'ch1': [0.1, 0.2, 0.3],
        'ch2': [0.1, 0.2, 0.3],
        'ch3': [0.1, 0.2, 0.3],
        'ch4': [0.1, 0.2, 0.3],
    })

    # Only RMS and onset in consensus (freq disagreed)
    detection_results = {
        'consensus_time': 2.0,
        'consensus_methods': ['rms', 'onset'],
        'rms_time': 1.95,
        'onset_time': 2.05,
        'freq_time': 5.0,  # Not in consensus
    }

    record = create_sync_record_from_data(
        sync_file_name="test_sync.pkl",
        synced_df=synced_df,
        pass_number=1,
        speed="normal",
        audio_stomp_time=2.0,
        detection_results=detection_results,
    )

    # Agreement span should only consider methods in consensus
    assert record.method_agreement_span == pytest.approx(0.1, abs=0.01)
    # Freq time is still recorded but not in consensus_methods
    assert record.freq_time == 5.0
    assert record.consensus_methods == "rms, onset"
