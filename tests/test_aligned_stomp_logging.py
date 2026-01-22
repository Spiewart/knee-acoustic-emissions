"""Tests for aligned stomp times and offset calculations in synchronization logging."""

import pandas as pd
import pytest
from pathlib import Path
from src.orchestration.processing_log import (
    create_sync_record_from_data,
)


def test_stomp_offset_calculation():
    """Test that stomp offset is correctly calculated."""
    # Create a simple synced dataframe
    synced_df = pd.DataFrame({
        "tt": pd.to_timedelta([0, 1, 2], unit="s"),
        "ch1": [0.1, 0.2, 0.3],
    })

    # Audio stomp at 5s, bio stomp at 10s
    record = create_sync_record_from_data(
        sync_file_name="test_sync",
        synced_df=synced_df,
        audio_stomp_time=5.0,
        bio_left_stomp_time=10.0,
        bio_right_stomp_time=12.0,
        knee_side="left",
        pass_number=1,
        speed="slow",
    )

    # Offset should be bio - audio = 10 - 5 = 5s
    # New fields are timedeltas, convert to seconds for comparison
    assert record.sync_offset.total_seconds() == pytest.approx(5.0)
    assert record.audio_sync_time.total_seconds() == pytest.approx(5.0)
    assert record.bio_left_sync_time.total_seconds() == pytest.approx(10.0)
    assert record.bio_right_sync_time.total_seconds() == pytest.approx(12.0)


def test_aligned_stomp_times():
    """Test that aligned stomp times are correctly calculated."""
    synced_df = pd.DataFrame({
        "tt": pd.to_timedelta([0, 1, 2], unit="s"),
        "ch1": [0.1, 0.2, 0.3],
    })

    # Audio stomp at 5s, bio stomp at 10s
    record = create_sync_record_from_data(
        sync_file_name="test_sync",
        synced_df=synced_df,
        audio_stomp_time=5.0,
        bio_left_stomp_time=10.0,
        bio_right_stomp_time=12.0,
        knee_side="left",
        pass_number=1,
        speed="slow",
    )

    # After alignment, audio stomp appears at: audio_stomp + offset = 5 + 5 = 10s
    # Bio stomp remains at 10s (synced timeline is in bio coords)
    assert record.aligned_audio_sync_time.total_seconds() == pytest.approx(10.0)
    assert record.aligned_bio_sync_time.total_seconds() == pytest.approx(10.0)


def test_aligned_stomp_times_right_knee():
    """Test aligned stomp calculation for right knee."""
    synced_df = pd.DataFrame({
        "tt": pd.to_timedelta([0, 1, 2], unit="s"),
        "ch1": [0.1, 0.2, 0.3],
    })

    # Audio stomp at 3s, bio right stomp at 8s
    record = create_sync_record_from_data(
        sync_file_name="test_sync",
        synced_df=synced_df,
        audio_stomp_time=3.0,
        bio_left_stomp_time=10.0,
        bio_right_stomp_time=8.0,
        knee_side="right",
        pass_number=1,
        speed="medium",
    )

    # Offset = bio_right - audio = 8 - 3 = 5s
    assert record.sync_offset.total_seconds() == pytest.approx(5.0)
    # Aligned audio = 3 + 5 = 8s
    assert record.aligned_audio_sync_time.total_seconds() == pytest.approx(8.0)
    # Aligned bio = 8s (right knee)
    assert record.aligned_bio_sync_time.total_seconds() == pytest.approx(8.0)


def test_stomp_offset_with_timedelta_input():
    """Test that timedelta inputs are correctly converted to seconds."""
    synced_df = pd.DataFrame({
        "tt": pd.to_timedelta([0, 1, 2], unit="s"),
        "ch1": [0.1, 0.2, 0.3],
    })

    # Pass stomp times as timedeltas
    record = create_sync_record_from_data(
        sync_file_name="test_sync",
        synced_df=synced_df,
        audio_stomp_time=pd.Timedelta(seconds=5.0),
        bio_left_stomp_time=pd.Timedelta(seconds=10.0),
        bio_right_stomp_time=pd.Timedelta(seconds=12.0),
        knee_side="left",
    )

    assert record.sync_offset.total_seconds() == pytest.approx(5.0)
    assert record.aligned_audio_sync_time.total_seconds() == pytest.approx(10.0)
    assert record.aligned_bio_sync_time.total_seconds() == pytest.approx(10.0)


def test_sync_record_to_dict_includes_new_fields():
    """Test that to_dict includes aligned stomp and offset fields."""
    synced_df = pd.DataFrame({
        "tt": pd.to_timedelta([0, 1, 2], unit="s"),
        "ch1": [0.1, 0.2, 0.3],
    })

    record = create_sync_record_from_data(
        sync_file_name="test_sync",
        synced_df=synced_df,
        audio_stomp_time=5.0,
        bio_left_stomp_time=10.0,
        bio_right_stomp_time=12.0,
        knee_side="left",
    )

    data_dict = record.to_dict()

    # Check for new field names with timedeltas
    assert "Sync Offset" in data_dict
    assert "Aligned Audio Sync Time" in data_dict
    assert "Aligned Bio Sync Time" in data_dict
    # Values are timedeltas in the dict
    assert data_dict["Sync Offset"].total_seconds() == pytest.approx(5.0)
    assert data_dict["Aligned Audio Sync Time"].total_seconds() == pytest.approx(10.0)
    assert data_dict["Aligned Bio Sync Time"].total_seconds() == pytest.approx(10.0)


def test_stomp_offset_none_when_missing_data():
    """Test that offset is zero timedelta when audio or bio stomp is missing."""
    synced_df = pd.DataFrame({
        "tt": pd.to_timedelta([0, 1, 2], unit="s"),
        "ch1": [0.1, 0.2, 0.3],
    })

    # Missing audio stomp
    record = create_sync_record_from_data(
        sync_file_name="test_sync",
        synced_df=synced_df,
        audio_stomp_time=None,
        bio_left_stomp_time=10.0,
        bio_right_stomp_time=12.0,
        knee_side="left",
    )

    # When data is missing, defaults to zero timedelta
    assert record.sync_offset.total_seconds() == 0.0
    assert record.aligned_audio_sync_time.total_seconds() == 0.0
    assert record.aligned_bio_sync_time.total_seconds() == 0.0


def test_stomp_offset_none_when_knee_side_missing():
    """Test that offset is zero timedelta when knee_side is not specified."""
    synced_df = pd.DataFrame({
        "tt": pd.to_timedelta([0, 1, 2], unit="s"),
        "ch1": [0.1, 0.2, 0.3],
    })

    # Missing knee_side - can't determine which bio stomp to use
    record = create_sync_record_from_data(
        sync_file_name="test_sync",
        synced_df=synced_df,
        audio_stomp_time=5.0,
        bio_left_stomp_time=10.0,
        bio_right_stomp_time=12.0,
        knee_side=None,
    )

    # Without knee_side, can't determine which bio stomp to use, so defaults to 0
    assert record.sync_offset.total_seconds() == 0.0
    assert record.aligned_audio_sync_time.total_seconds() == 0.0
    assert record.aligned_bio_sync_time.total_seconds() == 0.0
