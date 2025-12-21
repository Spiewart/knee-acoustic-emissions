"""Tests for event metadata extraction with whitespace handling."""

import pandas as pd
import pytest

from src.synchronization.sync import get_event_metadata


def test_get_event_metadata_no_whitespace():
    """Test get_event_metadata with clean event names."""
    bio_meta = pd.DataFrame({
        "Event Info": ["Sync Left", "Sync Right", "Other Event"],
        "Time (sec)": [1.5, 2.5, 3.5],
    })

    result = get_event_metadata(bio_meta, "Sync Left")
    assert len(result) == 1
    assert result["Time (sec)"].iloc[0] == 1.5


def test_get_event_metadata_leading_whitespace_in_data():
    """Test get_event_metadata with leading whitespace in data."""
    bio_meta = pd.DataFrame({
        "Event Info": ["  Sync Left", "Sync Right", "Other Event"],
        "Time (sec)": [1.5, 2.5, 3.5],
    })

    result = get_event_metadata(bio_meta, "Sync Left")
    assert len(result) == 1
    assert result["Time (sec)"].iloc[0] == 1.5


def test_get_event_metadata_trailing_whitespace_in_data():
    """Test get_event_metadata with trailing whitespace in data."""
    bio_meta = pd.DataFrame({
        "Event Info": ["Sync Left  ", "Sync Right", "Other Event"],
        "Time (sec)": [1.5, 2.5, 3.5],
    })

    result = get_event_metadata(bio_meta, "Sync Left")
    assert len(result) == 1
    assert result["Time (sec)"].iloc[0] == 1.5


def test_get_event_metadata_both_whitespace():
    """Test get_event_metadata with whitespace on both sides."""
    bio_meta = pd.DataFrame({
        "Event Info": ["  Sync Left  ", "Sync Right", "Other Event"],
        "Time (sec)": [1.5, 2.5, 3.5],
    })

    result = get_event_metadata(bio_meta, "  Sync Left  ")
    assert len(result) == 1
    assert result["Time (sec)"].iloc[0] == 1.5


def test_get_event_metadata_preserves_internal_spaces():
    """Test that get_event_metadata preserves spaces between words."""
    bio_meta = pd.DataFrame({
        "Event Info": ["Sync Left", "Sync Right", "Other Event"],
        "Time (sec)": [1.5, 2.5, 3.5],
    })

    # Should NOT find "SyncLeft" when looking for "Sync Left"
    with pytest.raises(ValueError, match="No events found for: SyncLeft"):
        get_event_metadata(bio_meta, "SyncLeft")


def test_get_event_metadata_not_found():
    """Test get_event_metadata when event is not found."""
    bio_meta = pd.DataFrame({
        "Event Info": ["Sync Left", "Sync Right"],
        "Time (sec)": [1.5, 2.5],
    })

    with pytest.raises(ValueError, match="No events found for: Nonexistent"):
        get_event_metadata(bio_meta, "Nonexistent")
