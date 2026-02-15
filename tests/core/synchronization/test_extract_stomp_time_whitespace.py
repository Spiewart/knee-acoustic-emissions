"""Tests for _extract_stomp_time whitespace handling."""

import pandas as pd
import pytest

from src.biomechanics.importers import _extract_stomp_time


def test_extract_stomp_time_no_whitespace():
    """Test that _extract_stomp_time works with clean event names."""
    event_df = pd.DataFrame(
        {
            "Event Info": ["Sync Left", "Sync Right", "Other Event"],
            "Time (sec)": [1.5, 2.5, 3.5],
        }
    )

    result = _extract_stomp_time(event_df, "Sync Left")
    assert result == 1.5

    result = _extract_stomp_time(event_df, "Sync Right")
    assert result == 2.5


def test_extract_stomp_time_leading_whitespace_in_data():
    """Test that _extract_stomp_time handles leading whitespace in data."""
    event_df = pd.DataFrame(
        {
            "Event Info": ["  Sync Left", "Sync Right", "Other Event"],
            "Time (sec)": [1.5, 2.5, 3.5],
        }
    )

    # Should find it even with leading spaces in the data
    result = _extract_stomp_time(event_df, "Sync Left")
    assert result == 1.5


def test_extract_stomp_time_trailing_whitespace_in_data():
    """Test that _extract_stomp_time handles trailing whitespace in data."""
    event_df = pd.DataFrame(
        {
            "Event Info": ["Sync Left  ", "Sync Right", "Other Event"],
            "Time (sec)": [1.5, 2.5, 3.5],
        }
    )

    # Should find it even with trailing spaces in the data
    result = _extract_stomp_time(event_df, "Sync Left")
    assert result == 1.5


def test_extract_stomp_time_both_whitespace_in_data():
    """Test that _extract_stomp_time handles whitespace on both sides in data."""
    event_df = pd.DataFrame(
        {
            "Event Info": ["  Sync Left  ", "Sync Right", "Other Event"],
            "Time (sec)": [1.5, 2.5, 3.5],
        }
    )

    # Should find it even with spaces on both sides in the data
    result = _extract_stomp_time(event_df, "Sync Left")
    assert result == 1.5


def test_extract_stomp_time_whitespace_in_search_term():
    """Test that _extract_stomp_time handles whitespace in the search term."""
    event_df = pd.DataFrame(
        {
            "Event Info": ["Sync Left", "Sync Right", "Other Event"],
            "Time (sec)": [1.5, 2.5, 3.5],
        }
    )

    # Should find it even when search term has extra spaces
    result = _extract_stomp_time(event_df, "  Sync Left  ")
    assert result == 1.5


def test_extract_stomp_time_whitespace_both_sides():
    """Test that _extract_stomp_time handles whitespace in both data and search term."""
    event_df = pd.DataFrame(
        {
            "Event Info": ["  Sync Left  ", "Sync Right", "Other Event"],
            "Time (sec)": [1.5, 2.5, 3.5],
        }
    )

    # Should find it when both have extra spaces
    result = _extract_stomp_time(event_df, "  Sync Left  ")
    assert result == 1.5


def test_extract_stomp_time_preserves_internal_spaces():
    """Test that _extract_stomp_time preserves spaces between words."""
    event_df = pd.DataFrame(
        {
            "Event Info": ["Sync Left", "Sync Right", "Other Event"],
            "Time (sec)": [1.5, 2.5, 3.5],
        }
    )

    # Should NOT find "SyncLeft" when looking for "Sync Left"
    with pytest.raises(ValueError, match="Event 'SyncLeft' not found"):
        _extract_stomp_time(event_df, "SyncLeft")

    # Should find "Sync Left" when spaces are preserved
    result = _extract_stomp_time(event_df, "Sync Left")
    assert result == 1.5


def test_extract_stomp_time_not_found():
    """Test that _extract_stomp_time raises error when event not found."""
    event_df = pd.DataFrame(
        {
            "Event Info": ["Sync Left", "Sync Right"],
            "Time (sec)": [1.5, 2.5],
        }
    )

    with pytest.raises(ValueError, match="Event 'Nonexistent' not found"):
        _extract_stomp_time(event_df, "Nonexistent")


def test_extract_stomp_time_multiple_whitespace_types():
    """Test handling of tabs and multiple spaces."""
    event_df = pd.DataFrame(
        {
            "Event Info": ["\t Sync Left \t ", "Sync Right", "Other Event"],
            "Time (sec)": [1.5, 2.5, 3.5],
        }
    )

    # Should handle tabs and multiple spaces
    result = _extract_stomp_time(event_df, "  Sync Left  ")
    assert result == 1.5
