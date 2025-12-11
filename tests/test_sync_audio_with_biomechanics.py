"""Test data for sync_audio_with_biomechanics module."""

import pandas as pd
import pytest

from process_biomechanics import import_biomechanics_recordings
from sync_audio_with_biomechanics import (
    get_audio_stomp_time,
    get_bio_end_time,
    get_bio_start_time,
    get_right_stomp_time,
    get_stomp_time,
    load_audio_data,
    sync_audio_with_biomechanics,
)


def test_get_stomp_time(fake_participant_directory):
    """Test the get_stomp_time function for both feet."""

    # Create a DataFrame from the fake biomechanics events
    events_data = fake_participant_directory["biomechanics"][
        "events_data"
    ]
    bio_meta = pd.DataFrame(events_data)

    # Test left foot stomp time
    left_stomp_time = get_stomp_time(bio_meta, foot="left")
    expected_left_time = pd.to_timedelta(16.23, unit='s').to_pytimedelta()
    assert left_stomp_time == expected_left_time, (
        f"Left stomp time mismatch: expected {expected_left_time}, "
        f"got {left_stomp_time}"
    )

    # Test right foot stomp time
    right_stomp_time = get_stomp_time(bio_meta, foot="right")
    expected_right_time = pd.to_timedelta(17.48, unit='s').to_pytimedelta()
    assert right_stomp_time == expected_right_time, (
        f"Right stomp time mismatch: expected {expected_right_time}, "
        f"got {right_stomp_time}"
    )


def test_load_audio_data(fake_participant_directory):
    """Test loading audio data from pickle file."""
    audio_path = fake_participant_directory["audio_paths"]["left"][
        "Walking"
    ]
    audio_data = load_audio_data(audio_path)

    assert not audio_data.empty, "Loaded audio data is empty."


def test_get_audio_stomp_time(fake_participant_directory):
    """Test getting stomp time from audio data."""
    audio_path = fake_participant_directory["audio_paths"]["left"][
        "Walking"
    ]
    audio_data = load_audio_data(audio_path)

    stomp_time = get_audio_stomp_time(audio_data)

    # The stomp time should be around 16.23 seconds (Sync Left time)
    # Just verify that the stomp time is a reasonable value
    assert stomp_time is not None, "Stomp time should not be None"
    # Check string representation to avoid type checking issues
    time_str = str(stomp_time)
    # Should be in the 16.2x second range
    assert "0:00:16" in time_str, (
        f"Audio stomp time unexpected: got {time_str}"
    )


def test_sync_audio_with_biomechanics(fake_participant_directory):
    """Test the full audio-biomechanics synchronization pipeline."""
    from pathlib import Path

    # Load biomechanics data
    excel_file = Path(fake_participant_directory["biomechanics"]["excel_path"])
    biomechanics_cycles = import_biomechanics_recordings(
        biomechanics_file=excel_file,
        maneuver="walk",
        speed="slow",
    )

    audio_path = fake_participant_directory["audio_paths"]["left"][
        "Walking"
    ]
    audio_data = load_audio_data(audio_path)

    bio_meta = fake_participant_directory["biomechanics"]["events_data"]

    audio_stomp_time = get_audio_stomp_time(audio_data)
    bio_right_stomp_time = get_right_stomp_time(pd.DataFrame(bio_meta))

    synchronized_df = sync_audio_with_biomechanics(
        audio_stomp_time,
        bio_right_stomp_time,
        audio_data,
        # Use the first biomechanics DataFrame for testing
        biomechanics_cycles[0].data,
    )

    assert not synchronized_df.empty, "Synchronized DataFrame is empty."


def test_get_bio_start_time_walk(fake_participant_directory):
    """Test get_bio_start_time for walking maneuver."""
    events_df = pd.read_excel(
        fake_participant_directory["biomechanics"]["excel_path"],
        sheet_name=fake_participant_directory["biomechanics"][
            "events_sheets"
        ]["walk_pass"],
    )

    start_time = get_bio_start_time(
        event_metadata=events_df,
        maneuver="walk",
        speed="slow",
        pass_number=1,
    )

    assert start_time is not None
    td_type = type(pd.to_timedelta(1, unit="s").to_pytimedelta())
    assert isinstance(start_time, td_type)


def test_get_bio_start_time_sit_to_stand(fake_participant_directory):
    """Test get_bio_start_time for sit-to-stand maneuver."""
    events_df = pd.read_excel(
        fake_participant_directory["biomechanics"]["excel_path"],
        sheet_name=fake_participant_directory["biomechanics"][
            "events_sheets"
        ]["sit_to_stand"],
    )

    start_time = get_bio_start_time(
        event_metadata=events_df,
        maneuver="sit_to_stand",
    )

    assert start_time is not None
    td_type = type(pd.to_timedelta(1, unit="s").to_pytimedelta())
    assert isinstance(start_time, td_type)


def test_get_bio_start_time_flexion_extension(fake_participant_directory):
    """Test get_bio_start_time for flexion-extension maneuver."""
    events_df = pd.read_excel(
        fake_participant_directory["biomechanics"]["excel_path"],
        sheet_name=fake_participant_directory["biomechanics"][
            "events_sheets"
        ]["flexion_extension"],
    )

    start_time = get_bio_start_time(
        event_metadata=events_df,
        maneuver="flexion_extension",
    )

    assert start_time is not None
    td_type = type(pd.to_timedelta(1, unit="s").to_pytimedelta())
    assert isinstance(start_time, td_type)


def test_get_bio_start_time_walk_missing_params():
    """Test get_bio_start_time raises error for walk without speed."""
    events_df = pd.DataFrame(
        {"Event Info": ["SS Pass 1 Start"], "Time (sec)": [10.0]}
    )

    with pytest.raises(ValueError, match="speed and pass_number required"):
        get_bio_start_time(
            event_metadata=events_df,
            maneuver="walk",
        )


def test_get_bio_end_time_walk(fake_participant_directory):
    """Test get_bio_end_time for walking maneuver."""
    events_df = pd.read_excel(
        fake_participant_directory["biomechanics"]["excel_path"],
        sheet_name=fake_participant_directory["biomechanics"][
            "events_sheets"
        ]["walk_pass"],
    )

    end_time = get_bio_end_time(
        event_metadata=events_df,
        maneuver="walk",
        speed="slow",
        pass_number=1,
    )

    assert end_time is not None
    td_type = type(pd.to_timedelta(1, unit="s").to_pytimedelta())
    assert isinstance(end_time, td_type)


def test_get_bio_end_time_sit_to_stand(fake_participant_directory):
    """Test get_bio_end_time for sit-to-stand maneuver."""
    events_df = pd.read_excel(
        fake_participant_directory["biomechanics"]["excel_path"],
        sheet_name=fake_participant_directory["biomechanics"][
            "events_sheets"
        ]["sit_to_stand"],
    )

    end_time = get_bio_end_time(
        event_metadata=events_df,
        maneuver="sit_to_stand",
    )

    assert end_time is not None
    td_type = type(pd.to_timedelta(1, unit="s").to_pytimedelta())
    assert isinstance(end_time, td_type)


def test_get_bio_end_time_flexion_extension(fake_participant_directory):
    """Test get_bio_end_time for flexion-extension maneuver."""
    events_df = pd.read_excel(
        fake_participant_directory["biomechanics"]["excel_path"],
        sheet_name=fake_participant_directory["biomechanics"][
            "events_sheets"
        ]["flexion_extension"],
    )

    end_time = get_bio_end_time(
        event_metadata=events_df,
        maneuver="flexion_extension",
    )

    assert end_time is not None
    td_type = type(pd.to_timedelta(1, unit="s").to_pytimedelta())
    assert isinstance(end_time, td_type)


def test_get_bio_end_time_walk_missing_params():
    """Test get_bio_end_time raises error for walk without pass_number."""
    events_df = pd.DataFrame(
        {"Event Info": ["SS Pass 1 End"], "Time (sec)": [30.0]}
    )

    with pytest.raises(ValueError, match="speed and pass_number required"):
        get_bio_end_time(
            event_metadata=events_df,
            maneuver="walk",
            speed="slow",
        )


def test_bio_start_end_times_ordering(fake_participant_directory):
    """Test that start time is before end time for all maneuvers."""
    # Test walk maneuver
    events_df = pd.read_excel(
        fake_participant_directory["biomechanics"]["excel_path"],
        sheet_name=fake_participant_directory["biomechanics"][
            "events_sheets"
        ]["walk_pass"],
    )

    start_time = get_bio_start_time(
        event_metadata=events_df,
        maneuver="walk",
        speed="slow",
        pass_number=1,
    )

    end_time = get_bio_end_time(
        event_metadata=events_df,
        maneuver="walk",
        speed="slow",
        pass_number=1,
    )

    assert start_time < end_time

    # Test sit-to-stand maneuver
    events_df = pd.read_excel(
        fake_participant_directory["biomechanics"]["excel_path"],
        sheet_name=fake_participant_directory["biomechanics"][
            "events_sheets"
        ]["sit_to_stand"],
    )

    start_time = get_bio_start_time(
        event_metadata=events_df,
        maneuver="sit_to_stand",
    )

    end_time = get_bio_end_time(
        event_metadata=events_df,
        maneuver="sit_to_stand",
    )

    assert start_time < end_time
