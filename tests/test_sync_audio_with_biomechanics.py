"""Test data for sync_audio_with_biomechanics module."""


import pandas as pd

from acoustic_emissions_processing.process_biomechanics import (
    import_biomechanics_recordings,
)
from acoustic_emissions_processing.sync_audio_with_biomechanics import (
    get_audio_stomp_time,
    get_right_stomp_time,
    get_stomp_time,
    load_audio_data,
    sync_audio_with_biomechanics,
)

# Fake test data from biomechanics timing events
FAKE_BIOMECHANICS_EVENTS = [
    {"Event Info": "Sync Left", "Time (sec)": 16.23},
    {"Event Info": "Sync Right", "Time (sec)": 17.48},
    {"Event Info": "Slow Speed Start", "Time (sec)": 18.80},
    {"Event Info": "SS Pass 1 Start", "Time (sec)": 19.28},
    {"Event Info": "SS Pass 1 End", "Time (sec)": 24.96},
    {"Event Info": "SS Pass 2 Start", "Time (sec)": 27.80},
    {"Event Info": "SS Pass 2 End", "Time (sec)": 34.63},
    {"Event Info": "SS Pass 3 Start", "Time (sec)": 37.09},
    {"Event Info": "SS Pass 3 End", "Time (sec)": 44.03},
    {"Event Info": "SS Pass 4 Start", "Time (sec)": 47.00},
    {"Event Info": "SS Pass 4 End", "Time (sec)": 54.03},
    {"Event Info": "SS Pass 5 Start", "Time (sec)": 56.27},
    {"Event Info": "SS Pass 5 End", "Time (sec)": 62.95},
    {"Event Info": "SS Pass 6 Start", "Time (sec)": 66.70},
    {"Event Info": "SS Pass 6 End", "Time (sec)": 74.01},
    {"Event Info": "SS Pass 7 Start", "Time (sec)": 76.73},
    {"Event Info": "SS Pass 7 End", "Time (sec)": 84.33},
    {"Event Info": "SS Pass 8 Start", "Time (sec)": 88.16},
    {"Event Info": "SS Pass 8 End", "Time (sec)": 96.04},
    {"Event Info": "SS Pass 9 Start", "Time (sec)": 98.08},
    {"Event Info": "SS Pass 9 End", "Time (sec)": 105.65},
    {"Event Info": "SS Pass 10 Start", "Time (sec)": 108.96},
    {"Event Info": "SS Pass 10 End", "Time (sec)": 116.94},
    {"Event Info": "Slow Speed End", "Time (sec)": 119.88},
    {"Event Info": "Normal Speed Start", "Time (sec)": 135.80},
    {"Event Info": "NS Pass 1 Start", "Time (sec)": 136.96},
    {"Event Info": "NS Pass 1 End", "Time (sec)": 142.15},
    {"Event Info": "NS Pass 2 Start", "Time (sec)": 144.13},
    {"Event Info": "NS Pass 2 End", "Time (sec)": 148.94},
    {"Event Info": "NS Pass 3 Start", "Time (sec)": 150.41},
    {"Event Info": "NS Pass 3 End", "Time (sec)": 155.74},
    {"Event Info": "NS Pass 4 Start", "Time (sec)": 158.44},
    {"Event Info": "NS Pass 4 End", "Time (sec)": 163.73},
    {"Event Info": "NS Pass 5 Start", "Time (sec)": 165.16},
    {"Event Info": "NS Pass 5 End", "Time (sec)": 170.33},
    {"Event Info": "NS Pass 6 Start", "Time (sec)": 173.38},
    {"Event Info": "NS Pass 6 End", "Time (sec)": 178.78},
    {"Event Info": "NS Pass 7 Start", "Time (sec)": 180.33},
    {"Event Info": "NS Pass 7 End", "Time (sec)": 185.56},
    {"Event Info": "NS Pass 8 Start", "Time (sec)": 189.25},
    {"Event Info": "NS Pass 8 End", "Time (sec)": 194.54},
    {"Event Info": "NS Pass 9 Start", "Time (sec)": 196.51},
    {"Event Info": "NS Pass 9 End", "Time (sec)": 201.48},
    {"Event Info": "NS Pass 10 Start", "Time (sec)": 204.89},
    {"Event Info": "NS Pass 10 End", "Time (sec)": 210.26},
    {"Event Info": "NS Pass 11 Start", "Time (sec)": 212.00},
    {"Event Info": "NS Pass 11 End", "Time (sec)": 217.16},
    {"Event Info": "NS Pass 12 Start", "Time (sec)": 220.62},
    {"Event Info": "NS Pass 12 End", "Time (sec)": 225.50},
    {"Event Info": "NS Pass 13 Start", "Time (sec)": 227.53},
    {"Event Info": "NS Pass 13 End", "Time (sec)": 232.63},
    {"Event Info": "NS Pass 14 Start", "Time (sec)": 236.00},
    {"Event Info": "NS Pass 14 End", "Time (sec)": 240.99},
    {"Event Info": "NS Pass 15 Start", "Time (sec)": 243.15},
    {"Event Info": "NS Pass 15 End", "Time (sec)": 248.08},
    {"Event Info": "NS Pass 16 Start", "Time (sec)": 251.63},
    {"Event Info": "NS Pass 16 End", "Time (sec)": 256.54},
    {"Event Info": "NS Pass 17 Start", "Time (sec)": 294.36},
    {"Event Info": "NS Pass 17 End", "Time (sec)": 298.81},
    {"Event Info": "NS Pass 18 Start", "Time (sec)": 301.14},
    {"Event Info": "NS Pass 18 End", "Time (sec)": 305.74},
    {"Event Info": "NS Pass 19 Start", "Time (sec)": 308.01},
    {"Event Info": "NS Pass 19 End", "Time (sec)": 312.63},
    {"Event Info": "NS Pass 20 Start", "Time (sec)": 315.52},
    {"Event Info": "NS Pass 20 End", "Time (sec)": 320.44},
    {"Event Info": "NS Pass 21 Start", "Time (sec)": 322.44},
    {"Event Info": "NS Pass 21 End", "Time (sec)": 327.12},
    {"Event Info": "NS Pass 22 Start", "Time (sec)": 332.19},
    {"Event Info": "NS Pass 22 End", "Time (sec)": 337.59},
    {"Event Info": "NS Pass 23 Start", "Time (sec)": 344.81},
    {"Event Info": "NS Pass 23 End", "Time (sec)": 349.33},
    {"Event Info": "NS Pass 24 Start", "Time (sec)": 355.13},
    {"Event Info": "NS Pass 24 End", "Time (sec)": 359.68},
    {"Event Info": "NS Pass 25 Start", "Time (sec)": 362.78},
    {"Event Info": "NS Pass 25 End", "Time (sec)": 367.78},
    {"Event Info": "NS Pass 26 Start", "Time (sec)": 370.88},
    # Note: Duplicate event in source data
    {"Event Info": "NS Pass 26 Start", "Time (sec)": 371.20},
    {"Event Info": "NS Pass 26 End", "Time (sec)": 375.78},
    {"Event Info": "NS Pass 27 Start", "Time (sec)": 377.42},
    {"Event Info": "NS Pass 27 End", "Time (sec)": 382.39},
    {"Event Info": "NS Pass 28 Start", "Time (sec)": 386.13},
    {"Event Info": "NS Pass 28 End", "Time (sec)": 390.69},
    {"Event Info": "NS Pass 29 Start", "Time (sec)": 392.76},
    {"Event Info": "NS Pass 29 End", "Time (sec)": 397.77},
    {"Event Info": "NS Pass 30 Start", "Time (sec)": 401.03},
    {"Event Info": "NS Pass 30 End", "Time (sec)": 406.10},
    {"Event Info": "Normal Speed End", "Time (sec)": 408.44},
    {"Event Info": "Fast Speed Start", "Time (sec)": 438.97},
    {"Event Info": "FS Pass 1 Start", "Time (sec)": 439.88},
    {"Event Info": "FS Pass 1 End", "Time (sec)": 443.73},
    {"Event Info": "FS Pass 2 Start", "Time (sec)": 445.75},
    {"Event Info": "FS Pass 2 End", "Time (sec)": 449.72},
    {"Event Info": "FS Pass 3 Start", "Time (sec)": 450.93},
    {"Event Info": "FS Pass 3 End", "Time (sec)": 455.13},
    {"Event Info": "FS Pass 4 Start", "Time (sec)": 457.35},
    {"Event Info": "FS Pass 4 End", "Time (sec)": 461.94},
    {"Event Info": "FS Pass 5 Start", "Time (sec)": 463.46},
    {"Event Info": "FS Pass 5 End", "Time (sec)": 467.53},
    {"Event Info": "FS Pass 6 Start", "Time (sec)": 470.13},
    {"Event Info": "FS Pass 6 End", "Time (sec)": 474.78},
    {"Event Info": "FS Pass 7 Start", "Time (sec)": 475.92},
    {"Event Info": "FS Pass 7 End", "Time (sec)": 480.09},
    {"Event Info": "FS Pass 8 Start", "Time (sec)": 483.03},
    {"Event Info": "FS Pass 8 End", "Time (sec)": 487.36},
    {"Event Info": "FS Pass 9 Start", "Time (sec)": 488.72},
    {"Event Info": "FS Pass 9 End", "Time (sec)": 492.97},
    {"Event Info": "FS Pass 10 Start", "Time (sec)": 495.33},
    {"Event Info": "FS Pass 10 End", "Time (sec)": 499.67},
    {"Event Info": "FS Pass 11 Start", "Time (sec)": 500.96},
    {"Event Info": "FS Pass 11 End", "Time (sec)": 505.38},
    {"Event Info": "FS Pass 12 Start", "Time (sec)": 507.83},
    {"Event Info": "FS Pass 12 End", "Time (sec)": 512.23},
    {"Event Info": "FS Pass 13 Start", "Time (sec)": 513.61},
    {"Event Info": "FS Pass 13 End", "Time (sec)": 518.02},
    {"Event Info": "FS Pass 14 Start", "Time (sec)": 520.52},
    {"Event Info": "FS Pass 14 End", "Time (sec)": 524.95},
    {"Event Info": "FS Pass 15 Start", "Time (sec)": 553.80},
    {"Event Info": "FS Pass 15 End", "Time (sec)": 557.46},
    {"Event Info": "FS Pass 16 Start", "Time (sec)": 559.68},
    {"Event Info": "FS Pass 16 End", "Time (sec)": 563.92},
    {"Event Info": "FS Pass 17 Start", "Time (sec)": 565.22},
    {"Event Info": "FS Pass 17 End", "Time (sec)": 569.55},
    {"Event Info": "FS Pass 18 Start", "Time (sec)": 572.39},
    {"Event Info": "FS Pass 18 End", "Time (sec)": 577.04},
    {"Event Info": "FS Pass 19 Start", "Time (sec)": 584.13},
    {"Event Info": "FS Pass 19 End", "Time (sec)": 587.94},
    {"Event Info": "FS Pass 20 Start", "Time (sec)": 590.75},
    {"Event Info": "FS Pass 20 End", "Time (sec)": 595.06},
    {"Event Info": "Fast Speed End", "Time (sec)": 596.48},
]


def test_biomechanics_events_structure():
    """Test that biomechanics events have the expected structure."""
    assert len(FAKE_BIOMECHANICS_EVENTS) > 0

    for event in FAKE_BIOMECHANICS_EVENTS:
        assert "Event Info" in event
        assert "Time (sec)" in event
        assert isinstance(event["Event Info"], str)
        assert isinstance(event["Time (sec)"], (int, float))
        assert event["Time (sec)"] >= 0


def test_sync_events_present():
    """Test that sync events are present in the data."""
    event_names = [e["Event Info"] for e in FAKE_BIOMECHANICS_EVENTS]
    assert "Sync Left" in event_names
    assert "Sync Right" in event_names


def test_speed_sections_present():
    """Test that all speed sections are present."""
    event_names = [e["Event Info"] for e in FAKE_BIOMECHANICS_EVENTS]

    # Check slow speed section
    assert "Slow Speed Start" in event_names
    assert "Slow Speed End" in event_names

    # Check normal speed section
    assert "Normal Speed Start" in event_names
    assert "Normal Speed End" in event_names

    # Check fast speed section
    assert "Fast Speed Start" in event_names
    assert "Fast Speed End" in event_names


def test_pass_pairs_match():
    """Test that each pass has both start and end events."""
    event_names = [e["Event Info"] for e in FAKE_BIOMECHANICS_EVENTS]

    # Check slow speed passes
    for i in range(1, 11):
        assert f"SS Pass {i} Start" in event_names
        assert f"SS Pass {i} End" in event_names

    # Check normal speed passes (excluding the duplicate NS Pass 26 Start)
    for i in range(1, 31):
        assert f"NS Pass {i} Start" in event_names
        assert f"NS Pass {i} End" in event_names

    # Check fast speed passes
    for i in range(1, 21):
        assert f"FS Pass {i} Start" in event_names
        assert f"FS Pass {i} End" in event_names


def test_chronological_order():
    """Test that events are in chronological order."""
    times = [e["Time (sec)"] for e in FAKE_BIOMECHANICS_EVENTS]

    for i in range(1, len(times)):
        assert times[i] >= times[i-1], (
            f"Events not in chronological order at index {i}: "
            f"{times[i-1]} -> {times[i]}"
        )


def test_get_stomp_time():
    """Test the get_stomp_time function for both feet."""

    # Create a DataFrame from the fake biomechanics events
    bio_meta = pd.DataFrame(FAKE_BIOMECHANICS_EVENTS)

    # Test left foot stomp time
    left_stomp_time = get_stomp_time(bio_meta, foot="left")
    expected_left_time = pd.to_timedelta(16.23, unit='s').to_pytimedelta()
    assert left_stomp_time == expected_left_time, (
        f"Left stomp time mismatch: expected {expected_left_time}, got {left_stomp_time}"
    )

    # Test right foot stomp time
    right_stomp_time = get_stomp_time(bio_meta, foot="right")
    expected_right_time = pd.to_timedelta(17.48, unit='s').to_pytimedelta()
    assert right_stomp_time == expected_right_time, (
        f"Right stomp time mismatch: expected {expected_right_time}, got {right_stomp_time}"
    )


def test_load_audio_data():
    path = "/Users/spiewart/kae_signal_processing_ml/sample_files/HP_W11.2-1-20240126_135704_outputs/HP_W11.2-1-20240126_135704_with_freq.pkl"

    audio_data = load_audio_data(path)

    assert not audio_data.empty, "Loaded audio data is empty."


def test_get_audio_stomp_time():
    path = "/Users/spiewart/kae_signal_processing_ml/sample_files/HP_W11.2-1-20240126_135704_outputs/HP_W11.2-1-20240126_135704_with_freq.pkl"

    audio_data = load_audio_data(path)

    stomp_time = get_audio_stomp_time(audio_data)

    expected_time = pd.to_timedelta(6.998379, unit='s').to_pytimedelta()

    assert stomp_time == expected_time, (
        f"Audio stomp time mismatch: expected {expected_time}, got {stomp_time}"
    )


def test_sync_audio_with_biomechanics():
    # Load biomechanics data
    sample_biomechanics_file_path = "/Users/spiewart/kae_signal_processing_ml/sample_files/AOA1011_Biomechanics_Full_Set.xlsx"
    sample_biomechanics_sheet_name = "AOA1011_Slow_Walking"
    biomechanics_metadata_sheet_name = "AOA1011_Walk0001"

    biomechanics_cycles = import_biomechanics_recordings(
        biomechanics_file=sample_biomechanics_file_path,
        data_sheet_name=sample_biomechanics_sheet_name,
        event_data_sheet_name=biomechanics_metadata_sheet_name,
    )

    right_knee_audio_file_name = "HP_W11.2-1-20240126_135704"

    right_knee_audio_file_path = (
        "/Users/spiewart/kae_signal_processing_ml/sample_files/"
        f"{right_knee_audio_file_name}_outputs/"
        f"{right_knee_audio_file_name}_with_freq.pkl"
    )

    audio_data = load_audio_data(right_knee_audio_file_path)

    bio_meta = pd.DataFrame(FAKE_BIOMECHANICS_EVENTS)

    audio_stomp_time = get_audio_stomp_time(audio_data)
    bio_right_stomp_time = get_right_stomp_time(bio_meta)

    synchronized_df = sync_audio_with_biomechanics(
        audio_stomp_time,
        bio_right_stomp_time,
        audio_data,
        biomechanics_cycles[0].data,  # Use the first biomechanics DataFrame for testing
    )

    assert not synchronized_df.empty, "Synchronized DataFrame is empty."
