"""Global pytest fixtures for test suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_acoustic_legend_file(tmp_path):
    """Create a sample acoustic file legend Excel file for testing.

    This fixture generates an Excel file with acoustic metadata for left and
    right knees across three maneuvers (walk, flexion-extension, sit-to-stand).
    The structure matches the expected format for get_acoustics_metadata().
    """
    excel_path = tmp_path / "test_acoustic_file_legend.xlsx"

    # Create the structure as described in the docstring:
    # Two tables (L Knee and R Knee) separated by a blank row
    data = [
        ["L Knee", None, None, None, None, None],
        [
            "Maneuvers",
            "File Name",
            "Microphone",
            "Patellar Position",
            "Laterality",
            "Notes",
        ],
        [
            "Walk (slow,medium, fast)",
            "HP_W11.2-5-20240126_135702",
            1,
            "Infrapatellar",
            "Lateral",
            None,
        ],
        [None, None, 2, "Infrapatellar", "Medial", None],
        [None, None, 3, "Suprapatellar", "Medial", None],
        [None, None, 4, "Suprapatellar", "Lateral", None],
        [
            "Flexion - Extension",
            "HP_W11.2-1-20240126_135704",
            1,
            "Infrapatellar",
            "Lateral",
            None,
        ],
        [None, None, 2, "Infrapatellar", "Medial", None],
        [None, None, 3, "Suprapatellar", "Medial", None],
        [None, None, 4, "Suprapatellar", "Lateral", None],
        [
            "Sit - to - Stand",
            "HP_W11.2-3-20240126_135706",
            1,
            "Infrapatellar",
            "Lateral",
            None,
        ],
        [None, None, 2, "Infrapatellar", "Medial", None],
        [None, None, 3, "Suprapatellar", "Medial", None],
        [None, None, 4, "Suprapatellar", "Lateral", None],
        [None, None, None, None, None, None],  # Blank row separator
        ["R Knee", None, None, None, None, None],
        [
            "Maneuvers",
            "File Name",
            "Microphone",
            "Patellar Position",
            "Laterality",
            "Notes",
        ],
        [
            "Walk (slow,medium, fast)",
            "HP_W12.2-5-20240126_135802",
            1,
            "Infrapatellar",
            "Lateral",
            None,
        ],
        [None, None, 2, "Infrapatellar", "Medial", None],
        [None, None, 3, "Suprapatellar", "Medial", None],
        [None, None, 4, "Suprapatellar", "Lateral", None],
        [
            "Flexion - Extension",
            "HP_W12.2-1-20240126_135804",
            1,
            "Infrapatellar",
            "Lateral",
            None,
        ],
        [None, None, 2, "Infrapatellar", "Medial", None],
        [None, None, 3, "Suprapatellar", "Medial", None],
        [None, None, 4, "Suprapatellar", "Lateral", None],
        [
            "Sit - to - Stand",
            "HP_W12.2-3-20240126_135806",
            1,
            "Infrapatellar",
            "Lateral",
            None,
        ],
        [None, None, 2, "Infrapatellar", "Medial", None],
        [None, None, 3, "Suprapatellar", "Medial", None],
        [None, None, 4, "Suprapatellar", "Lateral", None],
    ]

    df = pd.DataFrame(data)

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(
            writer, sheet_name="Acoustic Notes", index=False, header=False
        )

    return excel_path


@pytest.fixture
def fake_biomechanics_excel(tmp_path):
    """Create a fake biomechanics Excel file for testing.

    This fixture generates an Excel file with two sheets:
    - Data sheet: Contains biomechanics data in V3D format with walking cycle data
    - Events sheet: Contains timing information for walking events

    Returns a dict with keys:
    - excel_path: Path to the Excel file
    - data_sheet: Name of the data sheet
    - events_sheet: Name of the events sheet
    - events_data: List of event dictionaries
    """
    excel_path = tmp_path / "test_biomechanics.xlsx"

    # Create event data
    events_data = [
        {"Event Info": "Sync Left", "Time (sec)": 16.23},
        {"Event Info": "Sync Right", "Time (sec)": 17.48},
        {"Event Info": "Slow Speed Start", "Time (sec)": 18.80},
        {"Event Info": "SS Pass 1 Start", "Time (sec)": 19.28},
        {"Event Info": "SS Pass 1 End", "Time (sec)": 24.96},
        {"Event Info": "SS Pass 2 Start", "Time (sec)": 27.80},
        {"Event Info": "SS Pass 2 End", "Time (sec)": 34.63},
        {"Event Info": "Slow Speed End", "Time (sec)": 119.88},
        {"Event Info": "Normal Speed Start", "Time (sec)": 135.80},
        {"Event Info": "NS Pass 1 Start", "Time (sec)": 136.96},
        {"Event Info": "NS Pass 1 End", "Time (sec)": 142.15},
        {"Event Info": "NS Pass 2 Start", "Time (sec)": 144.13},
        {"Event Info": "NS Pass 2 End", "Time (sec)": 148.94},
        {"Event Info": "Normal Speed End", "Time (sec)": 408.44},
        {"Event Info": "Fast Speed Start", "Time (sec)": 438.97},
        {"Event Info": "FS Pass 1 Start", "Time (sec)": 439.88},
        {"Event Info": "FS Pass 1 End", "Time (sec)": 443.73},
        {"Event Info": "FS Pass 2 Start", "Time (sec)": 445.75},
        {"Event Info": "FS Pass 2 End", "Time (sec)": 449.72},
        {"Event Info": "Fast Speed End", "Time (sec)": 596.48},
    ]
    events_df = pd.DataFrame(events_data)

    # Create biomechanics data sheet with proper V3D column structure
    study_id = "AOA1011"
    time_points = np.linspace(0, 10, 100)

    # Create columns in V3D format that match what process_biomechanics
    # expects. Columns must be named with "V3D\" prefix and unique IDs
    data_dict = {}

    # Create two recording passes
    for pass_num in [1, 2]:
        # Create uid in format: V3D\Study_Walk0001_SSP#_Filt.c3d
        uid_base = f"V3D\\{study_id}_Walk0001_SSP{pass_num}_Filt.c3d"

        # Frame column - each column name uses the uid as a prefix
        data_dict[f"{uid_base}"] = (
            ["Frame", ""]
            + time_points.tolist()
        )
        # Angle columns
        data_dict[f"{uid_base}.1"] = (
            ["LAnkleAngles", "X"]
            + (np.sin(time_points + pass_num) * 10).tolist()
        )
        data_dict[f"{uid_base}.2"] = (
            ["LAnkleAngles", "Y"]
            + (np.cos(time_points + pass_num) * 5).tolist()
        )
        data_dict[f"{uid_base}.3"] = (
            ["RAnkleAngles", "X"]
            + (np.sin(time_points + pass_num + 1) * 10).tolist()
        )

    biomechanics_df = pd.DataFrame(data_dict)

    # Write to Excel
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        biomechanics_df.to_excel(
            writer,
            sheet_name=f"{study_id}_Slow_Walking",
            index=False,
        )
        events_df.to_excel(
            writer,
            sheet_name=f"{study_id}_Walk0001",
            index=False,
        )

    return {
        "excel_path": excel_path,
        "data_sheet": f"{study_id}_Slow_Walking",
        "events_sheet": f"{study_id}_Walk0001",
        "events_data": events_data,
    }


@pytest.fixture
def fake_audio_data(tmp_path):
    """Create fake audio data pickle file for testing.

    This fixture generates a pickle file containing audio data in the format
    expected by sync_audio_with_biomechanics module. The data includes:
    - Time column (tt): Time stamps for each sample
    - Channel columns (ch1-ch4): Audio data from four microphones
    - A spike at ~16.23 seconds to simulate a stomp event
    - Duration covers slow speed events (0 to 130 seconds)

    Realistic sensor specs:
    - Sample rate: 52 kHz (52,000 samples per second)
    - Duration: ~130 seconds (covers slow speed walking events)

    Returns:
        Path to the pickle file containing the audio data.
    """
    # Realistic sample rate, but limited duration for test performance
    sample_rate = 52000  # 52 kHz
    # Duration covers slow speed events (Slow Speed End at 119.88 sec)
    duration = 130  # seconds
    n_points = int(duration * sample_rate)

    time_points = np.linspace(0, duration, n_points)

    # Create a DataFrame with very low-amplitude noise
    # Use much lower noise so the stomp spike is clearly detectable
    audio_data = pd.DataFrame({
        'tt': time_points,
        'ch1': np.random.randn(n_points) * 0.01,
        'ch2': np.random.randn(n_points) * 0.01,
        'ch3': np.random.randn(n_points) * 0.01,
        'ch4': np.random.randn(n_points) * 0.01,
    })

    # Add a prominent spike at 16.23 seconds for the stomp detection
    # Sync Left is at 16.23 seconds (from fake_biomechanics_excel fixture)
    stomp_time = 16.23
    stomp_idx = int(stomp_time * sample_rate)
    # Add spike over a small window (10ms) with high amplitude
    spike_samples = int(0.01 * sample_rate)  # 10ms spike
    audio_data.loc[
        stomp_idx:stomp_idx + spike_samples,
        ['ch1', 'ch2', 'ch3', 'ch4']
    ] = 10.0

    # Save to pickle
    audio_file_path = tmp_path / "test_audio_with_freq.pkl"
    audio_data.to_pickle(audio_file_path)

    return audio_file_path
