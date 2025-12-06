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

    This fixture generates an Excel file with sheets for walk, sit-to-stand,
    and flexion-extension maneuvers with their corresponding event sheets.

    Sheet naming convention:
    - Walk: AOAXXXX_Speed_Walking (with events: AOAXXXX_Walk{PassNumber})
    - Sit-to-stand: AOAXXXX_SitToStand (with events: AOAXXXX_StoS_Events)
    - Flexion-extension: AOAXXXX_FlexExt (with events: AOAXXXX_FE_Events)

    Returns a dict with keys:
    - excel_path: Path to the Excel file
    - data_sheets: Dict of data sheet names by maneuver
    - events_sheets: Dict of event sheet names by maneuver
    """
    # Use a filename that matches the study ID pattern
    # The first 7 characters should be the study ID
    excel_path = tmp_path / "AOA1011_Biomechanics_Full_Set.xlsx"

    study_id = "AOA1011"
    time_points = np.linspace(0, 10, 100)

    # Create biomechanics data for walk maneuver (separate DataFrame)
    walk_data_dict = {}
    for pass_num in [1, 2]:
        uid_base = f"V3D\\{study_id}_Walk0001_SSP{pass_num}_Filt.c3d"
        walk_data_dict[f"{uid_base}"] = (
            ["Frame", ""] + time_points.tolist()
        )
        walk_data_dict[f"{uid_base}.1"] = (
            ["LAnkleAngles", "X"]
            + (np.sin(time_points + pass_num) * 10).tolist()
        )
        walk_data_dict[f"{uid_base}.2"] = (
            ["LAnkleAngles", "Y"]
            + (np.cos(time_points + pass_num) * 5).tolist()
        )

    walk_df = pd.DataFrame(walk_data_dict)

    # Create biomechanics data for sit-to-stand (separate DataFrame)
    sts_data_dict = {}
    uid_base_sts = f"V3D\\{study_id}_SitToStand0001_Filt.c3d"
    sts_data_dict[f"{uid_base_sts}"] = (
        ["Frame", ""] + time_points.tolist()
    )
    sts_data_dict[f"{uid_base_sts}.1"] = (
        ["LKneeAngles", "X"]
        + (np.sin(time_points) * 15).tolist()
    )
    sts_data_dict[f"{uid_base_sts}.2"] = (
        ["RKneeAngles", "X"]
        + (np.cos(time_points) * 15).tolist()
    )
    sts_df = pd.DataFrame(sts_data_dict)

    # Create biomechanics data for flexion-extension (separate DataFrame)
    fe_data_dict = {}
    uid_base_fe = f"V3D\\{study_id}_FlexExt0001_Filt.c3d"
    fe_data_dict[f"{uid_base_fe}"] = (
        ["Frame", ""] + time_points.tolist()
    )
    fe_data_dict[f"{uid_base_fe}.1"] = (
        ["LKneeAngles", "X"]
        + (np.sin(time_points * 2) * 10).tolist()
    )
    fe_data_dict[f"{uid_base_fe}.2"] = (
        ["RKneeAngles", "X"]
        + (np.cos(time_points * 2) * 10).tolist()
    )
    fe_df = pd.DataFrame(fe_data_dict)

    # Create walking events
    walking_events_data = [
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
    ]
    walking_events_df = pd.DataFrame(walking_events_data)

    # Create sit-to-stand events
    sts_events_data = [
        {"Event Info": "Movement Start", "Time (sec)": 5.0},
        {"Event Info": "Movement End", "Time (sec)": 8.5},
    ]
    sts_events_df = pd.DataFrame(sts_events_data)

    # Create flexion-extension events
    fe_events_data = [
        {"Event Info": "Movement Start", "Time (sec)": 2.0},
        {"Event Info": "Movement End", "Time (sec)": 9.0},
    ]
    fe_events_df = pd.DataFrame(fe_events_data)

    # Write all sheets to Excel
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Walk sheets
        walk_df.to_excel(
            writer,
            sheet_name=f"{study_id}_Slow_Walking",
            index=False,
        )
        walking_events_df.to_excel(
            writer,
            sheet_name=f"{study_id}_Walk0001",
            index=False,
        )

        # Sit-to-stand sheets
        sts_df.to_excel(
            writer,
            sheet_name=f"{study_id}_SitToStand",
            index=False,
        )
        sts_events_df.to_excel(
            writer,
            sheet_name=f"{study_id}_StoS_Events",
            index=False,
        )

        # Flexion-extension sheets
        fe_df.to_excel(
            writer,
            sheet_name=f"{study_id}_FlexExt",
            index=False,
        )
        fe_events_df.to_excel(
            writer,
            sheet_name=f"{study_id}_FE_Events",
            index=False,
        )

    return {
        "excel_path": excel_path,
        # New dictionary format for all maneuvers
        "data_sheets": {
            "walk": f"{study_id}_Slow_Walking",
            "sit_to_stand": f"{study_id}_SitToStand",
            "flexion_extension": f"{study_id}_FlexExt",
        },
        "events_sheets": {
            "walk": f"{study_id}_Walk0001",
            "sit_to_stand": f"{study_id}_StoS_Events",
            "flexion_extension": f"{study_id}_FE_Events",
        },
        # Legacy keys for backward compatibility with existing tests
        "data_sheet": f"{study_id}_Slow_Walking",
        "events_sheet": f"{study_id}_Walk0001",
        "events_data": walking_events_data,
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
