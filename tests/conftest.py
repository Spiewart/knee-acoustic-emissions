"""Global pytest fixtures for the test suite.

⚠️ CRITICAL TESTING GUIDELINE ⚠️
==================================

DO NOT CREATE TEST DATA MANUALLY IN INDIVIDUAL TEST FILES!

This file contains CONSOLIDATED FIXTURE FACTORIES that are the single source of
truth for all test data creation. This pattern is MANDATORY for maintainability.

Why Consolidated Fixtures Matter
---------------------------------

1. **Single Source of Truth**: When metadata fields change (e.g., timedelta → float),
   update defaults in ONE place instead of 42+ scattered helper functions.

2. **Type Safety**: Factories always create valid instances with correct field types.
   No more ValidationError surprises from outdated test helpers.

3. **Maintainability**: Adding a new field? Update 4 factories, not 50+ test files.

4. **Consistency**: All tests use the same default values, making behavior predictable.

Historical Context: Phase 5 Metadata Refactoring
-------------------------------------------------

During Phase 5, we converted all time fields from `timedelta` to `float` (seconds).
Before consolidation: Had to update 42 separate helper functions across 15+ test files.
After consolidation: Only updated 4 factory functions in this file.
Time saved: ~8 hours. Bugs prevented: Many.

How to Use Factory Fixtures
----------------------------

Factories are automatically available to all tests via pytest's fixture discovery:

    def test_example(synchronization_factory):
        '''Factory fixtures are auto-injected by pytest.'''
        
        # ✅ CORRECT: Use factory with custom overrides
        sync = synchronization_factory(
            audio_sync_time=5.0,
            sync_duration=120.0,
            knee="left"
        )
        
        # Factories provide sensible defaults for all required fields
        assert sync.study == "AOA"  # Default value
        assert sync.audio_sync_time == 5.0  # Your override

    # ❌ INCORRECT: Do not create instances manually
    def test_bad_pattern():
        from src.metadata import Synchronization
        sync = Synchronization(...)  # NO! Use the factory!

Available Factories
-------------------

- synchronization_factory(**overrides) → Synchronization
- synchronization_metadata_factory(**overrides) → SynchronizationMetadata
- audio_processing_factory(**overrides) → AudioProcessing
- movement_cycle_factory(**overrides) → MovementCycle

See individual factory docstrings below for default values and usage examples.

Time Field Format (Post Phase-5)
---------------------------------

ALL time fields use float (seconds), NOT timedelta:

    # ✅ CORRECT
    sync = synchronization_factory(audio_sync_time=5.0)

    # ❌ INCORRECT
    from datetime import timedelta
    sync = synchronization_factory(audio_sync_time=timedelta(seconds=5.0))

When Adding New Metadata Fields
--------------------------------

1. Update the metadata class in src/metadata.py
2. Update the corresponding factory in THIS FILE with a sensible default
3. Update tests to use the new factory (if defaults changed)
4. DO NOT create new helper functions in individual test files!

For detailed guidelines, see:
- README.md "Testing" section
- ai_instructions.md "Testing Guidelines" section

"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def dummy_pkl_file(tmp_path):
    """Create a dummy pickle file with a DataFrame."""
    df = pd.DataFrame(
        {
            "tt": np.arange(4096) / 1000.0,
            "ch1": np.random.randn(4096),
            "ch2": np.random.randn(4096),
            "ch3": np.random.randn(4096),
            "ch4": np.random.randn(4096),
        }
    )
    pkl_path = tmp_path / "test.pkl"
    df.to_pickle(pkl_path)
    # Create a dummy meta file
    meta_path = tmp_path / "test_meta.json"
    with open(meta_path, "w") as f:
        json.dump({"fs": 1000.0}, f)

    return pkl_path


def _create_acoustic_legend(participant_dir: Path) -> Path:
    """Write an acoustic legend Excel file matching expected schema."""

    excel_path = participant_dir / "acoustic_file_legend.xlsx"
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
        [None, None, None, None, None, None],
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
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(
            writer,
            sheet_name="Acoustic Notes",
            index=False,
            header=False,
        )

    return excel_path


def _generate_audio_dataframe(
    duration_seconds: int = 40,
    sample_rate: int = 2000,
) -> pd.DataFrame:
    """Generate synthetic audio with a clear stomp spike."""

    n_points = duration_seconds * sample_rate
    time_points = np.arange(n_points) / sample_rate

    audio_df = pd.DataFrame(
        {
            "tt": time_points,
            "ch1": np.random.randn(n_points) * 0.01,
            "ch2": np.random.randn(n_points) * 0.01,
            "ch3": np.random.randn(n_points) * 0.01,
            "ch4": np.random.randn(n_points) * 0.01,
        }
    )

    stomp_time = 16.23
    stomp_idx = int(stomp_time * sample_rate)
    spike_samples = int(0.01 * sample_rate)
    audio_df.loc[
        stomp_idx : stomp_idx + spike_samples,
        ["ch1", "ch2", "ch3", "ch4"],
    ] = 10.0

    return audio_df


def _write_audio_files(
    knee_dir: Path,
    audio_df: pd.DataFrame,
) -> dict[str, Path]:
    """Create maneuver folders with .bin and processed pickle files."""

    maneuver_dirs = {
        "Flexion-Extension": knee_dir / "Flexion-Extension",
        "Sit-Stand": knee_dir / "Sit-Stand",
        "Walking": knee_dir / "Walking",
    }
    audio_paths: dict[str, Path] = {}

    for maneuver_name, maneuver_dir in maneuver_dirs.items():
        maneuver_dir.mkdir(parents=True, exist_ok=True)

        bin_file = maneuver_dir / "test_audio.bin"
        bin_file.touch()

        outputs_dir = maneuver_dir / "test_audio_outputs"
        outputs_dir.mkdir(exist_ok=True)

        # Base pickle without frequency data
        pkl_path = outputs_dir / "test_audio.pkl"
        audio_df.to_pickle(pkl_path)

        # Frequency-augmented pickle lives in the same outputs folder
        pkl_freq_path = outputs_dir / "test_audio_with_freq.pkl"
        audio_df.to_pickle(pkl_freq_path)

        audio_paths[maneuver_name] = pkl_path

    return audio_paths


def _walk_sheet(
    study_id: str,
    speed_code: str,
    pass_number: int,
    time_points: np.ndarray,
) -> pd.DataFrame:
    uid_base = (
        f"V3D\\AOA{study_id}_Walk{pass_number:04d}_"
        f"{speed_code}P{pass_number}_Filt.c3d"
    )
    data_dict = {
        f"{uid_base}": ["Frame", ""] + time_points.tolist(),
        f"{uid_base}.1": [
            "LAnkleAngles",
            "X",
        ]
        + (np.sin(time_points + pass_number) * 10).tolist(),
        f"{uid_base}.2": [
            "LAnkleAngles",
            "Y",
        ]
        + (np.cos(time_points + pass_number) * 5).tolist(),
    }
    return pd.DataFrame(data_dict)


def _non_walk_sheet(
    study_id: str,
    maneuver_stub: str,
    time_points: np.ndarray,
) -> pd.DataFrame:
    uid_base = f"V3D\\AOA{study_id}_{maneuver_stub}0001_Filt.c3d"
    data_dict = {
        f"{uid_base}": ["Frame", ""] + time_points.tolist(),
        f"{uid_base}.1": [
            "LKneeAngles",
            "X",
        ]
        + (np.sin(time_points) * 15).tolist(),
        f"{uid_base}.2": [
            "RKneeAngles",
            "X",
        ]
        + (np.cos(time_points) * 15).tolist(),
    }
    return pd.DataFrame(data_dict)


def _create_biomechanics_excel(
    motion_capture_dir: Path,
    study_id: str,
) -> dict[str, object]:
    """Create biomechanics Excel with walking, StoS,
    and FE data plus events.
    """

    motion_capture_dir.mkdir(parents=True, exist_ok=True)
    excel_path = motion_capture_dir / (f"AOA{study_id}_Biomechanics_Full_Set.xlsx")

    time_points = np.linspace(0, 10, 50)

    walk_sheets = {
        "Slow": pd.concat(
            [
                _walk_sheet(study_id, "SS", 1, time_points),
                _walk_sheet(study_id, "SS", 2, time_points + 0.5),
            ],
            axis=1,
        ),
        "Medium": pd.concat(
            [
                _walk_sheet(study_id, "NS", 1, time_points + 1),
                _walk_sheet(study_id, "NS", 2, time_points + 1.5),
            ],
            axis=1,
        ),
        "Fast": _walk_sheet(study_id, "FS", 1, time_points + 2),
    }
    sts_df = _non_walk_sheet(study_id, "SitToStand", time_points)
    fe_df = _non_walk_sheet(study_id, "FlexExt", time_points)

    walking_events_data = [
        {"Event Info": "Sync Left", "Time (sec)": 16.23},
        {"Event Info": "Sync Right", "Time (sec)": 17.48},
        {"Event Info": "SS Pass 1 Start", "Time (sec)": 19.28},
        {"Event Info": "SS Pass 1 End", "Time (sec)": 26.50},
        {"Event Info": "SS Pass 2 Start", "Time (sec)": 27.80},
        {"Event Info": "SS Pass 2 End", "Time (sec)": 34.95},
        {"Event Info": "NS Pass 1 Start", "Time (sec)": 136.96},
        {"Event Info": "NS Pass 1 End", "Time (sec)": 144.00},
        {"Event Info": "NS Pass 2 Start", "Time (sec)": 144.13},
        {"Event Info": "NS Pass 2 End", "Time (sec)": 151.25},
        {"Event Info": "FS Pass 1 Start", "Time (sec)": 210.15},
        {"Event Info": "FS Pass 1 End", "Time (sec)": 216.80},
    ]

    sts_events_df = pd.DataFrame(
        [
            {"Event Info": "Sync Left", "Time (sec)": 10.50},
            {"Event Info": "Sync Right", "Time (sec)": 11.75},
            {"Event Info": "Movement Start", "Time (sec)": 5.0},
            {"Event Info": "Movement End", "Time (sec)": 8.5},
        ]
    )
    fe_events_df = pd.DataFrame(
        [
            {"Event Info": "Sync Left", "Time (sec)": 8.90},
            {"Event Info": "Sync Right", "Time (sec)": 10.15},
            {"Event Info": "Movement Start", "Time (sec)": 2.0},
            {"Event Info": "Movement End", "Time (sec)": 9.0},
        ]
    )

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for speed_label, df in walk_sheets.items():
            df.to_excel(
                writer,
                sheet_name=f"AOA{study_id}_{speed_label}_Walking",
                index=False,
            )

        # Walk0001 contains pass metadata (which pass belongs to which speed)
        pass_data = pd.DataFrame(walking_events_data)
        pass_data.to_excel(
            writer,
            sheet_name=f"AOA{study_id}_Walk0001",
            index=False,
        )

        # Speed-specific event sheets contain stomp/event timing data
        slow_events = pd.DataFrame(
            [
                {"Event Info": "Sync Left", "Time (sec)": 16.23},
                {"Event Info": "Sync Right", "Time (sec)": 17.48},
                {"Event Info": "SS Pass 1 Start", "Time (sec)": 19.28},
                {"Event Info": "SS Pass 2 Start", "Time (sec)": 27.80},
            ]
        )
        slow_events.to_excel(
            writer,
            sheet_name=f"AOA{study_id}_Slow_Walking_Events",
            index=False,
        )

        medium_events = pd.DataFrame(
            [
                {"Event Info": "Sync Left", "Time (sec)": 16.23},
                {"Event Info": "Sync Right", "Time (sec)": 17.48},
                {"Event Info": "NS Pass 1 Start", "Time (sec)": 136.96},
                {"Event Info": "NS Pass 2 Start", "Time (sec)": 144.13},
            ]
        )
        medium_events.to_excel(
            writer,
            sheet_name=f"AOA{study_id}_Medium_Walking_Events",
            index=False,
        )

        fast_events = pd.DataFrame(
            [
                {"Event Info": "Sync Left", "Time (sec)": 16.23},
                {"Event Info": "Sync Right", "Time (sec)": 17.48},
                {"Event Info": "FS Pass 1 Start", "Time (sec)": 210.15},
            ]
        )
        fast_events.to_excel(
            writer,
            sheet_name=f"AOA{study_id}_Fast_Walking_Events",
            index=False,
        )

        sts_df.to_excel(
            writer,
            sheet_name=f"AOA{study_id}_SitToStand",
            index=False,
        )
        sts_events_df.to_excel(
            writer,
            sheet_name=f"AOA{study_id}_StoS_Events",
            index=False,
        )

        fe_df.to_excel(
            writer,
            sheet_name=f"AOA{study_id}_FlexExt",
            index=False,
        )
        fe_events_df.to_excel(
            writer,
            sheet_name=f"AOA{study_id}_FE_Events",
            index=False,
        )

    return {
        "excel_path": excel_path,
        "data_sheets": {
            "walk": {
                speed.lower(): f"AOA{study_id}_{speed}_Walking"
                for speed in ["Slow", "Medium", "Fast"]
            },
            "sit_to_stand": f"AOA{study_id}_SitToStand",
            "flexion_extension": f"AOA{study_id}_FlexExt",
        },
        "events_sheets": {
            "walk_pass": f"AOA{study_id}_Walk0001",
            "sit_to_stand": f"AOA{study_id}_StoS_Events",
            "flexion_extension": f"AOA{study_id}_FE_Events",
        },
        "events_data": walking_events_data,
    }


@pytest.fixture
def syncd_walk(tmp_path):
    """Create synchronized walking data with realistic gait cycles.

    Returns:
        pd.DataFrame with 3-4 complete gait cycles (heel strike to heel strike)
    """
    np.random.seed(42)

    # Walking: 4 gait cycles at ~1 second per cycle
    num_samples = 4000  # 4 seconds at 1000 Hz
    time_array = np.linspace(0, 4, num_samples)

    # Gait cycle: knee angle oscillates with minima at heel strikes
    # Heel strikes at 0, 1.0, 2.0, 3.0, 4.0 seconds
    gait_freq = 1.0  # Hz (1 cycle per second)
    # Use -cos to get minima at integer seconds
    # Reduce noise to 0.1° to avoid spurious minima
    knee_angle = (
        20
        - 30 * np.cos(2 * np.pi * gait_freq * time_array)
        + np.random.randn(num_samples) * 0.1
    )

    # Acoustic events at heel strikes (impacts)
    acoustic_base = np.random.randn(num_samples) * 0.001
    f_ch_base = np.random.randn(num_samples) * 0.01

    # Add acoustic spikes at heel strikes (0, 1, 2, 3 seconds)
    for heel_strike_time in [0.0, 1.0, 2.0, 3.0]:
        idx = int(heel_strike_time * 1000)
        spike_width = 50  # 50ms spike
        acoustic_base[idx : idx + spike_width] += (
            np.exp(-np.arange(spike_width) / 10) * 0.1
        )
        f_ch_base[idx : idx + spike_width] += (
            np.exp(-np.arange(spike_width) / 10) * 2.0
        )

    syncd_df = pd.DataFrame(
        {
            "tt": time_array,
            "ch1": acoustic_base + np.random.randn(num_samples) * 0.0005,
            "ch2": acoustic_base + np.random.randn(num_samples) * 0.0005,
            "ch3": acoustic_base + np.random.randn(num_samples) * 0.0005,
            "ch4": acoustic_base + np.random.randn(num_samples) * 0.0005,
            "f_ch1": f_ch_base + np.random.randn(num_samples) * 0.005,
            "f_ch2": f_ch_base + np.random.randn(num_samples) * 0.005,
            "f_ch3": f_ch_base + np.random.randn(num_samples) * 0.005,
            "f_ch4": f_ch_base + np.random.randn(num_samples) * 0.005,
            "TIME": pd.to_timedelta(time_array, unit="s"),
            "Knee Angle Z": knee_angle,
        }
    )

    return syncd_df


@pytest.fixture
def syncd_sit_to_stand(tmp_path):
    """Create synchronized sit-to-stand data with standing and seated phases.

    Returns:
        pd.DataFrame with 3 sit-to-stand cycles (seated -> standing -> seated)
    """
    np.random.seed(43)

    # Sit-to-stand: 3 cycles over 15 seconds
    num_samples = 15000  # 15 seconds at 1000 Hz
    time_array = np.linspace(0, 15, num_samples)

    # Knee angle: seated (~90°), transition, standing (~10°)
    # Cycles at 0-5s, 5-10s, 10-15s
    knee_angle = np.zeros(num_samples)
    for cycle in range(3):
        start_idx = cycle * 5000
        # Seated phase (0-2s): ~90°
        knee_angle[start_idx : start_idx + 2000] = 90 + np.random.randn(2000) * 3
        # Transition (2-3s): rapid extension
        transition = np.linspace(90, 10, 1000) + np.random.randn(1000) * 5
        knee_angle[start_idx + 2000 : start_idx + 3000] = transition
        # Standing phase (3-5s): ~10°
        knee_angle[start_idx + 3000 : start_idx + 5000] = 10 + np.random.randn(2000) * 3

    # Acoustic activity during transitions (standing peaks)
    acoustic_base = np.random.randn(num_samples) * 0.001
    f_ch_base = np.random.randn(num_samples) * 0.01

    for cycle in range(3):
        # Peak acoustic activity during extension (standing)
        peak_idx = cycle * 5000 + 2500
        spike_width = 500  # 500ms of activity
        acoustic_base[peak_idx : peak_idx + spike_width] += (
            np.exp(-np.arange(spike_width) / 100) * 0.05
        )
        f_ch_base[peak_idx : peak_idx + spike_width] += (
            np.exp(-np.arange(spike_width) / 100) * 0.5
        )

    syncd_df = pd.DataFrame(
        {
            "tt": time_array,
            "ch1": acoustic_base + np.random.randn(num_samples) * 0.0005,
            "ch2": acoustic_base + np.random.randn(num_samples) * 0.0005,
            "ch3": acoustic_base + np.random.randn(num_samples) * 0.0005,
            "ch4": acoustic_base + np.random.randn(num_samples) * 0.0005,
            "f_ch1": f_ch_base + np.random.randn(num_samples) * 0.005,
            "f_ch2": f_ch_base + np.random.randn(num_samples) * 0.005,
            "f_ch3": f_ch_base + np.random.randn(num_samples) * 0.005,
            "f_ch4": f_ch_base + np.random.randn(num_samples) * 0.005,
            "TIME": pd.to_timedelta(time_array, unit="s"),
            "Knee Angle Z": knee_angle,
        }
    )

    return syncd_df


@pytest.fixture
def syncd_flexion_extension(tmp_path):
    """Create synchronized flexion-extension data with extension peaks.

    Returns:
        pd.DataFrame with 5 flexion-extension cycles
    """
    np.random.seed(44)

    # Flexion-extension: 5 cycles over 10 seconds
    num_samples = 10000  # 10 seconds at 1000 Hz
    time_array = np.linspace(0, 10, num_samples)

    # Knee angle: smooth oscillation between flexion (~90°) and extension (~10°)
    # Cycles every 2 seconds, extension peaks at 1, 3, 5, 7, 9 seconds
    cycle_freq = 0.5  # Hz (1 cycle per 2 seconds)
    knee_angle = (
        50
        + 40 * np.sin(2 * np.pi * cycle_freq * time_array - np.pi / 2)
        + np.random.randn(num_samples) * 3
    )

    # Acoustic activity during extension (peak extension = minimum knee angle)
    acoustic_base = np.random.randn(num_samples) * 0.001
    f_ch_base = np.random.randn(num_samples) * 0.01

    # Add activity at extension peaks (1, 3, 5, 7, 9 seconds)
    for peak_time in [1.0, 3.0, 5.0, 7.0, 9.0]:
        idx = int(peak_time * 1000)
        spike_width = 200  # 200ms activity window
        acoustic_base[idx : idx + spike_width] += (
            np.exp(-np.arange(spike_width) / 50) * 0.03
        )
        f_ch_base[idx : idx + spike_width] += np.exp(-np.arange(spike_width) / 50) * 0.3

    syncd_df = pd.DataFrame(
        {
            "tt": time_array,
            "ch1": acoustic_base + np.random.randn(num_samples) * 0.0005,
            "ch2": acoustic_base + np.random.randn(num_samples) * 0.0005,
            "ch3": acoustic_base + np.random.randn(num_samples) * 0.0005,
            "ch4": acoustic_base + np.random.randn(num_samples) * 0.0005,
            "f_ch1": f_ch_base + np.random.randn(num_samples) * 0.005,
            "f_ch2": f_ch_base + np.random.randn(num_samples) * 0.005,
            "f_ch3": f_ch_base + np.random.randn(num_samples) * 0.005,
            "f_ch4": f_ch_base + np.random.randn(num_samples) * 0.005,
            "TIME": pd.to_timedelta(time_array, unit="s"),
            "Knee Angle Z": knee_angle,
        }
    )

    return syncd_df


@pytest.fixture
def syncd_data(syncd_walk, tmp_path):
    """Provide pickle path and DataFrame for visualization tests."""

    syncd_copy = syncd_walk.copy(deep=True)
    # Visualization tests expect only 'tt' to represent time in this fixture
    if "TIME" in syncd_copy.columns:
        syncd_copy = syncd_copy.drop(columns=["TIME"])
    pkl_path = tmp_path / "syncd_fixture.pkl"
    syncd_copy.to_pickle(pkl_path)

    return pkl_path, syncd_copy


@pytest.fixture
def fake_participant_directory(tmp_path_factory):
    """Create a fake participant directory tree for integration tests."""

    project_dir = tmp_path_factory.mktemp("project")
    participant_dir = project_dir / "#1011"
    participant_dir.mkdir()

    legend_path = _create_acoustic_legend(participant_dir)

    left_knee_dir = participant_dir / "Left Knee"
    right_knee_dir = participant_dir / "Right Knee"
    left_knee_dir.mkdir()
    right_knee_dir.mkdir()

    audio_df = _generate_audio_dataframe()
    left_audio_paths = _write_audio_files(left_knee_dir, audio_df)
    right_audio_paths = _write_audio_files(right_knee_dir, audio_df)

    biomechanics_info = _create_biomechanics_excel(
        motion_capture_dir=participant_dir / "Motion Capture",
        study_id="1011",
    )

    return {
        "project_dir": project_dir,
        "participant_dir": participant_dir,
        "legend_file": legend_path,
        "audio_paths": {
            "left": left_audio_paths,
            "right": right_audio_paths,
        },
        "biomechanics": biomechanics_info,
    }


# ===== Metadata Factory Fixtures =====
# Centralized test data creation to reduce duplication across test suite


@pytest.fixture
def synchronization_factory():
    """Factory for creating Synchronization test records with sensible defaults.
    
    Usage:
        sync = synchronization_factory(audio_sync_time=5.0, knee="left")
    """
    from datetime import datetime
    from src.metadata import Synchronization
    
    def _create(**overrides):
        defaults = {
            # StudyMetadata
            "study": "AOA",
            "study_id": 1001,
            # BiomechanicsMetadata
            "linked_biomechanics": True,
            "biomechanics_file": "test_biomech.xlsx",
            "biomechanics_type": "Gonio",
            "biomechanics_sample_rate": 100.0,
            # AcousticsFile
            "audio_file_name": "test_audio.bin",
            "device_serial": "TEST001",
            "firmware_version": 1,
            "file_time": datetime(2024, 1, 1, 10, 0, 0),
            "file_size_mb": 100.0,
            "recording_date": datetime(2024, 1, 1),
            "recording_time": datetime(2024, 1, 1, 10, 0, 0),
            "knee": "left",
            "maneuver": "walk",
            "num_channels": 4,
            "sample_rate": 46875.0,
            "mic_1_position": "IPM",
            "mic_2_position": "IPL",
            "mic_3_position": "SPM",
            "mic_4_position": "SPL",
            # SynchronizationMetadata (all times in seconds as float)
            "audio_sync_time": 5.0,
            "bio_left_sync_time": 10.0,
            "sync_offset": 5.0,
            "aligned_audio_sync_time": 10.0,
            "aligned_biomechanics_sync_time": 10.0,
            "sync_method": "consensus",
            "consensus_time": 5.0,
            "rms_time": 5.0,
            "onset_time": 5.0,
            "freq_time": 5.0,
            # Synchronization-specific
            "sync_file_name": "test_sync.pkl",
            "processing_date": datetime(2024, 1, 1, 12, 0, 0),
            "processing_status": "success",
            "sync_duration": 120.0,
            "total_cycles_extracted": 0,
            "clean_cycles": 0,
            "outlier_cycles": 0,
            "mean_cycle_duration_s": 0.0,
            "median_cycle_duration_s": 0.0,
            "min_cycle_duration_s": 0.0,
            "max_cycle_duration_s": 0.0,
        }
        defaults.update(overrides)
        return Synchronization(**defaults)
    
    return _create


@pytest.fixture
def synchronization_metadata_factory():
    """Factory for creating SynchronizationMetadata test records.
    
    Usage:
        sync_meta = synchronization_metadata_factory(audio_sync_time=10.0)
    """
    from datetime import datetime
    from src.metadata import SynchronizationMetadata
    
    def _create(**overrides):
        defaults = {
            "study": "AOA",
            "study_id": 1001,
            "linked_biomechanics": True,
            "biomechanics_file": "test.xlsx",
            "biomechanics_type": "Gonio",
            "biomechanics_sample_rate": 100.0,
            "audio_file_name": "test_audio.wav",
            "device_serial": "TEST001",
            "firmware_version": 1,
            "file_time": datetime.now(),
            "file_size_mb": 10.0,
            "recording_date": datetime.now(),
            "recording_time": datetime.now(),
            "knee": "left",
            "maneuver": "walk",
            "num_channels": 4,
            "sample_rate": 46875.0,
            "mic_1_position": "IPM",
            "mic_2_position": "IPL",
            "mic_3_position": "SPM",
            "mic_4_position": "SPL",
            # All time fields in seconds (float)
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
        }
        defaults.update(overrides)
        return SynchronizationMetadata(**defaults)
    
    return _create


@pytest.fixture
def audio_processing_factory():
    """Factory for creating AudioProcessing test records.
    
    Usage:
        audio = audio_processing_factory(processing_status="success")
    """
    from datetime import datetime
    from src.metadata import AudioProcessing
    
    def _create(**overrides):
        defaults = {
            "study": "AOA",
            "study_id": 1001,
            "audio_file_name": "test_audio.bin",
            "device_serial": "TEST001",
            "firmware_version": 1,
            "file_time": datetime(2024, 1, 1, 10, 0, 0),
            "file_size_mb": 100.0,
            "recording_date": datetime(2024, 1, 1),
            "recording_time": datetime(2024, 1, 1, 10, 0, 0),
            "knee": "left",
            "maneuver": "walk",
            "num_channels": 4,
            "sample_rate": 46875.0,
            "mic_1_position": "IPM",
            "mic_2_position": "IPL",
            "mic_3_position": "SPM",
            "mic_4_position": "SPL",
            "processing_date": datetime(2024, 1, 1, 12, 0, 0),
            "processing_status": "success",
            "duration_seconds": 120.0,
        }
        defaults.update(overrides)
        return AudioProcessing(**defaults)
    
    return _create


@pytest.fixture
def movement_cycle_factory():
    """Factory for creating MovementCycle test records.
    
    Usage:
        cycle = movement_cycle_factory(cycle_number=1, cycle_duration_s=1.2)
    """
    from datetime import datetime
    from src.metadata import MovementCycle
    
    def _create(**overrides):
        defaults = {
            # StudyMetadata
            "study": "AOA",
            "study_id": 1001,
            # BiomechanicsMetadata
            "linked_biomechanics": True,
            "biomechanics_file": "test_biomech.xlsx",
            "biomechanics_type": "Gonio",
            "biomechanics_sample_rate": 100.0,
            # AcousticsFile
            "audio_file_name": "test_audio.bin",
            "device_serial": "TEST001",
            "firmware_version": 1,
            "file_time": datetime(2024, 1, 1, 10, 0, 0),
            "file_size_mb": 100.0,
            "recording_date": datetime(2024, 1, 1),
            "recording_time": datetime(2024, 1, 1, 10, 0, 0),
            "knee": "left",
            "maneuver": "walk",
            "num_channels": 4,
            "sample_rate": 46875.0,
            "mic_1_position": "IPM",
            "mic_2_position": "IPL",
            "mic_3_position": "SPM",
            "mic_4_position": "SPL",
            # SynchronizationMetadata (times in seconds)
            "audio_sync_time": 5.0,
            "sync_offset": 5.0,
            "sync_method": "consensus",
            # AudioProcessing
            "processing_date": datetime(2024, 1, 1, 12, 0, 0),
            "processing_status": "success",
            "duration_seconds": 120.0,
            # MovementCycle-specific
            "cycle_file": "test_cycle_01.pkl",
            "cycle_number": 1,
            "cycle_start_time": 0.0,
            "cycle_end_time": 1.2,
            "cycle_duration_s": 1.2,
        }
        defaults.update(overrides)
        return MovementCycle(**defaults)
    
    return _create
