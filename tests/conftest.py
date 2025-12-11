"""Global pytest fixtures for the test suite."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


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
        stomp_idx: stomp_idx + spike_samples,
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
    excel_path = motion_capture_dir / (
        f"AOA{study_id}_Biomechanics_Full_Set.xlsx"
    )

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
        pass_metadata = pd.DataFrame(walking_events_data)
        pass_metadata.to_excel(
            writer,
            sheet_name=f"AOA{study_id}_Walk0001",
            index=False,
        )

        # Speed-specific event sheets contain stomp/event timing data
        slow_events = pd.DataFrame([
            {"Event Info": "Sync Left", "Time (sec)": 16.23},
            {"Event Info": "Sync Right", "Time (sec)": 17.48},
            {"Event Info": "SS Pass 1 Start", "Time (sec)": 19.28},
            {"Event Info": "SS Pass 2 Start", "Time (sec)": 27.80},
        ])
        slow_events.to_excel(
            writer,
            sheet_name=f"AOA{study_id}_Slow_Walking_Events",
            index=False,
        )

        medium_events = pd.DataFrame([
            {"Event Info": "Sync Left", "Time (sec)": 16.23},
            {"Event Info": "Sync Right", "Time (sec)": 17.48},
            {"Event Info": "NS Pass 1 Start", "Time (sec)": 136.96},
            {"Event Info": "NS Pass 2 Start", "Time (sec)": 144.13},
        ])
        medium_events.to_excel(
            writer,
            sheet_name=f"AOA{study_id}_Medium_Walking_Events",
            index=False,
        )

        fast_events = pd.DataFrame([
            {"Event Info": "Sync Left", "Time (sec)": 16.23},
            {"Event Info": "Sync Right", "Time (sec)": 17.48},
            {"Event Info": "FS Pass 1 Start", "Time (sec)": 210.15},
        ])
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
