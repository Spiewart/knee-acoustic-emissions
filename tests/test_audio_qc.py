from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from audio_qc import (
    qc_audio_directory,
    qc_audio_flexion_extension,
    qc_audio_sit_to_stand,
    qc_audio_walk,
)


def _pulse_series(times_s: list[float], duration_s: float = 10.0, fs: float = 52000.0) -> pd.DataFrame:
    n = int(duration_s * fs)
    tt = np.arange(n) / fs
    signal = np.zeros(n)
    width = int(0.05 * fs)
    for t in times_s:
        idx = int(t * fs)
        signal[idx:idx + width] = 1.0
    df = pd.DataFrame({
        "tt": tt,
        "ch1": signal,
        "ch2": signal,
    })
    return df


def test_qc_flexion_extension_passes_periodic_signal():
    # 0.25 Hz -> 4s period; create three pulses spaced 4s apart
    df = _pulse_series([1.0, 5.0, 9.0], duration_s=12.0)
    passed, coverage = qc_audio_flexion_extension(
        df=df,
        time_col="tt",
        audio_channels=["ch1", "ch2"],
        target_freq_hz=0.25,
        tail_length_s=1.0,
    )
    assert passed is True
    assert coverage > 0.5


def test_qc_flexion_extension_fails_wrong_period():
    # Pulses 1s apart → period mismatch vs 4s target
    df = _pulse_series([1.0, 2.0, 3.0], duration_s=6.0)
    passed, coverage = qc_audio_flexion_extension(
        df=df,
        time_col="tt",
        audio_channels=["ch1", "ch2"],
        target_freq_hz=0.25,
        tail_length_s=0.5,
    )
    assert passed is False
    assert coverage == 0.0


def test_qc_sit_to_stand_passes_two_cycles():
    df = _pulse_series([1.0, 5.0], duration_s=10.0)
    passed, coverage = qc_audio_sit_to_stand(
        df=df,
        time_col="tt",
        audio_channels=["ch1", "ch2"],
        target_freq_hz=0.25,
        tail_length_s=0.5,
    )
    assert passed is True
    assert coverage > 0.2


def test_qc_sit_to_stand_requires_channels():
    df = pd.DataFrame({"tt": [0, 1, 2]})
    passed, coverage = qc_audio_sit_to_stand(
        df=df,
        time_col="tt",
        audio_channels=["ch1", "ch2"],
        target_freq_hz=0.25,
        tail_length_s=0.5,
    )
    assert passed is False
    assert coverage == 0.0


def test_qc_returns_bandpower_ratio_when_requested():
    df = _pulse_series([1.0, 5.0, 9.0], duration_s=12.0)
    passed, coverage, ratio = qc_audio_flexion_extension(
        df=df,
        time_col="tt",
        audio_channels=["ch1", "ch2"],
        target_freq_hz=0.25,
        tail_length_s=1.0,
        return_bandpower=True,
    )
    assert passed is True
    assert coverage > 0.5
    assert 0.0 <= ratio <= 1.0
    assert ratio > 0.001


def test_qc_bandpower_threshold_blocks_when_ratio_low():
    df = _pulse_series([1.0, 5.0, 9.0], duration_s=12.0)
    passed, coverage, ratio = qc_audio_flexion_extension(
        df=df,
        time_col="tt",
        audio_channels=["ch1", "ch2"],
        target_freq_hz=0.25,
        tail_length_s=1.0,
        bandpower_min_ratio=0.95,
        return_bandpower=True,
    )
    assert passed is False
    assert coverage > 0.5
    assert ratio < 0.95


def _walking_signal(pass_specs: list[tuple[float, float]], gap_s: float = 3.0, fs: float = 1000.0) -> pd.DataFrame:
    """Create synthetic walking audio with multiple speed passes.

    pass_specs: list of (start_freq_hz, duration_s) pairs.
    """
    total_duration = sum(d for _, d in pass_specs) + gap_s * (len(pass_specs) - 1)
    n = int(total_duration * fs)
    tt = np.arange(n) / fs
    signal = np.zeros(n)

    t_cursor = 0.0
    for freq_hz, dur_s in pass_specs:
        step_period = 1.0 / freq_hz
        step_times = np.arange(t_cursor + 0.5 * step_period, t_cursor + dur_s, step_period)
        width = max(int(0.01 * fs), 1)
        for st in step_times:
            idx = int(st * fs)
            signal[idx: idx + width] = 1.0
        t_cursor += dur_s + gap_s

    df = pd.DataFrame({"tt": tt, "ch1": signal, "ch2": signal})
    return df


def test_qc_audio_walk_detects_passes_and_freqs():
    passes = [(1.0, 35.0), (1.5, 35.0), (2.0, 35.0)]
    df = _walking_signal(passes)
    results = qc_audio_walk(
        df=df,
        time_col="tt",
        audio_channels=["ch1", "ch2"],
    )

    assert len(results) == 3

    freqs = [r["gait_cycle_hz"] for r in results]
    assert freqs[0] == pytest.approx(1.0, rel=0.1)
    assert freqs[1] == pytest.approx(1.5, rel=0.1)
    assert freqs[2] == pytest.approx(2.0, rel=0.1)

    assert all(r["passed"] for r in results)
    assert all(r["coverage_frac"] > 0.6 for r in results)


def test_qc_audio_walk_requires_enough_peaks_in_pass():
    df = _walking_signal([(1.0, 5.0)], gap_s=5.0)
    results = qc_audio_walk(
        df=df,
        time_col="tt",
        audio_channels=["ch1", "ch2"],
        min_pass_peaks=10,
    )
    assert results == []


def _write_pickled_audio(tmp_path: Path, knee: str, folder: str, df: pd.DataFrame) -> Path:
    base_name = "test_audio"
    maneuver_dir = tmp_path / knee / folder
    maneuver_dir.mkdir(parents=True, exist_ok=True)
    (maneuver_dir / f"{base_name}.bin").touch()
    outputs_dir = maneuver_dir / f"{base_name}_outputs"
    outputs_dir.mkdir(exist_ok=True)
    pkl_path = outputs_dir / f"{base_name}_with_freq.pkl"
    df.to_pickle(pkl_path)
    return maneuver_dir


def test_qc_audio_directory_runs_per_maneuver(tmp_path: Path):
    flex_df = _pulse_series([1.0, 5.0, 9.0], duration_s=12.0)
    walk_df = _walking_signal([(1.2, 25.0)], gap_s=3.0, fs=1000.0)

    _write_pickled_audio(tmp_path, "Left Knee", "Flexion-Extension", flex_df)
    _write_pickled_audio(tmp_path, "Right Knee", "Walking", walk_df)

    results = qc_audio_directory(Path(tmp_path))

    assert len(results) == 2
    flex = next(r for r in results if r["maneuver"] == "flexion_extension")
    walk = next(r for r in results if r["maneuver"] == "walk")

    assert flex["passed"] is True
    assert walk["passes"]
    assert walk["passes"][0]["gait_cycle_hz"] == pytest.approx(1.2, rel=0.2)


def test_qc_audio_directory_filters_maneuver(tmp_path: Path):
    flex_df = _pulse_series([1.0, 5.0, 9.0], duration_s=12.0)
    _write_pickled_audio(tmp_path, "Left Knee", "Flexion-Extension", flex_df)

    results = qc_audio_directory(Path(tmp_path), maneuver="flexion_extension")

    assert len(results) == 1
    assert results[0]["maneuver"] == "flexion_extension"
