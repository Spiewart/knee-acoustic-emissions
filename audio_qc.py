"""Quality control utilities for audio recordings.

Provides QC checks for detecting periodic acoustic emissions during
flexion-extension maneuvers by analyzing voltage fluctuations in audio
channels at the target frequency.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def qc_audio_flexion_extension(
    df: pd.DataFrame,
    time_col: str,
    audio_channels: list[str],
    target_freq_hz: float,
    tail_length_s: float,
    *,
    min_peak_voltage: float = 0.1,
    period_tolerance_frac: float = 0.3,
    min_coverage_frac: float = 0.5,
    resample_hz: float = 25.0,
) -> Tuple[bool, float]:
    """QC check for audio during flexion-extension maneuvers.

    Detects periodic voltage fluctuations in audio channels at the
    specified frequency.

    Args:
        df: DataFrame with time and audio channel columns.
        time_col: Name of the time column (seconds or timedelta).
        audio_channels: List of audio channel column names
            (e.g., ['ch1', 'ch2', 'ch3', 'ch4']).
        target_freq_hz: Expected frequency in Hz (e.g., 0.25).
        tail_length_s: Seconds to exclude from the end of recording.
        min_peak_voltage: Minimum voltage fluctuation magnitude for
            valid peaks (default: 0.1).
        period_tolerance_frac: Allowed fractional deviation from target
            period (default: 0.3).
        min_coverage_frac: Minimum fraction of recording covered by
            valid cycles to pass QC (default: 0.5).
        resample_hz: Resampling rate for audio (default: 25 Hz).

    Returns:
        (passed, coverage_frac):
            - passed: True if coverage ≥ min_coverage_frac.
            - coverage_frac: Fraction of duration (excluding tail)
              covered by valid cycles.
    """
    if df.empty or time_col not in df:
        return False, 0.0

    # Check that at least one audio channel exists
    available_channels = [ch for ch in audio_channels if ch in df.columns]
    if not available_channels:
        return False, 0.0

    # Convert time to seconds
    time_s = pd.to_numeric(
        pd.to_timedelta(df[time_col]).dt.total_seconds(),
        errors='coerce'
    ).to_numpy()

    # Average all available audio channels
    audio_data = df[available_channels].mean(axis=1).to_numpy()

    # Drop NaNs
    valid = np.isfinite(time_s) & np.isfinite(audio_data)
    time_s = time_s[valid]
    audio_data = audio_data[valid]
    if len(time_s) < 3:
        return False, 0.0

    # Exclude tail
    t_end = np.max(time_s)
    mask_tail = time_s <= (t_end - tail_length_s)
    time_s = time_s[mask_tail]
    audio_data = audio_data[mask_tail]
    if len(time_s) < 3:
        return False, 0.0

    # Resample to uniform grid
    t_uni, aud_uni = _resample_uniform(time_s, audio_data, resample_hz)
    if len(t_uni) < 3:
        return False, 0.0

    # Detect peaks in audio signal
    peaks, _ = find_peaks(aud_uni, height=min_peak_voltage)

    if len(peaks) < 2:
        return False, 0.0

    # Compute periods between consecutive peaks
    peak_times = t_uni[peaks]
    periods = np.diff(peak_times)
    target_period = 1.0 / target_freq_hz
    lo = target_period * (1.0 - period_tolerance_frac)
    hi = target_period * (1.0 + period_tolerance_frac)

    valid_cycles = (periods >= lo) & (periods <= hi)
    if not np.any(valid_cycles):
        return False, 0.0

    # Estimate coverage
    valid_durations = periods[valid_cycles]
    total_valid_time = float(np.sum(valid_durations))
    total_time = float(np.max(time_s) - np.min(time_s))
    coverage = (total_valid_time / total_time) if total_time > 0 else 0.0

    passed = coverage >= min_coverage_frac
    return passed, coverage


def _resample_uniform(
    time_s: np.ndarray, signal: np.ndarray, fs: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample irregular signal onto a uniform time grid."""
    if len(time_s) == 0:
        return np.array([]), np.array([])
    t0, t1 = float(np.min(time_s)), float(np.max(time_s))
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        return np.array([]), np.array([])
    n = int(np.floor((t1 - t0) * fs)) + 1
    t_uniform = np.linspace(t0, t1, n)
    sig_uniform = np.interp(t_uniform, time_s, signal)
    return t_uniform, sig_uniform


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Audio flexion-extension QC checker"
    )
    parser.add_argument("pkl", help="Path to audio pickle file")
    parser.add_argument(
        "--time",
        default="tt",
        help="Time column name (default: tt)",
    )
    parser.add_argument(
        "--freq",
        type=float,
        default=0.25,
        help="Target frequency in Hz (default: 0.25)",
    )
    parser.add_argument(
        "--tail",
        type=float,
        default=5.0,
        help="Tail length in seconds to exclude (default: 5.0)",
    )

    args = parser.parse_args()

    df = pd.read_pickle(args.pkl)
    passed, coverage = qc_audio_flexion_extension(
        df=df,
        time_col=args.time,
        audio_channels=['ch1', 'ch2', 'ch3', 'ch4'],
        target_freq_hz=args.freq,
        tail_length_s=args.tail,
    )
    print(f"Audio QC passed: {passed}, coverage: {coverage:.2%}")
