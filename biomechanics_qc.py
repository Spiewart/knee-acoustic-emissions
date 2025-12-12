"""Quality control utilities for biomechanics recordings.

Provides a QC check for flexion–extension maneuvers: given a DataFrame with a
time column and knee angle column, assesses whether the recording contains a
sustained, periodic oscillation near the requested frequency with sufficient
amplitude.

The QC is designed to be robust and fast on long, uniformly sampled audio-
synchronized frames by operating on resampled knee angle signals.

Usage example:

    from pathlib import Path
    import pickle
    import pandas as pd
    from biomechanics_qc import qc_flexion_extension

    df = pd.read_pickle(Path("/path/to/Synced/Left_flexion_extension.pkl"))
    passed, coverage = qc_flexion_extension(
        df=df,
        time_col="TIME",
        knee_angle_col="Knee Angle Z",
        target_freq_hz=0.25,
        tail_length_s=5.0,
    )
    print(passed, coverage)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


@dataclass
class FlexExtQCParams:
    """Parameters for flexion–extension QC.

    Attributes:
        target_freq_hz: Expected maneuver frequency in Hz (e.g., 0.25).
        tail_length_s: Seconds at the end to exclude from QC.
        min_peak_amplitude_deg: Minimum magnitude (absolute) of extension peak
            expected for valid cycles (default: 60 deg).
        period_tolerance_frac: Allowed fractional deviation from target period
            (default: 0.3 → ±30%).
        min_coverage_frac: Minimum fraction of recording (excluding tail) that
            must be covered by valid cycles to pass QC (default: 0.8).
        resample_hz: Resampling rate for knee angle (default: 25 Hz).
    """

    target_freq_hz: float = 0.25
    tail_length_s: float = 5.0
    min_peak_amplitude_deg: float = 60.0
    period_tolerance_frac: float = 0.3
    min_coverage_frac: float = 0.8
    resample_hz: float = 25.0


def _timedelta_to_seconds(series: pd.Series) -> np.ndarray:
    """Convert a timedelta-like pandas Series to seconds as float."""
    # Handle both timedelta64[ns] and python timedelta
    values = pd.to_timedelta(series).dt.total_seconds().to_numpy()
    return values


def _resample_uniform(
    time_s: np.ndarray, signal: np.ndarray, fs: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample irregular (or high-rate) signal onto a uniform time grid.

    Uses linear interpolation on a uniform grid spanning
    [time_s.min(), time_s.max()].
    """
    if len(time_s) == 0:
        return np.array([]), np.array([])
    t0, t1 = float(np.min(time_s)), float(np.max(time_s))
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        return np.array([]), np.array([])
    n = int(np.floor((t1 - t0) * fs)) + 1
    t_uniform = np.linspace(t0, t1, n)
    sig_uniform = np.interp(t_uniform, time_s, signal)
    return t_uniform, sig_uniform


def qc_flexion_extension(
    df: pd.DataFrame,
    time_col: str,
    knee_angle_col: str,
    target_freq_hz: float,
    tail_length_s: float,
    *,
    params: FlexExtQCParams | None = None,
) -> Tuple[bool, float]:
    """Quality-control check for flexion–extension maneuvers.

    Checks whether the knee angle exhibits sustained cyclic behavior at the
    specified frequency with sufficient amplitude.

    Args:
        df: DataFrame with time and knee angle columns.
        time_col: Name of the time column (timedelta-like).
        knee_angle_col: Name of the knee angle column (degrees,
            flexion negative).
        target_freq_hz: Expected frequency in Hz (e.g., 0.25).
        tail_length_s: Seconds to exclude from the end of the recording.
        params: Optional `FlexExtQCParams` to override defaults.

    Returns:
                (passed, coverage_frac):
                        - passed: True if coverage ≥ min_coverage_frac.
                        - coverage_frac: Fraction of duration (excluding tail)
                            covered by valid cycles.
    """
    if params is None:
        params = FlexExtQCParams(
            target_freq_hz=target_freq_hz,
            tail_length_s=tail_length_s,
        )

    if df.empty or time_col not in df or knee_angle_col not in df:
        return False, 0.0

    # Extract time in seconds and knee angle as float
    time_s = _timedelta_to_seconds(df[time_col])
    angle = pd.to_numeric(df[knee_angle_col], errors="coerce").to_numpy()

    # Drop NaNs
    valid = np.isfinite(time_s) & np.isfinite(angle)
    time_s = time_s[valid]
    angle = angle[valid]
    if len(time_s) < 3:
        return False, 0.0

    # Exclude tail
    t_end = np.max(time_s)
    mask_tail = time_s <= (t_end - params.tail_length_s)
    time_s = time_s[mask_tail]
    angle = angle[mask_tail]
    if len(time_s) < 3:
        return False, 0.0

    # Resample to a modest uniform rate for robust peak detection
    t_uni, ang_uni = _resample_uniform(time_s, angle, params.resample_hz)
    if len(t_uni) < 3:
        return False, 0.0

    # We expect extension peaks to be negative and large in magnitude
    # (~ -70 deg)
    # Detect peaks on the inverted signal (so negative becomes positive)
    inv = -ang_uni
    # Peak height threshold (converted to positive magnitude)
    height_thresh = params.min_peak_amplitude_deg
    peaks, _ = find_peaks(inv, height=height_thresh)

    if len(peaks) < 2:
        return False, 0.0

    # Compute periods between consecutive extension peaks
    peak_times = t_uni[peaks]
    periods = np.diff(peak_times)
    target_period = 1.0 / params.target_freq_hz
    lo = target_period * (1.0 - params.period_tolerance_frac)
    hi = target_period * (1.0 + params.period_tolerance_frac)

    valid_cycles = (periods >= lo) & (periods <= hi)
    if not np.any(valid_cycles):
        return False, 0.0

    # Estimate coverage as total time spanned by valid cycles divided by
    # recording duration
    # A cycle is time from peak i to peak i+1 when period is valid
    valid_durations = periods[valid_cycles]
    total_valid_time = float(np.sum(valid_durations))
    total_time = float(np.max(time_s) - np.min(time_s))
    coverage = (total_valid_time / total_time) if total_time > 0 else 0.0

    passed = coverage >= params.min_coverage_frac
    return passed, coverage


if __name__ == "__main__":
    # Simple CLI for quick QC checks
    import argparse

    parser = argparse.ArgumentParser(
        description="Flexion–extension QC checker"
    )
    parser.add_argument("pkl", help="Path to synchronized pickle file")
    parser.add_argument(
        "--knee",
        default="Knee Angle Z",
        help="Knee angle column name (default: Knee Angle Z)",
    )
    parser.add_argument(
        "--time",
        default="TIME",
        help="Time column name (default: TIME)",
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
    passed, coverage = qc_flexion_extension(
        df=df,
        time_col=args.time,
        knee_angle_col=args.knee,
        target_freq_hz=args.freq,
        tail_length_s=args.tail,
    )
    print(f"QC passed: {passed}, coverage: {coverage:.2%}")
