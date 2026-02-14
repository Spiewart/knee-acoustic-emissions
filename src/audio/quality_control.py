"""Quality control utilities for audio recordings.

Provides QC checks for detecting periodic acoustic emissions during
flexion-extension maneuvers by analyzing voltage fluctuations in audio
channels at the target frequency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import find_peaks, welch

from src.orchestration.participant import get_audio_file_name


def _load_walk_events_from_biomechanics(
    biomech_file: Path,
    study_name: str = "AOA",
) -> Optional[dict[str, list[tuple[str, float]]]]:
    """Load walking events from biomechanics Excel file.

    Reads the walk event sheet from the biomechanics Excel file and extracts
    events associated with each walking speed. Sheet name, column names, and
    speed keywords are all resolved through the study config.

    Args:
        biomech_file: Path to the biomechanics Excel file.
        study_name: Study identifier for config dispatch.

    Returns:
        Dictionary mapping speed code to list of (event_info, time_sec)
        tuples, or None if sheet not found or required columns missing.
    """
    from src.studies import get_study_config

    study_config = get_study_config(study_name)
    base_name = study_config.get_walk_event_sheet_base_name()
    event_col = study_config.get_biomechanics_event_column()
    time_col = study_config.get_biomechanics_time_column()
    speed_keywords = study_config.get_speed_event_keywords()
    speed_codes = set(study_config.get_speed_code_map().values())

    # Try study-prefixed sheet name first, then bare base name
    raw_id = biomech_file.stem.split("_")[0]
    try:
        _, numeric_id = study_config.parse_participant_id(raw_id)
        study_id = study_config.format_study_prefix(numeric_id)
        prefixed_name = f"{study_id}_{base_name}"
    except (ValueError, IndexError):
        prefixed_name = None

    df = None
    for sheet_name in filter(None, [prefixed_name, base_name]):
        try:
            df = pd.read_excel(biomech_file, sheet_name=sheet_name)
            break
        except (FileNotFoundError, ValueError):
            continue

    if df is None:
        return None

    if not {event_col, time_col}.issubset(df.columns):
        return None

    # Build speed code set from config
    all_speed_codes = set(speed_keywords.values())
    events_by_speed: dict[str, list[tuple[str, float]]] = {
        code: [] for code in all_speed_codes
    }

    current_speed: str | None = None
    for _, row in df.iterrows():
        event_info = str(row[event_col]).strip()
        time_sec = float(row[time_col])

        # Check for speed section header keywords
        for keyword, code in speed_keywords.items():
            if keyword in event_info:
                current_speed = code
                break

        if current_speed:
            events_by_speed[current_speed].append(
                (event_info, time_sec),
            )

    return events_by_speed


def _parse_pass_intervals_from_events(
    events_by_speed: dict[str, list[tuple[str, float]]],
) -> dict[str, list[dict[str, float | int]]]:
    """Parse pass start/end times from walking event list.

    Scans events for "Pass X Start" and "Pass X End" patterns, extracting
    pass numbers and time intervals. Matches end events with corresponding
    start events by pass number to build complete intervals.

    Args:
        events_by_speed: Dictionary mapping speed code to event list
                        (from _load_walk_events_from_biomechanics).

    Returns:
        Dictionary mapping speed code to list of pass dictionaries with keys:
        - "pass_num" (int): Pass number
        - "start_s" (float): Start time in seconds
        - "end_s" (float): End time in seconds (or None if unpaired)
    """
    passes_by_speed: dict[str, list[dict[str, float | int]]] = {
        speed: [] for speed in events_by_speed
    }

    for speed, events in events_by_speed.items():
        for event_info, time_sec in events:
            # Look for "Pass X Start" pattern
            if "Pass" in event_info and "Start" in event_info:
                try:
                    parts = event_info.split()
                    idx = parts.index("Pass")
                    pass_num = int(parts[idx + 1])
                    passes_by_speed[speed].append(
                        {"pass_num": pass_num, "start_s": time_sec, "end_s": None}
                    )
                except (ValueError, IndexError):
                    pass
            # Look for "Pass X End" pattern to close the interval
            elif "Pass" in event_info and "End" in event_info:
                try:
                    parts = event_info.split()
                    idx = parts.index("Pass")
                    pass_num = int(parts[idx + 1])
                    # Find matching start and set end_s
                    for p in reversed(passes_by_speed[speed]):
                        if p["pass_num"] == pass_num and p["end_s"] is None:
                            p["end_s"] = time_sec
                            break
                except (ValueError, IndexError):
                    pass

    # Filter out incomplete passes
    for speed in passes_by_speed:
        passes_by_speed[speed] = [
            p for p in passes_by_speed[speed] if p["end_s"] is not None
        ]

    return passes_by_speed


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
    bandpower_min_ratio: float | None = None,
    bandpower_rel_width: float = 0.25,
    return_bandpower: bool = False,
) -> tuple[bool, float] | tuple[bool, float, float]:
    """QC check for audio during flexion-extension maneuvers.

    Detects periodic voltage fluctuations in audio channels at the
    specified frequency.

    Returns (passed, coverage_frac) unless return_bandpower is True, in
    which case (passed, coverage_frac, bandpower_ratio) is returned.
    coverage_frac is the fraction of duration (excluding tail) covered
    by valid cycles.
    """
    return _qc_periodic_audio(
        df=df,
        time_col=time_col,
        audio_channels=audio_channels,
        target_freq_hz=target_freq_hz,
        tail_length_s=tail_length_s,
        min_peak_voltage=min_peak_voltage,
        period_tolerance_frac=period_tolerance_frac,
        min_coverage_frac=min_coverage_frac,
        resample_hz=resample_hz,
        min_peaks_required=2,
        bandpower_min_ratio=bandpower_min_ratio,
        bandpower_rel_width=bandpower_rel_width,
        return_bandpower=return_bandpower,
    )


def qc_audio_sit_to_stand(
    df: pd.DataFrame,
    time_col: str,
    audio_channels: list[str],
    target_freq_hz: float,
    tail_length_s: float,
    *,
    min_peak_voltage: float = 0.08,
    period_tolerance_frac: float = 0.35,
    min_coverage_frac: float = 0.2,
    resample_hz: float = 25.0,
    bandpower_min_ratio: float | None = None,
    bandpower_rel_width: float = 0.3,
    return_bandpower: bool = False,
) -> tuple[bool, float] | tuple[bool, float, float]:
    """QC check for audio during sit-to-stand maneuvers.

    Uses the same periodic detection approach as flexion-extension but
    with slightly looser defaults and lower coverage expectations since
    fewer cycles may be present.

    Returns (passed, coverage_frac) unless return_bandpower is True, in
    which case (passed, coverage_frac, bandpower_ratio) is returned.
    coverage_frac is the fraction of duration (excluding tail) covered
    by valid cycles.
    """
    return _qc_periodic_audio(
        df=df,
        time_col=time_col,
        audio_channels=audio_channels,
        target_freq_hz=target_freq_hz,
        tail_length_s=tail_length_s,
        min_peak_voltage=min_peak_voltage,
        period_tolerance_frac=period_tolerance_frac,
        min_coverage_frac=min_coverage_frac,
        resample_hz=resample_hz,
        min_peaks_required=2,
        bandpower_min_ratio=bandpower_min_ratio,
        bandpower_rel_width=bandpower_rel_width,
        return_bandpower=return_bandpower,
    )


def _qc_walk_single_pass(
    df: pd.DataFrame,
    time_col: str,
    audio_channels: list[str],
    *,
    resample_hz: float = 100.0,
    min_peak_height: float = 0.005,
    peak_prominence: float = 0.001,
    min_step_hz: float = 0.2,
    max_step_hz: float = 5.0,
    min_steps: int = 1,
    period_tolerance_frac: float = 1.0,
    min_coverage_frac: float = 0.01,
    bandpower_min_ratio: float | None = None,
    bandpower_rel_width: float = 0.35,
) -> list[dict[str, float | bool | int | str]]:
    """QC check for a single pre-segmented walking pass (e.g., from synced file).

    Treats the entire DataFrame as one walking interval and evaluates
    heel strike regularity without gap-based splitting. Uses extremely lenient
    defaults suited to short, pre-segmented synced pass files with varying
    acoustic quality across different walking speeds.

    Args:
        df: Audio DataFrame containing one walking pass.
        time_col: Column containing timestamps.
        audio_channels: List of audio channel column names.
        resample_hz: Target uniform resample rate for detection.
        min_peak_height: Minimum heel-strike peak height (extreme: 0.005).
        peak_prominence: Minimum prominence for heel-strike detection (extreme: 0.001).
        min_step_hz: Lower bound on plausible step rate (0.2 Hz).
        max_step_hz: Upper bound on plausible step rate (5.0 Hz).
        min_steps: Minimum heel strikes required (1).
        period_tolerance_frac: Allowed fractional deviation around median step period (1.0).
        min_coverage_frac: Minimum fraction of interval covered by regular steps (0.01).
        bandpower_min_ratio: Optional minimum PSD bandpower ratio around detected step rate.
        bandpower_rel_width: Half-width (fractional) of the PSD band used for bandpower ratio.

    Returns:
        List with single dict if pass is valid, else empty list.
    """
    if df.empty or time_col not in df:
        return []

    available_channels = [ch for ch in audio_channels if ch in df.columns]
    if not available_channels:
        return []

    time_s = pd.to_numeric(
        pd.to_timedelta(df[time_col], unit="s").dt.total_seconds(),
        errors="coerce",
    ).to_numpy()

    audio_data = df[available_channels].mean(axis=1).to_numpy()

    valid = np.isfinite(time_s) & np.isfinite(audio_data)
    time_s = time_s[valid]
    audio_data = audio_data[valid]
    if len(time_s) < 2:
        return []

    t_uni, aud_uni = _resample_uniform(time_s, audio_data, resample_hz)
    if len(t_uni) < 2:
        return []

    heel_times = _detect_heel_strikes(
        t_uni,
        aud_uni,
        resample_hz=resample_hz,
        min_peak_height=min_peak_height,
        peak_prominence=peak_prominence,
        min_step_hz=min_step_hz,
        max_step_hz=max_step_hz,
    )
    if len(heel_times) < min_steps:
        return []

    start_s, end_s = float(np.min(time_s)), float(np.max(time_s))
    pass_result = _evaluate_walk_interval(
        heel_times=heel_times,
        interval=(start_s, end_s),
        period_tolerance_frac=period_tolerance_frac,
        min_coverage_frac=min_coverage_frac,
        min_step_hz=min_step_hz,
        max_step_hz=max_step_hz,
    )
    if pass_result is None:
        return []

    mask = (t_uni >= start_s) & (t_uni <= end_s)
    bandpower_ratio = _bandpower_ratio(
        signal=aud_uni[mask],
        fs=resample_hz,
        target_freq_hz=pass_result["gait_cycle_hz"],
        bandpower_rel_width=bandpower_rel_width,
    )

    meets_bandpower = (
        True
        if bandpower_min_ratio is None
        else bandpower_ratio >= bandpower_min_ratio
    )
    passed = bool(pass_result["passed"] and meets_bandpower)

    return [
        {
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": end_s - start_s,
            "heel_strike_count": pass_result["heel_strike_count"],
            "gait_cycle_hz": pass_result["gait_cycle_hz"],
            "coverage_frac": pass_result["coverage_frac"],
            "bandpower_ratio": bandpower_ratio,
            "passed": passed,
        }
    ]


def qc_audio_walk(
    df: pd.DataFrame,
    time_col: str,
    audio_channels: list[str],
    *,
    resample_hz: float = 100.0,
    min_peak_height: float = 0.02,
    peak_prominence: float = 0.02,
    min_step_hz: float = 0.5,
    max_step_hz: float = 3.0,
    min_gap_s: float = 2.0,
    min_pass_peaks: int = 6,
    period_tolerance_frac: float = 0.5,
    min_coverage_frac: float = 0.2,
    bandpower_min_ratio: float | None = None,
    bandpower_rel_width: float = 0.35,
    biomech_file: Path | None = None,
    use_frequency_segmentation: bool = False,
    frequency_tolerance_frac: float = 0.06,
    study_name: str = "AOA",
) -> list[dict[str, float | bool | int | str]]:
    """QC check for walking maneuvers with variable gait speeds.

    If biomech_file is provided, uses Walk0001 sheet to segment passes by
    speed (SS/NS/FS) and pass number. Otherwise, if use_frequency_segmentation
    is True, segments passes by step frequency clustering (each cluster
    represents a distinct walking speed). Otherwise, detects passes via gap analysis.

    Args:
        df: Audio DataFrame.
        time_col: Column containing timestamps (seconds or timedelta convertible).
        audio_channels: List of audio channel column names.
        resample_hz: Target uniform resample rate for detection.
        min_peak_height: Minimum heel-strike peak height (after mean-channel averaging).
        peak_prominence: Minimum prominence for heel-strike detection.
        min_step_hz: Lower bound on plausible step rate.
        max_step_hz: Upper bound on plausible step rate.
        min_gap_s: Gap size that separates consecutive passes (if gap-based mode).
        min_pass_peaks: Minimum heel strikes required inside a pass.
        period_tolerance_frac: Allowed fractional deviation around the median step period.
        min_coverage_frac: Minimum fraction of interval covered by regular steps.
        bandpower_min_ratio: Optional minimum PSD bandpower ratio around the detected step rate.
        bandpower_rel_width: Half-width (fractional) of the PSD band used for bandpower ratio.
        biomech_file: Optional path to biomechanics Excel file with Walk0001 sheet.
        use_frequency_segmentation: If True, segment passes by step frequency clustering.
        frequency_tolerance_frac: Fractional tolerance for grouping similar frequencies (0.2 = 20%).

    Returns:
        List of dicts, one per detected pass, with keys:
        start_s, end_s, duration_s, heel_strike_count, gait_cycle_hz,
        coverage_frac, bandpower_ratio, speed (if biomech provided), pass_num (if biomech provided), passed.
    """

    if df.empty or time_col not in df:
        return []

    available_channels = [ch for ch in audio_channels if ch in df.columns]
    if not available_channels:
        return []

    time_s = pd.to_numeric(
        pd.to_timedelta(df[time_col], unit="s").dt.total_seconds(),
        errors="coerce",
    ).to_numpy()

    audio_data = df[available_channels].mean(axis=1).to_numpy()

    valid = np.isfinite(time_s) & np.isfinite(audio_data)
    time_s = time_s[valid]
    audio_data = audio_data[valid]
    if len(time_s) < 3:
        return []

    t_uni, aud_uni = _resample_uniform(time_s, audio_data, resample_hz)
    if len(t_uni) < 3:
        return []

    heel_times = _detect_heel_strikes(
        t_uni,
        aud_uni,
        resample_hz=resample_hz,
        min_peak_height=min_peak_height,
        peak_prominence=peak_prominence,
        min_step_hz=min_step_hz,
        max_step_hz=max_step_hz,
    )
    if len(heel_times) < min_pass_peaks:
        return []

    # Determine how to segment passes
    if biomech_file and biomech_file.exists():
        # Biomechanics file takes priority
        events_by_speed = _load_walk_events_from_biomechanics(
            biomech_file, study_name=study_name,
        )
        if events_by_speed:
            passes_by_speed = _parse_pass_intervals_from_events(events_by_speed)
            intervals = []
            speed_pass_map: dict[tuple[float, float], tuple[str, int]] = {}
            for speed, passes in passes_by_speed.items():
                for pass_info in passes:
                    start_s_interval = float(pass_info["start_s"])
                    end_s_interval = float(pass_info["end_s"])
                    intervals.append((start_s_interval, end_s_interval))
                    speed_pass_map[
                        (start_s_interval, end_s_interval)
                    ] = (speed, int(pass_info["pass_num"]))
        else:
            # Fall back to frequency-based or gap-based
            if use_frequency_segmentation:
                intervals = _identify_walk_speed_intervals_by_frequency(
                    heel_times=heel_times,
                    min_pass_peaks=min_pass_peaks,
                    frequency_tolerance_frac=frequency_tolerance_frac,
                    extra_tolerances=(0.05, 0.04, 0.03, 0.025),
                )
                if not intervals:
                    intervals = _identify_walk_speed_intervals(
                        heel_times=heel_times,
                        min_gap_s=min_gap_s,
                        min_pass_peaks=min_pass_peaks,
                    )
            else:
                intervals = _identify_walk_speed_intervals(
                    heel_times=heel_times,
                    min_gap_s=min_gap_s,
                    min_pass_peaks=min_pass_peaks,
                )
            speed_pass_map = {}
    else:
        # Use frequency-based or gap-based segmentation
        if use_frequency_segmentation:
            intervals = _identify_walk_speed_intervals_by_frequency(
                heel_times=heel_times,
                min_pass_peaks=min_pass_peaks,
                frequency_tolerance_frac=frequency_tolerance_frac,
                extra_tolerances=(0.05, 0.04, 0.03, 0.025),
            )
            if not intervals:
                intervals = _identify_walk_speed_intervals(
                    heel_times=heel_times,
                    min_gap_s=min_gap_s,
                    min_pass_peaks=min_pass_peaks,
                )
        else:
            intervals = _identify_walk_speed_intervals(
                heel_times=heel_times,
                min_gap_s=min_gap_s,
                min_pass_peaks=min_pass_peaks,
            )
        speed_pass_map = {}

    results: list[dict[str, float | bool | int | str]] = []

    for start_s, end_s in intervals:
        pass_result = _evaluate_walk_interval(
            heel_times=heel_times,
            interval=(start_s, end_s),
            period_tolerance_frac=period_tolerance_frac,
            min_coverage_frac=min_coverage_frac,
            min_step_hz=min_step_hz,
            max_step_hz=max_step_hz,
        )
        if pass_result is None:
            continue

        mask = (t_uni >= start_s) & (t_uni <= end_s)
        bandpower_ratio = _bandpower_ratio(
            signal=aud_uni[mask],
            fs=resample_hz,
            target_freq_hz=pass_result["gait_cycle_hz"],
            bandpower_rel_width=bandpower_rel_width,
        )

        meets_bandpower = (
            True
            if bandpower_min_ratio is None
            else bandpower_ratio >= bandpower_min_ratio
        )
        passed = bool(pass_result["passed"] and meets_bandpower)

        result_dict: dict[str, float | bool | int | str] = {
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": end_s - start_s,
            "heel_strike_count": pass_result["heel_strike_count"],
            "gait_cycle_hz": pass_result["gait_cycle_hz"],
            "coverage_frac": pass_result["coverage_frac"],
            "bandpower_ratio": bandpower_ratio,
            "passed": passed,
        }

        # Add speed and pass_num if from biomechanics
        interval_key = (start_s, end_s)
        if interval_key in speed_pass_map:
            speed, pass_num = speed_pass_map[interval_key]
            result_dict["speed"] = speed
            result_dict["pass_num"] = pass_num

        results.append(result_dict)

    return results


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


def _bandpower_ratio(
    signal: np.ndarray,
    fs: float,
    target_freq_hz: float,
    bandpower_rel_width: float,
) -> float:
    """Compute fractional power in a band around target_freq_hz."""
    if len(signal) < 4 or not np.isfinite(target_freq_hz) or target_freq_hz <= 0:
        return 0.0

    centered = signal - float(np.mean(signal))
    desired_len = max(int(fs * 8.0), 256)
    nperseg = min(len(centered), desired_len)
    freqs, psd = welch(centered, fs=fs, nperseg=nperseg)
    if len(freqs) == 0:
        return 0.0

    band_lo = max(0.0, target_freq_hz * (1.0 - bandpower_rel_width))
    band_hi = target_freq_hz * (1.0 + bandpower_rel_width)
    band_mask = (freqs >= band_lo) & (freqs <= band_hi)
    if not np.any(band_mask):
        return 0.0

    band_power = float(trapezoid(psd[band_mask], freqs[band_mask]))
    total_power = float(trapezoid(psd, freqs))
    if total_power <= 0:
        return 0.0

    return band_power / total_power


def _detect_heel_strikes(
    time_s: np.ndarray,
    signal: np.ndarray,
    *,
    resample_hz: float,
    min_peak_height: float,
    peak_prominence: float,
    min_step_hz: float,
    max_step_hz: float,
) -> np.ndarray:
    """Detect heel strikes via peaks on absolute signal."""
    if len(time_s) < 3:
        return np.array([])

    min_distance_samples = int(resample_hz / max_step_hz) if max_step_hz > 0 else 1
    min_distance_samples = max(min_distance_samples, 1)

    peaks, _ = find_peaks(
        np.abs(signal),
        height=min_peak_height,
        prominence=peak_prominence,
        distance=min_distance_samples,
    )
    if len(peaks) == 0:
        return np.array([])

    heel_times = time_s[peaks]
    heel_times = heel_times[np.isfinite(heel_times)]
    heel_times.sort()
    return heel_times


def _identify_walk_speed_intervals_by_frequency(
    heel_times: np.ndarray,
    *,
    min_pass_peaks: int,
    frequency_tolerance_frac: float = 0.2,
    extra_tolerances: tuple[float, ...] = (),
) -> list[tuple[float, float]]:
    """Split heel strikes into passes based on step frequency clustering.

    Groups consecutive heel strikes by similar frequency (step rate).
    Each frequency cluster represents a distinct walking speed and becomes
    a separate pass interval.

    Args:
        heel_times: Array of detected heel strike times.
        min_pass_peaks: Minimum heel strikes required in a pass.
        frequency_tolerance_frac: Fractional tolerance for grouping similar frequencies.

    Returns:
        List of (start_time, end_time) tuples for each frequency cluster.
    """
    def _cluster_with_tol(times: np.ndarray, tol: float) -> list[tuple[float, float]]:
        if len(times) < min_pass_peaks:
            return []

        periods = np.diff(times)
        if len(periods) == 0:
            return []

        frequencies = 1.0 / periods
        intervals_local: list[tuple[float, float]] = []

        current_freq = frequencies[0]
        cluster_start_idx = 0

        for i in range(1, len(frequencies)):
            freq = frequencies[i]
            freq_deviation = abs(freq - current_freq) / current_freq

            if freq_deviation > tol:
                cluster_end_idx = i
                cluster_size = cluster_end_idx - cluster_start_idx + 1

                if cluster_size >= min_pass_peaks:
                    start_s = float(times[cluster_start_idx])
                    end_s = float(times[cluster_end_idx])
                    intervals_local.append((start_s, end_s))

                current_freq = freq
                cluster_start_idx = i

        cluster_end_idx = len(frequencies)
        cluster_size = cluster_end_idx - cluster_start_idx + 1
        if cluster_size >= min_pass_peaks:
            start_s = float(times[cluster_start_idx])
            end_s = float(times[cluster_end_idx])
            intervals_local.append((start_s, end_s))

        return intervals_local

    tolerances = [frequency_tolerance_frac, *extra_tolerances]
    best_intervals: list[tuple[float, float]] = []

    for tol in tolerances:
        intervals = _cluster_with_tol(heel_times, tol)
        if len(intervals) > len(best_intervals):
            best_intervals = intervals
        if len(intervals) >= 3:
            return intervals

    # If only two intervals found, try splitting the longest one with tighter t-shirt sizes
    if len(best_intervals) == 2:
        longest_idx = int(np.argmax([iv[1] - iv[0] for iv in best_intervals]))
        longest = best_intervals[longest_idx]
        mask = (heel_times >= longest[0]) & (heel_times <= longest[1])
        sub_times = heel_times[mask]
        for tol in (0.03, 0.02, 0.015):
            split = _cluster_with_tol(sub_times, tol)
            if len(split) > 1:
                new_intervals = best_intervals[:longest_idx] + split + best_intervals[longest_idx + 1 :]
                return new_intervals

    return best_intervals


def _identify_walk_speed_intervals(
    heel_times: np.ndarray,
    *,
    min_gap_s: float,
    min_pass_peaks: int,
) -> list[tuple[float, float]]:
    """Split heel strikes into passes separated by large gaps."""
    if len(heel_times) < min_pass_peaks:
        return []

    intervals: list[tuple[float, float]] = []
    start = heel_times[0]
    prev = heel_times[0]
    count = 1

    for ht in heel_times[1:]:
        if (ht - prev) > min_gap_s:
            if count >= min_pass_peaks:
                intervals.append((start, prev))
            start = ht
            count = 1
        else:
            count += 1
        prev = ht

    if count >= min_pass_peaks:
        intervals.append((start, prev))

    return intervals


def _evaluate_walk_interval(
    *,
    heel_times: np.ndarray,
    interval: tuple[float, float],
    period_tolerance_frac: float,
    min_coverage_frac: float,
    min_step_hz: float,
    max_step_hz: float,
) -> Optional[dict[str, Union[float, bool, int]]]:
    """Compute gait metrics and QC for a single pass."""
    start_s, end_s = interval
    if end_s <= start_s:
        return None
    mask = (heel_times >= start_s) & (heel_times <= end_s)
    window = heel_times[mask]
    if len(window) < 2:
        return None

    periods = np.diff(window)
    median_period = float(np.median(periods)) if len(periods) else np.nan
    if not np.isfinite(median_period) or median_period <= 0:
        return None

    freq_hz = 1.0 / median_period
    if freq_hz < min_step_hz or freq_hz > max_step_hz:
        return None

    lo = median_period * (1.0 - period_tolerance_frac)
    hi = median_period * (1.0 + period_tolerance_frac)
    valid = (periods >= lo) & (periods <= hi)
    interval_duration = end_s - start_s
    if not np.any(valid) or interval_duration <= 0:
        coverage = 0.0
    else:
        coverage = float(np.sum(periods[valid]) / interval_duration)

    passed = bool(coverage >= min_coverage_frac and np.sum(valid) >= 1)

    return {
        "heel_strike_count": int(len(window)),
        "gait_cycle_hz": freq_hz,
        "coverage_frac": coverage,
        "passed": passed,
    }


def _qc_periodic_audio(
    df: pd.DataFrame,
    time_col: str,
    audio_channels: list[str],
    target_freq_hz: float,
    tail_length_s: float,
    *,
    min_peak_voltage: float,
    period_tolerance_frac: float,
    min_coverage_frac: float,
    resample_hz: float,
    min_peaks_required: int,
    bandpower_min_ratio: float | None,
    bandpower_rel_width: float,
    return_bandpower: bool,
) -> tuple[bool, float] | tuple[bool, float, float]:
    """Shared periodic QC routine with optional spectral gating."""
    def _fail() -> tuple[bool, float] | tuple[bool, float, float]:
        return (False, 0.0, 0.0) if return_bandpower else (False, 0.0)

    if df.empty or time_col not in df:
        return _fail()

    available_channels = [ch for ch in audio_channels if ch in df.columns]
    if not available_channels:
        return _fail()

    time_s = pd.to_numeric(
        pd.to_timedelta(df[time_col], unit="s").dt.total_seconds(),
        errors="coerce",
    ).to_numpy()

    audio_data = df[available_channels].mean(axis=1).to_numpy()

    valid = np.isfinite(time_s) & np.isfinite(audio_data)
    time_s = time_s[valid]
    audio_data = audio_data[valid]
    if len(time_s) < 3:
        return _fail()

    t_end = np.max(time_s)
    mask_tail = time_s <= (t_end - tail_length_s)
    time_s = time_s[mask_tail]
    audio_data = audio_data[mask_tail]
    if len(time_s) < 3:
        return _fail()

    t_uni, aud_uni = _resample_uniform(time_s, audio_data, resample_hz)
    if len(t_uni) < 3:
        return _fail()

    envelope = np.abs(aud_uni)
    target_period = 1.0 / target_freq_hz
    window_sec = min(target_period * 0.05, 0.25)
    smooth_window = max(int(resample_hz * window_sec), 3)
    kernel = np.ones(smooth_window) / smooth_window
    smoothed = np.convolve(envelope, kernel, mode="same")

    min_distance = max(
        1,
        int(resample_hz * target_period * (1.0 - period_tolerance_frac)),
    )

    peaks, _ = find_peaks(
        smoothed,
        height=min_peak_voltage,
        distance=min_distance,
    )
    if len(peaks) < min_peaks_required:
        return _fail()

    peak_times = t_uni[peaks]
    periods = np.diff(peak_times)
    if len(periods) == 0:
        return _fail()

    target_period = 1.0 / target_freq_hz
    lo = target_period * (1.0 - period_tolerance_frac)
    hi = target_period * (1.0 + period_tolerance_frac)

    valid_cycles = (periods >= lo) & (periods <= hi)
    if not np.any(valid_cycles):
        return _fail()

    valid_durations = periods[valid_cycles]
    total_valid_time = float(np.sum(valid_durations))
    total_time = float(np.max(time_s) - np.min(time_s))
    coverage_fraction = (
        total_valid_time / total_time
    ) if total_time > 0 else 0.0

    bandpower_ratio = _bandpower_ratio(
        signal=aud_uni,
        fs=resample_hz,
        target_freq_hz=target_freq_hz,
        bandpower_rel_width=bandpower_rel_width,
    )

    meets_coverage = coverage_fraction >= min_coverage_frac
    meets_bandpower = (
        True
        if bandpower_min_ratio is None
        else bandpower_ratio >= bandpower_min_ratio
    )
    passed_qc = bool(meets_coverage and meets_bandpower)

    if return_bandpower:
        return passed_qc, coverage_fraction, bandpower_ratio
    return passed_qc, coverage_fraction


DEFAULT_CHANNELS = ["ch1", "ch2", "ch3", "ch4"]


def _run_qc_on_file(
    pkl_path: Path,
    maneuver: Literal["flexion_extension", "sit_to_stand", "walk"],
    *,
    time_col: str,
    audio_channels: list[str],
    target_freq_hz: float,
    tail_length_s: float,
    bandpower_min_ratio: float | None,
    resample_hz_walk: float,
    min_pass_peaks: int,
    min_gap_s: float,
    use_frequency_segmentation: bool = False,
    frequency_tolerance_frac: float = 0.06,
    study_name: str = "AOA",
) -> dict[str, object]:
    df = pd.read_pickle(pkl_path)

    if maneuver == "walk":
        walk_results = qc_audio_walk(
            df=df,
            time_col=time_col,
            audio_channels=audio_channels,
            resample_hz=resample_hz_walk,
            min_pass_peaks=min_pass_peaks,
            min_gap_s=min_gap_s,
            bandpower_min_ratio=bandpower_min_ratio,
            use_frequency_segmentation=use_frequency_segmentation,
            frequency_tolerance_frac=frequency_tolerance_frac,
            study_name=study_name,
        )
        return {
            "path": str(pkl_path),
            "maneuver": maneuver,
            "passes": walk_results,
        }

    passed, coverage = (
        qc_audio_flexion_extension(
            df=df,
            time_col=time_col,
            audio_channels=audio_channels,
            target_freq_hz=target_freq_hz,
            tail_length_s=tail_length_s,
            bandpower_min_ratio=bandpower_min_ratio,
        )
        if maneuver == "flexion_extension"
        else qc_audio_sit_to_stand(
            df=df,
            time_col=time_col,
            audio_channels=audio_channels,
            target_freq_hz=target_freq_hz,
            tail_length_s=tail_length_s,
            bandpower_min_ratio=bandpower_min_ratio,
        )
    )

    return {
        "path": str(pkl_path),
        "maneuver": maneuver,
        "passed": bool(passed),
        "coverage": float(coverage),
    }


def qc_audio_directory(
    participant_dir: Path,
    *,
    time_col: str = "tt",
    audio_channels: list[str] | None = None,
    maneuver: Literal["walk", "sit_to_stand", "flexion_extension", "all"] = "all",
    target_freq_hz: float = 0.25,
    tail_length_s: float = 5.0,
    bandpower_min_ratio: float | None = None,
    resample_hz_walk: float = 100.0,
    min_pass_peaks: int = 6,
    min_gap_s: float = 2.0,
    use_frequency_segmentation: bool = False,
    frequency_tolerance_frac: float = 0.06,
    study_name: str = "AOA",
) -> list[dict[str, object]]:
    """Run QC across a participant directory.

    Iterates Left/Right knee folders, detects maneuver folders, loads the
    processed audio pickle, and runs the appropriate QC.
    """
    from src.studies import get_study_config

    study_config = get_study_config(study_name)

    maneuver_keys: list[Literal["walk", "sit_to_stand", "flexion_extension"]] = [
        "walk", "sit_to_stand", "flexion_extension",
    ]

    channels = audio_channels or DEFAULT_CHANNELS
    selected = set(maneuver_keys) if maneuver == "all" else {maneuver}

    results: list[dict[str, object]] = []

    for knee_side in ("left", "right"):
        knee_dir = participant_dir / study_config.get_knee_directory_name(knee_side)
        if not knee_dir.exists():
            continue

        for key in maneuver_keys:
            if key not in selected:
                continue

            folder_name = study_config.get_maneuver_directory_name(key)
            contains_terms = study_config.get_maneuver_search_terms(key)
            maneuver_dir = knee_dir / folder_name
            if not maneuver_dir.exists() and contains_terms:
                # Allow variants like "Sit-to-Stand" by matching all required terms.
                for candidate in knee_dir.iterdir():
                    if not candidate.is_dir():
                        continue
                    lower_name = candidate.name.lower()
                    if all(term in lower_name for term in contains_terms):
                        maneuver_dir = candidate
                        break

            if not maneuver_dir.exists():
                continue

            # Prefer synced per-pass files when available
            synced_dir = maneuver_dir / "Synced"
            if key == "walk" and synced_dir.exists():
                for pkl_path in sorted(synced_dir.glob("*.pkl")):
                    df = pd.read_pickle(pkl_path)
                    # Evaluate each synced file as a single pre-segmented pass
                    pass_results = _qc_walk_single_pass(
                        df=df,
                        time_col=time_col,
                        audio_channels=channels,
                        resample_hz=resample_hz_walk,
                        bandpower_min_ratio=bandpower_min_ratio,
                    )
                    results.append(
                        {
                            "knee": knee_side,
                            "maneuver": key,
                            "path": str(pkl_path),
                            "passes": pass_results,
                        }
                    )
                continue

            # Fallback: use primary processed file
            try:
                audio_base = Path(
                    get_audio_file_name(maneuver_dir, with_freq=False)
                ).name
                pickle_base = Path(
                    get_audio_file_name(maneuver_dir, with_freq=True)
                ).name
            except FileNotFoundError:
                continue

            pkl_path = (
                maneuver_dir / f"{audio_base}_outputs" / f"{pickle_base}.pkl"
            )
            if not pkl_path.exists():
                continue

            qc_result = _run_qc_on_file(
                pkl_path=pkl_path,
                maneuver=key,  # type: ignore[arg-type]
                time_col=time_col,
                audio_channels=channels,
                target_freq_hz=target_freq_hz,
                tail_length_s=tail_length_s,
                bandpower_min_ratio=bandpower_min_ratio,
                resample_hz_walk=resample_hz_walk,
                use_frequency_segmentation=use_frequency_segmentation,
                frequency_tolerance_frac=frequency_tolerance_frac,
                min_pass_peaks=min_pass_peaks,
                min_gap_s=min_gap_s,
                study_name=study_name,
            )

            qc_result.update({"knee": knee_side, "maneuver": key})
            results.append(qc_result)

    return results


def _build_cli_parser() -> "argparse.ArgumentParser":
    import argparse

    parser = argparse.ArgumentParser(
        description="Audio QC utilities for maneuver-specific checks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--time",
        default="tt",
        help="Time column name in the audio pickle",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=DEFAULT_CHANNELS,
        help="Audio channel columns to average for QC",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    file_parser = subparsers.add_parser(
        "file", help="Run QC on a single audio pickle"
    )
    file_parser.add_argument("pkl", help="Path to audio pickle file")
    file_parser.add_argument(
        "--maneuver",
        required=True,
        choices=["flexion_extension", "sit_to_stand", "walk"],
        help="Maneuver type to QC",
    )
    file_parser.add_argument(
        "--freq",
        type=float,
        default=0.25,
        help="Target frequency (Hz) for flexion-extension or sit-to-stand",
    )
    file_parser.add_argument(
        "--tail",
        type=float,
        default=5.0,
        help="Tail length (s) to exclude for periodic maneuvers",
    )
    file_parser.add_argument(
        "--bandpower-min-ratio",
        type=float,
        help="Optional minimum bandpower ratio around target/detected freq",
    )
    file_parser.add_argument(
        "--resample-walk",
        type=float,
        default=100.0,
        help="Resample rate (Hz) for walking heel-strike detection",
    )
    file_parser.add_argument(
        "--min-pass-peaks",
        type=int,
        default=6,
        help="Minimum heel strikes required in a walking pass",
    )
    file_parser.add_argument(
        "--min-gap-s",
        type=float,
        default=2.0,
        help="Gap (s) that splits walking passes",
    )
    file_parser.add_argument(
        "--use-frequency-segmentation",
        action="store_true",
        help="Use step frequency clustering instead of gap-based segmentation for walking",
    )
    file_parser.add_argument(
        "--frequency-tolerance-frac",
        type=float,
        default=0.06,
        help="Fractional tolerance for grouping similar frequencies (0.06 = 6%%)",
    )

    dir_parser = subparsers.add_parser(
        "dir", help="Run QC across a participant directory"
    )
    dir_parser.add_argument(
        "participant_dir",
        help="Path to participant directory (contains Left/Right Knee)",
    )
    dir_parser.add_argument(
        "--maneuver",
        choices=["walk", "sit_to_stand", "flexion_extension", "all"],
        default="all",
        help="Restrict QC to a single maneuver or run all",
    )
    dir_parser.add_argument(
        "--freq",
        type=float,
        default=0.25,
        help="Target frequency (Hz) for non-walk maneuvers",
    )
    dir_parser.add_argument(
        "--tail",
        type=float,
        default=5.0,
        help="Tail length (s) to exclude for non-walk maneuvers",
    )
    dir_parser.add_argument(
        "--bandpower-min-ratio",
        type=float,
        help="Optional minimum bandpower ratio around target/detected freq",
    )
    dir_parser.add_argument(
        "--resample-walk",
        type=float,
        default=100.0,
        help="Resample rate (Hz) for walking heel-strike detection",
    )
    dir_parser.add_argument(
        "--min-pass-peaks",
        type=int,
        default=6,
        help="Minimum heel strikes required in a walking pass",
    )
    dir_parser.add_argument(
        "--min-gap-s",
        type=float,
        default=2.0,
        help="Gap (s) that splits walking passes",
    )
    dir_parser.add_argument(
        "--use-frequency-segmentation",
        action="store_true",
        help="Use step frequency clustering instead of gap-based segmentation for walking",
    )
    dir_parser.add_argument(
        "--frequency-tolerance-frac",
        type=float,
        default=0.06,
        help="Fractional tolerance for grouping similar frequencies (0.06 = 6%%)",
    )

    return parser


def _print_file_result(result: dict[str, object]) -> None:
    maneuver = result["maneuver"]
    print(f"QC result for {maneuver} on {result['path']}")
    if maneuver == "walk":
        passes = result.get("passes", [])
        has_numbering = any(("speed" in p) or ("pass_num" in p) for p in passes)
        if passes and not has_numbering:
            grouped = _group_passes_by_speed(passes)
            if not grouped:
                print("  No walking passes detected")
            else:
                print("  Grouped by speed (no pass numbers):")
                for g in grouped:
                    print(
                        f"  {g['speed_label']}: freq={g['gait_cycle_hz']:.2f} Hz, "
                        f"coverage={g['coverage_frac']:.2%}, "
                        f"bandpower={g['bandpower_ratio']:.3f}, "
                        f"passed={g['passed']} (passes={g['pass_count']}, pass_rate={g['pass_rate']:.0%})"
                    )
        else:
            for idx, p in enumerate(passes, start=1):
                print(
                    f"  Pass {idx}: freq={p['gait_cycle_hz']:.2f} Hz, "
                    f"coverage={p['coverage_frac']:.2%}, "
                    f"bandpower={p['bandpower_ratio']:.3f}, "
                    f"passed={p['passed']}"
                )
            if not passes:
                print("  No walking passes detected")
    else:
        print(
            f"  passed={result['passed']}, "
            f"coverage={result['coverage']:.2%}"
        )


def _print_dir_results(results: list[dict[str, object]]) -> None:
    if not results:
        print("No QC results found.")
        return
    for res in results:
        header = (
            f"{res['knee']} | {res['maneuver']} | {res['path']}"
        )
        print(header)
        if res["maneuver"] == "walk":
            passes = res.get("passes", [])
            has_numbering = any(("speed" in p) or ("pass_num" in p) for p in passes)
            if passes and not has_numbering:
                grouped = _group_passes_by_speed(passes)
                if not grouped:
                    print("  No walking passes detected")
                    continue
                print("  Grouped by speed (no pass numbers):")
                for g in grouped:
                    print(
                        f"  {g['speed_label']}: freq={g['gait_cycle_hz']:.2f} Hz, "
                        f"coverage={g['coverage_frac']:.2%}, "
                        f"bandpower={g['bandpower_ratio']:.3f}, "
                        f"passed={g['passed']} (passes={g['pass_count']}, pass_rate={g['pass_rate']:.0%})"
                    )
                continue
            if not passes:
                print("  No walking passes detected")
                continue
            for idx, p in enumerate(passes, start=1):
                print(
                    f"  Pass {idx}: freq={p['gait_cycle_hz']:.2f} Hz, "
                    f"coverage={p['coverage_frac']:.2%}, "
                    f"bandpower={p['bandpower_ratio']:.3f}, "
                    f"passed={p['passed']}"
                )
        else:
            print(
                f"  passed={res['passed']}, "
                f"coverage={res['coverage']:.2%}"
            )


def _group_passes_by_speed(
    passes: list[dict[str, object]],
    *,
    tolerance_frac: float = 0.06,
) -> list[dict[str, float | bool | int | str]]:
    """Group unlabeled passes by similar gait frequency.

    Uses a simple 1D clustering: sorted by frequency and started new group when
    the fractional deviation exceeds tolerance_frac.
    """
    if not passes:
        return []

    sorted_passes = sorted(passes, key=lambda p: float(p["gait_cycle_hz"]))
    groups: list[list[dict[str, object]]] = []
    current_group: list[dict[str, object]] = []
    current_freq: float | None = None

    for p in sorted_passes:
        freq = float(p["gait_cycle_hz"])
        if current_freq is None:
            current_group = [p]
            current_freq = freq
            continue

        deviation = abs(freq - current_freq) / current_freq if current_freq else 0.0
        if deviation <= tolerance_frac:
            current_group.append(p)
            # update representative frequency to mean of group for stability
            current_freq = float(np.mean([float(x["gait_cycle_hz"]) for x in current_group]))
        else:
            groups.append(current_group)
            current_group = [p]
            current_freq = freq

    if current_group:
        groups.append(current_group)

    grouped_results: list[dict[str, float | bool | int | str]] = []
    for idx, g in enumerate(groups, start=1):
        freqs = np.array([float(p["gait_cycle_hz"]) for p in g])
        coverages = np.array([float(p["coverage_frac"]) for p in g])
        bandpowers = np.array([float(p["bandpower_ratio"]) for p in g])
        durations = np.array([float(p["duration_s"]) for p in g])
        heel_counts = np.array([int(p["heel_strike_count"]) for p in g])
        passes_passed = np.array([bool(p["passed"]) for p in g])

        grouped_results.append(
            {
                "speed_label": f"Speed {idx}",
                "gait_cycle_hz": float(np.mean(freqs)),
                "coverage_frac": float(np.mean(coverages)),
                "bandpower_ratio": float(np.mean(bandpowers)),
                "duration_s": float(np.sum(durations)),
                "heel_strike_count": int(np.sum(heel_counts)),
                "passed": bool(np.all(passes_passed)),
                "pass_count": int(len(g)),
                "pass_rate": float(np.mean(passes_passed)),
            }
        )

    return grouped_results


def _cli_main() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()

    channels = list(args.channels)
    bandpower_min_ratio = args.bandpower_min_ratio

    if args.command == "file":
        pkl_path = Path(args.pkl)
        if not pkl_path.exists():
            print(f"Error: File does not exist: {pkl_path}")
            return
        result = _run_qc_on_file(
            pkl_path=pkl_path,
            maneuver=args.maneuver,
            time_col=args.time,
            audio_channels=channels,
            target_freq_hz=args.freq,
            tail_length_s=args.tail,
            bandpower_min_ratio=bandpower_min_ratio,
            resample_hz_walk=args.resample_walk,
            min_pass_peaks=args.min_pass_peaks,
            min_gap_s=args.min_gap_s,
            use_frequency_segmentation=args.use_frequency_segmentation,
            frequency_tolerance_frac=args.frequency_tolerance_frac,
        )
        _print_file_result(result)
    else:
        participant_dir = Path(args.participant_dir)
        if not participant_dir.exists():
            print(f"Error: Directory does not exist: {participant_dir}")
            return
        results = qc_audio_directory(
            participant_dir=participant_dir,
            time_col=args.time,
            audio_channels=channels,
            maneuver=args.maneuver,
            target_freq_hz=args.freq,
            tail_length_s=args.tail,
            bandpower_min_ratio=bandpower_min_ratio,
            resample_hz_walk=args.resample_walk,
            min_pass_peaks=args.min_pass_peaks,
            min_gap_s=args.min_gap_s,
            use_frequency_segmentation=args.use_frequency_segmentation,
            frequency_tolerance_frac=args.frequency_tolerance_frac,
        )
        _print_dir_results(results)


if __name__ == "__main__":
    _cli_main()
