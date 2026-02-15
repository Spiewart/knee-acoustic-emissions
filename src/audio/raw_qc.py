"""Raw audio quality control for detecting signal dropout and artifactual noise.

This module provides QC checks for unprocessed/raw audio recordings to identify:
1. Signal dropout from all or any of the microphones (silence/flatline detection)
2. Artifactual noise:
   - One-off or time-limited spikes (intermittent background noise)
   - Periodic background noise (consistent noise at specific frequencies)

Sections with dropout or artifacts are annotated with timestamps, and methods
are provided for clipping out poor quality sections while preserving timestamps
and synchronization metadata.

Usage Example
-------------
```python
from src.audio.raw_qc import run_raw_audio_qc, merge_bad_intervals, clip_bad_segments
import pandas as pd

# Load audio data
df = pd.read_pickle("audio_data.pkl")

# Run QC to detect dropout and artifacts
dropout_intervals, artifact_intervals = run_raw_audio_qc(df)

# Merge all bad intervals
bad_intervals = merge_bad_intervals(dropout_intervals, artifact_intervals)

# Optionally clip out bad segments
clean_df = clip_bad_segments(df, bad_intervals)

# The bad_intervals can be stored in processing logs as QC_not_passed
qc_not_passed = str(bad_intervals)  # Format for Excel logs
```

Quality Control Criteria
------------------------

**Dropout Detection:**
- Silence: RMS amplitude below threshold (default: 0.001)
- Flatline: Variance below threshold (default: 0.0001)
- Window size: 0.5 seconds (default)
- Minimum duration: 0.1 seconds (default)

**Artifact Detection:**
- One-off spikes: Values exceeding mean + N*std in local window (default: 5 sigma)
  Window size: 0.01 seconds (default)
  Minimum duration: 0.01 seconds (default)
- Periodic noise: Consistent background noise at specific frequencies (e.g., fans)
  Uses power spectral density analysis (Welch's method)
  Threshold range: 0-1 (default: 0.3, higher = less sensitive)

Thresholds can be adjusted based on signal characteristics and recording conditions.

Synchronization Support
-----------------------
Methods are provided to handle audio QC in the context of synchronized data:
- `adjust_bad_intervals_for_sync()`: Adjusts bad interval timestamps from audio
  coordinates to synchronized (biomechanics) coordinates using stomp times
- `check_cycle_in_bad_interval()`: Checks if a movement cycle overlaps with
  bad audio intervals and should be marked as failing audio QC

Integration with Processing Pipeline
------------------------------------
This module is integrated into the bin processing stage (`_process_bin_stage` in
`participant.py`). When processing .bin files:
1. Raw audio is read from .bin file
2. QC is performed to detect dropout and artifacts (including periodic noise)
3. Bad intervals are stored in AudioProcessingRecord as QC_not_passed
4. Processing logs are updated with QC results
5. Clean audio proceeds to frequency augmentation

The QC_not_passed field in logs contains a string representation of
list of (start_time, end_time) tuples indicating sections that did not pass QC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DropoutThresholds:
    """Default thresholds for signal dropout detection.

    These thresholds are calibrated for acoustic emission sensor data with DC offset
    (e.g., sensors that output ~1.5V at baseline, not 0V).

    Based on statistical analysis of participant recordings where:
    - Normal signal RMS: 1.491-1.608V (median 1.499V)
    - Normal signal variance: 0.000001-0.345 (median 0.00138)
    - True dropout: Bottom 1% of windows (RMS < 1.496, Variance < 0.00000013)

    Thresholds set conservatively to flag only clear sensor malfunction:
    - RMS drops significantly below normal baseline
    - Variance essentially zero (flat/stuck sensor)

    For other audio types (speech, music, etc.), create custom thresholds
    appropriate for your signal characteristics.
    """

    silence_threshold: float = 1.45  # Maximum RMS for silence detection
    flatline_threshold: float = 0.000001  # Maximum variance for flatline detection
    window_size_s: float = 0.5  # Sliding window size in seconds
    min_dropout_duration_s: float = 0.1  # Minimum continuous dropout to report


# Global default thresholds for dropout detection
# Override by passing custom values to detection functions
DEFAULT_DROPOUT_THRESHOLDS = DropoutThresholds()


def _convert_time_to_seconds(df: pd.DataFrame, time_col: str) -> np.ndarray:
    """Convert time column to seconds as numpy array.

    Handles both numeric seconds and timedelta inputs.

    Args:
        df: DataFrame containing time data
        time_col: Name of time column

    Returns:
        Array of time values in seconds
    """
    time_series = df[time_col]

    # Check if already numeric (seconds)
    if pd.api.types.is_numeric_dtype(time_series):
        return pd.to_numeric(time_series, errors="coerce").to_numpy()

    # Try converting as timedelta
    try:
        time_s = pd.to_numeric(
            pd.to_timedelta(time_series, unit="s").dt.total_seconds(),
            errors="coerce",
        ).to_numpy()
        return time_s
    except (ValueError, TypeError):
        # Fall back to direct numeric conversion
        return pd.to_numeric(time_series, errors="coerce").to_numpy()


def detect_signal_dropout(
    df: pd.DataFrame,
    time_col: str = "tt",
    audio_channels: list[str] | None = None,
    *,
    silence_threshold: float | None = None,
    flatline_threshold: float | None = None,
    window_size_s: float | None = None,
    min_dropout_duration_s: float | None = None,
) -> list[tuple[float, float]]:
    """Detect signal dropout (silence or flatline) in audio channels.

    For acoustic emission sensor data with DC offset (e.g., walking recordings),
    this detects when:
    1. Signal amplitude drops significantly below baseline (~1.5V for AE sensors)
    2. Signal variance becomes essentially zero (sensor stuck/flat)

    NOTE: Thresholds are calibrated for acoustic emission sensors with ~1.5V DC
    offset. For other audio types, pass custom thresholds appropriate for your data.

    Args:
        df: Audio DataFrame with time and channel data
        time_col: Name of time column
        audio_channels: List of channel names to check (default: ch1-ch4)
        silence_threshold: Maximum RMS amplitude for silence detection (default: from DEFAULT_DROPOUT_THRESHOLDS)
        flatline_threshold: Maximum variance for flatline detection (default: from DEFAULT_DROPOUT_THRESHOLDS)
        window_size_s: Size of sliding window for detection (seconds, default: from DEFAULT_DROPOUT_THRESHOLDS)
        min_dropout_duration_s: Minimum continuous dropout duration to report (default: from DEFAULT_DROPOUT_THRESHOLDS)

    Returns:
        List of (start_time, end_time) tuples indicating dropout periods
    """
    # Use centralized defaults if not provided
    if silence_threshold is None:
        silence_threshold = DEFAULT_DROPOUT_THRESHOLDS.silence_threshold
    if flatline_threshold is None:
        flatline_threshold = DEFAULT_DROPOUT_THRESHOLDS.flatline_threshold
    if window_size_s is None:
        window_size_s = DEFAULT_DROPOUT_THRESHOLDS.window_size_s
    if min_dropout_duration_s is None:
        min_dropout_duration_s = DEFAULT_DROPOUT_THRESHOLDS.min_dropout_duration_s
    if audio_channels is None:
        audio_channels = ["ch1", "ch2", "ch3", "ch4"]

    # Filter to available channels
    available_channels = [ch for ch in audio_channels if ch in df.columns]
    if not available_channels or time_col not in df.columns:
        return []

    # Convert time to seconds using helper
    time_s = _convert_time_to_seconds(df, time_col)

    # Get sampling rate
    valid_times = time_s[np.isfinite(time_s)]
    if len(valid_times) < 2:
        return []

    dt_median = float(np.median(np.diff(valid_times)))
    if dt_median <= 0:
        return []
    fs = 1.0 / dt_median
    window_samples = max(int(fs * window_size_s), 10)

    # Check each channel for dropout
    dropout_mask = np.zeros(len(df), dtype=bool)

    for ch in available_channels:
        ch_data = pd.to_numeric(df[ch], errors="coerce").to_numpy()

        # Sliding window RMS and variance
        rms_values = _sliding_window_rms(ch_data, window_samples)
        var_values = _sliding_window_variance(ch_data, window_samples)

        # Mark dropout where RMS is too low OR variance is too low
        ch_dropout = (rms_values < silence_threshold) | (var_values < flatline_threshold)
        dropout_mask |= ch_dropout

    # Convert dropout mask to time intervals
    dropout_intervals = _mask_to_intervals(dropout_mask, time_s, min_duration_s=min_dropout_duration_s)

    return dropout_intervals


def detect_artifactual_noise(
    df: pd.DataFrame,
    time_col: str = "tt",
    audio_channels: list[str] | None = None,
    *,
    spike_threshold_sigma: float = 5.0,
    spike_window_s: float = 0.01,
    min_artifact_duration_s: float = 0.01,
    detect_periodic_noise: bool = False,
    periodic_noise_threshold: float = 0.3,
) -> list[tuple[float, float]]:
    """Detect artifactual noise (spikes, outliers) in audio channels.

    Detects one-off or time-limited spikes via local statistical thresholds
    (e.g., someone talking, glass falling).

    NOTE: Periodic noise detection is disabled by default because acoustic
    emission signals naturally have sharp spectral peaks at specific frequencies
    (from impact and movement sounds) which can be mistakenly flagged as
    periodic background noise. Periodic detection causes too many false positives.

    Args:
        df: Audio DataFrame with time and channel data
        time_col: Name of time column
        audio_channels: List of channel names to check (default: ch1-ch4)
        spike_threshold_sigma: Number of standard deviations for spike detection
        spike_window_s: Window size for computing local statistics (seconds)
        min_artifact_duration_s: Minimum continuous artifact duration to report
        detect_periodic_noise: Whether to detect periodic background noise
        periodic_noise_threshold: Threshold for periodic noise detection (0-1),
                                  higher = less sensitive (default: 0.3)

    Returns:
        List of (start_time, end_time) tuples indicating artifact periods
    """
    if audio_channels is None:
        audio_channels = ["ch1", "ch2", "ch3", "ch4"]

    # Filter to available channels
    available_channels = [ch for ch in audio_channels if ch in df.columns]
    if not available_channels or time_col not in df.columns:
        return []

    # Convert time to seconds using helper
    time_s = _convert_time_to_seconds(df, time_col)

    # Get sampling rate
    valid_times = time_s[np.isfinite(time_s)]
    if len(valid_times) < 2:
        return []

    dt_median = float(np.median(np.diff(valid_times)))
    if dt_median <= 0:
        return []
    fs = 1.0 / dt_median
    window_samples = max(int(fs * spike_window_s), 5)

    # Check each channel for artifacts
    artifact_mask = np.zeros(len(df), dtype=bool)

    for ch in available_channels:
        ch_data = pd.to_numeric(df[ch], errors="coerce").to_numpy()

        # Detect one-off spikes via local statistics
        local_mean = _sliding_window_mean(ch_data, window_samples)
        local_std = _sliding_window_std(ch_data, window_samples)

        # Detect spikes: values exceeding threshold * std from local mean
        threshold = local_mean + spike_threshold_sigma * local_std
        ch_artifacts = np.abs(ch_data) > threshold
        artifact_mask |= ch_artifacts

        # Detect periodic noise if enabled
        if detect_periodic_noise:
            periodic_mask = _detect_periodic_noise(ch_data, fs, periodic_noise_threshold)
            artifact_mask |= periodic_mask

    # Convert artifact mask to time intervals
    artifact_intervals = _mask_to_intervals(artifact_mask, time_s, min_duration_s=min_artifact_duration_s)

    return artifact_intervals


def run_raw_audio_qc(
    df: pd.DataFrame,
    time_col: str = "tt",
    audio_channels: list[str] | None = None,
    *,
    silence_threshold: float | None = None,
    flatline_threshold: float | None = None,
    spike_threshold_sigma: float = 5.0,
    dropout_window_s: float | None = None,
    spike_window_s: float = 0.01,
    min_dropout_duration_s: float | None = None,
    min_artifact_duration_s: float = 0.01,
    detect_periodic_noise: bool = False,
    periodic_noise_threshold: float = 0.3,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Run comprehensive raw audio QC checks.

    Uses DEFAULT_DROPOUT_THRESHOLDS for any threshold parameters not explicitly provided.

    Detects signal dropout and artifactual noise in raw audio recordings.

    Artifactual noise detection includes:
    - One-off or time-limited spikes (intermittent background noise)

    NOTE: Periodic noise detection is disabled by default because acoustic
    emission signals naturally have sharp spectral peaks at specific frequencies
    that cause false positives.

    NOTE: Dropout thresholds are set conservatively to avoid false positives
    on normal quiet periods in walking recordings. Only flags truly silent/missing
    signal, not natural signal variations.

    Args:
        df: Audio DataFrame with time and channel data
        time_col: Name of time column
        audio_channels: List of channel names to check (default: ch1-ch4)
        silence_threshold: Maximum RMS amplitude for silence detection
        flatline_threshold: Maximum variance for flatline detection
        spike_threshold_sigma: Number of standard deviations for spike detection
        dropout_window_s: Window size for dropout detection (seconds)
        spike_window_s: Window size for spike detection (seconds)
        min_dropout_duration_s: Minimum dropout duration to report
        min_artifact_duration_s: Minimum artifact duration to report
        detect_periodic_noise: Whether to detect periodic background noise
        periodic_noise_threshold: Threshold for periodic noise detection (0-1),
                                  higher = less sensitive (default: 0.3)

    Returns:
        Tuple of (dropout_intervals, artifact_intervals) where each is a list
        of (start_time, end_time) tuples
    """
    dropout_intervals = detect_signal_dropout(
        df,
        time_col=time_col,
        audio_channels=audio_channels,
        silence_threshold=silence_threshold,
        flatline_threshold=flatline_threshold,
        window_size_s=dropout_window_s,
        min_dropout_duration_s=min_dropout_duration_s,
    )

    artifact_intervals = detect_artifactual_noise(
        df,
        time_col=time_col,
        audio_channels=audio_channels,
        spike_threshold_sigma=spike_threshold_sigma,
        spike_window_s=spike_window_s,
        min_artifact_duration_s=min_artifact_duration_s,
        detect_periodic_noise=detect_periodic_noise,
        periodic_noise_threshold=periodic_noise_threshold,
    )

    return dropout_intervals, artifact_intervals


def detect_signal_dropout_per_mic(
    df: pd.DataFrame,
    time_col: str = "tt",
    audio_channels: list[str] | None = None,
    *,
    silence_threshold: float | None = None,
    flatline_threshold: float | None = None,
    window_size_s: float | None = None,
    min_dropout_duration_s: float | None = None,
) -> dict[str, list[tuple[float, float]]]:
    """Detect signal dropout per microphone channel.

    Returns bad intervals for each channel separately, allowing per-mic QC assessment.

    For acoustic emission sensor data with DC offset (e.g., walking recordings where
    sensor output is biased around 1.5V), dropout is detected when:
    1. RMS amplitude drops significantly below the signal baseline (~1.5 for AE sensors)
    2. Variance becomes essentially zero (signal is completely flat/stuck)

    Default thresholds are calibrated for acoustic emission walking data where:
    - Normal signal RMS: 1.491-1.608 (median 1.499)
    - Normal signal variance: 0.000001-0.345 (median 0.00138)
    - True dropout: RMS < 1.45 AND variance < 0.000001 (bottom 1-5% of data)

    Args:
        df: Audio DataFrame with time and channel data
        time_col: Name of time column
        audio_channels: List of channel names to check (default: ch1-ch4)
        silence_threshold: Maximum RMS amplitude for silence detection (default: from DEFAULT_DROPOUT_THRESHOLDS)
        flatline_threshold: Maximum variance for flatline detection (default: from DEFAULT_DROPOUT_THRESHOLDS)
        window_size_s: Size of sliding window for detection (seconds, default: from DEFAULT_DROPOUT_THRESHOLDS)
        min_dropout_duration_s: Minimum continuous dropout duration to report (default: from DEFAULT_DROPOUT_THRESHOLDS)

    Returns:
        Dictionary mapping channel name to list of (start_time, end_time) tuples
    """
    # Use centralized defaults if not provided
    if silence_threshold is None:
        silence_threshold = DEFAULT_DROPOUT_THRESHOLDS.silence_threshold
    if flatline_threshold is None:
        flatline_threshold = DEFAULT_DROPOUT_THRESHOLDS.flatline_threshold
    if window_size_s is None:
        window_size_s = DEFAULT_DROPOUT_THRESHOLDS.window_size_s
    if min_dropout_duration_s is None:
        min_dropout_duration_s = DEFAULT_DROPOUT_THRESHOLDS.min_dropout_duration_s

    if audio_channels is None:
        audio_channels = ["ch1", "ch2", "ch3", "ch4"]

    # Filter to available channels
    available_channels = [ch for ch in audio_channels if ch in df.columns]
    if not available_channels or time_col not in df.columns:
        return {}

    # Convert time to seconds using helper
    time_s = _convert_time_to_seconds(df, time_col)

    # Get sampling rate
    valid_times = time_s[np.isfinite(time_s)]
    if len(valid_times) < 2:
        return {}

    dt_median = float(np.median(np.diff(valid_times)))
    if dt_median <= 0:
        return {}
    fs = 1.0 / dt_median
    window_samples = max(int(fs * window_size_s), 10)

    # Check each channel for dropout separately
    per_mic_intervals = {}

    for ch in available_channels:
        ch_data = pd.to_numeric(df[ch], errors="coerce").to_numpy()

        # Sliding window RMS and variance
        rms_values = _sliding_window_rms(ch_data, window_samples)
        var_values = _sliding_window_variance(ch_data, window_samples)

        # Mark dropout where RMS is too low OR variance is too low
        ch_dropout = (rms_values < silence_threshold) | (var_values < flatline_threshold)

        # Convert dropout mask to time intervals for this channel
        intervals = _mask_to_intervals(ch_dropout, time_s, min_duration_s=min_dropout_duration_s)

        per_mic_intervals[ch] = intervals

    return per_mic_intervals


@dataclass(frozen=True)
class ContinuousArtifactThresholds:
    """Thresholds for continuous background noise detection.

    Continuous artifacts are persistent narrowband noise sources (fans, MRI
    machines, HVAC) that last at least ``min_duration_s`` seconds. The 4-second
    default spans ≥2 full maneuver cycles at the 0.25 Hz STS/FE cadence,
    ensuring that exercise-related broadband sounds are not mis-classified.

    Detection uses sliding-window Welch PSD to identify frequencies where the
    peak-to-median power ratio exceeds ``peak_snr_threshold`` consistently.
    """

    min_duration_s: float = 4.0  # Minimum duration to classify as continuous
    window_size_s: float = 2.0  # PSD sliding-window length
    peak_snr_threshold: float = 10.0  # Peak/median power ratio for narrowband noise


# Global default thresholds for continuous artifact detection
DEFAULT_CONTINUOUS_THRESHOLDS = ContinuousArtifactThresholds()


def _classify_artifact_type(
    intervals: list[tuple[float, float]],
    *,
    intermittent_threshold_s: float = 4.0,
) -> list[str]:
    """Classify artifacts as 'Intermittent' or 'Continuous' based on duration.

    Args:
        intervals: List of (start_time, end_time) tuples
        intermittent_threshold_s: Max duration (seconds) to classify as intermittent.
            Default is 4.0s (≥2 full maneuver cycles at 0.25 Hz cadence).

    Returns:
        List of artifact type strings ('Intermittent' or 'Continuous') per interval
    """
    types = []
    for start, end in intervals:
        duration = end - start
        artifact_type = "Intermittent" if duration < intermittent_threshold_s else "Continuous"
        types.append(artifact_type)
    return types


def detect_artifactual_noise_per_mic(
    df: pd.DataFrame,
    time_col: str = "tt",
    audio_channels: list[str] | None = None,
    *,
    spike_threshold_sigma: float = 5.0,
    spike_window_s: float = 0.01,
    min_artifact_duration_s: float = 0.01,
    detect_periodic_noise: bool = False,
    periodic_noise_threshold: float = 0.3,
) -> tuple[dict[str, list[tuple[float, float]]], dict[str, list[str]]]:
    """Detect artifactual noise per microphone channel.

    Returns bad intervals and artifact types for each channel separately, allowing per-mic QC assessment.
    Detects both one-off spikes and periodic background noise.

    Args:
        df: Audio DataFrame with time and channel data
        time_col: Name of time column
        audio_channels: List of channel names to check (default: ch1-ch4)
        spike_threshold_sigma: Number of standard deviations for spike detection
        spike_window_s: Window size for computing local statistics (seconds)
        min_artifact_duration_s: Minimum continuous artifact duration to report
        detect_periodic_noise: Whether to detect periodic background noise
        periodic_noise_threshold: Threshold for periodic noise detection (0-1),
                                  higher = less sensitive (default: 0.3)

    Returns:
        Tuple of (intervals_dict, types_dict) where:
        - intervals_dict: Channel name -> list of (start_time, end_time) tuples
        - types_dict: Channel name -> list of artifact types ('Intermittent' or 'Continuous')
    """
    if audio_channels is None:
        audio_channels = ["ch1", "ch2", "ch3", "ch4"]

    # Filter to available channels
    available_channels = [ch for ch in audio_channels if ch in df.columns]
    if not available_channels or time_col not in df.columns:
        return {}, {}

    # Convert time to seconds using helper
    time_s = _convert_time_to_seconds(df, time_col)

    # Get sampling rate
    valid_times = time_s[np.isfinite(time_s)]
    if len(valid_times) < 2:
        return {}, {}

    dt_median = float(np.median(np.diff(valid_times)))
    if dt_median <= 0:
        return {}, {}
    fs = 1.0 / dt_median
    window_samples = max(int(fs * spike_window_s), 5)

    # Check each channel for artifacts separately
    per_mic_intervals = {}

    for ch in available_channels:
        ch_data = pd.to_numeric(df[ch], errors="coerce").to_numpy()
        artifact_mask = np.zeros(len(df), dtype=bool)

        # Detect one-off spikes via local statistics
        local_mean = _sliding_window_mean(ch_data, window_samples)
        local_std = _sliding_window_std(ch_data, window_samples)

        # Detect spikes: values exceeding threshold * std from local mean
        threshold = local_mean + spike_threshold_sigma * local_std
        ch_artifacts = np.abs(ch_data) > threshold
        artifact_mask |= ch_artifacts

        # Detect periodic noise if enabled
        if detect_periodic_noise:
            periodic_mask = _detect_periodic_noise(ch_data, fs, periodic_noise_threshold)
            artifact_mask |= periodic_mask

        # Convert artifact mask to time intervals for this channel
        intervals = _mask_to_intervals(artifact_mask, time_s, min_duration_s=min_artifact_duration_s)

        per_mic_intervals[ch] = intervals

    # Classify artifact types for each channel
    per_mic_types = {ch: _classify_artifact_type(intervals) for ch, intervals in per_mic_intervals.items()}

    return per_mic_intervals, per_mic_types


def run_raw_audio_qc_per_mic(
    df: pd.DataFrame,
    time_col: str = "tt",
    audio_channels: list[str] | None = None,
    *,
    silence_threshold: float | None = None,
    flatline_threshold: float | None = None,
    spike_threshold_sigma: float = 5.0,
    dropout_window_s: float | None = None,
    spike_window_s: float = 0.01,
    min_dropout_duration_s: float | None = None,
    min_artifact_duration_s: float = 0.01,
    detect_periodic_noise: bool = False,
    periodic_noise_threshold: float = 0.3,
    detect_continuous_noise: bool = True,
    continuous_min_duration_s: float | None = None,
    continuous_window_s: float | None = None,
    continuous_snr_threshold: float | None = None,
) -> dict[str, list[tuple[float, float]]]:
    """Run comprehensive raw audio QC checks per microphone.

    Detects signal dropout, spike artifacts, and continuous narrowband
    background noise in raw audio recordings, returning results separately
    for each microphone channel.

    Args:
        df: Audio DataFrame with time and channel data
        time_col: Name of time column
        audio_channels: List of channel names to check (default: ch1-ch4)
        silence_threshold: Maximum RMS amplitude for silence detection
        flatline_threshold: Maximum variance for flatline detection
        spike_threshold_sigma: Number of standard deviations for spike detection
        dropout_window_s: Window size for dropout detection (seconds)
        spike_window_s: Window size for spike detection (seconds)
        min_dropout_duration_s: Minimum dropout duration to report
        min_artifact_duration_s: Minimum artifact duration to report
        detect_periodic_noise: Whether to detect periodic background noise
        periodic_noise_threshold: Threshold for periodic noise detection (0-1)
        detect_continuous_noise: Whether to detect continuous narrowband noise
        continuous_min_duration_s: Minimum duration for continuous noise
        continuous_window_s: PSD window size for continuous detection
        continuous_snr_threshold: Peak/median ratio for narrowband detection

    Returns:
        Dictionary mapping channel name to list of (start_time, end_time) tuples
        representing bad intervals for that specific channel
    """
    dropout_per_mic = detect_signal_dropout_per_mic(
        df,
        time_col=time_col,
        audio_channels=audio_channels,
        silence_threshold=silence_threshold,
        flatline_threshold=flatline_threshold,
        window_size_s=dropout_window_s,
        min_dropout_duration_s=min_dropout_duration_s,
    )

    artifact_per_mic, artifact_types_per_mic = detect_artifactual_noise_per_mic(
        df,
        time_col=time_col,
        audio_channels=audio_channels,
        spike_threshold_sigma=spike_threshold_sigma,
        spike_window_s=spike_window_s,
        min_artifact_duration_s=min_artifact_duration_s,
        detect_periodic_noise=detect_periodic_noise,
        periodic_noise_threshold=periodic_noise_threshold,
    )

    # Continuous narrowband noise detection (spectral)
    continuous_per_mic: dict[str, list[tuple[float, float]]] = {}
    if detect_continuous_noise:
        continuous_per_mic = detect_continuous_background_noise_per_mic(
            df,
            time_col=time_col,
            audio_channels=audio_channels,
            min_duration_s=continuous_min_duration_s,
            window_size_s=continuous_window_s,
            peak_snr_threshold=continuous_snr_threshold,
        )

    # Merge dropout, artifact, and continuous intervals per channel
    per_mic_bad_intervals = {}
    all_channels = set(list(dropout_per_mic.keys()) + list(artifact_per_mic.keys()) + list(continuous_per_mic.keys()))

    for ch in all_channels:
        dropout_intervals = dropout_per_mic.get(ch, [])
        artifact_intervals = artifact_per_mic.get(ch, [])
        continuous_intervals = continuous_per_mic.get(ch, [])
        merged = merge_artifact_intervals(dropout_intervals, artifact_intervals, continuous_intervals)
        per_mic_bad_intervals[ch] = merged

    return per_mic_bad_intervals


def merge_bad_intervals(
    dropout_intervals: list[tuple[float, float]],
    artifact_intervals: list[tuple[float, float]],
    merge_gap_s: float = 0.5,
) -> list[tuple[float, float]]:
    """Merge overlapping or nearby bad intervals.

    Combines dropout and artifact intervals, merging those that are close
    together to avoid fragmentation.

    Args:
        dropout_intervals: List of dropout (start, end) tuples
        artifact_intervals: List of artifact (start, end) tuples
        merge_gap_s: Merge intervals separated by less than this gap (seconds)

    Returns:
        List of merged (start_time, end_time) tuples
    """
    # Combine all intervals
    all_intervals = dropout_intervals + artifact_intervals
    if not all_intervals:
        return []

    # Sort by start time
    sorted_intervals = sorted(all_intervals, key=lambda x: x[0])

    # Merge overlapping or nearby intervals
    merged = [sorted_intervals[0]]
    for start, end in sorted_intervals[1:]:
        prev_start, prev_end = merged[-1]

        # Check if current interval overlaps or is close to previous
        if start <= prev_end + merge_gap_s:
            # Merge by extending the end time
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            # Add as new interval
            merged.append((start, end))

    return merged


def clip_bad_segments(
    df: pd.DataFrame,
    bad_intervals: list[tuple[float, float]],
    time_col: str = "tt",
) -> pd.DataFrame:
    """Remove bad time segments from audio DataFrame.

    Clips out sections identified as having dropout or artifacts, returning
    a DataFrame with only good quality data. Timestamps are preserved.

    Args:
        df: Audio DataFrame with time and channel data
        bad_intervals: List of (start_time, end_time) tuples to remove
        time_col: Name of time column

    Returns:
        DataFrame with bad segments removed
    """
    if not bad_intervals:
        return df.copy()

    # Convert time to seconds for comparison
    time_s = pd.to_numeric(
        pd.to_timedelta(df[time_col], unit="s").dt.total_seconds(),
        errors="coerce",
    ).to_numpy()

    # Create mask for good data (inverse of bad intervals)
    keep_mask = np.ones(len(df), dtype=bool)
    for start, end in bad_intervals:
        bad_mask = (time_s >= start) & (time_s <= end)
        keep_mask &= ~bad_mask

    # Return filtered DataFrame
    return df[keep_mask].reset_index(drop=True)


def adjust_bad_intervals_for_sync(
    bad_intervals: list[tuple[float, float]],
    audio_stomp_time: float,
    bio_stomp_time: float,
) -> list[tuple[float, float]]:
    """Adjust bad interval timestamps for synchronization offset.

    When audio and biomechanics are synchronized via stomp times, the audio
    timestamps are shifted. This function adjusts the bad intervals from
    raw audio coordinates to synchronized (biomechanics) coordinates.

    Args:
        bad_intervals: List of (start, end) tuples in audio time coordinates
        audio_stomp_time: Stomp time in audio recording (seconds)
        bio_stomp_time: Stomp time in biomechanics recording (seconds)

    Returns:
        List of (start, end) tuples in synchronized time coordinates
    """
    if not bad_intervals:
        return []

    # Calculate the offset: bio_stomp - audio_stomp
    # This is the shift applied to audio timestamps during synchronization
    offset = bio_stomp_time - audio_stomp_time

    # Adjust all intervals by the offset
    adjusted_intervals = [(start + offset, end + offset) for start, end in bad_intervals]

    return adjusted_intervals


def check_cycle_in_bad_interval(
    cycle_start_time: float,
    cycle_end_time: float,
    bad_intervals: list[tuple[float, float]],
    overlap_threshold: float = 0.1,
) -> bool:
    """Check if a movement cycle overlaps with bad audio intervals.

    Returns True if the cycle has significant overlap with any bad interval,
    indicating that audio QC should be marked as failed for this cycle.

    Args:
        cycle_start_time: Start time of movement cycle (seconds)
        cycle_end_time: End time of movement cycle (seconds)
        bad_intervals: List of (start, end) bad interval tuples (seconds)
        overlap_threshold: Fraction of cycle duration that must overlap
                          to mark as failed (default: 0.1 = 10%)

    Returns:
        True if cycle fails audio QC due to bad intervals, False otherwise
    """
    if not bad_intervals:
        return False

    cycle_duration = cycle_end_time - cycle_start_time
    if cycle_duration <= 0:
        return False

    # Calculate total overlap with all bad intervals
    total_overlap = 0.0

    for bad_start, bad_end in bad_intervals:
        # Calculate overlap between cycle and this bad interval
        overlap_start = max(cycle_start_time, bad_start)
        overlap_end = min(cycle_end_time, bad_end)

        if overlap_end > overlap_start:
            total_overlap += overlap_end - overlap_start

    # Check if overlap exceeds threshold
    overlap_fraction = total_overlap / cycle_duration
    return overlap_fraction >= overlap_threshold


# Helper functions for sliding window operations


def _sliding_window_rms(data: np.ndarray, window_size: int) -> np.ndarray:
    """Compute RMS using vectorized cumulative sum approach (O(n) complexity).

    Fully vectorized using numpy cumulative sums and advanced indexing,
    avoiding all Python loops for maximum performance on large arrays.
    """
    if len(data) == 0:
        return np.array([])

    if len(data) == 1:
        return np.array([np.sqrt(data[0] ** 2)])

    # Replace NaN with 0 for RMS calculation
    data_clean = np.nan_to_num(data, nan=0.0).astype(np.float64)

    # RMS = sqrt(mean(x^2))
    squared = data_clean**2

    # Cumulative sum for O(n) window sum computation
    cumsum: np.ndarray = np.concatenate(([0.0], np.cumsum(squared)))  # type: ignore[arg-type]

    half_window = window_size // 2

    # Vectorized index computation
    indices = np.arange(len(data))
    start_indices = np.maximum(0, indices - half_window)
    end_indices = np.minimum(len(data), indices + half_window + 1)

    # Vectorized window sum computation using advanced indexing
    window_sums = cumsum[end_indices] - cumsum[start_indices]
    window_counts = end_indices - start_indices

    # Vectorized RMS computation
    result = np.sqrt(window_sums / window_counts)

    return result


def _sliding_window_variance(data: np.ndarray, window_size: int) -> np.ndarray:
    """Compute variance using vectorized cumulative sum approach (O(n) complexity).

    Fully vectorized using numpy cumulative sums and advanced indexing,
    avoiding all Python loops for maximum performance on large arrays.
    """
    if len(data) == 0:
        return np.array([])

    if len(data) == 1:
        return np.array([0.0])

    # Replace NaN with 0 for variance calculation
    data_clean = np.nan_to_num(data, nan=0.0).astype(np.float64)

    # Variance = E[x^2] - (E[x])^2
    # Cumulative sums for O(n) window sum computation
    cumsum_x: np.ndarray = np.concatenate(([0.0], np.cumsum(data_clean)))  # type: ignore[arg-type]
    cumsum_x2: np.ndarray = np.concatenate(([0.0], np.cumsum(data_clean**2)))  # type: ignore[arg-type]

    half_window = window_size // 2

    # Vectorized index computation
    indices = np.arange(len(data))
    start_indices = np.maximum(0, indices - half_window)
    end_indices = np.minimum(len(data), indices + half_window + 1)

    # Vectorized window sum computation
    sum_x = cumsum_x[end_indices] - cumsum_x[start_indices]
    sum_x2 = cumsum_x2[end_indices] - cumsum_x2[start_indices]
    window_counts = end_indices - start_indices

    # Vectorized variance computation
    mean_x = sum_x / window_counts
    mean_x2 = sum_x2 / window_counts

    result = mean_x2 - mean_x**2

    # Clamp to zero (numerical precision can cause tiny negative values)
    result = np.maximum(result, 0.0)

    return result


def _sliding_window_mean(data: np.ndarray, window_size: int) -> np.ndarray:
    """Compute mean using vectorized cumulative sum approach (O(n) complexity).

    Fully vectorized using numpy cumulative sums and advanced indexing,
    avoiding all Python loops for maximum performance on large arrays.
    """
    if len(data) == 0:
        return np.array([])

    if len(data) == 1:
        return np.array([data[0]])

    # Replace NaN with 0 for mean calculation
    data_clean = np.nan_to_num(data, nan=0.0).astype(np.float64)

    # Cumulative sum for O(n) window sum computation
    cumsum_x: np.ndarray = np.concatenate(([0.0], np.cumsum(data_clean)))  # type: ignore[arg-type]

    half_window = window_size // 2

    # Vectorized index computation
    indices = np.arange(len(data))
    start_indices = np.maximum(0, indices - half_window)
    end_indices = np.minimum(len(data), indices + half_window + 1)

    # Vectorized window sum computation
    sum_x = cumsum_x[end_indices] - cumsum_x[start_indices]
    window_counts = end_indices - start_indices

    # Vectorized mean computation
    result = sum_x / window_counts

    return result


def _sliding_window_std(data: np.ndarray, window_size: int) -> np.ndarray:
    """Compute standard deviation using vectorized cumulative sum approach (O(n) complexity).

    Fully vectorized using numpy cumulative sums and advanced indexing,
    avoiding all Python loops for maximum performance on large arrays.
    """
    if len(data) == 0:
        return np.array([])

    if len(data) == 1:
        return np.array([0.0])

    # Replace NaN with 0 for std calculation
    data_clean = np.nan_to_num(data, nan=0.0).astype(np.float64)

    # StdDev = sqrt(variance)
    # Cumulative sums for O(n) window sum computation
    cumsum_x: np.ndarray = np.concatenate(([0.0], np.cumsum(data_clean)))  # type: ignore[arg-type]
    cumsum_x2: np.ndarray = np.concatenate(([0.0], np.cumsum(data_clean**2)))  # type: ignore[arg-type]

    half_window = window_size // 2

    # Vectorized index computation
    indices = np.arange(len(data))
    start_indices = np.maximum(0, indices - half_window)
    end_indices = np.minimum(len(data), indices + half_window + 1)

    # Vectorized window sum computation
    sum_x = cumsum_x[end_indices] - cumsum_x[start_indices]
    sum_x2 = cumsum_x2[end_indices] - cumsum_x2[start_indices]
    window_counts = end_indices - start_indices

    # Vectorized variance computation: var = E[x^2] - (E[x])^2
    mean_x = sum_x / window_counts
    mean_x2 = sum_x2 / window_counts

    variance = mean_x2 - mean_x**2

    # Clamp to zero (numerical precision can cause tiny negative values)
    variance = np.maximum(variance, 0.0)

    # Standard deviation = sqrt(variance)
    result = np.sqrt(variance)

    return result

    return result


def _mask_to_intervals(
    mask: np.ndarray,
    time_s: np.ndarray,
    min_duration_s: float = 0.0,
) -> list[tuple[float, float]]:
    """Convert boolean mask to list of time intervals.

    Args:
        mask: Boolean array where True indicates bad samples
        time_s: Time values in seconds
        min_duration_s: Minimum interval duration to include

    Returns:
        List of (start_time, end_time) tuples
    """
    if not np.any(mask):
        return []

    # Find transitions in mask
    diff = np.diff(np.concatenate([[False], mask, [False]]).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    intervals = []
    for start_idx, end_idx in zip(starts, ends):
        # Get time bounds (use nearest valid times)
        valid_mask = np.isfinite(time_s)
        if not np.any(valid_mask):
            continue

        # Find closest valid times
        if start_idx >= len(time_s):
            continue
        if end_idx > len(time_s):
            end_idx = len(time_s)

        start_time = time_s[start_idx] if np.isfinite(time_s[start_idx]) else np.nan
        end_time = time_s[end_idx - 1] if end_idx > 0 and np.isfinite(time_s[end_idx - 1]) else np.nan

        if np.isfinite(start_time) and np.isfinite(end_time):
            duration = end_time - start_time
            if duration >= min_duration_s:
                intervals.append((float(start_time), float(end_time)))

    return intervals


# NOTE: The _detect_periodic_noise function is now active and called during
# bin processing. It was previously deprecated but has been re-integrated into
# the raw audio QC pipeline to detect periodic background noise at specific
# frequencies using spectral analysis (Welch's method).


def _detect_periodic_noise(
    data: np.ndarray,
    fs: float,
    threshold: float = 0.3,
) -> np.ndarray:
    """Detect periodic background noise using spectral analysis.

    Identifies consistent periodic noise (e.g., fan running) by analyzing
    the power spectral density. Periods with elevated power at specific
    frequencies are marked as artifacts.

    Args:
        data: Audio signal data
        fs: Sampling frequency (Hz)
        threshold: Threshold for periodic noise detection (0-1)
                  Higher values = less sensitive

    Returns:
        Boolean mask where True indicates periodic noise
    """
    from scipy.signal import welch

    if len(data) < 256:
        return np.zeros(len(data), dtype=bool)

    # Compute power spectral density
    try:
        nperseg = min(len(data), int(fs * 2))  # 2 second windows
        freqs, psd = welch(data, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    except Exception:
        return np.zeros(len(data), dtype=bool)

    if len(psd) == 0:
        return np.zeros(len(data), dtype=bool)

    # Identify prominent spectral peaks (potential periodic noise)
    # Ignore DC and very low frequencies (< 5 Hz)
    freq_mask = freqs > 5.0
    if not np.any(freq_mask):
        return np.zeros(len(data), dtype=bool)

    psd_nondc = psd[freq_mask]
    if len(psd_nondc) == 0:
        return np.zeros(len(data), dtype=bool)

    # Calculate relative power: peak power / median power
    median_power = np.median(psd_nondc)
    if median_power <= 0:
        return np.zeros(len(data), dtype=bool)

    max_power = np.max(psd_nondc)
    relative_power = max_power / median_power

    # If relative power is high, we have a strong periodic component
    # This indicates consistent background noise at a specific frequency
    has_periodic_noise = relative_power > (1.0 / threshold)

    if has_periodic_noise:
        # Mark entire signal as artifact (periodic noise is typically present throughout)
        return np.ones(len(data), dtype=bool)
    else:
        return np.zeros(len(data), dtype=bool)


# ---------------------------------------------------------------------------
# Phase C: Continuous background noise detection via sliding-window PSD
# ---------------------------------------------------------------------------


def detect_continuous_background_noise(
    df: pd.DataFrame,
    time_col: str = "tt",
    audio_channels: list[str] | None = None,
    *,
    min_duration_s: float | None = None,
    window_size_s: float | None = None,
    peak_snr_threshold: float | None = None,
) -> list[tuple[float, float]]:
    """Detect continuous narrowband background noise across all channels.

    Uses a sliding-window Welch PSD approach to find time intervals where
    a strong narrowband peak persists for at least ``min_duration_s``.  This
    catches quiet but sustained noise sources (fans, HVAC, MRI hum) that
    the spike-based detector in ``detect_artifactual_noise`` may miss.

    Args:
        df: Audio DataFrame with time and channel data.
        time_col: Name of time column.
        audio_channels: Channel names to check (default: ch1-ch4).
        min_duration_s: Minimum duration to flag (default from
            ``DEFAULT_CONTINUOUS_THRESHOLDS``).
        window_size_s: Sliding PSD window length in seconds.
        peak_snr_threshold: Peak-to-median power ratio for detection.

    Returns:
        Merged list of (start_time, end_time) tuples across all channels.
    """
    if min_duration_s is None:
        min_duration_s = DEFAULT_CONTINUOUS_THRESHOLDS.min_duration_s
    if window_size_s is None:
        window_size_s = DEFAULT_CONTINUOUS_THRESHOLDS.window_size_s
    if peak_snr_threshold is None:
        peak_snr_threshold = DEFAULT_CONTINUOUS_THRESHOLDS.peak_snr_threshold
    if audio_channels is None:
        audio_channels = ["ch1", "ch2", "ch3", "ch4"]

    available_channels = [ch for ch in audio_channels if ch in df.columns]
    if not available_channels or time_col not in df.columns:
        return []

    time_s = _convert_time_to_seconds(df, time_col)
    valid_times = time_s[np.isfinite(time_s)]
    if len(valid_times) < 2:
        return []

    dt_median = float(np.median(np.diff(valid_times)))
    if dt_median <= 0:
        return []
    fs = 1.0 / dt_median

    combined_mask = np.zeros(len(df), dtype=bool)
    for ch in available_channels:
        ch_data = pd.to_numeric(df[ch], errors="coerce").to_numpy()
        ch_mask = _sliding_window_psd_narrowband(ch_data, fs, window_size_s, peak_snr_threshold)
        combined_mask |= ch_mask

    intervals = _mask_to_intervals(combined_mask, time_s, min_duration_s=min_duration_s)
    return intervals


def detect_continuous_background_noise_per_mic(
    df: pd.DataFrame,
    time_col: str = "tt",
    audio_channels: list[str] | None = None,
    *,
    min_duration_s: float | None = None,
    window_size_s: float | None = None,
    peak_snr_threshold: float | None = None,
) -> dict[str, list[tuple[float, float]]]:
    """Detect continuous narrowband background noise per microphone channel.

    Same algorithm as :func:`detect_continuous_background_noise` but returns
    results per channel so that per-mic QC columns can be populated.

    Returns:
        Dictionary mapping channel name to list of (start, end) intervals.
    """
    if min_duration_s is None:
        min_duration_s = DEFAULT_CONTINUOUS_THRESHOLDS.min_duration_s
    if window_size_s is None:
        window_size_s = DEFAULT_CONTINUOUS_THRESHOLDS.window_size_s
    if peak_snr_threshold is None:
        peak_snr_threshold = DEFAULT_CONTINUOUS_THRESHOLDS.peak_snr_threshold
    if audio_channels is None:
        audio_channels = ["ch1", "ch2", "ch3", "ch4"]

    available_channels = [ch for ch in audio_channels if ch in df.columns]
    if not available_channels or time_col not in df.columns:
        return {}

    time_s = _convert_time_to_seconds(df, time_col)
    valid_times = time_s[np.isfinite(time_s)]
    if len(valid_times) < 2:
        return {}

    dt_median = float(np.median(np.diff(valid_times)))
    if dt_median <= 0:
        return {}
    fs = 1.0 / dt_median

    per_mic: dict[str, list[tuple[float, float]]] = {}
    for ch in available_channels:
        ch_data = pd.to_numeric(df[ch], errors="coerce").to_numpy()
        ch_mask = _sliding_window_psd_narrowband(ch_data, fs, window_size_s, peak_snr_threshold)
        per_mic[ch] = _mask_to_intervals(ch_mask, time_s, min_duration_s=min_duration_s)

    return per_mic


def _sliding_window_psd_narrowband(
    data: np.ndarray,
    fs: float,
    window_size_s: float,
    peak_snr_threshold: float,
) -> np.ndarray:
    """Return a boolean mask marking samples with narrowband spectral peaks.

    Slides a window across *data*, computes a Welch PSD inside each window,
    and marks the window as *narrowband noise* when the peak-to-median
    spectral power ratio exceeds ``peak_snr_threshold`` (ignoring DC and
    frequencies < 5 Hz).
    """
    from scipy.signal import welch as scipy_welch

    n = len(data)
    if n < 256:
        return np.zeros(n, dtype=bool)

    window_samples = max(int(fs * window_size_s), 256)
    step = window_samples // 2  # 50 % overlap between consecutive windows

    mask = np.zeros(n, dtype=bool)

    for start_idx in range(0, n - window_samples + 1, step):
        end_idx = start_idx + window_samples
        segment = data[start_idx:end_idx]

        try:
            nperseg = min(window_samples, int(fs * 0.5))  # 0.5 s PSD sub-windows
            nperseg = max(nperseg, 64)
            freqs, psd = scipy_welch(segment, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
        except Exception:
            continue

        if len(psd) == 0:
            continue

        # Ignore DC and very low frequencies
        freq_mask = freqs > 5.0
        if not np.any(freq_mask):
            continue

        psd_nondc = psd[freq_mask]
        median_power = np.median(psd_nondc)
        if median_power <= 0:
            continue

        peak_power = np.max(psd_nondc)
        if peak_power / median_power >= peak_snr_threshold:
            mask[start_idx:end_idx] = True

    return mask


# ---------------------------------------------------------------------------
# Phase D: Interval trimming and merging utilities for cycle-level QC
# ---------------------------------------------------------------------------


def trim_intervals_to_cycle(
    intervals: list[tuple[float, float]],
    cycle_start: float,
    cycle_end: float,
) -> list[tuple[float, float]]:
    """Clip artifact intervals to a movement cycle's time bounds.

    Each interval that overlaps ``[cycle_start, cycle_end]`` is trimmed to
    fit within the cycle bounds.  Non-overlapping intervals are discarded.

    Args:
        intervals: List of (start, end) artifact time intervals.
        cycle_start: Cycle start time in seconds.
        cycle_end: Cycle end time in seconds.

    Returns:
        List of trimmed (start, end) tuples within the cycle.
    """
    trimmed: list[tuple[float, float]] = []
    for start, end in intervals:
        # Skip intervals with no overlap
        if end <= cycle_start or start >= cycle_end:
            continue
        trimmed.append((max(start, cycle_start), min(end, cycle_end)))
    return trimmed


def merge_artifact_intervals(
    *interval_lists: list[tuple[float, float]],
    merge_gap_s: float = 0.0,
) -> list[tuple[float, float]]:
    """Merge multiple lists of artifact intervals into one sorted, non-overlapping list.

    This generalises :func:`merge_bad_intervals` to accept an arbitrary number
    of interval lists (intermittent, continuous, periodic, …) and merge them.

    Args:
        *interval_lists: One or more lists of (start, end) tuples.
        merge_gap_s: Intervals separated by less than this gap are merged.

    Returns:
        Sorted, merged list of (start, end) tuples.
    """
    all_intervals: list[tuple[float, float]] = []
    for ilist in interval_lists:
        all_intervals.extend(ilist)
    if not all_intervals:
        return []

    sorted_intervals = sorted(all_intervals, key=lambda x: x[0])

    merged = [sorted_intervals[0]]
    for start, end in sorted_intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + merge_gap_s:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged
