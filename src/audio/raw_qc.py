"""Raw audio quality control for detecting signal dropout and artifactual noise.

This module provides QC checks for unprocessed/raw audio recordings to identify:
1. Signal dropout from all or any of the microphones (silence/flatline detection)
2. Artifactual noise (spikes, outliers, abnormal patterns)

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
- Spikes: Values exceeding mean + N*std in local window (default: 5 sigma)
- Window size: 0.01 seconds (default)
- Minimum duration: 0.01 seconds (default)

Thresholds can be adjusted based on signal characteristics and recording conditions.

Integration with Processing Pipeline
------------------------------------
This module is integrated into the bin processing stage (`_process_bin_stage` in
`participant.py`). When processing .bin files:
1. Raw audio is read from .bin file
2. QC is performed to detect dropout and artifacts
3. Bad intervals are stored in AudioProcessingRecord as QC_not_passed
4. Processing logs are updated with QC results
5. Clean audio proceeds to frequency augmentation

The QC_not_passed field in logs contains a string representation of
list of (start_time, end_time) tuples indicating sections that did not pass QC.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def detect_signal_dropout(
    df: pd.DataFrame,
    time_col: str = "tt",
    audio_channels: list[str] | None = None,
    *,
    silence_threshold: float = 0.001,
    flatline_threshold: float = 0.0001,
    window_size_s: float = 0.5,
    min_dropout_duration_s: float = 0.1,
) -> List[Tuple[float, float]]:
    """Detect signal dropout (silence or flatline) in audio channels.

    Signal dropout is identified when:
    1. Signal amplitude is below silence_threshold (near-zero voltage)
    2. Signal variance is below flatline_threshold (unchanging/stuck sensor)

    Args:
        df: Audio DataFrame with time and channel data
        time_col: Name of time column
        audio_channels: List of channel names to check (default: ch1-ch4)
        silence_threshold: Maximum RMS amplitude for silence detection
        flatline_threshold: Maximum variance for flatline detection
        window_size_s: Size of sliding window for detection (seconds)
        min_dropout_duration_s: Minimum continuous dropout duration to report

    Returns:
        List of (start_time, end_time) tuples indicating dropout periods
    """
    if audio_channels is None:
        audio_channels = ["ch1", "ch2", "ch3", "ch4"]

    # Filter to available channels
    available_channels = [ch for ch in audio_channels if ch in df.columns]
    if not available_channels or time_col not in df.columns:
        return []

    # Convert time to seconds
    time_s = pd.to_numeric(
        pd.to_timedelta(df[time_col], unit="s").dt.total_seconds(),
        errors="coerce",
    ).to_numpy()

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
    dropout_intervals = _mask_to_intervals(
        dropout_mask, time_s, min_duration_s=min_dropout_duration_s
    )

    return dropout_intervals


def detect_artifactual_noise(
    df: pd.DataFrame,
    time_col: str = "tt",
    audio_channels: list[str] | None = None,
    *,
    spike_threshold_sigma: float = 5.0,
    spike_window_s: float = 0.01,
    min_artifact_duration_s: float = 0.01,
) -> List[Tuple[float, float]]:
    """Detect artifactual noise (spikes, outliers) in audio channels.

    Artifacts are identified when signal values exceed a threshold based on
    local statistics, indicating abnormal spikes or noise bursts.

    Args:
        df: Audio DataFrame with time and channel data
        time_col: Name of time column
        audio_channels: List of channel names to check (default: ch1-ch4)
        spike_threshold_sigma: Number of standard deviations for spike detection
        spike_window_s: Window size for computing local statistics (seconds)
        min_artifact_duration_s: Minimum continuous artifact duration to report

    Returns:
        List of (start_time, end_time) tuples indicating artifact periods
    """
    if audio_channels is None:
        audio_channels = ["ch1", "ch2", "ch3", "ch4"]

    # Filter to available channels
    available_channels = [ch for ch in audio_channels if ch in df.columns]
    if not available_channels or time_col not in df.columns:
        return []

    # Convert time to seconds
    time_s = pd.to_numeric(
        pd.to_timedelta(df[time_col], unit="s").dt.total_seconds(),
        errors="coerce",
    ).to_numpy()

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

        # Compute local mean and std using sliding window
        local_mean = _sliding_window_mean(ch_data, window_samples)
        local_std = _sliding_window_std(ch_data, window_samples)

        # Detect spikes: values exceeding threshold * std from local mean
        threshold = local_mean + spike_threshold_sigma * local_std
        ch_artifacts = np.abs(ch_data) > threshold
        artifact_mask |= ch_artifacts

    # Convert artifact mask to time intervals
    artifact_intervals = _mask_to_intervals(
        artifact_mask, time_s, min_duration_s=min_artifact_duration_s
    )

    return artifact_intervals


def run_raw_audio_qc(
    df: pd.DataFrame,
    time_col: str = "tt",
    audio_channels: list[str] | None = None,
    *,
    silence_threshold: float = 0.001,
    flatline_threshold: float = 0.0001,
    spike_threshold_sigma: float = 5.0,
    dropout_window_s: float = 0.5,
    spike_window_s: float = 0.01,
    min_dropout_duration_s: float = 0.1,
    min_artifact_duration_s: float = 0.01,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Run comprehensive raw audio QC checks.

    Detects both signal dropout and artifactual noise in raw audio recordings.

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
    )

    return dropout_intervals, artifact_intervals


def merge_bad_intervals(
    dropout_intervals: List[Tuple[float, float]],
    artifact_intervals: List[Tuple[float, float]],
    merge_gap_s: float = 0.5,
) -> List[Tuple[float, float]]:
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
    bad_intervals: List[Tuple[float, float]],
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


# Helper functions for sliding window operations

def _sliding_window_rms(data: np.ndarray, window_size: int) -> np.ndarray:
    """Compute RMS using sliding window."""
    result = np.zeros(len(data))
    half_window = window_size // 2

    for i in range(len(data)):
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        window = data[start:end]
        valid_window = window[np.isfinite(window)]
        if len(valid_window) > 0:
            result[i] = np.sqrt(np.mean(valid_window ** 2))
        else:
            result[i] = 0.0

    return result


def _sliding_window_variance(data: np.ndarray, window_size: int) -> np.ndarray:
    """Compute variance using sliding window."""
    result = np.zeros(len(data))
    half_window = window_size // 2

    for i in range(len(data)):
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        window = data[start:end]
        valid_window = window[np.isfinite(window)]
        if len(valid_window) > 1:
            result[i] = np.var(valid_window)
        else:
            result[i] = 0.0

    return result


def _sliding_window_mean(data: np.ndarray, window_size: int) -> np.ndarray:
    """Compute mean using sliding window."""
    result = np.zeros(len(data))
    half_window = window_size // 2

    for i in range(len(data)):
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        window = data[start:end]
        valid_window = window[np.isfinite(window)]
        if len(valid_window) > 0:
            result[i] = np.mean(valid_window)
        else:
            result[i] = 0.0

    return result


def _sliding_window_std(data: np.ndarray, window_size: int) -> np.ndarray:
    """Compute standard deviation using sliding window."""
    result = np.zeros(len(data))
    half_window = window_size // 2

    for i in range(len(data)):
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        window = data[start:end]
        valid_window = window[np.isfinite(window)]
        if len(valid_window) > 1:
            result[i] = np.std(valid_window)
        else:
            result[i] = 0.0

    return result


def _mask_to_intervals(
    mask: np.ndarray,
    time_s: np.ndarray,
    min_duration_s: float = 0.0,
) -> List[Tuple[float, float]]:
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
