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
- One-off spikes: Values exceeding mean + N*std in local window (default: 5 sigma)
- Periodic noise: Consistent background noise at specific frequencies (e.g., fan)
- Window size: 0.01 seconds (default)
- Minimum duration: 0.01 seconds (default)

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
    periodic_noise_threshold: float = 0.3,
    detect_periodic_noise: bool = True,
) -> List[Tuple[float, float]]:
    """Detect artifactual noise (spikes, outliers) in audio channels.

    Detects two types of artifacts:
    1. One-off or time-limited spikes: Intermittent background noise (e.g., 
       someone talking, glass falling) detected via local statistical thresholds
    2. Consistent periodic background noise: Noise at specific frequencies 
       (e.g., fan running) detected via spectral analysis

    Args:
        df: Audio DataFrame with time and channel data
        time_col: Name of time column
        audio_channels: List of channel names to check (default: ch1-ch4)
        spike_threshold_sigma: Number of standard deviations for spike detection
        spike_window_s: Window size for computing local statistics (seconds)
        min_artifact_duration_s: Minimum continuous artifact duration to report
        periodic_noise_threshold: Threshold for detecting periodic noise (0-1)
        detect_periodic_noise: Whether to detect periodic background noise

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

        # Type 1: Detect one-off spikes via local statistics
        local_mean = _sliding_window_mean(ch_data, window_samples)
        local_std = _sliding_window_std(ch_data, window_samples)

        # Detect spikes: values exceeding threshold * std from local mean
        threshold = local_mean + spike_threshold_sigma * local_std
        ch_artifacts = np.abs(ch_data) > threshold
        artifact_mask |= ch_artifacts

        # Type 2: Detect periodic background noise via spectral analysis
        if detect_periodic_noise:
            periodic_mask = _detect_periodic_noise(ch_data, fs, periodic_noise_threshold)
            artifact_mask |= periodic_mask

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
    periodic_noise_threshold: float = 0.3,
    detect_periodic_noise: bool = False,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Run comprehensive raw audio QC checks.

    Detects both signal dropout and artifactual noise in raw audio recordings.
    
    Artifactual noise detection includes:
    - One-off or time-limited spikes (intermittent background noise)
    - Consistent periodic background noise at specific frequencies

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
        periodic_noise_threshold: Threshold for periodic noise detection (0-1)
        detect_periodic_noise: Whether to detect periodic background noise

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
        periodic_noise_threshold=periodic_noise_threshold,
        detect_periodic_noise=detect_periodic_noise,
    )

    return dropout_intervals, artifact_intervals


def detect_signal_dropout_per_mic(
    df: pd.DataFrame,
    time_col: str = "tt",
    audio_channels: list[str] | None = None,
    *,
    silence_threshold: float = 0.001,
    flatline_threshold: float = 0.0001,
    window_size_s: float = 0.5,
    min_dropout_duration_s: float = 0.1,
) -> dict[str, List[Tuple[float, float]]]:
    """Detect signal dropout per microphone channel.

    Returns bad intervals for each channel separately, allowing per-mic QC assessment.

    Args:
        df: Audio DataFrame with time and channel data
        time_col: Name of time column
        audio_channels: List of channel names to check (default: ch1-ch4)
        silence_threshold: Maximum RMS amplitude for silence detection
        flatline_threshold: Maximum variance for flatline detection
        window_size_s: Size of sliding window for detection (seconds)
        min_dropout_duration_s: Minimum continuous dropout duration to report

    Returns:
        Dictionary mapping channel name to list of (start_time, end_time) tuples
    """
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
        intervals = _mask_to_intervals(
            ch_dropout, time_s, min_duration_s=min_dropout_duration_s
        )
        
        per_mic_intervals[ch] = intervals

    return per_mic_intervals


def detect_artifactual_noise_per_mic(
    df: pd.DataFrame,
    time_col: str = "tt",
    audio_channels: list[str] | None = None,
    *,
    spike_threshold_sigma: float = 5.0,
    spike_window_s: float = 0.01,
    min_artifact_duration_s: float = 0.01,
    periodic_noise_threshold: float = 0.3,
    detect_periodic_noise: bool = True,
) -> dict[str, List[Tuple[float, float]]]:
    """Detect artifactual noise per microphone channel.

    Returns bad intervals for each channel separately, allowing per-mic QC assessment.

    Args:
        df: Audio DataFrame with time and channel data
        time_col: Name of time column
        audio_channels: List of channel names to check (default: ch1-ch4)
        spike_threshold_sigma: Number of standard deviations for spike detection
        spike_window_s: Window size for computing local statistics (seconds)
        min_artifact_duration_s: Minimum continuous artifact duration to report
        periodic_noise_threshold: Threshold for detecting periodic noise (0-1)
        detect_periodic_noise: Whether to detect periodic background noise

    Returns:
        Dictionary mapping channel name to list of (start_time, end_time) tuples
    """
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
    window_samples = max(int(fs * spike_window_s), 5)

    # Check each channel for artifacts separately
    per_mic_intervals = {}

    for ch in available_channels:
        ch_data = pd.to_numeric(df[ch], errors="coerce").to_numpy()
        artifact_mask = np.zeros(len(df), dtype=bool)

        # Type 1: Detect one-off spikes via local statistics
        local_mean = _sliding_window_mean(ch_data, window_samples)
        local_std = _sliding_window_std(ch_data, window_samples)

        # Detect spikes: values exceeding threshold * std from local mean
        threshold = local_mean + spike_threshold_sigma * local_std
        ch_artifacts = np.abs(ch_data) > threshold
        artifact_mask |= ch_artifacts

        # Type 2: Detect periodic background noise via spectral analysis
        if detect_periodic_noise:
            periodic_mask = _detect_periodic_noise(ch_data, fs, periodic_noise_threshold)
            artifact_mask |= periodic_mask

        # Convert artifact mask to time intervals for this channel
        intervals = _mask_to_intervals(
            artifact_mask, time_s, min_duration_s=min_artifact_duration_s
        )
        
        per_mic_intervals[ch] = intervals

    return per_mic_intervals


def run_raw_audio_qc_per_mic(
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
    periodic_noise_threshold: float = 0.3,
    detect_periodic_noise: bool = False,
) -> dict[str, List[Tuple[float, float]]]:
    """Run comprehensive raw audio QC checks per microphone.

    Detects both signal dropout and artifactual noise in raw audio recordings,
    returning results separately for each microphone channel.

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
        periodic_noise_threshold: Threshold for periodic noise detection (0-1)
        detect_periodic_noise: Whether to detect periodic background noise

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

    artifact_per_mic = detect_artifactual_noise_per_mic(
        df,
        time_col=time_col,
        audio_channels=audio_channels,
        spike_threshold_sigma=spike_threshold_sigma,
        spike_window_s=spike_window_s,
        min_artifact_duration_s=min_artifact_duration_s,
        periodic_noise_threshold=periodic_noise_threshold,
        detect_periodic_noise=detect_periodic_noise,
    )

    # Merge dropout and artifact intervals per channel
    per_mic_bad_intervals = {}
    all_channels = set(list(dropout_per_mic.keys()) + list(artifact_per_mic.keys()))
    
    for ch in all_channels:
        dropout_intervals = dropout_per_mic.get(ch, [])
        artifact_intervals = artifact_per_mic.get(ch, [])
        merged = merge_bad_intervals(dropout_intervals, artifact_intervals)
        per_mic_bad_intervals[ch] = merged

    return per_mic_bad_intervals


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


def adjust_bad_intervals_for_sync(
    bad_intervals: List[Tuple[float, float]],
    audio_stomp_time: float,
    bio_stomp_time: float,
) -> List[Tuple[float, float]]:
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
    adjusted_intervals = [
        (start + offset, end + offset)
        for start, end in bad_intervals
    ]
    
    return adjusted_intervals


def check_cycle_in_bad_interval(
    cycle_start_time: float,
    cycle_end_time: float,
    bad_intervals: List[Tuple[float, float]],
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
            total_overlap += (overlap_end - overlap_start)
    
    # Check if overlap exceeds threshold
    overlap_fraction = total_overlap / cycle_duration
    return overlap_fraction >= overlap_threshold


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
