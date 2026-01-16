"""Module for synchronizing audio data (pickled DataFrame) with biomechanics data.
Uses foot stomp events (Sync Left and Sync Right) to align the two datasets.
Stomp events are identified by the first peak in the audio channels exceeding
a threshold defined by the overall signal statistics."""

import logging
from datetime import timedelta
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd

try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# Modules for getting biomechanics metadata and stomp times
def get_biomechanics_metadata(
    directory: Path,
    sheet_name: str,
) -> pd.DataFrame:
    """Load biomechanics metadata from a pickled DataFrame in the given directory."""

    # Find the file that contains the term 'biomechanics' in its name
    bio_file = next((
        (f for f in directory.iterdir() if "biomechanics" in f.name.lower()),
    ), None)
    if not bio_file or not bio_file.is_file():
        raise FileNotFoundError(f"BiomechanicsRecording data file not found: {bio_file}")

    # Load the biomechanics metadata, which will be in an Excel
    # file with the specified sheet name.
    return pd.read_excel(
        bio_file,
        sheet_name=sheet_name if sheet_name else "metadata")


def get_event_metadata(
    bio_meta: pd.DataFrame,
    event_name: str,
) -> pd.DataFrame:
    """Extract event metadata for a specific event from biomechanics metadata.

    Strips leading and trailing whitespace from event names to handle
    erroneous keystrokes in the data.
    """
    event_name_stripped = event_name.strip()
    event_metadata = bio_meta.loc[
        bio_meta["Event Info"].str.strip() == event_name_stripped
    ]
    if event_metadata.empty:
        raise ValueError(f"No events found for: {event_name_stripped}")

    return event_metadata


def get_stomp_time(
    bio_meta: pd.DataFrame,
    foot: str,
) -> timedelta:
    """Extract the timestamp of the foot stomp event from biomechanics metadata.
    foot: 'left' or 'right'"""

    event_name = "Sync Left" if foot.lower() == "left" else "Sync Right"

    event_meta_data = get_event_metadata(bio_meta, event_name)

    stomp_times = event_meta_data["Time (sec)"].dropna().tolist()
    if not stomp_times:
        raise ValueError(f"No {foot} foot stomp events found in biomechanics metadata.")

    # Return the first stomp time as a timedelta object
    return pd.to_timedelta(stomp_times[0], unit='s').to_pytimedelta()


def get_right_stomp_time(
    bio_meta: pd.DataFrame,
) -> timedelta:
    """Extract the timestamp of the right foot stomp event from biomechanics metadata.
    Sync Right is a item in the first row Event Info. The second column Time (sec)
    will contain the timestamp."""

    return get_stomp_time(bio_meta, foot="right")


def get_left_stomp_time(
    bio_meta: pd.DataFrame,
) -> timedelta:
    """Extract the timestamp of the left foot stomp event from biomechanics metadata."""

    return get_stomp_time(bio_meta, foot="left")


# Modules for loading pickled audio data and labeling as right or left
# via audio metadata
def load_audio_data(
    audio_file: Path,
) -> pd.DataFrame:
    """Load the pickled audio data from the specified file."""

    return pd.read_pickle(audio_file)


def _tt_series_to_seconds(tt_series: pd.Series) -> np.ndarray:
    """Convert a `tt` time series to seconds.

    Args:
        tt_series: Series of time values in seconds or `pd.Timedelta`.

    Returns:
        Numpy array of seconds as floats.
    """
    if isinstance(tt_series.iloc[0], pd.Timedelta):
        return tt_series.dt.total_seconds().to_numpy()
    return tt_series.astype(float).to_numpy()


def _estimate_sampling_rate(tt_seconds: np.ndarray) -> float:
    """Estimate sampling rate (Hz) from `tt` seconds array.

    Computes the median time delta between consecutive timestamps and inverts
    to get frequency. Validates that timestamps are strictly monotonic increasing.

    Args:
        tt_seconds: Monotonic increasing array of timestamps in seconds.

    Returns:
        Sampling rate in Hz (1 / median_dt).

    Raises:
        ValueError: If timestamps are not strictly increasing or median dt is non-positive.
    """
    diffs = np.diff(tt_seconds)
    dt = float(np.median(diffs))
    if not np.all(diffs > 0) or dt <= 0:
        raise ValueError("Cannot estimate sampling rate: non-increasing tt")
    return 1.0 / dt


def _detect_stomp_by_rms_energy(
    audio_channels: np.ndarray,
    tt_seconds: np.ndarray,
    sr: float,
    search_duration: float = 20.0,
) -> tuple[float, float]:
    """Detect stomp using rolling RMS (root mean square) energy.

    Identifies the highest energy event in the first N seconds of recording.

    Args:
        audio_channels: Audio data as numpy array (samples × channels).
        tt_seconds: Time values in seconds (1D array).
        sr: Sampling rate in Hz.
        search_duration: Duration in seconds to search (default 20s).

    Returns:
        Tuple of (stomp_time_seconds, max_rms_energy).
    """
    # Limit search to first 20 seconds
    search_mask = tt_seconds <= search_duration
    search_audio = audio_channels[search_mask]
    search_tt = tt_seconds[search_mask]

    if len(search_audio) == 0:
        return float(tt_seconds[0]), 0.0

    # Downsample audio for faster processing (stomps are low-frequency events)
    # Target ~4kHz effective rate for stomp detection
    downsample_factor = max(1, int(sr / 4000))
    if downsample_factor > 1:
        search_audio = search_audio[::downsample_factor]
        search_tt = search_tt[::downsample_factor]
        effective_sr = sr / downsample_factor
    else:
        effective_sr = sr

    # Compute RMS energy in 50ms windows (typical stomp duration is 50-200ms)
    window_samples = max(2, int(0.05 * effective_sr))  # 50ms window

    # Use strided convolution for memory-efficient RMS computation
    # Instead of creating all windows at once, compute RMS in strides
    stride = max(1, window_samples // 4)  # 75% overlap for smooth detection
    n_windows = (len(search_audio) - window_samples) // stride + 1

    rms_energies = np.zeros(n_windows)
    rms_times = np.zeros(n_windows)

    for i in range(n_windows):
        start_idx = i * stride
        end_idx = start_idx + window_samples
        window = search_audio[start_idx:end_idx]
        # RMS across all channels: sqrt(mean(all_values^2))
        rms_energies[i] = np.sqrt(np.mean(window**2))
        mid_idx = start_idx + window_samples // 2
        rms_times[i] = search_tt[mid_idx]

    if len(rms_energies) > 0:
        peak_idx = int(np.argmax(rms_energies))
        return rms_times[peak_idx], rms_energies[peak_idx]

    return float(search_tt[0]), 0.0


def _detect_stomp_by_impact_onset(
    audio_channels: np.ndarray,
    tt_seconds: np.ndarray,
    sr: float,
    search_duration: float = 20.0,
) -> tuple[float, float]:
    """Detect stomp using impact/sudden onset detection.

    Identifies the timestamp where energy rises most sharply, characteristic
    of an impact (stomp). Uses derivative of smoothed energy envelope.

    Args:
        audio_channels: Audio data as numpy array (samples × channels).
        tt_seconds: Time values in seconds (1D array).
        sr: Sampling rate in Hz.
        search_duration: Duration in seconds to search (default 20s).

    Returns:
        Tuple of (stomp_time_seconds, max_onset_magnitude).
    """
    # Limit search to first 20 seconds
    search_mask = tt_seconds <= search_duration
    search_audio = audio_channels[search_mask]
    search_tt = tt_seconds[search_mask]

    if len(search_audio) == 0:
        return float(tt_seconds[0]), 0.0

    # Compute instantaneous energy (absolute value, smoothed)
    window_samples = max(1, int(0.01 * sr))  # 10ms smoothing window
    energy = np.sqrt(np.mean(search_audio**2, axis=1))

    # Apply smoothing to reduce noise
    if len(energy) > window_samples:
        smoothed_energy = np.convolve(
            energy,
            np.ones(window_samples) / window_samples,
            mode='same'
        )
    else:
        smoothed_energy = energy

    # Compute derivative (rate of change in energy)
    energy_derivative = np.gradient(smoothed_energy)

    # Stomp has sharp onset, so look for maximum positive derivative
    # Use absolute value to catch both rising and falling edges
    onset_magnitude = np.abs(energy_derivative)

    if len(onset_magnitude) > 0:
        peak_idx = int(np.argmax(onset_magnitude))
        return search_tt[peak_idx], onset_magnitude[peak_idx]

    return float(search_tt[0]), 0.0


def _detect_stomp_by_frequency_content(
    audio_channels: np.ndarray,
    tt_seconds: np.ndarray,
    sr: float,
    search_duration: float = 20.0,
) -> tuple[float, float]:
    """Detect stomp using frequency domain analysis.

    Contact microphone stomp signals typically have dominant frequency
    content in the 100-1000 Hz range. This method identifies the time window
    with the highest energy in this frequency band.

    Args:
        audio_channels: Audio data as numpy array (samples × channels).
        tt_seconds: Time values in seconds (1D array).
        sr: Sampling rate in Hz.
        search_duration: Duration in seconds to search (default 20s).

    Returns:
        Tuple of (stomp_time_seconds, max_frequency_energy).
    """
    try:
        from scipy import signal as scipy_signal
    except ImportError:
        logging.warning("scipy not available; skipping frequency-based stomp detection")
        return float(tt_seconds[0]), 0.0

    # Limit search to first 20 seconds
    search_mask = tt_seconds <= search_duration
    search_audio = audio_channels[search_mask]
    search_tt = tt_seconds[search_mask]

    if len(search_audio) == 0:
        return float(tt_seconds[0]), 0.0

    # Use sliding window STFT to find time-frequency content
    # Window size: 500ms (typical stomp has significant energy for ~100-300ms)
    window_duration = 0.5
    hop_duration = 0.1  # 100ms hop for sliding windows
    window_samples = int(window_duration * sr)
    hop_samples = int(hop_duration * sr)

    # Compute combined signal across all channels
    combined_signal = np.mean(np.abs(search_audio), axis=1)

    # Compute STFT for time-frequency representation
    try:
        frequencies, times, Sxx = scipy_signal.spectrogram(
            combined_signal,
            sr,
            window='hamming',
            nperseg=window_samples,
            noverlap=window_samples - hop_samples,
            scaling='spectrum'
        )
    except Exception as e:
        logging.warning(f"STFT computation failed: {e}")
        return float(search_tt[0]), 0.0

    # Extract energy in typical stomp frequency band (100-1000 Hz)
    freq_mask = (frequencies >= 100) & (frequencies <= 1000)
    if not np.any(freq_mask):
        # Fallback to all frequencies if band is empty
        freq_mask = np.ones(len(frequencies), dtype=bool)

    # Sum energy across selected frequency band
    band_energy = np.sum(Sxx[freq_mask, :], axis=0)

    if len(band_energy) > 0:
        peak_idx = int(np.argmax(band_energy))
        # Map back to time in original recording
        peak_time = search_tt[min(peak_idx * hop_samples + window_samples // 2,
                                   len(search_tt) - 1)]
        return peak_time, band_energy[peak_idx]

    return float(search_tt[0]), 0.0


def get_audio_stomp_time(
    audio_df: pd.DataFrame,
    recorded_knee: Optional[Literal["left", "right"]] = None,
    right_stomp_time: Optional[timedelta] = None,
    left_stomp_time: Optional[timedelta] = None,
    return_details: bool = False,
) -> Union[timedelta, tuple[timedelta, dict]]:
    """Detect the audio stomp time using multi-method approach.

    Identifies stomp events in the first 20 seconds of recording using three
    complementary methods:
    1. Rolling RMS (root mean square) energy to find the loudest event
    2. Impact/onset detection to find sharp energy transitions
    3. Frequency domain analysis for typical stomp frequency content (100-1000 Hz)

    When biomechanics metadata is available, uses it to validate the detected
    stomp times and determine which stomp (left/right) corresponds to the
    recorded knee.

    Args:
        audio_df: DataFrame with columns `tt`, `ch1`–`ch4`.
        recorded_knee: `"left"` or `"right"` when using biomechanics data.
        right_stomp_time: Biomechanics right stomp as `timedelta`.
        left_stomp_time: Biomechanics left stomp as `timedelta`.

    Returns:
        By default returns the selected stomp time as `timedelta` for backward compatibility.
        If `return_details=True`, returns a tuple of `(selected_stomp_time, detection_results_dict)` where
        detection_results_dict contains:
        - 'consensus_time': Median of three methods (seconds)
        - 'rms_time': RMS energy detection (seconds)
        - 'rms_energy': RMS energy value
        - 'onset_time': Impact onset detection (seconds)
        - 'onset_magnitude': Onset magnitude value
        - 'freq_time': Frequency domain detection (seconds)
        - 'freq_energy': Frequency band energy value

    Raises:
        ValueError: Invalid parameters or inability to estimate sampling rate.
    """
    # Validate parameters for biomechanics-guided detection
    if recorded_knee is not None:
        if right_stomp_time is None or left_stomp_time is None:
            raise ValueError(
                "When recorded_knee is specified, both right_stomp_time and left_stomp_time must be provided"
            )

    # Gather audio channels - prefer filtered channels (f_ch) if available, fall back to raw (ch)
    filtered_channels = ["f_ch1", "f_ch2", "f_ch3", "f_ch4"]
    raw_channels = ["ch1", "ch2", "ch3", "ch4"]

    if all(ch in audio_df.columns for ch in filtered_channels):
        # Use filtered channels for better stomp detection (higher SNR)
        channel_names = filtered_channels
    else:
        # Fall back to raw channels if filtered not available
        channel_names = raw_channels

    audio_channels = audio_df.loc[:, audio_df.columns.isin(channel_names)].values
    tt_series = audio_df["tt"]
    tt_seconds = _tt_series_to_seconds(tt_series)
    sr = _estimate_sampling_rate(tt_seconds)

    # Apply multi-method stomp detection on first 20 seconds
    rms_time, rms_energy = _detect_stomp_by_rms_energy(audio_channels, tt_seconds, sr)
    onset_time, onset_mag = _detect_stomp_by_impact_onset(audio_channels, tt_seconds, sr)
    freq_time, freq_energy = _detect_stomp_by_frequency_content(audio_channels, tt_seconds, sr)

    logging.debug(
        "Stomp detection methods: RMS (t=%.3fs, E=%.2f), "
        "Onset (t=%.3fs, M=%.2f), Frequency (t=%.3fs, E=%.2f)",
        rms_time, rms_energy, onset_time, onset_mag, freq_time, freq_energy
    )

    # Consensus approach: use median of the three detected times
    # This provides robustness against outliers from any single method
    detected_times = [rms_time, onset_time, freq_time]
    consensus_time = float(np.median(detected_times))

    logging.debug(
        "Consensus stomp time: %.3fs (from RMS: %.3fs, Onset: %.3fs, Freq: %.3fs)",
        consensus_time, rms_time, onset_time, freq_time
    )

    # Store detection results for visualization and logging
    detection_results = {
        'consensus_time': consensus_time,
        'rms_time': rms_time,
        'rms_energy': float(rms_energy),
        'onset_time': onset_time,
        'onset_magnitude': float(onset_mag),
        'freq_time': freq_time,
        'freq_energy': float(freq_energy),
    }

    # If we have biomechanics stomp metadata, use it to refine detection
    if recorded_knee is not None and right_stomp_time is not None and left_stomp_time is not None:
        right_target = float(right_stomp_time.total_seconds())
        left_target = float(left_stomp_time.total_seconds())

        # Find the two peaks closest to the biomechanics stomp times
        # by looking for local maxima in RMS energy

        # Search in first 20 seconds
        search_mask = tt_seconds <= 20.0
        search_audio = audio_channels[search_mask]
        search_tt = tt_seconds[search_mask]

        # Downsample for faster processing
        downsample_factor = max(1, int(sr / 4000))
        if downsample_factor > 1:
            search_audio = search_audio[::downsample_factor]
            search_tt = search_tt[::downsample_factor]
            effective_sr = sr / downsample_factor
        else:
            effective_sr = sr

        window_samples = max(2, int(0.1 * effective_sr))  # 100ms window

        # Memory-efficient rolling energy computation with striding
        stride = max(1, window_samples // 4)
        n_windows = (len(search_audio) - window_samples) // stride + 1

        rolling_energies = np.zeros(n_windows)
        rolling_times = np.zeros(n_windows)

        for i in range(n_windows):
            start_idx = i * stride
            end_idx = start_idx + window_samples
            window = search_audio[start_idx:end_idx]
            rolling_energies[i] = np.sqrt(np.mean(window**2))
            mid_idx = start_idx + window_samples // 2
            rolling_times[i] = search_tt[mid_idx]

        # Find peaks in the energy signal (local maxima)
        if SCIPY_AVAILABLE:
            try:
                # Distance in terms of rolling_energies indices (after striding)
                # 0.5 seconds * effective_sr samples/sec / stride samples/window
                min_peak_distance = int(0.5 * effective_sr / stride)
                peaks, peak_props = find_peaks(
                    rolling_energies,
                    height=np.max(rolling_energies) * 0.3,  # At least 30% of max energy
                    distance=min_peak_distance  # At least 0.5s apart
                )
            except Exception as e:
                logging.debug("find_peaks failed: %s; using fallback", e)
                # Fallback if find_peaks fails
                peaks = np.array([np.argmax(rolling_energies)])
                peak_props = {"peak_heights": np.array([rolling_energies[peaks[0]]])}
        else:
            # Fallback when scipy not available
            peaks = np.array([np.argmax(rolling_energies)])
            peak_props = {"peak_heights": np.array([rolling_energies[peaks[0]]])}

        if len(peaks) > 0:
            peak_times = rolling_times[peaks]
            peak_heights = peak_props["peak_heights"]

            logging.debug(
                "Detected %d peaks in RMS energy: times=%s, heights=%s",
                len(peaks), [f"{t:.3f}" for t in peak_times], [f"{h:.2f}" for h in peak_heights]
            )

            # If we have exactly 2 peaks, assign them to left/right based on biomechanics
            if len(peaks) >= 2:
                # Get the two highest peaks
                sorted_indices = np.argsort(peak_heights)[-2:]
                first_peak_idx, second_peak_idx = sorted(sorted_indices)
                first_peak_time = peak_times[first_peak_idx]
                second_peak_time = peak_times[second_peak_idx]

                # Determine which stomp occurred first based on BIOMECHANICS times
                if right_target < left_target:
                    # Right stomp is first in biomechanics
                    right_peak_time = first_peak_time
                    left_peak_time = second_peak_time
                else:
                    # Left stomp is first in biomechanics
                    left_peak_time = first_peak_time
                    right_peak_time = second_peak_time

                selected_time = right_peak_time if recorded_knee == "right" else left_peak_time

                logging.debug(
                    "Audio stomp detection (with biomechanics guidance): "
                    "left=%.3fs, right=%.3fs, selected=%s=%.3fs",
                    left_peak_time, right_peak_time, recorded_knee, selected_time
                )
            else:
                # Only 1 peak found; use consensus detection
                selected_time = consensus_time
                logging.debug(
                    "Only %d peak(s) found; using consensus stomp time: %.3fs",
                    len(peaks), selected_time
                )
        else:
            # No peaks found; use consensus detection
            selected_time = consensus_time
            logging.debug("No peaks found in RMS energy; using consensus stomp time: %.3fs", consensus_time)

        selected_td = pd.Timedelta(seconds=selected_time).to_pytimedelta()
        return (selected_td, detection_results) if return_details else selected_td

    # No biomechanics metadata: return consensus stomp time
    selected_td = pd.Timedelta(seconds=consensus_time).to_pytimedelta()
    return (selected_td, detection_results) if return_details else selected_td


def sync_audio_with_biomechanics(
    audio_stomp_time: timedelta,
    bio_stomp_time: timedelta,
    audio_df: "pd.DataFrame",
    bio_df: "pd.DataFrame",
    bio_start_time: Optional[timedelta] = None,
    bio_end_time: Optional[timedelta] = None,
    maneuver_key: Optional[str] = None,
    knee_side: Optional[str] = None,
    pass_number: Optional[int] = None,
    speed: Optional[str] = None,
) -> "pd.DataFrame":
    """Synchronize audio with biomechanics using stomp times.

        Clips biomechanics and audio to maneuver window ±0.5s before merging to
        reduce file size. Then interpolates biomechanics to match audio
        sampling rate.

    Args:
        audio_stomp_time: Timestamp of the foot stomp event in
            the audio data.
        bio_stomp_time: Timestamp of the foot stomp event in
            the biomechanics data.
        audio_df: DataFrame containing audio data with timestamps.
        bio_df: DataFrame containing biomechanics data with
            timestamps.
        bio_start_time: Start time of biomechanics maneuver
            window (timedelta). If provided, biomechanics before
            (start - 0.5s) are excluded.
        bio_end_time: End time of biomechanics maneuver window
            (timedelta). If provided, biomechanics after (end + 0.5s)
            are excluded.
        maneuver_key: Type of maneuver (walk, sit_to_stand, flexion_extension)
            for improved logging. Optional.
        knee_side: Side of knee (Left, Right) for improved logging. Optional.
        pass_number: Pass number (for walking maneuvers) for improved logging.
            Optional, None for non-walk maneuvers.
        speed: Speed level (for walking maneuvers) for improved logging.
            Optional, None for non-walk maneuvers.

    Returns:
        Synchronized audio DataFrame with interpolated biomechanics
        data.
    """

    # Calculate time difference between biomechanics and audio stomp times
    time_difference = bio_stomp_time - audio_stomp_time
    # Convert the audio_df 'tt' column to timedelta (copy to avoid modifying original)
    audio_df = audio_df.copy()
    audio_df['tt'] = pd.to_timedelta(audio_df['tt'], unit='s')

    # Warn if audio coverage is shorter than biomechanics time span
    audio_duration = audio_df['tt'].max() - audio_df['tt'].min()
    bio_duration = bio_df['TIME'].max() - bio_df['TIME'].min()
    if bio_duration > audio_duration:
        print(
            "Warning: biomechanics duration exceeds audio coverage; "
            "synced output will be truncated to audio range.",
        )

    # Drop any trailing NaT values in the biomechanics TIME column
    bio_df = bio_df.dropna(subset=['TIME']).reset_index(drop=True)

    # Note: We will clip biomechanics AFTER adjusting audio timestamps
    # to ensure the clipped ranges are in the same time coordinate system

    # Clip audio to match biomechanics data range (if clipping was applied)
    # This ensures no stale unmatched rows in the result
    if bio_start_time is not None and bio_end_time is not None:
        margin = pd.Timedelta(seconds=0.5)
        bio_clip_start = bio_start_time - margin
        bio_clip_end = bio_end_time + margin

        # Get the actual TIME range from clipped biomechanics
        # (accounting for the time shift between audio and bio)
        time_diff_td = pd.to_timedelta(time_difference, unit='s')
        # Clipped bio TIME range in audio time coordinates:
        audio_match_start = bio_clip_start - time_diff_td
        audio_match_end = bio_clip_end - time_diff_td

        # QC: Validate that biomechanics window overlaps with audio recording
        audio_start = audio_df['tt'].min()
        audio_end = audio_df['tt'].max()

        if audio_match_end < audio_start or audio_match_start > audio_end:
            audio_start_bio_coords = audio_start + time_diff_td
            audio_end_bio_coords = audio_end + time_diff_td
            raise ValueError(
                f"Biomechanics time window [{bio_clip_start}, {bio_clip_end}] "
                f"does not overlap with audio recording range [{audio_start_bio_coords}, "
                f"{audio_end_bio_coords}] (in biomechanics time coordinates). "
                f"\n  Diagnostics:"
                f"\n  - Audio stomp time: {audio_stomp_time}"
                f"\n  - Bio stomp time: {bio_stomp_time}"
                f"\n  - Time difference (bio - audio): {time_difference} seconds"
                f"\n  - Audio duration: {(audio_end - audio_start).total_seconds()} seconds"
                f"\n  - Requested bio window duration: {(bio_clip_end - bio_clip_start).total_seconds()} seconds"
            )

        # Warn if only partial overlap
        if audio_match_start < audio_start or audio_match_end > audio_end:
            overlap_start = max(audio_match_start, audio_start)
            overlap_end = min(audio_match_end, audio_end)
            overlap_duration = (overlap_end - overlap_start).total_seconds()
            requested_duration = (audio_match_end - audio_match_start).total_seconds()
            coverage = overlap_duration / requested_duration if requested_duration > 0 else 0

            print(
                f"Warning: Biomechanics window partially outside audio range. "
                f"Coverage: {coverage:.1%}. "
                f"Audio range: [{audio_start}, {audio_end}], "
                f"Requested: [{audio_match_start}, {audio_match_end}]"
            )

        # Clip audio to ensure all kept rows will have matching biomechanics
        audio_df = audio_df[
            (audio_df['tt'] >= audio_match_start) &
            (audio_df['tt'] <= audio_match_end)
        ].reset_index(drop=True)

        # QC: Verify audio clipping didn't remove all data
        if audio_df.empty:
            raise ValueError(
                f"Audio clipping resulted in empty DataFrame. "
                f"Biomechanics window [{bio_clip_start}, {bio_clip_end}] "
                f"may be completely outside audio recording range."
            )
    # Adjust audio timestamps by the time difference (after clipping)
    audio_df['tt'] = (
        audio_df['tt'] + pd.to_timedelta(time_difference, unit='s')
    )

    # NOW clip biomechanics to match the adjusted audio range
    # (both are now in biomechanics time coordinates)
    if bio_start_time is not None and bio_end_time is not None:
        margin = pd.Timedelta(seconds=0.5)
        bio_clip_start = bio_start_time - margin
        bio_clip_end = bio_end_time + margin

        # Get actual audio time range after adjustment
        audio_start_adjusted = audio_df['tt'].min()
        audio_end_adjusted = audio_df['tt'].max()

        # Clip biomechanics to the overlap region
        # Use the intersection of requested biomechanics window and actual audio range
        effective_start = max(bio_clip_start, audio_start_adjusted)
        effective_end = min(bio_clip_end, audio_end_adjusted)

        bio_df = bio_df[
            (bio_df['TIME'] >= effective_start) &
            (bio_df['TIME'] <= effective_end)
        ].reset_index(drop=True)

        # QC: Verify biomechanics clipping didn't remove all data
        if bio_df.empty:
            raise ValueError(
                f"Biomechanics clipping resulted in empty DataFrame. "
                f"Requested bio window [{bio_clip_start}, {bio_clip_end}] "
                f"does not overlap with adjusted audio range [{audio_start_adjusted}, {audio_end_adjusted}]."
            )

        # Log effective window for diagnostics
        # Build maneuver context string
        if maneuver_key == "walk" and speed is not None and pass_number is not None:
            maneuver_context = f"{speed.capitalize()} Pass {pass_number}"
        elif maneuver_key and knee_side:
            maneuver_context = f"{maneuver_key.replace('_', '-')} {knee_side} Knee"
        else:
            maneuver_context = "Recording"

        print(
            f"Syncing {maneuver_context}: bio window [{bio_start_time}, {bio_end_time}], "
            f"effective window [{effective_start}, {effective_end}], "
            f"audio rows: {len(audio_df)}, bio rows: {len(bio_df)}"
        )

    # Use tolerance based on biomechanics sampling interval (defaults to 20 ms)
    tolerance = pd.Timedelta(milliseconds=20)
    if len(bio_df) > 1:
        median_step = bio_df['TIME'].diff().median()
        if pd.notna(median_step) and isinstance(median_step, pd.Timedelta):
            tolerance = max(tolerance, median_step)

    # Merge clipped audio with clipped biomechanics
    synchronized_df = pd.merge_asof(
        audio_df.sort_values('tt'),
        bio_df.sort_values('TIME'),
        right_on='TIME',
        left_on='tt',
        direction='nearest',  # Nearest with tolerance to avoid stale rows
        tolerance=tolerance,
    )
    # Interpolate biomechanics columns to upsample from ~120Hz to 52kHz
    # Identify biomechanics columns (everything except audio & tt)
    audio_cols = {'tt', 'ch1', 'ch2', 'ch3', 'ch4',
                  'f_ch1', 'f_ch2', 'f_ch3', 'f_ch4'}
    bio_cols = [col for col in synchronized_df.columns
                if col not in audio_cols and col != 'TIME']

    # Use linear interpolation (much faster than time-based for
    # large datasets). This is valid since audio samples are
    # uniformly spaced
    for col in bio_cols:
        # Only interpolate numeric columns
        if pd.api.types.is_numeric_dtype(synchronized_df[col]):
            synchronized_df[col] = (
                synchronized_df[col].interpolate(
                    method='linear',
                    limit_area='inside'  # Only interpolate valid values
                )
            )

    # QC: Validate synchronized DataFrame contains both audio and biomechanics data
    _validate_synchronized_dataframe(synchronized_df)

    return synchronized_df


def _validate_synchronized_dataframe(df: "pd.DataFrame") -> None:
    """Validate that synchronized DataFrame contains both audio and biomechanics data.

    Args:
        df: Synchronized DataFrame to validate.

    Raises:
        ValueError: If DataFrame is missing audio or biomechanics data.
    """
    if df.empty:
        raise ValueError(
            "Synchronized DataFrame is empty. No data after merge."
        )

    # Check for audio channels
    audio_channels = [col for col in df.columns if col in ['ch1', 'ch2', 'ch3', 'ch4']]
    if not audio_channels:
        raise ValueError(
            "Synchronized DataFrame contains no audio channels (ch1-ch4). "
            "Merge may have failed or audio data was lost during clipping."
        )

    # Check that audio channels have non-NaN data
    audio_data_present = False
    for ch in audio_channels:
        if df[ch].notna().any():
            audio_data_present = True
            break

    if not audio_data_present:
        raise ValueError(
            f"Synchronized DataFrame has audio channels but all values are NaN. "
            f"Audio data may be completely outside the biomechanics window. "
            f"DataFrame shape: {df.shape}, Time range: [{df['tt'].min()}, {df['tt'].max()}]"
        )

    # Check for biomechanics columns (exclude audio, tt, TIME, and frequency channels)
    audio_cols = {'tt', 'ch1', 'ch2', 'ch3', 'ch4', 'f_ch1', 'f_ch2', 'f_ch3', 'f_ch4', 'TIME'}
    bio_cols = [col for col in df.columns if col not in audio_cols]

    if not bio_cols:
        raise ValueError(
            "Synchronized DataFrame contains no biomechanics columns. "
            "Merge may have failed or biomechanics data was lost."
        )

    # Check that at least one biomechanics column has non-NaN data
    bio_data_present = False
    for col in bio_cols:
        series = df[col]

        # Attempt to coerce non-numeric columns (common when Excel data is parsed as object)
        if not pd.api.types.is_numeric_dtype(series):
            coerced = pd.to_numeric(series, errors="coerce")
            if coerced.notna().any():
                df[col] = coerced
                bio_data_present = True
                break
            continue

        if series.notna().any():
            bio_data_present = True
            break

    if not bio_data_present:
        raise ValueError(
            f"Synchronized DataFrame has biomechanics columns but all numeric values are NaN. "
            f"Biomechanics data may be completely outside the audio window. "
            f"DataFrame shape: {df.shape}, Time range: [{df['tt'].min()}, {df['tt'].max()}]"
        )

    # Calculate coverage statistics for informative messages
    total_rows = len(df)
    audio_valid_rows = df[audio_channels].notna().any(axis=1).sum()
    audio_coverage = audio_valid_rows / total_rows if total_rows > 0 else 0

    if audio_coverage < 0.5:
        print(
            f"Warning: Only {audio_coverage:.1%} of synchronized rows contain valid audio data. "
            f"({audio_valid_rows}/{total_rows} rows)"
        )


def _get_walking_event_name(
    speed: Literal["slow", "normal", "fast"],
    pass_number: int,
    event_type: Literal["Start", "End"],
) -> str:
    """Construct walking event name from speed and pass number.

    Args:
        speed: Walking speed ("slow", "normal", "fast").
        pass_number: Pass number (1, 2, etc.).
        event_type: Event type ("Start" or "End").

    Returns:
        Event name like "SS Pass 1 Start" or "NS Pass 2 End".
    """
    speed_map = {
        "slow": "SS",
        "normal": "NS",
        "fast": "FS",
    }
    speed_code = speed_map[speed]
    return f"{speed_code} Pass {pass_number} {event_type}"


def get_bio_start_time(
    event_metadata: "pd.DataFrame",
    maneuver: Literal["walk", "sit_to_stand", "flexion_extension"],
    speed: Optional[Literal["slow", "normal", "fast"]] = None,
    pass_number: Optional[int] = None,
) -> timedelta:
    """Get the start time of the biomechanics data for the specified maneuver.

    For sit-to-stand and flexion-extension, looks for "Movement Start" event.
    For walking, constructs event name from speed and pass number
    (e.g., "SS Pass 1 Start" for slow speed pass 1).

    Args:
        event_metadata: DataFrame containing biomechanics event metadata.
        maneuver: The maneuver type
            ("walk", "sit_to_stand", "flexion_extension").
        speed: Speed of the maneuver
            ("slow", "normal", "fast"), required for walk.
        pass_number: Pass number for walking maneuvers, required for walk.

    Returns:
        Start time of the biomechanics data for the specified maneuver.

    Raises:
        ValueError: If required parameters are missing or event not found.
    """
    if maneuver == "walk":
        if speed is None or pass_number is None:
            raise ValueError(
                f"speed and pass_number required for walk maneuver, "
                f"got speed={speed}, pass_number={pass_number}"
            )
        event_name = _get_walking_event_name(speed, pass_number, "Start")
    elif maneuver in ["sit_to_stand", "flexion_extension"]:
        event_name = "Movement Start"
    else:
        raise ValueError(f"Unknown maneuver: {maneuver}")

    event_data = get_event_metadata(event_metadata, event_name)
    start_times = event_data["Time (sec)"].dropna().tolist()

    if not start_times:
        raise ValueError(f"No start time found for event: {event_name}")

    return pd.to_timedelta(start_times[0], unit="s").to_pytimedelta()


def get_bio_end_time(
    event_metadata: "pd.DataFrame",
    maneuver: Literal["walk", "sit_to_stand", "flexion_extension"],
    speed: Optional[Literal["slow", "normal", "fast"]] = None,
    pass_number: Optional[int] = None,
) -> timedelta:
    """Get the end time of the biomechanics data for the specified maneuver.

    For sit-to-stand and flexion-extension, looks for "Movement End" event.
    For walking, constructs event name from speed and pass number
    (e.g., "SS Pass 1 End" for slow speed pass 1).

    Args:
        event_metadata: DataFrame containing biomechanics event metadata.
        maneuver: The maneuver type
            ("walk", "sit_to_stand", "flexion_extension").
        speed: Speed of the maneuver
            ("slow", "normal", "fast"), required for walk.
        pass_number: Pass number for walking maneuvers, required for walk.

    Returns:
        End time of the biomechanics data for the specified maneuver.

    Raises:
        ValueError: If required parameters are missing or event not found.
    """
    if maneuver == "walk":
        if speed is None or pass_number is None:
            raise ValueError(
                f"speed and pass_number required for walk maneuver, "
                f"got speed={speed}, pass_number={pass_number}"
            )
        event_name = _get_walking_event_name(speed, pass_number, "End")
    elif maneuver in ["sit_to_stand", "flexion_extension"]:
        event_name = "Movement End"
    else:
        raise ValueError(f"Unknown maneuver: {maneuver}")

    event_data = get_event_metadata(event_metadata, event_name)
    end_times = event_data["Time (sec)"].dropna().tolist()

    if not end_times:
        raise ValueError(f"No end time found for event: {event_name}")

    return pd.to_timedelta(end_times[0], unit="s").to_pytimedelta()


def plot_stomp_detection(
    audio_df: "pd.DataFrame",
    bio_df: "pd.DataFrame",
    synced_df: "pd.DataFrame",
    audio_stomp_time: timedelta,
    bio_stomp_left: timedelta,
    bio_stomp_right: timedelta,
    output_path: Path,
    detection_results: Optional[dict] = None,
) -> None:
    """Create a visualization of stomp detection with overview and per-channel plots.

    Top row: Full recording RMS analysis, synchronized window, RMS zoom, and timing summary.
    Bottom row: Per-channel voltage and RMS energy (dual y-axes).

    Saved to the same directory as synchronized data.

    Args:
        audio_df: DataFrame with full audio data (columns: tt, ch1-ch4)
        bio_df: DataFrame with full biomechanics data (includes TIME, joint angles)
        synced_df: DataFrame with synchronized/clipped audio and biomechanics
        audio_stomp_time: Detected audio stomp time
        bio_stomp_left: Biomechanics left stomp time
        bio_stomp_right: Biomechanics right stomp time
        output_path: Path where to save the figure (e.g., same dir as synced pickle)
        detection_results: Dict with detection method times and metrics (rms_time, onset_time, freq_time, etc.)
    """
    if not MATPLOTLIB_AVAILABLE:
        logging.debug("Matplotlib not available; skipping stomp visualization")
        return

    try:
        # Convert times to seconds
        audio_stomp_s = float(audio_stomp_time.total_seconds())

        # Extract detection method times if available
        rms_time_s = detection_results.get('rms_time', audio_stomp_s) if detection_results else audio_stomp_s
        onset_time_s = detection_results.get('onset_time', audio_stomp_s) if detection_results else audio_stomp_s
        freq_time_s = detection_results.get('freq_time', audio_stomp_s) if detection_results else audio_stomp_s
        bio_left_s = float(bio_stomp_left.total_seconds())
        bio_right_s = float(bio_stomp_right.total_seconds())

        # Create figure with 2x4 subplots (top row: overview, bottom row: per-channel)
        fig, axes = plt.subplots(2, 4, figsize=(28, 10))
        ax_left = axes[0, 0]
        ax_right = axes[0, 1]
        ax_zoom = axes[0, 2]
        ax_summary = axes[0, 3]
        ax_ch1 = axes[1, 0]
        ax_ch2 = axes[1, 1]
        ax_ch3 = axes[1, 2]
        ax_ch4 = axes[1, 3]

        # ===== LEFT PLOT: Full biomechanics + audio with stomp markers =====
        # Extract full audio channels - prefer filtered channels if available
        filtered_channels = ["f_ch1", "f_ch2", "f_ch3", "f_ch4"]
        raw_channels = ["ch1", "ch2", "ch3", "ch4"]

        if all(ch in audio_df.columns for ch in filtered_channels):
            channel_names = filtered_channels
        else:
            channel_names = raw_channels

        audio_channels = audio_df.loc[:, audio_df.columns.isin(channel_names)]
        tt_audio = _tt_series_to_seconds(audio_df["tt"])

        # Truncate to start of recording through a few seconds after both stomps
        last_stomp_time = max(audio_stomp_s, bio_left_s, bio_right_s)
        time_buffer = 2.0  # Show 2 seconds after last stomp
        truncate_end = last_stomp_time + time_buffer

        # Mask for audio data within truncated range
        audio_mask = (tt_audio >= tt_audio.min()) & (tt_audio <= truncate_end)
        tt_audio_trunc = tt_audio[audio_mask]
        audio_channels_trunc = audio_channels[audio_mask]

        # Mask for biomechanics data within truncated range
        bio_time = _tt_series_to_seconds(bio_df["TIME"])
        bio_mask = (bio_time >= bio_time.min()) & (bio_time <= truncate_end)
        bio_time_trunc = bio_time[bio_mask]
        bio_df_trunc = bio_df[bio_mask]

        # Compute RMS energy time series (same as stomp detection algorithm)
        # This shows what the algorithm actually analyzes to find the stomp
        audio_channels_np = audio_channels_trunc.values

        # Use same parameters as stomp detection for consistency
        sr = 52000  # Typical sampling rate
        downsample_factor = max(1, int(sr / 4000))
        if downsample_factor > 1:
            audio_downsampled = audio_channels_np[::downsample_factor]
            tt_downsampled = tt_audio_trunc[::downsample_factor]
            effective_sr = sr / downsample_factor
        else:
            audio_downsampled = audio_channels_np
            tt_downsampled = tt_audio_trunc
            effective_sr = sr

        # Compute RMS in 50ms sliding windows (same as detection algorithm)
        window_samples = max(2, int(0.05 * effective_sr))
        stride = max(1, window_samples // 4)
        n_windows = max(1, (len(audio_downsampled) - window_samples) // stride + 1)

        rms_energies = np.zeros(n_windows)
        rms_times = np.zeros(n_windows)

        for i in range(n_windows):
            start_idx = i * stride
            end_idx = start_idx + window_samples
            window = audio_downsampled[start_idx:end_idx]
            rms_energies[i] = np.sqrt(np.mean(window**2))
            mid_idx = start_idx + window_samples // 2
            if mid_idx < len(tt_downsampled):
                rms_times[i] = tt_downsampled[mid_idx]

        # Downsample RMS for plotting if needed
        max_plot_points = 10000
        rms_downsample = max(1, len(rms_energies) // max_plot_points)
        rms_times_plot = rms_times[::rms_downsample]
        rms_energies_plot = rms_energies[::rms_downsample]

        # Plot RMS energy (detection metric) on left (secondary axis)
        ax_left_audio = ax_left.twinx()
        ax_left_audio.plot(rms_times_plot, rms_energies_plot, 'b-', linewidth=1.5, alpha=0.8, label="RMS Energy (Detection Metric)")
        ax_left_audio.fill_between(rms_times_plot, 0, rms_energies_plot, alpha=0.2, color='blue')

        ax_left_audio.set_ylabel("RMS Energy\n(Stomp Detection Metric)", fontsize=11, color="blue")
        ax_left_audio.tick_params(axis="y", labelcolor="blue")

        # Find and plot knee angle from truncated bio_df
        knee_angle_col = None
        # Prefer explicit Z-axis knee angle, fall back to any knee angle (exclude velocities)
        preferred_cols = [
            col
            for col in bio_df_trunc.columns
            if "knee angle z" in col.lower() and "velocity" not in col.lower()
        ]
        if preferred_cols:
            knee_angle_col = preferred_cols[0]
        else:
            for col in bio_df_trunc.columns:
                col_lower = col.lower()
                if (
                    "knee angle" in col_lower
                    and "velocity" not in col_lower
                    and "_x" not in col_lower
                    and "_y" not in col_lower
                ):
                    knee_angle_col = col
                    break

        if knee_angle_col:
            bio_downsample = max(1, len(bio_time_trunc) // max_plot_points)
            bio_time_plot = bio_time_trunc[::bio_downsample]
            knee_angle_plot = bio_df_trunc[knee_angle_col].values[::bio_downsample]

            ax_left.plot(bio_time_plot, knee_angle_plot, "k-", linewidth=2, alpha=0.8, label="Knee Angle")
            ax_left.set_ylabel("Knee Angle (degrees)", fontsize=11, color="black")
            ax_left.tick_params(axis="y", labelcolor="black")

        # Add stomp markers with clear labels
        # Draw consensus time (median of three methods) as red dashed line
        if detection_results:
            consensus_time = detection_results.get('consensus_time', audio_stomp_s)
            ax_left.axvline(
                consensus_time, color="red", linestyle="--", linewidth=2.5,
                label=f"Consensus\n{consensus_time:.2f}s", zorder=10
            )
        else:
            ax_left.axvline(
                audio_stomp_s, color="red", linestyle="--", linewidth=2.5,
                label=f"Stomp\n{audio_stomp_s:.2f}s", zorder=10
            )
        # Add detection method lines if available
        if detection_results:
            ax_left.axvline(
                rms_time_s, color="darkred", linestyle="-", linewidth=1.5, alpha=0.6,
                label=f"RMS\n{rms_time_s:.2f}s", zorder=9
            )
            ax_left.axvline(
                onset_time_s, color="maroon", linestyle=":", linewidth=1.5, alpha=0.6,
                label=f"Onset\n{onset_time_s:.2f}s", zorder=9
            )
            ax_left.axvline(
                freq_time_s, color="brown", linestyle="-.", linewidth=1.5, alpha=0.6,
                label=f"Freq\n{freq_time_s:.2f}s", zorder=9
            )
        ax_left.axvline(
            bio_left_s, color="green", linestyle=":", linewidth=2.5,
            label=f"Bio Left\n{bio_left_s:.2f}s", zorder=10
        )
        ax_left.axvline(
            bio_right_s, color="orange", linestyle=":", linewidth=2.5,
            label=f"Bio Right\n{bio_right_s:.2f}s", zorder=10
        )

        ax_left.set_xlabel("Time (seconds)", fontsize=11)
        ax_left.set_title("Stomp Detection: RMS Energy Analysis", fontsize=12, fontweight="bold")
        ax_left.set_xlim(tt_audio.min(), truncate_end)
        ax_left.grid(True, alpha=0.3)

        # Combine legends
        lines1, labels1 = ax_left.get_legend_handles_labels()
        lines2, labels2 = ax_left_audio.get_legend_handles_labels()
        ax_left.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

        # ===== TOP RIGHT PLOT: Zoomed RMS around detected stomp =====
        # Zoom around consensus time if available, otherwise around selected stomp
        zoom_center = detection_results.get('consensus_time', audio_stomp_s) if detection_results else audio_stomp_s
        zoom_window = 1.5  # seconds around detected stomp
        zoom_mask = (rms_times >= zoom_center - zoom_window) & (rms_times <= zoom_center + zoom_window)
        if np.any(zoom_mask):
            ax_zoom.plot(rms_times[zoom_mask], rms_energies[zoom_mask], 'b-', linewidth=1.5, alpha=0.9, label="RMS Energy (zoom)")
            ax_zoom.fill_between(rms_times[zoom_mask], 0, rms_energies[zoom_mask], alpha=0.2, color='blue')
        # Draw consensus time as red dashed line
        if detection_results:
            consensus_time = detection_results.get('consensus_time', audio_stomp_s)
            ax_zoom.axvline(consensus_time, color="red", linestyle="--", linewidth=2,
                           label=f"Consensus ({consensus_time:.2f}s)")
        else:
            ax_zoom.axvline(audio_stomp_s, color="red", linestyle="--", linewidth=2,
                           label=f"Stomp ({audio_stomp_s:.2f}s)")
        # Add detection method lines if available
        if detection_results:
            ax_zoom.axvline(rms_time_s, color="darkred", linestyle="-", linewidth=1.5, alpha=0.6,
                           label=f"RMS ({rms_time_s:.2f}s)")
            ax_zoom.axvline(onset_time_s, color="maroon", linestyle=":", linewidth=1.5, alpha=0.6,
                           label=f"Onset ({onset_time_s:.2f}s)")
            ax_zoom.axvline(freq_time_s, color="brown", linestyle="-.", linewidth=1.5, alpha=0.6,
                           label=f"Freq ({freq_time_s:.2f}s)")
        ax_zoom.axvline(bio_left_s, color="green", linestyle=":", linewidth=2, label=f"Bio Left ({bio_left_s:.2f}s)")
        ax_zoom.axvline(bio_right_s, color="orange", linestyle=":", linewidth=2, label=f"Bio Right ({bio_right_s:.2f}s)")
        ax_zoom.set_title("RMS Zoom (±1.5s)", fontsize=12, fontweight="bold")
        ax_zoom.set_xlabel("Time (seconds)")
        ax_zoom.set_ylabel("RMS Energy")
        ax_zoom.grid(True, alpha=0.3)
        ax_zoom.legend(fontsize=8, loc="upper left")

        # ===== RIGHT PLOT: Synchronized/clipped data =====
        # Use filtered channels if available, otherwise use raw
        filtered_channels = ["f_ch1", "f_ch2", "f_ch3", "f_ch4"]
        raw_channels = ["ch1", "ch2", "ch3", "ch4"]

        if all(ch in synced_df.columns for ch in filtered_channels):
            channel_names = filtered_channels
        else:
            channel_names = raw_channels

        synced_audio_channels = synced_df.loc[:, synced_df.columns.isin(channel_names)]

        # Compute RMS energy time series (same as detection algorithm)
        synced_audio_np = synced_audio_channels.values

        # Use same RMS window parameters
        sr = 52000
        downsample_factor = max(1, int(sr / 4000))
        if downsample_factor > 1 and len(synced_audio_np) > downsample_factor:
            synced_downsampled = synced_audio_np[::downsample_factor]
            effective_sr = sr / downsample_factor
        else:
            synced_downsampled = synced_audio_np
            effective_sr = sr

        tt_synced = _tt_series_to_seconds(synced_df["tt"])
        if downsample_factor > 1 and len(tt_synced) > downsample_factor:
            tt_synced_ds = tt_synced[::downsample_factor]
        else:
            tt_synced_ds = tt_synced

        # Compute RMS in 50ms windows
        window_samples = max(2, int(0.05 * effective_sr))
        stride = max(1, window_samples // 4)
        n_windows = max(1, (len(synced_downsampled) - window_samples) // stride + 1)

        synced_rms = np.zeros(n_windows)
        synced_rms_times = np.zeros(n_windows)

        for i in range(n_windows):
            start_idx = i * stride
            end_idx = min(start_idx + window_samples, len(synced_downsampled))
            if end_idx > start_idx:
                window = synced_downsampled[start_idx:end_idx]
                synced_rms[i] = np.sqrt(np.mean(window**2))
                mid_idx = start_idx + (end_idx - start_idx) // 2
                if mid_idx < len(tt_synced_ds):
                    synced_rms_times[i] = tt_synced_ds[mid_idx]

        # Downsample for plotting
        max_plot_points = 10000
        synced_downsample = max(1, len(synced_rms) // max_plot_points)
        synced_rms_times_plot = synced_rms_times[::synced_downsample]
        synced_rms_plot = synced_rms[::synced_downsample]

        # Plot RMS energy on right (secondary axis)
        ax_right_audio = ax_right.twinx()
        ax_right_audio.plot(synced_rms_times_plot, synced_rms_plot, 'b-', linewidth=1.5, alpha=0.8, label="RMS Energy")
        ax_right_audio.fill_between(synced_rms_times_plot, 0, synced_rms_plot, alpha=0.2, color='blue')

        ax_right_audio.set_ylabel("RMS Energy\n(Stomp Detection Metric)", fontsize=11, color="blue")
        ax_right_audio.tick_params(axis="y", labelcolor="blue")

        # Find and plot knee angle from synced data
        synced_knee_col = None
        preferred_synced = [
            col
            for col in synced_df.columns
            if "knee angle z" in col.lower() and "velocity" not in col.lower()
        ]
        if preferred_synced:
            synced_knee_col = preferred_synced[0]
        else:
            for col in synced_df.columns:
                col_lower = col.lower()
                if (
                    "knee angle" in col_lower
                    and "velocity" not in col_lower
                    and "_x" not in col_lower
                    and "_y" not in col_lower
                ):
                    synced_knee_col = col
                    break

        if synced_knee_col:
            # Use the full time array for knee angle since it's already downsampled in bio data
            knee_downsample = max(1, len(tt_synced) // max_plot_points)
            tt_knee_plot = tt_synced[::knee_downsample]
            knee_synced_plot = synced_df[synced_knee_col].values[::knee_downsample]
            ax_right.plot(tt_knee_plot, knee_synced_plot, "k-", linewidth=2, alpha=0.8, label="Knee Angle")
            ax_right.set_ylabel("Knee Angle (degrees)", fontsize=11, color="black")
            ax_right.tick_params(axis="y", labelcolor="black")

        ax_right.set_xlabel("Time (seconds)", fontsize=11)
        ax_right.set_title("Synchronized Pass/Maneuver Window", fontsize=12, fontweight="bold")
        ax_right.grid(True, alpha=0.3)

        # Combine legends for right plot
        lines3, labels3 = ax_right.get_legend_handles_labels()
        lines4, labels4 = ax_right_audio.get_legend_handles_labels()
        ax_right.legend(lines3 + lines4, labels3 + labels4, fontsize=8, loc="upper left")

        # ===== TOP RIGHT 2: Summary info =====
        summary_text = (
            f"Stomp Detection Summary\n\n"
        )
        if detection_results:
            consensus_time = detection_results.get('consensus_time', audio_stomp_s)
            summary_text += (
                f"Consensus: {consensus_time:.3f}s\n"
                f"  RMS: {rms_time_s:.3f}s\n"
                f"  Onset: {onset_time_s:.3f}s\n"
                f"  Freq: {freq_time_s:.3f}s\n\n"
            )
        else:
            summary_text += f"Selected Stomp: {audio_stomp_s:.3f}s\n\n"
        summary_text += (
            f"Bio Left: {bio_left_s:.3f}s\n"
            f"Bio Right: {bio_right_s:.3f}s\n\n"
        )
        if detection_results:
            consensus_time = detection_results.get('consensus_time', audio_stomp_s)
            summary_text += (
                f"Δ (Consensus - Left): {consensus_time - bio_left_s:.3f}s\n"
                f"Δ (Consensus - Right): {consensus_time - bio_right_s:.3f}s"
            )
        else:
            summary_text += (
                f"Δ (Stomp - Left): {audio_stomp_s - bio_left_s:.3f}s\n"
                f"Δ (Stomp - Right): {audio_stomp_s - bio_right_s:.3f}s"
            )
        ax_summary.text(0.5, 0.5,
                       summary_text,
                       ha="center", va="center", fontsize=10,
                       bbox=dict(boxstyle="round,pad=1", facecolor="lightgray", alpha=0.8),
                       family='monospace')
        ax_summary.set_title("Timing Summary", fontsize=12, fontweight="bold")
        ax_summary.axis("off")

        # Define zoom window around detected stomp for bottom plots
        zoom_window = 1.5
        # Zoom around consensus time if available, otherwise around selected stomp
        zoom_center = detection_results.get('consensus_time', audio_stomp_s) if detection_results else audio_stomp_s
        zoom_start = max(tt_synced.min(), zoom_center - zoom_window)
        zoom_end = zoom_center + zoom_window

        # ===== BOTTOM ROW: Per-channel voltage and RMS =====
        channel_colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
        channel_axes = [ax_ch1, ax_ch2, ax_ch3, ax_ch4]

        # Prepare data
        voltage_downsample = max(1, len(tt_audio_trunc) // max_plot_points)
        tt_voltage_plot = tt_audio_trunc[::voltage_downsample]
        volt_zoom_mask = (tt_voltage_plot >= zoom_start) & (tt_voltage_plot <= zoom_end)

        # Compute per-channel RMS using downsampled full audio
        per_ch_n_windows = max(1, (len(audio_downsampled) - window_samples) // stride + 1)
        rms_times_plot_full = rms_times
        rms_times_plot = rms_times_plot_full[::rms_downsample]
        rms_zoom_mask = (rms_times_plot >= zoom_start) & (rms_times_plot <= zoom_end)

        for idx, (ch, ax_ch) in enumerate(zip(channel_names, channel_axes)):
            if ch not in audio_channels_trunc.columns:
                continue

            # Plot voltage on primary axis
            ch_voltage = audio_channels_trunc[ch].values[::voltage_downsample]
            if np.any(volt_zoom_mask):
                ax_ch.plot(tt_voltage_plot[volt_zoom_mask], ch_voltage[volt_zoom_mask],
                          color=channel_colors[idx], linewidth=0.8, alpha=0.7, label="Voltage")
            else:
                ax_ch.plot(tt_voltage_plot, ch_voltage,
                          color=channel_colors[idx], linewidth=0.8, alpha=0.7, label="Voltage")

            # Compute and plot RMS on secondary axis
            if idx < audio_downsampled.shape[1]:
                ax_ch_rms = ax_ch.twinx()
                rms_ch = np.zeros(per_ch_n_windows)
                for i in range(per_ch_n_windows):
                    start_idx = i * stride
                    end_idx = min(start_idx + window_samples, len(audio_downsampled))
                    if end_idx > start_idx:
                        window = audio_downsampled[start_idx:end_idx, idx]
                        rms_ch[i] = np.sqrt(np.mean(window**2))

                rms_ch_plot = rms_ch[::rms_downsample]
                if np.any(rms_zoom_mask):
                    ax_ch_rms.plot(rms_times_plot[rms_zoom_mask], rms_ch_plot[rms_zoom_mask],
                                  color=channel_colors[idx], linewidth=1.2, alpha=0.5,
                                  linestyle='--', label="RMS")
                else:
                    ax_ch_rms.plot(rms_times_plot, rms_ch_plot,
                                  color=channel_colors[idx], linewidth=1.2, alpha=0.5,
                                  linestyle='--', label="RMS")

                ax_ch_rms.set_ylabel("RMS Energy", fontsize=9, color=channel_colors[idx], alpha=0.7)
                ax_ch_rms.tick_params(axis="y", labelcolor=channel_colors[idx], labelsize=8)

            # Draw consensus time as red dashed line
            if detection_results:
                consensus_time = detection_results.get('consensus_time', audio_stomp_s)
                ax_ch.axvline(consensus_time, color="red", linestyle="--", linewidth=1.0, alpha=0.6,
                             label=f"Consensus ({consensus_time:.2f}s)")
            else:
                ax_ch.axvline(audio_stomp_s, color="red", linestyle="--", linewidth=1.0, alpha=0.6,
                             label=f"Stomp ({audio_stomp_s:.2f}s)")
            # Add detection method lines if available
            if detection_results:
                ax_ch.axvline(rms_time_s, color="darkred", linestyle="-", linewidth=0.8, alpha=0.5,
                             label=f"RMS ({rms_time_s:.2f}s)")
                ax_ch.axvline(onset_time_s, color="maroon", linestyle=":", linewidth=0.7, alpha=0.5,
                             label=f"Onset ({onset_time_s:.2f}s)")
                ax_ch.axvline(freq_time_s, color="brown", linestyle="-.", linewidth=0.7, alpha=0.5,
                             label=f"Freq ({freq_time_s:.2f}s)")
            ax_ch.axvline(bio_left_s, color="green", linestyle=":", linewidth=0.8, alpha=0.6,
                         label=f"Left ({bio_left_s:.2f}s)")
            ax_ch.axvline(bio_right_s, color="orange", linestyle=":", linewidth=0.8, alpha=0.6,
                         label=f"Right ({bio_right_s:.2f}s)")

            # Formatting
            ax_ch.set_title(f"{ch.upper()}", fontsize=11, fontweight="bold")
            ax_ch.set_xlabel("Time (seconds)", fontsize=9)
            ax_ch.set_ylabel("Amplitude", fontsize=9, color=channel_colors[idx])
            ax_ch.tick_params(axis="y", labelcolor=channel_colors[idx], labelsize=8)
            ax_ch.tick_params(axis="x", labelsize=8)
            ax_ch.grid(True, alpha=0.3)
            ax_ch.legend(fontsize=6, loc="upper right", ncol=1)

        fig.tight_layout()
        # Save figure
        fig_path = output_path.with_suffix(".png")
        # Ensure parent directory exists
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        logging.info("Saved stomp detection visualization to %s", fig_path)

    except Exception as e:
        logging.warning("Failed to create stomp visualization: %s", str(e))
