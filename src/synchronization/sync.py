"""Module for synchronizing audio data (pickled DataFrame) with biomechanics data.
Uses foot stomp events (Sync Left and Sync Right) to align the two datasets.
Stomp events are identified by the first peak in the audio channels exceeding
a threshold defined by the overall signal statistics."""

import logging
from datetime import timedelta
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

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
        raise FileNotFoundError(f"BiomechanicsCycle data file not found: {bio_file}")

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


def get_audio_stomp_time(
    audio_df: pd.DataFrame,
    recorded_knee: Optional[Literal["left", "right"]] = None,
    right_stomp_time: Optional[timedelta] = None,
    left_stomp_time: Optional[timedelta] = None,
) -> timedelta:
    """Detect the audio stomp time based on biomechanics stomp times using area-under-curve.

    Uses area-under-curve (AUC) integration of acoustic energy across all 4 channels
    to identify stomp events. Searches within ±2s windows around biomechanics Sync
    Left/Right times. Determines temporal order from acoustic energy (earlier stomp
    typically has higher energy earlier in the recording).

    Args:
        audio_df: DataFrame with columns `tt`, `ch1`–`ch4`.
        recorded_knee: `"left"` or `"right"` when using biomechanics data.
        right_stomp_time: Biomechanics right stomp as `timedelta`.
        left_stomp_time: Biomechanics left stomp as `timedelta`.

    Returns:
        `timedelta` for the stomp corresponding to `recorded_knee`, or the
        peak energy time when metadata is absent.

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

    def _compute_auc_in_window(target_time_s: float, window_s: float = 0.5) -> tuple[float, float]:
        """Compute area-under-curve in a window and return (time_of_max_auc, total_auc).

        Args:
            target_time_s: Center time in seconds
            window_s: Half-window size in seconds (default 0.5s = ±500ms)

        Returns:
            Tuple of (time_of_peak_energy, total_area_under_curve)
        """
        # Find indices in window
        mask = (tt_seconds >= target_time_s - window_s) & (tt_seconds <= target_time_s + window_s)
        if not np.any(mask):
            return target_time_s, 0.0

        # Extract window data
        window_indices = np.where(mask)[0]
        window_tt = tt_seconds[window_indices]
        window_audio = audio_channels[mask]

        # Compute total AUC for each channel (use absolute value to capture all energy)
        channel_aucs = []
        for ch_idx in range(window_audio.shape[1]):
            ch_data = np.abs(window_audio[:, ch_idx])
            # Use trapezoidal integration for AUC
            try:
                auc = np.trapezoid(ch_data, window_tt)
            except AttributeError:
                # Fallback for older numpy versions
                auc = np.trapz(ch_data, window_tt)
            channel_aucs.append(auc)

        # Total acoustic energy is sum across all channels
        total_auc = sum(channel_aucs)

        # Find time with maximum instantaneous energy using a 100ms rolling window
        rolling_window_samples = max(2, int(0.1 * sr))  # 100ms, minimum 2 samples

        if len(window_audio) < rolling_window_samples:
            # Window too small, just use center
            peak_time = target_time_s
        else:
            rolling_energies = []
            rolling_times = []

            for i in range(len(window_audio) - rolling_window_samples + 1):
                frame = window_audio[i:i+rolling_window_samples]
                # Sum absolute energy across all channels for this frame
                frame_energy = np.sum(np.abs(frame))
                rolling_energies.append(frame_energy)
                mid_idx = i + rolling_window_samples // 2
                rolling_times.append(window_tt[mid_idx])

            if rolling_energies:
                peak_idx = int(np.argmax(rolling_energies))
                peak_time = rolling_times[peak_idx]
            else:
                peak_time = target_time_s

        return peak_time, total_auc

    # If we have biomechanics stomp metadata, search near those times
    if recorded_knee is not None and right_stomp_time is not None and left_stomp_time is not None:
        right_target = float(right_stomp_time.total_seconds())
        left_target = float(left_stomp_time.total_seconds())

        # Compute AUC and peak times for both knees
        right_peak_time, right_auc = _compute_auc_in_window(right_target)
        left_peak_time, left_auc = _compute_auc_in_window(left_target)

        logging.debug(
            "Audio stomp detection: left_auc=%.1f (t=%.2fs), right_auc=%.1f (t=%.2fs)",
            left_auc, left_peak_time, right_auc, right_peak_time
        )

        # Determine which stomp occurred first based on BIOMECHANICS times
        # This is the ground truth order
        if right_target < left_target:
            # Right stomp is first (temporally earlier in bio)
            if recorded_knee == "right":
                # We're recording right knee, so return the first peak
                selected_time = right_peak_time
            else:
                # We're recording left knee, so return the second peak
                selected_time = left_peak_time
        else:
            # Left stomp is first (temporally earlier in bio)
            if recorded_knee == "left":
                # We're recording left knee, so return the first peak
                selected_time = left_peak_time
            else:
                # We're recording right knee, so return the second peak
                selected_time = right_peak_time

        return pd.Timedelta(seconds=selected_time).to_pytimedelta()

    # No biomechanics metadata: find earliest high-energy event
    # Scan through entire recording with rolling window
    rolling_window_samples = max(1, int(0.1 * sr))  # 100ms
    rolling_energies = []
    rolling_times = []

    for i in range(len(audio_channels) - rolling_window_samples + 1):
        frame = audio_channels[i:i+rolling_window_samples]
        frame_energy = np.sum(np.abs(frame))
        rolling_energies.append(frame_energy)
        rolling_times.append(tt_seconds[i + rolling_window_samples // 2])

    if rolling_energies:
        peak_idx = np.argmax(rolling_energies)
        peak_time = rolling_times[peak_idx]
    else:
        peak_time = float(tt_seconds[0])

    return pd.Timedelta(seconds=peak_time).to_pytimedelta()


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
    # Convert the audio_df 'tt' column to timedelta
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
) -> None:
    """Create a visualization of stomp detection with two side-by-side plots.

    Left plot: Full biomechanics recording showing joint angle and audio channels with stomp markers.
    Right plot: Synchronized (clipped) audio and biomechanics data for the specific pass/maneuver.

    Saved to the same directory as synchronized data.

    Args:
        audio_df: DataFrame with full audio data (columns: tt, ch1-ch4)
        bio_df: DataFrame with full biomechanics data (includes TIME, joint angles)
        synced_df: DataFrame with synchronized/clipped audio and biomechanics
        audio_stomp_time: Detected audio stomp time
        bio_stomp_left: Biomechanics left stomp time
        bio_stomp_right: Biomechanics right stomp time
        output_path: Path where to save the figure (e.g., same dir as synced pickle)
    """
    if not MATPLOTLIB_AVAILABLE:
        logging.debug("Matplotlib not available; skipping stomp visualization")
        return

    try:
        # Convert times to seconds
        audio_stomp_s = float(audio_stomp_time.total_seconds())
        bio_left_s = float(bio_stomp_left.total_seconds())
        bio_right_s = float(bio_stomp_right.total_seconds())

        # Create figure with 2 side-by-side subplots
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(20, 6))

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

        # Downsample for plotting
        max_plot_points = 10000
        audio_downsample = max(1, len(tt_audio_trunc) // max_plot_points)
        tt_audio_plot = tt_audio_trunc[::audio_downsample]

        # Plot audio on left (secondary axis)
        ax_left_audio = ax_left.twinx()
        for i, ch in enumerate(channel_names):
            if ch in audio_channels_trunc.columns:
                ch_data = audio_channels_trunc[ch].values[::audio_downsample]
                ax_left_audio.plot(tt_audio_plot, ch_data, linewidth=0.5, alpha=0.4, label=f"Audio {ch}")

        ax_left_audio.set_ylabel("Audio Amplitude", fontsize=11, color="blue")
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

        # Add stomp markers
        ax_left.axvline(
            audio_stomp_s, color="red", linestyle="--", linewidth=2,
            label=f"Audio Stomp ({audio_stomp_s:.1f}s)"
        )
        ax_left.axvline(
            bio_left_s, color="green", linestyle=":", linewidth=2,
            label=f"Bio Left ({bio_left_s:.1f}s)"
        )
        ax_left.axvline(
            bio_right_s, color="orange", linestyle=":", linewidth=2,
            label=f"Bio Right ({bio_right_s:.1f}s)"
        )

        ax_left.set_xlabel("Time (seconds)", fontsize=11)
        ax_left.set_title(f"Stomp Detection: Start to {truncate_end:.1f}s", fontsize=12, fontweight="bold")
        ax_left.set_xlim(tt_audio.min(), truncate_end)
        ax_left.grid(True, alpha=0.3)

        # Combine legends
        lines1, labels1 = ax_left.get_legend_handles_labels()
        lines2, labels2 = ax_left_audio.get_legend_handles_labels()
        ax_left.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

        # ===== RIGHT PLOT: Synchronized/clipped data =====
        # Use filtered channels if available, otherwise use raw
        filtered_channels = ["f_ch1", "f_ch2", "f_ch3", "f_ch4"]
        raw_channels = ["ch1", "ch2", "ch3", "ch4"]

        if all(ch in synced_df.columns for ch in filtered_channels):
            channel_names = filtered_channels
        else:
            channel_names = raw_channels

        synced_audio_channels = synced_df.loc[:, synced_df.columns.isin(channel_names)]
        tt_synced = _tt_series_to_seconds(synced_df["tt"])
        synced_downsample = max(1, len(tt_synced) // max_plot_points)
        tt_synced_plot = tt_synced[::synced_downsample]

        # Plot synced audio on right (secondary axis)
        ax_right_audio = ax_right.twinx()
        colors = ["b", "g", "r", "m"]
        for i, ch in enumerate(channel_names):
            if ch in synced_audio_channels.columns:
                ch_data = synced_audio_channels[ch].values[::synced_downsample]
                ax_right_audio.plot(tt_synced_plot, ch_data, colors[i], linewidth=0.8, alpha=0.6, label=f"{ch}")

        ax_right_audio.set_ylabel("Audio Amplitude", fontsize=11, color="blue")
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
            knee_synced_plot = synced_df[synced_knee_col].values[::synced_downsample]
            ax_right.plot(tt_synced_plot, knee_synced_plot, "k-", linewidth=2, alpha=0.8, label="Knee Angle")
            ax_right.set_ylabel("Knee Angle (degrees)", fontsize=11, color="black")
            ax_right.tick_params(axis="y", labelcolor="black")

        ax_right.set_xlabel("Time (seconds)", fontsize=11)
        ax_right.set_title("Synchronized Pass/Maneuver Window", fontsize=12, fontweight="bold")
        ax_right.grid(True, alpha=0.3)

        # Combine legends for right plot
        lines3, labels3 = ax_right.get_legend_handles_labels()
        lines4, labels4 = ax_right_audio.get_legend_handles_labels()
        ax_right.legend(lines3 + lines4, labels3 + labels4, fontsize=8, loc="upper left")

        # Save figure
        fig_path = output_path.with_suffix(".png")
        fig.savefig(fig_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        logging.info("Saved stomp detection visualization to %s", fig_path)

    except Exception as e:
        logging.warning("Failed to create stomp visualization: %s", str(e))
