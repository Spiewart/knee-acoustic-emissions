"""Module for synchronizing audio data (pickled DataFrame) with biomechanics data.
Uses foot stomp events (Sync Left and Sync Right) to align the two datasets.
Stomp events are identified by the first peak in the audio channels exceeding
a threshold defined by the overall signal statistics."""

from datetime import timedelta
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

if TYPE_CHECKING:

    from pathlib import Path


# Modules for getting biomechanics metadata and stomp times
def get_biomechanics_metadata(
    directory: "Path",
    sheet_name: str,
) -> "pd.DataFrame":
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
    bio_meta: "pd.DataFrame",
    event_name: str,
) -> "pd.DataFrame":
    """Extract event metadata for a specific event from biomechanics metadata."""

    event_metadata = bio_meta.loc[
        bio_meta["Event Info"] == event_name
    ]
    if event_metadata.empty:
        raise ValueError(f"No events found for: {event_name}")

    return event_metadata


def get_stomp_time(
    bio_meta: "pd.DataFrame",
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
    bio_meta: "pd.DataFrame",
) -> timedelta:
    """Extract the timestamp of the right foot stomp event from biomechanics metadata.
    Sync Right is a item in the first row Event Info. The second column Time (sec)
    will contain the timestamp."""

    return get_stomp_time(bio_meta, foot="right")


def get_left_stomp_time(
    bio_meta: "pd.DataFrame",
) -> timedelta:
    """Extract the timestamp of the left foot stomp event from biomechanics metadata."""

    return get_stomp_time(bio_meta, foot="left")


# Modules for loading pickled audio data and labeling as right or left
# via audio metadata
def load_audio_data(
    audio_file: "Path",
) -> "pd.DataFrame":
    """Load the pickled audio data from the specified file."""

    return pd.read_pickle(audio_file)


def get_audio_stomp_time(
    audio_df: "pd.DataFrame",
    recorded_knee: Optional[Literal["left", "right"]] = None,
    right_stomp_time: Optional[timedelta] = None,
    left_stomp_time: Optional[timedelta] = None,
) -> timedelta:
    """Get the timestamp (column = tt) of the foot stomp event from the
    audio data DataFrame.

    When both knees are stomped (dual recording), both stomps may appear
    in the audio. This function can disambiguate by finding 2 peaks and
    selecting the one that temporally matches the recorded knee's stomp
    from biomechanics data.

    Args:
        audio_df: DataFrame containing audio data with 'tt' and channel
            columns (ch1-ch4).
        recorded_knee: Which knee is being recorded ("left" or "right").
            Required if providing stomp times from biomechanics.
        right_stomp_time: Timestamp of right knee stomp from biomechanics
            metadata (timedelta).
        left_stomp_time: Timestamp of left knee stomp from biomechanics
            metadata (timedelta).

    Returns:
        Timestamp of the stomp event for the recorded knee.

    Raises:
        ValueError: If recorded_knee is provided without both stomp times,
            or if fewer than 2 peaks are found when expected.
    """
    # Validate parameters
    if recorded_knee is not None:
        if right_stomp_time is None or left_stomp_time is None:
            raise ValueError(
                "When recorded_knee is specified, both right_stomp_time "
                "and left_stomp_time must be provided"
            )

    # Get audio channel data
    audio_channels = audio_df.loc[
        :, audio_df.columns.isin(["ch1", "ch2", "ch3", "ch4"])
    ].values

    # Define threshold for stomp detection
    threshold = audio_channels.mean() + (3 * audio_channels.std())

    # If dual-knee disambiguation is needed
    if (recorded_knee is not None and
            right_stomp_time is not None and
            left_stomp_time is not None):

        # Compute max signal across all channels at each time point
        max_signal = audio_channels.max(axis=1)

        # Find peaks above threshold
        peak_indices, _ = find_peaks(
            max_signal,
            height=threshold,
            distance=int(0.5 * 52000),  # At least 0.5s apart (52kHz)
        )

        if len(peak_indices) < 2:
            raise ValueError(
                f"Expected at least 2 stomp peaks for dual-knee recording, "
                f"found {len(peak_indices)}"
            )

        # Get timestamps of top 2 peaks by amplitude
        peak_heights = max_signal[peak_indices]
        top_2_idx = np.argsort(peak_heights)[-2:]
        top_2_peaks = peak_indices[top_2_idx]
        top_2_peaks_sorted = np.sort(top_2_peaks)  # Sort by time

        # Get corresponding timestamps as datetime.timedelta
        peak_times = []
        for idx in top_2_peaks_sorted:
            tt_value = audio_df.loc[idx, 'tt']
            if isinstance(tt_value, pd.Timedelta):
                peak_times.append(tt_value.to_pytimedelta())
            elif isinstance(tt_value, float):
                peak_times.append(
                    pd.Timedelta(seconds=tt_value).to_pytimedelta()
                )
            else:  # Already a timedelta
                peak_times.append(tt_value)

        # Biomechanics stomp times are already datetime.timedelta
        # Determine which biomechanics stomp came first
        bio_stomps_sorted = sorted([
            (right_stomp_time, "right"),
            (left_stomp_time, "left"),
        ], key=lambda x: x[0])

        # Match audio peaks to biomechanics stomps based on temporal order
        # First audio peak -> first bio stomp, second audio peak -> second
        audio_to_bio_mapping = {
            "right": peak_times[0] if bio_stomps_sorted[0][1] == "right"
            else peak_times[1],
            "left": peak_times[0] if bio_stomps_sorted[0][1] == "left"
            else peak_times[1],
        }

        return audio_to_bio_mapping[recorded_knee]

    # Fallback: original single-peak detection
    # Find the first index where any channel exceeds the threshold
    stomp_index = audio_df.loc[:, audio_df.columns.isin([
        "ch1", "ch2", "ch3", "ch4"]
            )
        ].gt(threshold).idxmax()

    # Get the corresponding timestamp
    stomp_times = audio_df.loc[stomp_index, 'tt']
    # Get the earliest stomp_time
    stomp_time = stomp_times.min()

    return pd.to_timedelta(stomp_time, unit="s").to_pytimedelta()


def sync_audio_with_biomechanics(
    audio_stomp_time: timedelta,
    bio_stomp_time: timedelta,
    audio_df: "pd.DataFrame",
    bio_df: "pd.DataFrame",
    bio_start_time: Optional[timedelta] = None,
    bio_end_time: Optional[timedelta] = None,
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

    # Clip biomechanics to maneuver window ±0.5s to avoid stale rows
    # during merge (movement window provided)
    if bio_start_time is not None and bio_end_time is not None:
        margin = pd.Timedelta(seconds=0.5)
        bio_clip_start = bio_start_time - margin
        bio_clip_end = bio_end_time + margin

        bio_df = bio_df[
            (bio_df['TIME'] >= bio_clip_start) &
            (bio_df['TIME'] <= bio_clip_end)
        ].reset_index(drop=True)

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

        # Clip audio to ensure all kept rows will have matching biomechanics
        audio_df = audio_df[
            (audio_df['tt'] >= audio_match_start) &
            (audio_df['tt'] <= audio_match_end)
        ].reset_index(drop=True)
    # Adjust audio timestamps by the time difference (after clipping)
    audio_df['tt'] = (
        audio_df['tt'] + pd.to_timedelta(time_difference, unit='s')
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

    return synchronized_df


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
