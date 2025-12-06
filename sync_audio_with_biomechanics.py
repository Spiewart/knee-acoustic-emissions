"""Module for synchronizing audio data (pickled DataFrame) with biomechanics data.
Uses foot stomp events (Sync Left and Sync Right) to align the two datasets.
Stomp events are identified by the first peak in the audio channels exceeding
a threshold defined by the overall signal statistics."""

from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

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
) -> datetime:
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
) -> datetime:
    """Extract the timestamp of the right foot stomp event from biomechanics metadata.
    Sync Right is a item in the first row Event Info. The second column Time (sec)
    will contain the timestamp."""

    return get_stomp_time(bio_meta, foot="right")


def get_left_stomp_time(
    bio_meta: "pd.DataFrame",
) -> datetime:
    """Extract the timestamp of the left foot stomp event from biomechanics metadata."""

    return get_stomp_time(bio_meta, foot="left")


# Modules for loading pickled audio data and labeling as right or left via audio metadata
def load_audio_data(
    audio_file: "Path",
) -> "pd.DataFrame":
    """Load the pickled audio data from the specified file."""

    return pd.read_pickle(audio_file)


def get_audio_stomp_time(
    audio_df: "pd.DataFrame",
) -> datetime:
    """Get the timestamp (column = tt) of the foot stomp event from the
    audio data DataFrame by finding the first peak in the audio channels
    exceeding a threshold."""

    # Define threshold for stomp detection
    threshold = (
        audio_df.loc[:, audio_df.columns.isin(["ch1", "ch2", "ch3", "ch4"])].values.mean() + (
                3 * audio_df.loc[:, audio_df.columns.isin(["ch1", "ch2", "ch3", "ch4"])].values.std()
        )
    )

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
    audio_stomp_time: datetime,
    bio_stomp_time: datetime,
    audio_df: "pd.DataFrame",
    bio_df: "pd.DataFrame",
) -> "pd.DataFrame":
    """Synchronize audio DataFrame with biomechanics DataFrame using stomp times.

    Args:
        audio_stomp_time (datetime): Timestamp of the foot stomp event in the audio data.
        bio_stomp_time (datetime): Timestamp of the foot stomp event in the biomechanics data.
        audio_df (pd.DataFrame): DataFrame containing audio data with timestamps.
        bio_df (pd.DataFrame): DataFrame containing biomechanics data with timestamps.

    Returns:
        pd.DataFrame: Synchronized audio DataFrame with adjusted timestamps and
                      aligned biomechanics data.
    """

    # Calculate time difference between biomechanics and audio stomp times
    time_difference = bio_stomp_time - audio_stomp_time
    # Convert the audio_df 'tt' column to timedelta
    audio_df['tt'] = pd.to_timedelta(audio_df['tt'], unit='s')

    # Adjust audio timestamps by the time difference
    audio_df['tt'] = audio_df['tt'] + pd.to_timedelta(time_difference, unit='s')
    # Drop any trailing NaT values in the biomechanics TIME column
    bio_df = bio_df.dropna(subset=['TIME']).reset_index(drop=True)

    # Merge biomechanics into audio to preserve all audio data
    # using adjusted 'tt' and biomechanics 'TIME' columns
    synchronized_df = pd.merge_asof(
        audio_df.sort_values('tt'),
        bio_df.sort_values('TIME'),
        right_on='TIME',
        left_on='tt',
        direction='nearest',
        tolerance=pd.Timedelta(seconds=0.00001)  # Adjust tolerance as needed
    )

    return synchronized_df


def clip_synchronized_data(
    synchronized_df: "pd.DataFrame",
    bio_df: "pd.DataFrame",
) -> "pd.DataFrame":
    """Clip synchronized DataFrame to the exact biomechanics data time range.

    Args:
        synchronized_df (pd.DataFrame): DataFrame containing synchronized audio and biomechanics data.
        bio_df (pd.DataFrame): DataFrame containing biomechanics data with timestamps.

    Returns:
        pd.DataFrame: Clipped synchronized DataFrame within biomechanics time range.
    """

    # Get biomechanics start and end times
    bio_start_time = bio_df['TIME'].min()
    bio_end_time = bio_df['TIME'].max()

    # Clip synchronized DataFrame to biomechanics time range
    clipped_df = synchronized_df[
        (synchronized_df['tt'] >= bio_start_time) &
        (synchronized_df['tt'] <= bio_end_time)
    ].reset_index(drop=True)

    return clipped_df