from typing import Literal, Optional

import pandas as pd

# Module that processes a DataFrame containing synchronized audio and
# biomechanics data to extract a list of movement cycles (flexion-extension,
# sit-to-stand, or walking cycles) based on knee angle data.

# The syncd_df has redundant biomechanics rows because the sampling rate for
# biomechanics motion capture is 120 or 60 Hz while the audio is sampled at 52
# kHz. Thus, multiple audio rows correspond to a single biomechanics row.
# The function accounts for this by using the biomechanics_sampling_rate
# parameter to determine how many audio rows# correspond to each biomechanics
# row.


def extract_fe_sts_cycles(
    syncd_df: pd.DataFrame,
    knee: Literal["left", "right", "Left", "Right"],
    angle_col_suffix: Literal["Z"] | None = "Z",
    biomechanics_sampling_rate: int = 120,
    cycle_threshold: int = 30,
    deflection_threshold: int = 15,
) -> list[pd.DataFrame]:
    """Method that takes a an audio DataFrame synchronized with biomechanics
    and extracts flexion-extension or sit-to-stand cycles based on the
    flexion-extension angle of the knee (column name = Left Knee Angle Z).

    Cycles include 500 ms before and after the movement. If there is not
    enough data before or after the movement, whatever is available is used.

    Args:
        syncd_df (pd.DataFrame): DataFrame containing synchronized audio
            and biomechanics data.

    Returns:
        list[pd.DataFrame]: List of DataFrames, each containing
            data for one flexion-extension or sit-to-stand cycle.
    """

    if knee.lower() == "left":
        angle_col = "Left Knee Angle"
    elif knee.lower() == "right":
        angle_col = "Right Knee Angle"
    else:
        raise ValueError("knee must be 'left' or 'right'")

    if angle_col_suffix is not None:
        angle_col += f" {angle_col_suffix}"

    fe_cycles = []
    cycle_start_idx = get_initial_fe_sts_index(
        syncd_df,
        angle_col,
        biomechanics_sampling_rate,
        cycle_threshold=cycle_threshold,
        deflection_threshold=deflection_threshold,
    )
    start_angle = syncd_df.loc[cycle_start_idx, angle_col]

    i = cycle_start_idx
    while i < len(syncd_df):
        window = syncd_df[i: i + biomechanics_sampling_rate * 2]


def get_initial_fe_sts_index(
    syncd_df: pd.DataFrame,
    angle_col: str,
    biomechanics_sampling_rate: int,
    cycle_threshold: int = 30,
    deflection_threshold: int = 15,
) -> int:
    """Helper function to find the initial index of the first flexion-extension
    or sit-to-stand cycle in the synchronized DataFrame. Looks for the first instance where
    the joint angle changes more than 30 degrees over a 2-second window and returns the index
    where the deflection starts.

    Args:
        syncd_df (pd.DataFrame): DataFrame containing synchronized audio
            and biomechanics data.
        angle_col (str): Name of the column containing knee angle data.
        biomechanics_sampling_rate (int): Sampling rate of the biomechanics data.

    Returns:
        int: Index of the start of the first flexion-extension or sit-to-stand cycle

    Raises:
        KeyError: if the specified angle_col does not exist in syncd_df.
        RuntimeError: if a 30-degree deflection is not found in the data.
    """
    window_size = biomechanics_sampling_rate * 2  # 2 seconds

    try:
        angle_data = syncd_df[angle_col].reset_index(drop=True)
    except KeyError as e:
        raise KeyError(f"Column '{angle_col}' not found in DataFrame.") from e

    for i in range(len(angle_data) - window_size):
        window = angle_data[i: i + window_size]
        angle_change = window.max() - window.min()
        if angle_change > 30:  # 15 degree / second threshold
            max_index: int = window.idxmax()
            deflection_index = get_initial_deflection_index(
                window,
                max_index,
                deflection_threshold=deflection_threshold,
                sampling_rate=biomechanics_sampling_rate,
            )
            if deflection_index is not None:
                return deflection_index

    raise RuntimeError(
        "No flexion-extension or sit-to-stand cycle found in the data."
    )


def get_initial_deflection_index(
    window: pd.Series,
    max_index: int,
    deflection_threshold: int = 5,
    sampling_rate: int = 120,
) -> Optional[int]:
    """Helper function to find the index where the knee angle starts to
    deflect significantly within a given window of angle data. Does so by
    looking backwards from the maximum angle index to find where the angle
    starts to deflect towards the max (earliest index in the window where
    the angle change exceeds the specified threshold in degrees / second).

    Args:
        window (pd.Series): Series containing knee angle data
            for a specific time window.
        max_index (int): Index of the maximum angle within the window.
        deflection_threshold (int): Minimum angle change to consider as significant
            deflection.
        sampling_rate (int): Sampling rate of the biomechanics data.

    Returns:
        Optional[int]: Index of the start of the deflection within the window.

    Raises:
        RuntimeError: if no significant deflection is found in the window.
    """
    for j in range(max_index, 0, -1):
        angle_change = abs(window[j] - window[j - 1])
        if angle_change >= (deflection_threshold / sampling_rate):
            return j

    raise RuntimeError(
        "No significant deflection found in the provided window."
    )
