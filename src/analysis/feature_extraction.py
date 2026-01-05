"""Feature extraction from movement cycle data."""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def extract_channel_features(channel_data: np.ndarray) -> Dict[str, float]:
    """Extract statistical features from a single audio channel.

    Args:
        channel_data: 1D array of audio samples

    Returns:
        Dictionary of feature name -> value
    """
    features = {
        "mean": float(np.mean(channel_data)),
        "std": float(np.std(channel_data)),
        "rms": float(np.sqrt(np.mean(channel_data ** 2))),
        "peak": float(np.max(np.abs(channel_data))),
        "peak_to_peak": float(np.ptp(channel_data)),
        "energy": float(np.sum(channel_data ** 2)),
        "kurtosis": float(pd.Series(channel_data).kurtosis()),
        "skewness": float(pd.Series(channel_data).skew()),
    }

    # Spectral features (simple)
    fft = np.fft.rfft(channel_data)
    magnitude = np.abs(fft)

    features["spectral_centroid"] = float(
        np.sum(np.arange(len(magnitude)) * magnitude) / (np.sum(magnitude) + 1e-10)
    )
    features["spectral_spread"] = float(np.std(magnitude))
    features["spectral_energy"] = float(np.sum(magnitude ** 2))

    return features


def extract_cycle_features(
    cycle_df: pd.DataFrame,
    channels: List[str] = None,
) -> Dict[str, float]:
    """Extract features from a movement cycle DataFrame.

    Args:
        cycle_df: DataFrame with audio channels (ch1-4, f_ch1-4) and biomechanics
        channels: List of channel names to extract, or None for default
            (default: ["f_ch1", "f_ch2", "f_ch3", "f_ch4"])

    Returns:
        Dictionary of feature name -> value
    """
    if channels is None:
        # Use filtered channels by default
        channels = ["f_ch1", "f_ch2", "f_ch3", "f_ch4"]

    features = {}

    # Extract features from each audio channel
    for ch in channels:
        if ch not in cycle_df.columns:
            logger.warning(f"Channel {ch} not found in cycle DataFrame")
            continue

        channel_data = cycle_df[ch].values
        ch_features = extract_channel_features(channel_data)

        # Prefix with channel name
        for feat_name, feat_val in ch_features.items():
            features[f"{ch}_{feat_name}"] = feat_val

    # Cross-channel features
    if all(ch in cycle_df.columns for ch in channels):
        # Average across channels
        all_channels = np.column_stack([cycle_df[ch].values for ch in channels])
        avg_signal = np.mean(all_channels, axis=1)

        avg_features = extract_channel_features(avg_signal)
        for feat_name, feat_val in avg_features.items():
            features[f"avg_{feat_name}"] = feat_val

        # Channel correlations
        for i, ch1 in enumerate(channels):
            for ch2 in channels[i+1:]:
                corr = np.corrcoef(cycle_df[ch1], cycle_df[ch2])[0, 1]
                features[f"corr_{ch1}_{ch2}"] = float(corr)

    # Biomechanics features if available
    if "Knee Angle Z" in cycle_df.columns:
        knee_angle = cycle_df["Knee Angle Z"].values
        features["knee_angle_mean"] = float(np.mean(knee_angle))
        features["knee_angle_std"] = float(np.std(knee_angle))
        features["knee_angle_min"] = float(np.min(knee_angle))
        features["knee_angle_max"] = float(np.max(knee_angle))
        features["knee_angle_range"] = float(np.ptp(knee_angle))

    # Temporal features
    if "tt" in cycle_df.columns:
        tt = cycle_df["tt"].values
        # Handle both timedelta and numpy.timedelta64
        time_delta = tt[-1] - tt[0]
        if isinstance(time_delta, pd.Timedelta):
            duration = time_delta.total_seconds()
        elif isinstance(time_delta, np.timedelta64):
            # Convert to seconds
            duration = float(time_delta / np.timedelta64(1, 's'))
        elif hasattr(time_delta, "total_seconds"):
            duration = time_delta.total_seconds()
        else:
            duration = float(time_delta)

        features["duration"] = duration
        features["num_samples"] = len(cycle_df)
        if duration > 0:
            features["sample_rate"] = len(cycle_df) / duration

    return features


def extract_features_from_cycles(cycles_df: pd.DataFrame) -> pd.DataFrame:
    """Extract features from all cycles in a DataFrame.

    Args:
        cycles_df: DataFrame from load_all_participant_cycles with cycle_df column

    Returns:
        DataFrame with extracted features, one row per cycle
    """
    feature_rows = []

    for idx, row in cycles_df.iterrows():
        try:
            features = extract_cycle_features(row["cycle_df"])

            # Add metadata
            features["study_id"] = row["study_id"]
            features["knee"] = row["knee"]
            features["maneuver"] = row["maneuver"]
            features["speed"] = row["speed"]
            features["cycle_index"] = row["cycle_index"]
            features["cycle_acoustic_energy"] = row["cycle_acoustic_energy"]

            feature_rows.append(features)
        except Exception as e:
            logger.warning(f"Failed to extract features for cycle {idx}: {e}")

    features_df = pd.DataFrame(feature_rows)
    logger.info(f"Extracted {len(features_df.columns)} features from {len(features_df)} cycles")

    return features_df
