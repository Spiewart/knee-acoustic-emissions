"""Module for extracting movement cycles from synchronized audio and biomechanics data.

Movement cycles are segmented based on biomechanics joint angle (Z-dimension) patterns
that differ by maneuver type (walking, sit-to-stand, flexion-extension).
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd


class MovementCycleExtractor:
    """Extract movement cycles from synchronized DataFrame based on joint angle patterns."""

    def __init__(
        self,
        maneuver: Literal["walk", "sit_to_stand", "flexion_extension"],
        speed: Optional[Literal["slow", "medium", "fast"]] = None,
    ):
        """Initialize extractor for a specific maneuver type.

        Args:
            maneuver: Type of movement (walk, sit_to_stand, flexion_extension)
            speed: Speed level for walking maneuvers (slow, medium, fast)
        """
        self.maneuver = maneuver
        self.speed = speed

    def extract_cycles(self, synced_df: pd.DataFrame) -> list[pd.DataFrame]:
        """Extract movement cycles from synchronized data.

        Identifies cycle boundaries (heel strikes, standing phases, or extension extrema)
        based on maneuver type, then segments the DataFrame into individual cycles.

        Cycles are defined by:
        - **Walk**: Successive heel strikes (local minima in knee angle).
        - **Sit-to-stand**: Transitions between sitting and standing (standing minima).
        - **Flexion-extension**: Cycles between flexion/extension extrema.

        Args:
            synced_df: DataFrame with synchronized audio and biomechanics data.
                      Must have 'Knee Angle Z' column for joint angle data.

        Returns:
            List of DataFrames, each containing one complete movement cycle.
            Empty list if fewer than 2 cycle boundaries detected.

        Raises:
            ValueError: If required 'Knee Angle Z' column is missing or maneuver type is invalid.
        """
        if synced_df.empty or len(synced_df) < 2:
            return []

        if "Knee Angle Z" not in synced_df.columns:
            raise ValueError("DataFrame must contain 'Knee Angle Z' column")

        if self.maneuver == "walk":
            return self._extract_walking_cycles(synced_df)
        elif self.maneuver == "sit_to_stand":
            return self._extract_sit_to_stand_cycles(synced_df)
        elif self.maneuver == "flexion_extension":
            return self._extract_flexion_extension_cycles(synced_df)
        else:
            raise ValueError(f"Unsupported maneuver: {self.maneuver}")

    def _extract_walking_cycles(self, synced_df: pd.DataFrame) -> list[pd.DataFrame]:
        """Extract gait cycles from walking data.

        A gait cycle is defined from one heel strike to the next heel strike
        of the same foot. Uses local minima in knee angle (Z-component) as
        a proxy for heel strike events. Requires that start and end heel
        strikes have similar knee extension angles to be considered valid.

        Applies Savitzky-Golay smoothing to reduce noise and improve peak detection.

        Args:
            synced_df: Synchronized walking data with 'Knee Angle Z' column.

        Returns:
            List of cycle DataFrames. Empty if fewer than 2 heel strikes detected
            or if cycles fail angle tolerance checks.
        """
        from scipy.signal import find_peaks, savgol_filter

        knee_angle = synced_df["Knee Angle Z"].values

        # Smooth the signal to reduce noise impact on peak detection
        # Use Savitzky-Golay filter with window of 51 samples (~0.05s at 1000Hz)
        if len(knee_angle) > 51:
            knee_angle_smooth = savgol_filter(knee_angle, window_length=51, polyorder=3)
        else:
            knee_angle_smooth = knee_angle

        # Find local minima (heel strike events) by inverting signal and finding peaks
        # Use distance parameter to ensure peaks are at least 0.8 seconds apart
        dt = synced_df["tt"].iloc[1] - synced_df["tt"].iloc[0]
        dt_seconds = dt.total_seconds() if hasattr(dt, 'total_seconds') else float(dt)
        sample_rate = 1 / dt_seconds
        min_peak_distance = int(
            0.8 * sample_rate
        )  # 0.8 seconds minimum between heel strikes

        minima_indices, _ = find_peaks(
            -knee_angle_smooth, distance=min_peak_distance, prominence=5
        )

        if len(minima_indices) < 2:
            return []

        # Validate that start/end heel strikes have similar extension angles
        angle_tolerance = 7.5  # degrees
        cycles = []
        for i in range(len(minima_indices) - 1):
            start_idx = minima_indices[i]
            end_idx = minima_indices[i + 1]

            start_angle = knee_angle_smooth[start_idx]
            end_angle = knee_angle_smooth[end_idx]

            if abs(end_angle - start_angle) <= angle_tolerance:
                # Include the endpoint only for the last cycle to avoid off-by-one
                # under-coverage without duplicating shared boundaries across cycles.
                end_slice = end_idx + 1 if i == (len(minima_indices) - 2) else end_idx
                cycle_df = synced_df.iloc[start_idx:end_slice].reset_index(
                    drop=True
                )
                if len(cycle_df) > 10:  # Minimum cycle length
                    cycles.append(cycle_df)
        return cycles

    def _extract_sit_to_stand_cycles(
        self, synced_df: pd.DataFrame
    ) -> list[pd.DataFrame]:
        """Extract sit-to-stand cycles.

        For sit-to-stand maneuvers, cycles are segmented by standing phases
        (low knee angle minima). Each standing phase marks a transition point.
        If no clear standing phases are found, returns the entire recording
        as a single cycle.

        Args:
            synced_df: Synchronized sit-to-stand data with 'Knee Angle Z' column.

        Returns:
            List of cycle DataFrames. At minimum returns entire recording as one cycle.
        """
        from scipy.signal import find_peaks

        knee_angle = synced_df["Knee Angle Z"].values

        # Find minima (standing positions - low knee angle)
        # Standing phases should be at least 2 seconds apart
        dt = synced_df["tt"].iloc[1] - synced_df["tt"].iloc[0]
        dt_seconds = dt.total_seconds() if hasattr(dt, 'total_seconds') else float(dt)
        sample_rate = 1 / dt_seconds
        min_peak_distance = int(2.0 * sample_rate)

        standing_indices, _ = find_peaks(
            -knee_angle, distance=min_peak_distance, prominence=20
        )

        if len(standing_indices) < 1:
            # No clear cycles, return entire dataframe as single cycle
            return [synced_df.reset_index(drop=True)]

        cycles = []

        # Segment between consecutive standing phases
        if len(standing_indices) == 1:
            cycles.append(synced_df.reset_index(drop=True))
        else:
            for i in range(len(standing_indices) - 1):
                start_idx = standing_indices[i]
                end_idx = standing_indices[i + 1]
                cycle_df = synced_df.iloc[start_idx:end_idx].reset_index(drop=True)
                if len(cycle_df) > 10:
                    cycles.append(cycle_df)

        return cycles

    def _extract_flexion_extension_cycles(
        self, synced_df: pd.DataFrame
    ) -> list[pd.DataFrame]:
        """Extract flexion-extension cycles.

        A complete cycle consists of extension (minimum angle) → flexion (maximum angle) → extension.
        Each cycle is bounded by two consecutive extension peaks.

        Returns:
            List of cycle DataFrames.
        """
        from scipy.signal import find_peaks

        knee_angle = synced_df["Knee Angle Z"].values

        # Find extension peaks (minima in knee angle) - these mark cycle boundaries
        # Minimum distance is 90% of expected cycle time to avoid double-counting while
        # still detecting slightly faster cycles (e.g., 1.8s for 2s cycles, 4.5s for 5s cycles)
        dt = synced_df["tt"].iloc[1] - synced_df["tt"].iloc[0]
        dt_seconds = dt.total_seconds() if hasattr(dt, 'total_seconds') else float(dt)
        sample_rate = 1 / dt_seconds
        min_peak_distance = int(1.8 * sample_rate)

        extension_indices, _ = find_peaks(-knee_angle, distance=min_peak_distance, prominence=10)

        if len(extension_indices) < 2:
            return []

        # Now find flexion peaks (maxima) to validate complete cycles
        flexion_indices, _ = find_peaks(knee_angle, prominence=10)

        cycles = []
        for i in range(len(extension_indices) - 1):
            start_idx = extension_indices[i]
            end_idx = extension_indices[i + 1]

            # Verify there's at least one flexion peak between these two extensions
            flexions_in_cycle = [idx for idx in flexion_indices if start_idx < idx < end_idx]

            if flexions_in_cycle and len(synced_df.iloc[start_idx:end_idx]) > 10:
                cycle_df = synced_df.iloc[start_idx:end_idx].reset_index(drop=True)
                cycles.append(cycle_df)

        return cycles

    @staticmethod
    def _find_local_minima(signal: np.ndarray, min_distance: float = 0.1) -> list[int]:
        """Find local minima in a signal.

        Args:
            signal: Input signal array.
            min_distance: Minimum distance between peaks (as fraction of signal length).

        Returns:
            List of indices where local minima occur.
        """
        min_samples = max(2, int(len(signal) * min_distance))

        # Simple minima detection: compare each point with neighbors
        minima = []
        for i in range(min_samples, len(signal) - min_samples):
            window = signal[i - min_samples : i + min_samples]
            if signal[i] == np.min(window):
                minima.append(i)

        # Remove consecutive duplicates
        if minima:
            minima = [minima[0]] + [
                m for i, m in enumerate(minima[1:], 1) if m != minima[i - 1]
            ]

        return minima

    @staticmethod
    def _find_local_maxima(signal: np.ndarray, min_distance: float = 0.1) -> list[int]:
        """Find local maxima in a signal.

        Args:
            signal: Input signal array.
            min_distance: Minimum distance between peaks (as fraction of signal length).

        Returns:
            List of indices where local maxima occur.
        """
        min_samples = max(2, int(len(signal) * min_distance))

        # Simple maxima detection: compare each point with neighbors
        maxima = []
        for i in range(min_samples, len(signal) - min_samples):
            window = signal[i - min_samples : i + min_samples]
            if signal[i] == np.max(window):
                maxima.append(i)

        # Remove consecutive duplicates
        if maxima:
            maxima = [maxima[0]] + [
                m for i, m in enumerate(maxima[1:], 1) if m != maxima[i - 1]
            ]

        return maxima


def extract_movement_cycles(
    synced_df: pd.DataFrame,
    maneuver: Literal["walk", "sit_to_stand", "flexion_extension"],
    speed: Optional[Literal["slow", "medium", "fast"]] = None,
) -> list[pd.DataFrame]:
    """Convenience function to extract movement cycles from synchronized data.

    Args:
        synced_df: Synchronized audio and biomechanics DataFrame.
        maneuver: Type of movement.
        speed: Speed level (for walking only).

    Returns:
        List of DataFrames, each containing one movement cycle.
    """
    extractor = MovementCycleExtractor(maneuver=maneuver, speed=speed)
    return extractor.extract_cycles(synced_df)
