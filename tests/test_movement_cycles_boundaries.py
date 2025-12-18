"""Boundary tests for movement cycle extraction.

Covers variable sampling rates, minimal data lengths, and missing columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from parse_movement_cycles import MovementCycleExtractor, extract_movement_cycles


def _make_walk_df(duration_s: float, fs: int) -> pd.DataFrame:
    n = int(duration_s * fs)
    tt = np.arange(n) / fs
    # Heel strikes as minima at integer seconds using -cos
    knee = 20 - 30 * np.cos(2 * np.pi * 1.0 * tt) + np.random.randn(n) * 0.2
    noise = np.random.randn(n) * 0.001
    df = pd.DataFrame(
        {
            "tt": tt,
            "ch1": noise,
            "ch2": noise,
            "ch3": noise,
            "ch4": noise,
            "f_ch1": noise,
            "f_ch2": noise,
            "f_ch3": noise,
            "f_ch4": noise,
            "TIME": pd.to_timedelta(tt, unit="s"),
            "Knee Angle Z": knee,
        }
    )
    return df


def _make_fe_df(duration_s: float, fs: int, freq_hz: float = 0.5) -> pd.DataFrame:
    n = int(duration_s * fs)
    tt = np.linspace(0, duration_s, n)
    knee = (
        50
        + 40 * np.sin(2 * np.pi * freq_hz * tt - np.pi / 2)
        + np.random.randn(n) * 1.0
    )
    noise = np.random.randn(n) * 0.001
    return pd.DataFrame(
        {
            "tt": tt,
            "ch1": noise,
            "ch2": noise,
            "ch3": noise,
            "ch4": noise,
            "f_ch1": noise,
            "f_ch2": noise,
            "f_ch3": noise,
            "f_ch4": noise,
            "TIME": pd.to_timedelta(tt, unit="s"),
            "Knee Angle Z": knee,
        }
    )


def _make_sts_single_peak_df(duration_s: float, fs: int) -> pd.DataFrame:
    n = int(duration_s * fs)
    tt = np.linspace(0, duration_s, n)
    knee = np.full(n, 60.0)
    # Single standing phase (minimum knee angle) near mid
    mid = n // 2
    width = int(0.5 * fs)
    if width > 0:
        knee[mid : mid + width] = 15.0
    noise = np.random.randn(n) * 0.001
    return pd.DataFrame(
        {
            "tt": tt,
            "ch1": noise,
            "ch2": noise,
            "ch3": noise,
            "ch4": noise,
            "f_ch1": noise,
            "f_ch2": noise,
            "f_ch3": noise,
            "f_ch4": noise,
            "TIME": pd.to_timedelta(tt, unit="s"),
            "Knee Angle Z": knee,
        }
    )


class TestVariableSamplingRates:
    def test_walking_cycles_detected_500hz(self):
        df = _make_walk_df(duration_s=2.5, fs=500)
        cycles = extract_movement_cycles(df, "walk")
        # Expect at least one complete cycle in 2.5s window
        assert len(cycles) >= 1

    def test_walking_cycles_detected_2000hz(self):
        df = _make_walk_df(duration_s=2.5, fs=2000)
        cycles = extract_movement_cycles(df, "walk")
        assert len(cycles) >= 1

    def test_fe_cycle_durations_near_2s(self):
        df = _make_fe_df(duration_s=8.0, fs=1000, freq_hz=0.5)
        cycles = extract_movement_cycles(df, "flexion_extension")
        # With 8s of data at 0.5 Hz (2s cycles), we get 4 complete periods
        # But extracting complete cycles (extension→flexion→extension) requires
        # N+1 extension peaks to find N cycles, so expect 2-3 cycles
        assert len(cycles) >= 2
        for c in cycles:
            dur = float(c["tt"].max() - c["tt"].min())
            assert 1.6 <= dur <= 2.4


class TestMinimalDataLengths:
    def test_two_rows_returns_empty(self):
        df = pd.DataFrame(
            {
                "tt": [0.0, 0.001],
                "ch1": [0, 0],
                "ch2": [0, 0],
                "ch3": [0, 0],
                "ch4": [0, 0],
                "f_ch1": [0, 0],
                "f_ch2": [0, 0],
                "f_ch3": [0, 0],
                "f_ch4": [0, 0],
                "TIME": pd.to_timedelta([0.0, 0.001], unit="s"),
                "Knee Angle Z": [10.0, 10.1],
            }
        )
        extractor = MovementCycleExtractor("walk")
        assert extractor.extract_cycles(df) == []


class TestMissingColumns:
    def test_missing_knee_angle_raises(self):
        df = pd.DataFrame(
            {
                "tt": [0.0, 0.001, 0.002],
                "ch1": [0, 0, 0],
                "ch2": [0, 0, 0],
                "ch3": [0, 0, 0],
                "ch4": [0, 0, 0],
                "f_ch1": [0, 0, 0],
                "f_ch2": [0, 0, 0],
                "f_ch3": [0, 0, 0],
                "f_ch4": [0, 0, 0],
                "TIME": pd.to_timedelta([0.0, 0.001, 0.002], unit="s"),
            }
        )
        extractor = MovementCycleExtractor("walk")
        try:
            extractor.extract_cycles(df)
            assert False, "Expected ValueError for missing Knee Angle Z"
        except ValueError as e:
            assert "Knee Angle Z" in str(e)


class TestSitToStandSingleStandingPhase:
    def test_single_standing_peak_returns_whole_df(self):
        df = _make_sts_single_peak_df(duration_s=5.0, fs=1000)
        cycles = extract_movement_cycles(df, "sit_to_stand")
        assert len(cycles) == 1
        # It should be the whole DataFrame when only a single standing phase exists
        assert len(cycles[0]) == len(df)
