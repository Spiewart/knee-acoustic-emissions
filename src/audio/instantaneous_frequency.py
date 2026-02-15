"""Add instantaneous frequency to a DataFrame."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4) -> tuple[np.ndarray, np.ndarray]:
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def apply_bandpass(
    y: np.ndarray, fs: float, lowcut: float = 10.0, highcut: float = 5000.0, order: int = 4
) -> np.ndarray:
    try:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y_filt = filtfilt(b, a, y)
        return y_filt
    except (ValueError, TypeError) as e:
        logging.exception("Bandpass filtering failed; returning raw signal: %s", e)
        return y


def add_instantaneous_frequency(
    df: pd.DataFrame,
    fs: float,
    lowcut: float = 10.0,
    highcut: float = 5000.0,
    order: int = 4,
    channels: list[str] | None = None,
) -> pd.DataFrame:
    """Add instantaneous frequency to a DataFrame using a Hilbert transform.

    Args:
        df (pd.DataFrame): DataFrame with audio channels.
        fs (float): Sampling frequency.
        lowcut (float, optional): Lowcut frequency for bandpass filter.
        highcut (float, optional): Highcut frequency for bandpass filter.
        order (int, optional): Butterworth filter order.
        channels (List[str], optional): List of channels to process.
                                        If None, uses all `ch*` columns.

    Returns:
        pd.DataFrame: DataFrame with added `f_ch*` columns.
    """
    if channels is None:
        channels = [col for col in df.columns if col.startswith("ch")]

    dt = 1.0 / fs

    for ch in channels:
        col_freq = f"f_{ch}"
        if ch not in df.columns or not df[ch].notna().any():
            logging.info("%s not present or empty; filling %s with NaN", ch, col_freq)
            df[col_freq] = np.nan
            continue

        y = pd.to_numeric(df[ch], errors="coerce").to_numpy()
        y_centered = y - np.nanmean(y)
        y_filtered = apply_bandpass(y_centered, fs, lowcut=lowcut, highcut=highcut, order=order)

        try:
            analytic = hilbert(y_filtered)
        except (ValueError, TypeError) as e:
            logging.exception("Hilbert transform failed for %s; filling with NaN: %s", ch, e)
            df[col_freq] = np.nan
            continue

        phase = np.unwrap(np.angle(analytic))
        dphase_dt = np.gradient(phase, dt)
        inst_freq = dphase_dt / (2.0 * np.pi)
        df[col_freq] = inst_freq

        # Suppress verbose frequency statistics logging
        # finite = inst_freq[np.isfinite(inst_freq)]
        # if finite.size > 0:
        #     logging.debug(
        #         "%s: min=%.3f Hz, max=%.3f Hz, mean=%.3f Hz",
        #         col_freq,
        #         finite.min(),
        #         finite.max(),
        #         finite.mean(),
        #     )

    return df
