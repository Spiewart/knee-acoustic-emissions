import numpy as np
import pandas as pd

from src.audio.instantaneous_frequency import (
    add_instantaneous_frequency,
    apply_bandpass,
)


def test_apply_bandpass_returns_raw_on_invalid_params():
    # fs too low so highcut > Nyquist -> filter should fail and return raw
    fs = 100.0
    y = np.sin(2 * np.pi * 1.0 * np.arange(100) / fs)
    y_filt = apply_bandpass(y, fs=fs, lowcut=10.0, highcut=5000.0, order=4)
    assert np.allclose(y_filt, y)


def test_add_instantaneous_frequency_handles_missing_channel():
    # ch1 present but all NaN -> f_ch1 created and filled with NaN
    df = pd.DataFrame({"ch1": [np.nan] * 128})
    out = add_instantaneous_frequency(df.copy(), fs=1000.0, channels=["ch1"])
    assert "f_ch1" in out.columns
    assert len(out["f_ch1"]) == len(df)
    assert np.isnan(out["f_ch1"]).all()
