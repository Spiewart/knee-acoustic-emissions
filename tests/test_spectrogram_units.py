import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.audio.spectrogram import compute_spectrogram_arrays, get_fs_from_df_or_meta


def test_compute_spectrogram_arrays_no_channels():
    df = pd.DataFrame({"foo": [1, 2, 3]})
    f, t, specs = compute_spectrogram_arrays(df, fs=1000.0, nperseg=64, noverlap=32)
    assert f.size == 0
    assert t.size == 0
    assert specs == {}


def test_get_fs_from_df_or_meta_from_tt():
    # tt increments by 0.001 -> fs = 1000 Hz
    tt = np.arange(0.0, 1.0, 0.001)
    df = pd.DataFrame({"tt": tt, "ch1": np.sin(2 * np.pi * 10 * tt)})
    fs = get_fs_from_df_or_meta(df)
    assert np.isclose(fs, 1000.0)


def test_get_fs_from_df_or_meta_from_meta(tmp_path):
    # No tt; get fs from meta json
    df = pd.DataFrame({"ch1": np.arange(100)})
    meta = {"fs": 2000.0}
    meta_json = tmp_path / "a_meta.json"
    meta_json.write_text(json.dumps(meta), encoding="utf-8")
    fs = get_fs_from_df_or_meta(df, meta_json)
    assert np.isclose(fs, 2000.0)
