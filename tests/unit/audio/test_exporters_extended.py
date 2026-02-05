"""Extended tests for audio exporters: multi-channel and edge cases."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.audio.exporters import dump_channels_to_csv


def test_dump_channels_to_csv_multichannel_with_tt():
    """Test multi-channel export with tt column present."""
    N = 50
    tt = np.linspace(0.0, 1.0, N)
    df = pd.DataFrame({
        "tt": tt,
        "ch1": np.sin(2 * np.pi * 1.0 * tt),
        "ch2": np.cos(2 * np.pi * 2.0 * tt),
        "ch3": np.sin(2 * np.pi * 0.5 * tt),
    })

    csv_path = Path("/tmp/test_multi_out.csv")
    dump_channels_to_csv(df, csv_path)

    out_df = pd.read_csv(csv_path)
    assert out_df.shape == (N, 5)  # tt + ch1..ch4
    assert out_df["tt"].shape == (N,)
    # ch1, ch2, ch3 should have data; ch4 should be NaN
    assert out_df["ch1"].notna().all()
    assert out_df["ch2"].notna().all()
    assert out_df["ch3"].notna().all()
    assert out_df["ch4"].isna().all()


def test_dump_channels_to_csv_meta_invalid_json(tmp_path):
    """Test fallback to indices when meta JSON is malformed."""
    N = 30
    df = pd.DataFrame({"ch1": np.arange(N)})
    csv_out = tmp_path / "out_bad_meta.csv"

    # Create malformed JSON file
    meta_json = tmp_path / "file_meta.json"
    meta_json.write_text("{invalid json", encoding="utf-8")

    dump_channels_to_csv(df, csv_out, meta_json)

    out_df = pd.read_csv(csv_out)
    # Should fallback to indices
    assert np.allclose(out_df["tt"].to_numpy(), np.arange(N))


def test_dump_channels_to_csv_meta_missing_fs(tmp_path):
    """Test valid meta JSON but missing startTime/stopTime or fs."""
    N = 25
    df = pd.DataFrame({"ch2": np.random.randn(N)})
    csv_out = tmp_path / "out_incomplete_meta.csv"

    meta = {"some_other_field": 123}  # No startTime/stopTime
    meta_json = tmp_path / "file_meta.json"
    meta_json.write_text(json.dumps(meta), encoding="utf-8")

    dump_channels_to_csv(df, csv_out, meta_json)

    out_df = pd.read_csv(csv_out)
    # Should fallback to indices
    assert np.allclose(out_df["tt"].to_numpy(), np.arange(N))
