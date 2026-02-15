import json

import numpy as np
import pandas as pd

from src.audio.exporters import dump_channels_to_csv


def test_dump_channels_to_csv_meta_linspace(tmp_path):
    # Create DataFrame without tt, with ch1 of length N
    N = 100
    df = pd.DataFrame({"ch1": np.linspace(0.0, 1.0, N)})
    csv_out = tmp_path / "out.csv"
    meta = {"startTime": 2.0, "stopTime": 5.0}
    meta_json = tmp_path / "file_meta.json"
    meta_json.write_text(json.dumps(meta), encoding="utf-8")

    dump_channels_to_csv(df, csv_out, meta_json)

    out_df = pd.read_csv(csv_out)
    assert out_df.shape == (N, 5)  # tt + ch1..ch4
    # tt should be a linspace between start/stop
    assert np.isclose(out_df["tt"].iloc[0], 2.0)
    assert np.isclose(out_df["tt"].iloc[-1], 5.0)
    # account for CSV rounding by verifying the timesteps are nearly constant
    expected_step = (5.0 - 2.0) / (N - 1)
    steps = np.diff(out_df["tt"])
    # Steps should be nearly constant, with small variation due to rounding
    assert np.allclose(steps, expected_step, rtol=1e-4)


def test_dump_channels_to_csv_no_meta_uses_indices(tmp_path):
    # No tt and no meta json -> tt should be indices
    N = 50
    df = pd.DataFrame({"ch2": np.arange(N)})
    csv_out = tmp_path / "out2.csv"

    dump_channels_to_csv(df, csv_out, meta_json=None)

    out_df = pd.read_csv(csv_out)
    assert out_df.shape == (N, 5)
    assert np.allclose(out_df["tt"].to_numpy(), np.arange(N))
