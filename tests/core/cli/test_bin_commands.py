"""Tests for CLI entry points."""

import json
import sys
from unittest.mock import patch
import warnings

import pandas as pd
import pytest

from cli.add_instantaneous_frequency import main as add_inst_freq_main
from cli.compute_spectrogram import main as compute_spectrogram_main
from cli.dump_channels_to_csv import main as dump_channels_main
from cli.plot_per_channel import main as plot_per_channel_main
from cli.read_audio import main as read_audio_main


def test_ae_read_audio(tmp_path, capsys):
    """Test ae-read-audio CLI command."""
    # Create a dummy .bin file with a valid header
    bin_file = tmp_path / "test.bin"
    header = bytearray(512)
    # Set devFirmwareVersion to a known value
    header[24:28] = (2).to_bytes(4, byteorder="little")
    # Set numSDBlocks to 1
    header[61:65] = (1).to_bytes(4, byteorder="little")
    # Set a valid fileTime to avoid OverflowError
    # This is 2025-12-20 12:00:00 UTC
    file_time = 134292720000000000
    header[65:73] = file_time.to_bytes(8, byteorder="big")
    bin_file.write_bytes(header)
    # Write one block of data
    bin_file.write_bytes(bytearray(512))

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Unrecognized firmware version! Using default number of bits and sample rate",
            category=UserWarning,
        )
        with patch.object(sys, "argv", ["ae-read-audio", str(bin_file), "--out", str(tmp_path)]):
            read_audio_main()

    captured = capsys.readouterr()
    assert "Data saved to" in captured.out
    assert (tmp_path / "test.pkl").exists()
    assert (tmp_path / "test_meta.json").exists()


def test_ae_dump_channels(dummy_pkl_file, capsys):
    """Test ae-dump-channels CLI command."""
    with patch.object(sys, "argv", ["ae-dump-channels", str(dummy_pkl_file)]):
        dump_channels_main()

    captured = capsys.readouterr()
    assert "Wrote CSV" in captured.out
    assert dummy_pkl_file.with_name(f"{dummy_pkl_file.stem}_channels.csv").exists()


def test_ae_plot_per_channel(dummy_pkl_file, capsys):
    """Test ae-plot-per-channel CLI command."""
    with patch.object(sys, "argv", ["ae-plot-per-channel", str(dummy_pkl_file)]):
        plot_per_channel_main()

    captured = capsys.readouterr()
    assert "Saved per-channel waveform" in captured.out
    assert dummy_pkl_file.with_name(f"{dummy_pkl_file.stem}_waveform_per_channel.png").exists()


def test_ae_add_inst_freq(dummy_pkl_file, capsys):
    """Test ae-add-inst-freq CLI command."""
    with patch.object(sys, "argv", ["ae-add-inst-freq", str(dummy_pkl_file)]):
        add_inst_freq_main()

    captured = capsys.readouterr()
    assert "Saving updated pickle to" in captured.out
    assert dummy_pkl_file.with_name(f"{dummy_pkl_file.stem}_with_freq.pkl").exists()


def test_ae_add_inst_freq_uses_meta_fs(tmp_path, capsys):
    """Fallback to meta fs when tt is absent."""
    pkl_path = tmp_path / "no_tt.pkl"
    meta_path = tmp_path / "no_tt_meta.json"

    pd.DataFrame({"ch1": [0.1, 0.2, 0.3, 0.4]}).to_pickle(pkl_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"fs": 1000.0}, f)

    with patch.object(sys, "argv", ["ae-add-inst-freq", str(pkl_path)]):
        add_inst_freq_main()

    out_pkl = tmp_path / "no_tt_with_freq.pkl"
    assert out_pkl.exists()
    df_out = pd.read_pickle(out_pkl)
    for col in ("f_ch1", "f_ch2", "f_ch3", "f_ch4"):
        # Only ch1 exists, others absent
        if col == "f_ch1":
            assert col in df_out.columns
        else:
            assert col not in df_out.columns


def test_ae_add_inst_freq_bad_meta_json(tmp_path):
    """Invalid meta JSON should raise a runtime error."""
    pkl_path = tmp_path / "bad_meta.pkl"
    meta_path = tmp_path / "bad_meta_meta.json"

    pd.DataFrame({"ch1": [0.1, 0.2, 0.3, 0.4]}).to_pickle(pkl_path)
    # Write malformed JSON
    meta_path.write_text("not-json", encoding="utf-8")

    with patch.object(sys, "argv", ["ae-add-inst-freq", str(pkl_path)]):
        with pytest.raises(RuntimeError):
            add_inst_freq_main()


def test_ae_add_inst_freq_no_tt_no_meta(tmp_path):
    """Missing tt and meta.fs should raise a runtime error."""
    pkl_path = tmp_path / "no_info.pkl"
    pd.DataFrame({"ch1": [0.1, 0.2, 0.3, 0.4]}).to_pickle(pkl_path)

    with patch.object(sys, "argv", ["ae-add-inst-freq", str(pkl_path)]):
        with pytest.raises(RuntimeError):
            add_inst_freq_main()


def test_ae_compute_spectrogram_no_channels(tmp_path, capsys):
    """Spectrogram should still produce NPZ when no channels exist."""
    pkl_path = tmp_path / "tt_only.pkl"
    tt = [0.0, 0.001, 0.002, 0.003, 0.004]
    pd.DataFrame({"tt": tt}).to_pickle(pkl_path)

    with patch.object(sys, "argv", ["ae-compute-spectrogram", str(pkl_path)]):
        exit_code = compute_spectrogram_main()

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Saved spectrogram arrays" in captured.out
    npz_path = tmp_path / "tt_only_spectrogram.npz"
    assert npz_path.exists()
    # No per-channel PNGs should be created
    assert not list(tmp_path.glob("tt_only_spec_*.png"))


def test_ae_dump_channels_unreadable_pickle(tmp_path):
    """Unreadable pickle should exit with code 3."""
    bad_path = tmp_path / "bad.pkl"
    bad_path.write_text("not a pickle", encoding="utf-8")

    with patch.object(sys, "argv", ["ae-dump-channels", str(bad_path)]):
        with pytest.raises(SystemExit) as excinfo:
            dump_channels_main()

    assert excinfo.value.code == 3


def test_ae_compute_spectrogram(dummy_pkl_file, capsys):
    """Test ae-compute-spectrogram CLI command."""
    with patch.object(sys, "argv", ["ae-compute-spectrogram", str(dummy_pkl_file)]):
        compute_spectrogram_main()

    captured = capsys.readouterr()
    assert "Saved spectrogram arrays" in captured.out
    assert dummy_pkl_file.with_name(f"{dummy_pkl_file.stem}_spectrogram.npz").exists()


def test_ae_dump_channels_missing_file(tmp_path):
    """Missing pickle should exit with code 2."""
    missing = tmp_path / "does_not_exist.pkl"
    with patch.object(sys, "argv", ["ae-dump-channels", str(missing)]):
        with pytest.raises(SystemExit) as excinfo:
            dump_channels_main()

    assert excinfo.value.code == 2


def test_ae_plot_per_channel_missing_file(tmp_path):
    """Missing pickle should exit with code 1."""
    missing = tmp_path / "no_pickle.pkl"
    with patch.object(sys, "argv", ["ae-plot-per-channel", str(missing)]):
        with pytest.raises(SystemExit) as excinfo:
            plot_per_channel_main()

    assert excinfo.value.code == 1


def test_ae_add_inst_freq_missing_file(tmp_path):
    """Missing pickle should exit with code 1."""
    missing = tmp_path / "no_file.pkl"
    with patch.object(sys, "argv", ["ae-add-inst-freq", str(missing)]):
        with pytest.raises(SystemExit) as excinfo:
            add_inst_freq_main()

    assert excinfo.value.code == 1


def test_ae_compute_spectrogram_missing_fs(tmp_path, capsys, caplog):
    """Pickle without tt/meta fs should log and return non-zero."""
    pkl_path = tmp_path / "no_fs.pkl"
    pd.DataFrame({"ch1": [1.0, 2.0, 3.0]}).to_pickle(pkl_path)

    with patch.object(sys, "argv", ["ae-compute-spectrogram", str(pkl_path)]):
        exit_code = compute_spectrogram_main()

    assert exit_code == 1
    assert "Cannot determine sampling frequency" in caplog.text
