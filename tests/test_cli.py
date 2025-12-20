"""Tests for CLI entry points."""

import sys
from pathlib import Path
from unittest.mock import patch

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

    with patch.object(
        sys, "argv", ["ae-read-audio", str(bin_file), "--out", str(tmp_path)]
    ):
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
    assert dummy_pkl_file.with_name(
        f"{dummy_pkl_file.stem}_waveform_per_channel.png"
    ).exists()


def test_ae_add_inst_freq(dummy_pkl_file, capsys):
    """Test ae-add-inst-freq CLI command."""
    with patch.object(sys, "argv", ["ae-add-inst-freq", str(dummy_pkl_file)]):
        add_inst_freq_main()

    captured = capsys.readouterr()
    assert "Saving updated pickle to" in captured.out
    assert dummy_pkl_file.with_name(
        f"{dummy_pkl_file.stem}_with_freq.pkl"
    ).exists()


def test_ae_compute_spectrogram(dummy_pkl_file, capsys):
    """Test ae-compute-spectrogram CLI command."""
    with patch.object(sys, "argv", ["ae-compute-spectrogram", str(dummy_pkl_file)]):
        compute_spectrogram_main()

    captured = capsys.readouterr()
    assert "Saved spectrogram arrays" in captured.out
    assert dummy_pkl_file.with_name(
        f"{dummy_pkl_file.stem}_spectrogram.npz"
    ).exists()
