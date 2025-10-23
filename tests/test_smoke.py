import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _load_module_from_path(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_synthetic_bin(path: Path) -> None:
    """Create a small synthetic .bin with a 512-byte header and one full packet.

    The format matches the expectations of `read_audio_board_file.py` helpers
    (header fields at specific offsets and a single packet of uint32 words).
    """
    header = bytearray(512)
    # devFirmwareVersion at bytes 24:28 -> set to 2 (little-endian)
    header[24:28] = (2).to_bytes(4, byteorder="little", signed=False)
    # numSDBlocks at 61:65 -> set to 1
    header[61:65] = (1).to_bytes(4, byteorder="little", signed=False)
    # fileTime (8 bytes) at 65:73 -> set to zero
    header[65:73] = (0).to_bytes(8, byteorder="little", signed=False)

    # Parameters from reader implementation
    adc_half_buffer = 5504
    tt_block_length = 128
    num_uint32_per_write = adc_half_buffer + tt_block_length

    # Create a single packet of uint32 words with a simple pattern
    raw = np.arange(num_uint32_per_write, dtype=np.uint32)
    # ensure values fit in 32 bits and provide both upper/lower words
    raw = (raw & 0xFFFFFFFF).astype('<u4')
    raw_bytes = raw.tobytes()

    # For the sync region (last 128 uint32) we want first few entries to be sensible
    # We'll overwrite the last 128 uint32s with a sync vector where first 4 entries are non-zero
    sync = np.zeros(tt_block_length, dtype='<u4')
    sync[0] = 1000
    sync[1] = 2000
    sync[2] = 3000
    sync[3] = 4000
    sync_bytes = sync.tobytes()

    # Compose the file: header + data-with-sync
    # raw_bytes length is num_uint32_per_write * 4
    # Replace final 128*4 bytes with sync_bytes
    data_bytes = bytearray(raw_bytes)
    data_bytes[-(tt_block_length * 4) :] = sync_bytes

    with open(path, "wb") as f:
        f.write(header)
        f.write(data_bytes)


def test_end_to_end_smoke(tmp_path: Path):
    """Smoke test: create synthetic .bin, run reader and CSV exporter, check outputs."""
    bin_path = tmp_path / "synthetic.bin"
    _make_synthetic_bin(bin_path)

    out_dir = tmp_path

    # Dynamically load and run the core reader module (from project root)
    project_root = Path.cwd()
    reader_mod = _load_module_from_path(project_root / "read_audio_board_file.py")
    reader_mod.read_audio_board_file(str(bin_path), str(out_dir))

    base = bin_path.stem
    pkl = out_dir / (base + ".pkl")
    meta = out_dir / (base + "_meta.json")
    assert pkl.exists(), "Pickle not created"
    assert meta.exists(), "Meta JSON not created"

    # Load pickle (may be empty for minimal synthetic file)
    df = pd.read_pickle(pkl)
    # It's acceptable for the synthetic file to produce an empty DataFrame,
    # but if data exists it should contain channels ch1..ch4.
    if len(df) > 0:
        cols = [c.lower() for c in df.columns]
        assert "ch1" in cols or "ch2" in cols, "Expected channel columns in DataFrame when data present"

    # Load CSV exporter module and run its main() with adjusted argv
    csv_mod = _load_module_from_path(project_root / "dump_channels_to_csv.py")
    old_argv = sys.argv[:]
    try:
        sys.argv = ["dump_channels_to_csv.py", str(pkl)]
        csv_mod.main()
    finally:
        sys.argv = old_argv

    csv = out_dir / (base + "_channels.csv")
    assert csv.exists(), "Channels CSV not created"
    out_df = pd.read_csv(csv)
    # CSV should include the timestamp column and the 4 channel columns (may be NaN/empty)
    assert set(["tt", "ch1", "ch2", "ch3", "ch4"]).issubset(set(out_df.columns)), "CSV missing expected columns"
