Project Overview
================

Python utilities for two workflows:

1) **Audio board .bin processing**: convert device `.bin` files to DataFrames/JSON, export CSVs, plot waveforms, compute instantaneous frequency, and generate spectrograms.
2) **Participant synchronization**: parse participant directories, import biomechanics Excel, and synchronize audio with biomechanics using stomp events (supports dual-knee recordings with disambiguation). Run AFTER audio board processing.

Quick Start
-----------

1) Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
```

2) Install runtime dependencies:

```bash
pip install -r requirements.txt
```

3) (Optional) install developer tools:

```bash
pip install -r dev-requirements.txt
```

Common Commands
---------------

- Convert a `.bin` to pickle + JSON metadata:
  ```bash
  python read_audio_board_file.py path/to/file.bin --out ./outputs
  ```
- Export per-channel CSV:
  ```bash
  python dump_channels_to_csv.py ./outputs/file.pkl
  ```
- Plot per-channel waveforms:
  ```bash
  python plot_per_channel.py ./outputs/file.pkl
  ```
- Add instantaneous frequency:
  ```bash
  python add_instantaneous_frequency.py ./outputs/file.pkl
  ```
- Compute spectrograms:
  ```bash
  python compute_spectrogram.py ./outputs/file.pkl
  ```
- Process a participant directory (sync audio + biomechanics):
  ```bash
  python process_participant_directory.py /path/to/study/#1011
  ```

Biomechanics Excel (walking)
----------------------------

- `Walk0001`: shared pass metadata + sync events (`Sync Left`/`Sync Right`) for all walking speeds.
- Speed data sheets: `Slow_Walking`, `Medium_Walking`, `Fast_Walking` (V3D UIDs encode pass numbers/speed).
- Speed-specific `*_Walking_Events` exports may exist but stomp sync uses `Walk0001`.

Outputs
-------

- `<base>.pkl`: audio DataFrame (`tt`, `ch1`-`ch4`, optional `f_ch*`).
- `<base>_meta.json`: header metadata (fs, firmware, timestamps).
- `<base>_channels.csv`: per-channel CSV.
- `<base>_waveform_per_channel.png`: 4-row waveform plot.
- `<base>_with_freq.pkl` / `_with_freq_channels.csv`: instantaneous frequency results.
- `<base>_spectrogram.npz` + `_spec_ch*.png`: STFT outputs.
- Synchronized outputs saved under maneuver folders per participant.

Dual-Knee Synchronization Notes
-------------------------------

- `get_audio_stomp_time` can accept biomechanics stomp times for both knees and disambiguate two audio peaks by temporal order (recorded vs. contralateral knee).
- Non-walk maneuvers clip synchronized output to Movement Start/End ±0.5s.
- Walking uses pass-specific start/end based on speed and pass number.

Testing
-------

Run the full suite (from repo root with venv active):

```bash
PYTHONPATH=. pytest -v
```

Troubleshooting
---------------

- Missing sampling frequency (`fs`): scripts fall back to `tt` or metadata JSON; if absent, some analyses fail.
- Hilbert instantaneous frequency is sensitive to filtering; adjust `--lowcut/--highcut` as needed.
- Large spectrograms may need `nperseg`/`noverlap` tweaks to manage memory.
