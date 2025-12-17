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

Bin processing (raw audio → analysis-ready files)
- Convert `.bin` to pickle + JSON metadata: `python read_audio_board_file.py path/to/file.bin --out ./outputs` ([read_audio_board_file.py](read_audio_board_file.py))
- Export per-channel CSV: `python dump_channels_to_csv.py ./outputs/file.pkl` ([dump_channels_to_csv.py](dump_channels_to_csv.py))
- Plot per-channel waveforms: `python plot_per_channel.py ./outputs/file.pkl` ([plot_per_channel.py](plot_per_channel.py))
- Add instantaneous frequency: `python add_instantaneous_frequency.py ./outputs/file.pkl` ([add_instantaneous_frequency.py](add_instantaneous_frequency.py))
- Compute spectrograms: `python compute_spectrogram.py ./outputs/file.pkl` ([compute_spectrogram.py](compute_spectrogram.py))

Synchronize audio with biomechanics
- Process all participants under a root path: `python process_participant_directory.py /path/to/studies` ([process_participant_directory.py](process_participant_directory.py))
- Process specific participants (with or without '#'): `python process_participant_directory.py /path/to/studies --participant 1011 #2024`
- Limit the number of participants: `python process_participant_directory.py /path/to/studies --limit 5`
- Sync a single unsynced audio pickle from a maneuver `_outputs/`: `python process_participant_directory.py /path/to/studies --sync-single /path/to/audio.pkl`
- Write logs to file: `python process_participant_directory.py /path/to/studies --log run.log`

Audio QC (maneuver-specific)
----------------------------

- Single file QC (flexion-extension or sit-to-stand):
  ```bash
  python audio_qc.py file /path/to/audio.pkl --maneuver flexion_extension --freq 0.25 --tail 5
  python audio_qc.py file /path/to/audio.pkl --maneuver sit_to_stand --freq 0.25 --tail 5
  ```
- Walking QC on a single file (detects passes and step rates):
  ```bash
  python audio_qc.py file /path/to/walk_audio.pkl --maneuver walk --resample-walk 100 --min-pass-peaks 6 --min-gap-s 2
  ```
- QC an entire participant directory (Left/Right knees, all maneuvers):
  ```bash
  python audio_qc.py dir /Users/spiewart/kae_signal_processing_ml/sample_project_directory/#1013 --maneuver all
  ```
- Useful flags:
  - `--time tt` to set the time column name in the pickle
  - `--channels ch1 ch2 ch3 ch4` to choose audio channels to average
  - `--bandpower-min-ratio 0.2` to require spectral support around the target/detected frequency
  - `--resample-walk`, `--min-pass-peaks`, `--min-gap-s` tune walking heel-strike detection
  - Walking defaults are lenient for coverage and period tolerance (tuned for real study data); tighten via `--min-pass-peaks`, `--period-tolerance`, or `--min-coverage` if your data are very clean.
  - Periodic maneuvers can return bandpower ratios with `--bandpower-min-ratio` to gate passes on spectral energy.

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
