Project: Audio board .bin processing (Python)
=============================================

This folder contains Python utilities converted from the original
`read_audio_board_file.m` MATLAB reader and several helper scripts to
export CSVs, plot waveforms, compute instantaneous frequency, and
generate spectrograms. The scripts were refactored to provide a
command-line interface and basic logging.

Files of interest
-----------------

- `read_audio_board_file.py` — Core translator that parses the 512-byte
  header and audio payload, converts ADC counts to voltages, builds a
  pandas DataFrame and writes a pickle (`<base>.pkl`) and metadata JSON
  (`<base>_meta.json`). Use as: `python read_audio_board_file.py <file.bin> --out <outdir>`.

Project: Audio-board .bin processing (Python)
===========================================

This folder contains Python utilities converted from the original
`read_audio_board_file.m` MATLAB reader and several helper scripts to:

- convert a device `.bin` to a pandas DataFrame and metadata JSON
- export per-channel CSVs
- produce per-channel waveform plots
- compute instantaneous frequency (Hilbert) per channel
- compute and save spectrograms (STFT)

Quick start
-----------

1) Create and activate a virtual environment (PowerShell example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install runtime deps (pinned versions are in `requirements.txt`):

```powershell
pip install -r requirements.txt
```

3) (Optional) install developer tools for formatting and testing:

```powershell
pip install -r dev-requirements.txt
```

Basic usage examples
--------------------

Convert a single `.bin` (writes `<base>.pkl` and `<base>_meta.json` by default):

```powershell
python read_audio_board_file.py path\to\HP_... .bin --out ./outputs
```

Export channels CSV from the pickle produced above:

```powershell
python dump_channels_to_csv.py ./outputs\HP_... .pkl
```

Save stacked per-channel waveform (4 rows):

```powershell
python plot_per_channel.py ./outputs\HP_... .pkl
```

Compute instantaneous frequency per channel (default band-pass 10–5000 Hz):

```powershell
python add_instantaneous_frequency.py ./outputs\HP_... .pkl
```

Compute spectrograms (STFT) and save PNGs + compressed NPZ:

```powershell
python compute_spectrogram.py ./outputs\HP_... .pkl
```

Run the full pipeline on a single file or a directory (wrapper):

```powershell
python process_bin_files.py path\to\file.bin
# or
python process_bin_files.py path\to\directory_with_bin_files
```

What each script produces
-------------------------

- `<base>.pkl` — pandas DataFrame pickle with columns like `tt,ch1,ch2,ch3,ch4`
- `<base>_meta.json` — extracted header metadata (fs, board firmware, times)
- `<base>_channels.csv` — CSV with `tt,ch1..ch4` (if available)
- `<base>_waveform_per_channel.png` — stacked 4-channel time series
- `<base>_with_freq.pkl` and `_with_freq_channels.csv` — DataFrame + CSV with `f_ch1..f_ch4`
- `<base>_spec_ch1.png` .. `_spec_ch4.png` and `<base>_spectrogram.npz` — STFT visuals + arrays
- `<base>_outputs\<base>.log` — per-run log captured by the wrapper

Troubleshooting & notes
-----------------------

- If the scripts cannot determine a sampling frequency (`fs`), they
  will try `tt` in the DataFrame, then `<base>_meta.json`. If both are
  missing, some operations (Hilbert, STFT) will raise an error.
- Hilbert-based instantaneous frequency is noisy without pre-filtering;
  we apply a default 10–5000 Hz band-pass which you can change with
  `--lowcut` / `--highcut` on `add_instantaneous_frequency.py`.
- Large files may use a lot of memory during STFT; consider using
  `nperseg`/`noverlap` adjustments in `compute_spectrogram.py`.
- The repository includes a `dev-requirements.txt` with formatter and
  test tools (black, isort, pytest). Formatting was already applied to
  the project files; the virtualenv folder `matlab_conv` is excluded.

Next steps I can take (pick any)
--------------------------------

- add minimal pytest-based smoke tests for header parsing and CSV export
- consolidate logging configuration into a shared helper module
- add a short PowerShell script to create the venv and install deps
- freeze the full environment into `requirements-full.txt` via `pip freeze`

If you'd like one of these, tell me which and I'll implement it.
# acoustic_emissions_processing
# acoustic_emissions_processing
