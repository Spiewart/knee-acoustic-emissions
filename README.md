Project Overview
================

Python utilities for two workflows:

1) **Audio board .bin processing**: convert device `.bin` files to DataFrames/JSON, export CSVs, plot waveforms, compute instantaneous frequency, and generate spectrograms.
2) **Participant synchronization**: parse participant directories, import biomechanics Excel, and synchronize audio with biomechanics using stomp events (supports dual-knee recordings with disambiguation). Run AFTER audio board processing.

Project Structure
-----------------

The project is organized into clear submodules:

```
acoustic_emissions_processing/
├── src/                           # Library code (import from here)
│   ├── audio/                     # Audio processing
│   ├── biomechanics/              # Biomechanics processing
│   ├── synchronization/           # Audio-biomechanics sync
│   ├── visualization/             # Plotting utilities
│   └── orchestration/             # High-level workflows
├── cli/                           # Command-line interfaces
└── tests/                         # Test suite (230+ tests)
```

Data Models
-----------

### Data Validation Architecture

This project uses Pydantic models (`src/models.py`) as the single source of truth for data validation:

**Processing Log Models** (used for logging and Excel export):
- **AudioProcessingMetadata**: Audio file processing and QC metadata (validated via Pydantic)
- **BiomechanicsImportMetadata**: Biomechanics data import tracking (validated via Pydantic)
- **SynchronizationMetadata**: Audio-biomechanics synchronization details (validated via Pydantic)
- **MovementCyclesMetadata**: Movement cycle extraction and QC metadata (validated via Pydantic)

**Recording Models** (used during processing):
- **AcousticsFileMetadata / AcousticsRecording**: Audio-specific fields (`audio_file_name`, microphones 1-4, optional QC/timestamps/notes) layered on top of scripted maneuver + knee metadata.
- **BiomechanicsFileMetadata / BiomechanicsRecording**: Biomechanics file metadata (`biomech_file_name`, system, sync times, QC) with required walk details.
- **SynchronizedRecording**: Combines acoustics + biomechanics metadata and data after alignment.
- **MovementCycleMetadata / MovementCycle**: Per-cycle metadata (cycle IDs, energy, QC, notes, periodic noise detection, sync quality) plus the synchronized data slice used for aggregation/DB export.

**Key Design Principle**: Pydantic models contain metadata and recording properties (file names, QC parameters, timestamps, sample rates). Data-derived statistics (channel RMS values, per-sample counts) are stored separately in dataclass wrappers for Excel export but NOT validated through Pydantic.

See [ai_instructions.md](ai_instructions.md) for detailed information on adding new fields to models, [MIGRATION.md](MIGRATION.md) for module mappings, [QC_VERSIONING.md](docs/QC_VERSIONING.md) for QA/QC version tracking, and [CYCLE_QC.md](docs/CYCLE_QC.md) for movement cycle quality control details.

Prerequisites
-----------

- **Python 3.12 or higher** is required. Verify your Python version:
  ```bash
  python --version
  ```

Quick Start
-----------

### Option 1: Install as Package (Recommended)

Install in development mode for access to CLI commands:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Install package with dependencies
pip install -e .

# Or with dev dependencies (pytest, black, mypy, etc.)
pip install -e ".[dev]"
```

This makes CLI commands available: `ae-process-directory`, `ae-sync-qc`, `ae-visualize`, etc.

### Option 2: Manual Dependency Installation

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Install runtime dependencies
pip install -r requirements.txt

# (Optional) install developer tools
pip install -r dev-requirements.txt
```

Common Commands
---------------

### Using CLI Entry Points (After `pip install -e .`)

**Bin processing (raw audio → analysis-ready files):**
- Convert `.bin` to pickle + JSON metadata: `ae-read-audio path/to/file.bin --out ./outputs`
- Export per-channel CSV: `ae-dump-channels ./outputs/file.pkl`
- Plot per-channel waveforms: `ae-plot-per-channel ./outputs/file.pkl`
- Add instantaneous frequency: `ae-add-inst-freq ./outputs/file.pkl`
- Compute spectrograms: `ae-compute-spectrogram ./outputs/file.pkl`

Edge-Case Behavior
------------------

These CLIs include explicit handling for common error cases:

- **Missing pickle**:
  - `ae-dump-channels` exits with code 2 and logs an error.
  - `ae-plot-per-channel` exits with code 1 and logs an error.
  - `ae-add-inst-freq` exits with code 1 and logs an error.
- **Unreadable pickle** (corrupted or non-pickle):
  - `ae-dump-channels` exits with code 3.
- **Missing sampling frequency**:
  - `ae-compute-spectrogram` returns non-zero and logs “Cannot determine sampling frequency” when `tt` and meta `fs` are absent.
- **Meta `fs` fallback**:
  - `ae-add-inst-freq` falls back to `*_meta.json` `fs` when `tt` is absent.
- **No channels present**:
  - `ae-compute-spectrogram` still writes the NPZ with `f`/`t` arrays and returns success, without creating per-channel PNGs.

Firmware Warning
----------------

When reading `.bin` files with unknown firmware (`devFirmwareVersion`), `ae-read-audio` issues a warning and defaults to 16-bit, 46.875 kHz. This mirrors the original MATLAB logic and ensures consistent downstream processing.

**Synchronize audio with biomechanics:**
- Process all participants under a root path: `ae-process-directory /path/to/studies`
- Process specific participants: `ae-process-directory /path/to/studies --participant 1011 '#'2024`
- Limit the number of participants: `ae-process-directory /path/to/studies --limit 5`
- Sync a single unsynced audio pickle: `ae-process-directory --sync-single /path/to/audio.pkl`
- Write logs to file: `ae-process-directory /path/to/studies --log run.log`
- Choose pipeline entrypoint (bin → sync → cycles): `ae-process-directory /path/to/studies --entrypoint bin|sync|cycles` (default: sync)

**Audio QC (maneuver-specific):**
- Single file QC: `ae-audio-qc file /path/to/audio.pkl --maneuver flexion_extension`
- Walking QC: `ae-audio-qc file /path/to/walk_audio.pkl --maneuver walk`
- QC entire participant directory: `ae-audio-qc dir /path/to/participant --maneuver all`

**Synchronization QC:**
- Parse and QC synced files: `ae-sync-qc /path/to/synced_data.pkl`
- Run on directory: `ae-sync-qc /path/to/participant/Knee/Maneuver/Synced/`

**Visualization:**
- Plot synced data: `ae-visualize synced_data.pkl --save-path output.png`

**Cleanup outputs (for testing):**
- Clean single participant: `ae-cleanup-outputs /path/to/#1011`
- Clean entire study: `ae-cleanup-outputs /path/to/studies`
- Dry run (preview): `ae-cleanup-outputs /path/to/studies --dry-run`
- Limit participants: `ae-cleanup-outputs /path/to/studies --limit 5`

**Machine Learning (trains on processed participants with complete synced cycles):**

The ML commands automatically discover participant directories with fully processed acoustics and biomechanics data (indicated by `knee_processing_log_*_{Left,Right}.xlsx` with all maneuvers having ≥1 synced files). They then extract acoustic features from movement cycles and train logistic regression models with leave-one-out cross-validation (for small datasets) or train/test split (for larger datasets).

**Important: Data Leakage Prevention for Knee-Level Models**

For knee-level models (TFM KL, PFM KL, Varus Thrust), the training process automatically excludes the contralateral knee of any participant whose other knee appears in the test set. This prevents data leakage from shared physiological characteristics between a participant's two knees.

- **Knee Pain (participant-level outcome):**
  ```bash
  ae-ml-kneepain /path/to/project --demographics /path/to/demographics.xlsx
  ae-ml-kneepain /path/to/project --demographics /path/to/demographics.xlsx --maneuvers walk sit_to_stand
  ae-ml-kneepain /path/to/project --demographics /path/to/demographics.xlsx --cycle-type clean --aggregation mean --allow-partial-knees
  ```

- **Tibiofemoral KL Grade (knee-level outcome):**
  ```bash
  ae-ml-tfmkl /path/to/project --outcome-file /path/to/outcomes.xlsx --sheet "TFM KL"
  ae-ml-tfmkl /path/to/project --outcome-file /path/to/outcomes.xlsx --sheet "TFM KL" --maneuvers walk
  ```

- **Patellofemoral KL Grade (knee-level outcome):**
  ```bash
  ae-ml-pfmkl /path/to/project --outcome-file /path/to/outcomes.xlsx --sheet "PFM KL"
  ae-ml-pfmkl /path/to/project --outcome-file /path/to/outcomes.xlsx --sheet "PFM KL" --allow-partial-knees
  ```

- **Varus Thrust (knee-level outcome):**
  ```bash
  ae-ml-varusthrust /path/to/project --outcome-file /path/to/outcomes.xlsx --sheet "Varus Thrust"
  ae-ml-varusthrust /path/to/project --outcome-file /path/to/outcomes.xlsx --sheet "Varus Thrust" --aggregation median
  ```

**ML Options:**

*Participant-level (`ae-ml-kneepain`):**
- `--demographics`: Path to demographics Excel file with outcome column (required).
- `--maneuvers`: Filter to specific maneuvers (e.g., `walk sit_to_stand`); defaults to all.
- `--cycle-type`: Load "clean" or "outliers" cycles (default: "clean").
- `--aggregation`: How to aggregate features per participant: "mean", "median", "max", "min" (default: "mean").
- `--allow-partial-knees`: Include participants with only one processed knee (default: require both).

*Knee-level (`ae-ml-tfmkl`, `ae-ml-pfmkl`, `ae-ml-varusthrust`):**
- `--outcome-file`: Path to Excel file with knee-level outcome variable (defaults to `cohort_chars_PRELIM_12_22_2025.xlsx` in project root).
- `--sheet`: Sheet name when both knees are in one sheet (e.g., "Varus Thrust", "TFM KL").
- `--left-sheet` / `--right-sheet`: Use separate sheets for left/right knees (e.g., "KOOS R Knee", "KOOS L Knee").
- `--side-column`: Column indicating knee side in outcome data (default: "Knee").
- `--maneuvers`: Filter to specific maneuvers (e.g., `walk sit_to_stand`); defaults to all.
- `--cycle-type`: Load "clean" or "outliers" cycles (default: "clean").
- `--allow-partial-knees`: Include participants with only one processed knee (default: require both).
- `--no-contralateral-exclusion`: Disable exclusion of contralateral knees from training; by default, contralateral knees are excluded to prevent data leakage.


### Using Python Modules Directly (No installation)

**Bin processing:**
- Convert `.bin`: `python -m cli.read_audio path/to/file.bin --out ./outputs`
- (Other bin processing tools remain unchanged: `dump_channels_to_csv.py`, etc.)

**Synchronize audio with biomechanics:**
- Process participants: `python -m cli.process_directory /path/to/studies`
- Process specific: `python -m cli.process_directory /path/to/studies --participant 1011`
- Single file sync: `python -m cli.process_directory --sync-single /path/to/audio.pkl`
- Start at a specific stage: `python -m cli.process_directory /path/to/studies --entrypoint bin|sync|cycles` (default: sync)

**Audio QC:**
- Single file: `python -m cli.audio_qc file /path/to/audio.pkl --maneuver walk`
- Directory: `python -m cli.audio_qc dir /path/to/participant --maneuver all`

**Sync QC:**
- Parse and QC: `python -m cli.sync_qc /path/to/synced_data.pkl`

**Visualization:**
- Plot synced data: `python -m cli.visualize synced_data.pkl`

Raw Audio QC (signal quality)
-----------------------------

The raw audio QC module automatically detects signal dropout and artifactual noise when processing .bin files. This QC runs during the bin processing stage (before other QC checks) and ensures raw sensor data quality.

**Automatic QC during bin processing:**
```bash
ae-process-directory /path/to/studies --entrypoint bin
```

**What it detects:**
- **Signal dropout**: Silence or flatline conditions (sensor failure/disconnection)
- **Artifactual noise**: Spikes, outliers, or abnormal signal patterns

**Results storage:**
- Bad intervals are stored in processing logs as `QC_not_passed` column
- Format: List of (start_time, end_time) tuples in seconds
- Example: `[(1.5, 2.3), (5.0, 6.2)]` indicates two problematic sections

**Programmatic usage:**
```python
from src.audio.raw_qc import run_raw_audio_qc, merge_bad_intervals, clip_bad_segments
import pandas as pd

# Load audio data
df = pd.read_pickle("audio_data.pkl")

# Run QC
dropout_intervals, artifact_intervals = run_raw_audio_qc(df)

# Merge and optionally clip out bad segments
bad_intervals = merge_bad_intervals(dropout_intervals, artifact_intervals)
clean_df = clip_bad_segments(df, bad_intervals)
```

See [docs/RAW_AUDIO_QC.md](docs/RAW_AUDIO_QC.md) for detailed documentation, tuning guidelines, and API reference.

Audio QC (maneuver-specific)
----------------------------

The `ae-audio-qc` CLI provides quality control checks for audio recordings.

- **Single file QC (flexion-extension or sit-to-stand):**
  ```bash
  ae-audio-qc file /path/to/audio.pkl --maneuver flexion_extension --freq 0.25 --tail 5
  ae-audio-qc file /path/to/audio.pkl --maneuver sit_to_stand --freq 0.25 --tail 5
  ```

- **Walking QC on a single file (detects passes and step rates):**
  ```bash
  ae-audio-qc file /path/to/walk_audio.pkl --maneuver walk --resample-walk 100 --min-pass-peaks 6
  ```

- **QC an entire participant directory (Left/Right knees, all maneuvers):**
  ```bash
  ae-audio-qc dir /path/to/participant/ --maneuver all
  ```

- **Useful flags:**
  - `--time tt` to set the time column name in the pickle
  - `--channels ch1 ch2 ch3 ch4` to choose audio channels to average
  - `--bandpower-min-ratio 0.2` to require spectral support around the target/detected frequency
  - `--resample-walk`, `--min-pass-peaks`, `--min-gap-s` tune walking heel-strike detection
  - Walking defaults are lenient for coverage and period tolerance (tuned for real study data)
  - Periodic maneuvers can return bandpower ratios with `--bandpower-min-ratio` to gate passes on spectral energy

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

Parsing Synchronized Data for Movement Cycles
---------------------------------------------

After synchronizing audio and biomechanics data, the `ae-sync-qc` CLI can be used to parse the synchronized `.pkl` files, identify movement cycles, and perform quality control to separate clean cycles from outliers based on acoustic energy.

The script saves the clean and outlier cycles as separate `.pkl` files in a `MovementCycles` subdirectory.

### Command-Line Usage

The script can be run on a single file, a `Synced` directory, or an entire participant directory.

-   **Run QC on a single synced file:**

    ```bash
    ae-sync-qc /path/to/participant/Knee/Maneuver/Synced/data_synced.pkl
    ```

-   **Run QC on all synced files in a directory:**

    ```bash
    ae-sync-qc /path/to/participant/Knee/Maneuver/Synced/
    ```

-   **Run QC on an entire participant directory (finds all synced files):**

    ```bash
    ae-sync-qc /path/to/participant/
    ```

### Optional Arguments

-   `--threshold`: Adjust the acoustic energy threshold for classifying cycles as clean or outliers. The default is `100.0`.

    ```bash
    ae-sync-qc /path/to/data_synced.pkl --threshold 50.0
    ```

-   `--no-plots`: Skip the creation of visualization plots for each cycle.

    ```bash
    ae-sync-qc /path/to/data_synced.pkl --no-plots
    ```

-   `--maneuver` and `--speed`: Manually specify the maneuver and speed to override the script's automatic inference from the file path.

    ```bash
    ae-sync-qc /path/to/data.pkl --maneuver walk --speed fast
    ```

-   `-v` or `--verbose`: Enable verbose logging for detailed output.

    ```bash
    ae-sync-qc /path/to/data_synced.pkl -v
    ```

Testing
-------

Run the full suite (from repo root with venv active):

```bash
pytest tests/ -v
```

All 230+ tests should pass. If you've made changes, ensure tests pass before committing:

```bash
pytest tests/ -v --tb=short
```

Using as a Library
------------------

Import functions from the `src` package in your own code:

```python
# Audio processing
from src.audio.readers import read_audio_board_file
from src.audio.parsers import get_acoustics_metadata
from src.audio.quality_control import qc_audio_walk, qc_audio_flexion_extension

# Biomechanics processing
from src.biomechanics.importers import import_biomechanics_recordings
from src.biomechanics.cycle_parsing import extract_movement_cycles

# Synchronization
from src.synchronization.sync import sync_audio_with_biomechanics
from src.synchronization.quality_control import perform_sync_qc

# Visualization
from src.visualization.plots import plot_syncd_data

# High-level workflows
from src.orchestration.participant import process_participant
```

See [MIGRATION.md](MIGRATION.md) for detailed import paths and examples.

Troubleshooting
---------------

- Missing sampling frequency (`fs`): scripts fall back to `tt` or metadata JSON; if absent, some analyses fail.
- Hilbert instantaneous frequency is sensitive to filtering; adjust `--lowcut/--highcut` as needed.
- Large spectrograms may need `nperseg`/`noverlap` tweaks to manage memory.

### Updated CLI Usage

#### `ae-process-directory`

Process participant directories with additional filtering options:

```bash
# Process all participants under a root path
$ ae-process-directory /path/to/studies

# Process specific participants
$ ae-process-directory /path/to/studies --participant 1011 2024

# Limit the number of participants
$ ae-process-directory /path/to/studies --limit 5

# Specify knee side (left or right)
$ ae-process-directory /path/to/studies --knee left

# Specify maneuver type (walk, fe, sts)
$ ae-process-directory /path/to/studies --maneuver walk

# Combine filters
$ ae-process-directory /path/to/studies --knee right --maneuver fe
```
