# Study-Specific Configuration

## Overview

All study-specific logic — directory layouts, Excel sheet naming, UID parsing, event
name construction, speed codes — is encapsulated behind the `StudyConfig` protocol
(`src/studies/base.py`). Core pipeline modules (`biomechanics/importers.py`,
`audio/quality_control.py`, `synchronization/sync.py`, `audio/parsers.py`) use
this protocol and **never** reference study-specific conventions directly.

## Architecture

```
src/studies/
├── __init__.py          # Registry: get_study_config(name) → StudyConfig
├── base.py              # StudyConfig Protocol definition
└── aoa/
    ├── __init__.py      # Registers AOAConfig on import
    ├── config.py        # AOAConfig (implements StudyConfig)
    └── legend.py        # AOA-specific legend sheet parsing
```

### Registry Pattern

Studies are registered via `register_study()` and looked up via `get_study_config()`:

```python
from src.studies import get_study_config

config = get_study_config("AOA")
config.get_knee_directory_name("left")  # → "Left Knee"
```

Known studies are auto-imported on first access. To add a new study, create a package
under `src/studies/` and call `register_study()` in its `__init__.py`.

## Protocol Methods by Processing Stage

The `StudyConfig` protocol organizes methods into four processing stages. Each stage
corresponds to a pipeline phase and the pipeline modules that consume those methods.

### 1. Identity & Directory Structure

| Method | Purpose | AOA Example |
|--------|---------|-------------|
| `study_name` | Short study identifier | `"AOA"` |
| `get_knee_directory_name(knee)` | Folder name per knee side | `"Left Knee"` |
| `parse_participant_id(dir_name)` | Extract (study, numeric_id) | `("#AOA1011") → ("AOA", 1011)` |
| `format_study_prefix(id)` | Build full prefixed ID | `(1011) → "AOA1011"` |
| `get_maneuver_directory_name(maneuver)` | Maneuver folder name | `"Walking"`, `"Sit-Stand"` |
| `get_maneuver_from_directory(dir_name)` | Reverse lookup | `"Walking" → "walk"` |
| `get_maneuver_search_terms(maneuver)` | Fuzzy match terms | `("sit", "stand")` |
| `get_motion_capture_directory_name()` | Biomechanics folder | `"Motion Capture"` |
| `find_excel_file(dir, pattern)` | Find .xlsx/.xlsm by stem | generic utility |

**Consumers**: `orchestration/`, `audio/`, `biomechanics/`

### 2. Audio Metadata (Legend Parsing)

| Method | Purpose | AOA Example |
|--------|---------|-------------|
| `get_acoustics_sheet_name()` | Metadata sheet name | `"Acoustic Notes"` |
| `get_legend_file_pattern()` | Legend file glob | `"*acoustic_file_legend*"` |
| `parse_legend_fallback(path, maneuver, knee)` | Secondary sheet data | Mic Setup sheet |
| `get_default_microphones()` | Default mic layout | 4-mic standard positions |

**Consumers**: `audio/parsers.py`

### 3. Biomechanics Import

| Method | Purpose | AOA Example |
|--------|---------|-------------|
| `get_biomechanics_file_pattern(id)` | Excel filename stem | `"AOA1011_Biomechanics_Full_Set"` |
| `construct_biomechanics_sheet_names(id, maneuver, speed)` | Data + event sheets | `{"data_sheet": "AOA1011_Slow_Walking", "event_sheet": "AOA1011_Walk0001"}` |
| `get_walk_event_sheet_base_name()` | Walk event sheet suffix | `"Walk0001"` |
| `parse_biomechanics_uid(uid)` | UID → BiomechanicsFileMetadata | `"AOA1011_Walk0001_NSP1_Filt" → {maneuver: "walk", speed: "normal", pass: 1}` |
| `get_speed_code_map()` | Speed → event prefix code | `{"slow": "SS", "normal": "NS", "fast": "FS"}` |
| `get_speed_event_keywords()` | Section header → speed code | `{"Slow Speed": "SS", "Medium Speed": "NS", ...}` |

**Consumers**: `biomechanics/importers.py`, `audio/quality_control.py`

### 4. Synchronization (Event Names & Columns)

| Method | Purpose | AOA Example |
|--------|---------|-------------|
| `get_stomp_event_name(foot)` | Sync stomp label | `"Sync Left"` |
| `get_movement_start_event(maneuver, speed, pass)` | Start event label | `"SS Pass 1 Start"` |
| `get_movement_end_event(maneuver, speed, pass)` | End event label | `"SS Pass 1 End"` |
| `get_biomechanics_event_column()` | Event label column | `"Event Info"` |
| `get_biomechanics_time_column()` | Time column | `"Time (sec)"` |
| `get_knee_angle_column()` | Knee angle column | `"Knee Angle Z"` |

**Consumers**: `synchronization/sync.py`, `biomechanics/importers.py`

## Adding a New Study

1. Create `src/studies/{study_name}/`:
   ```
   src/studies/preoa/
   ├── __init__.py
   ├── config.py
   └── legend.py  (optional)
   ```

2. In `config.py`, implement a class with **all** methods from `StudyConfig`:
   ```python
   class PreOAConfig:
       @property
       def study_name(self) -> str:
           return "preOA"
       # ... implement all protocol methods
   ```

3. In `__init__.py`, register the config:
   ```python
   from src.studies import register_study
   from src.studies.preoa.config import PreOAConfig
   register_study("preOA", PreOAConfig())
   ```

4. Add `"src.studies.preoa"` to the `_KNOWN_STUDIES` list in `src/studies/__init__.py`.

5. Add tests in `tests/studies/preoa/test_preoa_config.py`.

## Rules for Pipeline Modules

- **NEVER** hardcode study-specific values (speed codes, sheet names, event names,
  column names, directory names) in pipeline modules.
- **ALWAYS** obtain study-specific values through `get_study_config(study_name)`.
- Pipeline functions that need study-specific behavior must accept a
  `study_name: str = "AOA"` parameter and delegate through the config.
