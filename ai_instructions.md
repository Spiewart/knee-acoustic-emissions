# AI Assistant Instructions for Acoustic Emissions Processing

## Project Overview

This project processes acoustic emissions data from knee joint recordings during various biomechanical maneuvers (walking, sit-to-stand, flexion-extension). It integrates audio data with biomechanics motion capture data from participants in a research study.

### Key Components

1. **Audio Processing**: Converts `.bin` files from audio boards to pandas DataFrames
2. **Biomechanics Data**: Reads and processes motion capture data from Excel files
3. **Data Synchronization**: Aligns audio and biomechanics data using foot stomp events (supports dual-knee recordings)
4. **Analysis**: Computes spectrograms, instantaneous frequency, and per-channel metrics

---

## Virtual Environment Setup

### Requirements

**Python 3.12+** is required.

The project uses the following dependencies:

**Runtime Dependencies** (`requirements.txt`):
- `numpy==2.3.4` - Numerical computing
- `openpyxl==3.1.3` - Excel file handling
- `pandas==2.3.3` - Data manipulation and analysis
- `scipy==1.16.2` - Scientific computing (spectrograms, signal processing)
- `matplotlib==3.10.7` - Data visualization
- `pydantic==2.12.4` - Data validation and modeling

**Development Dependencies** (`dev-requirements.txt`):
- `black==25.9.0` - Code formatting
- `isort==7.0.0` - Import sorting
- `pytest==8.4.2` - Testing framework
- `mypy==1.8.0` - Type checking
- `flake8==6.1.0` - Linting

### Installation Steps

1. Create virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   # or
   .\.venv\Scripts\Activate.ps1  # PowerShell
   ```

2. Install runtime dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install development tools (optional):
   ```bash
   pip install -r dev-requirements.txt
   ```

---

## Project Structure & Key Files

### Core Data Models (`models.py`)

Pydantic models for data validation:

- **`MicrophonePosition`**: Microphone placement metadata (patellar position, laterality)
- **`AcousticsMetadata`**: Audio recording metadata with conditional validation
  - `scripted_maneuver`: "walk", "sit_to_stand", or "flexion_extension"
  - `speed`: "slow", "medium", or "fast" (only for walk)
  - `knee`: "left" or "right"
  - Microphone dictionary with keys 1-4

- **`BiomechanicsMetadata`**: Motion capture recording metadata with **conditional validation rules**:
  - **When `maneuver="walk"`**: `pass_number` (int, required) and `speed` (required, one of "slow", "normal", "fast")
  - **When `maneuver="sit_to_stand"` or `"flexion_extension"`**: `pass_number` and `speed` must be `None`

- **`BiomechanicsCycle`**: Biomechanics data + metadata
- **`AcousticsCycle`**: Audio data + metadata
- **`SynchronizedCycle`**: Combined audio + biomechanics data

### Processing Modules

- **`process_biomechanics.py`**: Extract, normalize, and import biomechanics recordings
  - `get_biomechanics_metadata()`: Parses UIDs to extract maneuver, speed, pass_number (with conditional logic)
  - `import_walk_biomechanics()` / `import_fe_sts_biomechanics()`: Specialized imports for walk vs. non-walk maneuvers
  - `import_biomechanics_recordings()`: Dispatcher that validates inputs and routes to the specialized functions

- **`parse_acoustic_file_legend.py`**: Reads Excel legends to extract audio metadata

- **`sync_audio_with_biomechanics.py`**: Synchronizes audio and biomechanics using foot stomp events; `get_audio_stomp_time` supports dual-knee disambiguation when both biomechanics stomps are provided

- **`read_audio_board_file.py`**: Core translator for `.bin` files → DataFrames + JSON metadata

- **`process_participant_directory.py`**: Orchestrates full participant data processing

- **`plot_per_channel.py`**: Generates per-channel waveform visualizations

- **`compute_spectrogram.py`**: Computes STFT spectrograms

- **`cli/add_instantaneous_frequency.py`**: Computes Hilbert transform instantaneous frequency

### Testing

Tests located in `tests/` directory:
- `test_process_biomechanics.py` - Biomechanics processing tests
- `test_parse_acoustic_file_legend.py` - Audio metadata parsing tests
- `test_process_participant_directory.py` - Directory processing tests
- `test_sync_audio_with_biomechanics.py` - Synchronization tests
- `test_cli.py` - CLI entry point tests
- `test_smoke.py` - End-to-end smoke tests
- `conftest.py` - Shared pytest fixtures

**Running Tests**: Use the Pylance MCP server to run tests, as it's much faster and more reliable than running pytest in the terminal:

```python
# Using Pylance server to run tests
mcp_pylance_mcp_s_pylanceRunCodeSnippet(
    workspaceRoot="file:///path/to/workspace",
    workingDirectory="/path/to/workspace",
    codeSnippet="""
import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-v"],
    capture_output=True,
    text=True,
    cwd="/path/to/workspace"
)
print(result.stdout)
print(f"Exit code: {result.returncode}")
"""
)
```

Avoid running `pytest` directly in the terminal as it often hangs.

---

## Coding Standards & Conventions

### Code Style

- **Formatting**: Use `black` for auto-formatting
- **Imports**: Use `isort` for consistent import ordering
- **Type Hints**: Prefer type annotations; use `mypy` for static type checking
- **Linting**: Follow `flake8` rules

### Documentation

- Module docstrings explain purpose and key functions
- Function docstrings include:
  - Brief description
  - Args with types
  - Returns with type
  - Raises (if applicable)
- Complex logic should have explanatory comments

### Data Validation

- Use Pydantic models for all data structures
- Field validators should include clear error messages
- Conditional validation rules must be documented in docstrings
- All DataFrames must be wrapped in custom Pydantic DataFrame subclasses

### Testing

- Write tests for new functionality
- All tests should pass before commits
- Use descriptive test names: `test_<function>_<scenario>`
- Use pytest fixtures for common setup (see `conftest.py`)

---

## Important Implementation Details

### BiomechanicsMetadata Validation

The `BiomechanicsMetadata` class enforces conditional validation:

```python
@field_validator('speed')
@classmethod
def validate_speed_for_maneuver(cls, v: str | None, info) -> str | None:
    if info.data.get('maneuver') == 'walk':
        if v is None:
            raise ValueError("speed is required when maneuver is 'walk'")
    else:  # sit_to_stand or flexion_extension
        if v is not None:
            raise ValueError(f"speed must be None when maneuver is '{info.data.get('maneuver')}'")
    return v
```

Similar logic applies to `pass_number`. When extending to non-walk maneuvers, ensure:
1. `get_biomechanics_metadata()` returns `speed=None, pass_number=None` for non-walk maneuvers
2. `import_biomechanics_recordings()` handles the TODO for sit_to_stand and flexion_extension
3. Start time extraction logic: walking uses pass-specific start/end; non-walk uses Movement Start/End with no frameshift

### UID Format

Biomechanics unique identifiers follow the pattern:
```
Study{StudyID}_{Maneuver}{StudyNum}_{SpeedPass}_{Filt}
Example: Study123_Walk0001_NSP1_Filt
```

Where:
- `Maneuver`: Walk, SitToStand, FlexExt (normalized to lowercase)
- `SpeedPass`: Two-letter speed code (SS/NS/FS) + P + pass number
  - SS = Slow
  - NS = Normal
  - FS = Fast

### Biomechanics Excel sheets (walking)

- `Walk0001` is the pass-metadata sheet for *all* walking speeds. It is always named the same for every participant; the `0001` is export nomenclature and not a true pass number. It carries sync events (`Sync Left` / `Sync Right`) and pass timing markers used for stomp alignment.
- Speed-specific data sheets are `Slow_Walking`, `Medium_Walking`, and `Fast_Walking`. Their V3D column UIDs (e.g., `Walk0001_SSP4_Filt`) encode both speed and pass number.
- Speed-specific `*_Walking_Events` sheets may exist in exports but are not used for stomp-time sync; rely on `Walk0001` for sync events.

### DataFrame Columns

**Audio Data (AcousticsRecording)**:

- `tt`: time vector
- `ch1`, `ch2`, `ch3`, `ch4`: audio channels
- `f_ch1`, `f_ch2`, `f_ch3`, `f_ch4`: instantaneous frequency

**Biomechanics Data (BiomechanicsRecording)**:

- `TIME`: time vector (timedelta, relative to start)
- Various motion capture columns (joint angles, positions, etc.)

**Synchronized Data (SynchronizedRecording)**:

- All columns from both audio and biomechanics

---

## Common Tasks for AI Assistants

### Adding New Features

1. Define Pydantic models in `models.py` if introducing new data structures
2. Implement processing logic in appropriate module
3. Add field validators for data validation
4. Write comprehensive tests
5. Update docstrings and this guide if adding major functionality

### Debugging Data Issues

1. Check that DataFrames have required columns (validated in `__get_pydantic_core_schema__`)
2. Verify conditional validation in BiomechanicsMetadata (maneuver → speed/pass_number rules)
3. Ensure TIME column is properly relative (not absolute timestamps)
4. Check for proper handling of multiple passes/speeds in walking data

### Extending to Non-Walk Maneuvers

Currently, only "walk" maneuvers are fully supported. To extend:

1. Update `import_biomechanics_recordings()` to process sit_to_stand and flexion_extension
2. Implement alternative start-time detection (currently uses walking event data)
3. Handle cases where speed/pass_number are None for these maneuvers
4. Add tests for new maneuver types
5. Update `get_walking_start_time()` or create equivalent for other maneuvers

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Import errors for dependencies | Ensure venv activated and `pip install -r requirements.txt` run |
| Pydantic validation errors | Check field types and conditional rules in `BiomechanicsMetadata` |
| Test failures | Run `pytest tests/test_smoke.py` first to check end-to-end functionality |
| DataFrame column missing | Verify data source has required columns before DataFrame creation |
| Time synchronization issues | Check foot stomp detection in biomechanics data |

---

## References & Resources

- **Pydantic Documentation**: <https://docs.pydantic.dev/>
- **Pandas Documentation**: <https://pandas.pydata.org/docs/>
- **SciPy Signal Processing**: <https://docs.scipy.org/doc/scipy/reference/signal.html>
- **Project Repository**: Department of Veterans Affairs - acoustic_emissions_processing

---

## Contact & Questions

For questions about the project structure, data models, or implementation details, refer to the existing code, docstrings, and tests. The codebase is self-documenting with comprehensive type hints and validation.
