# AI Assistant Instructions for Acoustic Emissions Processing

## Project Overview

This project processes acoustic emissions data from knee joint recordings during various biomechanical maneuvers (walking, sit-to-stand, flexion-extension). It integrates audio data with biomechanics motion capture data from participants in a research study.

### Key Components

1. **Audio Processing**: Converts `.bin` files from audio boards to pandas DataFrames
2. **Biomechanics Data**: Reads and processes motion capture data from Excel files
3. **Data Synchronization**: Aligns audio and biomechanics data using foot stomp events (supports dual-knee recordings)
4. **Analysis**: Computes spectrograms, instantaneous frequency, and per-channel metrics

---

## Data Validation Architecture

**CRITICAL: Unified Pydantic Dataclasses as Single Source of Truth**

This project uses unified Pydantic dataclasses (`src/metadata.py`) that combine validation and Excel export in a single definition.

### Unified Pydantic Dataclasses (`src/metadata.py`)

**Purpose**: Single-class definitions that provide both Pydantic validation AND Excel export functionality. These are the single source of truth for all processing metadata.

**Location**: `src/metadata.py`

**Key Classes**:
- `AudioProcessing`: Audio file processing and QC metadata (replaces AudioProcessingMetadata + AudioProcessingRecord)
- `BiomechanicsImport`: Biomechanics data import tracking (replaces BiomechanicsImportMetadata + BiomechanicsImportRecord)
- `Synchronization`: Audio-biomechanics synchronization details (replaces SynchronizationMetadata + SynchronizationRecord)
- `MovementCycles`: Movement cycle extraction and QC metadata (replaces MovementCyclesMetadata + MovementCyclesRecord)
- `MovementCycle`: Individual movement cycle with embedded upstream processing info

**Field Guidelines**:
- ✅ **Include**: File names, processing status, QC parameters, timestamps, recording characteristics (sample rate, duration), QC version tracking
- ✅ **Include**: Data-derived statistics (channel RMS/peak values, per-sample counts) - now part of the unified class
- ❌ **Exclude**: Large data objects, complex nested structures that can't be exported to Excel

**Field Naming**: All fields use `snake_case` for direct mapping to database columns and Excel headers.

**Creation Pattern**:
```python
# Direct instantiation with validation
audio = AudioProcessing(
    audio_file_name="recording.bin",
    processing_status="success",
    sample_rate=46875.0,  # Validated by Pydantic
    duration_seconds=120.5,
    channel_1_rms=150.3,  # Data-derived field, included in same class
    channel_2_rms=145.8,
)

# Export to Excel (built-in to_dict method)
excel_dict = audio.to_dict()
```

### Adding New Fields: SIMPLIFIED WORKFLOW

**When adding fields to metadata, you only update ONE location:**

1. **Add to Pydantic dataclass** (`src/metadata.py`):
   ```python
   @dataclass
   class AudioProcessing:
       # Add new field
       new_qc_parameter: Optional[float] = None
       
       # Add validator if needed
       @field_validator("new_qc_parameter")
       @classmethod
       def validate_new_parameter(cls, value: Optional[float]) -> Optional[float]:
           if value is not None and value < 0:
               raise ValueError("new_qc_parameter must be non-negative")
           return value
   ```

2. **Update to_dict() for Excel export**:
   ```python
   def to_dict(self) -> Dict[str, Any]:
       return {
           "New QC Parameter": self.new_qc_parameter,
           # ... other fields
       }
   ```

3. **Update helper functions** (`src/orchestration/processing_log.py`):
   ```python
   def create_audio_record_from_data(data: dict) -> AudioProcessing:
       return AudioProcessing(
           new_qc_parameter=data.get("new_qc_parameter"),
           # ... other fields
       )
   ```

4. **Update load_from_excel()** to read from Excel.

### File Metadata Classes (`src/models.py`)

**Purpose**: Base classes for file-level metadata used during synchronization workflows.

**Location**: `src/models.py`

**Key Classes**:
- `AcousticsFileMetadata`: Audio file metadata with microphone positions
- `BiomechanicsFileMetadata`: Biomechanics file metadata with system info
- `FullMovementCycleMetadata`: Complete cycle metadata with file metadata inheritance (for sync QC and data processing workflows)
- `SynchronizedRecording`: Combined acoustics + biomechanics with data
- `MovementCycle`: Movement cycle with synchronized data field (for processing workflows)

**Note**: These BaseModel classes are separate from the unified Pydantic dataclasses in `metadata.py`. They are used for:
1. Synchronization QC workflows that need inheritance from multiple file metadata classes
2. Carrying actual data (DataFrame) objects during processing (not suitable for Excel export)

### Key Differences

**Old Architecture (deprecated)**:
- Two separate classes: Pydantic BaseModel for validation + Python dataclass for Excel export
- Complex `from_metadata()` pattern
- Fields split across two definitions
- Easy to get out of sync

**New Architecture (current)**:
- Single Pydantic @dataclass combining validation + export
- Direct instantiation
- All fields in one place
- Single source of truth
   def to_dict(self):
       return {
           "Ch5 RMS": self.channel_5_rms,
           # ... other fields
       }
   
   # C. Update helper functions to calculate and set the field
   # D. Update load_from_excel() to read and set directly on record
   ```

### Helper Functions

**Location**: `src/orchestration/processing_log.py`

Helper functions create records with proper validation:
- `create_audio_record_from_data()`: Calculates duration/stats → validates → creates record → sets channel stats
- `create_biomechanics_record_from_data()`: Similar pattern
- `create_sync_record_from_data()`: Similar pattern
- `create_cycles_record_from_data()`: Similar pattern

**Pattern**: All helpers build a data dict → validate through Pydantic → create record from metadata → set data-derived fields separately.

### Excel Loading

Excel files contain BOTH metadata and data-derived fields. The `load_from_excel()` function:
1. Reads all fields from Excel
2. Separates metadata fields from data-derived fields
3. Creates Pydantic metadata model with metadata fields only
4. Creates record from validated metadata
5. Sets data-derived fields directly on record

### Testing Guidelines

When writing tests:
- Create Pydantic metadata models first (validates fields)
- Use `Record.from_metadata()` to create records
- Set data-derived fields separately if needed
- Never create records without validated metadata

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
- **`AcousticsFileMetadata`**: Audio recording metadata with conditional validation
  - `scripted_maneuver`: "walk", "sit_to_stand", or "flexion_extension"
  - `speed`: "slow", "medium", or "fast" (only for walk)
  - `knee`: "left" or "right"
  - Microphone dictionary with keys 1-4

- **`BiomechanicsFileMetadata`**: Motion capture recording metadata with **conditional validation rules**:
  - **When `maneuver="walk"`**: `pass_number` (int, required) and `speed` (required, one of "slow", "normal", "fast")
  - **When `maneuver="sit_to_stand"` or `"flexion_extension"`**: `pass_number` and `speed` must be `None`

- **`BiomechanicsRecording`**: Biomechanics data + metadata
- **`AcousticsRecording`**: Audio data + metadata
- **`SynchronizedRecording`**: Combined audio + biomechanics data

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

### BiomechanicsFileMetadata Validation

The `BiomechanicsFileMetadata` class enforces conditional validation:

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

**Audio Data (AcousticsData)**:

- `tt`: time vector
- `ch1`, `ch2`, `ch3`, `ch4`: audio channels
- `f_ch1`, `f_ch2`, `f_ch3`, `f_ch4`: instantaneous frequency

**Biomechanics Data (BiomechanicsData)**:

- `TIME`: time vector (timedelta, relative to start)
- Various motion capture columns (joint angles, positions, etc.)

**Synchronized Data (SynchronizedData)**:

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
2. Verify conditional validation in BiomechanicsFileMetadata (maneuver → speed/pass_number rules)
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
| Pydantic validation errors | Check field types and conditional rules in `BiomechanicsFileMetadata` |
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

## Change Documentation & Changelog

### Overview
All change documentation should be organized in `/docs/changelog/` with type-specific subfolders rather than cluttering the root directory.

### Folder Structure
```
docs/changelog/
├── patch/           # Bug fixes, minor improvements
├── feature/         # New features, significant additions
├── bugfix/          # Critical bug fixes
├── refactor/        # Code refactoring without behavior changes
└── README.md        # Detailed instructions and index
```

### Naming Convention
Use format: `YYYYMMDD_descriptive_name.md`

Example: `20260115_entrypoint_filter_processing.md`

### For AI Assistants
When creating change documentation:
1. **Always** save to `/docs/changelog/{type}/` subdirectories
2. **Never** save to root directory
3. Include: problem, solution, affected files, test results, backward compatibility
4. Use precise line numbers and file paths
5. Make documents self-contained for future reference

### For Users
- Review changes in `/docs/changelog/` organized by type and date
- Check `docs/changelog/README.md` for detailed guidelines
- Filter by change type based on your interest

**Full details**: See `/docs/changelog/README.md`

---

## Contact & Questions

For questions about the project structure, data models, or implementation details, refer to the existing code, docstrings, and tests. The codebase is self-documenting with comprehensive type hints and validation.
