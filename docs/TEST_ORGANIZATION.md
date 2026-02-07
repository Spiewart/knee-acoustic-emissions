# Test Organization Strategy

## Overview

This document defines the organizational structure for the test suite (`tests/` directory). Tests are grouped by functional domain with nested subdirectories for related concerns. This structure improves discoverability, reduces file clutter, and makes it clear where new tests should be added.

## Directory Structure

```
tests/
├── conftest.py                          # Shared fixtures (MANDATORY factories)
│
├── unit/                                # Unit tests for individual modules
│   ├── audio/                           # Audio processing tests
│   │   ├── test_readers.py              # .bin file reading (read_audio_board_file)
│   │   ├── test_parsers.py              # Audio metadata parsing (parse_acoustic_file_legend)
│   │   ├── test_exporters.py            # CSV/channel export functionality
│   │   ├── test_analysis.py             # Spectrogram, instantaneous frequency
│   │   └── test_qc/                     # Audio quality control
│   │       ├── test_raw_qc.py           # Raw signal QC (dropout, artifacts)
│   │       ├── test_maneuver_qc.py      # Maneuver-specific QC (walk, fe, sts)
│   │       ├── test_per_mic_qc.py       # Per-microphone QC checks
│   │       └── test_qc_thresholds.py    # QC threshold validation
│   │
│   ├── biomechanics/                    # Biomechanics processing tests
│   │   ├── test_importers.py            # Excel import (import_biomechanics_recordings)
│   │   ├── test_metadata_parsing.py     # UID parsing, maneuver detection
│   │   ├── test_normalization.py        # Maneuver normalization (walk vs non-walk)
│   │   ├── test_cycle_parsing.py        # Movement cycle extraction
│   │   └── test_cycle_boundaries.py     # Cycle start/end detection
│   │
│   ├── synchronization/                 # Audio-biomechanics sync tests
│   │   ├── test_stomp_detection.py      # Foot stomp detection (single + dual-knee)
│   │   ├── test_sync_methods.py         # Sync algorithm (consensus, RMS, guided)
│   │   ├── test_sync_qc.py              # Synchronization quality control
│   │   ├── test_clipping.py             # Time clipping for maneuvers
│   │   └── test_deduplication.py        # Duplicate cycle prevention
│   │
│   ├── metadata/                        # Metadata and validation tests
│   │   ├── test_models.py               # Pydantic models (AcousticsFileMetadata, etc.)
│   │   ├── test_validators.py           # Field validators (conditional validation)
│   │   ├── test_dataframe_validation.py # DataFrame column/type validation
│   │   └── test_whitespace_handling.py  # Edge cases: whitespace in UIDs/events
│   │
│   ├── database/                        # Database layer tests
│   │   ├── test_models.py               # SQLAlchemy models (ORM)
│   │   ├── test_repository.py           # Repository CRUD operations
│   │   ├── test_persistence.py          # Record persistence workflows
│   │   └── test_migrations.py           # Schema migration tests
│   │
│   ├── processing_log/                  # Processing log tests
│   │   ├── test_excel_io.py             # Excel read/write operations
│   │   ├── test_record_creation.py      # Record factory functions
│   │   ├── test_sheet_integrity.py      # Sheet structure validation
│   │   ├── test_summary_generation.py   # Processing log summaries
│   │   └── test_edge_cases.py           # Missing sheets, empty records, etc.
│   │
│   ├── qc_versioning/                   # QC version tracking
│   │   ├── test_version_models.py       # QC version metadata
│   │   ├── test_version_tracking.py     # Version persistence
│   │   └── test_version_reports.py      # Report generation with versions
│   │
│   ├── visualization/                   # Plotting and visualization
│   │   ├── test_plots.py                # Waveform/spectrogram plotting
│   │   ├── test_sync_plots.py           # Synchronized data visualization
│   │   └── test_cycle_plots.py          # Movement cycle plots
│   │
│   └── cli/                             # CLI command tests
│       ├── test_bin_commands.py         # ae-read-audio, ae-dump-channels, etc.
│       ├── test_pipeline_commands.py    # ae-process-directory entrypoints
│       ├── test_qc_commands.py          # ae-audio-qc, ae-sync-qc
│       ├── test_ml_commands.py          # ae-ml-* commands
│       └── test_cleanup_commands.py     # ae-cleanup-outputs
│
├── integration/                         # Integration tests (multi-module)
│   ├── test_bin_to_sync_pipeline.py     # bin → audio → sync workflow
│   ├── test_sync_to_cycles_pipeline.py  # sync → cycles → QC workflow
│   ├── test_full_pipeline.py            # End-to-end: bin → cycles
│   ├── test_participant_processing.py   # process_participant_directory
│   ├── test_database_workflows.py       # Multi-stage DB persistence
│   ├── test_metadata_integration.py     # Metadata flow through pipeline
│   └── test_entrypoint_filtering.py     # Entrypoint-specific processing
│
├── regression/                          # Regression tests (prevent past bugs)
│   ├── test_session_6_regression.py     # Session 6-specific issues
│   ├── test_phase_2_regression.py       # Phase 2 implementation regressions
│   ├── test_phase_2d_performance.py     # Performance regression tests
│   └── test_whitespace_regressions.py   # Whitespace handling bugs
│
├── edge_cases/                          # Edge case tests (boundary conditions)
│   ├── test_missing_files.py            # Missing .bin, Excel, sync files
│   ├── test_corrupt_data.py             # Malformed pickles, invalid Excel
│   ├── test_empty_recordings.py         # Zero-length audio/biomechanics
│   ├── test_boundary_conditions.py      # Min/max values, extreme inputs
│   └── test_special_characters.py       # Unicode, special chars in filenames
│
└── performance/                         # Performance benchmarks
    ├── test_bin_stage_optimization.py   # .bin processing speed
    ├── test_sync_performance.py         # Sync algorithm performance
    └── test_large_datasets.py           # Scalability tests
```

## File Naming Conventions

### Test File Names
- **Unit tests**: `test_<module_name>.py` (e.g., `test_readers.py`)
- **Integration tests**: `test_<workflow_name>_pipeline.py` (e.g., `test_bin_to_sync_pipeline.py`)
- **Regression tests**: `test_<issue_name>_regression.py` (e.g., `test_session_6_regression.py`)
- **Edge case tests**: `test_<scenario>.py` (e.g., `test_missing_files.py`)

### Test Function Names
- Follow pattern: `test_<function>_<scenario>_<expected_outcome>`
- Examples:
  - `test_read_audio_board_file_valid_bin_returns_dataframe()`
  - `test_qc_audio_walk_insufficient_passes_fails()`
  - `test_sync_audio_with_biomechanics_dual_knee_disambiguates_stomps()`

## Test Categories

### 1. Unit Tests (`unit/`)
Tests for individual functions/classes in isolation. Mock external dependencies.

**Characteristics**:
- Fast execution (< 0.1s per test)
- No file I/O (use fixtures/mocks)
- Single responsibility (one function/class)
- High coverage of edge cases

**Example**:
```python
# tests/unit/audio/test_readers.py
def test_read_audio_board_file_invalid_firmware_warns_and_defaults():
    """Unknown firmware should warn but continue with defaults."""
    # Test implementation
```

### 2. Integration Tests (`integration/`)
Tests for multi-module workflows. May use real files (small test fixtures).

**Characteristics**:
- Moderate execution time (< 1s per test)
- May use temp directories and small test files
- Test module interactions
- Focus on data flow between components

**Example**:
```python
# tests/integration/test_bin_to_sync_pipeline.py
def test_bin_to_sync_workflow_produces_synced_pickle(tmp_path):
    """End-to-end: .bin → audio → sync → cycles."""
    # Test implementation
```

### 3. Regression Tests (`regression/`)
Tests that prevent specific past bugs from reoccurring.

**Characteristics**:
- Document the original issue (bug report, phase)
- Include failing case before fix
- Verify fix prevents recurrence
- Link to documentation (e.g., changelog)

**Example**:
```python
# tests/regression/test_session_6_regression.py
def test_session_6_sync_stomp_detection_bug():
    """Regression test for Session 6 stomp detection failure.

    Original issue: Biomechanics stomp events contained extra whitespace,
    causing sync to fail. Fixed in Phase X.

    See: docs/changelog/bugfix/20260115_whitespace_handling.md
    """
    # Test implementation
```

### 4. Edge Case Tests (`edge_cases/`)
Tests for boundary conditions, invalid inputs, and error handling.

**Characteristics**:
- Test error paths (exceptions, validation errors)
- Extreme values (empty, very large, negative)
- Invalid file formats
- Missing required data

**Example**:
```python
# tests/edge_cases/test_missing_files.py
def test_read_audio_board_file_missing_bin_raises_file_not_found():
    """Missing .bin file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        read_audio_board_file("nonexistent.bin")
```

### 5. Performance Tests (`performance/`)
Tests for execution speed and scalability.

**Characteristics**:
- Measure execution time
- Test with large datasets
- Identify bottlenecks
- Mark with `@pytest.mark.slow` for optional execution

**Example**:
```python
# tests/performance/test_bin_stage_optimization.py
@pytest.mark.slow
def test_read_audio_board_file_performance_large_file():
    """Should process 500MB .bin file in < 5 seconds."""
    start = time.time()
    read_audio_board_file("large_test_file.bin")
    assert time.time() - start < 5.0
```

## Fixture Organization

### conftest.py Structure

```python
# tests/conftest.py

# ============================================================================
# MANDATORY FIXTURE FACTORIES
# ============================================================================
# See docs/TESTING_GUIDELINES.md for usage instructions
# DO NOT create test data manually in test files!

# Section 1: Metadata Factories
# - synchronization_factory()
# - synchronization_metadata_factory()
# - audio_processing_factory()
# - biomechanics_import_factory()
# - movement_cycle_factory()

# Section 2: File/Path Fixtures
# - tmp_path (pytest built-in)
# - test_data_dir (fixtures for small test files)

# Section 3: Database Fixtures
# - test_db_engine (PostgreSQL test connection)
# - test_db_session (scoped session)
# - test_repository (repository with test DB)

# Section 4: Mock Data Fixtures
# - mock_audio_dataframe()
# - mock_biomechanics_dataframe()
# - mock_synced_dataframe()
```

### Nested conftest.py Files

Each subdirectory can have its own `conftest.py` for domain-specific fixtures:

```python
# tests/unit/audio/conftest.py
@pytest.fixture
def mock_bin_file(tmp_path):
    """Create a small valid .bin file for testing."""
    # Implementation
```

## Migration Guide

### Current State → New Structure

#### Step 1: Create Directory Structure
```bash
cd tests/
mkdir -p unit/{audio/test_qc,biomechanics,synchronization,metadata,database,processing_log,qc_versioning,visualization,cli}
mkdir -p integration regression edge_cases performance
```

#### Step 2: Move Files to New Locations

**Audio Tests** (`unit/audio/`):
```bash
# Readers
test_excel_file_finding.py → unit/audio/test_parsers.py (combine)
test_parse_acoustic_file_legend.py → unit/audio/test_parsers.py (combine)

# Exporters
test_audio_exporters.py → unit/audio/test_exporters.py
test_audio_exporters_extended.py → unit/audio/test_exporters.py (merge)

# Analysis
test_instantaneous_frequency_unit.py → unit/audio/test_analysis.py (combine)
test_instantaneous_frequency_extended.py → unit/audio/test_analysis.py (combine)
test_spectrogram_units.py → unit/audio/test_analysis.py (combine)
test_spectrogram_units_extended.py → unit/audio/test_analysis.py (combine)

# QC
test_raw_audio_qc.py → unit/audio/test_qc/test_raw_qc.py
test_audio_qc.py → unit/audio/test_qc/test_maneuver_qc.py
test_per_mic_audio_qc.py → unit/audio/test_qc/test_per_mic_qc.py
test_audio_qc_fields_persistence.py → unit/audio/test_qc/test_qc_thresholds.py (merge)
```

**Biomechanics Tests** (`unit/biomechanics/`):
```bash
test_process_biomechanics.py → unit/biomechanics/test_importers.py
test_characteristics_importers.py → unit/biomechanics/test_importers.py (merge)
test_biomechanics_metadata_integration.py → unit/biomechanics/test_metadata_parsing.py
test_maneuver_normalization.py → unit/biomechanics/test_normalization.py
test_maneuver_normalization_and_validation.py → unit/biomechanics/test_normalization.py (merge)
test_parse_movement_cycles.py → unit/biomechanics/test_cycle_parsing.py
test_movement_cycles_boundaries.py → unit/biomechanics/test_cycle_boundaries.py
test_cycle_details_regeneration.py → unit/biomechanics/test_cycle_parsing.py (merge)
```

**Synchronization Tests** (`unit/synchronization/`):
```bash
test_sync_audio_with_biomechanics.py → unit/synchronization/test_sync_methods.py
test_stomp_detection_methods.py → unit/synchronization/test_stomp_detection.py
test_biomech_guided_stomp_detection.py → unit/synchronization/test_stomp_detection.py (merge)
test_extract_stomp_time_whitespace.py → unit/synchronization/test_stomp_detection.py (merge)
test_aligned_stomp_logging.py → unit/synchronization/test_stomp_detection.py (merge)
test_sync_qc.py → unit/synchronization/test_sync_qc.py
test_sync_qc_audio_integration.py → unit/synchronization/test_sync_qc.py (merge)
test_sync_clipping.py → unit/synchronization/test_clipping.py
test_sync_deduplication_and_cycle_persistence.py → unit/synchronization/test_deduplication.py
test_sync_record_fields.py → unit/synchronization/test_sync_methods.py (merge)
```

**Metadata Tests** (`unit/metadata/`):
```bash
test_db_models.py → unit/metadata/test_models.py
test_analysis_models.py → unit/metadata/test_models.py (merge)
test_movement_cycle_models.py → unit/metadata/test_models.py (merge)
test_metadata_validators.py → unit/metadata/test_validators.py
test_dataframe_validation.py → unit/metadata/test_dataframe_validation.py
test_event_metadata_whitespace.py → unit/metadata/test_whitespace_handling.py
```

**Database Tests** (`unit/database/`):
```bash
test_database.py → unit/database/test_repository.py
test_orchestration_database_persistence.py → unit/database/test_persistence.py
```

**Processing Log Tests** (`unit/processing_log/`):
```bash
test_processing_log.py → unit/processing_log/test_excel_io.py
test_processing_log_helpers.py → unit/processing_log/test_record_creation.py
test_processing_log_standalone.py → unit/processing_log/test_excel_io.py (merge)
test_processing_log_sheet_integrity.py → unit/processing_log/test_sheet_integrity.py
test_processing_log_summary.py → unit/processing_log/test_summary_generation.py
test_processing_log_edge_cases.py → unit/processing_log/test_edge_cases.py
test_processing_log_integration.py → integration/test_database_workflows.py (move)
test_audio_sheet_data_population.py → unit/processing_log/test_excel_io.py (merge)
test_biomechanics_processing_status.py → unit/processing_log/test_record_creation.py (merge)
test_knee_level_logging.py → unit/processing_log/test_excel_io.py (merge)
```

**QC Versioning Tests** (`unit/qc_versioning/`):
```bash
test_qc_versions.py → unit/qc_versioning/test_version_models.py
test_qc_version_integration.py → unit/qc_versioning/test_version_tracking.py
test_qc_metadata.py → unit/qc_versioning/test_version_models.py (merge)
test_qc_report_integrity.py → unit/qc_versioning/test_version_reports.py
```

**Visualization Tests** (`unit/visualization/`):
```bash
test_visualizations.py → unit/visualization/test_plots.py
test_cycle_qc.py → unit/visualization/test_cycle_plots.py (if plotting-focused)
```

**CLI Tests** (`unit/cli/`):
```bash
test_cli.py → unit/cli/test_bin_commands.py
test_cli_ml_commands.py → unit/cli/test_ml_commands.py
test_ml_cli_helpers.py → unit/cli/test_ml_commands.py (merge)
test_cleanup_outputs.py → unit/cli/test_cleanup_commands.py
```

**Integration Tests** (`integration/`):
```bash
test_full_pipeline_integration.py → integration/test_full_pipeline.py
test_fake_directory_processing_integration.py → integration/test_participant_processing.py
test_participant_processor.py → integration/test_participant_processing.py (merge)
test_process_participant_directory.py → integration/test_participant_processing.py (merge)
test_run_process_participant_directory.py → integration/test_participant_processing.py (merge)
test_persistent_processor.py → integration/test_participant_processing.py (merge)
test_entrypoint_processing.py → integration/test_entrypoint_filtering.py
test_filtered_processing.py → integration/test_entrypoint_filtering.py (merge)
test_metadata_integration.py → integration/test_metadata_integration.py
```

**Regression Tests** (`regression/`):
```bash
test_session_6_integration.py → regression/test_session_6_regression.py
test_phase_2_implementation.py → regression/test_phase_2_regression.py
test_phase_2d_performance.py → regression/test_phase_2d_performance.py
test_energy_ratio_validation.py → regression/test_energy_ratio_regression.py (if fixing a bug)
```

**Edge Case Tests** (`edge_cases/`):
```bash
# Create new files combining edge cases from existing tests
# Look for tests with "missing", "empty", "invalid", "error" in names
```

**Performance Tests** (`performance/`):
```bash
test_bin_stage_optimization.py → performance/test_bin_stage_optimization.py
```

**Keep at Root** (special purpose):
```bash
test_smoke.py → tests/test_smoke.py (smoke tests run first)
conftest.py → tests/conftest.py (shared fixtures)
```

#### Step 3: Update Imports

When moving tests, update import paths:
```python
# Before
from src.audio.readers import read_audio_board_file

# After (if using relative imports)
from src.audio.readers import read_audio_board_file  # No change needed
```

#### Step 4: Combine Related Tests

When merging files, group tests by the function/class they test:

```python
# unit/audio/test_analysis.py (combined from 4 files)

# ============================================================================
# Instantaneous Frequency Tests
# ============================================================================
class TestInstantaneousFrequency:
    """Tests for Hilbert transform instantaneous frequency computation."""

    def test_compute_instantaneous_frequency_basic(self):
        """Unit test: basic frequency computation."""
        # From test_instantaneous_frequency_unit.py

    def test_compute_instantaneous_frequency_edge_cases(self):
        """Extended test: boundary conditions."""
        # From test_instantaneous_frequency_extended.py

# ============================================================================
# Spectrogram Tests
# ============================================================================
class TestSpectrogram:
    """Tests for STFT spectrogram computation."""

    def test_compute_spectrogram_basic(self):
        """Unit test: basic spectrogram."""
        # From test_spectrogram_units.py

    def test_compute_spectrogram_large_data(self):
        """Extended test: performance with large arrays."""
        # From test_spectrogram_units_extended.py
```

### Migration Timeline

1. **Phase 1** (Week 1): Create directory structure, move audio/biomechanics tests
2. **Phase 2** (Week 2): Move synchronization/metadata tests, combine duplicates
3. **Phase 3** (Week 3): Move database/processing_log tests, update imports
4. **Phase 4** (Week 4): Reorganize integration/regression tests, verify all pass
5. **Phase 5** (Week 5): Update documentation, remove old empty directories

## Testing Checklist for New Features

When adding a new feature, create tests in the following order:

### 1. Unit Tests (REQUIRED)
- [ ] Happy path: basic functionality works
- [ ] Edge cases: boundary conditions, empty inputs, extreme values
- [ ] Error cases: invalid inputs raise appropriate exceptions
- [ ] Validation: Pydantic models validate fields correctly

### 2. Integration Tests (if multi-module)
- [ ] Data flows correctly between modules
- [ ] Temp files created and cleaned up
- [ ] Database persistence works end-to-end
- [ ] CLI commands work with real files

### 3. Regression Tests (if fixing a bug)
- [ ] Original failing case now passes
- [ ] Document the bug in test docstring
- [ ] Link to changelog or bug report

### 4. Documentation (REQUIRED)
- [ ] Update this file if adding new test category
- [ ] Update ai_instructions.md with test placement rules
- [ ] Add examples to TESTING_GUIDELINES.md if new pattern

## Missing Test Coverage (TODO)

Based on current analysis, the following areas need additional tests:

### Edge Cases (High Priority)
- [ ] **Missing .bin files**: Test all CLI commands with missing inputs
- [ ] **Corrupt pickle files**: Test loading malformed/truncated pickles
- [ ] **Empty recordings**: Zero-length audio, no movement cycles
- [ ] **Unicode filenames**: Special characters in participant IDs, file names
- [ ] **Disk full errors**: Test behavior when unable to write outputs
- [ ] **Network issues**: Test PostgreSQL connection failures (timeout, refused)

### Error Handling (Medium Priority)
- [ ] **Excel validation**: Missing sheets, wrong column names, invalid data types
- [ ] **Time synchronization**: Stomp events outside recording duration
- [ ] **QC thresholds**: Invalid threshold values (negative, NaN, infinity)
- [ ] **Concurrent access**: Multiple processes writing same processing log

### Performance (Low Priority)
- [ ] **Large datasets**: Test with 1GB+ .bin files, 1000+ movement cycles
- [ ] **Memory usage**: Profile memory consumption on large pipelines
- [ ] **Parallel processing**: Test concurrent participant processing

### Regression Prevention (Ongoing)
- [ ] **Phase 5 time field refactoring**: Ensure all time fields are float, not timedelta
- [ ] **Dual-knee disambiguation**: Test stomp detection with both knees present
- [ ] **Whitespace handling**: Test UIDs with leading/trailing whitespace

## pytest Configuration

### Running Tests by Category

```bash
# Run all tests
pytest tests/ -v

# Run only unit tests
pytest tests/unit/ -v

# Run only integration tests
pytest tests/integration/ -v

# Run only fast tests (exclude performance)
pytest tests/ -v -m "not slow"

# Run specific module
pytest tests/unit/audio/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### pytest.ini Configuration

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Markers for test categories
markers =
    unit: Unit tests (fast, isolated)
    integration: Integration tests (multi-module)
    regression: Regression tests (prevent past bugs)
    edge_case: Edge case tests (boundary conditions)
    slow: Slow tests (> 1s, often marked for optional execution)
    requires_db: Tests requiring PostgreSQL
    requires_files: Tests requiring test data files

# Timeout for slow tests
timeout = 60

# Show local variables on failure
showlocals = true

# Verbose output
addopts = -ra --strict-markers
```

## AI Assistant Guidelines

When creating new tests, AI assistants should:

1. **Determine test category** based on what's being tested:
   - Single function → `unit/`
   - Multi-module workflow → `integration/`
   - Past bug → `regression/`
   - Error handling → `edge_cases/`

2. **Place in correct subdirectory**:
   - Audio processing → `unit/audio/`
   - Biomechanics → `unit/biomechanics/`
   - Synchronization → `unit/synchronization/`
   - Database → `unit/database/`
   - Processing logs → `unit/processing_log/`

3. **Use fixture factories** (MANDATORY):
   - `synchronization_factory()`
   - `audio_processing_factory()`
   - `biomechanics_import_factory()`
   - `movement_cycle_factory()`

4. **Follow naming conventions**:
   - File: `test_<module>.py`
   - Function: `test_<function>_<scenario>_<outcome>()`
   - Class: `TestClassName` (if grouping related tests)

5. **Document test purpose** in docstring:
   - What is being tested
   - Why this test exists (especially for regressions)
   - Link to relevant documentation

6. **Update this document** if creating a new test category or pattern.

## Summary

This organization provides:
- **Discoverability**: Find tests by domain (audio, biomechanics, sync)
- **Maintainability**: Related tests grouped together
- **Scalability**: Clear structure for adding new tests
- **Documentation**: Self-documenting directory structure

When in doubt about test placement, ask:
1. Does it test one function/class? → `unit/`
2. Does it test multiple modules? → `integration/`
3. Does it prevent a past bug? → `regression/`
4. Does it test error handling? → `edge_cases/`
5. Does it measure performance? → `performance/`

For questions or suggestions, update this document and notify the team.
