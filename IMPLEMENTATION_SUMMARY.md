# QA/QC Versioning Implementation Summary

## Overview

This implementation establishes a comprehensive versioning system for Quality Assurance/Quality Control (QA/QC) methods used throughout the acoustic emissions processing pipeline. This allows tracking which version of QC each file passed, enabling proper handling of evolving QC methods over time.

## Problem Statement

The issue described the need to version-control QA/QC iterations because:
- QC methods will evolve and change over time
- Not all files will be reprocessed prior to analysis
- There needs to be metadata indicating which version of QC each file passed
- This metadata will ultimately be saved to a database

## Solution

### Three QC Stages Identified

Based on the agent instructions and code review, there are three distinct QC stages:

1. **Audio QC** - Performed on audio after processing .bin files
2. **Biomechanics QC** - Performed on biomechanics pre-synchronization with audio
3. **Cycle QC** - Performed on synchronized audio/biomechanics after parsing into individual movement cycles

### Implementation Components

#### 1. Centralized Version Management (`src/qc_versions.py`)

Created a single source of truth for all QC versions:

```python
# Version constants
AUDIO_QC_VERSION = 1
BIOMECH_QC_VERSION = 1
CYCLE_QC_VERSION = 1

# Getter functions
get_audio_qc_version() -> int
get_biomech_qc_version() -> int
get_cycle_qc_version() -> int
get_qc_version(qc_type: Literal["audio", "biomech", "cycle"]) -> int
```

#### 2. Model Updates (`src/models.py`)

Updated three model classes to use dynamic QC version defaults:

- `AcousticsFileMetadata`: Uses `Field(default_factory=get_audio_qc_version)`
- `BiomechanicsFileMetadata`: Uses `Field(default_factory=get_biomech_qc_version)`
- `MovementCycleMetadata`: Uses all three version getters

This ensures that any metadata created will automatically use the current QC version.

#### 3. Integration Points

Updated metadata creation in three locations:

1. **`src/audio/parsers.py`**: Audio metadata creation now explicitly uses `get_audio_qc_version()`
2. **`src/biomechanics/importers.py`**: Biomechanics metadata creation uses `get_biomech_qc_version()`
3. **`src/synchronization/quality_control.py`**: Cycle metadata creation uses all three version getters

### Workflow

#### Creating Metadata (Current State)

When metadata is created, it automatically includes the current QC version:

```python
# Example: Creating audio metadata
audio_metadata = AcousticsFileMetadata(
    # ... other fields ...
)
# audio_metadata.audio_qc_version is automatically set to current version (1)
```

#### Updating QC Methods (Future)

When QC methods are modified:

1. Update the QC implementation (e.g., in `src/audio/quality_control.py`)
2. Increment the version in `src/qc_versions.py`:
   ```python
   AUDIO_QC_VERSION = 2  # Increment from 1 to 2
   ```
3. Document the change in the version history docstring
4. All new metadata will automatically use version 2
5. Old data retains version 1 for proper analysis

#### Database Usage

The version fields are designed for database queries:

```sql
-- Get only data processed with latest QC
SELECT * FROM movement_cycles 
WHERE audio_qc_version = 2 
  AND biomech_qc_version = 1 
  AND cycle_qc_version = 1

-- Compare results across QC versions
SELECT audio_qc_version, AVG(cycle_acoustic_energy)
FROM movement_cycles
GROUP BY audio_qc_version
```

## Testing

Comprehensive test coverage with 20 new tests:

### Unit Tests (`tests/test_qc_versions.py`) - 14 tests
- Version constants are properly defined
- Getter functions return correct values
- Version consistency across calls
- Error handling for invalid QC types

### Integration Tests (`tests/test_qc_version_integration.py`) - 6 tests
- Audio metadata includes correct version
- Biomechanics metadata includes correct version
- Cycle metadata includes all three versions
- Versions are included in serialization
- Default values work correctly when not explicitly specified

### Regression Testing
- All 331 existing tests pass
- No breaking changes to existing functionality

## Documentation

Created comprehensive documentation:

### User Documentation (`docs/QC_VERSIONING.md`)
- Overview of the versioning system
- Usage examples
- Step-by-step guide for updating versions
- Best practices
- Example: implementing a new QC version

### Updated README
- Added reference to QC versioning documentation
- Integrated into existing data models documentation

## Benefits

1. **Traceability**: Every file has a clear record of which QC version it passed
2. **Backward Compatibility**: Old data remains valid with its original QC version
3. **Flexible Reprocessing**: Can selectively reprocess files that used older QC
4. **Analysis Accuracy**: Can filter or compare results based on QC version
5. **Easy Updates**: Single location to increment versions when QC methods change
6. **Database Ready**: Fields are designed for database storage and querying

## Future Enhancements

Potential future improvements:
- Add version history tracking in database
- Create CLI tool to list files by QC version
- Implement automated reprocessing for older versions
- Add version compatibility checks
- Create migration utilities for major QC changes

## Files Changed

- **New Files**:
  - `src/qc_versions.py` - Version management module
  - `tests/test_qc_versions.py` - Unit tests
  - `tests/test_qc_version_integration.py` - Integration tests
  - `docs/QC_VERSIONING.md` - User documentation

- **Modified Files**:
  - `src/models.py` - Updated to use default_factory for versions
  - `src/audio/parsers.py` - Uses get_audio_qc_version()
  - `src/biomechanics/importers.py` - Uses get_biomech_qc_version()
  - `src/synchronization/quality_control.py` - Uses all version getters
  - `README.md` - Added reference to versioning docs

## Test Results

✅ 331 tests passing (311 existing + 20 new)
- 14 unit tests for qc_versions module
- 6 integration tests for version tracking
- All existing tests pass with no regressions
