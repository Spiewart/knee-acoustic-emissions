# QC Versioning Guide

## Overview

The acoustic emissions processing codebase includes a versioning system for Quality Control (QC) methods. As QC methods evolve and improve over time, version tracking ensures that metadata correctly indicates which version of QC each file passed.

## QC Types

There are three types of QC, each with its own version:

1. **Audio QC** - Applied to audio after processing .bin files
2. **Biomechanics QC** - Applied to biomechanics data pre-synchronization
3. **Cycle QC** - Applied to synchronized audio/biomechanics after parsing into individual movement cycles

## Using QC Versions

### Getting Current Versions

```python
from src.qc_versions import (
    get_audio_qc_version,
    get_biomech_qc_version,
    get_cycle_qc_version,
    get_qc_version,
)

# Get specific QC version
audio_version = get_audio_qc_version()  # Returns 1
biomech_version = get_biomech_qc_version()  # Returns 1
cycle_version = get_cycle_qc_version()  # Returns 1

# Or use the generic function
audio_version = get_qc_version("audio")  # Returns 1
biomech_version = get_qc_version("biomech")  # Returns 1
cycle_version = get_qc_version("cycle")  # Returns 1
```

### Automatic Version Tracking

The QC versions are automatically included when creating metadata objects:

```python
from src.models import AcousticsFileMetadata, BiomechanicsFileMetadata, MovementCycleMetadata

# Audio metadata automatically includes audio_qc_version
audio_metadata = AcousticsFileMetadata(
    # ... other fields ...
)
print(audio_metadata.audio_qc_version)  # Will be current version (1)

# Biomechanics metadata automatically includes biomech_qc_version
biomech_metadata = BiomechanicsFileMetadata(
    # ... other fields ...
)
print(biomech_metadata.biomech_qc_version)  # Will be current version (1)

# Cycle metadata automatically includes all QC versions
cycle_metadata = MovementCycleMetadata(
    # ... other fields ...
)
print(cycle_metadata.audio_qc_version)  # Current audio QC version
print(cycle_metadata.biomech_qc_version)  # Current biomech QC version
print(cycle_metadata.cycle_qc_version)  # Current cycle QC version
```

## Updating QC Versions

When you modify QC methods, you should increment the appropriate version constant in `src/qc_versions.py`:

```python
# Before making QC method changes
AUDIO_QC_VERSION = 1
BIOMECH_QC_VERSION = 1
CYCLE_QC_VERSION = 1

# After improving audio QC method
AUDIO_QC_VERSION = 2  # Increment this
BIOMECH_QC_VERSION = 1
CYCLE_QC_VERSION = 1
```

### Version History

Document significant changes to QC methods in the docstring of `src/qc_versions.py`:

```python
"""QC Version Management

Version History:
    Audio QC v1: Initial implementation with periodic audio detection
    Audio QC v2: Added bandpower ratio filtering (2026-01-15)

    Biomechanics QC v1: Basic biomechanics validation

    Cycle QC v1: Initial cycle-level acoustic energy thresholding
"""
```

## Database Usage

The QC version fields in metadata are designed to be saved to a database. When querying data:

```python
# Example database query to filter by QC version
# SELECT * FROM movement_cycles
# WHERE audio_qc_version = 2
#   AND biomech_qc_version = 1
#   AND cycle_qc_version = 1
```

This allows analysis to:
- Use only data processed with specific QC versions
- Compare results across different QC method iterations
- Reprocess only files that used older QC versions

## Best Practices

1. **Increment versions** whenever you change QC logic that could affect pass/fail decisions
2. **Document changes** in the version history docstring
3. **Don't decrement** versions - always increment forward
4. **Coordinate updates** if multiple QC types are modified together
5. **Reprocess data** when critical QC improvements are made to ensure consistency

## Example: Implementing a New QC Version

1. Modify QC method in appropriate module (e.g., `src/audio/quality_control.py`)
2. Increment version constant in `src/qc_versions.py`:
   ```python
   AUDIO_QC_VERSION = 2
   ```
3. Document the change:
   ```python
   """
   Version History:
       Audio QC v1: Initial implementation
       Audio QC v2: Added stricter peak detection thresholds
   """
   ```
4. Commit changes with clear message
5. Metadata created after this change will automatically use v2
6. Consider reprocessing important datasets to apply new QC
