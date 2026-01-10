# Biomechanics QA/QC for Movement Cycles

## Overview

The biomechanics QA/QC feature validates that movement cycles extracted from synchronized audio and biomechanics data contain appropriate knee angle fluctuations. This ensures that only cycles with valid biomechanical movement patterns are used for downstream analysis.

## Background

Each scripted maneuver has stereotypic (characteristic) fluctuations in the knee angle z-axis (flexion-extension) that are measured biomechanically through:
- Goniometers
- Inertial Measurement Units (IMUs)
- Motion Analysis systems

The biomechanics QC validates these expected patterns through **waveform-level analysis**, checking both the range of motion (ROM) and the shape of the knee angle trajectory to ensure it matches the stereotypic pattern for each maneuver.

## QC Stages

The `MovementCycleQC` class performs two-stage quality control:

### Stage 1: Acoustic Signal Validation
Filters cycles by acoustic signal strength using area-under-curve (AUC). Cycles with insufficient acoustic energy are flagged as outliers.

### Stage 2: Biomechanics Waveform Validation
Validates that each cycle exhibits the expected stereotypic knee angle waveform pattern. This includes:

1. **Range of Motion (ROM) Check**: Minimum ROM thresholds specific to each maneuver
2. **Waveform Pattern Validation**: Verifies the shape and characteristics of the knee angle trajectory

#### Walking Gait Validation
- Start and end angles must be similar (heel strike positions)
- Single dominant flexion peak during swing phase (middle 20-80% of cycle)
- Peak should not be at the extremes of the cycle

#### Sit-to-Stand Validation
- High angle at start (sitting position, ~90° flexion)
- Decrease to low angle at end (standing position, ~0-10° extension)
- Angle change must be substantial (≥50% of ROM)
- General downward trend (later thirds should have lower mean angles)

#### Flexion-Extension Validation
- Start and end angles should be similar (extension position)
- Clear flexion peak in the middle portion (25-75% of cycle)
- Not more than 3 peaks (avoids noisy/erratic signals)

## Expected Patterns by Maneuver

| Maneuver | ROM Threshold | Waveform Characteristics |
|----------|--------------|--------------------------|
| **Walk** | 20° | Start low (heel strike) → peak during swing → end low (heel strike) |
| **Sit-to-stand** | 40° | Start high (sitting ~90°) → decrease to low (standing ~0-10°) |
| **Flexion-extension** | 40° | Start low (extension) → peak during flexion → end low (extension) |

## Usage

### Command Line Interface

```bash
# Use default thresholds for each maneuver
ae-sync-qc /path/to/synced_data.pkl

# Customize acoustic threshold
ae-sync-qc /path/to/synced_data.pkl --threshold 150.0

# Customize biomechanics ROM threshold
ae-sync-qc /path/to/synced_data.pkl --biomech-min-rom 30.0

# Process multiple files in a directory
ae-sync-qc /path/to/participant/Knee/Maneuver/Synced/

# Disable plot generation
ae-sync-qc /path/to/synced_data.pkl --no-plots

# Override maneuver detection
ae-sync-qc /path/to/synced_data.pkl --maneuver sit_to_stand

# Verbose output for debugging
ae-sync-qc /path/to/synced_data.pkl -v
```

### Python API

```python
from pathlib import Path
from src.synchronization.quality_control import MovementCycleQC, perform_sync_qc

# Option 1: Use high-level pipeline function
clean_cycles, outlier_cycles, output_dir = perform_sync_qc(
    synced_pkl_path=Path("/path/to/synced_data.pkl"),
    maneuver="walk",
    acoustic_threshold=100.0,
    biomech_min_rom=20.0,  # Custom ROM threshold
    create_plots=True
)

# Option 2: Use QC class directly with custom cycles
import pandas as pd
from src.biomechanics.cycle_parsing import extract_movement_cycles

# Load synchronized data
synced_df = pd.read_pickle("/path/to/synced_data.pkl")

# Extract cycles
cycles = extract_movement_cycles(synced_df, maneuver="walk")

# Initialize QC analyzer
qc = MovementCycleQC(
    maneuver="walk",
    speed="medium",
    acoustic_threshold=100.0,
    biomech_min_rom=25.0,  # Custom threshold
    acoustic_channel="filtered"
)

# Perform QC analysis
clean_cycles, outlier_cycles = qc.analyze_cycles(cycles)

print(f"Clean cycles: {len(clean_cycles)}")
print(f"Outlier cycles: {len(outlier_cycles)}")
```

## Output Structure

The QC pipeline creates the following directory structure:

```
ManeuverDirectory/
└── Synced/
    └── MovementCycles/
        ├── clean/
        │   ├── file_cycle_000.pkl
        │   ├── file_cycle_000.json
        │   ├── file_cycle_000.png
        │   ├── file_cycle_001.pkl
        │   └── ...
        └── outliers/
            ├── file_outlier_000.pkl
            ├── file_outlier_000.json
            ├── file_outlier_000.png
            └── ...
```

- **clean/**: Cycles passing both acoustic and biomechanics QC
- **outliers/**: Cycles failing either QC check
- **.pkl files**: Cycle DataFrames with synchronized audio and biomechanics data
- **.json files**: Metadata including QC results and parameters
- **.png files**: Visualization plots (if enabled)

## Metadata

Each cycle's metadata JSON includes:
- `cycle_qc_pass`: Boolean indicating if cycle passed QC
- `cycle_qc_version`: Version of QC algorithm used (v2 includes biomechanics validation)
- `cycle_acoustic_energy`: Computed acoustic energy (AUC)
- Biomechanics file information and sync times
- Maneuver, speed, and pass number details

## Troubleshooting

### High Outlier Rate

If many cycles are being flagged as outliers:

1. **Check ROM threshold**: Use `-v` verbose flag to see ROM values for each cycle
   ```bash
   ae-sync-qc /path/to/synced_data.pkl -v
   ```

2. **Adjust threshold**: Lower the ROM threshold if biomechanics data quality is lower
   ```bash
   ae-sync-qc /path/to/synced_data.pkl --biomech-min-rom 15.0
   ```

3. **Verify data**: Ensure `Knee Angle Z` column exists and contains valid data

### Missing Knee Angle Data

If cycles are missing the `Knee Angle Z` column:
- Verify biomechanics data was properly imported during synchronization
- Check that the correct knee (left/right) was selected
- Ensure biomechanics Excel file contains knee angle data

## QC Version History

- **v1**: Initial acoustic energy thresholding only
- **v2**: Added biomechanics validation with knee angle ROM checks (current)

See `src/qc_versions.py` for version management details.

## Related Modules

- `src/synchronization/quality_control.py`: Main QC implementation
- `src/biomechanics/cycle_parsing.py`: Movement cycle extraction
- `src/qc_versions.py`: QC version management
- `tests/test_sync_qc.py`: QC test suite
