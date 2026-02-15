# Movement Cycle Quality Control (QC)

This document describes the comprehensive quality control system for movement cycles in synchronized acoustic emissions and biomechanics data.

## Overview

The movement cycle QC system implements three types of quality checks:

1. **Audio Cycle QC**: Detects periodic background noise in individual movement cycles
2. **Biomechanics QC**: Validates knee angle waveform patterns for each maneuver type
3. **Cross-Modal Sync QC**: Validates synchronization quality by comparing acoustic features across movement phases

## Audio Cycle QC

### Periodic Noise Detection

The audio cycle QC detects periodic background noise (e.g., from fans, HVAC systems) using spectral analysis with Welch's method. This check is expensive computationally, so it's performed on individual cycles rather than the entire recording.

**Implementation**: `src/audio/cycle_qc.py`

**Key Functions**:
- `_detect_periodic_noise_in_cycle()`: Core detection algorithm using Welch's method
- `check_cycle_periodic_noise()`: Per-channel periodic noise detection
- `run_cycle_audio_qc()`: Comprehensive audio QC for a cycle

**How it Works**:
1. Computes power spectral density (PSD) using Welch's method with 2-second windows
2. Identifies prominent spectral peaks above 5 Hz
3. Calculates relative power (peak / median)
4. Flags cycles with high relative power as having periodic noise

**Thresholds**:
- Default threshold: 0.3 (less sensitive to spurious peaks)
- Lower threshold (e.g., 0.1): More sensitive, may detect weaker noise
- Higher threshold (e.g., 0.5): More conservative, only strong periodic components

**Usage Example**:
```python
from src.audio.cycle_qc import check_cycle_periodic_noise

# Check for periodic noise in a cycle
periodic_noise = check_cycle_periodic_noise(cycle_df)

# Returns: {'ch1': False, 'ch2': True, 'ch3': False, 'ch4': False}
# In this example, channel 2 has detected periodic noise
```

## Biomechanics QC

The biomechanics QC validates that knee angle waveforms exhibit expected patterns for each maneuver type.

**Implementation**: `src/biomechanics/quality_control.py` (already implemented)

**Validation Checks**:
- **Range of Motion (ROM)**: Minimum ROM thresholds by maneuver
  - Walking: 20°
  - Sit-to-stand: 40°
  - Flexion-extension: 40°
- **Waveform Patterns**:
  - Walking: Proper gait cycle with flexion peak during swing phase
  - Sit-to-stand: Decreasing angle from sitting to standing
  - Flexion-extension: Cyclic pattern with clear flexion/extension phases

## Cross-Modal Sync QC

The most innovative feature validates synchronization quality by analyzing acoustic features across movement phases defined by joint angle.

**Implementation**: `src/audio/cycle_qc.py::check_sync_quality_by_phase()`

### How It Works

1. **Phase Definition**: Divides each cycle into 3 phases based on knee angle terciles
   - Walking: Extension → Mid-swing → Flexion
   - Sit-to-stand: Sitting → Transition → Standing
   - Flexion-extension: Extension → Mid-phase → Flexion

2. **Acoustic Feature Extraction**: Computes RMS acoustic energy for each phase

3. **Reference Range Comparison**: Compares phase-specific features to expected ranges

4. **Quality Score**: Calculates fraction of phases within expected ranges

### Reference Ranges

Default reference ranges are conservative and based on expected acoustic patterns:

**Constants** (defined in `src/audio/cycle_qc.py`):
```python
DEFAULT_MIN_RMS_ENERGY = 0.001      # Minimum detectable
DEFAULT_MAX_RMS_ENERGY = 10.0       # Maximum expected
DEFAULT_LOW_ENERGY_MAX = 5.0        # For low-energy phases
DEFAULT_MID_ENERGY_MIN = 0.01       # For moderate-energy phases
DEFAULT_HIGH_ENERGY_MIN = 0.005     # For active phases
```

**Walking**:
- Extension (heel strike): Higher energy expected (0.01 - 10.0)
- Mid-phase (stance): Moderate to high energy (0.01 - 10.0)
- Flexion (swing): Lower energy acceptable (0.001 - 5.0)

**Sit-to-Stand**:
- Sitting: Lower energy (0.001 - 5.0)
- Transition: Higher energy expected (0.01 - 10.0)
- Standing: Lower energy (0.001 - 5.0)

**Flexion-Extension**:
- Extension: Moderate energy (0.005 - 10.0)
- Mid-phase: Higher energy expected (0.01 - 10.0)
- Flexion: Moderate energy (0.005 - 10.0)

### Calibration

**Important**: The default reference ranges are conservative placeholders. For production use, these should be calibrated from a reference dataset:

1. Collect clean, validated cycles from multiple participants
2. Compute phase-specific acoustic features for each maneuver
3. Calculate mean and standard deviation for each phase
4. Set reference ranges as mean ± 2-3 standard deviations

**Custom Reference Ranges**:
```python
from src.audio.cycle_qc import check_sync_quality_by_phase

# Define custom reference ranges from your calibration dataset
custom_ranges = {
    'extension': (0.015, 8.0),
    'mid_phase': (0.020, 9.0),
    'flexion': (0.002, 4.5),
}

# Use custom ranges
results = check_sync_quality_by_phase(
    cycle_df,
    maneuver="walk",
    reference_ranges=custom_ranges
)
```

### Quality Score and Pass/Fail

- **Sync Quality Score**: Fraction of phases within expected ranges (0.0 - 1.0)
- **Pass Threshold**: ≥ 0.75 (at least 75% of phases must be in range)

```python
results = check_sync_quality_by_phase(cycle_df, maneuver="walk")

print(f"Quality Score: {results['sync_quality_score']:.2f}")
print(f"Pass: {results['sync_qc_pass']}")

# Output:
# Quality Score: 0.67
# Pass: False
# (Only 2 out of 3 phases were in expected range)
```

## Integration into perform_sync_qc()

All cycle-level QC checks are automatically run during `perform_sync_qc()`:

```python
from src.synchronization.quality_control import perform_sync_qc

clean_cycles, outlier_cycles, output_dir = perform_sync_qc(
    synced_pkl_path="path/to/synced.pkl",
    maneuver="walk",
    speed="medium",
    acoustic_threshold=100.0,
)

# Cycle metadata includes all QC results:
# - periodic_noise_detected (bool)
# - periodic_noise_ch1/ch2/ch3/ch4 (bool)
# - sync_quality_score (float)
# - sync_qc_pass (bool)
```

## Metadata Fields

The following fields are added to `MovementCycleMetadata`:

```python
class MovementCycleMetadata:
    # Audio cycle QC results
    periodic_noise_detected: bool = False
    periodic_noise_ch1: bool = False
    periodic_noise_ch2: bool = False
    periodic_noise_ch3: bool = False
    periodic_noise_ch4: bool = False

    # Cross-modal sync QC results
    sync_quality_score: Optional[float] = None
    sync_qc_pass: Optional[bool] = None
```

These fields are automatically populated during QC and saved to JSON metadata files alongside cycle pickle files.

## Comprehensive Cycle QC

For advanced use cases, you can run all QC checks together:

```python
from src.audio.cycle_qc import run_comprehensive_cycle_qc

results = run_comprehensive_cycle_qc(
    cycle_df,
    maneuver="walk",
    check_periodic_noise=True,
    check_sync_quality=True,
    fail_on_periodic_noise=False,  # Optional: fail QC if periodic noise detected
)

# Results structure:
{
    'audio_qc': {
        'periodic_noise': {'ch1': False, 'ch2': False, ...},
        'has_periodic_noise': False,
        'qc_pass': True
    },
    'sync_qc': {
        'phase_acoustic_features': {...},
        'phase_in_range': {...},
        'sync_quality_score': 0.85,
        'sync_qc_pass': True
    },
    'overall_qc_pass': True
}
```

## Performance Considerations

- **Periodic Noise Detection**: Expensive (Welch's method). Only run on extracted cycles, not entire recordings.
- **Sync Quality Checks**: Moderate cost. Requires phase definition and RMS computation per phase.
- **Typical Performance**: ~10-50ms per cycle on modern hardware

## Future Enhancements

1. **Machine Learning-Based Sync QC**: Train ML models to learn phase-specific acoustic patterns from reference data
2. **Adaptive Reference Ranges**: Automatically adjust ranges based on participant characteristics (BMI, age, etc.)
3. **Multi-Modal Features**: Include additional features beyond RMS (e.g., spectral centroid, zero-crossing rate)
4. **Real-Time Feedback**: Flag synchronization issues during data collection

## Related Documentation

- [QC Versioning](QC_VERSIONING.md): Understanding QC version tracking
- [Biomechanics QC](../src/biomechanics/quality_control.py): Detailed biomechanics validation
- [Audio QC](../src/audio/quality_control.py): Raw audio quality control

## References

1. Welch, P. (1967). "The use of fast Fourier transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms". IEEE Transactions on Audio and Electroacoustics.
2. Project-specific calibration data and protocols (to be established)
