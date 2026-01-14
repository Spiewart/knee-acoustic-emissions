# Periodic Noise Detection Integration

## Summary

The `_detect_periodic_noise()` function has been successfully re-integrated into the raw audio QC processing pipeline. This function detects consistent periodic background noise (e.g., from fans, HVAC systems, motors) using spectral analysis (Welch's method).

## Changes Made

### 1. Function Signatures Updated

All artifact detection functions now accept two new parameters:

- **`detect_periodic_noise`** (bool, default=True): Enable/disable periodic noise detection
- **`periodic_noise_threshold`** (float, default=0.3): Detection threshold (0-1 scale)
  - Lower values: More sensitive to periodic noise
  - Higher values: More conservative, only strong periodic components

### 2. Functions Modified

#### Core Detection Functions
- `detect_artifactual_noise()`: Now detects both spikes and periodic noise
- `detect_artifactual_noise_per_mic()`: Per-channel detection with periodic noise support
- `run_raw_audio_qc()`: Comprehensive QC including periodic noise
- `run_raw_audio_qc_per_mic()`: Per-channel comprehensive QC

#### Internal Implementation
- `_detect_periodic_noise()`: Moved from deprecated status back to active use
  - Uses Welch's method for power spectral density analysis
  - Identifies prominent spectral peaks above 5 Hz
  - Calculates relative power (peak / median)
  - Marks samples with strong periodic components as artifacts

### 3. Processing Pipeline Integration

Raw audio QC is executed during **bin processing stage** when processing .bin files:

```python
from src.audio.raw_qc import run_raw_audio_qc, merge_bad_intervals

# Load raw audio from .bin file
df = load_audio_data()

# Run comprehensive QC (now includes periodic noise detection)
dropout_intervals, artifact_intervals = run_raw_audio_qc(df)

# Periodic noise detection is enabled by default
# To disable: run_raw_audio_qc(df, detect_periodic_noise=False)

# Merge all bad intervals
bad_intervals = merge_bad_intervals(dropout_intervals, artifact_intervals)

# Store QC results in processing log
```

## Usage Examples

### Basic Usage (Periodic Noise Detection Enabled by Default)

```python
from src.audio.raw_qc import run_raw_audio_qc
import pandas as pd

df = pd.read_pickle("audio_data.pkl")

# Detects both spikes and periodic noise
dropout_intervals, artifact_intervals = run_raw_audio_qc(df)
```

### Disable Periodic Noise Detection

```python
# Only detect spikes, not periodic noise
dropout_intervals, artifact_intervals = run_raw_audio_qc(
    df,
    detect_periodic_noise=False
)
```

### Custom Threshold for Periodic Noise

```python
# More sensitive to periodic noise
dropout_intervals, artifact_intervals = run_raw_audio_qc(
    df,
    periodic_noise_threshold=0.2,  # Lower = more sensitive
)

# More conservative (only strong periodic components)
dropout_intervals, artifact_intervals = run_raw_audio_qc(
    df,
    periodic_noise_threshold=0.5,  # Higher = less sensitive
)
```

### Per-Microphone Detection with Periodic Noise

```python
from src.audio.raw_qc import run_raw_audio_qc_per_mic

# Returns bad intervals per channel (includes periodic noise per channel)
per_mic_intervals = run_raw_audio_qc_per_mic(df)

for channel, intervals in per_mic_intervals.items():
    print(f"{channel}: {intervals}")
```

## Documentation Updates

- Module docstring updated to describe periodic noise detection
- Function docstrings updated with new parameters
- Deprecation notes removed from `_detect_periodic_noise()`
- Notes about moving to cycle_qc.py have been removed

## Testing

All tests pass successfully:

- ✅ Raw audio QC tests: 19/19 passed
- ✅ Per-microphone audio QC tests: 6/6 passed
- ✅ Cycle QC tests: 29/29 passed
- ✅ Audio QC tests: 13/13 passed
- ✅ Process participant directory tests: 59/59 passed

### Test Adjustments

One test in `test_per_mic_audio_qc.py` was updated:
- `test_detect_artifactual_noise_per_mic()`: Now explicitly disables periodic noise detection to focus on spike detection testing, since the test uses a sinusoidal signal which naturally has strong periodic components.

## Implementation Notes

### How Periodic Noise Detection Works

1. **Signal Processing**: Audio signal is analyzed using Welch's method for power spectral density (PSD)
2. **Spectral Analysis**:
   - Uses 2-second windows with 50% overlap
   - Ignores DC and frequencies below 5 Hz
3. **Peak Detection**: Identifies prominent spectral peaks
4. **Relative Power Calculation**: `peak_power / median_power`
5. **Threshold Comparison**: If relative power > (1.0 / threshold), periodic noise is detected

### Performance Characteristics

- **Processing Cost**: Moderate (uses Welch's method on full recording)
- **Sensitivity**: Configurable via threshold parameter
- **False Positives**: Can occur with strong harmonic content (e.g., speech)
- **Optimization Note**: For very large recordings, consider applying only to identified cycles

## Backward Compatibility

The integration maintains backward compatibility:
- Default behavior now includes periodic noise detection
- Existing code continues to work (periodic noise detection is enabled by default)
- Users can disable periodic noise detection with `detect_periodic_noise=False` if needed

## Future Enhancements

Possible future improvements:
1. Adaptive threshold based on signal characteristics
2. Frequency-specific noise detection (identify problem frequencies)
3. Time-localized periodic noise detection for targeted cycles
4. Integration with sync_qc for cycle-specific periodic noise analysis
