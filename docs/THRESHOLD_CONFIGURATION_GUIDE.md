# Signal Dropout Detection Threshold Configuration

## Overview

All signal dropout detection functions in `src/audio/raw_qc.py` now use **centralized threshold configuration** through the `DEFAULT_DROPOUT_THRESHOLDS` dataclass.

## Benefits

✅ **Single Source of Truth**: Thresholds defined in one place
✅ **Easy Maintenance**: Change thresholds once, applies everywhere
✅ **Consistent Behavior**: All functions use same defaults
✅ **Type Safety**: Dataclass provides structure and documentation
✅ **Override When Needed**: Pass custom thresholds to any function

## Default Thresholds

```python
DEFAULT_DROPOUT_THRESHOLDS = DropoutThresholds(
    silence_threshold=1.45,         # Maximum RMS for silence detection
    flatline_threshold=0.000001,    # Maximum variance for flatline detection
    window_size_s=0.5,              # Sliding window size in seconds
    min_dropout_duration_s=0.1      # Minimum continuous dropout to report
)
```

These values are calibrated for **acoustic emission sensor data** with DC offset (~1.5V baseline characteristic of knee accelerometers).

## Using Default Thresholds

All functions automatically use `DEFAULT_DROPOUT_THRESHOLDS` when parameters are not provided:

```python
from src.audio.raw_qc import detect_signal_dropout_per_mic

# Uses DEFAULT_DROPOUT_THRESHOLDS automatically
dropout_dict = detect_signal_dropout_per_mic(df, time_col="tt")
```

## Customizing Thresholds

### Option 1: Pass Custom Values to Functions

For one-off customization when processing non-standard sensor data, pass threshold parameters directly:

```python
# Use custom thresholds for different sensor characteristics
dropout_dict = detect_signal_dropout_per_mic(
    df,
    time_col="tt",
    silence_threshold=0.5,      # Higher for high-gain sensors
    flatline_threshold=0.0001,  # Lower for more sensitive detection
)
```

### Option 2: Create Custom Threshold Configuration

For consistent use across multiple calls with different sensor types, create a custom configuration:

```python
from src.audio.raw_qc import DropoutThresholds

# Define thresholds for different sensor gain settings
HIGH_GAIN_THRESHOLDS = DropoutThresholds(
    silence_threshold=0.5,
    flatline_threshold=0.0001,
    window_size_s=0.5,
    min_dropout_duration_s=0.1
)

# Use in functions
dropout_dict = detect_signal_dropout_per_mic(
    df,
    silence_threshold=HIGH_GAIN_THRESHOLDS.silence_threshold,
    flatline_threshold=HIGH_GAIN_THRESHOLDS.flatline_threshold,
    window_size_s=HIGH_GAIN_THRESHOLDS.window_size_s,
    min_dropout_duration_s=HIGH_GAIN_THRESHOLDS.min_dropout_duration_s,
)
```

### Option 3: Modify Global Defaults (Not Recommended)

You can modify the global default, but this affects all subsequent calls:

```python
from src.audio import raw_qc

# NOT RECOMMENDED: Mutating global state
# (DropoutThresholds is frozen, so you'd need to replace the whole object)
```

## Calibrating Thresholds for Different Sensor Configurations

If you're working with different accelerometer sensors or gain settings:

1. **Analyze your signal characteristics**:
   ```python
   # Compute statistics on 0.5s windows
   window_samples = int(fs * 0.5)
   rms_values = []
   var_values = []

   for i in range(0, len(data) - window_samples, window_samples):
       window = data[i:i+window_samples]
       rms_values.append(np.sqrt(np.mean(window**2)))
       var_values.append(np.var(window))

   # Find bottom percentiles (true dropout/sensor failure)
   rms_1st = np.percentile(rms_values, 1)
   var_1st = np.percentile(var_values, 1)
   ```

2. **Set thresholds below normal signal range**:
   - `silence_threshold`: Just below 1st percentile of RMS values
   - `flatline_threshold`: Just below 1st percentile of variance values
   - These should flag only true sensor disconnection/malfunction, not normal signal variations

3. **Test and validate**:
   - Verify false positive rate is acceptable
   - Ensure true dropout/sensor failure is detected
   - Document the sensor configuration used for calibration

## Functions Using Centralized Thresholds

All four functions now use `DEFAULT_DROPOUT_THRESHOLDS`:

1. **`detect_signal_dropout()`** - Overall dropout detection
2. **`detect_signal_dropout_per_mic()`** - Per-channel dropout detection
3. **`run_raw_audio_qc()`** - Combined dropout + artifact detection
4. **`run_raw_audio_qc_per_mic()`** - Per-channel combined QC

## Migration Notes

### Before (Multiple Hardcoded Defaults)
```python
def detect_signal_dropout_per_mic(
    df,
    silence_threshold: float = 0.05,  # Hardcoded here
    flatline_threshold: float = 0.01,  # And here
    ...
)
```

### After (Centralized Configuration)
```python
def detect_signal_dropout_per_mic(
    df,
    silence_threshold: float | None = None,  # Uses DEFAULT_DROPOUT_THRESHOLDS
    flatline_threshold: float | None = None,  # if None
    ...
):
    if silence_threshold is None:
        silence_threshold = DEFAULT_DROPOUT_THRESHOLDS.silence_threshold
    ...
```

### Why This Is Better
- **Before**: To change thresholds, update 4 function signatures
- **After**: To change thresholds, update 1 dataclass definition
- **Backward Compatible**: Existing code with explicit thresholds still works
- **Forward Compatible**: Easy to add new threshold parameters

## Example: Processing Different Sensor Configurations

```python
from src.audio.raw_qc import detect_signal_dropout_per_mic, DropoutThresholds

# Default thresholds (knee accelerometer, standard gain)
ae_dropout = detect_signal_dropout_per_mic(ae_audio_df)

# High-gain accelerometer (more sensitive, lower baseline RMS)
high_gain_dropout = detect_signal_dropout_per_mic(
    high_gain_audio_df,
    silence_threshold=0.5,
    flatline_threshold=0.0001,
)

# Low-gain accelerometer (less sensitive, higher baseline RMS)
low_gain_dropout = detect_signal_dropout_per_mic(
    low_gain_audio_df,
    silence_threshold=2.5,
    flatline_threshold=0.00001,
)
```
