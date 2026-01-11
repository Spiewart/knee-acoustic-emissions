# Raw Audio Quality Control (Audio QC)

## Overview

The raw audio QC module provides automated quality checks for unprocessed acoustic recordings from .bin files. It detects two main types of issues:

1. **Signal Dropout**: Silence or flatline conditions in microphones
2. **Artifactual Noise**: Spikes, outliers, or abnormal signal patterns

Bad sections are annotated with timestamps and stored in processing logs for review and downstream analysis.

## Features

### Dropout Detection

Identifies periods where microphones are not recording properly:
- **Silence Detection**: RMS amplitude below threshold (near-zero voltage)
- **Flatline Detection**: Variance below threshold (unchanging/stuck sensor)

Default thresholds:
- Silence: RMS < 0.001
- Flatline: Variance < 0.0001
- Window size: 0.5 seconds
- Minimum duration: 0.1 seconds

### Artifact Detection

Identifies abnormal signal spikes or noise bursts:
- Detects values exceeding mean + N×std in local sliding window
- Default: 5 standard deviations
- Window size: 0.01 seconds
- Minimum duration: 0.01 seconds

### Audio Segment Clipping

Provides utilities to:
- Remove bad time segments from audio data
- Preserve original timestamps
- Maintain synchronization metadata

## Integration with Processing Pipeline

Raw audio QC is automatically executed during the **bin processing stage** when processing .bin files:

```bash
# Process bin files with QC
ae-process-directory /path/to/studies --entrypoint bin
```

**Processing Flow:**
1. Read .bin file → base pickle + metadata
2. **Run raw audio QC** → detect dropout and artifacts
3. Store QC results in processing log (QC_not_passed field)
4. Add instantaneous frequency
5. Save frequency-augmented pickle

## Processing Log Integration

QC results are stored in the `AudioProcessingRecord` within the maneuver processing log:

**QC_not_passed Field:**
- Format: String representation of list of (start_time, end_time) tuples
- Example: `"[(1.5, 2.3), (5.0, 6.2)]"`
- Empty/None if no issues detected

**Excel Log Location:**
- File: `processing_log_{study_id}_{knee_side}_{maneuver}.xlsx`
- Sheet: "Audio"
- Column: "QC_not_passed"

## API Usage

### Basic QC

```python
from src.audio.raw_qc import run_raw_audio_qc, merge_bad_intervals
import pandas as pd

# Load audio data
df = pd.read_pickle("audio_data.pkl")

# Run QC
dropout_intervals, artifact_intervals = run_raw_audio_qc(df)

# Merge overlapping/nearby intervals
bad_intervals = merge_bad_intervals(dropout_intervals, artifact_intervals)

print(f"Detected {len(dropout_intervals)} dropout periods")
print(f"Detected {len(artifact_intervals)} artifact periods")
print(f"Merged into {len(bad_intervals)} bad intervals")
```

### Custom Thresholds

```python
from src.audio.raw_qc import run_raw_audio_qc

# Adjust sensitivity
dropout_intervals, artifact_intervals = run_raw_audio_qc(
    df,
    silence_threshold=0.005,      # More sensitive to silence
    flatline_threshold=0.0005,    # More sensitive to flatline
    spike_threshold_sigma=3.0,    # More sensitive to spikes
    dropout_window_s=0.2,         # Smaller window for dropout
    min_dropout_duration_s=0.2,   # Longer minimum duration
)
```

### Clipping Bad Segments

```python
from src.audio.raw_qc import clip_bad_segments

# Remove bad segments while preserving timestamps
clean_df = clip_bad_segments(df, bad_intervals)

print(f"Original length: {len(df)} samples")
print(f"Clean length: {len(clean_df)} samples")
print(f"Removed: {len(df) - len(clean_df)} samples")
```

### Individual Detection Methods

```python
from src.audio.raw_qc import detect_signal_dropout, detect_artifactual_noise

# Detect only dropout
dropout_intervals = detect_signal_dropout(
    df,
    time_col="tt",
    audio_channels=["ch1", "ch2", "ch3", "ch4"],
    silence_threshold=0.001,
    flatline_threshold=0.0001,
)

# Detect only artifacts
artifact_intervals = detect_artifactual_noise(
    df,
    time_col="tt",
    audio_channels=["ch1", "ch2", "ch3", "ch4"],
    spike_threshold_sigma=5.0,
)
```

## Interpretation of Results

### Dropout Intervals

Intervals indicate periods where signal quality is compromised:
- Short dropouts (< 1s): May be transient sensor issues
- Long dropouts (> 5s): Likely sensor disconnection or failure
- Multiple dropouts: Check sensor connections and hardware

### Artifact Intervals

Intervals indicate periods with abnormal signal characteristics:
- Brief spikes: May be electrical interference or EMI
- Sustained artifacts: Check for environmental noise sources
- Frequent artifacts: May indicate sensor saturation or clipping

### QC_not_passed in Logs

The `QC_not_passed` column in Excel logs shows:
- **None/Empty**: Clean recording, no issues detected
- **List of tuples**: Timestamps of problematic sections
  - Format: `[(start1, end1), (start2, end2), ...]`
  - Times in seconds relative to recording start

**Example interpretations:**
- `[]`: Perfect recording
- `[(1.5, 2.0)]`: Single 0.5s issue at 1.5-2.0 seconds
- `[(0.0, 0.5), (10.0, 15.0)]`: Two issues: 0.5s at start, 5s at 10-15s

## Tuning Recommendations

### Conservative (Fewer false positives)
```python
run_raw_audio_qc(
    df,
    silence_threshold=0.0005,     # Very low threshold
    spike_threshold_sigma=8.0,    # High tolerance
    min_dropout_duration_s=0.5,   # Longer minimum
)
```

### Sensitive (More detections)
```python
run_raw_audio_qc(
    df,
    silence_threshold=0.01,       # Higher threshold
    spike_threshold_sigma=3.0,    # Low tolerance
    min_dropout_duration_s=0.05,  # Shorter minimum
)
```

### Application-specific

For different recording conditions:
- **Lab/controlled**: Use default or conservative settings
- **Field/uncontrolled**: Use sensitive settings
- **High-quality sensors**: Use conservative settings
- **Low-cost sensors**: Use sensitive settings

## Relationship to Other QC

The raw audio QC (audio_QC) is distinct from:

1. **sync_QC**: Synchronization quality checks between audio and biomechanics
   - Runs during synchronization stage
   - Focuses on alignment and temporal consistency
   - Does not check raw signal quality

2. **Maneuver-specific QC**: Activity-specific checks
   - Runs on processed/synchronized data
   - Examples: heel-strike detection (walking), periodicity (flexion-extension)
   - Assumes raw signal quality is acceptable

Raw audio QC is the **first line of defense** and runs before other QC checks. It ensures that raw sensor data is usable before downstream processing.

## Testing

The module includes comprehensive unit tests in `tests/test_raw_audio_qc.py`:

```bash
# Run tests
pytest tests/test_raw_audio_qc.py -v

# Run specific test
pytest tests/test_raw_audio_qc.py::test_detect_signal_dropout -v
```

Tests cover:
- Dropout detection (silence and flatline)
- Artifact detection (spikes)
- Interval merging
- Segment clipping
- Edge cases (empty data, no channels, etc.)

## Troubleshooting

**No issues detected when expected:**
- Lower detection thresholds
- Reduce minimum duration requirements
- Check that correct channels are specified

**Too many false positives:**
- Raise detection thresholds
- Increase window sizes for smoother detection
- Increase minimum duration to filter transients

**Performance issues:**
- Increase window sizes (reduces computation)
- Process data in chunks
- Use only necessary channels

## Future Enhancements

Potential improvements:
- Per-channel QC reporting (currently aggregated)
- Adaptive thresholds based on signal statistics
- Machine learning-based artifact detection
- Real-time QC during recording
- Visualization tools for QC results
