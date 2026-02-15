# Raw Audio Quality Control (Audio QC)

## Overview

The raw audio QC module provides automated quality checks for unprocessed acoustic recordings from .bin files. It detects two main types of issues:

1. **Signal Dropout**: Silence or flatline conditions in microphones
2. **Artifactual Noise**: Two types of artifacts:
   - **Type 1**: One-off or time-limited spikes (intermittent background noise: talking, objects falling)
   - **Type 2**: Consistent periodic background noise at specific frequencies (fans, motors)

Bad sections are annotated with timestamps and stored in processing logs for review and downstream analysis. The module also provides synchronization support for aligning QC results with biomechanics data.

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

Identifies two types of abnormal signal patterns:

**Type 1: One-off Spikes**
- Detects intermittent background noise (people talking, glass breaking)
- Uses local statistical thresholds (mean + N×std)
- Default: 5 standard deviations
- Window size: 0.01 seconds

**Type 2: Periodic Background Noise**
- Detects consistent noise at specific frequencies (fan running, motor hum)
- Uses power spectral density analysis (Welch's method)
- Identifies prominent spectral peaks
- Configurable threshold (0-1 scale, default: 0.3)

### Synchronization Support

Methods to handle audio QC in synchronized data context:

**`adjust_bad_intervals_for_sync()`**
- Adjusts bad interval timestamps from audio coordinates to synchronized (biomechanics) coordinates
- Uses stomp times to calculate offset
- Ensures QC results align with synchronized data

**`check_cycle_in_bad_interval()`**
- Checks if movement cycles overlap with bad audio segments
- Configurable overlap threshold (default: 10%)
- Returns whether cycle should be marked as failing audio QC
- Can populate `audio_QC_passed` field in cycle metadata

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
    detect_periodic_noise=True,
    periodic_noise_threshold=0.3,
)
```

### Synchronization Workflow

When working with synchronized audio-biomechanics data:

```python
from src.audio.raw_qc import (
    run_raw_audio_qc,
    merge_bad_intervals,
    adjust_bad_intervals_for_sync,
    check_cycle_in_bad_interval,
)

# Step 1: Run QC on raw audio
dropout, artifacts = run_raw_audio_qc(audio_df)
bad_intervals_audio = merge_bad_intervals(dropout, artifacts)

# Step 2: Get stomp times from synchronization
audio_stomp_time = 3.5  # seconds (in audio coordinates)
bio_stomp_time = 12.0   # seconds (in biomechanics coordinates)

# Step 3: Adjust bad intervals to synchronized coordinates
bad_intervals_synced = adjust_bad_intervals_for_sync(
    bad_intervals_audio,
    audio_stomp_time,
    bio_stomp_time,
)

# Step 4: Check if movement cycles have bad audio
for cycle in movement_cycles:
    cycle_start = cycle['start_time']  # in synced coordinates
    cycle_end = cycle['end_time']

    audio_qc_passed = not check_cycle_in_bad_interval(
        cycle_start,
        cycle_end,
        bad_intervals_synced,
        overlap_threshold=0.1,  # 10% overlap threshold
    )

    # Update cycle metadata
    cycle['audio_qc_pass'] = audio_qc_passed
    if not audio_qc_passed:
        print(f"Cycle {cycle['id']} failed audio QC due to bad segments")
```

### Controlling Artifact Detection Types

```python
from src.audio.raw_qc import run_raw_audio_qc

# Detect only spikes, not periodic noise
dropout, artifacts = run_raw_audio_qc(
    df,
    detect_periodic_noise=False,
)

# Detect both types with custom thresholds
dropout, artifacts = run_raw_audio_qc(
    df,
    spike_threshold_sigma=3.0,          # More sensitive to spikes
    periodic_noise_threshold=0.2,       # More sensitive to periodic noise
    detect_periodic_noise=True,
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

**Type 1: One-off Spikes**
- Brief, isolated spikes: Electrical interference or EMI
- Burst patterns: Someone talking, doors closing, objects falling
- Irregular timing: Environmental disturbances

**Type 2: Periodic Background Noise**
- Entire recording flagged: Consistent fan or motor running throughout
- Specific frequency dominance: HVAC system, machinery
- Steady presence: Background equipment noise

**General Guidelines:**
- Short artifacts (< 0.1s): Often transient electrical noise
- Medium artifacts (0.1-1s): Likely environmental events
- Long artifacts (> 1s): Sustained background noise or periodic interference
- Frequent short artifacts: Check for EMI sources or grounding issues

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
- Disable periodic noise detection (use `detect_periodic_noise=False`, which is the default)
- Process data in chunks
- Use only necessary channels

## Future Enhancements

Potential improvements:
- Adaptive thresholds based on signal statistics
- Machine learning-based artifact detection
- Real-time QC during recording
- Visualization tools for QC results
