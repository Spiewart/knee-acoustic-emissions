# Processing Log System

## Overview

The processing log system provides comprehensive tracking of all processing stages for acoustic emissions data, including:

- **Audio Processing**: Conversion from raw .bin files, QC metrics, channel statistics
- **Biomechanics Import**: Data extraction from Excel files, recording details
- **Synchronization**: Audio-biomechanics alignment using stomp detection
- **Movement Cycles**: Cycle extraction and quality control analysis

All information is saved in a single Excel file (`.xlsx`) per knee/maneuver combination, with separate sheets for each processing stage.

## Features

### Automatic Logging
Processing logs are **automatically created and updated** during participant processing. No manual intervention is required.

### Incremental Updates
When re-processing a participant directory, only the portions of the log that correspond to the re-run stages are updated. For example:
- Re-running only the `cycles` stage updates only the Movement Cycles sheet
- Re-running `sync` stage updates Synchronization and Movement Cycles sheets
- Re-running `bin` stage updates all sheets

### Comprehensive Information
Each log includes:

#### Summary Sheet
- Study ID, knee side, maneuver type
- Processing timestamps (created, last updated)
- Overall status indicators
- Counts of processed files

#### Audio Sheet
- File names and paths
- Sample rate, duration, file size
- Device metadata (serial number, firmware version, recording time)
- Per-channel RMS and peak values
- Instantaneous frequency indicator

#### Biomechanics Sheet
- Source Excel file and sheet name
- Number of recordings and passes
- Duration and sample rate
- Time range (start/end times)

#### Synchronization Sheet
- One row per synchronized file
- Pass number and speed (for walking)
- Stomp detection times (audio, left knee, right knee)
- Sample count and duration
- Sync QC status

#### Movement Cycles Sheet
- One row per cycle extraction analysis
- Total cycles extracted
- Clean vs. outlier cycle counts
- Acoustic threshold used
- Output directory and plot status

## File Location

Processing logs are saved in each maneuver directory with the naming convention:

```
processing_log_{study_id}_{knee_side}_{maneuver}.xlsx
```

**Examples:**
```
Left Knee/Walking/processing_log_1011_Left_walk.xlsx
Right Knee/Sit-Stand/processing_log_1011_Right_sit_to_stand.xlsx
Left Knee/Flexion-Extension/processing_log_1011_Left_flexion_extension.xlsx
```

## Usage

### Automatic Usage (Recommended)

Processing logs are created automatically when processing participants:

```bash
# Process a participant - logs are created automatically
python cli/process_directory.py /path/to/studies --participant 1011

# Re-process only cycles stage - only cycles info is updated
python cli/process_directory.py /path/to/studies --participant 1011 --entrypoint cycles
```

### Programmatic Usage

You can also create and manage logs programmatically:

```python
from pathlib import Path
from src.orchestration.processing_log import ManeuverProcessingLog

# Get or create a log
log = ManeuverProcessingLog.get_or_create(
    study_id="1011",
    knee_side="Left",
    maneuver="walk",
    maneuver_directory=Path("/path/to/participant/#1011/Left Knee/Walking"),
)

# Add records (automatically done during processing)
# ... processing code ...

# Save the log
log.save_to_excel()

# Load an existing log
loaded_log = ManeuverProcessingLog.load_from_excel(
    Path("/path/to/processing_log_1011_Left_walk.xlsx")
)
```

### Using Helper Functions

Helper functions are provided to create records from processing data:

```python
from src.orchestration.processing_log import (
    create_audio_record_from_data,
    create_biomechanics_record_from_data,
    create_sync_record_from_data,
    create_cycles_record_from_data,
)

# Create audio record from DataFrame and metadata
audio_record = create_audio_record_from_data(
    audio_file_name="audio_with_freq",
    audio_df=audio_dataframe,
    audio_bin_path=Path("audio.bin"),
    audio_pkl_path=Path("audio_with_freq.pkl"),
    metadata=audio_metadata_dict,
)

# Add to log
log.update_audio_record(audio_record)
```

## Example Log Structure

A typical processing log Excel file contains:

```
processing_log_1011_Left_walk.xlsx
├── Summary Sheet
│   └── Overall processing status and counts
├── Audio Sheet
│   └── Audio file conversion and QC details
├── Biomechanics Sheet
│   └── Biomechanics import details
├── Synchronization Sheet
│   ├── left_walk_slow_pass1
│   ├── left_walk_slow_pass2
│   ├── left_walk_slow_pass3
│   ├── left_walk_medium_pass1
│   ├── left_walk_medium_pass2
│   ├── left_walk_medium_pass3
│   ├── left_walk_fast_pass1
│   ├── left_walk_fast_pass2
│   └── left_walk_fast_pass3
└── Movement Cycles Sheet
    ├── Cycles for left_walk_slow_pass1
    ├── Cycles for left_walk_slow_pass2
    └── ... (one row per synchronized file)
```

## Implementation Details

### Core Classes

- **`ManeuverProcessingLog`**: Main log container for a knee/maneuver
- **`AudioProcessingRecord`**: Audio conversion and QC data
- **`BiomechanicsImportRecord`**: Biomechanics import data
- **`SynchronizationRecord`**: Sync data for one file
- **`MovementCyclesRecord`**: Cycle extraction data for one file

### Integration Points

The processing log system is integrated into:

1. **`_sync_maneuver_data()`**: Captures audio, biomechanics, and sync data
2. **`process_participant()`**: Captures movement cycles data
3. **`_save_or_update_processing_log()`**: Updates log with new data

### Error Handling

The logging system is designed to be non-blocking:
- If log creation fails, a warning is logged but processing continues
- Partial data is acceptable - only available information is recorded
- Missing optional fields are set to `None`

## Example Output

See `examples/processing_log_example.py` for a demonstration of creating and using processing logs programmatically.

To run the example:

```bash
python examples/processing_log_example.py
```

This creates `example_processing_log.xlsx` with sample data demonstrating the log structure.

## Benefits

1. **Traceability**: Complete record of all processing steps and parameters
2. **Quality Control**: Easy review of QC metrics across all files
3. **Debugging**: Quick identification of processing issues
4. **Reproducibility**: All parameters and timestamps recorded
5. **Incremental Processing**: Clear indication of what has been processed
6. **Data Management**: Single file per maneuver simplifies organization

## Future Enhancements

Potential future additions:
- Processing duration metrics
- Error/warning counts
- Audio QC results (periodic signal detection)
- Sync QC detailed metrics
- Visualization links or embedded plots
- Processing parameter history
