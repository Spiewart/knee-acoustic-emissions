# Artifact Type QC Fix - Complete Summary

## Problem Statement
Excel reports were not displaying artifact type QC data correctly when reprocessing participant audio:
- Overall `QC Artifact Type` column showed empty `'[]'`
- Per-channel `QC Artifact Type Ch1/2/3/4` columns showed string representations like `"['Continuous']"` instead of readable values

## Root Causes Identified

### Issue 1: Metadata Type Mismatch
**Location**: `src/metadata.py` lines 127-153
**Problem**: AudioProcessing class defined artifact type fields as single string values:
```python
# BEFORE (incorrect)
qc_artifact_type: Optional[Literal["intermittent", "continuous"]] = None
```
**Impact**: Fields couldn't accept list data, causing validation errors or data loss

### Issue 2: Missing Overall Artifact Type Computation
**Location**: `src/orchestration/participant_processor.py` line 183
**Problem**: Code tried to access non-existent "overall" key:
```python
# BEFORE (broken)
"qc_artifact_type": artifact_types_per_mic.get("overall", []) if artifact_intervals else [],
```
**Impact**: Overall artifact type always empty `[]` even when channels had artifacts

### Issue 3: List-to-String Conversion in Excel Export
**Location**: `src/reports/report_generator.py` lines 167-171
**Problem**: PostgreSQL ARRAY columns returned as Python lists, which pandas/Excel converted to string representations:
```python
# BEFORE (produces "['Continuous']" in Excel)
'QC Artifact Type': record.qc_artifact_type,  # Python list → string representation
```
**Impact**: Excel showed `"['Continuous']"` instead of readable `"Continuous"`

## Solutions Implemented

### Fix 1: Updated Metadata Field Types
**File**: `src/metadata.py`
```python
# AFTER (correct)
qc_artifact_type: Optional[List[Literal["Intermittent", "Continuous"]]] = None
qc_artifact_type_ch1: Optional[List[Literal["Intermittent", "Continuous"]]] = None
qc_artifact_type_ch2: Optional[List[Literal["Intermittent", "Continuous"]]] = None
qc_artifact_type_ch3: Optional[List[Literal["Intermittent", "Continuous"]]] = None
qc_artifact_type_ch4: Optional[List[Literal["Intermittent", "Continuous"]]] = None
```
**Changes**: 5 field definitions updated (overall + 4 channels)

### Fix 2: Compute Overall Artifact Type from Channels
**File**: `src/orchestration/participant_processor.py`
```python
# AFTER (combined from all channels)
overall_artifact_types = []
for ch_num in range(1, 5):
    ch_name = f"ch{ch_num}"
    if ch_name in artifact_types_per_mic:
        overall_artifact_types.extend(artifact_types_per_mic[ch_name])
# Remove duplicates while preserving order
seen = set()
overall_artifact_types = [x for x in overall_artifact_types if not (x in seen or seen.add(x))]

qc_data = {
    ...
    "qc_artifact_type": overall_artifact_types if artifact_intervals else [],
}
```
**Changes**: Lines 175-204 refactored to compute overall type

### Fix 3: Add List-to-String Formatter for Excel
**File**: `src/reports/report_generator.py`
```python
@staticmethod
def _format_list_for_excel(value) -> Optional[str]:
    """Convert list to comma-separated string for Excel.

    PostgreSQL ARRAY columns return as Python lists, but pandas/Excel
    will convert these to string representations like "['value']".
    This method properly converts them to readable strings like "value".
    """
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return None
        return ", ".join(str(v) for v in value)
    return value
```

Then applied to all artifact type columns:
```python
'QC Artifact Type': self._format_list_for_excel(record.qc_artifact_type),
'QC Artifact Type Ch1': self._format_list_for_excel(record.qc_artifact_type_ch1),
'QC Artifact Type Ch2': self._format_list_for_excel(record.qc_artifact_type_ch2),
'QC Artifact Type Ch3': self._format_list_for_excel(record.qc_artifact_type_ch3),
'QC Artifact Type Ch4': self._format_list_for_excel(record.qc_artifact_type_ch4),
```

## Verification

All fixes verified with comprehensive integration test:

✓ Metadata accepts `List[Literal["Intermittent", "Continuous"]]`
✓ List-to-string formatting produces correct Excel values:
  - `["Intermittent"]` → `"Intermittent"`
  - `["Continuous"]` → `"Continuous"`
  - `["Intermittent", "Continuous"]` → `"Intermittent, Continuous"`
  - `[]` → `None` (empty)
✓ Overall artifact type correctly combines all channels
✓ Case consistency verified: "Intermittent"/"Continuous" (capital letters)

## Data Flow After Fixes

```
1. detect_artifactual_noise_per_mic()
   ↓ Returns: {"ch1": [...], "ch2": [...], ...}

2. participant_processor.py (compute overall)
   ↓ Combines: ["Continuous", "Intermittent"] (deduplicated)

3. create_audio_record_from_data()
   ↓ Extracts qc_data fields → AudioProcessing model

4. _update_audio_processing_record()
   ↓ Saves to database: ARRAY(String) columns

5. report_generator.py
   ↓ Queries database, formats lists: "Continuous, Intermittent"

6. Excel File
   ✓ Displays: "Continuous, Intermittent" (readable!)
```

## Files Modified

1. `src/metadata.py` - Updated artifact type field definitions (14 lines)
2. `src/orchestration/participant_processor.py` - Added overall type computation (30 lines)
3. `src/reports/report_generator.py` - Added list formatter and applied to all artifact type fields (25 lines)

## Testing

Run reprocessing to verify:
```bash
ae-process-directory "/Users/spiewart/kae_signal_processing_ml/sample_project_directory/" \
  --entrypoint bin --participant 1016 --maneuver walk
```

Then check Excel file:
```
/Users/spiewart/kae_signal_processing_ml/sample_project_directory/#1016/Right Knee/Walking/processing_log_1016_Right_walk.xlsx
```

Expected columns now populated correctly:
- `QC Fail` - Boolean (True/False)
- `QC Signal Dropout` - Boolean (True/False)
- `QC Artifact` - Boolean (True/False)
- `QC Artifact Type` - String (e.g., "Intermittent" or "Continuous, Intermittent")
- `QC Artifact Type Ch1-4` - String (e.g., "Continuous")
