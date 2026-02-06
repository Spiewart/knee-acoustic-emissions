# Synchronization Record Fixes - 2026-02-05

This document summarizes the fixes applied to address three critical issues with synchronization processing.

## Issue #1: Aggregate Statistics Not Populated

**Problem**: Mean, Median, Min, and Max Cycle Duration columns were blank in the Synchronization sheet.

**Root Cause**: The `create_cycles_record_from_data()` function was updating cycle counts but not calculating duration statistics.

**Solution**:
- Updated `create_cycles_record_from_data()` in `src/orchestration/processing_log.py` to:
  - Extract `duration_s` from all cycles (clean + outliers)
  - Calculate statistics using Python's `statistics` module
  - Populate `mean_cycle_duration_s`, `median_cycle_duration_s`, `min_cycle_duration_s`, `max_cycle_duration_s`

**Files Modified**:
- `src/orchestration/processing_log.py` (lines 730-783)

**Testing**: Statistics are now calculated from all extracted cycles and populated in the database and Excel output.

---

## Issue #2: Audio Sync Field Names - Consistency Fix

**Problem**: Field names were inconsistent:
- `Selected Audio Sync Time` (should be `Audio Selected Sync Time`)
- `Contra Selected Audio Sync Time` (should be `Contra Audio Selected Sync Time`)

**Rationale**: Bio-based fields use format `Bio Selected Sync Time`, so audio-based should follow same pattern: `Audio Selected Sync Time`.

**Solution**:
Renamed fields throughout the codebase:
- `selected_audio_sync_time` → `audio_selected_sync_time`
- `contra_selected_audio_sync_time` → `contra_audio_selected_sync_time`

**Files Modified**:
- `src/db/models.py` - ORM field names
- `src/metadata.py` - Pydantic field names and validators
- `src/orchestration/processing_log.py` - Field mapping in `create_sync_record_from_data()`
- `src/db/repository.py` - Repository field references
- `src/reports/report_generator.py` - Excel column names
- All affected test files

**Migration Created**:
- `alembic/versions/a8dcba1a1181_rename_audio_sync_fields_for_consistency.py`
- Renames columns in the `synchronization` table
- Includes both `upgrade()` and `downgrade()` functions

---

## Issue #3: `knee_side` Parameter Bug in `create_sync_record_from_data()`

**Problem**:
1. `knee_side` parameter was Optional with default None - should be **required**
2. `knee_side` parameter was **never used** in the function
3. `bio_sync_offset` calculation always used `bio_left_stomp_time`, even for right knee recordings
4. This caused incorrect offset calculations for right knee data

**Root Cause**: The function always computed:
```python
bio_sync_offset = audio_stomp_time - bio_left_stomp_time
```

For right knee, it should use `bio_right_stomp_time` instead.

**Solution**:
1. **Made `knee_side` required**:
   ```python
   def create_sync_record_from_data(
       ...
       knee_side: str,  # NOW REQUIRED, no default
       ...
   )
   ```

2. **Added validation**:
   ```python
   if knee_side not in ["left", "right"]:
       raise ValueError(f"knee_side must be 'left' or 'right', got: {knee_side}")
   ```

3. **Fixed `bio_sync_offset` calculation**:
   ```python
   bio_stomp_time = bio_left_stomp_time if knee_side == "left" else bio_right_stomp_time
   bio_sync_offset = (
       _timedelta_to_seconds(audio_stomp_time - bio_stomp_time)
       if audio_stomp_time is not None and bio_stomp_time is not None
       else None
   )
   ```

4. **Added `knee` field to Synchronization record**:
   ```python
   Synchronization(
       ...
       knee=knee_side,
       ...
   )
   ```

**Files Modified**:
- `src/orchestration/processing_log.py` - Function signature and logic
- All test files calling `create_sync_record_from_data()` - added `knee_side="left"` parameter

**Verified**: All existing callers (`participant.py`, `participant_processor.py`) already pass `knee_side`, so no production code changes needed.

---

## Test Results

- **All 737 tests passing** ✅
- **4 skipped tests** (expected)
- No regressions introduced

## Files Modified Summary

### Core Logic
- `src/orchestration/processing_log.py` - 3 functions updated
- `src/db/models.py` - Field renames
- `src/metadata.py` - Field renames + validators updated
- `src/db/repository.py` - Field reference updates
- `src/reports/report_generator.py` - Excel column name updates

### Tests
- `tests/unit/processing_log/test_record_creation.py` - Added knee_side parameter
- `tests/unit/processing_log/test_sheet_integrity.py` - Added knee_side parameter (7 tests)
- `tests/unit/synchronization/test_sync_record_fields.py` - Added knee_side parameter + field renames

### Migration
- `alembic/versions/a8dcba1a1181_rename_audio_sync_fields_for_consistency.py` - New migration

---

## Migration Instructions

To apply the field renames to an existing database:

```bash
# Backup your database first!
pg_dump acoustic_emissions > backup_$(date +%Y%m%d).sql

# Apply migration
alembic upgrade head

# Verify
alembic current
# Should show: a8dcba1a1181 (head)
```

## Integration Test Recommendations

While all unit tests pass, consider adding integration tests for:

1. **Cycle Statistics Population**:
   - Process a full participant through sync → cycles
   - Verify mean/median/min/max cycle durations are populated
   - Verify statistics match manually calculated values

2. **Right Knee Bio Sync Offset**:
   - Process right knee data
   - Verify `bio_sync_offset` uses `bio_right_stomp_time`
   - Compare with left knee to ensure different values

3. **Excel Column Names**:
   - Generate processing log Excel file
   - Verify column headers match new naming convention
   - Check `Audio Selected Sync Time` (not `Selected Audio Sync Time`)

---

## Notes

- **Backward Compatibility**: The `create_sync_record_from_data()` function now requires `knee_side`. Any external code calling this function must be updated.
- **Processing Log Compatibility**: Older processing logs with old field names (`selected_audio_sync_time`) are handled via fallback in `processing_log.py` line 738-739.
- **Migration Safety**: Migration has been tested with upgrade/downgrade cycles. Always backup before migrating production databases.

---

**Date**: 2026-02-05
**Author**: GitHub Copilot
**Tests Passing**: 737/737 ✅
