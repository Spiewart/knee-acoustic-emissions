# Synchronization Processing Log Schema Changes

## Overview

This document describes the schema changes made to the `synchronizations` table to improve clarity and support new synchronization methods (audio, consensus, biomechanics).

## Key Concepts

- **Biomechanics is synchronized to audio**: The entire audio recording is included, and audio time 0 is the synchronized data time 0.
- **Aligned Sync Time**: A unified field replacing the redundant `audio_sync_time`, `aligned_audio_sync_time`, and `aligned_biomechanics_sync_time`.
- **Multiple Detection Methods**: Supports using multiple stomp detection methods (audio, consensus, biomechanics) with a selected method.

## Database Schema Changes

### Fields Removed
1. `audio_sync_time` - Redundant with aligned fields
2. `aligned_audio_sync_time` - Redundant
3. `aligned_biomechanics_sync_time` - Redundant
4. `audio_qc_version` - QC done at other stages
5. `biomech_qc_version` - QC done at other stages
6. `cycle_qc_version` - QC done at other stages

### Fields Renamed
1. `sync_offset` → `bio_sync_offset` - Clarifies this is biomechanics sync offset between legs
2. `selected_audio_sync_time` → `bio_selected_sync_time` - Clarifies this is biomechanics-based
3. `contra_selected_audio_sync_time` → `contra_bio_selected_sync_time` - Clarifies this is biomechanics-based
4. `audio_stomp_method` → `selected_stomp_method` - Now a single selected method

### Fields Added
1. `aligned_sync_time` (Float, nullable) - Unified aligned sync time on merged dataframes
2. `stomp_detection_methods` (ARRAY of String, nullable) - List of methods used: ['audio', 'consensus', 'biomechanics']
3. `audio_sync_time_left` (Float, nullable) - Optional: time between mic on and participant stopping (left leg)
4. `audio_sync_time_right` (Float, nullable) - Optional: time between mic on and participant stopping (right leg)
5. `audio_sync_offset` (Float, nullable) - Required if both left and right audio sync times present
6. `selected_audio_sync_time` (Float, nullable) - NEW: Required if 'audio' in stomp_detection_methods
7. `contra_selected_audio_sync_time` (Float, nullable) - NEW: Required if 'audio' in stomp_detection_methods

## Alembic Migration Script

When Alembic is initialized, create a migration with the following operations:

```python
def upgrade():
    # Remove redundant fields
    op.drop_column('synchronizations', 'audio_sync_time')
    op.drop_column('synchronizations', 'aligned_audio_sync_time')
    op.drop_column('synchronizations', 'aligned_biomechanics_sync_time')
    op.drop_column('synchronizations', 'audio_qc_version')
    op.drop_column('synchronizations', 'biomech_qc_version')
    op.drop_column('synchronizations', 'cycle_qc_version')

    # Rename fields
    op.alter_column('synchronizations', 'sync_offset', new_column_name='bio_sync_offset')
    op.alter_column('synchronizations', 'selected_audio_sync_time', new_column_name='bio_selected_sync_time')
    op.alter_column('synchronizations', 'contra_selected_audio_sync_time', new_column_name='contra_bio_selected_sync_time')
    op.alter_column('synchronizations', 'audio_stomp_method', new_column_name='selected_stomp_method')

    # Add new fields
    op.add_column('synchronizations', sa.Column('aligned_sync_time', sa.Float(), nullable=True))
    op.add_column('synchronizations', sa.Column('stomp_detection_methods', sa.ARRAY(sa.String()), nullable=True))
    op.add_column('synchronizations', sa.Column('audio_sync_time_left', sa.Float(), nullable=True))
    op.add_column('synchronizations', sa.Column('audio_sync_time_right', sa.Float(), nullable=True))
    op.add_column('synchronizations', sa.Column('audio_sync_offset', sa.Float(), nullable=True))
    op.add_column('synchronizations', sa.Column('selected_audio_sync_time', sa.Float(), nullable=True))
    op.add_column('synchronizations', sa.Column('contra_selected_audio_sync_time', sa.Float(), nullable=True))

def downgrade():
    # Remove new fields
    op.drop_column('synchronizations', 'contra_selected_audio_sync_time')
    op.drop_column('synchronizations', 'selected_audio_sync_time')
    op.drop_column('synchronizations', 'audio_sync_offset')
    op.drop_column('synchronizations', 'audio_sync_time_right')
    op.drop_column('synchronizations', 'audio_sync_time_left')
    op.drop_column('synchronizations', 'stomp_detection_methods')
    op.drop_column('synchronizations', 'aligned_sync_time')

    # Revert renames
    op.alter_column('synchronizations', 'selected_stomp_method', new_column_name='audio_stomp_method')
    op.alter_column('synchronizations', 'contra_bio_selected_sync_time', new_column_name='contra_selected_audio_sync_time')
    op.alter_column('synchronizations', 'bio_selected_sync_time', new_column_name='selected_audio_sync_time')
    op.alter_column('synchronizations', 'bio_sync_offset', new_column_name='sync_offset')

    # Re-add removed fields
    op.add_column('synchronizations', sa.Column('cycle_qc_version', sa.String(20), nullable=True))
    op.add_column('synchronizations', sa.Column('biomech_qc_version', sa.String(20), nullable=True))
    op.add_column('synchronizations', sa.Column('audio_qc_version', sa.String(20), nullable=True))
    op.add_column('synchronizations', sa.Column('aligned_biomechanics_sync_time', sa.Float(), nullable=True))
    op.add_column('synchronizations', sa.Column('aligned_audio_sync_time', sa.Float(), nullable=True))
    op.add_column('synchronizations', sa.Column('audio_sync_time', sa.Float(), nullable=True))
```

## Excel Report Column Changes

### Columns Removed
- Audio Sync Time
- Aligned Audio Sync Time
- Aligned Biomechanics Sync Time
- Audio QC Version
- Biomech QC Version
- Cycle QC Version

### Columns Renamed
- Sync Offset → Bio Sync Offset
- Selected Audio Sync Time → Bio Selected Sync Time
- Contra Selected Audio Sync Time → Contra Bio Selected Sync Time
- Audio Stomp Method → Selected Stomp Method

### Columns Added
- Aligned Sync Time
- Stomp Detection Methods (displayed as comma-separated list)
- Audio Sync Time Left
- Audio Sync Time Right
- Audio Sync Offset
- Selected Audio Sync Time (new audio-based)
- Contra Selected Audio Sync Time (new audio-based)

### Column Reordering
- Method Agreement Span moved next to consensus-related columns

## Code Changes Summary

### Files Modified
1. `src/db/models.py` - Updated SynchronizationRecord model
2. `src/metadata.py` - Updated Synchronization Pydantic model
3. `src/reports/report_generator.py` - Updated Excel column mapping
4. `src/orchestration/processing_log.py` - Updated create_sync_record_from_data()
5. `src/db/repository.py` - Updated synchronization record creation/update
6. `src/synchronization/sync.py` - Updated detection results field names and documentation
7. `tests/unit/synchronization/test_sync_record_fields.py` - Updated test field names
8. `tests/unit/synchronization/test_sync_methods.py` - Fixed field name assertion

### Backward Compatibility Notes
- Old code using `audio_stomp_method` will need to use `selected_stomp_method`
- Old code using `selected_audio_sync_time` for biomechanics-based values will need to use `bio_selected_sync_time`
- The NEW `selected_audio_sync_time` is for audio-based detection (different purpose)
- Detection results dictionaries from `get_audio_stomp_time()` now use:
  - `stomp_detection_methods` (list) instead of just method indicator
  - `selected_stomp_method` (str) for the selected method
  - `bio_selected_time` and `contra_bio_selected_time` for biomechanics-based values

## Conditional Field Requirements

### bio_selected_sync_time and contra_bio_selected_sync_time
- Required when `'biomechanics'` is in `stomp_detection_methods`

### selected_audio_sync_time and contra_selected_audio_sync_time
- Required when `'audio'` is in `stomp_detection_methods`

### audio_sync_offset
- Required when both `audio_sync_time_left` and `audio_sync_time_right` are present

## Testing

All synchronization unit tests pass:
- ✅ test_synchronization_has_biomech_guided_fields
- ✅ test_synchronization_fields_include_new_values
- ✅ test_create_sync_record_populates_new_fields_from_detection_results
- ✅ test_synchronization_biomech_guided_can_be_none
- ✅ test_create_sync_record_consensus_without_biomech_guided
- ✅ test_synchronization_consensus_method_agreement_span_calculation
- ✅ test_synchronization_audio_sync_times (NEW)
- ✅ test_synchronization_audio_based_sync_fields (NEW)
- ✅ test_synchronization_multiple_detection_methods (NEW)
- ✅ test_synchronization_agreement_span_with_partial_consensus

## Next Steps

1. ✅ Update database models
2. ✅ Update Pydantic models
3. ✅ Update report generator
4. ✅ Update processing_log
5. ✅ Update repository layer
6. ✅ Update synchronization detection code
7. ✅ Update unit tests
8. ⬜ Initialize Alembic and create migration
9. ⬜ Test migration on development database
10. ⬜ Run full integration tests
11. ⬜ Add field validators for conditional requirements (future enhancement)
12. ⬜ Implement actual audio stomp detection methods (future enhancement)
