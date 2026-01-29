# DB Normalization Implementation Checklist

**Status**: Phase 1 Complete ✅, Phase 2-6 Pending  
**Last Updated**: January 29, 2026

---

## 📊 PHASE 1: ORM Redesign ✅ COMPLETED

### What Was Done
- [x] Analyzed inheritance-based structure
- [x] Designed normalized FK-based schema
- [x] Rewrote AudioProcessingRecord (removed biomechanics fields)
- [x] Rewrote BiomechanicsImportRecord (standalone with import stats)
- [x] Rewrote SynchronizationRecord (FK to audio + biomech)
- [x] Rewrote MovementCycleRecord (FK to audio, optional biomech, optional sync)
- [x] Removed MovementCycleDetailRecord (simplified)
- [x] Added proper relationships with cascade deletes
- [x] Updated constraints (knee, maneuver, status validation)
- [x] Documented in `docs/DB_NORMALIZATION_PLAN.md`

### Files Modified
- `src/db/models.py` - Complete redesign

### Status
✅ Ready for Phase 2

---

## 📝 PHASE 2: Pydantic Model Refactoring 🔜 NEXT

### Complexity: HIGH (~800 lines to rewrite)
**Estimated Time**: 2-3 hours

### What Needs Doing

#### A. AudioProcessing Model
- [ ] Remove: BiomechanicsMetadata inheritance
- [ ] Remove: AcousticsFile inheritance
- [ ] Remove: SynchronizationMetadata inheritance
- [ ] Remove: walk-specific fields (pass_number, speed)
- [ ] Remove: Biomechanics linkage fields
- [ ] Keep: All audio file metadata, QC fields
- [ ] Add: audio_qc_version from get_audio_qc_version()
- [ ] Remove: to_dict() method or simplify
- [ ] Remove: Validators that reference inherited fields

**Key Fields to Keep**
```python
study, study_id (from parent)
audio_file_name, device_serial, firmware_version, file_time, file_size_mb
recording_date, recording_time
knee, maneuver
num_channels, sample_rate
mic_1/2/3/4_position, mic_1/2/3/4_notes, notes
pkl_file_path
audio_qc_version
qc_not_passed, qc_not_passed_mic_1/2/3/4
qc_fail_segments, qc_fail_segments_ch1/2/3/4
qc_signal_dropout, qc_signal_dropout_segments, qc_signal_dropout_ch1/2/3/4
qc_artifact, qc_artifact_type, qc_artifact_segments, qc_artifact_ch1/2/3/4
processing_date, processing_status, error_message, duration_seconds
```

#### B. BiomechanicsImport Model (NEW)
- [ ] Create standalone model
- [ ] Inherit only from StudyMetadata
- [ ] Add: all biomechanics-specific fields
- [ ] Add: import statistics (num_sub_recordings, duration_seconds, num_data_points, num_passes)
- [ ] Add: walk-specific fields (pass_number, speed)
- [ ] Add: Validators for walk maneuver requirements
- [ ] Add: Validators for biomechanics field requirements

**Key Fields**
```python
study, study_id (from parent)
biomechanics_file, sheet_name
biomechanics_type (Gonio, IMU, Motion Analysis)
knee, maneuver
pass_number, speed (optional, required for walk)
biomechanics_sync_method, biomechanics_sample_rate, biomechanics_notes
num_sub_recordings, duration_seconds, num_data_points, num_passes
biomech_qc_version, biomechanics_qc_fail, biomechanics_qc_notes
processing_date, processing_status, error_message
```

#### C. Synchronization Model
- [ ] Remove: All audio file metadata duplication
- [ ] Remove: All biomechanics metadata duplication
- [ ] Add: audio_processing_id (int)
- [ ] Add: biomechanics_import_id (int)
- [ ] Keep: Sync-specific fields
- [ ] Keep: Walk-specific fields (pass_number, speed)
- [ ] Keep: Aggregate statistics
- [ ] Add: Validator to ensure both IDs are set
- [ ] Update: to_dict() to handle FK references

**Key Fields**
```python
study, study_id (from parent)
audio_processing_id, biomechanics_import_id (NEW - FKs)
pass_number, speed
audio_sync_time, bio_left_sync_time, bio_right_sync_time, sync_offset
aligned_audio_sync_time, aligned_biomechanics_sync_time
sync_method, consensus_methods, consensus_time, rms_time, onset_time, freq_time
selected_audio_sync_time, contra_selected_audio_sync_time
audio_visual_sync_time, audio_visual_sync_time_contralateral
audio_stomp_method
sync_file_name, sync_file_path
sync_duration, total_cycles_extracted, clean_cycles, outlier_cycles
mean/median/min/max_cycle_duration_s, method_agreement_span
audio_qc_version, biomech_qc_version, cycle_qc_version
sync_qc_fail, processing_date, processing_status, error_message
```

#### D. MovementCycle Model
- [ ] Remove: All audio file metadata duplication
- [ ] Remove: All biomechanics metadata duplication
- [ ] Remove: All sync metadata duplication
- [ ] Add: audio_processing_id (int, required)
- [ ] Add: biomechanics_import_id (int, optional)
- [ ] Add: synchronization_id (int, optional)
- [ ] Keep: Cycle-specific fields only
- [ ] Add: Validator for bio/sync consistency
- [ ] Update: to_dict() to handle FK references

**Key Fields**
```python
study, study_id (from parent)
audio_processing_id, biomechanics_import_id, synchronization_id (NEW - FKs)
cycle_file, cycle_index, is_outlier
start_time_s, end_time_s, duration_s
audio_start_time, audio_end_time
bio_start_time, bio_end_time (optional)
biomechanics_qc_fail, sync_qc_fail
biomechanics_qc_version, sync_qc_version
cycle_file_path, cycle_file_checksum, cycle_file_size_mb, cycle_file_modified
```

### Files to Modify
- `src/metadata.py` - Completely rewrite (backup as `src/metadata_old.py` already made)

### Files That May Need Updates
- Any file that imports from `src.metadata` and expects certain field inheritance

### Validation Rules to Add/Update
- [ ] AudioProcessing: No walk-specific field validation needed
- [ ] BiomechanicsImport: Require pass_number/speed if maneuver="walk"
- [ ] Synchronization: Require both audio_processing_id and biomechanics_import_id
- [ ] MovementCycle: If biomechanics_import_id is set, sync_qc_fail must be False if not synced

---

## 🔧 PHASE 3: Repository Layer Refactor

### Complexity: HIGH (~591 lines to update)
**Estimated Time**: 2-3 hours

### Changes Needed

#### New Methods to Create
- [ ] `save_biomechanics_import(biomech: BiomechanicsImport) -> BiomechanicsImportRecord`
- [ ] `get_biomechanics_import(study, study_id, knee, maneuver) -> BiomechanicsImportRecord`
- [ ] `get_audio_processing(study, study_id, knee, maneuver) -> AudioProcessingRecord`

#### Methods to Refactor
- [ ] `save_audio_processing()` - Remove biomechanics field handling
- [ ] `save_synchronization()` - Accept IDs instead of full audio/biomech objects
- [ ] `save_movement_cycle()` - Accept IDs instead of inherited fields
- [ ] `_create_audio_processing_record()` - Simplify
- [ ] `_create_synchronization_record()` - Add FK support
- [ ] `_create_movement_cycle_record()` - Add FK support
- [ ] Query methods - Add joins where needed

#### Example Changes
```python
# OLD
def save_synchronization(self, sync: Synchronization) -> SynchronizationRecord:
    # Gets participant, creates record with all audio/biomech data
    
# NEW
def save_synchronization(self, sync: Synchronization) -> SynchronizationRecord:
    # Gets participant
    # Validates that audio_processing_id and biomechanics_import_id exist
    # Creates record with FKs only
```

### Files to Modify
- `src/db/repository.py` - Major refactoring

---

## 🧪 PHASE 4: Test Updates

### Complexity: MEDIUM (~355 lines to update)
**Estimated Time**: 1-2 hours

### Changes Needed
- [ ] Update fixtures to create AudioProcessing without biomechanics
- [ ] Update fixtures to create BiomechanicsImport separately
- [ ] Create Synchronization with IDs (not full objects)
- [ ] Create MovementCycle with IDs
- [ ] Update test queries to use FKs
- [ ] Add tests for relationships/cascades

### Files to Modify
- `tests/test_database.py` - Complete rewrite of test fixtures and methods

---

## 🔄 PHASE 5: Orchestration Pipeline Updates

### Complexity: MEDIUM-HIGH (Multiple files)
**Estimated Time**: 2-3 hours

### Affected Files
- `src/orchestration/participant.py` - Main processing orchestrator
- `src/orchestration/processing_log.py` - Excel/DB writing
- Audio processing modules
- Biomechanics processing modules
- Sync processing modules
- Cycle processing modules

### Key Changes
```python
# OLD FLOW: Pass inheritance-based objects through pipeline
audio_proc = AudioProcessing(...)
sync = Synchronization(...)  # inherits audio fields from parent
cycle = MovementCycle(...)   # inherits from sync

# NEW FLOW: Save at each stage, pass IDs
audio_proc = AudioProcessing(...)
audio_record = repo.save_audio_processing(audio_proc)

biomech = BiomechanicsImport(...)
biomech_record = repo.save_biomechanics_import(biomech)

sync = Synchronization(
    audio_processing_id=audio_record.id,
    biomechanics_import_id=biomech_record.id,
    ...
)
sync_record = repo.save_synchronization(sync)

cycle = MovementCycle(
    audio_processing_id=audio_record.id,
    biomechanics_import_id=biomech_record.id,
    synchronization_id=sync_record.id,
    ...
)
```

### Search Patterns
- [ ] Find all `AudioProcessing(` instantiations
- [ ] Find all `Synchronization(` instantiations
- [ ] Find all `MovementCycle(` instantiations
- [ ] Check all places that pass objects to repository methods

---

## 📚 PHASE 6: Integration & Phase 2

### Complexity: MEDIUM (Testing + cleanup)
**Estimated Time**: 1-2 hours

### Tasks
- [ ] Integration test: Full processing pipeline with DB writes
- [ ] Test with actual audio/biomech data
- [ ] Verify data integrity in database
- [ ] Test query operations
- [ ] Implement Phase 2 dual-write logic (if proceeding)
- [ ] Update README and QUICKSTART documentation

---

## ⚠️ Known Issues / Decisions

### Q: Where do walk-specific fields live?
**Decision**: In Synchronization and MovementCycle, not in audio/biomech
- Rationale: Audio file can have multiple passes, biomech has one, sync/cycle are pass-specific

### Q: How to handle queries?
**Decision**: Repository methods will need to join tables
- Example: `repo.get_audio_processing()` returns AudioProcessingRecord, but to get full context, need joins

### Q: What about backward compat with Excel?
**Decision**: None needed per user requirement
- No Excel export required during this phase

### Q: How to handle existing Excel files?
**Decision**: Ignore them
- If needed later, create migration script in Phase 7

---

## 📋 Session Planning Guide

### Recommended Session Breakdown

**Session 2** (2-3 hours)
- Task: Complete Pydantic model rewrite
- Deliverable: src/metadata.py compiles and imports work
- Files: src/metadata.py only
- Testing: Can import models, create instances

**Session 3** (2-3 hours)
- Task: Repository layer refactoring
- Deliverable: Repository methods work with new FK-based models
- Files: src/db/repository.py
- Testing: Basic CRUD operations pass

**Session 4** (1-2 hours)
- Task: Test rewrite
- Deliverable: tests/test_database.py passes
- Files: tests/test_database.py
- Testing: 8 tests pass

**Session 5** (2-3 hours)
- Task: Orchestration pipeline updates
- Deliverable: Processing pipeline creates proper FK-based objects
- Files: src/orchestration/*.py, cli/*.py
- Testing: Individual processing modules work

**Session 6** (1-2 hours)
- Task: Integration test + Phase 2 prep
- Deliverable: Full pipeline works, ready for Phase 2 dual-write
- Files: Tests, docs
- Testing: End-to-end processing works

---

## 🚀 Next Session Checklist

When starting Session 2, verify:
- [ ] ORM changes committed to git
- [ ] This checklist document created
- [ ] src/metadata_old.py backup exists
- [ ] No uncommitted changes in other files

Then proceed with:
1. Open src/metadata.py (the rewrite will be substantial)
2. Start with AudioProcessing model
3. Verify each model compiles before moving to next
4. Test imports work at end of session

---

## 📌 References

- Design: `docs/DB_NORMALIZATION_PLAN.md`
- Old Models: `src/metadata_old.py` (backup)
- ORM Reference: `src/db/models.py` (already updated)
- Tests: `tests/test_database.py` (will be updated)

