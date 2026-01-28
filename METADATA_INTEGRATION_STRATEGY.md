# Metadata Integration Strategy

## Overview

Integrating the new metadata class hierarchy (WalkMetadata, SynchronizationMetadata, refactored Synchronization, refactored MovementCycle) requires changes across the processing pipeline and comprehensive testing. This document outlines the approach to minimize breakage and ensure data consistency.

## Key Integration Points

### 1. **processing_log.py** - Record Creation
   - **Current State**: Creates single `Synchronization` records with mixed data
   - **Issue**: Synchronization records currently contain cycle-level fields that should be moved to MovementCycle
   - **Change Needed**: Split record creation into two parallel tracks
     - Track 1: Create `Synchronization` records (sync process summary)
     - Track 2: Create `MovementCycle` records for each cycle (cycle-specific QC)

### 2. **participant_processor.py** - Sheet Writing
   - **Current State**: Saves maneuver processing log to single Excel file with sheets: Summary, Audio, Biomechanics, Synchronization, Movement Cycles
   - **Issue**: Need to ensure both Synchronization and MovementCycle records are properly saved to correct sheets
   - **Change Needed**: Update excel writing to handle new metadata structure

### 3. **All Tests** - Field Expectations
   - **Current State**: Tests expect old Synchronization/MovementCycle field layout
   - **Issue**: Will break when metadata classes are refactored
   - **Change Needed**: Comprehensive integration tests to catch issues early

## Phased Integration Approach

### Phase 1: Implement New Metadata Classes (NO FUNCTIONAL CHANGES YET)
**Goal**: Add new classes alongside old ones, ensure they work in isolation

**Tasks**:
1. Create `WalkMetadata` mixin
2. Create `SynchronizationMetadata` class  
3. Create empty integration test file (`test_metadata_integration.py`)
4. Verify new classes can be instantiated in isolation

**Success Criteria**:
- All new classes instantiate correctly
- Field inheritance works as designed
- No breaking changes to existing code

**Files Modified**:
- `src/metadata.py` (additions only)
- `tests/test_metadata_integration.py` (new file)

---

### Phase 2: Update Record Creation in processing_log.py
**Goal**: Split Synchronization record creation, prepare both old and new formats in parallel

**Key Decision**: "Dual-track mode"
- Continue creating old-format `Synchronization` records (for backward compatibility)
- ALSO create new-format records for validation (in-memory, not saved yet)
- Compare the two formats to catch discrepancies early

**Tasks**:
1. Add helper function `_create_synchronization_metadata()` that extracts stomp times and sync method
2. Add logic to create `SynchronizationMetadata` alongside existing `Synchronization` creation
3. Add validation in `ManeuverProcessingLog` to ensure fields match between old/new formats
4. Add dual-track logic to `add_movement_cycle()` method
5. Add comprehensive logging of what data goes where

**Methods to Update**:
- `ManeuverProcessingLog.add_synchronization_records()`
- `SynchronizationLogHelper.process_synchronization()` (or equivalent)
- `SynchronizationLogHelper.add_cycles()` (or equivalent)

**Success Criteria**:
- Old format `Synchronization` records still created correctly
- New metadata created in parallel without breaking anything
- Validation catches any field mismatches
- All existing tests still pass

**Files Modified**:
- `src/orchestration/processing_log.py` (new helper functions, validation logic)

---

### Phase 3: Refactor Synchronization and MovementCycle Classes
**Goal**: Update class definitions to use new inheritance structure

**Tasks**:
1. Update `Synchronization` class:
   - Change parent from `SynchronizationMetadata` (after Phase 2 is working)
   - Remove cycle-level fields (biomechanics_qc_fail, sync_qc_fail, per_cycle_details)
   - Keep aggregate statistics (clean_cycles, mean_duration_s, etc.)
   - Override linked_biomechanics = True

2. Update `MovementCycle` class:
   - Change parents to `SynchronizationMetadata, AudioProcessing`
   - Keep only cycle-specific fields (cycle_file, cycle_index, start_time_s, etc.)
   - Add biomechanics_qc_fail, sync_qc_fail (cycle-level only)
   - Inherit all audio QC fields from AudioProcessing

**Success Criteria**:
- All fields properly inherited
- No field duplication
- Old tests still pass (because we're updating class definitions, old code still works)

**Files Modified**:
- `src/metadata.py` (class definitions)

---

### Phase 4: Integration Testing
**Goal**: Comprehensive testing to ensure metadata flows correctly through entire pipeline

**New Test File**: `tests/test_metadata_integration.py`

**Test Categories**:

#### A. Field Population Tests
```python
def test_synchronization_has_sync_aggregate_fields()
    # Verify Synchronization contains: clean_cycles, mean_duration_s, etc.

def test_movement_cycle_has_cycle_specific_fields()
    # Verify MovementCycle contains: cycle_index, start_time_s, etc.

def test_movement_cycle_inherits_audio_qc_fields()
    # Verify all AudioProcessing QC fields present on MovementCycle

def test_synchronization_does_not_have_cycle_fields()
    # Verify Synchronization doesn't have: cycle_file, cycle_index, etc.

def test_walk_metadata_validates_pass_and_speed()
    # Verify pass_number/speed required for walk, optional for others
```

#### B. Data Flow Tests
```python
def test_full_pipeline_sync_data_flow()
    # Process participant through sync stage
    # Verify Synchronization record created with correct fields
    # Verify can be saved to Excel and read back

def test_full_pipeline_cycle_data_flow()
    # Process participant through cycle stage
    # Verify MovementCycle records created with all inherited fields
    # Verify can be saved to Excel and read back

def test_sync_and_cycle_sheets_both_present()
    # Verify Excel file has both Synchronization and Movement Cycles sheets
    # Verify correct data in each sheet

def test_dual_track_validation_consistency()
    # Verify new metadata matches old format for field values
    # Catch any discrepancies early
```

#### C. Inheritance Tests
```python
def test_synchronization_metadata_inheritance()
    # Verify SynchronizationMetadata properly inherits from AcousticsFile, WalkMetadata

def test_movement_cycle_full_inheritance()
    # Verify MovementCycle inherits from SynchronizationMetadata and AudioProcessing
    # Verify all fields from all parents present

def test_no_field_duplication_across_parents()
    # Verify no field defined in multiple parent classes
    # Catch any field shadowing issues
```

#### D. Resume/Persistence Tests
```python
def test_resume_at_sync_stage_preserves_metadata()
    # Stop after sync, resume from sync
    # Verify Synchronization record properly restored

def test_resume_at_cycle_stage_preserves_metadata()
    # Stop after cycles, resume from cycles
    # Verify MovementCycle records properly restored
    # Verify can still be saved to Excel

def test_load_from_excel_populates_all_fields()
    # Save to Excel, load from Excel
    # Verify all inherited fields present
```

#### E. QC Field Tests
```python
def test_biomechanics_qc_fail_on_cycle_only()
    # Verify biomechanics_qc_fail is on MovementCycle, not Synchronization

def test_sync_qc_fail_on_cycle_only()
    # Verify sync_qc_fail is on MovementCycle, not Synchronization

def test_audio_qc_fields_on_cycle()
    # Verify qc_artifact, qc_signal_dropout, etc. on MovementCycle
    # Not on Synchronization
```

#### F. Sheet Writing Tests
```python
def test_synchronization_sheet_has_correct_columns()
    # Verify "Synchronization" sheet has all Synchronization fields
    # Verify no cycle-specific fields

def test_movement_cycles_sheet_has_correct_columns()
    # Verify "Movement Cycles" sheet has all MovementCycle fields
    # Verify includes inherited fields from SynchronizationMetadata and AudioProcessing

def test_excel_file_roundtrip_preserves_data()
    # Save to Excel, read back, verify field values match
```

#### G. Backward Compatibility Tests
```python
def test_old_log_files_can_be_loaded()
    # Ensure old Excel files can still be loaded if they exist

def test_resuming_from_old_logs()
    # If processing is resumed from old log format, verify graceful handling
```

**Files to Create**:
- `tests/test_metadata_integration.py` (new comprehensive integration test suite)

**Success Criteria**:
- All new integration tests pass
- All existing tests still pass
- Data consistency verified across entire pipeline
- Resume/persistence scenarios covered

---

### Phase 5: Refactor Processing Logic (If Needed)
**Goal**: Update processing_log.py to properly handle both record types

**Only if Phase 4 tests reveal issues**:
1. Update record creation to properly populate both old and new formats
2. Add explicit validation between formats
3. Add logging/debugging info for field tracking

**Files Modified**:
- `src/orchestration/processing_log.py` (production fixes only)

---

## Implementation Order

**Week 1**:
1. Phase 1: Add new metadata classes
2. Phase 2: Add dual-track record creation

**Week 2**:
1. Phase 3: Refactor class definitions
2. Phase 4: Create comprehensive integration tests
3. Run all tests, fix any failures

**Week 3**:
1. Phase 5: Fix production code if needed
2. Final validation across entire pipeline
3. Clean up and commit

---

## Integration Testing Architecture

### Test Fixture Setup

```python
# conftest.py additions
@pytest.fixture
def sample_processed_maneuver():
    """Fixture providing a fully processed maneuver through all stages."""
    # Create temp directory with sample data
    # Run through ParticipantProcessor
    # Stop at each stage (audio, biomech, sync, cycles)
    # Yield processed data

@pytest.fixture
def sample_synchronization_record():
    """Create valid Synchronization record with all required fields."""

@pytest.fixture
def sample_movement_cycle_record():
    """Create valid MovementCycle record with all inherited fields."""
```

### Validation Helpers

```python
def assert_has_all_fields(obj, expected_fields):
    """Verify object has all expected fields with correct types."""
    for field_name, field_type in expected_fields.items():
        assert hasattr(obj, field_name), f"Missing field: {field_name}"
        value = getattr(obj, field_name)
        if value is not None:
            assert isinstance(value, field_type), f"Field {field_name} wrong type"

def assert_no_field_duplication(cls):
    """Verify no field defined in multiple parent classes."""
    # Check class __dataclass_fields__ against all parent classes
    
def assert_all_parents_fields_inherited(child_cls):
    """Verify child has all fields from all parent classes."""
    # Check each parent's fields present in child
```

---

## Risk Mitigation

### Risk 1: Breaking Existing Functionality
**Mitigation**:
- Phase 2: Dual-track mode ensures old format still works
- Phase 1-2: Don't change class definitions until tests confirm new structure works
- All new code is additive (no deletions until Phase 3)

### Risk 2: Field Value Mismatches
**Mitigation**:
- Phase 2: Validation compares old vs new format side-by-side
- Phase 4: Integration tests verify field values match at each stage
- Add extensive logging to track field population

### Risk 3: Resume/Persistence Issues
**Mitigation**:
- Phase 4: Dedicated tests for resume at each stage
- Phase 4: Tests for loading from Excel and reading back
- Include backward compatibility for old log files

### Risk 4: Sheet Writing Errors
**Mitigation**:
- Phase 4: Tests verify correct columns in each sheet
- Phase 4: Tests verify field values match across save/load cycle
- Phase 4: End-to-end tests that save and read back from Excel

### Risk 5: Validation/Type Errors
**Mitigation**:
- Phase 4: Tests instantiate records with all field combinations
- Phase 1: New classes tested in isolation first
- Add type hints and validators to catch issues early

---

## Checklist for Success

- [ ] Phase 1 Complete: New classes added, isolated tests pass
- [ ] Phase 2 Complete: Dual-track record creation works, validation passes
- [ ] Phase 3 Complete: Class definitions refactored, old tests still pass
- [ ] Phase 4 Complete: All integration tests written and passing
- [ ] Phase 5 Complete: Any production fixes applied and tested
- [ ] All 600+ existing tests still pass
- [ ] No test skips or failures
- [ ] Excel files save and load correctly
- [ ] Resume scenarios work at all entry points
- [ ] Documentation updated with new field structure
- [ ] Git commit with all changes

---

## Key Metrics to Monitor

1. **Test Pass Rate**: Should stay at 100% (currently 588 passing, 1 skipped)
2. **Test Count**: Should increase (adding comprehensive integration tests)
3. **Code Coverage**: Should not decrease (new classes should have tests)
4. **Execution Time**: Should not increase significantly (no slow operations added)
5. **Field Consistency**: Dual-track validation should catch 100% of discrepancies

---

## Long-term Maintenance

After integration complete:
1. Keep integration test suite as part of CI/CD
2. Any future changes to metadata must update tests
3. Regular validation of sheet structure against expected columns
4. Monitor for field value discrepancies in production logs
