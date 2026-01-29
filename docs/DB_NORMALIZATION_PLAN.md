# Database Normalization Refactoring Plan

**Date**: January 29, 2026  
**Branch**: `features/postgres`  
**Status**: IN PROGRESS

## Overview

Refactoring from inheritance-based Pydantic models (designed for flat file export) to normalized relational models using foreign key references.

## Phase 1: ORM Models ✅ COMPLETED

### Changes Made
- **AudioProcessingRecord**: Removed biomechanics fields, added relationships
- **BiomechanicsImportRecord**: Standalone table with all biomech-specific fields
- **SynchronizationRecord**: Now references audio_processing_id and biomechanics_import_id via FKs
- **MovementCycleRecord**: References audio_processing_id, biomechanics_import_id (optional), synchronization_id (optional)
- Removed MovementCycleDetailRecord (simplified to single table)

### New Schema Structure
```
StudyRecord
  └─ ParticipantRecord
      ├─ AudioProcessingRecord
      │   ├─ SynchronizationRecord (FK to audio + biomech)
      │   └─ MovementCycleRecord (FK to audio, optional biomech, optional sync)
      └─ BiomechanicsImportRecord
          ├─ SynchronizationRecord (FK to audio + biomech)
          └─ MovementCycleRecord (FK to audio, optional biomech, optional sync)
```

## Phase 2: Pydantic Models 🔄 IN PROGRESS

### Current Model Structure (Inheritance-Based)
```python
StudyMetadata (study, study_id)
  └─ BiomechanicsMetadata (linked_biomechanics, biomech fields)
      └─ AcousticsFile (audio file metadata)
          └─ SynchronizationMetadata (sync times, methods)
              └─ AudioProcessing (QC fields)
                  └─ Synchronization (aggregate stats)
                  └─ MovementCycle (cycle-specific)
```

### New Model Structure (FK-Based)
```python
# Base class - keep as-is
StudyMetadata (study, study_id)

# Standalone models - no inheritance beyond StudyMetadata
AudioProcessing(StudyMetadata):
    - All audio file metadata
    - QC fields
    - NO biomechanics fields
    - NO walk-specific fields (pass_number, speed)
    
BiomechanicsImport(StudyMetadata):
    - All biomechanics import metadata
    - Import statistics
    - QC fields
    - Walk-specific fields (pass_number, speed) IF maneuver="walk"

Synchronization(StudyMetadata):
    - audio_processing_id: int
    - biomechanics_import_id: int
    - Sync times and methods
    - Aggregate cycle statistics
    - Walk-specific fields (pass_number, speed) from parents
    - QC versions

MovementCycle(StudyMetadata):
    - audio_processing_id: int
    - biomechanics_import_id: Optional[int]
    - synchronization_id: Optional[int]
    - Cycle-specific fields only
    - Cycle timestamps
    - Cycle QC flags
```

### Key Design Decisions

1. **Study Metadata**: Keep as base class for participant/study identification
2. **No Multi-Level Inheritance**: Only StudyMetadata as parent
3. **Foreign Key Fields**: Use integer IDs to reference related records
4. **Walk-Specific Fields**: 
   - Store in AudioProcessing/BiomechanicsImport? NO - not audio/biomech specific
   - Store in Synchronization/MovementCycle? YES - these are maneuver execution records
   - But how to handle when audio file itself has pass_number? Need to think through this...

### Questions to Resolve

**Q1: Where do walk-specific fields live?**
- `pass_number` and `speed` relate to a specific execution of a walk maneuver
- Audio file can contain multiple passes
- Biomechanics file typically has one pass
- Sync is for a specific pass
- Cycles are from a specific pass

**Answer**: Walk fields should live at the maneuver-execution level:
- **AudioProcessing**: NO walk fields (file can contain multiple passes)
- **BiomechanicsImport**: YES walk fields (each import is for one pass)
- **Synchronization**: YES walk fields (syncing a specific pass)
- **MovementCycle**: Inherit from sync/biomech parent

**Q2: How to handle maneuver field?**
- Currently in AcousticsFile (knee + maneuver)
- Needed to identify what was recorded

**Answer**: Keep in both AudioProcessing and BiomechanicsImport
- Each record represents processing of one maneuver
- Sync/Cycle infer maneuver from parents via FKs

### Pydantic Model Changes

#### AudioProcessing
```python
@dataclass(kw_only=True)
class AudioProcessing(StudyMetadata):
    # File identification
    audio_file_name: str
    device_serial: str
    firmware_version: int
    file_time: datetime
    file_size_mb: float
    
    # Recording metadata
    recording_date: datetime
    recording_time: datetime
    
    # Maneuver identification
    knee: Literal["right", "left"]
    maneuver: Literal["fe", "sts", "walk"]
    
    # Audio characteristics
    num_channels: int
    sample_rate: float = 46875.0
    mic_1_position: Literal["IPM", "IPL", "SPM", "SPL"]
    mic_2_position: Literal["IPM", "IPL", "SPM", "SPL"]
    mic_3_position: Literal["IPM", "IPL", "SPM", "SPL"]
    mic_4_position: Literal["IPM", "IPL", "SPM", "SPL"]
    
    # Optional notes
    mic_1_notes: Optional[str] = None
    mic_2_notes: Optional[str] = None
    mic_3_notes: Optional[str] = None
    mic_4_notes: Optional[str] = None
    notes: Optional[str] = None
    
    # Pickle file
    pkl_file_path: Optional[str] = None
    
    # QC metadata (all the qc_* fields)
    audio_qc_version: int
    qc_not_passed: bool = False
    # ... all QC fields ...
    
    # Processing metadata
    processing_date: datetime
    processing_status: Literal["not_processed", "success", "error"] = "not_processed"
    error_message: Optional[str] = None
    duration_seconds: Optional[float] = None
```

#### BiomechanicsImport
```python
@dataclass(kw_only=True)
class BiomechanicsImport(StudyMetadata):
    # File identification
    biomechanics_file: str
    sheet_name: Optional[str] = None
    biomechanics_type: Literal["Gonio", "IMU", "Motion Analysis"]
    
    # Maneuver identification
    knee: Literal["right", "left"]
    maneuver: Literal["fe", "sts", "walk"]
    
    # Walk-specific (conditional)
    pass_number: Optional[int] = None
    speed: Optional[Literal["slow", "normal", "fast", "medium", "comfortable"]] = None
    
    # Biomechanics characteristics
    biomechanics_sync_method: Literal["flick", "stomp"]
    biomechanics_sample_rate: float
    biomechanics_notes: Optional[str] = None
    
    # Import statistics
    num_sub_recordings: int
    duration_seconds: float
    num_data_points: int
    num_passes: int = 0
    
    # QC metadata
    biomech_qc_version: int
    biomechanics_qc_fail: bool = False
    biomechanics_qc_notes: Optional[str] = None
    
    # Processing metadata
    processing_date: datetime
    processing_status: Literal["not_processed", "success", "error"] = "not_processed"
    error_message: Optional[str] = None
```

#### Synchronization
```python
@dataclass(kw_only=True)
class Synchronization(StudyMetadata):
    # Foreign keys (IDs to be populated after saving related records)
    audio_processing_id: int
    biomechanics_import_id: int
    
    # Walk-specific (inherited conceptually from parents, but stored here)
    pass_number: Optional[int] = None
    speed: Optional[Literal["slow", "normal", "fast", "medium", "comfortable"]] = None
    
    # Sync times
    audio_sync_time: Optional[float] = None
    bio_left_sync_time: Optional[float] = None
    bio_right_sync_time: Optional[float] = None
    sync_offset: Optional[float] = None
    aligned_audio_sync_time: Optional[float] = None
    aligned_biomechanics_sync_time: Optional[float] = None
    
    # Sync method details
    sync_method: Optional[Literal["consensus", "biomechanics"]] = None
    consensus_methods: Optional[str] = None
    consensus_time: Optional[float] = None
    rms_time: Optional[float] = None
    onset_time: Optional[float] = None
    freq_time: Optional[float] = None
    
    # Biomechanics-guided
    selected_audio_sync_time: Optional[float] = None
    contra_selected_audio_sync_time: Optional[float] = None
    audio_visual_sync_time: Optional[float] = None
    audio_visual_sync_time_contralateral: Optional[float] = None
    audio_stomp_method: Optional[str] = None
    
    # Pickle file
    sync_file_name: str
    sync_file_path: Optional[str] = None
    
    # Duration & stats
    sync_duration: Optional[float] = None
    total_cycles_extracted: int = 0
    clean_cycles: int = 0
    outlier_cycles: int = 0
    mean_cycle_duration_s: Optional[float] = None
    median_cycle_duration_s: Optional[float] = None
    min_cycle_duration_s: Optional[float] = None
    max_cycle_duration_s: Optional[float] = None
    method_agreement_span: Optional[float] = None
    
    # QC versions
    audio_qc_version: int
    biomech_qc_version: int
    cycle_qc_version: int
    sync_qc_fail: bool = False
    
    # Processing metadata
    processing_date: datetime
    processing_status: Literal["not_processed", "success", "error"] = "not_processed"
    error_message: Optional[str] = None
```

#### MovementCycle
```python
@dataclass(kw_only=True)
class MovementCycle(StudyMetadata):
    # Foreign keys
    audio_processing_id: int
    biomechanics_import_id: Optional[int] = None
    synchronization_id: Optional[int] = None
    
    # Cycle identification
    cycle_file: str
    cycle_index: int
    is_outlier: bool
    
    # Cycle temporal characteristics
    start_time_s: float
    end_time_s: float
    duration_s: float
    
    # Audio timestamps (always present)
    audio_start_time: datetime
    audio_end_time: datetime
    
    # Biomechanics timestamps (optional)
    bio_start_time: Optional[datetime] = None
    bio_end_time: Optional[datetime] = None
    
    # QC flags (cycle-specific)
    biomechanics_qc_fail: bool
    sync_qc_fail: bool
    
    # QC versions
    biomechanics_qc_version: int
    sync_qc_version: int
```

## Phase 3: Repository Layer 🔜 TODO

### Changes Needed

1. **save_audio_processing()**: Simplified, no biomech fields
2. **save_biomechanics_import()**: NEW method
3. **save_synchronization()**: Must accept IDs, fetch parent records
4. **save_movement_cycle()**: Must accept IDs, fetch parent records
5. **Query methods**: Need joins to fetch related data

### Example: Saving a Sync Record
```python
# Old way (all data in one object)
sync = Synchronization(...)  # 50+ fields
repo.save_synchronization(sync)

# New way (references)
audio = AudioProcessing(...)
audio_record = repo.save_audio_processing(audio)

biomech = BiomechanicsImport(...)
biomech_record = repo.save_biomechanics_import(biomech)

sync = Synchronization(
    audio_processing_id=audio_record.id,
    biomechanics_import_id=biomech_record.id,
    ...  # only sync-specific fields
)
repo.save_synchronization(sync)
```

## Phase 4: Tests 🔜 TODO

All test fixtures and helper functions need updating to use the new FK-based approach.

## Phase 5: Processing Pipeline Integration 🔜 TODO

### Files to Update
- `src/orchestration/participant.py` - main processing orchestration
- `src/orchestration/processing_log.py` - Excel/DB writing
- All audio/biomech/sync/cycle processing modules

### Key Changes
1. Save audio processing → get audio_record.id
2. Save biomech import → get biomech_record.id
3. Create sync with IDs
4. Create cycles with IDs

## Phase 6: Documentation 🔜 TODO

Update all docs to reflect normalized schema.

## Migration Strategy

Since no production data exists:
1. Drop old database
2. Recreate with new schema
3. No backfill needed

## Risks & Mitigation

**Risk**: Large refactor touches many files  
**Mitigation**: Work incrementally, test at each layer

**Risk**: Processing pipeline complex  
**Mitigation**: Update tests first, then implementation

**Risk**: Breaking changes to Pydantic models  
**Mitigation**: No backward compat needed per user

## Next Steps

1. ✅ Complete Pydantic model refactor
2. ⬜ Update repository layer
3. ⬜ Update tests
4. ⬜ Update processing pipeline
5. ⬜ Integration test with real data
