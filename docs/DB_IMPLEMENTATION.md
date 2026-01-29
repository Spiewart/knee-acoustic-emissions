# Database Schema + ORM Implementation Summary

**Date**: January 29, 2026
**Branch**: `features/postgres`

## What Was Created

### 1. Database Module (`src/db/`)

Complete PostgreSQL-backed metadata storage system with:

#### Core Files
- **`models.py`**: SQLAlchemy ORM models
  - `StudyRecord` - Study metadata (AOA, preOA, SMoCK)
  - `ParticipantRecord` - Participant records
  - `AudioProcessingRecord` - Audio file metadata + QC
  - `BiomechanicsImportRecord` - Biomechanics import metadata
  - `SynchronizationRecord` - Sync results
  - `MovementCycleRecord` - Cycle extraction summary
  - `MovementCycleDetailRecord` - Per-cycle details

- **`repository.py`**: High-level CRUD operations
  - `Repository` class with save/query methods
  - Abstracts SQLAlchemy session management
  - Converts between Pydantic models and ORM records

- **`session.py`**: Database connection management
  - `get_engine()` - Create SQLAlchemy engine
  - `get_session()` - Context manager for sessions
  - `init_db()` - Initialize schema
  - Reads `AE_DATABASE_URL` from environment

- **`__init__.py`**: Public API exports
- **`README.md`**: Comprehensive documentation

### 2. CLI Tools (`cli/db_admin.py`)

Database management commands:
- `ae-db-init` - Initialize database schema
- `ae-db-check` - Verify database connection

### 3. Configuration

- **`.env.example`**: Updated with database URL example
- **`requirements.txt`**: Added dependencies:
  - `sqlalchemy==2.0.36`
  - `psycopg==3.2.3` (PostgreSQL driver)
  - `alembic==1.15.1` (migrations, future use)

- **`setup.py`**: Added new CLI entry points

### 4. Tests (`tests/test_database.py`)

Comprehensive test suite:
- ORM model creation tests
- Unique constraint tests
- Repository CRUD operations
- Query filtering tests
- Uses SQLite in-memory for testing (no Postgres required)

## Schema Design

### Key Features

1. **Hierarchical Structure**
   ```
   Study → Participant → [Audio, Biomech, Sync, Cycles]
   ```

2. **Unique Constraints**
   - Prevents duplicate records per participant/maneuver/knee
   - Enforces data integrity

3. **File Storage Pattern**
   - Large `.pkl` files stay on disk
   - Database stores: path, checksum, size, modified time
   - Metadata fully queryable

4. **QC Metadata**
   - Per-channel QC flags
   - Artifact types (arrays)
   - Fail segments (arrays)
   - Signal dropout tracking

5. **Walk-Specific Fields**
   - `pass_number`, `speed` (nullable)
   - Only populated for walk maneuvers

### Schema Highlights

**Audio Processing** (most complex table):
- All file identification fields
- Microphone positions (4 channels)
- Per-channel QC flags and artifact types
- QC fail segments (stored as arrays)
- Signal dropout tracking per channel
- Biomechanics linkage metadata

**Synchronization**:
- All sync timing fields
- Sync method + consensus details
- QC flags and notes
- Pickle file path tracking

**Movement Cycles**:
- Summary metadata
- Links to cycle details (1:many)
- Per-cycle QC in separate table

## Usage Examples

### Initialize Database

```bash
# Set environment variable
export AE_DATABASE_URL="postgresql+psycopg://user:password@localhost:5432/acoustic_emissions"

# Initialize schema
ae-db-init

# Verify connection
ae-db-check
```

### Save Audio Processing Record

```python
from datetime import datetime
from src.db import get_session
from src.db.repository import Repository
from src.metadata import AudioProcessing

audio = AudioProcessing(
    study="AOA",
    study_id=1011,
    linked_biomechanics=False,
    audio_file_name="test.bin",
    device_serial="SN123",
    firmware_version=1,
    file_time=datetime(2024, 1, 1, 10, 0, 0),
    file_size_mb=100.0,
    recording_date=datetime(2024, 1, 1),
    recording_time=datetime(2024, 1, 1, 10, 0, 0),
    knee="left",
    maneuver="walk",
    num_channels=4,
    mic_1_position="IPM",
    mic_2_position="IPL",
    mic_3_position="SPM",
    mic_4_position="SPL",
)

with get_session() as session:
    repo = Repository(session)
    record = repo.save_audio_processing(audio, pkl_file_path="/path/to/audio.pkl")
```

### Query Records

```python
with get_session() as session:
    repo = Repository(session)

    # Find all walk maneuvers for participant 1011
    records = repo.get_audio_processing_records(
        study_name="AOA",
        participant_number=1011,
        maneuver="walk"
    )
```

## Migration Path

### Current: Phase 1 ✓
- Schema defined
- ORM models complete
- Repository layer functional
- CLI tools available
- Tests passing

### Next: Phase 2 (Dual Write)
- Integrate into `ManeuverProcessingLog`
- Write to both Excel AND database
- Feature flag: `AE_ENABLE_DB_WRITES`
- Maintain backward compatibility

### Future: Phase 3 (Read Path)
- Prefer database reads
- Fall back to Excel when DB empty
- Backfill script for existing logs

### Future: Phase 4 (Excel Deprecation)
- Optional Excel export command
- Database as primary source

## Testing

Run database tests:
```bash
pytest tests/test_database.py -v
```

All tests use SQLite in-memory database (no Postgres installation required for testing).

## Files Created/Modified

### Created
- `src/db/__init__.py`
- `src/db/models.py` (525 lines)
- `src/db/repository.py` (442 lines)
- `src/db/session.py` (85 lines)
- `src/db/README.md`
- `cli/db_admin.py`
- `tests/test_database.py`
- `docs/DB_IMPLEMENTATION.md` (this file)

### Modified
- `requirements.txt` - Added SQLAlchemy, psycopg, Alembic
- `setup.py` - Added CLI entry points

## Next Steps

1. **Test locally**:
   ```bash
   createdb acoustic_emissions
   export AE_DATABASE_URL="postgresql+psycopg://postgres@localhost:5432/acoustic_emissions"
   ae-db-init
   pytest tests/test_database.py
   ```

2. **Integrate with processing pipeline**:
   - Modify `ManeuverProcessingLog.save_to_excel()` to also write to DB
   - Add feature flag for dual-write mode
   - Test with real processing runs

3. **Create backfill script**:
   - Read existing Excel logs
   - Convert to Pydantic models
   - Save to database via repository

4. **Add Alembic migrations**:
   - Track schema changes
   - Version control for database evolution

## Design Decisions

1. **SQLAlchemy 2.0 style**: Modern declarative base with `Mapped[]` type hints
2. **Repository pattern**: Abstracts database operations from business logic
3. **Pydantic → ORM conversion**: Repository handles translation
4. **Array storage**: PostgreSQL ARRAY type for segments and artifact lists
5. **Timestamps**: Auto-managed `created_at` and `updated_at` fields
6. **Soft relationships**: Foreign keys with cascade deletes
7. **Unique constraints**: Prevent duplicate maneuver records
8. **Optional fields**: Nullable for walk-specific or conditional metadata

## Benefits

✓ **Queryable metadata**: SQL queries instead of parsing Excel files
✓ **Data integrity**: Foreign keys, unique constraints, type validation
✓ **Performance**: Indexed queries vs. file I/O
✓ **Scalability**: Supports concurrent access, read replicas
✓ **Audit trail**: Created/updated timestamps on all records
✓ **Type safety**: Pydantic validation + SQLAlchemy ORM
✓ **Testing**: SQLite in-memory for fast unit tests

## Notes

- Large `.pkl` files remain on disk (not in database)
- Database stores paths, checksums, and metadata only
- Backward compatible with existing Excel-based logs
- Can run Excel and DB side-by-side during migration
- No breaking changes to existing code
