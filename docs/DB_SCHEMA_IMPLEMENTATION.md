# Database Schema + ORM Implementation Summary

## Status: ✅ Implemented

The PostgreSQL schema and SQLAlchemy ORM have been implemented on the `features/postgres` branch.

## What's Included

### 1. **Database Models** (`src/db/models.py`)

Seven ORM models mirror the Pydantic dataclasses in `src/metadata.py`:

- **`StudyRecord`** - Top-level study (AOA, preOA, SMoCK)
- **`ParticipantRecord`** - Participant within a study
- **`AudioProcessingRecord`** - Audio file processing metadata and QC results
- **`BiomechanicsImportRecord`** - Biomechanics import metadata and QC
- **`SynchronizationRecord`** - Audio-biomechanics synchronization results
- **`MovementCycleRecord`** - Movement cycle summary statistics
- **`MovementCycleDetailRecord`** - Per-cycle details and QC flags

**Key features:**
- Proper foreign key relationships between tables
- Check constraints for valid enum values (knee, maneuver, study name, etc.)
- Unique constraints to prevent duplicates
- Timestamps (created_at, updated_at) for audit tracking
- Arrays for QC segments and artifact types (PostgreSQL-specific)

### 2. **Session Management** (`src/db/session.py`)

Provides:
- `get_database_url()` - Reads `AE_DATABASE_URL` from environment
- `get_engine()` - Creates SQLAlchemy engine with connection pooling
- `get_session()` - Context manager for database sessions
- `init_db()` - Creates all tables in the database

### 3. **Repository Layer** (`src/db/repository.py`)

High-level CRUD operations abstracting away SQLAlchemy queries:

- `get_or_create_study()` - Get or create study record
- `get_or_create_participant()` - Get or create participant record
- `save_audio_processing()` - Save or update audio processing metadata
- `save_biomechanics_import()` - Save or update biomechanics metadata
- `save_synchronization()` - Save or update synchronization results
- `save_movement_cycle()` - Save or update movement cycle data
- Query methods for retrieving records by various criteria

### 4. **Tests** (`tests/test_database.py`)

Comprehensive test suite covering:
- Model creation and validation
- Constraint enforcement (unique, check, foreign key)
- Session management and transactions
- Repository CRUD operations
- Relationship traversal

Uses SQLite in-memory for fast testing (compatible with PostgreSQL in production).

### 5. **Documentation**

- [src/db/README.md](src/db/README.md) - Database module overview and setup
- [docs/POSTGRES_SETUP.md](docs/POSTGRES_SETUP.md) - Native PostgreSQL installation guide
- [docs/POSTGRES_TRANSITION.md](docs/POSTGRES_TRANSITION.md) - Migration strategy from Excel/JSON

## Design Decisions

### Binary Files Stay on Disk

Large `.pkl` files (audio, sync, cycles) remain on disk. The database stores:
- File path (absolute or relative to `AE_DATA_ROOT`)
- Checksum (SHA-256, for future integrity verification)
- File size and modification timestamp

This keeps the database lightweight and performant while maintaining referential integrity.

### Denormalization for Query Performance

Some fields are intentionally duplicated across tables (e.g., audio file info in both `audio_processing` and `synchronization`) to:
- Avoid complex joins for common queries
- Match the structure of existing Pydantic models
- Support independent processing stages

This follows the "Synchronization inherits from AcousticsFile" pattern in the Pydantic models.

### PostgreSQL-Specific Features

- **ARRAY columns** for QC segments and artifact lists (automatically converted to JSON for SQLite testing)
- **Check constraints** for enum-like fields
- **Unique constraints** on compound keys (study + participant, audio file + knee + maneuver)

For testing, SQLite is used with custom `TypeDecorator` classes (`StringList`, `FloatList`) that:
- Use native PostgreSQL arrays in production
- Serialize to JSON in SQLite for testing
- Provide transparent cross-database compatibility

## Configuration

### Environment Variables

Set in `.env.local` (gitignored):

```bash
# Database connection
AE_DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/acoustic_emissions

# Optional: default data root for participants
AE_DATA_ROOT=/absolute/path/to/studies
```

### Python Dependencies

Already in `requirements.txt`:
- `sqlalchemy==2.0.36` - ORM
- `psycopg==3.2.3` - PostgreSQL adapter (modern psycopg3)
- `alembic==1.15.1` - Database migrations (not yet configured)

## Next Steps

### Phase 2: Dual Write (Excel + Database)

1. **Add database writes to processing pipeline:**
   - Modify `src/orchestration/processing_log.py` to call repository methods
   - Keep existing Excel writes for backward compatibility
   - Add feature flag to enable/disable DB writes

2. **Path management:**
   - Store file paths relative to `AE_DATA_ROOT` for portability
   - Compute and store checksums for file integrity

3. **Migration script:**
   - Create backfill script to populate database from existing Excel logs
   - Verify data integrity after migration

### Phase 3: Read Path Migration

1. **Prefer database reads:**
   - Update `ManeuverProcessingLog.load_from_excel()` to first try database
   - Fall back to Excel if database record doesn't exist
   - Log which source is used

2. **Validation:**
   - Compare database vs Excel results
   - Ensure both sources produce identical outputs

### Phase 4: Database Migrations

1. **Configure Alembic:**
   - Initialize Alembic migration environment
   - Create initial migration from existing models
   - Document migration workflow for schema changes

## Testing the Implementation

### Run Database Tests

```bash
# Requires AE_DATABASE_URL to be set in environment or .env.local
pytest tests/test_database.py -v
```

### Initialize Database Manually

```python
from src.db import init_db, get_engine

# Create all tables
engine = get_engine()
init_db(engine)
```

### Example Usage

```python
from src.db import get_session
from src.db.repository import Repository
from src.metadata import AudioProcessing
from datetime import datetime

# Create audio processing record
audio = AudioProcessing(
    study="AOA",
    study_id=1011,
    audio_file_name="test.bin",
    device_serial="TEST123",
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
    linked_biomechanics=False,
)

# Save to database
with get_session() as session:
    repo = Repository(session)
    record = repo.save_audio_processing(audio, pkl_file_path="/path/to/audio.pkl")
    print(f"Saved audio processing record with ID: {record.id}")
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Processing Pipeline                       │
│  (cli/process_directory.py, src/orchestration/...)          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ Creates Pydantic models
                     │ (src/metadata.py)
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                  Repository Layer                            │
│            (src/db/repository.py)                            │
│  • save_audio_processing()                                   │
│  • save_synchronization()                                    │
│  • save_movement_cycle()                                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ Converts to ORM models
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                SQLAlchemy ORM Models                         │
│               (src/db/models.py)                             │
│  • StudyRecord                                               │
│  • ParticipantRecord                                         │
│  • AudioProcessingRecord                                     │
│  • SynchronizationRecord                                     │
│  • MovementCycleRecord                                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ SQL queries via SQLAlchemy
                     ↓
┌─────────────────────────────────────────────────────────────┐
│              PostgreSQL Database                             │
│  • studies                                                   │
│  • participants                                              │
│  • audio_processing                                          │
│  • synchronizations                                          │
│  • movement_cycles                                           │
│  • movement_cycle_details                                    │
└─────────────────────────────────────────────────────────────┘

                     Disk (separate)
         ┌──────────────────────────────┐
         │  Binary files (.pkl):         │
         │  • Audio data                 │
         │  • Sync results               │
         │  • Cycle data                 │
         │  (paths stored in DB)         │
         └──────────────────────────────┘
```

## Comparison: Excel vs Database

| Aspect | Excel/JSON | PostgreSQL |
|--------|-----------|------------|
| **Query Performance** | Must load entire file | Indexed queries, efficient |
| **Concurrent Access** | File locking issues | Multi-user, ACID transactions |
| **Data Integrity** | Manual validation | Enforced constraints |
| **Scalability** | Limited by file size | Scales to millions of records |
| **Versioning** | Git-based | Schema migrations (Alembic) |
| **Backup** | File copy | pg_dump, continuous archiving |
| **Integration** | Manual parsing | Standard SQL interface |
| **Audit Trail** | Git history | Timestamps, optional logging |

## Conclusion

The database schema and ORM are fully implemented and ready for integration. The next step is Phase 2: adding dual writes to the processing pipeline while maintaining Excel compatibility. This allows gradual migration with fallback options.
