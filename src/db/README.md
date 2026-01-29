# Database Module

PostgreSQL-backed metadata storage for acoustic emissions processing.

## Overview

This module provides:
- **SQLAlchemy ORM models** (`src/db/models.py`) mirroring Pydantic dataclasses from `src/metadata.py`
- **Repository layer** (`src/db/repository.py`) for CRUD operations
- **Session management** (`src/db/session.py`) for database connections
- **CLI tools** for database initialization and management

Large binary files (`.pkl` audio and cycle data) remain on disk. The database stores only:
- Metadata
- File paths
- Checksums (future)
- Processing status and QC results

**Important:** Models use PostgreSQL-specific features (ARRAY types). SQLite is NOT supported.

## Setup

### 1. Install PostgreSQL

See [POSTGRES_SETUP.md](../../docs/POSTGRES_SETUP.md) for detailed installation instructions.

**macOS (Postgres.app)**:
```bash
# Download from https://postgresapp.com/
# Or via Homebrew:
brew install postgresql@15
brew services start postgresql@15
```

**Windows**:
```bash
# Download from https://www.postgresql.org/download/windows/
```

### 2. Create Database

```bash
createdb acoustic_emissions

# For testing, create a separate test database
createdb acoustic_emissions_test
```

### 3. Configure Environment

Copy `.env.example` to `.env.local` and update:

```bash
# .env.local
AE_DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/acoustic_emissions
AE_DATA_ROOT=/path/to/your/studies
```

### 4. Install Python Dependencies

```bash
pip install -e .
# Or for development:
pip install -e ".[dev]"
```

### 5. Initialize Database Schema

```bash
ae-db-init
```

This creates all tables. Safe to run multiple times (won't drop existing data).

## Usage

### Check Database Connection

```bash
ae-db-check
```

### Using the Repository Layer

```python
from datetime import datetime
from src.db import get_session
from src.db.repository import Repository
from src.metadata import AudioProcessing

# Create metadata record
audio = AudioProcessing(
    study="AOA",
    study_id=1011,
    linked_biomechanics=False,
    audio_file_name="test_audio.bin",
    device_serial="SN12345",
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

# Save to database
with get_session() as session:
    repo = Repository(session)
    record = repo.save_audio_processing(audio, pkl_file_path="/path/to/audio.pkl")
    print(f"Saved record with ID: {record.id}")
```

### Querying Records

```python
from src.db import get_session
from src.db.repository import Repository

with get_session() as session:
    repo = Repository(session)

    # Query all audio processing records for a participant
    records = repo.get_audio_processing_records(
        study_name="AOA",
        participant_number=1011,
        maneuver="walk"
    )

    for record in records:
        print(f"{record.audio_file_name}: {record.knee} {record.maneuver}")
```

## Schema

### Tables

- **`studies`**: Study metadata (AOA, preOA, SMoCK)
- **`participants`**: Participant records linked to studies
- **`audio_processing`**: Audio file conversion and QC metadata
- **`biomechanics_imports`**: Biomechanics file import metadata
- **`synchronizations`**: Audio-biomechanics synchronization results
- **`movement_cycles`**: Movement cycle extraction summary
- **`movement_cycle_details`**: Per-cycle QC and timing details

### Relationships

```
studies
  └── participants (1:many)
        ├── audio_processing (1:many)
        ├── biomechanics_imports (1:many)
        ├── synchronizations (1:many)
        └── movement_cycles (1:many)
              └── movement_cycle_details (1:many)
```

### Unique Constraints

Each table enforces uniqueness on key combinations:
- Audio processing: `(participant_id, audio_file_name, knee, maneuver)`
- Synchronization: `(participant_id, audio_file_name, knee, maneuver)`
- Movement cycles: `(participant_id, audio_file_name, knee, maneuver)`

## Migration Strategy

### Phase 1: Schema & ORM ✓ (Current)
- SQLAlchemy models defined
- Repository layer implemented
- CLI tools for initialization

### Phase 2: Dual Write (Next)
- Write to both Excel and database
- Feature flag to enable/disable DB writes
- Maintain backward compatibility

### Phase 3: Read Path Migration
- Prefer database reads
- Fall back to Excel when DB empty
- Backfill script to migrate existing Excel logs

### Phase 4: Deprecate Excel
- Optional Excel export command
- Database as primary source

## Testing

Run database tests:

```bash
pytest tests/test_database.py -v
```

Tests use SQLite in-memory database (no PostgreSQL required).

## Connection String Format

```
postgresql+psycopg://username:password@host:port/database
```

Examples:
```bash
# Local development
postgresql+psycopg://postgres:password@localhost:5432/acoustic_emissions

# With custom user
postgresql+psycopg://ae_user:secret@localhost:5432/acoustic_emissions

# Remote server
postgresql+psycopg://user:pass@db.example.com:5432/ae_prod
```

## Environment Variables

- **`AE_DATABASE_URL`** (required): PostgreSQL connection string
- **`AE_DATA_ROOT`** (optional): Default participant data directory

## Future Enhancements

- [ ] File checksums (SHA-256) for integrity verification
- [ ] Automatic backfill script from Excel logs
- [ ] Dual-write mode in processing pipeline
- [ ] Alembic migrations for schema versioning
- [ ] Connection pooling configuration
- [ ] Read replicas for query scaling
