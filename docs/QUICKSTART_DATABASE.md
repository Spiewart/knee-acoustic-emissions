# Quick Start: PostgreSQL Database

## Prerequisites

1. **PostgreSQL installed and running**
   - macOS: Postgres.app or `brew install postgresql@15`
   - Windows: Download from postgresql.org
   - See [POSTGRES_SETUP.md](POSTGRES_SETUP.md) for detailed instructions

2. **Python 3.12+ virtual environment activated**
   ```bash
   source .venv/bin/activate  # macOS/Linux
   .venv\Scripts\activate     # Windows
   ```

3. **Dependencies installed**
   ```bash
   pip install -r requirements.txt
   ```

## Setup Steps

### 1. Create Databases

Create **both** a production and test database:

```bash
# Production database - for actual data processing
createdb acoustic_emissions

# Test database - for running tests (gets tables created/dropped)
createdb acoustic_emissions_test
```

### 2. Configure Environment

Copy `.env.example` to `.env.local` and update:

```bash
cp .env.example .env.local
```

Edit `.env.local`:
```bash
AE_DATA_ROOT=/path/to/your/participant/data

# Main database for processing data
AE_DATABASE_URL=postgresql+psycopg://username:password@localhost:5432/acoustic_emissions

# Test database (optional - will auto-append _test if not specified)
AE_TEST_DATABASE_URL=postgresql+psycopg://username:password@localhost:5432/acoustic_emissions_test
```

**Common connection strings:**
- Local PostgreSQL: `postgresql+psycopg://postgres:@localhost:5432/acoustic_emissions`
- Postgres.app (macOS): `postgresql+psycopg://username@localhost:5432/acoustic_emissions`
- Remote server: `postgresql+psycopg://user:pass@hostname:5432/acoustic_emissions`

**Important:** Keep test and production databases separate!
- **Production DB** (`acoustic_emissions`) - For actual data processing
- **Test DB** (`acoustic_emissions_test`) - For running tests (gets tables dropped/recreated)

### 3. Initialize Databases

Initialize both databases:

```bash
# Initialize production database
python scripts/init_database.py --init

# Initialize test database
python scripts/init_database.py --init --test

# Verify both connections
python scripts/init_database.py --test-connection
python scripts/init_database.py --test-connection --test
```

Expected output:
```
Initializing production database...
  Database: localhost/acoustic_emissions
✅ Database initialized successfully!
   Tables created: studies, participants, audio_processing, ...
```

### 4. (Optional) Create Sample Data

```bash
python scripts/init_database.py --sample
```

This creates a test record you can use to verify everything works.

## Verify Installation

### Check Tables Were Created

```bash
psql acoustic_emissions -c "\dt"
```

You should see:
```
                    List of relations
 Schema |           Name            | Type  |  Owner
--------+---------------------------+-------+----------
 public | audio_processing          | table | username
 public | biomechanics_imports      | table | username
 public | movement_cycle_details    | table | username
 public | movement_cycles           | table | username
 public | participants              | table | username
 public | studies                   | table | username
 public | synchronizations          | table | username
```

### Run Database Tests

```bash
pytest tests/test_database.py -v
```

**Note:** Tests automatically use a separate test database:
1. Checks `AE_TEST_DATABASE_URL` environment variable first
2. Falls back to `AE_DATABASE_URL` with `_test` appended
3. Defaults to `acoustic_emissions_test` if neither is set

**Safety:** Tests will create and drop tables in the test database. Never point `AE_TEST_DATABASE_URL` at your production database!

```bash
# Create test database (one time)
createdb acoustic_emissions_test

# Tests will use it automatically
pytest tests/test_database.py -v
```

The models use PostgreSQL-specific features (ARRAY types) so SQLite is not supported.

## Usage Examples

### Manual Database Operations

```python
from src.db import get_session
from src.db.repository import Repository
from src.metadata import AudioProcessing
from datetime import datetime

# Create metadata
audio = AudioProcessing(
    study="AOA",
    study_id=1011,
    audio_file_name="example.bin",
    device_serial="DEVICE123",
    firmware_version=1,
    file_time=datetime(2024, 1, 1, 10, 0, 0),
    file_size_mb=150.0,
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
    print(f"Saved record ID: {record.id}")
```

### Query Data

```python
from src.db import get_session
from src.db.models import AudioProcessingRecord, ParticipantRecord, StudyRecord
from sqlalchemy import select

with get_session() as session:
    # Get all audio files for participant AOA 1011
    stmt = select(AudioProcessingRecord).join(ParticipantRecord).join(StudyRecord).where(
        StudyRecord.name == "AOA",
        ParticipantRecord.participant_number == 1011
    )
    results = session.execute(stmt).scalars().all()

    for audio in results:
        print(f"{audio.audio_file_name}: {audio.knee} {audio.maneuver}")
```

## Troubleshooting

### "AE_DATABASE_URL environment variable not set"

Make sure:
1. You created `.env.local` from `.env.example`
2. You're running commands from the project root
3. Your shell loads `.env.local` (or use `python-dotenv`)

### "Connection refused"

PostgreSQL is not running. Start it:
- macOS (Homebrew): `brew services start postgresql@15`
- macOS (Postgres.app): Launch the app
- Windows: Check Services panel for PostgreSQL service

### "database does not exist"

Create the database:
```bash
createdb acoustic_emissions
```

### Permission denied / authentication failed

Update your connection string with the correct username/password:
```bash
# Find your PostgreSQL user
psql -l

# Update .env.local with correct credentials
AE_DATABASE_URL=postgresql+psycopg://YOUR_USERNAME:YOUR_PASSWORD@localhost:5432/acoustic_emissions
```

### Import errors

Make sure you're in the virtual environment and dependencies are installed:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Next Steps

Once the database is set up and tested:

1. **Review the schema**: See [DB_SCHEMA_IMPLEMENTATION.md](DB_SCHEMA_IMPLEMENTATION.md)
2. **Understand the migration plan**: See [POSTGRES_TRANSITION.md](POSTGRES_TRANSITION.md)
3. **Start dual-write implementation**: Modify processing pipeline to write to both Excel and DB

## Useful Commands

```bash
# Test database connection
python scripts/init_database.py --test

# Reset database (careful!)
python scripts/init_database.py --drop --init

# Connect to database with psql
psql acoustic_emissions

# Backup database
pg_dump acoustic_emissions > backup.sql

# Restore database
psql acoustic_emissions < backup.sql

# View table schemas
psql acoustic_emissions -c "\d audio_processing"
```
