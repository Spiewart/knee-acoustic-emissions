# PostgreSQL Database Operations

This guide provides comprehensive instructions for managing the PostgreSQL database for acoustic emissions processing, including setup, schema management with **Alembic**, migrations, data inspection, backup/restore, and troubleshooting.

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Initial Setup](#initial-setup)
3. [Schema Management with Alembic](#schema-management-with-alembic) ⭐ NEW
4. [Database Schema Management](#database-schema-management)
5. [Data Inspection and Queries](#data-inspection-and-queries)
6. [Backup and Restore](#backup-and-restore)
7. [Legacy Migrations](#legacy-migrations)
8. [Testing Database](#testing-database)
9. [Troubleshooting](#troubleshooting)
10. [Related Documentation](#related-documentation)

---

## Quick Reference

### Common Commands (Alembic - Recommended)

```bash
# Check current migration status
alembic current

# Apply all migrations
alembic upgrade head

# View migration history
alembic history --verbose

# Rollback one migration
alembic downgrade -1

# Create new migration
alembic revision --autogenerate -m "description"
```

### Legacy Commands (Deprecated)

```bash
# ⚠️ DEPRECATED: Initialize fresh database (drops all data!)
# Use: alembic upgrade head instead
python init_fresh_db.py

# ⚠️ DEPRECATED: Initialize database (creates missing tables only)
# Use: alembic upgrade head instead
python scripts/init_database.py --init

# Test database connection (still valid)
python scripts/init_database.py --test-connection
```

### Database Inspection

```bash
# View database schema and record counts
psql $AE_DATABASE_URL -c "\dt"
psql $AE_DATABASE_URL -c "SELECT COUNT(*) FROM audio_processing;"
```

---

## Initial Setup

### 1. Install PostgreSQL

See [POSTGRES_SETUP.md](POSTGRES_SETUP.md) for detailed installation instructions for macOS and Windows.

**Quick install (macOS with Homebrew):**
```bash
brew install postgresql@15
brew services start postgresql@15
```

### 2. Create Database

```bash
# Create production database
createdb acoustic_emissions

# Create test database (for running test suite)
createdb acoustic_emissions_test
```

### 3. Configure Connection String

Create or edit `.env.local` in the project root:

```bash
# Production database
AE_DATABASE_URL=postgresql+psycopg://USERNAME@localhost:5432/acoustic_emissions

# Test database (optional, defaults to acoustic_emissions_test)
AE_TEST_DATABASE_URL=postgresql+psycopg://USERNAME@localhost:5432/acoustic_emissions_test

# Data root directory (optional)
AE_DATA_ROOT=/path/to/participant/directories
```

Replace `USERNAME` with your PostgreSQL username. For local development with no password:
```bash
AE_DATABASE_URL=postgresql+psycopg://spiewart@localhost/acoustic_emissions
```

**Connection String Format:**
```
postgresql+psycopg://USER:PASSWORD@HOST:PORT/DATABASE
```

See [.env.example](.env.example) for reference.

### 4. Initialize Database Schema (Alembic)

**Recommended: Use Alembic for all schema initialization and updates**

```bash
# Activate virtual environment
workon kae_processing

# Apply all migrations to create schema
alembic upgrade head

# Verify schema was created
psql $AE_DATABASE_URL -c "\dt"
```

**What this does:**
- Creates all tables with current schema
- Tracks migration history in `alembic_version` table
- Enables proper version control for future schema changes

### Working with Git Worktrees

If you're using git worktrees, `.env.local` won't be present in the worktree directory. You need to set it up:

**Option 1: Symlink (Recommended)**
```bash
# From inside your worktree directory
ln -s /path/to/main/repo/.env.local .env.local

# Example:
# ln -s ~/acoustic_emissions_processing/.env.local .env.local
```

**Option 2: Copy**
```bash
# Copy from main repo to worktree
cp /path/to/main/repo/.env.local .env.local
```

**Why?** The symlink keeps configuration synchronized with the main repository. Changes to `.env.local` in the main repo automatically apply to all worktrees.

### 4b. Legacy Initialization (Deprecated)

**⚠️ These methods are deprecated - use Alembic instead**

**Option A: Fresh initialization (for new databases)**

Creates all tables from scratch:
```bash
python scripts/init_database.py --init
```

**Option B: Complete reset (drops all data!)**

Use when starting fresh or for development:
```bash
python init_fresh_db.py
```

This script:
1. Drops the entire `public` schema (destroys all data)
2. Recreates the `public` schema
3. Creates all tables from the current models

**⚠️ WARNING:** `init_fresh_db.py` deletes ALL data. Use only when:
- Setting up a new database
- You have backups of important data
- Working in a development environment

### 5. Verify Installation

```bash
# Test connection and view current data
python scripts/init_database.py --test-connection
```

Expected output:
```
Testing database connection...
✅ Successfully connected to database!
   Database URL: postgresql+psycopg://...
   Current records:
     • Studies: 0
     • Participants: 0
     • Audio files: 0
```

---

## Schema Management with Alembic

**Alembic is now the recommended way to manage database schema changes.** It provides version control for your database schema, similar to Git for code.

### Why Alembic?

- **Version Control**: Track all schema changes in migration files
- **Reversible**: Downgrade to previous schema versions if needed
- **Auditable**: See exactly what changed and when
- **Collaborative**: Multiple developers can work on schema changes
- **Production-Safe**: Test migrations before applying to production

### Alembic Workflow

#### 1. Check Current Status

```bash
# View current migration version
alembic current

# View migration history
alembic history --verbose

# See if you're up to date
alembic history | head -n 5
```

#### 2. Apply Migrations

```bash
# Apply all pending migrations
alembic upgrade head

# Apply one migration at a time
alembic upgrade +1

# Upgrade to specific revision
alembic upgrade <revision_id>
```

#### 3. Rollback Migrations

```bash
# Downgrade one migration
alembic downgrade -1

# Downgrade to specific revision
alembic downgrade <revision_id>

# Downgrade everything (DANGEROUS!)
alembic downgrade base
```

### Creating New Migrations

When you modify database models in `src/db/models.py`:

#### Auto-Generate Migration

```bash
# Generate migration from model changes
alembic revision --autogenerate -m "add_new_column_to_audio"

# Review the generated file in alembic/versions/
# File will be named: <revision>_add_new_column_to_audio.py
```

**⚠️ Important**: Always review auto-generated migrations! Alembic may:
- Miss some changes (complex alterations)
- Generate unnecessary operations
- Need manual adjustments for data migrations

#### Manual Migration

```bash
# Create empty migration template
alembic revision -m "custom_data_migration"

# Edit the generated file to add your changes
```

### Migration Best Practices

1. **Always backup before migrating production**:
   ```bash
   pg_dump acoustic_emissions > backup_$(date +%Y%m%d).sql
   ```

2. **Test migrations on development first**:
   ```bash
   # Test upgrade
   alembic upgrade head
   
   # Test downgrade
   alembic downgrade -1
   alembic upgrade head
   ```

3. **Include both upgrade() and downgrade()**:
   ```python
   def upgrade():
       op.add_column('table', sa.Column('new_col', sa.String()))
   
   def downgrade():
       op.drop_column('table', 'new_col')
   ```

4. **Use descriptive migration messages**:
   ```bash
   # Good
   alembic revision -m "add_audio_sync_time_columns"
   
   # Bad
   alembic revision -m "changes"
   ```

5. **One logical change per migration**

### Current Migrations

**b68cac4282f5** - `refactor_synchronization_fields` (Current)
- Refactored synchronization table fields
- Removed redundant audio sync time fields  
- Added support for multiple stomp detection methods
- Added optional audio sync time fields per leg
- See [SYNCHRONIZATION_SCHEMA_CHANGES.md](../SYNCHRONIZATION_SCHEMA_CHANGES.md) for details

### Alembic Configuration

The Alembic configuration is in:
- `alembic.ini` - Main configuration file
- `alembic/env.py` - Environment setup (uses `AE_DATABASE_URL`)
- `alembic/versions/` - Migration scripts

**Environment Variable**: Alembic automatically uses `AE_DATABASE_URL` from your environment.

### Troubleshooting Alembic

#### "Can't locate revision identified by 'xyz'"

Your database is out of sync. Options:
```bash
# Option 1: Start fresh (DESTROYS DATA!)
dropdb acoustic_emissions && createdb acoustic_emissions
alembic upgrade head

# Option 2: Stamp current version manually
alembic stamp head  # Use with caution!
```

#### Migration Conflicts

Multiple developers creating migrations:
```bash
# List all branches
alembic branches

# Merge conflicting branches
alembic merge <rev1> <rev2> -m "merge_branches"
```

#### Auto-generate Doesn't Detect Changes

Some changes require manual migrations:
- Renaming columns (detected as drop + add)
- Data migrations
- Complex constraint changes

---

## Database Schema Management

### Current Schema

The database consists of six main tables:

1. **study** - Research studies (AOA, preOA, SMoCK)
2. **participant** - Study participants
3. **audio_processing** - Audio file processing and QC metadata
4. **biomechanics_import** - Biomechanics data import tracking
5. **synchronization** - Audio-biomechanics synchronization records
6. **movement_cycles** - Individual movement cycle data

### View Schema

**List all tables:**
```bash
psql $AE_DATABASE_URL -c "\dt"
```

**View specific table schema:**
```bash
psql $AE_DATABASE_URL -c "\d audio_processing"
```

**Export full schema to file:**
```bash
pg_dump $AE_DATABASE_URL --schema-only > schema_backup.sql
```

### Python Scripts for Schema Management

#### `init_database.py` - Multi-purpose Database Tool

Located in `scripts/init_database.py`, this is the primary tool for database operations.

**Usage:**
```bash
# Initialize production database
python scripts/init_database.py --init

# Initialize test database
python scripts/init_database.py --init --test

# Test connection
python scripts/init_database.py --test-connection

# Create sample data
python scripts/init_database.py --sample

# Drop all tables (requires confirmation)
python scripts/init_database.py --drop
```

**Key Features:**
- Creates tables that don't exist
- Preserves existing data
- Can target production or test database with `--test` flag
- Includes connection testing and sample data creation

#### `init_fresh_db.py` - Complete Database Reset

Located in project root, this script performs a destructive reset.

**Usage:**
```bash
# Reset production database (hardcoded to 'acoustic_emissions')
python init_fresh_db.py

# Reset specific database via environment variable
AE_DATABASE_URL="postgresql+psycopg://user@localhost/my_db" python init_fresh_db.py
```

**⚠️ This script:**
- Drops the entire `public` schema (CASCADE delete)
- Recreates empty schema
- Creates all tables from scratch
- **Destroys ALL data** - no confirmation prompt

**When to use:**
- Initial setup of new database
- Development environment reset
- After making incompatible schema changes in models
- Before restoring from backup

**When NOT to use:**
- Production database with real data (use migrations instead)
- When you only need to add new tables (use `init_database.py --init`)
- Without a recent backup

#### `sync_production_db.py` - Safe Schema Updates

Located in `scripts/sync_production_db.py`, this script updates the schema while preserving data.

**Usage:**
```bash
python scripts/sync_production_db.py
```

**Features:**
- Prompts for confirmation before making changes
- Creates new tables that don't exist
- Preserves existing data in all tables
- Shows database connection details

**Limitations:**
- Cannot add columns to existing tables (use migrations)
- Cannot modify existing column types (use migrations)
- Cannot rename columns (use migrations)

Use this when:
- You added new tables to the models
- You want a safer alternative to `init_fresh_db.py`
- You're not sure if tables exist

---

## Data Inspection and Queries

### Using Python Scripts

**Check database status:**
```bash
python scripts/init_database.py --test-connection
```

**Create test data:**
```bash
python scripts/init_database.py --sample
```

### Using psql Command Line

**Connect to database:**
```bash
# Connect to production database
psql $AE_DATABASE_URL

# Connect with explicit URL
psql postgresql://user@localhost:5432/acoustic_emissions
```

**Common queries inside psql:**

```sql
-- List all tables
\dt

-- View table schema
\d audio_processing

-- Count records in each table
SELECT 'audio_processing' as table, COUNT(*) FROM audio_processing
UNION ALL
SELECT 'biomechanics_import', COUNT(*) FROM biomechanics_import
UNION ALL
SELECT 'synchronization', COUNT(*) FROM synchronization
UNION ALL
SELECT 'movement_cycles', COUNT(*) FROM movement_cycles
UNION ALL
SELECT 'participant', COUNT(*) FROM participant
UNION ALL
SELECT 'study', COUNT(*) FROM study;

-- View recent audio processing records
SELECT id, audio_file_name, knee, maneuver, processing_status, processing_date
FROM audio_processing
ORDER BY processing_date DESC
LIMIT 10;

-- Check for participants
SELECT p.id, p.study_participant_id, s.name as study_name
FROM participant p
JOIN study s ON p.study_id = s.id;

-- View audio QC summary
SELECT
    COUNT(*) as total_files,
    SUM(CASE WHEN audio_qc_fail THEN 1 ELSE 0 END) as failed_qc,
    SUM(CASE WHEN qc_signal_dropout THEN 1 ELSE 0 END) as signal_dropout,
    SUM(CASE WHEN qc_artifact THEN 1 ELSE 0 END) as artifacts
FROM audio_processing;

-- View synchronization statistics
SELECT
    COUNT(*) as total_syncs,
    biomechanics_sync_method,
    AVG(num_cycles) as avg_cycles
FROM synchronization
GROUP BY biomechanics_sync_method;

-- Exit psql
\q
```

### Using Python Directly

**Quick inspection script:**
```python
from src.db import get_session
from src.db.models import AudioProcessingRecord, ParticipantRecord

with get_session() as session:
    # Count records
    audio_count = session.query(AudioProcessingRecord).count()
    print(f"Audio records: {audio_count}")

    # Get recent audio files
    recent = session.query(AudioProcessingRecord)\
                   .order_by(AudioProcessingRecord.processing_date.desc())\
                   .limit(5).all()
    for record in recent:
        print(f"  {record.audio_file_name} - {record.knee} {record.maneuver}")
```

---

## Backup and Restore

### Full Database Backup

**Backup entire database:**
```bash
# Backup with data (most common)
pg_dump $AE_DATABASE_URL > backup_$(date +%Y%m%d).sql

# Backup schema only (no data)
pg_dump $AE_DATABASE_URL --schema-only > schema_backup.sql

# Backup data only (no schema)
pg_dump $AE_DATABASE_URL --data-only > data_backup.sql

# Backup specific tables
pg_dump $AE_DATABASE_URL -t audio_processing -t biomechanics_import > tables_backup.sql
```

**Compressed backup:**
```bash
pg_dump $AE_DATABASE_URL | gzip > backup_$(date +%Y%m%d).sql.gz
```

### Restore Database

**From SQL dump:**
```bash
# Drop and recreate database (for complete restore)
dropdb acoustic_emissions
createdb acoustic_emissions
psql $AE_DATABASE_URL < backup_20260202.sql

# Or restore to existing database (may have conflicts)
psql $AE_DATABASE_URL < backup_20260202.sql
```

**From compressed backup:**
```bash
gunzip -c backup_20260202.sql.gz | psql $AE_DATABASE_URL
```

**Restore after init_fresh_db.py:**
```bash
# 1. Reset database schema
python init_fresh_db.py

# 2. Restore data (data-only backup)
psql $AE_DATABASE_URL < data_backup.sql
```

### Copy Database

**Copy to a new database:**
```bash
# Create new database from template
createdb -T acoustic_emissions acoustic_emissions_backup

# Or dump and restore
pg_dump acoustic_emissions | psql acoustic_emissions_backup
```

### Automated Backup Script Example

Create `scripts/backup_db.sh`:
```bash
#!/bin/bash
BACKUP_DIR="$HOME/ae_backups"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p "$BACKUP_DIR"

pg_dump $AE_DATABASE_URL | gzip > "$BACKUP_DIR/ae_backup_$DATE.sql.gz"

# Keep only last 7 days of backups
find "$BACKUP_DIR" -name "ae_backup_*.sql.gz" -mtime +7 -delete

echo "Backup completed: ae_backup_$DATE.sql.gz"
```

---

## Legacy Migrations (SQL Files)

**⚠️ DEPRECATED**: This section describes the old SQL-based migration system. **Use Alembic instead** (see [Schema Management with Alembic](#schema-management-with-alembic) above).

### Historical Context

Before Alembic, schema changes were managed with manual SQL migration files in `scripts/migrations/`. These are preserved for reference but should not be used for new changes.

### When to Use SQL Migrations (Rare)

Use migrations when you need to:
- Add columns to existing tables
- Modify column types
- Add/remove constraints or indexes
- Rename columns or tables
- Preserve existing data

**Do NOT use migrations when:**
- Setting up a new database (use `init_fresh_db.py`)
- Database has no data (use `init_fresh_db.py`)
- Making incompatible breaking changes in development

### Creating a Migration

**1. Create migration SQL file:**

Location: `scripts/migrations/NNN_description.sql`

Format: `NNN` = sequential number (e.g., `002_add_timezone.sql`)

Example structure:
```sql
-- Migration: Add recording timezone column
-- Date: 2026-02-02
-- Description: Adds timezone field to audio_processing table

-- Add the column
ALTER TABLE audio_processing
ADD COLUMN IF NOT EXISTS recording_timezone VARCHAR(10);

-- Set default value for existing records
UPDATE audio_processing
SET recording_timezone = 'UTC'
WHERE recording_timezone IS NULL;

-- Verify the change
SELECT column_name, data_type, character_maximum_length
FROM information_schema.columns
WHERE table_name = 'audio_processing'
AND column_name = 'recording_timezone';
```

**Best practices:**
- Use `IF NOT EXISTS` / `IF EXISTS` for idempotent migrations
- Include verification queries at the end
- Add comments explaining the purpose
- Test on a copy of production data first
- Make migrations reversible when possible

**2. Update Python models:**

Ensure `src/db/models.py` matches the migration:
```python
class AudioProcessingRecord(Base):
    # ... existing columns ...
    recording_timezone: Mapped[str] = mapped_column(String(10), nullable=True)
```

And update Pydantic model in `src/metadata.py`:
```python
@dataclass
class AudioProcessing:
    # ... existing fields ...
    recording_timezone: str = Field(default="UTC")
```

**3. Test migration:**

```bash
# Test on test database first
psql $AE_TEST_DATABASE_URL -f scripts/migrations/002_add_timezone.sql

# Verify it worked
psql $AE_TEST_DATABASE_URL -c "\d audio_processing"

# Run test suite
pytest tests/
```

**4. Apply to production:**

```bash
# Backup first!
pg_dump $AE_DATABASE_URL > backup_before_migration.sql

# Apply migration
psql $AE_DATABASE_URL -f scripts/migrations/002_add_timezone.sql

# Verify
python scripts/init_database.py --test-connection
```

### Example Migration: Add Foreign Key

```sql
-- Migration 001: Add biomechanics_import_id foreign key
-- Date: 2026-01-30

ALTER TABLE audio_processing
ADD COLUMN IF NOT EXISTS biomechanics_import_id INTEGER;

ALTER TABLE audio_processing
DROP CONSTRAINT IF EXISTS fk_audio_biomechanics;

ALTER TABLE audio_processing
ADD CONSTRAINT fk_audio_biomechanics
FOREIGN KEY (biomechanics_import_id)
REFERENCES biomechanics_import(id)
ON DELETE SET NULL;
```

### Rollback Migrations

Create a rollback script for each migration:

`scripts/migrations/002_add_timezone_rollback.sql`:
```sql
-- Rollback for migration 002
ALTER TABLE audio_processing
DROP COLUMN IF EXISTS recording_timezone;
```

Apply rollback:
```bash
# Restore from backup (safest)
psql $AE_DATABASE_URL < backup_before_migration.sql

# Or run rollback script
psql $AE_DATABASE_URL -f scripts/migrations/002_add_timezone_rollback.sql
```

---

## Testing Database

### Setup Test Database

The test database is automatically used by pytest when running tests.

**Create and initialize:**
```bash
# Create database
createdb acoustic_emissions_test

# Initialize schema
python scripts/init_database.py --init --test

# Or use init_fresh_db.py with environment override
AE_DATABASE_URL="postgresql+psycopg://user@localhost/acoustic_emissions_test" \
  python init_fresh_db.py
```

**Configure in `.env.local`:**
```bash
AE_TEST_DATABASE_URL=postgresql+psycopg://user@localhost:5432/acoustic_emissions_test
```

If not set, tests will default to `acoustic_emissions_test` database.

### Test Database Operations

**Create sample data:**
```bash
python scripts/init_database.py --sample --test
```

**Reset test database:**
```bash
# Quick reset (preserves schema)
psql acoustic_emissions_test -c "TRUNCATE study, participant, audio_processing,
  biomechanics_import, synchronization, movement_cycles CASCADE;"

# Complete reset
AE_DATABASE_URL="postgresql+psycopg://user@localhost/acoustic_emissions_test" \
  python init_fresh_db.py
```

**Run tests:**
```bash
# Run full test suite
pytest tests/

# Run specific database tests
pytest tests/test_database.py

# Run with database debugging
pytest tests/ --log-cli-level=DEBUG
```

### Test Database Fixture

Tests use the `use_test_db` fixture from `tests/conftest.py`:

```python
@pytest.fixture(scope="session")
def use_test_db():
    """Initialize test database for the entire test session."""
    # Creates engine for acoustic_emissions_test
    # Drops and recreates all tables
    # Yields engine for tests
    # Cleans up on teardown
```

---

## Troubleshooting

### Connection Errors

**Error:** `could not connect to server`
```bash
# Check if PostgreSQL is running
brew services list  # macOS
# or
pg_isready

# Start PostgreSQL if needed
brew services start postgresql@15
```

**Error:** `database "acoustic_emissions" does not exist`
```bash
# Create the database
createdb acoustic_emissions
```

**Error:** `FATAL: role "user" does not exist`
```bash
# Create PostgreSQL user
createuser -s username

# Or connect as existing user
AE_DATABASE_URL=postgresql+psycopg://existinguser@localhost/acoustic_emissions
```

### Schema Errors

**Error:** `column "recording_timezone" does not exist`

**Cause:** Database schema is out of date.

**Solution:**
```bash
# Option 1: Apply migration (if data exists)
psql $AE_DATABASE_URL -f scripts/migrations/add_recording_timezone.sql

# Option 2: Reset database (if no important data)
python init_fresh_db.py

# Option 3: Run init (adds missing tables only)
python scripts/init_database.py --init
```

**Error:** `relation "audio_processing" does not exist`

**Solution:**
```bash
# Initialize database schema
python scripts/init_database.py --init
```

### Test Database Issues

**Error:** Tests fail with `UndefinedColumn` errors

**Solution:**
```bash
# Reset test database schema
AE_DATABASE_URL="postgresql+psycopg://user@localhost/acoustic_emissions_test" \
  python init_fresh_db.py

# Run tests again
pytest tests/
```

### Performance Issues

**Slow queries:**
```sql
-- Check for missing indexes
SELECT schemaname, tablename, indexname
FROM pg_indexes
WHERE schemaname = 'public';

-- Add index on frequently queried columns
CREATE INDEX idx_audio_participant ON audio_processing(participant_id);
CREATE INDEX idx_audio_knee_maneuver ON audio_processing(knee, maneuver);
```

**Large database:**
```bash
# Check database size
psql $AE_DATABASE_URL -c "SELECT pg_size_pretty(pg_database_size('acoustic_emissions'));"

# Check table sizes
psql $AE_DATABASE_URL -c "SELECT tablename,
  pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
  FROM pg_tables WHERE schemaname = 'public' ORDER BY
  pg_total_relation_size(schemaname||'.'||tablename) DESC;"
```

### Data Integrity Issues

**Check for orphaned records:**
```sql
-- Audio records without participant
SELECT COUNT(*) FROM audio_processing
WHERE participant_id NOT IN (SELECT id FROM participant);

-- Synchronization records without audio
SELECT COUNT(*) FROM synchronization
WHERE audio_id NOT IN (SELECT id FROM audio_processing);
```

**Fix foreign key violations:**
```sql
-- Remove orphaned records
DELETE FROM audio_processing
WHERE participant_id NOT IN (SELECT id FROM participant);
```

### Recovery from Corruption

**If database is corrupted:**
```bash
# 1. Stop all connections
# 2. Try to dump what you can
pg_dump $AE_DATABASE_URL > rescue_backup.sql

# 3. Drop and recreate
dropdb acoustic_emissions
createdb acoustic_emissions

# 4. Restore schema
python init_fresh_db.py

# 5. Try to restore data
psql $AE_DATABASE_URL < rescue_backup.sql
```

---

## Related Documentation

### Database and Schema Management
- **[SYNCHRONIZATION_SCHEMA_CHANGES.md](../SYNCHRONIZATION_SCHEMA_CHANGES.md)** - Recent synchronization table refactoring ⭐ NEW
- **[Alembic Documentation](https://alembic.sqlalchemy.org/)** - Official Alembic migration tool
- **[SQLAlchemy ORM](https://docs.sqlalchemy.org/en/20/orm/)** - SQLAlchemy ORM documentation

### PostgreSQL Setup and Configuration
- [POSTGRES_SETUP.md](POSTGRES_SETUP.md) - PostgreSQL installation for macOS/Windows
- [POSTGRES_TRANSITION.md](POSTGRES_TRANSITION.md) - Historical context of database migration

### Testing
- [TESTING_GUIDELINES.md](TESTING_GUIDELINES.md) - How to write tests with database fixtures
- [TESTING_PROCESSING_LOG.md](TESTING_PROCESSING_LOG.md) - Testing database-backed reports

### Data Models
- [src/db/models.py](../src/db/models.py) - SQLAlchemy ORM models
- [src/metadata.py](../src/metadata.py) - Pydantic validation models
- [src/db/repository.py](../src/db/repository.py) - Database access layer

### Processing Pipeline
- [PROCESSING_LOG.md](PROCESSING_LOG.md) - Excel report generation using database
- [README.md](../README.md) - Overall project documentation

---

## Quick Troubleshooting Checklist

When things go wrong, try these steps in order:

1. **Check PostgreSQL is running:**
   ```bash
   pg_isready
   ```

2. **Verify connection string:**
   ```bash
   echo $AE_DATABASE_URL
   ```

3. **Test database connection:**
   ```bash
   python scripts/init_database.py --test-connection
   ```

4. **Check if database exists:**
   ```bash
   psql -l | grep acoustic_emissions
   ```

5. **Verify schema is current:**
   ```bash
   psql $AE_DATABASE_URL -c "\d audio_processing" | grep recording_timezone
   ```

6. **Reset if needed (DEV ONLY):**
   ```bash
   python init_fresh_db.py
   ```

7. **Run tests:**
   ```bash
   pytest tests/test_database.py -v
   ```

---

## Environment Variables Reference

| Variable | Purpose | Example |
|----------|---------|---------|
| `AE_DATABASE_URL` | Production database connection | `postgresql+psycopg://user@localhost/acoustic_emissions` |
| `AE_TEST_DATABASE_URL` | Test database connection (optional) | `postgresql+psycopg://user@localhost/acoustic_emissions_test` |
| `AE_DATA_ROOT` | Default participant data directory | `/path/to/studies` |

Set these in `.env.local` (never commit this file).

---

**Last Updated:** February 2026
**Maintainer:** See project README for contact information
