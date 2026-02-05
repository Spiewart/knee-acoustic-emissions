# Database Usage and Migration Guide

## Overview

This project uses **PostgreSQL** for persistent data storage and **Alembic** for schema migrations. All database operations are managed through SQLAlchemy ORM.

## Prerequisites

1. **PostgreSQL** server running (version 12+)
2. **Python environment** with dependencies installed
3. **Environment variable** `AE_DATABASE_URL` configured

## Environment Setup

### 1. Database Connection

Set the `AE_DATABASE_URL` environment variable in your `.env.local` file (or export it):

```bash
# PostgreSQL format:
export AE_DATABASE_URL="postgresql+psycopg://username:password@localhost:5432/acoustic_emissions"

# Alternative drivers:
# postgresql+psycopg2://username:password@localhost:5432/acoustic_emissions
# postgresql://username:password@localhost:5432/acoustic_emissions
```

**Important**: This environment variable is used by both the application code and Alembic migrations.

### 2. Create Database

First, create the PostgreSQL database:

```bash
# Connect to PostgreSQL as superuser
psql -U postgres

# Create database
CREATE DATABASE acoustic_emissions;

# Create user (if needed)
CREATE USER ae_user WITH PASSWORD 'your_password';

# Grant privileges
GRANT ALL PRIVILEGES ON DATABASE acoustic_emissions TO ae_user;

# Exit psql
\q
```

## Schema Management with Alembic

### Initial Setup (First Time Only)

The Alembic configuration is already initialized. To apply the schema:

```bash
# Activate virtual environment
workon kae_processing  # or: source ~/.virtualenvs/kae_processing/bin/activate

# Apply all migrations to create/update schema
alembic upgrade head
```

### Common Alembic Commands

```bash
# Check current migration status
alembic current

# Show migration history
alembic history --verbose

# Upgrade to latest version
alembic upgrade head

# Upgrade one version at a time
alembic upgrade +1

# Downgrade one version
alembic downgrade -1

# Downgrade to specific revision
alembic downgrade <revision_id>

# Downgrade everything
alembic downgrade base
```

### Creating New Migrations

When you modify the database models in `src/db/models.py`:

```bash
# Auto-generate migration from model changes
alembic revision --autogenerate -m "description_of_changes"

# Manually create empty migration
alembic revision -m "description_of_changes"

# Review the generated migration file in alembic/versions/
# Edit if necessary, then apply:
alembic upgrade head
```

**Best Practices**:
- Always review auto-generated migrations before applying
- Test migrations on development database first
- Include both `upgrade()` and `downgrade()` functions
- Add descriptive docstrings explaining the changes

## Application Database Usage

### Basic Usage

```python
from src.db.session import get_session
from src.db.models import StudyRecord

# Using context manager (recommended)
with get_session() as session:
    study = session.query(StudyRecord).filter_by(name="AOA").first()
    print(study.name)
    # Commits automatically on success, rolls back on exception
```

### Repository Pattern (Recommended)

```python
from src.db.repository import Repository
from src.metadata import AudioProcessing

# Create repository with session
with get_session() as session:
    repo = Repository(session)

    # Save audio processing record
    audio_record = AudioProcessing(
        study="AOA",
        study_id=1001,
        audio_file_name="test.bin",
        # ... other fields
    )
    saved_record = repo.save_audio_processing(audio_record, audio_file_path="/path/to/file.bin")
    print(f"Saved with ID: {saved_record.id}")
```

### Direct SQLAlchemy (Advanced)

```python
from src.db.session import get_engine
from src.db.models import Base, ParticipantRecord
from sqlalchemy.orm import sessionmaker

engine = get_engine()
SessionLocal = sessionmaker(bind=engine)

session = SessionLocal()
try:
    participant = session.query(ParticipantRecord).filter_by(study_id=1001).first()
    session.commit()
except Exception:
    session.rollback()
    raise
finally:
    session.close()
```

## Testing

### Unit Tests

Unit tests use **in-memory SQLite** or **test fixtures** and don't require a real database:

```bash
pytest tests/unit/ -v
```

### Integration Tests

Integration tests require a **real PostgreSQL database** with migrations applied:

```bash
# 1. Set up test database
export AE_DATABASE_URL="postgresql+psycopg://user:pass@localhost:5432/ae_test"

# 2. Apply migrations
alembic upgrade head

# 3. Run integration tests
pytest tests/integration/ -v
```

**Important**: Integration tests will fail if:
- `AE_DATABASE_URL` is not set
- Database doesn't exist
- Schema is not up to date (migrations not applied)

## Migration History

### Current Migrations

1. **b68cac4282f5** - `refactor_synchronization_fields`
   - Refactored synchronization table fields
   - Removed redundant audio sync time fields
   - Added support for multiple stomp detection methods
   - Added optional audio sync time fields per leg
   - See `SYNCHRONIZATION_SCHEMA_CHANGES.md` for details

## Troubleshooting

### "AE_DATABASE_URL environment variable not set"

**Solution**: Set the environment variable in your shell or `.env.local` file:

```bash
export AE_DATABASE_URL="postgresql+psycopg://user:pass@localhost:5432/acoustic_emissions"
```

### "alembic.util.exc.CommandError: Can't locate revision identified by 'xyz'"

**Solution**: Your database schema is out of sync. Either:
- Apply all migrations: `alembic upgrade head`
- Or reset to base: `alembic downgrade base` then `alembic upgrade head`

### "relation 'synchronizations' does not exist"

**Solution**: Run migrations to create the schema:

```bash
alembic upgrade head
```

### Migration Conflicts

If you have multiple branches with different migrations:

```bash
# Check current revision
alembic current

# View all branches
alembic branches

# Merge branches (if needed)
alembic merge <rev1> <rev2> -m "merge_branches"
```

## Schema Initialization (Legacy)

**⚠️ Deprecated**: The `init_db()` function is deprecated. Use Alembic migrations instead.

```python
# OLD WAY (deprecated):
from src.db.session import init_db
init_db()  # Don't use this anymore

# NEW WAY (recommended):
# Use: alembic upgrade head
```

## Production Deployment

1. **Backup database** before applying migrations:
   ```bash
   pg_dump -U user acoustic_emissions > backup_$(date +%Y%m%d).sql
   ```

2. **Test migration** on staging environment first

3. **Apply migration** during maintenance window:
   ```bash
   alembic upgrade head
   ```

4. **Verify** application works with new schema

5. **Monitor** for issues

6. **Rollback if needed**:
   ```bash
   alembic downgrade -1  # Go back one version
   ```

## Best Practices

1. **Always use Alembic** for schema changes (never use `init_db()` or manual SQL)
2. **Test migrations** on development database before production
3. **Version control** all migration files in `alembic/versions/`
4. **Document** significant schema changes in docstrings and separate .md files
5. **Use transactions** - Alembic automatically wraps migrations in transactions
6. **Backup** production database before applying migrations
7. **Review** auto-generated migrations - they're not always perfect

## Additional Resources

- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [SQLAlchemy ORM Tutorial](https://docs.sqlalchemy.org/en/20/orm/quickstart.html)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- Project-specific: `SYNCHRONIZATION_SCHEMA_CHANGES.md`

## Contact

For database-related questions or issues, contact the development team or file an issue in the repository.
