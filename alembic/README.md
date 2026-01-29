# Alembic Migration Configuration

This directory is reserved for future Alembic migrations when schema changes are needed.

## Setup (Future)

When ready to use Alembic for schema versioning:

```bash
# Initialize Alembic
alembic init alembic

# Configure alembic.ini to use AE_DATABASE_URL

# Create first migration
alembic revision --autogenerate -m "Initial schema"

# Apply migrations
alembic upgrade head
```

## Current Status

**Phase 1**: Using `init_db()` for simple schema creation.
**Future**: Will use Alembic for production schema changes.

See [DB_IMPLEMENTATION.md](../DB_IMPLEMENTATION.md) for full migration strategy.
