# Transition Plan: Excel/JSON → PostgreSQL

This document outlines a phased migration plan to store metadata in PostgreSQL while keeping large `.pkl` files on disk.

## Goals

- Replace Excel/JSON as the primary metadata store.
- Preserve backward compatibility during migration.
- Keep large binary data (`.pkl`) outside the DB; store paths + checksums.

## Scope

### Metadata to store in Postgres
- Audio processing metadata
- Biomechanics import metadata
- Synchronization metadata
- Movement cycles (summary + per-cycle records)

### Data to keep on disk
- `.pkl` audio files
- `.pkl` movement cycles
- QC plots and artifacts

Store the absolute path or URI in the DB, plus:
- file size
- checksum (e.g., SHA-256)
- last modified timestamp

## Proposed Schema (High Level)

- `studies`
- `participants`
- `audio_processing`
- `biomechanics_import`
- `synchronization`
- `movement_cycles`
- `movement_cycle_details`

Each table mirrors the corresponding Pydantic dataclass in `src/metadata.py`.

## Implementation Phases

### Phase 1 — Schema & ORM
- Add SQLAlchemy models mirroring `src/metadata.py`.
- Add migration tooling (Alembic).
- Add connection config via `AE_DATABASE_URL`.

### Phase 2 — Dual Write
- Continue writing Excel/JSON.
- Add DB writes alongside Excel writes.
- Provide a feature flag to toggle DB writes.

### Phase 3 — Read Path Migration
- Prefer DB reads; fall back to Excel when DB missing.
- Add tooling to backfill DB from existing Excel logs.

### Phase 4 — Deprecate Excel
- Optional: keep Excel export as an explicit command.
- Remove Excel as default data source.

## Local Configuration

Use `.env.local` for machine-specific config:
```
AE_DATA_ROOT=/absolute/path/to/studies
AE_DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/acoustic_emissions
```

`.env.local` is created from `.env.example` by the setup scripts.

## Where the Data Root Lives in Code

- Config helper: `src/config.py`
- Used by:
  - `cli/process_directory.py`
  - `src/orchestration/participant.py`

If the CLI `PATH` argument is omitted, the code uses `AE_DATA_ROOT` from the environment.

## Next Step Proposal

- Add SQLAlchemy models in `src/db/models.py`
- Add a DB repository layer in `src/db/repository.py`
- Add a migration script to backfill from Excel logs
