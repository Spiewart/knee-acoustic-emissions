#!/usr/bin/env python
"""Delete stale participant ID=1 and all related records from the database."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import create_engine, delete

from src.db.models import (
    AudioProcessingRecord,
    BiomechanicsImportRecord,
    MovementCycleRecord,
    ParticipantRecord,
    SynchronizationRecord,
)

DB_URL = "postgresql://spiewart@localhost/acoustic_emissions"

engine = create_engine(DB_URL)

# Delete in order of dependencies (child first, then parent)
with engine.begin() as conn:
    # Delete movement cycles for this participant
    stmt = delete(MovementCycleRecord).where(MovementCycleRecord.participant_id == 1)
    result = conn.execute(stmt)
    print(f"Deleted {result.rowcount} movement cycle record(s)")

    # Delete synchronization records for this participant
    stmt = delete(SynchronizationRecord).where(SynchronizationRecord.participant_id == 1)
    result = conn.execute(stmt)
    print(f"Deleted {result.rowcount} synchronization record(s)")

    # Delete biomechanics import records for this participant
    stmt = delete(BiomechanicsImportRecord).where(BiomechanicsImportRecord.participant_id == 1)
    result = conn.execute(stmt)
    print(f"Deleted {result.rowcount} biomechanics import record(s)")

    # Delete audio processing records for this participant
    stmt = delete(AudioProcessingRecord).where(AudioProcessingRecord.participant_id == 1)
    result = conn.execute(stmt)
    print(f"Deleted {result.rowcount} audio processing record(s)")

    # Finally, delete the participant
    stmt = delete(ParticipantRecord).where(ParticipantRecord.id == 1)
    result = conn.execute(stmt)
    print(f"Deleted {result.rowcount} participant record(s) with ID=1")

print("\nDatabase cleanup complete!")
