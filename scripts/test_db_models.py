#!/usr/bin/env python3
"""Quick test to verify database models work with SQLite."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine

from src.db import Base, ParticipantRecord, StudyRecord

# Create in-memory SQLite database
engine = create_engine("sqlite:///:memory:", echo=False)

try:
    # Create all tables (this would fail with ARRAY issue)
    Base.metadata.create_all(engine)
    print("✅ Successfully created all tables in SQLite")
    print(f"   Tables: {', '.join(Base.metadata.tables.keys())}")

    # Test creating a record
    from sqlalchemy.orm import Session
    with Session(engine) as session:
        study = StudyRecord(name="AOA")
        session.add(study)
        session.commit()
        print(f"✅ Successfully created study record (ID: {study.id})")

        participant = ParticipantRecord(study_id=study.id, participant_number=1011)
        session.add(participant)
        session.commit()
        print(f"✅ Successfully created participant record (ID: {participant.id})")

    print("\n✅ All database model tests passed!")

except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
