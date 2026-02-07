#!/usr/bin/env python3
"""Quick test to verify database models work with PostgreSQL."""

import os
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db import Base, ParticipantRecord, StudyRecord


def _get_test_db_url() -> str:
    """Resolve PostgreSQL test database URL from environment."""
    test_url = os.getenv("AE_TEST_DATABASE_URL")
    if test_url:
        return test_url

    prod_url = os.getenv("AE_DATABASE_URL")
    if prod_url and "acoustic_emissions" in prod_url:
        return prod_url.replace("acoustic_emissions", "acoustic_emissions_test")

    return "postgresql+psycopg://postgres@localhost/acoustic_emissions_test"


def test_models_with_postgres():
    """Verify models work with PostgreSQL (ARRAY types supported)."""
    db_url = _get_test_db_url()
    if not db_url.startswith("postgresql"):
        pytest.skip(f"PostgreSQL required, got: {db_url}")

    engine = create_engine(db_url, echo=False)
    try:
        with engine.connect() as conn:
            pass
    except Exception as e:
        pytest.skip(
            f"PostgreSQL not available: {e}\n"
            "Make sure PostgreSQL is running and AE_DATABASE_URL is set."
        )

    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        study = StudyRecord(name="AOA")
        session.add(study)
        session.commit()
        assert study.id is not None

        participant = ParticipantRecord(study_participant_id=study.id, study_id=1011)
        session.add(participant)
        session.commit()
        assert participant.id is not None

    Base.metadata.drop_all(engine)
    engine.dispose()


if __name__ == "__main__":
    try:
        test_models_with_postgres()
        print("✅ PostgreSQL model test passed")
    except pytest.SkipTest as exc:
        print(f"⚠️  Skipped: {exc}")
