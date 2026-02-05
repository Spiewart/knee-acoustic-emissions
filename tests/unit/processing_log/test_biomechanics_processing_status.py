"""Test that biomechanics records are saved with correct processing_status."""

from pathlib import Path

import pytest

from src.db.repository import Repository
from src.metadata import BiomechanicsImport


def test_biomechanics_processing_status_persists(db_session):
    """Test that processing_status='success' is saved to database."""
    repository = Repository(db_session)

    # Create a biomechanics import record with processing_status='success'
    biomech = BiomechanicsImport(
        study="AOA",
        study_id=1016,
        biomechanics_file="/path/to/biomech.xlsx",
        sheet_name="Sheet1",
        biomechanics_type="Motion Analysis",
        knee="left",
        maneuver="fe",
        biomechanics_sync_method="stomp",
        biomechanics_sample_rate=100.0,
        num_sub_recordings=3,
        duration_seconds=30.0,
        num_data_points=3000,
        num_passes=3,
        processing_status="success",  # This should be saved
    )

    # Save to database
    record = repository.save_biomechanics_import(biomech)
    db_session.commit()

    # Verify it was saved with the correct status
    assert record.processing_status == "success"

    # Query it back to ensure it persists
    retrieved = repository.get_biomechanics_imports(
        study_name="AOA",
        participant_number=1016,
        knee="left",
        maneuver="fe",
    )
    assert len(retrieved) == 1
    assert retrieved[0].processing_status == "success"


def test_biomechanics_processing_status_update(db_session):
    """Test that processing_status is updated correctly when record exists."""
    repository = Repository(db_session)

    # Create initial record with 'not_processed' status
    biomech = BiomechanicsImport(
        study="AOA",
        study_id=1017,
        biomechanics_file="/path/to/biomech2.xlsx",
        sheet_name="Sheet1",
        biomechanics_type="Motion Analysis",
        knee="right",
        maneuver="walk",
        biomechanics_sync_method="stomp",
        biomechanics_sample_rate=100.0,
        num_sub_recordings=9,
        duration_seconds=90.0,
        num_data_points=9000,
        num_passes=9,
        processing_status="not_processed",
    )

    # Save initial record
    record1 = repository.save_biomechanics_import(biomech)
    db_session.commit()
    assert record1.processing_status == "not_processed"

    # Update with success status
    biomech.processing_status = "success"
    record2 = repository.save_biomechanics_import(biomech)
    db_session.commit()

    # Verify it was updated
    assert record2.processing_status == "success"
    assert record1.id == record2.id  # Same record

    # Query to ensure update persists
    retrieved = repository.get_biomechanics_imports(
        study_name="AOA",
        participant_number=1017,
        knee="right",
        maneuver="walk",
    )
    assert len(retrieved) == 1
    assert retrieved[0].processing_status == "success"
