"""Integration tests for DB-backed report generation."""

from src.orchestration.processing_log import ManeuverProcessingLog
from src.reports.report_generator import ReportGenerator


def test_maneuver_log_save_uses_database(
    db_session,
    repository,
    audio_processing_factory,
    biomechanics_import_factory,
    synchronization_factory,
    tmp_path,
):
    """Test that ManeuverProcessingLog.save_to_excel generates report from database."""
    audio = audio_processing_factory(study="AOA", study_id=1111, knee="left", maneuver="walk")
    audio_record = repository.save_audio_processing(audio)
    db_session.commit()

    biomech = biomechanics_import_factory(study="AOA", study_id=1111, knee="left", maneuver="walk")
    biomech_record = repository.save_biomechanics_import(biomech, audio_processing_id=audio_record.id)
    db_session.commit()

    sync = synchronization_factory(
        study="AOA",
        study_id=1111,
        audio_processing_id=audio_record.id,
        biomechanics_import_id=biomech_record.id,
        sync_file_name="AOA1111_walk_sync",
    )
    repository.save_synchronization(
        sync,
        audio_processing_id=audio_record.id,
        biomechanics_import_id=biomech_record.id,
    )
    db_session.commit()

    # Create log and save to Excel
    log = ManeuverProcessingLog(
        study_id="AOA1111",
        knee_side="Left",
        maneuver="walk",
        maneuver_directory=tmp_path,
    )
    output_path = log.save_to_excel(session=db_session)

    # Verify output file was created
    assert output_path.exists()
    assert output_path.suffix == ".xlsx"


def test_report_generator_queries_database(
    db_session,
    repository,
    audio_processing_factory,
    biomechanics_import_factory,
    synchronization_factory,
    movement_cycle_factory,
):
    """Test that ReportGenerator queries database for all sheets."""
    # Create audio and biomechanics records
    audio = audio_processing_factory(study="AOA", study_id=2222, knee="right", maneuver="sts")
    audio_record = repository.save_audio_processing(audio)
    db_session.commit()

    biomech = biomechanics_import_factory(study="AOA", study_id=2222, knee="right", maneuver="sts")
    biomech_record = repository.save_biomechanics_import(biomech, audio_processing_id=audio_record.id)
    db_session.commit()

    # Link audio to biomechanics
    audio_record.biomechanics_import_id = biomech_record.id
    db_session.commit()

    # Create synchronization record
    sync = synchronization_factory(
        study="AOA",
        study_id=2222,
        audio_processing_id=audio_record.id,
        biomechanics_import_id=biomech_record.id,
    )
    sync_record = repository.save_synchronization(
        sync,
        audio_processing_id=audio_record.id,
        biomechanics_import_id=biomech_record.id,
    )
    db_session.commit()

    # Create movement cycle record
    cycle = movement_cycle_factory(
        study="AOA",
        study_id=2222,
        audio_processing_id=audio_record.id,
        biomechanics_import_id=biomech_record.id,
        synchronization_id=sync_record.id,
    )
    repository.save_movement_cycle(
        cycle,
        audio_processing_id=audio_record.id,
        biomechanics_import_id=biomech_record.id,
        synchronization_id=sync_record.id,
    )
    db_session.commit()

    # Generate report sheets
    generator = ReportGenerator(db_session)

    # Test audio sheet
    audio_sheet = generator.generate_audio_sheet(
        study_id=audio_record.study_id,
        maneuver="sts",
        knee="right",
    )
    assert len(audio_sheet) > 0
    assert "Audio File Name" in audio_sheet.columns

    # Test synchronization sheet
    sync_sheet = generator.generate_synchronization_sheet(
        study_id=sync_record.study_id,
        maneuver="sts",
        knee="right",
    )
    assert len(sync_sheet) > 0

    # Test cycles sheet
    cycles_sheet = generator.generate_movement_cycles_sheet(
        study_id=audio_record.study_id,
        maneuver="sts",
        knee="right",
    )
    assert len(cycles_sheet) > 0
    assert "Cycle Index" in cycles_sheet.columns


def test_log_incremental_database_updates(
    db_session,
    repository,
    audio_processing_factory,
    biomechanics_import_factory,
    synchronization_factory,
):
    """Test that database-backed log handles incremental updates correctly."""
    # Create initial audio record
    audio = audio_processing_factory(study="AOA", study_id=3333, knee="left", maneuver="fe")
    audio_record = repository.save_audio_processing(audio)
    db_session.commit()

    # Verify audio record was saved
    assert audio_record.id is not None
    assert audio_record.audio_file_name == audio.audio_file_name

    # Add biomechanics data
    biomech = biomechanics_import_factory(study="AOA", study_id=3333, knee="left", maneuver="fe")
    biomech_record = repository.save_biomechanics_import(biomech, audio_processing_id=audio_record.id)
    db_session.commit()

    # Update audio record to link biomechanics (simulating simultaneous recording)
    audio_record.biomechanics_import_id = biomech_record.id
    db_session.commit()

    # Add synchronization data
    sync = synchronization_factory(
        study="AOA",
        study_id=3333,
        audio_processing_id=audio_record.id,
        biomechanics_import_id=biomech_record.id,
    )
    sync_record = repository.save_synchronization(
        sync,
        audio_processing_id=audio_record.id,
        biomechanics_import_id=biomech_record.id,
    )
    db_session.commit()

    # Verify all records are in database with correct relationships
    assert audio_record.biomechanics_import_id == biomech_record.id
    assert sync_record.audio_processing_id == audio_record.id
    assert sync_record.biomechanics_import_id == biomech_record.id
