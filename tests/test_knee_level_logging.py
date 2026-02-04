"""Knee-level DB report tests."""

import pandas as pd

from src.orchestration.processing_log import KneeProcessingLog


def test_knee_log_generates_summary_sheet(
    db_session,
    repository,
    audio_processing_factory,
    biomechanics_import_factory,
    synchronization_factory,
    movement_cycle_factory,
    tmp_path,
):
    audio = audio_processing_factory(study="AOA", study_id=4001, knee="right", maneuver="walk")
    audio_record = repository.save_audio_processing(audio)

    biomech = biomechanics_import_factory(study="AOA", study_id=4001, knee="right", maneuver="walk")
    biomech_record = repository.save_biomechanics_import(biomech, audio_processing_id=audio_record.id)

    sync = synchronization_factory(
        study="AOA",
        study_id=4001,
        audio_processing_id=audio_record.id,
        biomechanics_import_id=biomech_record.id,
        sync_file_name="AOA4001_walk_sync",
    )
    sync_record = repository.save_synchronization(sync, audio_processing_id=audio_record.id, biomechanics_import_id=biomech_record.id)

    cycle = movement_cycle_factory(
        study="AOA",
        study_id=4001,
        audio_processing_id=audio_record.id,
        biomechanics_import_id=biomech_record.id,
        synchronization_id=sync_record.id,
        cycle_file="AOA4001_walk_cycle_0",
    )
    repository.save_movement_cycle(cycle, audio_processing_id=audio_record.id, biomechanics_import_id=biomech_record.id, synchronization_id=sync_record.id)

    knee_log = KneeProcessingLog(
        study_id="AOA4001",
        knee_side="Right",
        knee_directory=tmp_path,
    )
    output_path = knee_log.save_to_excel(session=db_session)

    summary = pd.read_excel(output_path, sheet_name="Knee Summary")
    assert "walk" in set(summary["Maneuver"])
