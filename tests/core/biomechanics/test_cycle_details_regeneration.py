def test_cycle_upsert_preserves_primary_key(
    repository,
    audio_processing_factory,
    biomechanics_import_factory,
    synchronization_factory,
    movement_cycle_factory,
):
    audio = audio_processing_factory(study="AOA", study_id=5001, knee="left", maneuver="walk")
    audio_record = repository.save_audio_processing(audio)

    biomech = biomechanics_import_factory(study="AOA", study_id=5001, knee="left", maneuver="walk")
    biomech_record = repository.save_biomechanics_import(biomech, audio_processing_id=audio_record.id)

    sync = synchronization_factory(
        study="AOA",
        study_id=5001,
        audio_processing_id=audio_record.id,
        biomechanics_import_id=biomech_record.id,
        sync_file_name="AOA5001_walk_sync",
    )
    sync_record = repository.save_synchronization(
        sync, audio_processing_id=audio_record.id, biomechanics_import_id=biomech_record.id
    )

    cycle = movement_cycle_factory(
        study="AOA",
        study_id=5001,
        audio_processing_id=audio_record.id,
        biomechanics_import_id=biomech_record.id,
        synchronization_id=sync_record.id,
        cycle_file="AOA5001_walk_cycle_0",
        duration_s=1.2,
    )
    first = repository.save_movement_cycle(
        cycle,
        audio_processing_id=audio_record.id,
        biomechanics_import_id=biomech_record.id,
        synchronization_id=sync_record.id,
    )

    updated_cycle = movement_cycle_factory(
        study="AOA",
        study_id=5001,
        audio_processing_id=audio_record.id,
        biomechanics_import_id=biomech_record.id,
        synchronization_id=sync_record.id,
        cycle_file="AOA5001_walk_cycle_0",
        duration_s=2.4,
    )
    second = repository.save_movement_cycle(
        updated_cycle,
        audio_processing_id=audio_record.id,
        biomechanics_import_id=biomech_record.id,
        synchronization_id=sync_record.id,
    )

    assert first.id == second.id
    assert second.duration_s == 2.4
