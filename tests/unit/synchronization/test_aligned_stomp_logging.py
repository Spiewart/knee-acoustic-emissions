"""Tests for aligned stomp times and offset calculations in synchronization logging.

DB-backed tests verify that synchronization records correctly store and report
stomp time calculations and offsets.
"""

import pytest


def test_stomp_offset_calculation(db_session, repository, synchronization_factory, audio_processing_factory, biomechanics_import_factory):
    """Test that stomp offset is correctly calculated and stored in DB.

    Verifies that audio_sync_time, bio_left_sync_time, and sync_offset
    are properly calculated and persisted.
    """
    # Create and save prerequisite audio and biomechanics records
    audio_data = audio_processing_factory()
    audio_record = repository.save_audio_processing(audio_data)
    db_session.commit()

    biomech_data = biomechanics_import_factory()
    biomech_record = repository.save_biomechanics_import(biomech_data)
    db_session.commit()

    # Create a synchronization record with specific stomp times
    sync_data = synchronization_factory(
        aligned_sync_time=5.0,
        bio_left_sync_time=10.0,
        bio_right_sync_time=12.0,
        bio_sync_offset=5.0,  # bio_left - audio = 10 - 5 = 5
    )

    # Save via repository with actual foreign key IDs
    sync_record = repository.save_synchronization(
        sync_data,
        audio_processing_id=audio_record.id,
        biomechanics_import_id=biomech_record.id,
    )

    db_session.commit()

    # Verify stored values
    assert sync_record.aligned_sync_time == pytest.approx(5.0)
    assert sync_record.bio_left_sync_time == pytest.approx(10.0)
    assert sync_record.bio_sync_offset == pytest.approx(5.0)
    assert sync_record.bio_right_sync_time == pytest.approx(12.0)


def test_aligned_stomp_times(db_session, repository, synchronization_factory, audio_processing_factory, biomechanics_import_factory):
    """Test that aligned stomp times are correctly stored.

    After alignment, audio stomp appears at: audio_stomp + offset
    Bio stomp remains at the aligned value (synced timeline is in bio coords).
    """
    # Create and save prerequisite records
    audio_data = audio_processing_factory()
    audio_record = repository.save_audio_processing(audio_data)
    db_session.commit()

    biomech_data = biomechanics_import_factory()
    biomech_record = repository.save_biomechanics_import(biomech_data)
    db_session.commit()

    sync_data = synchronization_factory(
        aligned_sync_time=10.0,  # Combined aligned time for both audio and bio
        bio_left_sync_time=10.0,
        bio_right_sync_time=None,
        bio_sync_offset=5.0,
    )

    sync_record = repository.save_synchronization(
        sync_data,
        audio_processing_id=audio_record.id,
        biomechanics_import_id=biomech_record.id,
    )

    db_session.commit()

    assert sync_record.aligned_sync_time == pytest.approx(10.0)


def test_aligned_stomp_times_right_knee(db_session, repository, synchronization_factory, audio_processing_factory, biomechanics_import_factory):
    """Test aligned stomp calculation for right knee."""
    # Create and save prerequisite records
    audio_data = audio_processing_factory()
    audio_record = repository.save_audio_processing(audio_data)
    db_session.commit()

    biomech_data = biomechanics_import_factory()
    biomech_record = repository.save_biomechanics_import(biomech_data)
    db_session.commit()

    sync_data = synchronization_factory(
        aligned_sync_time=8.0,  # Combined aligned time
        bio_left_sync_time=10.0,  # Not used for right knee
        bio_right_sync_time=8.0,
        bio_sync_offset=5.0,  # 8.0 - 3.0
    )

    sync_record = repository.save_synchronization(
        sync_data,
        audio_processing_id=audio_record.id,
        biomechanics_import_id=biomech_record.id,
    )

    db_session.commit()

    assert sync_record.bio_sync_offset == pytest.approx(5.0)
    assert sync_record.aligned_sync_time == pytest.approx(8.0)


def test_stomp_offset_with_different_methods(db_session, repository, synchronization_factory, audio_processing_factory, biomechanics_import_factory):
    """Test that different sync methods store stomp offsets correctly."""
    # Create prerequisite records
    audio_data = audio_processing_factory()
    audio_record = repository.save_audio_processing(audio_data)
    db_session.commit()

    biomech_data = biomechanics_import_factory()
    biomech_record = repository.save_biomechanics_import(biomech_data)
    db_session.commit()

    # Consensus method with RMS time set
    sync_consensus = synchronization_factory(
        aligned_sync_time=2.0,
        bio_left_sync_time=7.0,
        bio_sync_offset=5.0,
        sync_method="consensus",
        rms_time=2.0,
    )

    record_consensus = repository.save_synchronization(
        sync_consensus,
        audio_processing_id=audio_record.id,
        biomechanics_import_id=biomech_record.id,
    )

    db_session.commit()

    assert record_consensus.selected_stomp_method == "consensus"
    assert record_consensus.bio_sync_offset == pytest.approx(5.0)
    assert record_consensus.rms_time == pytest.approx(2.0)

    # Create second prerequisite set for biomechanics method
    audio_data2 = audio_processing_factory(audio_file_name="test_audio_2.bin")
    audio_record2 = repository.save_audio_processing(audio_data2)
    db_session.commit()

    biomech_data2 = biomechanics_import_factory(biomechanics_file="test_biomech_2.xlsx")
    biomech_record2 = repository.save_biomechanics_import(biomech_data2)
    db_session.commit()

    # Biomechanics method
    sync_biomech = synchronization_factory(
        aligned_sync_time=1.5,
        bio_left_sync_time=6.5,
        bio_sync_offset=5.0,
        sync_method="biomechanics",
        sync_file_name="test_sync_biomech.pkl",
    )

    record_biomech = repository.save_synchronization(
        sync_biomech,
        audio_processing_id=audio_record2.id,
        biomechanics_import_id=biomech_record2.id,
    )

    db_session.commit()

    assert record_biomech.sync_method == "biomechanics"
    assert record_biomech.bio_sync_offset == pytest.approx(5.0)
