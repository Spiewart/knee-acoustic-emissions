"""Regression tests for sync record deduplication.

These tests guard against the double-counting bug where case variations
in sync filenames (e.g., "Left_walk_Pass0001_slow" vs "left_walk_Pass0001_slow")
created duplicate sync records — 20 records for 10 walking passes.

The fix uses:
1. Lowercase normalization in _generate_sync_output_path
2. Semantic multi-column unique constraints instead of filename-based
3. Short, normalized filenames from generate_sync_filename()
"""

from src.studies.file_naming import generate_sync_filename


class TestSyncOutputPathAlwaysLowercase:
    """Verify that sync filenames are always lowercase."""

    def test_walk_filename_lowercase(self):
        filename = generate_sync_filename("left", "walk", pass_number=1, speed="slow")
        assert filename == filename.lower(), f"Filename not lowercase: {filename}"

    def test_fe_filename_lowercase(self):
        filename = generate_sync_filename("right", "fe")
        assert filename == filename.lower(), f"Filename not lowercase: {filename}"

    def test_sts_filename_lowercase(self):
        filename = generate_sync_filename("left", "sts")
        assert filename == filename.lower(), f"Filename not lowercase: {filename}"

    def test_uppercase_input_normalized(self):
        """Even if inputs have mixed case, output should be lowercase."""
        filename = generate_sync_filename("Left", "Walk", pass_number=1, speed="Slow")
        assert filename == filename.lower()

    def test_no_case_variant_duplicates(self):
        """Same logical file should produce the same filename regardless of input case."""
        f1 = generate_sync_filename("left", "walk", pass_number=1, speed="slow")
        f2 = generate_sync_filename("Left", "Walk", pass_number=1, speed="Slow")
        f3 = generate_sync_filename("LEFT", "WALK", pass_number=1, speed="SLOW")
        assert f1 == f2 == f3


class TestSyncUniqueConstraintPreventsDoubles:
    """Verify DB unique constraint blocks duplicate sync records.

    These tests require a PostgreSQL test database.
    """

    def test_same_semantic_key_upserts(
        self,
        db_session,
        synchronization_factory,
        audio_processing_factory,
        biomechanics_import_factory,
    ):
        """Two sync records with same (participant, knee, maneuver, pass, speed) should upsert."""
        from src.db.repository import Repository

        repo = Repository(db_session)

        audio = audio_processing_factory(study="AOA", study_id=9001, knee="left", maneuver="walk")
        audio_record = repo.save_audio_processing(audio)
        db_session.flush()

        biomech = biomechanics_import_factory(study="AOA", study_id=9001, knee="left", maneuver="walk")
        biomech_record = repo.save_biomechanics_import(biomech, audio_processing_id=audio_record.id)
        db_session.flush()

        # Save first sync record
        sync1 = synchronization_factory(
            study="AOA",
            study_id=9001,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
            knee="left",
            maneuver="walk",
            pass_number=1,
            speed="medium",
            sync_file_name="left_walk_p1_medium.pkl",
        )
        record1 = repo.save_synchronization(
            sync1,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
        )
        db_session.flush()

        # Save second sync record with same semantic key
        sync2 = synchronization_factory(
            study="AOA",
            study_id=9001,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
            knee="left",
            maneuver="walk",
            pass_number=1,
            speed="medium",
            sync_file_name="left_walk_p1_medium.pkl",
            aligned_sync_time=99.0,  # Different value to verify upsert
        )
        record2 = repo.save_synchronization(
            sync2,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
        )
        db_session.flush()

        # Should have updated the existing record, not created a new one
        assert record1.id == record2.id, (
            f"Expected upsert (same ID), but got different IDs: {record1.id} vs {record2.id}"
        )

    def test_one_sync_record_per_walking_pass(
        self,
        db_session,
        synchronization_factory,
        audio_processing_factory,
        biomechanics_import_factory,
    ):
        """Each walking pass should produce exactly one sync record."""
        from src.db.models import SynchronizationRecord
        from src.db.repository import Repository

        repo = Repository(db_session)

        audio = audio_processing_factory(study="AOA", study_id=9002, knee="left", maneuver="walk")
        audio_record = repo.save_audio_processing(audio)
        db_session.flush()

        biomech = biomechanics_import_factory(study="AOA", study_id=9002, knee="left", maneuver="walk")
        biomech_record = repo.save_biomechanics_import(biomech, audio_processing_id=audio_record.id)
        db_session.flush()

        # Simulate processing 10 walking passes
        for pass_num in range(1, 11):
            sync = synchronization_factory(
                study="AOA",
                study_id=9002,
                audio_processing_id=audio_record.id,
                biomechanics_import_id=biomech_record.id,
                knee="left",
                maneuver="walk",
                pass_number=pass_num,
                speed="medium",
                sync_file_name=f"left_walk_p{pass_num}_medium.pkl",
            )
            repo.save_synchronization(
                sync,
                audio_processing_id=audio_record.id,
                biomechanics_import_id=biomech_record.id,
            )
        db_session.flush()

        # Count sync records for this participant
        count = (
            db_session.query(SynchronizationRecord)
            .filter(
                SynchronizationRecord.study_id == audio_record.study_id,
                SynchronizationRecord.knee == "left",
                SynchronizationRecord.maneuver == "walk",
            )
            .count()
        )

        assert count == 10, f"Expected 10 sync records (one per pass), but got {count}. Possible double-counting bug."
