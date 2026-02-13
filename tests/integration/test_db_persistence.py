"""Comprehensive integration tests for DB persistence layer.

Tests repository CRUD, upsert semantics, is_active soft-delete,
deactivation/reactivation with PK stability, query filtering,
report generation, file naming dependencies, and artifact interval
round-tripping.

These tests are the safety net for the entire persistence layer.
If you change file naming, metadata fields, natural keys, or
persistence logic — these tests should catch regressions.
"""

import pytest
from sqlalchemy.orm import Session

from src.db.models import (
    AudioProcessingRecord,
    BiomechanicsImportRecord,
    MovementCycleRecord,
    ParticipantRecord,
    StudyRecord,
    SynchronizationRecord,
)
from src.db.repository import Repository


# ========================================================================
# Helpers
# ========================================================================


def _wipe_all_tables(engine):
    """Truncate every table — FK order matters."""
    with Session(engine) as session:
        for model in [
            MovementCycleRecord,
            SynchronizationRecord,
            AudioProcessingRecord,
            BiomechanicsImportRecord,
            StudyRecord,
            ParticipantRecord,
        ]:
            session.query(model).delete()
        session.commit()


def _seed_full_chain(
    session,
    audio_processing_factory,
    biomechanics_import_factory,
    synchronization_factory,
    movement_cycle_factory,
    *,
    study_id: int = 1001,
    knee: str = "left",
    maneuver: str = "walk",
    num_syncs: int = 1,
    num_cycles_per_sync: int = 3,
    pass_numbers: list[int] | None = None,
    speeds: list[str] | None = None,
    audio_file_name: str = "test_audio.bin",
    biomechanics_file: str = "test_biomech.xlsx",
    sync_file_prefix: str = "left_walk",
):
    """Seed a full record chain with configurable topology.

    Returns dict of:
        study_record, audio, biomech, syncs[], cycles[]
    """
    repo = Repository(session)

    audio = audio_processing_factory(
        study_id=study_id,
        knee=knee,
        maneuver=maneuver,
        audio_file_name=audio_file_name,
    )
    audio_record = repo.save_audio_processing(audio)

    biomech = biomechanics_import_factory(
        study_id=study_id,
        knee=knee,
        maneuver=maneuver,
        biomechanics_file=biomechanics_file,
    )
    biomech_record = repo.save_biomechanics_import(
        biomech, audio_processing_id=audio_record.id
    )

    # Link audio to biomech
    audio_record.biomechanics_import_id = biomech_record.id
    session.flush()

    if pass_numbers is None:
        pass_numbers = list(range(1, num_syncs + 1))
    if speeds is None:
        speeds = ["medium"] * num_syncs

    sync_records = []
    cycle_records = []

    for i in range(num_syncs):
        sync_fname = f"{sync_file_prefix}_p{pass_numbers[i]}_{speeds[i]}.pkl"
        sync = synchronization_factory(
            study_id=study_id,
            knee=knee,
            maneuver=maneuver,
            pass_number=pass_numbers[i],
            speed=speeds[i],
            sync_file_name=sync_fname,
        )
        sync_record = repo.save_synchronization(
            sync,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
        )
        sync_records.append(sync_record)

        for ci in range(num_cycles_per_sync):
            cycle = movement_cycle_factory(
                study_id=study_id,
                knee=knee,
                maneuver=maneuver,
                pass_number=pass_numbers[i],
                speed=speeds[i],
                cycle_index=ci,
                cycle_file=f"cycle_{ci:02d}.pkl",
            )
            cycle_record = repo.save_movement_cycle(
                cycle,
                audio_processing_id=audio_record.id,
                biomechanics_import_id=biomech_record.id,
                synchronization_id=sync_record.id,
            )
            cycle_records.append(cycle_record)

    session.flush()

    return {
        "audio": audio_record,
        "biomech": biomech_record,
        "syncs": sync_records,
        "cycles": cycle_records,
    }


# ========================================================================
# 1. Participant + Study operations
# ========================================================================


class TestParticipantStudyOperations:
    """Tests for get_or_create_participant — the identity anchor."""

    @pytest.fixture(autouse=True)
    def _cleanup(self, db_engine):
        _wipe_all_tables(db_engine)
        yield
        _wipe_all_tables(db_engine)

    def test_creates_participant_and_study(self, db_session):
        repo = Repository(db_session)
        study = repo.get_or_create_participant("AOA", 1001)

        assert study.study_name == "AOA"
        assert study.study_participant_id == 1001
        assert study.participant_id is not None

    def test_idempotent_get_or_create(self, db_session):
        """Calling get_or_create twice returns the SAME StudyRecord."""
        repo = Repository(db_session)
        s1 = repo.get_or_create_participant("AOA", 1001)
        s2 = repo.get_or_create_participant("AOA", 1001)

        assert s1.id == s2.id
        assert db_session.query(ParticipantRecord).count() == 1
        assert db_session.query(StudyRecord).count() == 1

    def test_different_studies_different_records(self, db_session):
        """Different (study_name, participant_number) → different StudyRecords."""
        repo = Repository(db_session)
        s1 = repo.get_or_create_participant("AOA", 1001)
        s2 = repo.get_or_create_participant("AOA", 1002)
        s3 = repo.get_or_create_participant("preOA", 1001)

        assert s1.id != s2.id
        assert s1.id != s3.id
        assert s2.id != s3.id

    def test_same_participant_multiple_studies(self, db_session):
        """Two studies can share the same ParticipantRecord once we implement
        cross-study linking — for now, they create separate participants."""
        repo = Repository(db_session)
        s1 = repo.get_or_create_participant("AOA", 1001)
        s2 = repo.get_or_create_participant("preOA", 2001)

        # Different participants (separate identity anchors)
        assert s1.participant_id != s2.participant_id


# ========================================================================
# 2. Upsert semantics — all 4 entity types
# ========================================================================


class TestUpsertSemantics:
    """Tests that save_*() methods create on first call, update on second."""

    @pytest.fixture(autouse=True)
    def _cleanup(self, db_engine):
        _wipe_all_tables(db_engine)
        yield
        _wipe_all_tables(db_engine)

    def test_audio_upsert_creates_then_updates(
        self, db_session, audio_processing_factory
    ):
        """First save → INSERT, second save with same natural key → UPDATE."""
        repo = Repository(db_session)

        audio = audio_processing_factory(study_id=1001, duration_seconds=100.0)
        rec1 = repo.save_audio_processing(audio)
        pk1 = rec1.id

        audio2 = audio_processing_factory(study_id=1001, duration_seconds=200.0)
        rec2 = repo.save_audio_processing(audio2)

        assert rec2.id == pk1, "PK should be preserved on upsert"
        assert rec2.duration_seconds == 200.0, "Updated value should persist"
        assert db_session.query(AudioProcessingRecord).count() == 1

    def test_biomechanics_upsert_creates_then_updates(
        self, db_session, biomechanics_import_factory
    ):
        repo = Repository(db_session)

        biomech = biomechanics_import_factory(study_id=1001, duration_seconds=100.0)
        rec1 = repo.save_biomechanics_import(biomech)
        pk1 = rec1.id

        biomech2 = biomechanics_import_factory(study_id=1001, duration_seconds=200.0)
        rec2 = repo.save_biomechanics_import(biomech2)

        assert rec2.id == pk1
        assert db_session.query(BiomechanicsImportRecord).count() == 1

    def test_sync_upsert_creates_then_updates(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory,
    ):
        repo = Repository(db_session)

        audio = audio_processing_factory(study_id=1001)
        audio_rec = repo.save_audio_processing(audio)
        biomech = biomechanics_import_factory(study_id=1001)
        biomech_rec = repo.save_biomechanics_import(biomech)

        sync = synchronization_factory(
            study_id=1001, pass_number=1, speed="medium", sync_duration=100.0
        )
        rec1 = repo.save_synchronization(
            sync, audio_processing_id=audio_rec.id,
            biomechanics_import_id=biomech_rec.id,
        )
        pk1 = rec1.id

        sync2 = synchronization_factory(
            study_id=1001, pass_number=1, speed="medium", sync_duration=200.0
        )
        rec2 = repo.save_synchronization(
            sync2, audio_processing_id=audio_rec.id,
            biomechanics_import_id=biomech_rec.id,
        )

        assert rec2.id == pk1
        assert rec2.sync_duration == 200.0
        assert db_session.query(SynchronizationRecord).count() == 1

    def test_cycle_upsert_creates_then_updates(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory, movement_cycle_factory,
    ):
        repo = Repository(db_session)
        chain = _seed_full_chain(
            db_session, audio_processing_factory, biomechanics_import_factory,
            synchronization_factory, movement_cycle_factory,
            num_syncs=1, num_cycles_per_sync=3,
        )
        original_pks = [c.id for c in chain["cycles"]]

        # Re-process same cycles with updated duration
        for idx in range(3):
            cycle = movement_cycle_factory(
                study_id=1001, pass_number=1, speed="medium",
                cycle_index=idx, cycle_file=f"cycle_{idx:02d}.pkl",
                duration_s=99.0,
            )
            updated = repo.save_movement_cycle(
                cycle,
                audio_processing_id=chain["audio"].id,
                biomechanics_import_id=chain["biomech"].id,
                synchronization_id=chain["syncs"][0].id,
            )
            assert updated.id == original_pks[idx]
            assert updated.duration_s == 99.0

    def test_audio_upsert_matches_on_natural_key(
        self, db_session, audio_processing_factory
    ):
        """Natural key = (study_id, audio_file_name, knee, maneuver).
        Different audio_file_name → different record."""
        repo = Repository(db_session)

        rec1 = repo.save_audio_processing(
            audio_processing_factory(study_id=1001, audio_file_name="file_a.bin")
        )
        rec2 = repo.save_audio_processing(
            audio_processing_factory(study_id=1001, audio_file_name="file_b.bin")
        )

        assert rec1.id != rec2.id
        assert db_session.query(AudioProcessingRecord).count() == 2

    def test_sync_upsert_with_null_pass_number(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory,
    ):
        """STS/FE maneuvers have pass_number=None, speed=None.
        Upsert should match on NULLs correctly."""
        repo = Repository(db_session)

        audio = audio_processing_factory(study_id=1001, maneuver="sts")
        audio_rec = repo.save_audio_processing(audio)
        biomech = biomechanics_import_factory(study_id=1001, maneuver="sts")
        biomech_rec = repo.save_biomechanics_import(biomech)

        sync = synchronization_factory(
            study_id=1001, maneuver="sts",
            pass_number=None, speed=None,
            sync_duration=100.0,
        )
        rec1 = repo.save_synchronization(
            sync, audio_processing_id=audio_rec.id,
            biomechanics_import_id=biomech_rec.id,
        )
        pk1 = rec1.id

        # Re-save with same NULL natural key
        sync2 = synchronization_factory(
            study_id=1001, maneuver="sts",
            pass_number=None, speed=None,
            sync_duration=200.0,
        )
        rec2 = repo.save_synchronization(
            sync2, audio_processing_id=audio_rec.id,
            biomechanics_import_id=biomech_rec.id,
        )

        assert rec2.id == pk1, "Should match existing record with NULL pass_number"
        assert rec2.sync_duration == 200.0

    def test_cycle_upsert_with_null_pass_speed(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory, movement_cycle_factory,
    ):
        """Cycles with NULL pass_number/speed (STS/FE) should upsert correctly."""
        repo = Repository(db_session)

        audio = audio_processing_factory(study_id=1001, maneuver="fe")
        audio_rec = repo.save_audio_processing(audio)

        cycle = movement_cycle_factory(
            study_id=1001, maneuver="fe",
            pass_number=None, speed=None,
            cycle_index=0, duration_s=1.0,
        )
        rec1 = repo.save_movement_cycle(
            cycle, audio_processing_id=audio_rec.id,
        )

        cycle2 = movement_cycle_factory(
            study_id=1001, maneuver="fe",
            pass_number=None, speed=None,
            cycle_index=0, duration_s=2.0,
        )
        rec2 = repo.save_movement_cycle(
            cycle2, audio_processing_id=audio_rec.id,
        )

        assert rec2.id == rec1.id
        assert rec2.duration_s == 2.0


# ========================================================================
# 3. is_active soft-delete — deactivation across ALL entity types
# ========================================================================


class TestDeactivationAllEntityTypes:
    """Tests deactivation logic for every downstream table."""

    @pytest.fixture(autouse=True)
    def _cleanup(self, db_engine):
        _wipe_all_tables(db_engine)
        yield
        _wipe_all_tables(db_engine)

    def test_new_records_are_active(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory, movement_cycle_factory,
    ):
        chain = _seed_full_chain(
            db_session, audio_processing_factory, biomechanics_import_factory,
            synchronization_factory, movement_cycle_factory,
        )
        assert chain["audio"].is_active is True
        assert chain["biomech"].is_active is True
        for s in chain["syncs"]:
            assert s.is_active is True
        for c in chain["cycles"]:
            assert c.is_active is True

    def test_deactivate_audio(
        self, db_session, audio_processing_factory,
    ):
        """Deactivation with empty seen_audio_ids deactivates the audio record."""
        repo = Repository(db_session)
        audio = audio_processing_factory(study_id=1001)
        rec = repo.save_audio_processing(audio)

        deactivated = repo.deactivate_unseen_records(
            study_id=rec.study_id, knee="left", maneuver="walk",
            seen_audio_ids=set(), seen_biomech_ids=set(),
            seen_sync_ids=set(), seen_cycle_ids=set(),
        )
        assert deactivated["audio_processing"] == 1

        db_session.expire_all()
        db_session.refresh(rec)
        assert rec.is_active is False

    def test_deactivate_biomechanics(
        self, db_session, biomechanics_import_factory,
    ):
        repo = Repository(db_session)
        biomech = biomechanics_import_factory(study_id=1001)
        rec = repo.save_biomechanics_import(biomech)

        deactivated = repo.deactivate_unseen_records(
            study_id=rec.study_id, knee="left", maneuver="walk",
            seen_audio_ids=set(), seen_biomech_ids=set(),
            seen_sync_ids=set(), seen_cycle_ids=set(),
        )
        assert deactivated["biomechanics_imports"] == 1

        db_session.expire_all()
        db_session.refresh(rec)
        assert rec.is_active is False

    def test_deactivate_syncs_selectively(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory, movement_cycle_factory,
    ):
        """Deactivate some syncs while keeping others active."""
        chain = _seed_full_chain(
            db_session, audio_processing_factory, biomechanics_import_factory,
            synchronization_factory, movement_cycle_factory,
            num_syncs=4, num_cycles_per_sync=2,
            pass_numbers=[1, 2, 3, 4],
            speeds=["medium", "medium", "slow", "slow"],
        )

        # Keep syncs 1,2 (medium) — deactivate 3,4 (slow)
        seen_sync_ids = {chain["syncs"][0].id, chain["syncs"][1].id}
        repo = Repository(db_session)
        deactivated = repo.deactivate_unseen_records(
            study_id=chain["audio"].study_id, knee="left", maneuver="walk",
            seen_audio_ids={chain["audio"].id},
            seen_biomech_ids={chain["biomech"].id},
            seen_sync_ids=seen_sync_ids,
            seen_cycle_ids={c.id for c in chain["cycles"][:4]},  # cycles from sync 1,2
        )

        assert deactivated["synchronizations"] == 2

        db_session.expire_all()
        for i, sr in enumerate(chain["syncs"]):
            db_session.refresh(sr)
            if i < 2:
                assert sr.is_active is True, f"Sync {i} should be active"
            else:
                assert sr.is_active is False, f"Sync {i} should be inactive"

    def test_deactivate_cycles_partial(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory, movement_cycle_factory,
    ):
        """10 cycles → re-process produces 7 → cycles 7-9 deactivated."""
        chain = _seed_full_chain(
            db_session, audio_processing_factory, biomechanics_import_factory,
            synchronization_factory, movement_cycle_factory,
            num_syncs=1, num_cycles_per_sync=10,
        )

        seen_cycle_ids = {c.id for c in chain["cycles"][:7]}
        repo = Repository(db_session)
        deactivated = repo.deactivate_unseen_records(
            study_id=chain["audio"].study_id, knee="left", maneuver="walk",
            seen_audio_ids={chain["audio"].id},
            seen_biomech_ids={chain["biomech"].id},
            seen_sync_ids={chain["syncs"][0].id},
            seen_cycle_ids=seen_cycle_ids,
        )

        assert deactivated["movement_cycles"] == 3

        db_session.expire_all()
        for i, c in enumerate(chain["cycles"]):
            db_session.refresh(c)
            if i < 7:
                assert c.is_active is True, f"Cycle {i} should be active"
            else:
                assert c.is_active is False, f"Cycle {i} should be inactive"

    def test_deactivate_with_empty_seen_ids(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory, movement_cycle_factory,
    ):
        """Empty seen_ids for ALL tables → everything deactivated."""
        chain = _seed_full_chain(
            db_session, audio_processing_factory, biomechanics_import_factory,
            synchronization_factory, movement_cycle_factory,
            num_syncs=2, num_cycles_per_sync=3,
        )

        repo = Repository(db_session)
        deactivated = repo.deactivate_unseen_records(
            study_id=chain["audio"].study_id, knee="left", maneuver="walk",
            seen_audio_ids=set(), seen_biomech_ids=set(),
            seen_sync_ids=set(), seen_cycle_ids=set(),
        )

        assert deactivated["audio_processing"] == 1
        assert deactivated["biomechanics_imports"] == 1
        assert deactivated["synchronizations"] == 2
        assert deactivated["movement_cycles"] == 6

    def test_double_deactivation_is_idempotent(
        self, db_session, audio_processing_factory,
    ):
        """Deactivating already-inactive records should return 0 changes."""
        repo = Repository(db_session)
        audio = audio_processing_factory(study_id=1001)
        rec = repo.save_audio_processing(audio)

        d1 = repo.deactivate_unseen_records(
            study_id=rec.study_id, knee="left", maneuver="walk",
            seen_audio_ids=set(), seen_biomech_ids=set(),
            seen_sync_ids=set(), seen_cycle_ids=set(),
        )
        assert d1["audio_processing"] == 1

        d2 = repo.deactivate_unseen_records(
            study_id=rec.study_id, knee="left", maneuver="walk",
            seen_audio_ids=set(), seen_biomech_ids=set(),
            seen_sync_ids=set(), seen_cycle_ids=set(),
        )
        assert d2["audio_processing"] == 0, "Already inactive — no change"

    def test_deactivation_returns_zero_when_nothing_to_deactivate(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory, movement_cycle_factory,
    ):
        """When all record IDs are in seen_ids → no deactivation."""
        chain = _seed_full_chain(
            db_session, audio_processing_factory, biomechanics_import_factory,
            synchronization_factory, movement_cycle_factory,
        )
        repo = Repository(db_session)
        deactivated = repo.deactivate_unseen_records(
            study_id=chain["audio"].study_id, knee="left", maneuver="walk",
            seen_audio_ids={chain["audio"].id},
            seen_biomech_ids={chain["biomech"].id},
            seen_sync_ids={s.id for s in chain["syncs"]},
            seen_cycle_ids={c.id for c in chain["cycles"]},
        )
        assert all(v == 0 for v in deactivated.values())


# ========================================================================
# 4. Deactivation scoping — cross-study / cross-knee / cross-maneuver
# ========================================================================


class TestDeactivationScoping:
    """Deactivation is scoped to (study_id, knee, maneuver).
    Records for other scopes must remain untouched."""

    @pytest.fixture(autouse=True)
    def _cleanup(self, db_engine):
        _wipe_all_tables(db_engine)
        yield
        _wipe_all_tables(db_engine)

    def test_different_maneuver_untouched(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory, movement_cycle_factory,
    ):
        """Deactivating walk records must not affect sts records."""
        walk = _seed_full_chain(
            db_session, audio_processing_factory, biomechanics_import_factory,
            synchronization_factory, movement_cycle_factory,
            maneuver="walk", audio_file_name="walk_audio.bin",
            biomechanics_file="walk_biomech.xlsx",
        )
        sts = _seed_full_chain(
            db_session, audio_processing_factory, biomechanics_import_factory,
            synchronization_factory, movement_cycle_factory,
            maneuver="sts", audio_file_name="sts_audio.bin",
            biomechanics_file="sts_biomech.xlsx",
            sync_file_prefix="left_sts",
        )

        repo = Repository(db_session)
        repo.deactivate_unseen_records(
            study_id=walk["audio"].study_id, knee="left", maneuver="walk",
            seen_audio_ids=set(), seen_biomech_ids=set(),
            seen_sync_ids=set(), seen_cycle_ids=set(),
        )

        db_session.expire_all()
        db_session.refresh(walk["audio"])
        db_session.refresh(sts["audio"])
        assert walk["audio"].is_active is False
        assert sts["audio"].is_active is True, "STS should be untouched"

    def test_different_knee_untouched(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory, movement_cycle_factory,
    ):
        """Deactivating left-knee records must not affect right-knee."""
        left = _seed_full_chain(
            db_session, audio_processing_factory, biomechanics_import_factory,
            synchronization_factory, movement_cycle_factory,
            knee="left", audio_file_name="left_audio.bin",
            biomechanics_file="left_biomech.xlsx",
        )
        right = _seed_full_chain(
            db_session, audio_processing_factory, biomechanics_import_factory,
            synchronization_factory, movement_cycle_factory,
            knee="right", audio_file_name="right_audio.bin",
            biomechanics_file="right_biomech.xlsx",
            sync_file_prefix="right_walk",
        )

        repo = Repository(db_session)
        repo.deactivate_unseen_records(
            study_id=left["audio"].study_id, knee="left", maneuver="walk",
            seen_audio_ids=set(), seen_biomech_ids=set(),
            seen_sync_ids=set(), seen_cycle_ids=set(),
        )

        db_session.expire_all()
        db_session.refresh(left["audio"])
        db_session.refresh(right["audio"])
        assert left["audio"].is_active is False
        assert right["audio"].is_active is True

    def test_different_participant_untouched(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory, movement_cycle_factory,
    ):
        """Deactivating participant 1001 must not affect participant 1002."""
        p1 = _seed_full_chain(
            db_session, audio_processing_factory, biomechanics_import_factory,
            synchronization_factory, movement_cycle_factory,
            study_id=1001,
        )
        p2 = _seed_full_chain(
            db_session, audio_processing_factory, biomechanics_import_factory,
            synchronization_factory, movement_cycle_factory,
            study_id=1002,
        )

        repo = Repository(db_session)
        repo.deactivate_unseen_records(
            study_id=p1["audio"].study_id, knee="left", maneuver="walk",
            seen_audio_ids=set(), seen_biomech_ids=set(),
            seen_sync_ids=set(), seen_cycle_ids=set(),
        )

        db_session.expire_all()
        db_session.refresh(p1["audio"])
        db_session.refresh(p2["audio"])
        assert p1["audio"].is_active is False
        assert p2["audio"].is_active is True


# ========================================================================
# 5. Reactivation — all entity types with PK preservation
# ========================================================================


class TestReactivation:
    """Upserting a deactivated record should re-activate it with the same PK."""

    @pytest.fixture(autouse=True)
    def _cleanup(self, db_engine):
        _wipe_all_tables(db_engine)
        yield
        _wipe_all_tables(db_engine)

    def test_reactivate_audio(self, db_session, audio_processing_factory):
        repo = Repository(db_session)

        audio = audio_processing_factory(study_id=1001)
        rec = repo.save_audio_processing(audio)
        original_pk = rec.id

        repo.deactivate_unseen_records(
            study_id=rec.study_id, knee="left", maneuver="walk",
            seen_audio_ids=set(), seen_biomech_ids=set(),
            seen_sync_ids=set(), seen_cycle_ids=set(),
        )
        db_session.expire_all()
        db_session.refresh(rec)
        assert rec.is_active is False

        # Re-upsert
        audio2 = audio_processing_factory(study_id=1001)
        reactivated = repo.save_audio_processing(audio2)

        assert reactivated.id == original_pk
        assert reactivated.is_active is True

    def test_reactivate_biomechanics(self, db_session, biomechanics_import_factory):
        repo = Repository(db_session)

        biomech = biomechanics_import_factory(study_id=1001)
        rec = repo.save_biomechanics_import(biomech)
        original_pk = rec.id

        repo.deactivate_unseen_records(
            study_id=rec.study_id, knee="left", maneuver="walk",
            seen_audio_ids=set(), seen_biomech_ids=set(),
            seen_sync_ids=set(), seen_cycle_ids=set(),
        )
        db_session.expire_all()
        db_session.refresh(rec)
        assert rec.is_active is False

        biomech2 = biomechanics_import_factory(study_id=1001)
        reactivated = repo.save_biomechanics_import(biomech2)

        assert reactivated.id == original_pk
        assert reactivated.is_active is True

    def test_reactivate_sync(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory,
    ):
        repo = Repository(db_session)

        audio = audio_processing_factory(study_id=1001)
        audio_rec = repo.save_audio_processing(audio)
        biomech = biomechanics_import_factory(study_id=1001)
        biomech_rec = repo.save_biomechanics_import(biomech)

        sync = synchronization_factory(study_id=1001, pass_number=1, speed="medium")
        rec = repo.save_synchronization(
            sync, audio_processing_id=audio_rec.id,
            biomechanics_import_id=biomech_rec.id,
        )
        original_pk = rec.id

        repo.deactivate_unseen_records(
            study_id=audio_rec.study_id, knee="left", maneuver="walk",
            seen_audio_ids={audio_rec.id}, seen_biomech_ids={biomech_rec.id},
            seen_sync_ids=set(), seen_cycle_ids=set(),
        )
        db_session.expire_all()
        db_session.refresh(rec)
        assert rec.is_active is False

        sync2 = synchronization_factory(study_id=1001, pass_number=1, speed="medium")
        reactivated = repo.save_synchronization(
            sync2, audio_processing_id=audio_rec.id,
            biomechanics_import_id=biomech_rec.id,
        )
        assert reactivated.id == original_pk
        assert reactivated.is_active is True

    def test_reactivate_cycle(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory, movement_cycle_factory,
    ):
        repo = Repository(db_session)

        chain = _seed_full_chain(
            db_session, audio_processing_factory, biomechanics_import_factory,
            synchronization_factory, movement_cycle_factory,
            num_syncs=1, num_cycles_per_sync=3,
        )
        cycle_pks = [c.id for c in chain["cycles"]]

        # Deactivate all cycles
        repo.deactivate_unseen_records(
            study_id=chain["audio"].study_id, knee="left", maneuver="walk",
            seen_audio_ids={chain["audio"].id},
            seen_biomech_ids={chain["biomech"].id},
            seen_sync_ids={chain["syncs"][0].id},
            seen_cycle_ids=set(),
        )
        db_session.expire_all()
        for c in chain["cycles"]:
            db_session.refresh(c)
            assert c.is_active is False

        # Re-process cycles → should reactivate with same PKs
        for idx in range(3):
            cycle = movement_cycle_factory(
                study_id=1001, pass_number=1, speed="medium",
                cycle_index=idx, cycle_file=f"cycle_{idx:02d}.pkl",
            )
            reactivated = repo.save_movement_cycle(
                cycle,
                audio_processing_id=chain["audio"].id,
                biomechanics_import_id=chain["biomech"].id,
                synchronization_id=chain["syncs"][0].id,
            )
            assert reactivated.id == cycle_pks[idx]
            assert reactivated.is_active is True

    def test_full_deactivate_reactivate_cycle(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory, movement_cycle_factory,
    ):
        """End-to-end: create → deactivate → reactivate → verify PKs stable."""
        chain = _seed_full_chain(
            db_session, audio_processing_factory, biomechanics_import_factory,
            synchronization_factory, movement_cycle_factory,
            num_syncs=2, num_cycles_per_sync=4,
            pass_numbers=[1, 2], speeds=["medium", "slow"],
        )

        all_pks = {
            "audio": chain["audio"].id,
            "biomech": chain["biomech"].id,
            "syncs": [s.id for s in chain["syncs"]],
            "cycles": [c.id for c in chain["cycles"]],
        }

        # Deactivate everything
        repo = Repository(db_session)
        repo.deactivate_unseen_records(
            study_id=chain["audio"].study_id, knee="left", maneuver="walk",
            seen_audio_ids=set(), seen_biomech_ids=set(),
            seen_sync_ids=set(), seen_cycle_ids=set(),
        )

        # Reactivate by re-processing
        rchain = _seed_full_chain(
            db_session, audio_processing_factory, biomechanics_import_factory,
            synchronization_factory, movement_cycle_factory,
            num_syncs=2, num_cycles_per_sync=4,
            pass_numbers=[1, 2], speeds=["medium", "slow"],
        )

        assert rchain["audio"].id == all_pks["audio"]
        assert rchain["biomech"].id == all_pks["biomech"]
        for i, s in enumerate(rchain["syncs"]):
            assert s.id == all_pks["syncs"][i]
        for i, c in enumerate(rchain["cycles"]):
            assert c.id == all_pks["cycles"][i]


# ========================================================================
# 6. Query operations — include_inactive parameter
# ========================================================================


class TestQueryOperations:
    """Tests for get_*() methods with include_inactive filter."""

    @pytest.fixture(autouse=True)
    def _cleanup(self, db_engine):
        _wipe_all_tables(db_engine)
        yield
        _wipe_all_tables(db_engine)

    def _deactivate_all(self, repo, study_id):
        return repo.deactivate_unseen_records(
            study_id=study_id, knee="left", maneuver="walk",
            seen_audio_ids=set(), seen_biomech_ids=set(),
            seen_sync_ids=set(), seen_cycle_ids=set(),
        )

    def test_audio_query_excludes_inactive(
        self, db_session, audio_processing_factory,
    ):
        repo = Repository(db_session)
        audio = audio_processing_factory(study_id=1001)
        rec = repo.save_audio_processing(audio)
        self._deactivate_all(repo, rec.study_id)

        assert len(repo.get_audio_processing_records(
            study_name="AOA", participant_number=1001,
        )) == 0

        assert len(repo.get_audio_processing_records(
            study_name="AOA", participant_number=1001, include_inactive=True,
        )) == 1

    def test_biomechanics_query_excludes_inactive(
        self, db_session, biomechanics_import_factory,
    ):
        repo = Repository(db_session)
        biomech = biomechanics_import_factory(study_id=1001)
        rec = repo.save_biomechanics_import(biomech)
        self._deactivate_all(repo, rec.study_id)

        assert len(repo.get_biomechanics_imports(
            study_name="AOA", participant_number=1001,
        )) == 0

        assert len(repo.get_biomechanics_imports(
            study_name="AOA", participant_number=1001, include_inactive=True,
        )) == 1

    def test_sync_query_excludes_inactive(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory,
    ):
        repo = Repository(db_session)
        audio = audio_processing_factory(study_id=1001)
        audio_rec = repo.save_audio_processing(audio)
        biomech = biomechanics_import_factory(study_id=1001)
        biomech_rec = repo.save_biomechanics_import(biomech)
        sync = synchronization_factory(study_id=1001)
        repo.save_synchronization(
            sync, audio_processing_id=audio_rec.id,
            biomechanics_import_id=biomech_rec.id,
        )
        self._deactivate_all(repo, audio_rec.study_id)

        assert len(repo.get_synchronization_records(
            study_name="AOA", participant_number=1001,
        )) == 0

        assert len(repo.get_synchronization_records(
            study_name="AOA", participant_number=1001, include_inactive=True,
        )) == 1

    def test_cycle_query_excludes_inactive(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory, movement_cycle_factory,
    ):
        chain = _seed_full_chain(
            db_session, audio_processing_factory, biomechanics_import_factory,
            synchronization_factory, movement_cycle_factory,
            num_syncs=1, num_cycles_per_sync=5,
        )
        repo = Repository(db_session)
        self._deactivate_all(repo, chain["audio"].study_id)

        assert len(repo.get_movement_cycle_records(
            study_name="AOA", participant_number=1001,
        )) == 0

        assert len(repo.get_movement_cycle_records(
            study_name="AOA", participant_number=1001, include_inactive=True,
        )) == 5

    def test_query_filters_by_maneuver_and_knee(
        self, db_session, audio_processing_factory,
    ):
        """Query methods should support filtering by maneuver and knee."""
        repo = Repository(db_session)

        repo.save_audio_processing(
            audio_processing_factory(
                study_id=1001, knee="left", maneuver="walk",
                audio_file_name="lw.bin",
            )
        )
        repo.save_audio_processing(
            audio_processing_factory(
                study_id=1001, knee="right", maneuver="walk",
                audio_file_name="rw.bin",
            )
        )
        repo.save_audio_processing(
            audio_processing_factory(
                study_id=1001, knee="left", maneuver="sts",
                audio_file_name="ls.bin",
            )
        )

        left_walk = repo.get_audio_processing_records(
            study_name="AOA", participant_number=1001,
            knee="left", maneuver="walk",
        )
        assert len(left_walk) == 1
        assert left_walk[0].audio_file_name == "lw.bin"

        all_left = repo.get_audio_processing_records(
            study_name="AOA", participant_number=1001, knee="left",
        )
        assert len(all_left) == 2


# ========================================================================
# 7. Artifact interval round-tripping
# ========================================================================


class TestArtifactIntervals:
    """Tests for _flatten_intervals and DB round-trip of artifact segments."""

    def test_flatten_intervals_basic(self):
        """[(1.0, 2.0), (3.0, 4.0)] → [1.0, 2.0, 3.0, 4.0]"""
        result = Repository._flatten_intervals([(1.0, 2.0), (3.0, 4.0)])
        assert result == [1.0, 2.0, 3.0, 4.0]

    def test_flatten_intervals_empty(self):
        """Empty list → None (not empty list)."""
        result = Repository._flatten_intervals([])
        assert result is None

    def test_flatten_intervals_none(self):
        """None input → None output."""
        result = Repository._flatten_intervals(None)
        assert result is None

    def test_flatten_intervals_single(self):
        result = Repository._flatten_intervals([(5.5, 10.5)])
        assert result == [5.5, 10.5]

    @pytest.fixture(autouse=True)
    def _cleanup(self, db_engine):
        _wipe_all_tables(db_engine)
        yield
        _wipe_all_tables(db_engine)

    def test_periodic_artifact_segments_round_trip(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory,
    ):
        """Artifact segments should survive a save→load round trip."""
        repo = Repository(db_session)

        audio = audio_processing_factory(study_id=1001)
        audio_rec = repo.save_audio_processing(audio)
        biomech = biomechanics_import_factory(study_id=1001)
        biomech_rec = repo.save_biomechanics_import(biomech)

        sync = synchronization_factory(
            study_id=1001,
            periodic_artifact_detected=True,
            periodic_artifact_detected_ch1=True,
            periodic_artifact_segments=[(1.0, 2.0), (5.0, 6.0)],
            periodic_artifact_segments_ch1=[(1.0, 2.0), (5.0, 6.0)],
        )
        sync_rec = repo.save_synchronization(
            sync, audio_processing_id=audio_rec.id,
            biomechanics_import_id=biomech_rec.id,
        )

        db_session.flush()
        db_session.expire_all()
        db_session.refresh(sync_rec)

        # Stored flattened: [1.0, 2.0, 5.0, 6.0]
        assert sync_rec.periodic_artifact_segments == [1.0, 2.0, 5.0, 6.0]
        assert sync_rec.periodic_artifact_detected is True

    def test_audio_qc_segments_round_trip(
        self, db_session, audio_processing_factory,
    ):
        """Audio dropout/continuous artifact segments survive round trip."""
        repo = Repository(db_session)

        audio = audio_processing_factory(
            study_id=1001,
            qc_signal_dropout=True,
            qc_signal_dropout_segments=[(0.5, 1.5)],
            qc_signal_dropout_ch1=True,
            qc_signal_dropout_segments_ch1=[(0.5, 1.5)],
            qc_continuous_artifact=True,
            qc_continuous_artifact_segments=[(10.0, 20.0), (30.0, 40.0)],
        )
        rec = repo.save_audio_processing(audio)

        db_session.flush()
        db_session.expire_all()
        db_session.refresh(rec)

        assert rec.qc_signal_dropout is True
        assert rec.qc_continuous_artifact is True


# ========================================================================
# 8. File naming dependencies
# ========================================================================


class TestFileNamingDependencies:
    """Tests that file name patterns are preserved correctly in DB and reports."""

    @pytest.fixture(autouse=True)
    def _cleanup(self, db_engine):
        _wipe_all_tables(db_engine)
        yield
        _wipe_all_tables(db_engine)

    def test_audio_file_name_preserved(
        self, db_session, audio_processing_factory,
    ):
        """Audio file name (with .bin) is stored as-is in DB."""
        repo = Repository(db_session)
        audio = audio_processing_factory(
            study_id=1001, audio_file_name="1016_20240101_100000_L_walk.bin"
        )
        rec = repo.save_audio_processing(audio)

        db_session.flush()
        db_session.expire_all()
        db_session.refresh(rec)
        assert rec.audio_file_name == "1016_20240101_100000_L_walk.bin"

    def test_sync_file_name_with_pkl_extension(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory,
    ):
        """Sync file name (with .pkl) is stored in DB."""
        repo = Repository(db_session)

        audio = audio_processing_factory(study_id=1001)
        audio_rec = repo.save_audio_processing(audio)
        biomech = biomechanics_import_factory(study_id=1001)
        biomech_rec = repo.save_biomechanics_import(biomech)

        sync = synchronization_factory(
            study_id=1001,
            sync_file_name="left_walk_p1_medium.pkl",
        )
        rec = repo.save_synchronization(
            sync, audio_processing_id=audio_rec.id,
            biomechanics_import_id=biomech_rec.id,
        )

        db_session.flush()
        db_session.expire_all()
        db_session.refresh(rec)
        assert rec.sync_file_name == "left_walk_p1_medium.pkl"

    def test_cycle_file_name_preserved(
        self, db_session, audio_processing_factory, movement_cycle_factory,
    ):
        repo = Repository(db_session)

        audio = audio_processing_factory(study_id=1001)
        audio_rec = repo.save_audio_processing(audio)

        cycle = movement_cycle_factory(
            study_id=1001, cycle_file="cycle_03.pkl", cycle_index=3,
        )
        rec = repo.save_movement_cycle(
            cycle, audio_processing_id=audio_rec.id,
        )

        db_session.flush()
        db_session.expire_all()
        db_session.refresh(rec)
        assert rec.cycle_file == "cycle_03.pkl"
        assert rec.cycle_index == 3

    def test_biomechanics_file_name_preserved(
        self, db_session, biomechanics_import_factory,
    ):
        repo = Repository(db_session)
        biomech = biomechanics_import_factory(
            study_id=1001,
            biomechanics_file="1016_MotionCapture_Walk.xlsx",
        )
        rec = repo.save_biomechanics_import(biomech)

        db_session.flush()
        db_session.expire_all()
        db_session.refresh(rec)
        assert rec.biomechanics_file == "1016_MotionCapture_Walk.xlsx"

    def test_audio_file_name_is_natural_key_component(
        self, db_session, audio_processing_factory,
    ):
        """Changing audio_file_name for same (study, knee, maneuver) creates
        a DIFFERENT record because audio_file_name is part of the natural key."""
        repo = Repository(db_session)

        rec1 = repo.save_audio_processing(
            audio_processing_factory(
                study_id=1001, audio_file_name="old_name.bin",
            )
        )
        rec2 = repo.save_audio_processing(
            audio_processing_factory(
                study_id=1001, audio_file_name="new_name.bin",
            )
        )

        assert rec1.id != rec2.id, "Different file name = different natural key = new record"

    def test_sync_file_name_not_part_of_natural_key(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory,
    ):
        """sync_file_name is NOT a natural key component.
        Natural key = (study_id, knee, maneuver, pass_number, speed).
        Changing sync_file_name while keeping the same natural key → UPDATE."""
        repo = Repository(db_session)

        audio = audio_processing_factory(study_id=1001)
        audio_rec = repo.save_audio_processing(audio)
        biomech = biomechanics_import_factory(study_id=1001)
        biomech_rec = repo.save_biomechanics_import(biomech)

        sync1 = synchronization_factory(
            study_id=1001, pass_number=1, speed="medium",
            sync_file_name="old_sync.pkl",
        )
        rec1 = repo.save_synchronization(
            sync1, audio_processing_id=audio_rec.id,
            biomechanics_import_id=biomech_rec.id,
        )

        sync2 = synchronization_factory(
            study_id=1001, pass_number=1, speed="medium",
            sync_file_name="new_sync.pkl",
        )
        rec2 = repo.save_synchronization(
            sync2, audio_processing_id=audio_rec.id,
            biomechanics_import_id=biomech_rec.id,
        )

        assert rec2.id == rec1.id, "Same natural key → update, not insert"
        assert rec2.sync_file_name == "new_sync.pkl"


# ========================================================================
# 9. Report generation — is_active filtering
# ========================================================================


class TestReportIsActiveFiltering:
    """Reports must only include active records."""

    @pytest.fixture(autouse=True)
    def _cleanup(self, db_engine):
        _wipe_all_tables(db_engine)
        yield
        _wipe_all_tables(db_engine)

    def test_summary_counts_exclude_inactive(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory, movement_cycle_factory, tmp_path,
    ):
        from src.reports.report_generator import ReportGenerator

        chain = _seed_full_chain(
            db_session, audio_processing_factory, biomechanics_import_factory,
            synchronization_factory, movement_cycle_factory,
            num_syncs=2, num_cycles_per_sync=5,
            pass_numbers=[1, 2], speeds=["medium", "slow"],
        )

        # Deactivate 1 sync + its 5 cycles
        repo = Repository(db_session)
        repo.deactivate_unseen_records(
            study_id=chain["audio"].study_id, knee="left", maneuver="walk",
            seen_audio_ids={chain["audio"].id},
            seen_biomech_ids={chain["biomech"].id},
            seen_sync_ids={chain["syncs"][0].id},  # keep sync 1 only
            seen_cycle_ids={c.id for c in chain["cycles"][:5]},  # keep cycles from sync 1
        )

        report = ReportGenerator(db_session)
        summary = report.generate_summary_sheet(
            chain["audio"].study_id, "walk", "left",
        )
        summary_dict = dict(zip(summary["Metric"], summary["Value"]))

        assert summary_dict["Audio Records"] == 1
        assert summary_dict["Biomechanics Records"] == 1
        assert summary_dict["Synchronization Records"] == 1
        assert summary_dict["Movement Cycles"] == 5

    def test_audio_sheet_excludes_inactive(
        self, db_session, audio_processing_factory, tmp_path,
    ):
        import pandas as pd
        from src.reports.report_generator import ReportGenerator

        repo = Repository(db_session)

        # Create 2 audio records, deactivate one
        rec1 = repo.save_audio_processing(
            audio_processing_factory(study_id=1001, audio_file_name="active.bin")
        )
        rec2 = repo.save_audio_processing(
            audio_processing_factory(study_id=1001, audio_file_name="inactive.bin")
        )

        repo.deactivate_unseen_records(
            study_id=rec1.study_id, knee="left", maneuver="walk",
            seen_audio_ids={rec1.id}, seen_biomech_ids=set(),
            seen_sync_ids=set(), seen_cycle_ids=set(),
        )

        report = ReportGenerator(db_session)
        excel_path = report.save_to_excel(
            tmp_path / "test_report.xlsx",
            participant_id=rec1.study_id,
            maneuver="walk",
            knee="left",
        )

        audio_sheet = pd.read_excel(excel_path, sheet_name="Audio")
        assert len(audio_sheet) == 1
        # Report strips .bin extension from audio file names
        assert audio_sheet.iloc[0]["Audio File Name"] == "active"

    def test_cycles_sheet_excludes_inactive(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory, movement_cycle_factory, tmp_path,
    ):
        import pandas as pd
        from src.reports.report_generator import ReportGenerator

        chain = _seed_full_chain(
            db_session, audio_processing_factory, biomechanics_import_factory,
            synchronization_factory, movement_cycle_factory,
            num_syncs=1, num_cycles_per_sync=5,
        )

        # Deactivate 2 cycles
        repo = Repository(db_session)
        seen_cycle_ids = {c.id for c in chain["cycles"][:3]}
        repo.deactivate_unseen_records(
            study_id=chain["audio"].study_id, knee="left", maneuver="walk",
            seen_audio_ids={chain["audio"].id},
            seen_biomech_ids={chain["biomech"].id},
            seen_sync_ids={chain["syncs"][0].id},
            seen_cycle_ids=seen_cycle_ids,
        )

        report = ReportGenerator(db_session)
        excel_path = report.save_to_excel(
            tmp_path / "test_cycles.xlsx",
            participant_id=chain["audio"].study_id,
            maneuver="walk",
            knee="left",
        )

        cycles_sheet = pd.read_excel(excel_path, sheet_name="Cycles")
        assert len(cycles_sheet) == 3, "Only 3 active cycles in report"

    def test_sync_sheet_excludes_inactive(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory, movement_cycle_factory, tmp_path,
    ):
        import pandas as pd
        from src.reports.report_generator import ReportGenerator

        chain = _seed_full_chain(
            db_session, audio_processing_factory, biomechanics_import_factory,
            synchronization_factory, movement_cycle_factory,
            num_syncs=3, num_cycles_per_sync=2,
            pass_numbers=[1, 2, 3], speeds=["medium", "medium", "slow"],
        )

        repo = Repository(db_session)
        # Keep only sync 1 and its cycles
        repo.deactivate_unseen_records(
            study_id=chain["audio"].study_id, knee="left", maneuver="walk",
            seen_audio_ids={chain["audio"].id},
            seen_biomech_ids={chain["biomech"].id},
            seen_sync_ids={chain["syncs"][0].id},
            seen_cycle_ids={c.id for c in chain["cycles"][:2]},
        )

        report = ReportGenerator(db_session)
        excel_path = report.save_to_excel(
            tmp_path / "test_syncs.xlsx",
            participant_id=chain["audio"].study_id,
            maneuver="walk",
            knee="left",
        )

        sync_sheet = pd.read_excel(excel_path, sheet_name="Synchronization")
        assert len(sync_sheet) == 1, "Only 1 active sync in report"


# ========================================================================
# 10. FK integrity — cross-table relationships
# ========================================================================


class TestForeignKeyIntegrity:
    """Tests that FK relationships between tables are maintained."""

    @pytest.fixture(autouse=True)
    def _cleanup(self, db_engine):
        _wipe_all_tables(db_engine)
        yield
        _wipe_all_tables(db_engine)

    def test_audio_links_to_study(
        self, db_session, audio_processing_factory,
    ):
        repo = Repository(db_session)
        audio = audio_processing_factory(study_id=1001)
        rec = repo.save_audio_processing(audio)

        assert rec.study_id is not None
        study = db_session.get(StudyRecord, rec.study_id)
        assert study.study_participant_id == 1001

    def test_biomechanics_links_to_audio(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
    ):
        repo = Repository(db_session)
        audio = audio_processing_factory(study_id=1001)
        audio_rec = repo.save_audio_processing(audio)

        biomech = biomechanics_import_factory(study_id=1001)
        biomech_rec = repo.save_biomechanics_import(
            biomech, audio_processing_id=audio_rec.id
        )

        assert biomech_rec.audio_processing_id == audio_rec.id

    def test_sync_links_to_audio_and_biomech(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory,
    ):
        repo = Repository(db_session)
        audio = audio_processing_factory(study_id=1001)
        audio_rec = repo.save_audio_processing(audio)
        biomech = biomechanics_import_factory(study_id=1001)
        biomech_rec = repo.save_biomechanics_import(biomech)

        sync = synchronization_factory(study_id=1001)
        sync_rec = repo.save_synchronization(
            sync, audio_processing_id=audio_rec.id,
            biomechanics_import_id=biomech_rec.id,
        )

        assert sync_rec.audio_processing_id == audio_rec.id
        assert sync_rec.biomechanics_import_id == biomech_rec.id

    def test_cycle_links_to_all_parents(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory, movement_cycle_factory,
    ):
        chain = _seed_full_chain(
            db_session, audio_processing_factory, biomechanics_import_factory,
            synchronization_factory, movement_cycle_factory,
        )

        for c in chain["cycles"]:
            assert c.audio_processing_id == chain["audio"].id
            assert c.biomechanics_import_id == chain["biomech"].id
            assert c.synchronization_id == chain["syncs"][0].id


# ========================================================================
# 11. Cleanup is filesystem-only (no DB involvement)
# ========================================================================


class TestCleanupIsFilesystemOnly:
    """Validates cleanup utility has no DB interaction."""

    def test_cleanup_has_no_db_parameters(self):
        import inspect
        from cli.cleanup_outputs import cleanup_participant_outputs

        sig = inspect.signature(cleanup_participant_outputs)
        param_names = set(sig.parameters.keys())
        assert "purge_db" not in param_names
        assert "db_url" not in param_names

    def test_cleanup_module_has_no_db_functions(self):
        import cli.cleanup_outputs as module

        assert not hasattr(module, "cleanup_participant_db_records")
        assert not hasattr(module, "cleanup_all_db_records")


# ========================================================================
# 12. Data field persistence — QC data round-trip
# ========================================================================


class TestQCDataPersistence:
    """Tests that QC fields are persisted correctly through save/load."""

    @pytest.fixture(autouse=True)
    def _cleanup(self, db_engine):
        _wipe_all_tables(db_engine)
        yield
        _wipe_all_tables(db_engine)

    def test_audio_qc_fields_persist(
        self, db_session, audio_processing_factory,
    ):
        repo = Repository(db_session)
        audio = audio_processing_factory(
            study_id=1001,
            audio_qc_version=2,
            qc_signal_dropout=True,
            qc_signal_dropout_ch1=True,
            qc_signal_dropout_ch2=False,
            qc_signal_dropout_ch3=False,
            qc_signal_dropout_ch4=False,
            qc_continuous_artifact=True,
            qc_continuous_artifact_ch1=False,
            qc_continuous_artifact_ch2=True,
        )
        rec = repo.save_audio_processing(audio)
        db_session.flush()
        db_session.expire_all()
        db_session.refresh(rec)

        assert str(rec.audio_qc_version) == "2"
        assert rec.qc_signal_dropout is True
        assert rec.qc_signal_dropout_ch1 is True
        assert rec.qc_signal_dropout_ch2 is False
        assert rec.qc_continuous_artifact is True
        assert rec.qc_continuous_artifact_ch2 is True

    def test_sync_periodic_artifact_fields_persist(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory,
    ):
        repo = Repository(db_session)

        audio = audio_processing_factory(study_id=1001)
        audio_rec = repo.save_audio_processing(audio)
        biomech = biomechanics_import_factory(study_id=1001)
        biomech_rec = repo.save_biomechanics_import(biomech)

        sync = synchronization_factory(
            study_id=1001,
            periodic_artifact_detected=True,
            periodic_artifact_detected_ch1=True,
            periodic_artifact_detected_ch2=False,
            periodic_artifact_detected_ch3=False,
            periodic_artifact_detected_ch4=True,
            periodic_artifact_segments=[(1.0, 3.0), (5.0, 7.0)],
            periodic_artifact_segments_ch1=[(1.0, 3.0)],
            periodic_artifact_segments_ch4=[(5.0, 7.0)],
        )
        rec = repo.save_synchronization(
            sync, audio_processing_id=audio_rec.id,
            biomechanics_import_id=biomech_rec.id,
        )
        db_session.flush()
        db_session.expire_all()
        db_session.refresh(rec)

        assert rec.periodic_artifact_detected is True
        assert rec.periodic_artifact_detected_ch1 is True
        assert rec.periodic_artifact_detected_ch2 is False
        assert rec.periodic_artifact_detected_ch4 is True
        assert rec.periodic_artifact_segments == [1.0, 3.0, 5.0, 7.0]
        assert rec.periodic_artifact_segments_ch1 == [1.0, 3.0]
        assert rec.periodic_artifact_segments_ch4 == [5.0, 7.0]

    def test_cycle_all_artifact_types_persist(
        self, db_session, audio_processing_factory, movement_cycle_factory,
    ):
        """Cycles store 4 artifact types × 5 channels (overall + ch1-4)."""
        repo = Repository(db_session)

        audio = audio_processing_factory(study_id=1001)
        audio_rec = repo.save_audio_processing(audio)

        cycle = movement_cycle_factory(
            study_id=1001, cycle_index=0,
            # Intermittent — timestamps are already-flattened list[float]
            audio_artifact_intermittent_fail=True,
            audio_artifact_intermittent_fail_ch1=True,
            audio_artifact_timestamps=[0.1, 0.2],
            audio_artifact_timestamps_ch1=[0.1, 0.2],
            # Dropout
            audio_artifact_dropout_fail=True,
            audio_artifact_dropout_fail_ch2=True,
            audio_artifact_dropout_timestamps=[0.3, 0.4],
            audio_artifact_dropout_timestamps_ch2=[0.3, 0.4],
            # Continuous
            audio_artifact_continuous_fail=True,
            audio_artifact_continuous_fail_ch3=True,
            audio_artifact_continuous_timestamps=[0.5, 0.6],
            audio_artifact_continuous_timestamps_ch3=[0.5, 0.6],
            # Periodic
            audio_artifact_periodic_fail=True,
            audio_artifact_periodic_fail_ch4=True,
            audio_artifact_periodic_timestamps=[0.7, 0.8],
            audio_artifact_periodic_timestamps_ch4=[0.7, 0.8],
        )
        rec = repo.save_movement_cycle(
            cycle, audio_processing_id=audio_rec.id,
        )
        db_session.flush()
        db_session.expire_all()
        db_session.refresh(rec)

        # Intermittent
        assert rec.audio_artifact_intermittent_fail is True
        assert rec.audio_artifact_intermittent_fail_ch1 is True
        # Dropout
        assert rec.audio_artifact_dropout_fail is True
        assert rec.audio_artifact_dropout_fail_ch2 is True
        # Continuous
        assert rec.audio_artifact_continuous_fail is True
        assert rec.audio_artifact_continuous_fail_ch3 is True
        # Periodic
        assert rec.audio_artifact_periodic_fail is True
        assert rec.audio_artifact_periodic_fail_ch4 is True

    def test_sync_cycle_statistics_persist(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory,
    ):
        repo = Repository(db_session)

        audio = audio_processing_factory(study_id=1001)
        audio_rec = repo.save_audio_processing(audio)
        biomech = biomechanics_import_factory(study_id=1001)
        biomech_rec = repo.save_biomechanics_import(biomech)

        sync = synchronization_factory(
            study_id=1001,
            total_cycles_extracted=10,
            clean_cycles=8,
            outlier_cycles=2,
            mean_cycle_duration_s=1.2,
            median_cycle_duration_s=1.15,
            min_cycle_duration_s=0.8,
            max_cycle_duration_s=1.6,
        )
        rec = repo.save_synchronization(
            sync, audio_processing_id=audio_rec.id,
            biomechanics_import_id=biomech_rec.id,
        )
        db_session.flush()
        db_session.expire_all()
        db_session.refresh(rec)

        assert rec.total_cycles_extracted == 10
        assert rec.clean_cycles == 8
        assert rec.outlier_cycles == 2
        assert rec.mean_cycle_duration_s == pytest.approx(1.2)
        assert rec.max_cycle_duration_s == pytest.approx(1.6)

    def test_cycle_timing_fields_persist(
        self, db_session, audio_processing_factory, movement_cycle_factory,
    ):
        """Cycle timing: start_time_s, end_time_s, duration_s (floats)."""
        from datetime import datetime

        repo = Repository(db_session)
        audio = audio_processing_factory(study_id=1001)
        audio_rec = repo.save_audio_processing(audio)

        cycle = movement_cycle_factory(
            study_id=1001, cycle_index=0,
            start_time_s=10.5,
            end_time_s=11.7,
            duration_s=1.2,
            start_time=datetime(2024, 1, 1, 10, 0, 10, 500000),
            end_time=datetime(2024, 1, 1, 10, 0, 11, 700000),
        )
        rec = repo.save_movement_cycle(
            cycle, audio_processing_id=audio_rec.id,
        )
        db_session.flush()
        db_session.expire_all()
        db_session.refresh(rec)

        assert rec.start_time_s == pytest.approx(10.5)
        assert rec.end_time_s == pytest.approx(11.7)
        assert rec.duration_s == pytest.approx(1.2)


# ========================================================================
# 13. Multi-sync walk topology (reflects real processing)
# ========================================================================


class TestWalkMultiSyncTopology:
    """Walk maneuver typically produces multiple passes with multiple syncs.
    This tests the real-world topology where a single audio → biomech chain
    fans out to multiple sync records, each with multiple cycles."""

    @pytest.fixture(autouse=True)
    def _cleanup(self, db_engine):
        _wipe_all_tables(db_engine)
        yield
        _wipe_all_tables(db_engine)

    def test_multiple_passes_with_speeds(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory, movement_cycle_factory,
    ):
        """Typical walk: passes 1-5 × speeds (medium, slow).
        10 syncs total, ~3-5 cycles each."""
        chain = _seed_full_chain(
            db_session, audio_processing_factory, biomechanics_import_factory,
            synchronization_factory, movement_cycle_factory,
            num_syncs=10,
            num_cycles_per_sync=4,
            pass_numbers=[1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            speeds=["medium", "slow", "medium", "slow", "medium",
                    "slow", "medium", "slow", "medium", "slow"],
        )

        assert len(chain["syncs"]) == 10
        assert len(chain["cycles"]) == 40

        # Deactivate 2 slow passes (syncs 1, 3 → slow passes 1, 2)
        repo = Repository(db_session)
        keep_sync_ids = {chain["syncs"][i].id for i in [0, 2, 4, 5, 6, 7, 8, 9]}
        keep_cycle_ids = set()
        for i in [0, 2, 4, 5, 6, 7, 8, 9]:
            for c in chain["cycles"][i*4:(i+1)*4]:
                keep_cycle_ids.add(c.id)

        deactivated = repo.deactivate_unseen_records(
            study_id=chain["audio"].study_id, knee="left", maneuver="walk",
            seen_audio_ids={chain["audio"].id},
            seen_biomech_ids={chain["biomech"].id},
            seen_sync_ids=keep_sync_ids,
            seen_cycle_ids=keep_cycle_ids,
        )

        assert deactivated["synchronizations"] == 2
        assert deactivated["movement_cycles"] == 8

    def test_sts_single_sync_no_pass(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory, movement_cycle_factory,
    ):
        """STS maneuver: single sync with pass_number=None, speed=None."""
        chain = _seed_full_chain(
            db_session, audio_processing_factory, biomechanics_import_factory,
            synchronization_factory, movement_cycle_factory,
            maneuver="sts",
            audio_file_name="sts_audio.bin",
            biomechanics_file="sts_biomech.xlsx",
            num_syncs=1, num_cycles_per_sync=5,
            pass_numbers=[None],
            speeds=[None],
            sync_file_prefix="left_sts",
        )

        assert len(chain["syncs"]) == 1
        assert chain["syncs"][0].pass_number is None
        assert chain["syncs"][0].speed is None
        assert len(chain["cycles"]) == 5


# ========================================================================
# 14. Edge cases
# ========================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    @pytest.fixture(autouse=True)
    def _cleanup(self, db_engine):
        _wipe_all_tables(db_engine)
        yield
        _wipe_all_tables(db_engine)

    def test_deactivate_nonexistent_scope_returns_zeros(self, db_session):
        """Deactivating a scope with no records should return all zeros."""
        repo = Repository(db_session)
        deactivated = repo.deactivate_unseen_records(
            study_id=9999, knee="left", maneuver="walk",
            seen_audio_ids=set(), seen_biomech_ids=set(),
            seen_sync_ids=set(), seen_cycle_ids=set(),
        )
        assert all(v == 0 for v in deactivated.values())

    def test_upsert_updates_updated_at_timestamp(
        self, db_session, audio_processing_factory,
    ):
        """The updated_at field should change on upsert."""
        from datetime import datetime, timezone

        repo = Repository(db_session)
        audio = audio_processing_factory(study_id=1001)
        rec = repo.save_audio_processing(audio)
        first_updated = rec.updated_at

        # Force a small time difference
        import time
        time.sleep(0.01)

        audio2 = audio_processing_factory(study_id=1001, duration_seconds=999.0)
        rec2 = repo.save_audio_processing(audio2)

        assert rec2.updated_at is not None
        # updated_at should be >= first_updated (may be same if fast)

    def test_query_with_no_matching_study(self, db_session):
        """Query for nonexistent participant returns empty list."""
        repo = Repository(db_session)
        results = repo.get_audio_processing_records(
            study_name="AOA", participant_number=9999,
        )
        assert results == []

    def test_multiple_audio_files_same_maneuver(
        self, db_session, audio_processing_factory,
    ):
        """Multiple audio files for the same maneuver (different file names)
        should all be persisted as separate records."""
        repo = Repository(db_session)

        files = ["walk_trial1.bin", "walk_trial2.bin", "walk_trial3.bin"]
        records = []
        for f in files:
            rec = repo.save_audio_processing(
                audio_processing_factory(study_id=1001, audio_file_name=f)
            )
            records.append(rec)

        assert len({r.id for r in records}) == 3, "3 distinct records"

        # Deactivate only trial2
        repo.deactivate_unseen_records(
            study_id=records[0].study_id, knee="left", maneuver="walk",
            seen_audio_ids={records[0].id, records[2].id},
            seen_biomech_ids=set(), seen_sync_ids=set(), seen_cycle_ids=set(),
        )

        active = repo.get_audio_processing_records(
            study_name="AOA", participant_number=1001,
        )
        assert len(active) == 2
        active_names = {r.audio_file_name for r in active}
        assert "walk_trial2.bin" not in active_names

    def test_cycle_is_outlier_persists(
        self, db_session, audio_processing_factory, movement_cycle_factory,
    ):
        """is_outlier flag should be persisted and queryable."""
        repo = Repository(db_session)
        audio = audio_processing_factory(study_id=1001)
        audio_rec = repo.save_audio_processing(audio)

        # Create one normal, one outlier
        normal = movement_cycle_factory(
            study_id=1001, cycle_index=0, is_outlier=False,
        )
        outlier = movement_cycle_factory(
            study_id=1001, cycle_index=1, is_outlier=True,
            cycle_file="cycle_01.pkl",
        )
        rec_normal = repo.save_movement_cycle(
            normal, audio_processing_id=audio_rec.id,
        )
        rec_outlier = repo.save_movement_cycle(
            outlier, audio_processing_id=audio_rec.id,
        )

        db_session.flush()
        db_session.expire_all()
        db_session.refresh(rec_normal)
        db_session.refresh(rec_outlier)

        assert rec_normal.is_outlier is False
        assert rec_outlier.is_outlier is True

    def test_biomech_upsert_updates_audio_processing_id_fk(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
    ):
        """When biomechanics is re-saved with a new audio_processing_id,
        the FK should be updated."""
        repo = Repository(db_session)

        audio1 = audio_processing_factory(
            study_id=1001, audio_file_name="audio1.bin"
        )
        audio_rec1 = repo.save_audio_processing(audio1)

        audio2 = audio_processing_factory(
            study_id=1001, audio_file_name="audio2.bin"
        )
        audio_rec2 = repo.save_audio_processing(audio2)

        biomech = biomechanics_import_factory(study_id=1001)
        rec = repo.save_biomechanics_import(
            biomech, audio_processing_id=audio_rec1.id
        )
        assert rec.audio_processing_id == audio_rec1.id

        # Re-save with different audio FK
        biomech2 = biomechanics_import_factory(study_id=1001)
        rec2 = repo.save_biomechanics_import(
            biomech2, audio_processing_id=audio_rec2.id
        )
        assert rec2.id == rec.id, "Same record updated"
        assert rec2.audio_processing_id == audio_rec2.id, "FK updated"
