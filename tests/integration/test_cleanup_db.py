"""Integration tests for is_active soft-delete and PK stability.

Validates that:
1. Re-processing updates existing records in-place (PKs preserved).
2. Records not present in a re-processing run are marked is_active=False.
3. Previously deactivated records are re-activated when they match again.
4. Reports only show active records.
5. Cleanup utility is file-system only (no DB involvement).
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


def _count_active(session) -> dict[str, int]:
    """Count active records in every downstream table."""
    return {
        "audio_processing": session.query(AudioProcessingRecord)
        .filter(AudioProcessingRecord.is_active == True)
        .count(),
        "biomechanics_imports": session.query(BiomechanicsImportRecord)
        .filter(BiomechanicsImportRecord.is_active == True)
        .count(),
        "synchronizations": session.query(SynchronizationRecord)
        .filter(SynchronizationRecord.is_active == True)
        .count(),
        "movement_cycles": session.query(MovementCycleRecord).filter(MovementCycleRecord.is_active == True).count(),
    }


def _count_all(session) -> dict[str, int]:
    """Count all records (active + inactive) in every table."""
    return {
        "participants": session.query(ParticipantRecord).count(),
        "studies": session.query(StudyRecord).count(),
        "audio_processing": session.query(AudioProcessingRecord).count(),
        "biomechanics_imports": session.query(BiomechanicsImportRecord).count(),
        "synchronizations": session.query(SynchronizationRecord).count(),
        "movement_cycles": session.query(MovementCycleRecord).count(),
    }


def _seed_participant(
    session,
    audio_processing_factory,
    biomechanics_import_factory,
    synchronization_factory,
    movement_cycle_factory,
    *,
    study_id: int = 1001,
    num_cycles: int = 2,
):
    """Create a full participant chain (audio -> biomech -> sync -> N cycles).

    Returns:
        Tuple of (audio_record, biomech_record, sync_record, [cycle_records]).
    """
    repo = Repository(session)

    audio = audio_processing_factory(study_id=study_id)
    audio_record = repo.save_audio_processing(audio)

    biomech = biomechanics_import_factory(study_id=study_id)
    biomech_record = repo.save_biomechanics_import(biomech)

    sync = synchronization_factory(study_id=study_id)
    sync_record = repo.save_synchronization(
        sync,
        audio_processing_id=audio_record.id,
        biomechanics_import_id=biomech_record.id,
    )

    cycle_records = []
    for idx in range(num_cycles):
        cycle = movement_cycle_factory(
            study_id=study_id,
            cycle_index=idx,
            cycle_file=f"cycle_{idx:02d}.pkl",
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
            synchronization_id=sync_record.id,
        )
        cycle_record = repo.save_movement_cycle(
            cycle,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
            synchronization_id=sync_record.id,
        )
        cycle_records.append(cycle_record)

    session.flush()
    return audio_record, biomech_record, sync_record, cycle_records


def _wipe_all_tables(engine):
    """Truncate all tables (used for test cleanup)."""
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


class TestIsActiveDeactivation:
    """Tests for is_active soft-delete via deactivate_unseen_records()."""

    @pytest.fixture(autouse=True)
    def _cleanup_tables(self, db_engine):
        """Ensure tables are clean before and after each test."""
        _wipe_all_tables(db_engine)
        yield
        _wipe_all_tables(db_engine)

    def test_new_records_are_active_by_default(
        self,
        db_session,
        audio_processing_factory,
        biomechanics_import_factory,
        synchronization_factory,
        movement_cycle_factory,
    ):
        """All records created via repository should have is_active=True."""
        audio_record, biomech_record, sync_record, cycle_records = _seed_participant(
            db_session,
            audio_processing_factory,
            biomechanics_import_factory,
            synchronization_factory,
            movement_cycle_factory,
        )

        assert audio_record.is_active is True
        assert biomech_record.is_active is True
        assert sync_record.is_active is True
        for cycle in cycle_records:
            assert cycle.is_active is True

    def test_deactivate_unseen_marks_cycles_inactive(
        self,
        db_session,
        audio_processing_factory,
        biomechanics_import_factory,
        synchronization_factory,
        movement_cycle_factory,
    ):
        """Cycles not in seen_ids should be marked inactive."""
        audio_record, biomech_record, sync_record, cycle_records = _seed_participant(
            db_session,
            audio_processing_factory,
            biomechanics_import_factory,
            synchronization_factory,
            movement_cycle_factory,
            num_cycles=5,
        )

        repo = Repository(db_session)

        # Simulate re-processing that only produces 3 cycles (0, 1, 2)
        seen_cycle_ids = {cycle_records[0].id, cycle_records[1].id, cycle_records[2].id}
        deactivated = repo.deactivate_unseen_records(
            study_id=audio_record.study_id,
            knee="left",
            maneuver="walk",
            seen_audio_ids={audio_record.id},
            seen_biomech_ids={biomech_record.id},
            seen_sync_ids={sync_record.id},
            seen_cycle_ids=seen_cycle_ids,
        )

        assert deactivated["movement_cycles"] == 2  # cycles 3, 4 deactivated

        # Verify directly
        db_session.expire_all()
        for i, cycle in enumerate(cycle_records):
            db_session.refresh(cycle)
            if i < 3:
                assert cycle.is_active is True, f"Cycle {i} should be active"
            else:
                assert cycle.is_active is False, f"Cycle {i} should be inactive"

    def test_deactivate_preserves_other_maneuvers(
        self,
        db_session,
        audio_processing_factory,
        biomechanics_import_factory,
        synchronization_factory,
        movement_cycle_factory,
    ):
        """Deactivation scoped to (study, knee, maneuver) — other maneuvers untouched."""
        repo = Repository(db_session)

        # Create walk records
        walk_audio = audio_processing_factory(study_id=1001, maneuver="walk")
        walk_audio_record = repo.save_audio_processing(walk_audio)

        # Create sts records
        sts_audio = audio_processing_factory(
            study_id=1001,
            maneuver="sts",
            audio_file_name="sts_audio.bin",
        )
        sts_audio_record = repo.save_audio_processing(sts_audio)

        db_session.flush()

        # Deactivate only walk records with empty seen_ids
        deactivated = repo.deactivate_unseen_records(
            study_id=walk_audio_record.study_id,
            knee="left",
            maneuver="walk",
            seen_audio_ids=set(),  # deactivate all walk audio
            seen_biomech_ids=set(),
            seen_sync_ids=set(),
            seen_cycle_ids=set(),
        )

        assert deactivated["audio_processing"] == 1

        db_session.expire_all()
        db_session.refresh(walk_audio_record)
        db_session.refresh(sts_audio_record)
        assert walk_audio_record.is_active is False
        assert sts_audio_record.is_active is True  # untouched

    def test_reactivation_on_re_processing(
        self,
        db_session,
        audio_processing_factory,
        biomechanics_import_factory,
        synchronization_factory,
        movement_cycle_factory,
    ):
        """A previously-deactivated record should be re-activated when upserted."""
        repo = Repository(db_session)

        # Create and deactivate
        audio = audio_processing_factory(study_id=1001)
        audio_record = repo.save_audio_processing(audio)
        original_pk = audio_record.id

        deactivated = repo.deactivate_unseen_records(
            study_id=audio_record.study_id,
            knee="left",
            maneuver="walk",
            seen_audio_ids=set(),
            seen_biomech_ids=set(),
            seen_sync_ids=set(),
            seen_cycle_ids=set(),
        )
        assert deactivated["audio_processing"] == 1

        db_session.expire_all()
        db_session.refresh(audio_record)
        assert audio_record.is_active is False

        # Re-process: upsert same record
        audio2 = audio_processing_factory(study_id=1001)
        reactivated_record = repo.save_audio_processing(audio2)

        assert reactivated_record.id == original_pk  # Same PK
        assert reactivated_record.is_active is True  # Re-activated


class TestPkStability:
    """Tests that PKs are preserved across re-processing runs."""

    @pytest.fixture(autouse=True)
    def _cleanup_tables(self, db_engine):
        _wipe_all_tables(db_engine)
        yield
        _wipe_all_tables(db_engine)

    def test_upsert_preserves_audio_pk(
        self,
        db_session,
        audio_processing_factory,
    ):
        """Re-processing the same audio file should update, not create a new row."""
        repo = Repository(db_session)

        audio = audio_processing_factory(study_id=1001)
        record1 = repo.save_audio_processing(audio)
        pk1 = record1.id

        # Re-process
        audio2 = audio_processing_factory(study_id=1001, duration_seconds=999.0)
        record2 = repo.save_audio_processing(audio2)

        assert record2.id == pk1
        assert record2.duration_seconds == 999.0

    def test_upsert_preserves_cycle_pks(
        self,
        db_session,
        audio_processing_factory,
        biomechanics_import_factory,
        synchronization_factory,
        movement_cycle_factory,
    ):
        """Re-processing should update existing cycle records with same PKs."""
        audio_record, biomech_record, sync_record, cycle_records = _seed_participant(
            db_session,
            audio_processing_factory,
            biomechanics_import_factory,
            synchronization_factory,
            movement_cycle_factory,
            num_cycles=5,
        )

        original_pks = [c.id for c in cycle_records]

        # Re-process with same 5 cycles (updated data)
        repo = Repository(db_session)
        for idx in range(5):
            cycle = movement_cycle_factory(
                study_id=1001,
                cycle_index=idx,
                cycle_file=f"cycle_{idx:02d}.pkl",
                audio_processing_id=audio_record.id,
                biomechanics_import_id=biomech_record.id,
                synchronization_id=sync_record.id,
                duration_s=99.0,  # updated value
            )
            updated = repo.save_movement_cycle(
                cycle,
                audio_processing_id=audio_record.id,
                biomechanics_import_id=biomech_record.id,
                synchronization_id=sync_record.id,
            )
            assert updated.id == original_pks[idx]
            assert updated.duration_s == 99.0


class TestIsActiveInReports:
    """Tests that reports only include active records."""

    @pytest.fixture(autouse=True)
    def _cleanup_tables(self, db_engine):
        _wipe_all_tables(db_engine)
        yield
        _wipe_all_tables(db_engine)

    def test_summary_counts_only_active_records(
        self,
        db_session,
        audio_processing_factory,
        biomechanics_import_factory,
        synchronization_factory,
        movement_cycle_factory,
        tmp_path,
    ):
        """Summary sheet counts should exclude inactive records."""
        from src.reports.report_generator import ReportGenerator

        audio_record, biomech_record, sync_record, cycle_records = _seed_participant(
            db_session,
            audio_processing_factory,
            biomechanics_import_factory,
            synchronization_factory,
            movement_cycle_factory,
            num_cycles=5,
        )

        # Deactivate 2 cycles
        repo = Repository(db_session)
        seen_cycle_ids = {c.id for c in cycle_records[:3]}
        repo.deactivate_unseen_records(
            study_id=audio_record.study_id,
            knee="left",
            maneuver="walk",
            seen_audio_ids={audio_record.id},
            seen_biomech_ids={biomech_record.id},
            seen_sync_ids={sync_record.id},
            seen_cycle_ids=seen_cycle_ids,
        )

        report = ReportGenerator(db_session)
        summary = report.generate_summary_sheet(
            audio_record.study_id,
            "walk",
            "left",
        )

        summary_dict = dict(zip(summary["Metric"], summary["Value"]))
        assert summary_dict["Movement Cycles"] == 3  # 5 total - 2 inactive

    def test_inactive_records_hidden_from_excel(
        self,
        db_session,
        audio_processing_factory,
        biomechanics_import_factory,
        synchronization_factory,
        movement_cycle_factory,
        tmp_path,
    ):
        """save_to_excel should not include inactive audio records."""
        import pandas as pd

        from src.reports.report_generator import ReportGenerator

        repo = Repository(db_session)

        # Create 2 audio records
        audio1 = audio_processing_factory(study_id=1001, audio_file_name="file_a.bin")
        rec1 = repo.save_audio_processing(audio1)

        audio2 = audio_processing_factory(
            study_id=1001,
            audio_file_name="file_b.bin",
        )
        repo.save_audio_processing(audio2)

        # Deactivate file_b
        repo.deactivate_unseen_records(
            study_id=rec1.study_id,
            knee="left",
            maneuver="walk",
            seen_audio_ids={rec1.id},
            seen_biomech_ids=set(),
            seen_sync_ids=set(),
            seen_cycle_ids=set(),
        )

        report = ReportGenerator(db_session)
        output_path = report.save_to_excel(
            tmp_path / "test_active_filter.xlsx",
            participant_id=rec1.study_id,
            maneuver="walk",
            knee="left",
        )

        audio_sheet = pd.read_excel(output_path, sheet_name="Audio")
        assert len(audio_sheet) == 1
        assert audio_sheet.iloc[0]["Audio File Name"] == "file_a"  # .bin stripped by report


class TestQueryIncludeInactive:
    """Tests for include_inactive parameter on repository query methods."""

    @pytest.fixture(autouse=True)
    def _cleanup_tables(self, db_engine):
        _wipe_all_tables(db_engine)
        yield
        _wipe_all_tables(db_engine)

    def test_get_audio_records_excludes_inactive_by_default(
        self,
        db_session,
        audio_processing_factory,
    ):
        """Default query should only return active records."""
        repo = Repository(db_session)

        audio = audio_processing_factory(study_id=1001)
        record = repo.save_audio_processing(audio)

        # Deactivate
        repo.deactivate_unseen_records(
            study_id=record.study_id,
            knee="left",
            maneuver="walk",
            seen_audio_ids=set(),
            seen_biomech_ids=set(),
            seen_sync_ids=set(),
            seen_cycle_ids=set(),
        )

        # Default: active only
        results = repo.get_audio_processing_records(
            study_name="AOA",
            participant_number=1001,
        )
        assert len(results) == 0

        # With include_inactive
        results_all = repo.get_audio_processing_records(
            study_name="AOA",
            participant_number=1001,
            include_inactive=True,
        )
        assert len(results_all) == 1
        assert results_all[0].is_active is False


class TestCleanupFilesystemOnly:
    """Tests that cleanup utility is file-system only."""

    def test_cleanup_has_no_db_parameters(self):
        """cleanup_participant_outputs should not accept purge_db or db_url."""
        import inspect

        from cli.cleanup_outputs import cleanup_participant_outputs

        sig = inspect.signature(cleanup_participant_outputs)
        param_names = set(sig.parameters.keys())
        assert "purge_db" not in param_names
        assert "db_url" not in param_names

    def test_cleanup_module_has_no_db_functions(self):
        """cleanup_outputs should not export cleanup_participant_db_records."""
        import cli.cleanup_outputs as module

        assert not hasattr(module, "cleanup_participant_db_records")
        assert not hasattr(module, "cleanup_all_db_records")
