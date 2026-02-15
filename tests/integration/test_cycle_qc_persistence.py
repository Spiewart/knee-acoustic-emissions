"""Integration tests for Cycle QC field persistence through DB and Excel.

Validates that:
1. MovementCycle audio QC fields (aggregate + per-source + per-channel)
   persist correctly to MovementCycleRecord in the database.
2. Cycles Excel sheet columns include all audio QC fields with correct values.
"""

from src.db.models import MovementCycleRecord
from src.db.repository import Repository
from src.reports.report_generator import ReportGenerator


class TestCycleAudioQCDatabasePersistence:
    """Verify cycle-level audio QC fields persist through DB layer."""

    def test_aggregate_audio_qc_fail_persists_true(
        self,
        db_session,
        audio_processing_factory,
        biomechanics_import_factory,
        synchronization_factory,
        movement_cycle_factory,
    ):
        """audio_qc_fail=True and audio_qc_failures list should persist."""
        repo = Repository(db_session)

        audio_record = repo.save_audio_processing(audio_processing_factory())
        biomech_record = repo.save_biomechanics_import(biomechanics_import_factory())
        sync = synchronization_factory()
        sync_record = repo.save_synchronization(
            sync,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
        )

        cycle = movement_cycle_factory(
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
            synchronization_id=sync_record.id,
            audio_qc_fail=True,
            audio_qc_failures=["dropout", "intermittent"],
        )
        cycle_record = repo.save_movement_cycle(
            cycle,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
            synchronization_id=sync_record.id,
        )
        db_session.flush()

        fetched = db_session.get(MovementCycleRecord, cycle_record.id)
        assert fetched.audio_qc_fail is True
        assert fetched.audio_qc_failures is not None
        assert "dropout" in fetched.audio_qc_failures
        assert "intermittent" in fetched.audio_qc_failures

    def test_aggregate_audio_qc_fail_persists_false(
        self,
        db_session,
        audio_processing_factory,
        biomechanics_import_factory,
        synchronization_factory,
        movement_cycle_factory,
    ):
        """Clean cycle should have audio_qc_fail=False and no failures."""
        repo = Repository(db_session)

        audio_record = repo.save_audio_processing(audio_processing_factory())
        biomech_record = repo.save_biomechanics_import(biomechanics_import_factory())
        sync = synchronization_factory()
        sync_record = repo.save_synchronization(
            sync,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
        )

        cycle = movement_cycle_factory(
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
            synchronization_id=sync_record.id,
            audio_qc_fail=False,
            audio_qc_failures=None,
        )
        cycle_record = repo.save_movement_cycle(
            cycle,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
            synchronization_id=sync_record.id,
        )
        db_session.flush()

        fetched = db_session.get(MovementCycleRecord, cycle_record.id)
        assert fetched.audio_qc_fail is False
        assert fetched.audio_qc_failures is None

    def test_intermittent_fail_flags_persist(
        self,
        db_session,
        audio_processing_factory,
        biomechanics_import_factory,
        synchronization_factory,
        movement_cycle_factory,
    ):
        """Per-channel intermittent fail flags and timestamps should persist."""
        repo = Repository(db_session)

        audio_record = repo.save_audio_processing(audio_processing_factory())
        biomech_record = repo.save_biomechanics_import(biomechanics_import_factory())
        sync = synchronization_factory()
        sync_record = repo.save_synchronization(
            sync,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
        )

        cycle = movement_cycle_factory(
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
            synchronization_id=sync_record.id,
            audio_qc_fail=True,
            audio_qc_failures=["intermittent"],
            audio_artifact_intermittent_fail=True,
            audio_artifact_intermittent_fail_ch1=False,
            audio_artifact_intermittent_fail_ch2=True,
            audio_artifact_intermittent_fail_ch3=False,
            audio_artifact_intermittent_fail_ch4=True,
            audio_artifact_timestamps=[0.5, 0.8, 0.3, 0.6],
            audio_artifact_timestamps_ch2=[0.5, 0.8],
            audio_artifact_timestamps_ch4=[0.3, 0.6],
        )
        cycle_record = repo.save_movement_cycle(
            cycle,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
            synchronization_id=sync_record.id,
        )
        db_session.flush()

        fetched = db_session.get(MovementCycleRecord, cycle_record.id)
        assert fetched.audio_artifact_intermittent_fail is True
        assert fetched.audio_artifact_intermittent_fail_ch1 is False
        assert fetched.audio_artifact_intermittent_fail_ch2 is True
        assert fetched.audio_artifact_intermittent_fail_ch3 is False
        assert fetched.audio_artifact_intermittent_fail_ch4 is True
        assert fetched.audio_artifact_timestamps is not None
        assert len(fetched.audio_artifact_timestamps) == 4
        assert fetched.audio_artifact_timestamps_ch2 == [0.5, 0.8]
        assert fetched.audio_artifact_timestamps_ch4 == [0.3, 0.6]

    def test_periodic_fail_flags_and_timestamps_persist(
        self,
        db_session,
        audio_processing_factory,
        biomechanics_import_factory,
        synchronization_factory,
        movement_cycle_factory,
    ):
        """Per-channel periodic fail flags and timestamps should persist."""
        repo = Repository(db_session)

        audio_record = repo.save_audio_processing(audio_processing_factory())
        biomech_record = repo.save_biomechanics_import(biomechanics_import_factory())
        sync = synchronization_factory()
        sync_record = repo.save_synchronization(
            sync,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
        )

        cycle = movement_cycle_factory(
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
            synchronization_id=sync_record.id,
            audio_qc_fail=True,
            audio_qc_failures=["periodic"],
            audio_artifact_periodic_fail=True,
            audio_artifact_periodic_fail_ch1=True,
            audio_artifact_periodic_fail_ch2=False,
            audio_artifact_periodic_fail_ch3=True,
            audio_artifact_periodic_fail_ch4=False,
            audio_artifact_periodic_timestamps=[0.0, 1.2, 0.2, 0.9],
            audio_artifact_periodic_timestamps_ch1=[0.0, 1.2],
            audio_artifact_periodic_timestamps_ch3=[0.2, 0.9],
        )
        cycle_record = repo.save_movement_cycle(
            cycle,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
            synchronization_id=sync_record.id,
        )
        db_session.flush()

        fetched = db_session.get(MovementCycleRecord, cycle_record.id)
        assert fetched.audio_artifact_periodic_fail is True
        assert fetched.audio_artifact_periodic_fail_ch1 is True
        assert fetched.audio_artifact_periodic_fail_ch2 is False
        assert fetched.audio_artifact_periodic_fail_ch3 is True
        assert fetched.audio_artifact_periodic_fail_ch4 is False
        assert fetched.audio_artifact_periodic_timestamps is not None
        assert len(fetched.audio_artifact_periodic_timestamps) == 4
        assert fetched.audio_artifact_periodic_timestamps_ch1 == [0.0, 1.2]
        assert fetched.audio_artifact_periodic_timestamps_ch3 == [0.2, 0.9]

    def test_all_four_failure_types_persist(
        self,
        db_session,
        audio_processing_factory,
        biomechanics_import_factory,
        synchronization_factory,
        movement_cycle_factory,
    ):
        """Cycle with all four audio QC failure types should persist all fields."""
        repo = Repository(db_session)

        audio_record = repo.save_audio_processing(audio_processing_factory())
        biomech_record = repo.save_biomechanics_import(biomechanics_import_factory())
        sync = synchronization_factory()
        sync_record = repo.save_synchronization(
            sync,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
        )

        cycle = movement_cycle_factory(
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
            synchronization_id=sync_record.id,
            audio_qc_fail=True,
            audio_qc_failures=["dropout", "continuous", "intermittent", "periodic"],
            audio_artifact_intermittent_fail=True,
            audio_artifact_intermittent_fail_ch2=True,
            audio_artifact_timestamps=[0.5, 0.8],
            audio_artifact_timestamps_ch2=[0.5, 0.8],
            audio_artifact_periodic_fail=True,
            audio_artifact_periodic_fail_ch1=True,
            audio_artifact_periodic_timestamps=[0.0, 1.2],
            audio_artifact_periodic_timestamps_ch1=[0.0, 1.2],
        )
        cycle_record = repo.save_movement_cycle(
            cycle,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
            synchronization_id=sync_record.id,
        )
        db_session.flush()

        fetched = db_session.get(MovementCycleRecord, cycle_record.id)
        assert fetched.audio_qc_fail is True
        assert set(fetched.audio_qc_failures) == {"dropout", "continuous", "intermittent", "periodic"}
        assert fetched.audio_artifact_intermittent_fail is True
        assert fetched.audio_artifact_periodic_fail is True


class TestCyclesExcelSheetQCColumns:
    """Verify Cycles Excel sheet includes all audio QC columns with correct values."""

    def _create_cycle_with_qc(
        self,
        db_session,
        audio_processing_factory,
        biomechanics_import_factory,
        synchronization_factory,
        movement_cycle_factory,
        **cycle_overrides,
    ):
        """Helper to create a full DB chain and return (cycle_record, audio_record)."""
        repo = Repository(db_session)

        audio_record = repo.save_audio_processing(audio_processing_factory())
        biomech_record = repo.save_biomechanics_import(biomechanics_import_factory())
        sync = synchronization_factory()
        sync_record = repo.save_synchronization(
            sync,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
        )

        defaults = {
            "audio_processing_id": audio_record.id,
            "biomechanics_import_id": biomech_record.id,
            "synchronization_id": sync_record.id,
        }
        defaults.update(cycle_overrides)
        cycle = movement_cycle_factory(**defaults)
        cycle_record = repo.save_movement_cycle(
            cycle,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
            synchronization_id=sync_record.id,
        )
        db_session.flush()
        return cycle_record, audio_record

    def test_cycles_sheet_has_all_audio_qc_columns(
        self,
        db_session,
        audio_processing_factory,
        biomechanics_import_factory,
        synchronization_factory,
        movement_cycle_factory,
    ):
        """Cycles sheet should contain aggregate + per-source audio QC columns."""
        cycle_record, audio_record = self._create_cycle_with_qc(
            db_session,
            audio_processing_factory,
            biomechanics_import_factory,
            synchronization_factory,
            movement_cycle_factory,
        )

        generator = ReportGenerator(db_session)
        sheet = generator.generate_movement_cycles_sheet(
            study_id=audio_record.study_id,
            maneuver="walk",
            knee="left",
        )

        assert len(sheet) > 0

        expected_columns = [
            # Aggregate
            "Audio QC Fail",
            "Audio QC Failures",
            # Intermittent
            "Audio Artifact Intermittent Fail",
            "Audio Artifact Intermittent Fail Ch1",
            "Audio Artifact Intermittent Fail Ch2",
            "Audio Artifact Intermittent Fail Ch3",
            "Audio Artifact Intermittent Fail Ch4",
            "Audio Artifact Timestamps",
            "Audio Artifact Timestamps Ch1",
            "Audio Artifact Timestamps Ch2",
            "Audio Artifact Timestamps Ch3",
            "Audio Artifact Timestamps Ch4",
            # Dropout (audio-stage, trimmed to cycle)
            "Audio Artifact Dropout Fail",
            "Audio Artifact Dropout Fail Ch1",
            "Audio Artifact Dropout Fail Ch2",
            "Audio Artifact Dropout Fail Ch3",
            "Audio Artifact Dropout Fail Ch4",
            "Audio Artifact Dropout Timestamps",
            "Audio Artifact Dropout Timestamps Ch1",
            "Audio Artifact Dropout Timestamps Ch2",
            "Audio Artifact Dropout Timestamps Ch3",
            "Audio Artifact Dropout Timestamps Ch4",
            # Continuous (audio-stage, trimmed to cycle)
            "Audio Artifact Continuous Fail",
            "Audio Artifact Continuous Fail Ch1",
            "Audio Artifact Continuous Fail Ch2",
            "Audio Artifact Continuous Fail Ch3",
            "Audio Artifact Continuous Fail Ch4",
            "Audio Artifact Continuous Timestamps",
            "Audio Artifact Continuous Timestamps Ch1",
            "Audio Artifact Continuous Timestamps Ch2",
            "Audio Artifact Continuous Timestamps Ch3",
            "Audio Artifact Continuous Timestamps Ch4",
            # Periodic
            "Audio Artifact Periodic Fail",
            "Audio Artifact Periodic Fail Ch1",
            "Audio Artifact Periodic Fail Ch2",
            "Audio Artifact Periodic Fail Ch3",
            "Audio Artifact Periodic Fail Ch4",
            "Audio Artifact Periodic Timestamps",
            "Audio Artifact Periodic Timestamps Ch1",
            "Audio Artifact Periodic Timestamps Ch2",
            "Audio Artifact Periodic Timestamps Ch3",
            "Audio Artifact Periodic Timestamps Ch4",
        ]
        for col in expected_columns:
            assert col in sheet.columns, (
                f"Missing column '{col}' in Cycles sheet. Available: {sorted(sheet.columns.tolist())}"
            )

    def test_cycles_sheet_aggregate_values_populated(
        self,
        db_session,
        audio_processing_factory,
        biomechanics_import_factory,
        synchronization_factory,
        movement_cycle_factory,
    ):
        """Cycles sheet should show audio_qc_fail=True with failure list."""
        cycle_record, audio_record = self._create_cycle_with_qc(
            db_session,
            audio_processing_factory,
            biomechanics_import_factory,
            synchronization_factory,
            movement_cycle_factory,
            audio_qc_fail=True,
            audio_qc_failures=["dropout", "periodic"],
        )

        generator = ReportGenerator(db_session)
        sheet = generator.generate_movement_cycles_sheet(
            study_id=audio_record.study_id,
            maneuver="walk",
            knee="left",
        )

        row = sheet.iloc[0]
        assert bool(row["Audio QC Fail"]) is True
        # audio_qc_failures is formatted for Excel by _format_list_for_excel
        failures_str = row["Audio QC Failures"]
        assert failures_str is not None
        assert "dropout" in str(failures_str)
        assert "periodic" in str(failures_str)

    def test_cycles_sheet_intermittent_values_populated(
        self,
        db_session,
        audio_processing_factory,
        biomechanics_import_factory,
        synchronization_factory,
        movement_cycle_factory,
    ):
        """Cycles sheet should show intermittent fail flags and timestamps."""
        cycle_record, audio_record = self._create_cycle_with_qc(
            db_session,
            audio_processing_factory,
            biomechanics_import_factory,
            synchronization_factory,
            movement_cycle_factory,
            audio_qc_fail=True,
            audio_qc_failures=["intermittent"],
            audio_artifact_intermittent_fail=True,
            audio_artifact_intermittent_fail_ch2=True,
            audio_artifact_timestamps=[0.5, 0.8],
            audio_artifact_timestamps_ch2=[0.5, 0.8],
        )

        generator = ReportGenerator(db_session)
        sheet = generator.generate_movement_cycles_sheet(
            study_id=audio_record.study_id,
            maneuver="walk",
            knee="left",
        )

        row = sheet.iloc[0]
        assert bool(row["Audio Artifact Intermittent Fail"]) is True
        assert bool(row["Audio Artifact Intermittent Fail Ch2"]) is True
        assert bool(row["Audio Artifact Intermittent Fail Ch1"]) is False
        # Timestamps should be formatted
        timestamps = row["Audio Artifact Timestamps"]
        assert timestamps is not None
        timestamps_ch2 = row["Audio Artifact Timestamps Ch2"]
        assert timestamps_ch2 is not None

    def test_cycles_sheet_periodic_values_populated(
        self,
        db_session,
        audio_processing_factory,
        biomechanics_import_factory,
        synchronization_factory,
        movement_cycle_factory,
    ):
        """Cycles sheet should show periodic fail flags and timestamps."""
        cycle_record, audio_record = self._create_cycle_with_qc(
            db_session,
            audio_processing_factory,
            biomechanics_import_factory,
            synchronization_factory,
            movement_cycle_factory,
            audio_qc_fail=True,
            audio_qc_failures=["periodic"],
            audio_artifact_periodic_fail=True,
            audio_artifact_periodic_fail_ch1=True,
            audio_artifact_periodic_fail_ch3=True,
            audio_artifact_periodic_timestamps=[0.0, 1.2, 0.2, 0.9],
            audio_artifact_periodic_timestamps_ch1=[0.0, 1.2],
            audio_artifact_periodic_timestamps_ch3=[0.2, 0.9],
        )

        generator = ReportGenerator(db_session)
        sheet = generator.generate_movement_cycles_sheet(
            study_id=audio_record.study_id,
            maneuver="walk",
            knee="left",
        )

        row = sheet.iloc[0]
        assert bool(row["Audio Artifact Periodic Fail"]) is True
        assert bool(row["Audio Artifact Periodic Fail Ch1"]) is True
        assert bool(row["Audio Artifact Periodic Fail Ch2"]) is False
        assert bool(row["Audio Artifact Periodic Fail Ch3"]) is True
        assert bool(row["Audio Artifact Periodic Fail Ch4"]) is False
        # Timestamps should be populated
        timestamps_ch1 = row["Audio Artifact Periodic Timestamps Ch1"]
        assert timestamps_ch1 is not None
        timestamps_ch3 = row["Audio Artifact Periodic Timestamps Ch3"]
        assert timestamps_ch3 is not None

    def test_clean_cycle_all_qc_false(
        self,
        db_session,
        audio_processing_factory,
        biomechanics_import_factory,
        synchronization_factory,
        movement_cycle_factory,
    ):
        """Clean cycle should show all audio QC fields as False/empty in Excel."""
        cycle_record, audio_record = self._create_cycle_with_qc(
            db_session,
            audio_processing_factory,
            biomechanics_import_factory,
            synchronization_factory,
            movement_cycle_factory,
            audio_qc_fail=False,
            audio_qc_failures=None,
        )

        generator = ReportGenerator(db_session)
        sheet = generator.generate_movement_cycles_sheet(
            study_id=audio_record.study_id,
            maneuver="walk",
            knee="left",
        )

        row = sheet.iloc[0]
        assert bool(row["Audio QC Fail"]) is False
        assert bool(row["Audio Artifact Intermittent Fail"]) is False
        assert bool(row["Audio Artifact Dropout Fail"]) is False
        assert bool(row["Audio Artifact Continuous Fail"]) is False
        assert bool(row["Audio Artifact Periodic Fail"]) is False
        for ch in range(1, 5):
            assert bool(row[f"Audio Artifact Intermittent Fail Ch{ch}"]) is False
            assert bool(row[f"Audio Artifact Dropout Fail Ch{ch}"]) is False
            assert bool(row[f"Audio Artifact Continuous Fail Ch{ch}"]) is False
            assert bool(row[f"Audio Artifact Periodic Fail Ch{ch}"]) is False
