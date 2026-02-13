"""Integration tests for audio-stage artifact timestamps flowing to cycle records.

Validates that dropout and continuous artifact segments detected at the audio
processing stage are correctly trimmed to cycle boundaries and persisted to:
1. MovementCycleRecord DB columns (bool fail flags + ARRAY(Float) timestamps)
2. Cycles Excel sheet columns via ReportGenerator
"""

from datetime import datetime

import pytest

from src.db.models import MovementCycleRecord
from src.db.repository import Repository
from src.reports.report_generator import ReportGenerator


class TestDropoutTimestampsDatabasePersistence:
    """Verify dropout artifact fail flags and timestamps persist through DB."""

    def _create_full_chain(
        self,
        db_session,
        audio_processing_factory,
        biomechanics_import_factory,
        synchronization_factory,
        movement_cycle_factory,
        **cycle_overrides,
    ):
        """Helper: create audio -> biomech -> sync -> cycle chain, return (cycle_record, audio_record)."""
        repo = Repository(db_session)

        audio_record = repo.save_audio_processing(audio_processing_factory())
        biomech_record = repo.save_biomechanics_import(biomechanics_import_factory())
        sync = synchronization_factory()
        sync_record = repo.save_synchronization(
            sync,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
        )

        defaults = dict(
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
            synchronization_id=sync_record.id,
        )
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

    def test_dropout_fail_flags_persist(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory, movement_cycle_factory,
    ):
        """Dropout fail booleans (aggregate + per-channel) should round-trip through DB."""
        cycle_record, _ = self._create_full_chain(
            db_session, audio_processing_factory, biomechanics_import_factory,
            synchronization_factory, movement_cycle_factory,
            audio_qc_fail=True,
            audio_qc_failures=["dropout"],
            audio_artifact_dropout_fail=True,
            audio_artifact_dropout_fail_ch1=True,
            audio_artifact_dropout_fail_ch2=False,
            audio_artifact_dropout_fail_ch3=True,
            audio_artifact_dropout_fail_ch4=False,
        )

        fetched = db_session.get(MovementCycleRecord, cycle_record.id)
        assert fetched.audio_artifact_dropout_fail is True
        assert fetched.audio_artifact_dropout_fail_ch1 is True
        assert fetched.audio_artifact_dropout_fail_ch2 is False
        assert fetched.audio_artifact_dropout_fail_ch3 is True
        assert fetched.audio_artifact_dropout_fail_ch4 is False

    def test_dropout_timestamps_persist(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory, movement_cycle_factory,
    ):
        """Dropout timestamp arrays (aggregate + per-channel) should round-trip."""
        cycle_record, _ = self._create_full_chain(
            db_session, audio_processing_factory, biomechanics_import_factory,
            synchronization_factory, movement_cycle_factory,
            audio_qc_fail=True,
            audio_qc_failures=["dropout"],
            audio_artifact_dropout_fail=True,
            audio_artifact_dropout_fail_ch1=True,
            audio_artifact_dropout_fail_ch3=True,
            # Flattened: [start1, end1, start2, end2]
            audio_artifact_dropout_timestamps=[0.1, 0.3, 0.5, 0.7],
            audio_artifact_dropout_timestamps_ch1=[0.1, 0.3],
            audio_artifact_dropout_timestamps_ch3=[0.5, 0.7],
        )

        fetched = db_session.get(MovementCycleRecord, cycle_record.id)
        assert fetched.audio_artifact_dropout_timestamps == [0.1, 0.3, 0.5, 0.7]
        assert fetched.audio_artifact_dropout_timestamps_ch1 == [0.1, 0.3]
        assert fetched.audio_artifact_dropout_timestamps_ch2 is None
        assert fetched.audio_artifact_dropout_timestamps_ch3 == [0.5, 0.7]
        assert fetched.audio_artifact_dropout_timestamps_ch4 is None

    def test_no_dropout_all_none(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory, movement_cycle_factory,
    ):
        """Clean cycle should have dropout fail=False and timestamps=None."""
        cycle_record, _ = self._create_full_chain(
            db_session, audio_processing_factory, biomechanics_import_factory,
            synchronization_factory, movement_cycle_factory,
            audio_artifact_dropout_fail=False,
            audio_artifact_dropout_fail_ch1=False,
            audio_artifact_dropout_fail_ch2=False,
            audio_artifact_dropout_fail_ch3=False,
            audio_artifact_dropout_fail_ch4=False,
            audio_artifact_dropout_timestamps=None,
            audio_artifact_dropout_timestamps_ch1=None,
            audio_artifact_dropout_timestamps_ch2=None,
            audio_artifact_dropout_timestamps_ch3=None,
            audio_artifact_dropout_timestamps_ch4=None,
        )

        fetched = db_session.get(MovementCycleRecord, cycle_record.id)
        assert fetched.audio_artifact_dropout_fail is False
        for ch in range(1, 5):
            assert getattr(fetched, f"audio_artifact_dropout_fail_ch{ch}") is False
            assert getattr(fetched, f"audio_artifact_dropout_timestamps_ch{ch}") is None
        assert fetched.audio_artifact_dropout_timestamps is None


class TestContinuousTimestampsDatabasePersistence:
    """Verify continuous artifact fail flags and timestamps persist through DB."""

    def _create_full_chain(
        self,
        db_session,
        audio_processing_factory,
        biomechanics_import_factory,
        synchronization_factory,
        movement_cycle_factory,
        **cycle_overrides,
    ):
        """Helper: create audio -> biomech -> sync -> cycle chain."""
        repo = Repository(db_session)

        audio_record = repo.save_audio_processing(audio_processing_factory())
        biomech_record = repo.save_biomechanics_import(biomechanics_import_factory())
        sync = synchronization_factory()
        sync_record = repo.save_synchronization(
            sync,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
        )

        defaults = dict(
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
            synchronization_id=sync_record.id,
        )
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

    def test_continuous_fail_flags_persist(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory, movement_cycle_factory,
    ):
        """Continuous fail booleans should round-trip through DB."""
        cycle_record, _ = self._create_full_chain(
            db_session, audio_processing_factory, biomechanics_import_factory,
            synchronization_factory, movement_cycle_factory,
            audio_qc_fail=True,
            audio_qc_failures=["continuous"],
            audio_artifact_continuous_fail=True,
            audio_artifact_continuous_fail_ch1=False,
            audio_artifact_continuous_fail_ch2=True,
            audio_artifact_continuous_fail_ch3=False,
            audio_artifact_continuous_fail_ch4=True,
        )

        fetched = db_session.get(MovementCycleRecord, cycle_record.id)
        assert fetched.audio_artifact_continuous_fail is True
        assert fetched.audio_artifact_continuous_fail_ch1 is False
        assert fetched.audio_artifact_continuous_fail_ch2 is True
        assert fetched.audio_artifact_continuous_fail_ch3 is False
        assert fetched.audio_artifact_continuous_fail_ch4 is True

    def test_continuous_timestamps_persist(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory, movement_cycle_factory,
    ):
        """Continuous timestamp arrays should round-trip."""
        cycle_record, _ = self._create_full_chain(
            db_session, audio_processing_factory, biomechanics_import_factory,
            synchronization_factory, movement_cycle_factory,
            audio_qc_fail=True,
            audio_qc_failures=["continuous"],
            audio_artifact_continuous_fail=True,
            audio_artifact_continuous_fail_ch2=True,
            audio_artifact_continuous_fail_ch4=True,
            audio_artifact_continuous_timestamps=[1.0, 2.5, 3.0, 4.0],
            audio_artifact_continuous_timestamps_ch2=[1.0, 2.5],
            audio_artifact_continuous_timestamps_ch4=[3.0, 4.0],
        )

        fetched = db_session.get(MovementCycleRecord, cycle_record.id)
        assert fetched.audio_artifact_continuous_timestamps == [1.0, 2.5, 3.0, 4.0]
        assert fetched.audio_artifact_continuous_timestamps_ch1 is None
        assert fetched.audio_artifact_continuous_timestamps_ch2 == [1.0, 2.5]
        assert fetched.audio_artifact_continuous_timestamps_ch3 is None
        assert fetched.audio_artifact_continuous_timestamps_ch4 == [3.0, 4.0]

    def test_no_continuous_all_none(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory, movement_cycle_factory,
    ):
        """Clean cycle should have continuous fail=False and timestamps=None."""
        cycle_record, _ = self._create_full_chain(
            db_session, audio_processing_factory, biomechanics_import_factory,
            synchronization_factory, movement_cycle_factory,
            audio_artifact_continuous_fail=False,
            audio_artifact_continuous_fail_ch1=False,
            audio_artifact_continuous_fail_ch2=False,
            audio_artifact_continuous_fail_ch3=False,
            audio_artifact_continuous_fail_ch4=False,
            audio_artifact_continuous_timestamps=None,
            audio_artifact_continuous_timestamps_ch1=None,
            audio_artifact_continuous_timestamps_ch2=None,
            audio_artifact_continuous_timestamps_ch3=None,
            audio_artifact_continuous_timestamps_ch4=None,
        )

        fetched = db_session.get(MovementCycleRecord, cycle_record.id)
        assert fetched.audio_artifact_continuous_fail is False
        for ch in range(1, 5):
            assert getattr(fetched, f"audio_artifact_continuous_fail_ch{ch}") is False
            assert getattr(fetched, f"audio_artifact_continuous_timestamps_ch{ch}") is None
        assert fetched.audio_artifact_continuous_timestamps is None


class TestCyclesExcelDropoutContinuousColumns:
    """Verify Cycles Excel sheet includes dropout/continuous QC columns."""

    def _create_full_chain(
        self,
        db_session,
        audio_processing_factory,
        biomechanics_import_factory,
        synchronization_factory,
        movement_cycle_factory,
        **cycle_overrides,
    ):
        """Helper: create audio -> biomech -> sync -> cycle chain."""
        repo = Repository(db_session)

        audio_record = repo.save_audio_processing(audio_processing_factory())
        biomech_record = repo.save_biomechanics_import(biomechanics_import_factory())
        sync = synchronization_factory()
        sync_record = repo.save_synchronization(
            sync,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
        )

        defaults = dict(
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
            synchronization_id=sync_record.id,
        )
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

    def test_cycles_sheet_has_dropout_and_continuous_columns(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory, movement_cycle_factory,
    ):
        """Cycles sheet should contain all dropout + continuous audio QC columns."""
        cycle_record, audio_record = self._create_full_chain(
            db_session, audio_processing_factory, biomechanics_import_factory,
            synchronization_factory, movement_cycle_factory,
        )

        generator = ReportGenerator(db_session)
        sheet = generator.generate_movement_cycles_sheet(
            study_id=audio_record.study_id,
            maneuver="walk",
            knee="left",
        )
        assert len(sheet) > 0

        expected_columns = [
            # Dropout
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
            # Continuous
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
        ]
        for col in expected_columns:
            assert col in sheet.columns, (
                f"Missing column '{col}' in Cycles sheet. "
                f"Available: {sorted(sheet.columns.tolist())}"
            )

    def test_cycles_sheet_dropout_values_populated(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory, movement_cycle_factory,
    ):
        """Cycles sheet should show dropout fail flags and timestamps."""
        cycle_record, audio_record = self._create_full_chain(
            db_session, audio_processing_factory, biomechanics_import_factory,
            synchronization_factory, movement_cycle_factory,
            audio_qc_fail=True,
            audio_qc_failures=["dropout"],
            audio_artifact_dropout_fail=True,
            audio_artifact_dropout_fail_ch1=True,
            audio_artifact_dropout_fail_ch3=True,
            audio_artifact_dropout_timestamps=[0.1, 0.3, 0.5, 0.7],
            audio_artifact_dropout_timestamps_ch1=[0.1, 0.3],
            audio_artifact_dropout_timestamps_ch3=[0.5, 0.7],
        )

        generator = ReportGenerator(db_session)
        sheet = generator.generate_movement_cycles_sheet(
            study_id=audio_record.study_id,
            maneuver="walk",
            knee="left",
        )

        row = sheet.iloc[0]
        assert bool(row["Audio Artifact Dropout Fail"]) is True
        assert bool(row["Audio Artifact Dropout Fail Ch1"]) is True
        assert bool(row["Audio Artifact Dropout Fail Ch2"]) is False
        assert bool(row["Audio Artifact Dropout Fail Ch3"]) is True
        assert bool(row["Audio Artifact Dropout Fail Ch4"]) is False
        # Timestamps should be formatted (non-empty)
        assert row["Audio Artifact Dropout Timestamps"] is not None
        assert row["Audio Artifact Dropout Timestamps Ch1"] is not None

    def test_cycles_sheet_continuous_values_populated(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory, movement_cycle_factory,
    ):
        """Cycles sheet should show continuous fail flags and timestamps."""
        cycle_record, audio_record = self._create_full_chain(
            db_session, audio_processing_factory, biomechanics_import_factory,
            synchronization_factory, movement_cycle_factory,
            audio_qc_fail=True,
            audio_qc_failures=["continuous"],
            audio_artifact_continuous_fail=True,
            audio_artifact_continuous_fail_ch2=True,
            audio_artifact_continuous_timestamps=[1.0, 2.5],
            audio_artifact_continuous_timestamps_ch2=[1.0, 2.5],
        )

        generator = ReportGenerator(db_session)
        sheet = generator.generate_movement_cycles_sheet(
            study_id=audio_record.study_id,
            maneuver="walk",
            knee="left",
        )

        row = sheet.iloc[0]
        assert bool(row["Audio Artifact Continuous Fail"]) is True
        assert bool(row["Audio Artifact Continuous Fail Ch1"]) is False
        assert bool(row["Audio Artifact Continuous Fail Ch2"]) is True
        assert row["Audio Artifact Continuous Timestamps"] is not None
        assert row["Audio Artifact Continuous Timestamps Ch2"] is not None

    def test_clean_cycle_dropout_continuous_false(
        self, db_session, audio_processing_factory, biomechanics_import_factory,
        synchronization_factory, movement_cycle_factory,
    ):
        """Clean cycle should show all dropout/continuous QC as False/empty."""
        cycle_record, audio_record = self._create_full_chain(
            db_session, audio_processing_factory, biomechanics_import_factory,
            synchronization_factory, movement_cycle_factory,
        )

        generator = ReportGenerator(db_session)
        sheet = generator.generate_movement_cycles_sheet(
            study_id=audio_record.study_id,
            maneuver="walk",
            knee="left",
        )

        row = sheet.iloc[0]
        assert bool(row["Audio Artifact Dropout Fail"]) is False
        assert bool(row["Audio Artifact Continuous Fail"]) is False
        for ch in range(1, 5):
            assert bool(row[f"Audio Artifact Dropout Fail Ch{ch}"]) is False
            assert bool(row[f"Audio Artifact Continuous Fail Ch{ch}"]) is False


class TestTrimIntervalsLogic:
    """Unit tests for trim_intervals_to_cycle used in cycle artifact population."""

    def test_interval_fully_within_cycle(self):
        """Interval within cycle should be returned unchanged."""
        from src.audio.raw_qc import trim_intervals_to_cycle

        result = trim_intervals_to_cycle([(1.0, 2.0)], 0.5, 3.0)
        assert result == [(1.0, 2.0)]

    def test_interval_partially_overlapping_start(self):
        """Interval starting before cycle should be trimmed to cycle start."""
        from src.audio.raw_qc import trim_intervals_to_cycle

        result = trim_intervals_to_cycle([(0.0, 2.0)], 1.0, 3.0)
        assert len(result) == 1
        assert result[0][0] == pytest.approx(1.0)
        assert result[0][1] == pytest.approx(2.0)

    def test_interval_partially_overlapping_end(self):
        """Interval ending after cycle should be trimmed to cycle end."""
        from src.audio.raw_qc import trim_intervals_to_cycle

        result = trim_intervals_to_cycle([(2.0, 5.0)], 1.0, 3.0)
        assert len(result) == 1
        assert result[0][0] == pytest.approx(2.0)
        assert result[0][1] == pytest.approx(3.0)

    def test_interval_no_overlap(self):
        """Interval outside cycle should be discarded entirely."""
        from src.audio.raw_qc import trim_intervals_to_cycle

        result = trim_intervals_to_cycle([(5.0, 7.0)], 1.0, 3.0)
        assert result == []

    def test_multiple_intervals_mixed_overlap(self):
        """Multiple intervals: some overlap, some not, some trimmed."""
        from src.audio.raw_qc import trim_intervals_to_cycle

        intervals = [
            (0.0, 0.5),   # before cycle — discarded
            (0.8, 1.5),   # overlaps start — trimmed
            (1.2, 1.8),   # fully within — unchanged
            (1.9, 3.5),   # overlaps end — trimmed
            (4.0, 5.0),   # after cycle — discarded
        ]
        result = trim_intervals_to_cycle(intervals, 1.0, 2.0)
        assert len(result) == 3
        # First: trimmed at start
        assert result[0][0] == pytest.approx(1.0)
        assert result[0][1] == pytest.approx(1.5)
        # Second: unchanged
        assert result[1][0] == pytest.approx(1.2)
        assert result[1][1] == pytest.approx(1.8)
        # Third: trimmed at end
        assert result[2][0] == pytest.approx(1.9)
        assert result[2][1] == pytest.approx(2.0)

    def test_empty_intervals(self):
        """Empty input should return empty output."""
        from src.audio.raw_qc import trim_intervals_to_cycle

        result = trim_intervals_to_cycle([], 1.0, 3.0)
        assert result == []
