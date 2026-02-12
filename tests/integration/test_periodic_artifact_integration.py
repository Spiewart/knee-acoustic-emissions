"""Integration tests for periodic artifact detection columns.

Validates that periodic artifact fields propagate correctly through:
1. Synchronization records (sync-level periodic detection)
2. MovementCycle records (cycle-level periodic propagation)
3. Excel report generation (both Synchronization and Cycles sheets)
4. Database persistence (new columns exist and accept data)
"""

from datetime import datetime

import pandas as pd
import pytest
from sqlalchemy import select

from src.db.models import (
    MovementCycleRecord,
    SynchronizationRecord,
)
from src.db.repository import Repository
from src.metadata import MovementCycle, Synchronization


class TestPeriodicArtifactDatabaseColumns:
    """Verify periodic artifact columns exist and accept real values in the DB."""

    def test_sync_periodic_columns_persist_true_values(
        self, db_session, synchronization_factory, audio_processing_factory,
        biomechanics_import_factory,
    ):
        """Sync record should store periodic_artifact_detected=True with segments."""
        repo = Repository(db_session)

        audio_record = repo.save_audio_processing(audio_processing_factory())
        biomech_record = repo.save_biomechanics_import(biomechanics_import_factory())

        sync = synchronization_factory(
            periodic_artifact_detected=True,
            periodic_artifact_detected_ch1=True,
            periodic_artifact_detected_ch2=False,
            periodic_artifact_detected_ch3=True,
            periodic_artifact_detected_ch4=False,
            periodic_artifact_segments=[(1.0, 3.5), (7.2, 9.0)],
            periodic_artifact_segments_ch1=[(1.0, 3.5)],
            periodic_artifact_segments_ch3=[(7.2, 9.0)],
        )
        sync_record = repo.save_synchronization(
            sync,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
        )

        db_session.flush()

        record = db_session.get(SynchronizationRecord, sync_record.id)
        assert record.periodic_artifact_detected is True
        assert record.periodic_artifact_detected_ch1 is True
        assert record.periodic_artifact_detected_ch2 is False
        assert record.periodic_artifact_detected_ch3 is True
        assert record.periodic_artifact_detected_ch4 is False
        assert record.periodic_artifact_segments is not None
        assert len(record.periodic_artifact_segments) == 4  # flattened [1.0, 3.5, 7.2, 9.0]
        assert record.periodic_artifact_segments_ch1 is not None
        assert len(record.periodic_artifact_segments_ch1) == 2  # [1.0, 3.5]

    def test_sync_periodic_columns_default_false(
        self, db_session, synchronization_factory, audio_processing_factory,
        biomechanics_import_factory,
    ):
        """Sync record should default periodic fields to False/None."""
        repo = Repository(db_session)

        audio_record = repo.save_audio_processing(audio_processing_factory())
        biomech_record = repo.save_biomechanics_import(biomechanics_import_factory())

        sync = synchronization_factory()  # No periodic overrides
        sync_record = repo.save_synchronization(
            sync,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
        )

        db_session.flush()

        record = db_session.get(SynchronizationRecord, sync_record.id)
        assert record.periodic_artifact_detected is False
        assert record.periodic_artifact_detected_ch1 is False
        assert record.periodic_artifact_segments is None

    def test_cycle_periodic_columns_persist_true_values(
        self, db_session, movement_cycle_factory, audio_processing_factory,
        synchronization_factory, biomechanics_import_factory,
    ):
        """Cycle record should store periodic fail flags and timestamps."""
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
            audio_artifact_periodic_fail=True,
            audio_artifact_periodic_fail_ch1=True,
            audio_artifact_periodic_fail_ch2=False,
            audio_artifact_periodic_fail_ch3=True,
            audio_artifact_periodic_fail_ch4=False,
            audio_artifact_periodic_timestamps=[2.0, 3.5, 8.1, 9.0],
            audio_artifact_periodic_timestamps_ch1=[2.0, 3.5],
            audio_artifact_periodic_timestamps_ch3=[8.1, 9.0],
        )
        cycle_record = repo.save_movement_cycle(
            cycle,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
            synchronization_id=sync_record.id,
        )

        db_session.flush()

        record = db_session.get(MovementCycleRecord, cycle_record.id)
        assert record.audio_artifact_periodic_fail is True
        assert record.audio_artifact_periodic_fail_ch1 is True
        assert record.audio_artifact_periodic_fail_ch2 is False
        assert record.audio_artifact_periodic_fail_ch3 is True
        assert record.audio_artifact_periodic_timestamps is not None
        assert len(record.audio_artifact_periodic_timestamps) == 4
        assert record.audio_artifact_periodic_timestamps_ch1 is not None
        assert len(record.audio_artifact_periodic_timestamps_ch1) == 2

    def test_cycle_periodic_columns_default_false(
        self, db_session, movement_cycle_factory, audio_processing_factory,
        synchronization_factory, biomechanics_import_factory,
    ):
        """Cycle record should default periodic fields to False/None."""
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
        )
        cycle_record = repo.save_movement_cycle(
            cycle,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
            synchronization_id=sync_record.id,
        )

        db_session.flush()

        record = db_session.get(MovementCycleRecord, cycle_record.id)
        assert record.audio_artifact_periodic_fail is False
        assert record.audio_artifact_periodic_fail_ch1 is False
        assert record.audio_artifact_periodic_timestamps is None


class TestPeriodicArtifactExcelReport:
    """Verify periodic artifact columns appear in Excel reports."""

    def test_sync_sheet_includes_periodic_columns(
        self, db_session, synchronization_factory, audio_processing_factory,
        biomechanics_import_factory,
    ):
        """Synchronization Excel sheet should contain periodic artifact columns."""
        from src.reports.report_generator import ReportGenerator

        repo = Repository(db_session)

        audio_record = repo.save_audio_processing(audio_processing_factory())
        biomech_record = repo.save_biomechanics_import(biomechanics_import_factory())

        sync = synchronization_factory(
            periodic_artifact_detected=True,
            periodic_artifact_detected_ch1=True,
            periodic_artifact_segments=[(1.0, 3.5)],
            periodic_artifact_segments_ch1=[(1.0, 3.5)],
        )
        sync_record = repo.save_synchronization(
            sync,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
        )
        db_session.flush()

        generator = ReportGenerator(db_session)
        sync_sheet = generator.generate_synchronization_sheet(
            participant_id=sync_record.participant_id,
            maneuver="walk",
            knee="left",
        )

        expected_columns = [
            "Periodic Artifact Detected",
            "Periodic Artifact Ch1",
            "Periodic Artifact Ch2",
            "Periodic Artifact Ch3",
            "Periodic Artifact Ch4",
            "Periodic Artifact Segments",
        ]
        for col in expected_columns:
            assert col in sync_sheet.columns, (
                f"Missing column '{col}' in Synchronization sheet. "
                f"Available: {list(sync_sheet.columns)}"
            )

    def test_cycles_sheet_includes_periodic_columns(
        self, db_session, movement_cycle_factory, audio_processing_factory,
        synchronization_factory, biomechanics_import_factory,
    ):
        """Movement Cycles Excel sheet should contain periodic artifact columns."""
        from src.reports.report_generator import ReportGenerator

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
            audio_artifact_periodic_fail=True,
            audio_artifact_periodic_fail_ch1=True,
            audio_artifact_periodic_timestamps=[2.0, 3.5],
            audio_artifact_periodic_timestamps_ch1=[2.0, 3.5],
        )
        repo.save_movement_cycle(
            cycle,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
            synchronization_id=sync_record.id,
        )
        db_session.flush()

        generator = ReportGenerator(db_session)
        cycles_sheet = generator.generate_movement_cycles_sheet(
            participant_id=audio_record.participant_id,
            maneuver="walk",
            knee="left",
        )

        expected_columns = [
            "Audio Artifact Periodic Fail",
            "Audio Artifact Periodic Fail Ch1",
            "Audio Artifact Periodic Timestamps",
        ]
        for col in expected_columns:
            assert col in cycles_sheet.columns, (
                f"Missing column '{col}' in Movement Cycles sheet. "
                f"Available: {list(cycles_sheet.columns)}"
            )


class TestPeriodicArtifactFactoryDefaults:
    """Verify factory fixtures produce valid objects with periodic fields."""

    def test_synchronization_factory_includes_periodic_fields(
        self, synchronization_factory
    ):
        """Default sync factory should include periodic artifact fields."""
        sync = synchronization_factory()
        assert hasattr(sync, "periodic_artifact_detected")
        assert sync.periodic_artifact_detected is False
        assert sync.periodic_artifact_segments is None

    def test_synchronization_factory_accepts_periodic_overrides(
        self, synchronization_factory
    ):
        """Sync factory should accept periodic artifact overrides."""
        sync = synchronization_factory(
            periodic_artifact_detected=True,
            periodic_artifact_detected_ch2=True,
            periodic_artifact_segments=[(5.0, 10.0)],
            periodic_artifact_segments_ch2=[(5.0, 10.0)],
        )
        assert sync.periodic_artifact_detected is True
        assert sync.periodic_artifact_detected_ch2 is True
        assert sync.periodic_artifact_segments == [(5.0, 10.0)]

    def test_movement_cycle_factory_includes_periodic_fields(
        self, movement_cycle_factory
    ):
        """Default cycle factory should include periodic artifact fields."""
        cycle = movement_cycle_factory()
        assert hasattr(cycle, "audio_artifact_periodic_fail")
        assert cycle.audio_artifact_periodic_fail is False
        assert cycle.audio_artifact_periodic_timestamps is None

    def test_movement_cycle_factory_accepts_periodic_overrides(
        self, movement_cycle_factory
    ):
        """Cycle factory should accept periodic artifact overrides."""
        cycle = movement_cycle_factory(
            audio_artifact_periodic_fail=True,
            audio_artifact_periodic_fail_ch3=True,
            audio_artifact_periodic_timestamps=[1.0, 2.0, 5.0, 6.0],
            audio_artifact_periodic_timestamps_ch3=[5.0, 6.0],
        )
        assert cycle.audio_artifact_periodic_fail is True
        assert cycle.audio_artifact_periodic_fail_ch3 is True
        assert cycle.audio_artifact_periodic_timestamps == [1.0, 2.0, 5.0, 6.0]


class TestContinuousArtifactThreshold:
    """Verify the continuous artifact threshold was changed from 1s to 4s."""

    def test_classify_artifact_type_uses_4s_threshold(self):
        """_classify_artifact_type should use 4.0s as the default threshold."""
        from src.audio.raw_qc import _classify_artifact_type

        intervals = [
            (0.0, 3.9),   # 3.9s -> Intermittent (< 4.0)
            (5.0, 9.1),   # 4.1s -> Continuous (>= 4.0)
            (10.0, 10.5), # 0.5s -> Intermittent
        ]

        types = _classify_artifact_type(intervals)
        assert types == ["Intermittent", "Continuous", "Intermittent"]

    def test_continuous_thresholds_dataclass_defaults(self):
        """ContinuousArtifactThresholds should default to 4s duration."""
        from src.audio.raw_qc import ContinuousArtifactThresholds

        thresholds = ContinuousArtifactThresholds()
        assert thresholds.min_duration_s == 4.0
        assert thresholds.window_size_s == 2.0
        assert thresholds.peak_snr_threshold == 10.0


class TestQCVersionsBumped:
    """Verify all QC versions are at v2."""

    def test_all_qc_versions_are_v2(self):
        from src.qc_versions import (
            AUDIO_QC_VERSION,
            BIOMECH_QC_VERSION,
            CYCLE_QC_VERSION,
        )
        assert AUDIO_QC_VERSION == 2, "Audio QC should be v2"
        assert BIOMECH_QC_VERSION == 2, "Biomech QC should be v2"
        assert CYCLE_QC_VERSION == 2, "Cycle QC should be v2"


class TestSyncQCRemovedColumns:
    """Verify sync_qc_fail and sync_qc_notes were removed from Synchronization."""

    def test_synchronization_has_no_sync_qc_fail(self, synchronization_factory):
        """Synchronization should not have sync_qc_fail field."""
        sync = synchronization_factory()
        assert not hasattr(sync, "sync_qc_fail") or "sync_qc_fail" not in type(sync).__dataclass_fields__

    def test_sync_record_has_no_sync_qc_fail_column(self, db_session):
        """SynchronizationRecord should not have sync_qc_fail column."""
        assert not hasattr(SynchronizationRecord, "sync_qc_fail")

    def test_sync_record_has_no_sync_qc_notes_column(self, db_session):
        """SynchronizationRecord should not have sync_qc_notes column."""
        assert not hasattr(SynchronizationRecord, "sync_qc_notes")


class TestPerformSyncQCReturnsSyncQCOutput:
    """Verify perform_sync_qc returns a SyncQCOutput object."""

    def test_return_is_sync_qc_output(self, syncd_walk, tmp_path):
        """perform_sync_qc should return a SyncQCOutput dataclass."""
        from src.synchronization.quality_control import perform_sync_qc, SyncQCOutput

        synced_dir = tmp_path / "Synced"
        synced_dir.mkdir()
        synced_pkl_path = synced_dir / "Test_L_Walk_Medium_P1_synced.pkl"
        syncd_walk.to_pickle(synced_pkl_path)

        result = perform_sync_qc(
            synced_pkl_path,
            output_dir=tmp_path / "QC_Output",
            create_plots=False,
            acoustic_threshold=0.05,
        )

        assert isinstance(result, SyncQCOutput)
        assert isinstance(result.clean_cycles, list)
        assert isinstance(result.outlier_cycles, list)
        assert result.sync_periodic_results is None or isinstance(result.sync_periodic_results, dict)
        assert isinstance(result.cycle_qc_results, list)


class TestMLGhostMethodsRaiseNotImplemented:
    """Verify ML ghost methods raise NotImplementedError."""

    def test_classify_cycle_artifacts_ml_raises(self):
        """ML cycle classifier should raise NotImplementedError."""
        from src.audio.cycle_qc import classify_cycle_artifacts_ml

        with pytest.raises(NotImplementedError):
            classify_cycle_artifacts_ml(pd.DataFrame(), "walk")

    def test_validate_sync_quality_model_based_raises(self):
        """Model-based sync validator should raise NotImplementedError."""
        from src.audio.cycle_qc import validate_sync_quality_model_based

        with pytest.raises(NotImplementedError):
            validate_sync_quality_model_based(pd.DataFrame(), "walk")
