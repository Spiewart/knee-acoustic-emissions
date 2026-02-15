"""Report generator sheet integrity tests."""

import numpy as np
import pandas as pd

from src.db.models import SynchronizationRecord
from src.reports.report_generator import ReportGenerator


def test_cycles_sheet_has_expected_columns(
    db_session,
    repository,
    audio_processing_factory,
    biomechanics_import_factory,
    synchronization_factory,
    movement_cycle_factory,
    tmp_path,
):
    audio = audio_processing_factory(study="AOA", study_id=3001, knee="left", maneuver="walk")
    audio_record = repository.save_audio_processing(audio)

    biomech = biomechanics_import_factory(study="AOA", study_id=3001, knee="left", maneuver="walk")
    biomech_record = repository.save_biomechanics_import(biomech, audio_processing_id=audio_record.id)

    sync = synchronization_factory(
        study="AOA",
        study_id=3001,
        audio_processing_id=audio_record.id,
        biomechanics_import_id=biomech_record.id,
        sync_file_name="AOA3001_walk_sync",
    )
    sync_record = repository.save_synchronization(
        sync,
        audio_processing_id=audio_record.id,
        biomechanics_import_id=biomech_record.id,
    )

    cycle = movement_cycle_factory(
        study="AOA",
        study_id=3001,
        audio_processing_id=audio_record.id,
        biomechanics_import_id=biomech_record.id,
        synchronization_id=sync_record.id,
        cycle_file="AOA3001_walk_cycle_0",
    )
    repository.save_movement_cycle(
        cycle,
        audio_processing_id=audio_record.id,
        biomechanics_import_id=biomech_record.id,
        synchronization_id=sync_record.id,
    )

    report = ReportGenerator(db_session)
    output_path = report.save_to_excel(
        tmp_path / "sheet_integrity.xlsx",
        participant_id=audio_record.study_id,
        maneuver="walk",
        knee="left",
    )

    cycles = pd.read_excel(output_path, sheet_name="Cycles")
    assert {"Cycle File", "Start Time (s)", "Duration (s)"}.issubset(cycles.columns)


from src.orchestration.processing_log import (
    create_sync_record_from_data,
)


class TestProcessingLogSheetIntegrity:
    """Test Excel sheet integrity through save/load/update cycles."""

    def test_method_agreement_span_persists_through_updates(
        self,
        db_session,
        repository,
        audio_processing_factory,
        biomechanics_import_factory,
        synchronization_factory,
        movement_cycle_factory,
        tmp_path,
    ):
        """Verify method_agreement_span persists through DB writes and report generation."""
        audio = audio_processing_factory(study="AOA", study_id=1011, knee="left", maneuver="walk")
        audio_record = repository.save_audio_processing(audio)

        biomech = biomechanics_import_factory(study="AOA", study_id=1011, knee="left", maneuver="walk")
        biomech_record = repository.save_biomechanics_import(biomech, audio_processing_id=audio_record.id)

        expected_span = 0.4
        sync = synchronization_factory(
            study="AOA",
            study_id=1011,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
            sync_file_name="test_sync.pkl",
            method_agreement_span=expected_span,
            consensus_methods="rms, onset, freq",
            rms_time=4.8,
            onset_time=5.1,
            freq_time=5.2,
            consensus_time=5.0,
        )
        sync_record = repository.save_synchronization(
            sync,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
        )

        cycle = movement_cycle_factory(
            study="AOA",
            study_id=1011,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
            synchronization_id=sync_record.id,
            cycle_file="test_cycle_01.pkl",
            pass_number=1,
            speed="medium",
            method_agreement_span=expected_span,
        )
        repository.save_movement_cycle(
            cycle,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
            synchronization_id=sync_record.id,
        )

        db_sync = db_session.get(SynchronizationRecord, sync_record.id)
        assert abs(db_sync.method_agreement_span - expected_span) < 0.001

        report = ReportGenerator(db_session)
        output_path = report.save_to_excel(
            tmp_path / "test_log.xlsx",
            participant_id=audio_record.study_id,
            maneuver="walk",
            knee="left",
        )

        sync_df = pd.read_excel(output_path, sheet_name="Synchronization")
        assert abs(sync_df["Method Agreement Span"].iloc[0] - expected_span) < 0.001

    def test_movement_cycles_has_correct_columns(
        self,
        db_session,
        repository,
        audio_processing_factory,
        biomechanics_import_factory,
        synchronization_factory,
        movement_cycle_factory,
        tmp_path,
    ):
        """Validate Movement Cycles sheet columns in DB-backed report."""
        audio = audio_processing_factory(study="AOA", study_id=1011, knee="left", maneuver="walk")
        audio_record = repository.save_audio_processing(audio)

        biomech = biomechanics_import_factory(study="AOA", study_id=1011, knee="left", maneuver="walk")
        biomech_record = repository.save_biomechanics_import(biomech, audio_processing_id=audio_record.id)

        sync = synchronization_factory(
            study="AOA",
            study_id=1011,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
            sync_file_name="test_sync.pkl",
        )
        sync_record = repository.save_synchronization(
            sync,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
        )

        cycle = movement_cycle_factory(
            study="AOA",
            study_id=1011,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
            synchronization_id=sync_record.id,
            cycle_file="test_cycle_01.pkl",
            pass_number=1,
            speed="medium",
        )
        repository.save_movement_cycle(
            cycle,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
            synchronization_id=sync_record.id,
        )

        report = ReportGenerator(db_session)
        output_path = report.save_to_excel(
            tmp_path / "test_log.xlsx",
            participant_id=audio_record.study_id,
            maneuver="walk",
            knee="left",
        )

        mc_df = pd.read_excel(output_path, sheet_name="Cycles")
        assert "Movement Cycle ID" in mc_df.columns
        assert "Participant ID" in mc_df.columns
        assert "Processing Date" in mc_df.columns

    def test_movement_cycles_has_log_updated(
        self,
        db_session,
        repository,
        audio_processing_factory,
        biomechanics_import_factory,
        synchronization_factory,
        movement_cycle_factory,
        tmp_path,
    ):
        """Test that Movement Cycles sheet has Processing Date timestamp."""
        audio = audio_processing_factory(study="AOA", study_id=1011, knee="left", maneuver="walk")
        audio_record = repository.save_audio_processing(audio)

        biomech = biomechanics_import_factory(study="AOA", study_id=1011, knee="left", maneuver="walk")
        biomech_record = repository.save_biomechanics_import(biomech, audio_processing_id=audio_record.id)

        sync = synchronization_factory(
            study="AOA",
            study_id=1011,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
            sync_file_name="test_sync.pkl",
        )
        sync_record = repository.save_synchronization(
            sync,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
        )

        cycle = movement_cycle_factory(
            study="AOA",
            study_id=1011,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
            synchronization_id=sync_record.id,
            cycle_file="test_cycle_01.pkl",
            pass_number=1,
            speed="medium",
        )
        repository.save_movement_cycle(
            cycle,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
            synchronization_id=sync_record.id,
        )

        report = ReportGenerator(db_session)
        output_path = report.save_to_excel(
            tmp_path / "test_log.xlsx",
            participant_id=audio_record.study_id,
            maneuver="walk",
            knee="left",
        )

        mc_df = pd.read_excel(output_path, sheet_name="Cycles")
        assert "Processing Date" in mc_df.columns
        assert not pd.isnull(mc_df["Processing Date"].iloc[0])


class TestMethodAgreementSpanCalculation:
    """Test method_agreement_span calculation in various scenarios."""

    def test_single_method_span_is_zero(self, tmp_path):
        """Test that method_agreement_span is 0 when only one method contributes."""
        maneuver_dir = tmp_path / "Left Knee" / "Walking"
        maneuver_dir.mkdir(parents=True)

        # Create synced dataframe
        synced_df = pd.DataFrame(
            {
                "tt": np.arange(0, 10.0, 0.01),
                "ch1": np.random.randn(1000),
            }
        )

        # Only RMS method contributes
        detection_results = {
            "consensus_time": 5.0,
            "rms_time": 5.0,
            "onset_time": 4.5,
            "freq_time": 5.5,
            "consensus_methods": ["rms"],  # Only RMS contributed
        }

        sync_record = create_sync_record_from_data(
            sync_file_name="test_sync",
            audio_stomp_time=5.0,
            synced_df=synced_df,
            knee_side="left",
            pass_number=1,
            speed="normal",
            detection_results=detection_results,
            audio_record=None,
            biomech_record=None,
            metadata={},
            study="AOA",
            study_id=1011,
        )

        # CRITICAL: With only one method, span should be 0
        assert sync_record.method_agreement_span == 0.0, (
            f"Method agreement span should be 0 with single method, got {sync_record.method_agreement_span}"
        )
        assert sync_record.consensus_methods == "rms", (
            f"Consensus methods should be 'rms', got {sync_record.consensus_methods}"
        )

    def test_two_methods_span_is_difference(self, tmp_path):
        """Test that method_agreement_span is the difference between two methods."""
        maneuver_dir = tmp_path / "Left Knee" / "Walking"
        maneuver_dir.mkdir(parents=True)

        synced_df = pd.DataFrame(
            {
                "tt": np.arange(0, 10.0, 0.01),
                "ch1": np.random.randn(1000),
            }
        )

        # RMS and Onset methods contribute
        detection_results = {
            "consensus_time": 5.0,
            "rms_time": 4.7,  # Earlier
            "onset_time": 5.2,  # Later
            "freq_time": 6.0,  # Not used
            "consensus_methods": ["rms", "onset"],
        }

        sync_record = create_sync_record_from_data(
            sync_file_name="test_sync",
            audio_stomp_time=5.0,
            synced_df=synced_df,
            knee_side="left",
            detection_results=detection_results,
            audio_record=None,
            biomech_record=None,
            metadata={"pass_number": 1, "speed": "normal"},
            study="AOA",
            study_id=1011,
            pass_number=1,
            speed="normal",
        )

        expected_span = 5.2 - 4.7  # 0.5
        assert abs(sync_record.method_agreement_span - expected_span) < 0.001, (
            f"Method agreement span should be {expected_span}, got {sync_record.method_agreement_span}"
        )
        assert "rms" in sync_record.consensus_methods and "onset" in sync_record.consensus_methods, (
            f"Consensus methods should contain 'rms' and 'onset', got {sync_record.consensus_methods}"
        )

    def test_three_methods_span_is_range(self, tmp_path):
        """Test that method_agreement_span is max-min with three methods."""
        maneuver_dir = tmp_path / "Left Knee" / "Walking"
        maneuver_dir.mkdir(parents=True)

        synced_df = pd.DataFrame(
            {
                "tt": np.arange(0, 10.0, 0.01),
                "ch1": np.random.randn(1000),
            }
        )

        # All three methods contribute with different times
        detection_results = {
            "consensus_time": 5.0,
            "rms_time": 4.8,  # Earliest
            "onset_time": 5.1,  # Middle
            "freq_time": 5.2,  # Latest
            "consensus_methods": ["rms", "onset", "freq"],
        }

        sync_record = create_sync_record_from_data(
            sync_file_name="test_sync",
            audio_stomp_time=5.0,
            synced_df=synced_df,
            knee_side="left",
            detection_results=detection_results,
            audio_record=None,
            biomech_record=None,
            metadata={"pass_number": 1, "speed": "normal"},
            study="AOA",
            study_id=1011,
            pass_number=1,
            speed="normal",
        )

        expected_span = 5.2 - 4.8  # 0.4
        assert abs(sync_record.method_agreement_span - expected_span) < 0.001, (
            f"Method agreement span should be {expected_span}, got {sync_record.method_agreement_span}"
        )
        assert all(method in sync_record.consensus_methods for method in ["rms", "onset", "freq"]), (
            f"Consensus methods should contain all three methods, got {sync_record.consensus_methods}"
        )

    def test_all_methods_same_time_span_is_zero(self, tmp_path):
        """Test that method_agreement_span is 0 when all methods agree perfectly."""
        maneuver_dir = tmp_path / "Left Knee" / "Walking"
        maneuver_dir.mkdir(parents=True)

        synced_df = pd.DataFrame(
            {
                "tt": np.arange(0, 10.0, 0.01),
                "ch1": np.random.randn(1000),
            }
        )

        # All methods at exactly the same time (perfect agreement)
        detection_results = {
            "consensus_time": 5.0,
            "rms_time": 5.0,
            "onset_time": 5.0,
            "freq_time": 5.0,
            "consensus_methods": ["rms", "onset", "freq"],
        }

        sync_record = create_sync_record_from_data(
            sync_file_name="test_sync",
            audio_stomp_time=5.0,
            synced_df=synced_df,
            knee_side="left",
            detection_results=detection_results,
            audio_record=None,
            biomech_record=None,
            metadata={"pass_number": 1, "speed": "normal"},
            study="AOA",
            study_id=1011,
            pass_number=1,
            speed="normal",
        )

        # Perfect agreement = 0 span
        assert sync_record.method_agreement_span == 0.0, (
            f"Method agreement span should be 0 with perfect agreement, got {sync_record.method_agreement_span}"
        )

    def test_large_disagreement_span(self, tmp_path):
        """Test method_agreement_span with large disagreement between methods."""
        maneuver_dir = tmp_path / "Left Knee" / "Walking"
        maneuver_dir.mkdir(parents=True)

        synced_df = pd.DataFrame(
            {
                "tt": np.arange(0, 30.0, 0.01),
                "ch1": np.random.randn(3000),
            }
        )

        # Large disagreement (could indicate poor signal quality)
        detection_results = {
            "consensus_time": 15.0,
            "rms_time": 10.0,  # 5 seconds early
            "onset_time": 15.0,  # On time
            "freq_time": 20.0,  # 5 seconds late
            "consensus_methods": ["rms", "onset", "freq"],
        }

        sync_record = create_sync_record_from_data(
            sync_file_name="test_sync",
            audio_stomp_time=5.0,
            synced_df=synced_df,
            knee_side="left",
            detection_results=detection_results,
            audio_record=None,
            biomech_record=None,
            metadata={"pass_number": 1, "speed": "normal"},
            study="AOA",
            study_id=1011,
            pass_number=1,
            speed="normal",
        )

        expected_span = 20.0 - 10.0  # 10.0 seconds
        assert abs(sync_record.method_agreement_span - expected_span) < 0.001, (
            f"Method agreement span should be {expected_span}, got {sync_record.method_agreement_span}"
        )

    def test_method_agreement_span_persists_in_cycles_record(self, tmp_path):
        """Test that method_agreement_span is inherited by cycles records."""
        maneuver_dir = tmp_path / "Left Knee" / "Walking"
        maneuver_dir.mkdir(parents=True)

        synced_df = pd.DataFrame(
            {
                "tt": np.arange(0, 10.0, 0.01),
                "ch1": np.random.randn(1000),
            }
        )

        # Create sync record with method_agreement_span
        detection_results = {
            "consensus_time": 5.0,
            "rms_time": 4.8,
            "onset_time": 5.1,
            "freq_time": 5.3,
            "consensus_methods": ["rms", "onset", "freq"],
        }

        sync_record = create_sync_record_from_data(
            sync_file_name="test_sync",
            audio_stomp_time=5.0,
            synced_df=synced_df,
            knee_side="left",
            detection_results=detection_results,
            audio_record=None,
            biomech_record=None,
            metadata={"maneuver": "walk"},
            pass_number=1,
            speed="normal",
            study="AOA",
            study_id=1011,
        )

        expected_span = 5.3 - 4.8  # 0.5
        assert abs(sync_record.method_agreement_span - expected_span) < 0.001, (
            f"Sync record method_agreement_span should be {expected_span}"
        )

        # Update sync record with cycle data (simulating what participant_processor does)
        cycle_data = pd.DataFrame(
            {
                "tt": np.arange(0, 1.0, 0.01),
                "ch1": np.random.randn(100),
            }
        )

        # Directly update sync record with cycle statistics
        clean_cycles = [cycle_data]
        outlier_cycles = []
        sync_record.total_cycles_extracted = len(clean_cycles) + len(outlier_cycles)
        sync_record.clean_cycles = len(clean_cycles)
        sync_record.outlier_cycles = len(outlier_cycles)

        # Sync record should preserve method_agreement_span after update
        assert abs(sync_record.method_agreement_span - expected_span) < 0.001, (
            f"Sync record should preserve method_agreement_span={expected_span} after cycle update, got {sync_record.method_agreement_span}"
        )

    def test_method_agreement_span_in_excel_output(
        self,
        db_session,
        repository,
        audio_processing_factory,
        biomechanics_import_factory,
        synchronization_factory,
        tmp_path,
    ):
        """Test that method_agreement_span appears correctly in Excel output."""
        audio = audio_processing_factory(study="AOA", study_id=1011, knee="left", maneuver="walk")
        audio_record = repository.save_audio_processing(audio)

        biomech = biomechanics_import_factory(study="AOA", study_id=1011, knee="left", maneuver="walk")
        biomech_record = repository.save_biomechanics_import(biomech, audio_processing_id=audio_record.id)

        expected_span = 0.7
        sync = synchronization_factory(
            study="AOA",
            study_id=1011,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
            sync_file_name="test_sync.pkl",
            method_agreement_span=expected_span,
            consensus_methods="rms, onset, freq",
            rms_time=4.6,
            onset_time=5.0,
            freq_time=5.3,
            consensus_time=5.0,
        )
        repository.save_synchronization(
            sync,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
        )

        report = ReportGenerator(db_session)
        output_path = report.save_to_excel(
            tmp_path / "test_log.xlsx",
            participant_id=audio_record.study_id,
            maneuver="walk",
            knee="left",
        )

        sync_df = pd.read_excel(output_path, sheet_name="Synchronization")
        actual_span = sync_df["Method Agreement Span"].iloc[0]
        assert abs(actual_span - expected_span) < 0.001

        consensus_methods = sync_df["Consensus Methods"].iloc[0]
        assert "rms" in consensus_methods and "onset" in consensus_methods and "freq" in consensus_methods
