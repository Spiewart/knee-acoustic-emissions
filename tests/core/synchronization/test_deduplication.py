"""Test sync deduplication and movement cycle persistence to database.

These tests validate the fixes for:
1. Synchronization record deduplication (using DISTINCT on sync_file_name)
2. Movement cycle persistence from disk to database
3. Correct aggregation in Excel reports (per-maneuver, not across all maneuvers)
"""

from pathlib import Path

import pandas as pd
import pytest

from src.metadata import (
    AudioProcessing,
    BiomechanicsImport,
    MovementCycle,
    Synchronization,
)
from src.reports.report_generator import ReportGenerator


class TestSyncDeduplication:
    """Test that sync records are properly deduplicated in reports."""

    def test_summary_sheet_counts_distinct_sync_files(
        self,
        db_session,
        repository,
        audio_processing_factory,
        biomechanics_import_factory,
        synchronization_factory,
        tmp_path,
    ):
        """Test that sync count in Summary sheet uses DISTINCT on sync_file_name."""
        # Create audio and biomechanics records
        audio = audio_processing_factory(
            study="AOA", study_id=5001, knee="right", maneuver="walk"
        )
        audio_record = repository.save_audio_processing(audio)
        db_session.commit()

        biomech = biomechanics_import_factory(
            study="AOA", study_id=5001, knee="right", maneuver="walk"
        )
        biomech_record = repository.save_biomechanics_import(
            biomech, audio_processing_id=audio_record.id
        )
        db_session.commit()

        # Create multiple sync records with SAME sync_file_name (simulating duplicate)
        sync_file_name = "AOA5001_walk_Pass0001_sync.pkl"
        for i in range(3):  # Add 3 records with same sync_file_name
            sync = synchronization_factory(
                study="AOA",
                study_id=5001,
                audio_processing_id=audio_record.id,
                biomechanics_import_id=biomech_record.id,
                knee="right",
                maneuver="walk",
                sync_file_name=sync_file_name,
            )
            repository.save_synchronization(
                sync,
                audio_processing_id=audio_record.id,
                biomechanics_import_id=biomech_record.id,
            )
        db_session.commit()

        # Generate report
        report = ReportGenerator(db_session)
        output_path = report.save_to_excel(
            tmp_path / "test_dedup.xlsx",
            participant_id=audio_record.participant_id,
            maneuver="walk",
            knee="right",
        )

        # Read Summary sheet
        summary = pd.read_excel(output_path, sheet_name="Summary")
        summary_dict = dict(zip(summary["Metric"], summary["Value"]))

        # Should count DISTINCT sync_file_name, so 1, not 3
        assert summary_dict["Synchronization Records"] == 1, (
            f"Expected 1 distinct sync file, but got {summary_dict['Synchronization Records']}. "
            "Deduplication on sync_file_name failed."
        )

    def test_sync_sheet_shows_one_record_per_unique_sync_file(
        self,
        db_session,
        repository,
        audio_processing_factory,
        biomechanics_import_factory,
        synchronization_factory,
        tmp_path,
    ):
        """Test that Synchronization sheet shows only unique sync files."""
        audio = audio_processing_factory(
            study="AOA", study_id=5002, knee="left", maneuver="sts"
        )
        audio_record = repository.save_audio_processing(audio)
        db_session.commit()

        biomech = biomechanics_import_factory(
            study="AOA", study_id=5002, knee="left", maneuver="sts"
        )
        biomech_record = repository.save_biomechanics_import(
            biomech, audio_processing_id=audio_record.id
        )
        db_session.commit()

        # Create 3 records with same sync_file_name
        sync_file_name = "AOA5002_sts_sync.pkl"
        for i in range(3):
            sync = synchronization_factory(
                study="AOA",
                study_id=5002,
                audio_processing_id=audio_record.id,
                biomechanics_import_id=biomech_record.id,
                knee="left",
                maneuver="sts",
                sync_file_name=sync_file_name,
                pass_number=None,
                speed=None,
            )
            repository.save_synchronization(
                sync,
                audio_processing_id=audio_record.id,
                biomechanics_import_id=biomech_record.id,
            )
        db_session.commit()

        # Generate report
        report = ReportGenerator(db_session)
        output_path = report.save_to_excel(
            tmp_path / "test_sync_sheet.xlsx",
            participant_id=audio_record.participant_id,
            maneuver="sts",
            knee="left",
        )

        # Read Synchronization sheet
        sync_sheet = pd.read_excel(output_path, sheet_name="Synchronization")

        # Should have only 1 row (deduplicated)
        assert len(sync_sheet) == 1, (
            f"Expected 1 unique sync record, but got {len(sync_sheet)}. "
            "Sync sheet deduplication failed."
        )


class TestMovementCyclePersistence:
    """Test that movement cycles are properly persisted to database and appear in reports."""

    def test_cycles_sheet_populated_when_cycles_exist(
        self,
        db_session,
        repository,
        audio_processing_factory,
        biomechanics_import_factory,
        synchronization_factory,
        movement_cycle_factory,
        tmp_path,
    ):
        """Test that Cycles sheet is populated when movement cycles exist in database."""
        audio = audio_processing_factory(
            study="AOA", study_id=5003, knee="right", maneuver="fe"
        )
        audio_record = repository.save_audio_processing(audio)
        db_session.commit()

        biomech = biomechanics_import_factory(
            study="AOA", study_id=5003, knee="right", maneuver="fe"
        )
        biomech_record = repository.save_biomechanics_import(
            biomech, audio_processing_id=audio_record.id
        )
        db_session.commit()

        sync = synchronization_factory(
            study="AOA",
            study_id=5003,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
            knee="right",
            maneuver="fe",
            pass_number=None,
            speed=None,
            sync_file_name="AOA5003_fe_sync.pkl",
        )
        sync_record = repository.save_synchronization(
            sync,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
        )
        db_session.commit()

        # Create 10 movement cycles
        for i in range(10):
            cycle = movement_cycle_factory(
                study="AOA",
                study_id=5003,
                audio_processing_id=audio_record.id,
                biomechanics_import_id=biomech_record.id,
                synchronization_id=sync_record.id,
                knee="right",
                maneuver="fe",
                pass_number=None,
                speed=None,
                cycle_file=f"fe_cycle_{i:02d}.pkl",
                cycle_index=i,
            )
            repository.save_movement_cycle(
                cycle,
                audio_processing_id=audio_record.id,
                biomechanics_import_id=biomech_record.id,
                synchronization_id=sync_record.id,
            )
        db_session.commit()

        # Generate report
        report = ReportGenerator(db_session)
        output_path = report.save_to_excel(
            tmp_path / "test_cycles.xlsx",
            participant_id=audio_record.participant_id,
            maneuver="fe",
            knee="right",
        )

        # Read Cycles sheet
        cycles_sheet = pd.read_excel(output_path, sheet_name="Cycles")

        # Should have 10 cycles
        assert len(cycles_sheet) == 10, (
            f"Expected 10 cycles in sheet, but got {len(cycles_sheet)}. "
            "Movement cycles not properly persisted to database."
        )

    def test_cycles_sheet_has_outlier_classification(
        self,
        db_session,
        repository,
        audio_processing_factory,
        biomechanics_import_factory,
        synchronization_factory,
        movement_cycle_factory,
        tmp_path,
    ):
        """Test that Cycles sheet includes 'Is Outlier' column for QC classification."""
        audio = audio_processing_factory(
            study="AOA", study_id=5004, knee="left", maneuver="walk"
        )
        audio_record = repository.save_audio_processing(audio)
        db_session.commit()

        biomech = biomechanics_import_factory(
            study="AOA", study_id=5004, knee="left", maneuver="walk"
        )
        biomech_record = repository.save_biomechanics_import(
            biomech, audio_processing_id=audio_record.id
        )
        db_session.commit()

        sync = synchronization_factory(
            study="AOA",
            study_id=5004,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
            sync_file_name="AOA5004_walk_sync.pkl",
        )
        sync_record = repository.save_synchronization(
            sync,
            audio_processing_id=audio_record.id,
            biomechanics_import_id=biomech_record.id,
        )
        db_session.commit()

        # Create 3 clean cycles and 2 outlier cycles
        for i in range(3):
            cycle = movement_cycle_factory(
                study="AOA",
                study_id=5004,
                audio_processing_id=audio_record.id,
                biomechanics_import_id=biomech_record.id,
                synchronization_id=sync_record.id,
                cycle_file=f"walk_clean_{i}.pkl",
                cycle_index=i,
                is_outlier=False,
            )
            repository.save_movement_cycle(
                cycle,
                audio_processing_id=audio_record.id,
                biomechanics_import_id=biomech_record.id,
                synchronization_id=sync_record.id,
            )

        for i in range(2):
            cycle = movement_cycle_factory(
                study="AOA",
                study_id=5004,
                audio_processing_id=audio_record.id,
                biomechanics_import_id=biomech_record.id,
                synchronization_id=sync_record.id,
                cycle_file=f"walk_outlier_{i}.pkl",
                cycle_index=3 + i,
                is_outlier=True,
            )
            repository.save_movement_cycle(
                cycle,
                audio_processing_id=audio_record.id,
                biomechanics_import_id=biomech_record.id,
                synchronization_id=sync_record.id,
            )
        db_session.commit()

        # Generate report
        report = ReportGenerator(db_session)
        output_path = report.save_to_excel(
            tmp_path / "test_outlier_col.xlsx",
            participant_id=audio_record.participant_id,
            maneuver="walk",
            knee="left",
        )

        # Read Cycles sheet
        cycles_sheet = pd.read_excel(output_path, sheet_name="Cycles")

        # Should have "Is Outlier" column
        assert "Is Outlier" in cycles_sheet.columns, (
            "Cycles sheet missing 'Is Outlier' column for QC classification"
        )

        # Check counts: 3 should be False, 2 should be True
        outlier_count = cycles_sheet["Is Outlier"].sum()
        assert outlier_count == 2, (
            f"Expected 2 outlier cycles, got {outlier_count}. "
            "Outlier classification not properly persisted."
        )

        clean_count = (~cycles_sheet["Is Outlier"]).sum()
        assert clean_count == 3, (
            f"Expected 3 clean cycles, got {clean_count}. "
            "Clean cycle classification not properly persisted."
        )


class TestAggregationPerManeuver:
    """Test that sync/cycle counts are properly scoped to maneuver/knee (not cross-aggregated)."""

    def test_summary_sheet_counts_only_requested_maneuver(
        self,
        db_session,
        repository,
        audio_processing_factory,
        biomechanics_import_factory,
        synchronization_factory,
        tmp_path,
    ):
        """Test that Summary sheet counts only sync records for requested maneuver."""
        # Create audio/biomech for walk
        audio_walk = audio_processing_factory(
            study="AOA", study_id=5005, knee="right", maneuver="walk"
        )
        audio_walk_record = repository.save_audio_processing(audio_walk)
        db_session.commit()

        biomech_walk = biomechanics_import_factory(
            study="AOA", study_id=5005, knee="right", maneuver="walk"
        )
        biomech_walk_record = repository.save_biomechanics_import(
            biomech_walk, audio_processing_id=audio_walk_record.id
        )
        db_session.commit()

        # Create 3 sync records for walk
        for i in range(3):
            sync = synchronization_factory(
                study="AOA",
                study_id=5005,
                audio_processing_id=audio_walk_record.id,
                biomechanics_import_id=biomech_walk_record.id,
                knee="right",
                maneuver="walk",
                sync_file_name=f"AOA5005_walk_Pass{i:04d}_sync.pkl",
                pass_number=i,
            )
            repository.save_synchronization(
                sync,
                audio_processing_id=audio_walk_record.id,
                biomechanics_import_id=biomech_walk_record.id,
            )
        db_session.commit()

        # Create audio/biomech for STS (different participant or same?)
        # For this test, let's use same participant so we can verify filtering
        audio_sts = audio_processing_factory(
            study="AOA", study_id=5005, knee="right", maneuver="sts"
        )
        audio_sts_record = repository.save_audio_processing(audio_sts)
        db_session.commit()

        biomech_sts = biomechanics_import_factory(
            study="AOA", study_id=5005, knee="right", maneuver="sts"
        )
        biomech_sts_record = repository.save_biomechanics_import(
            biomech_sts, audio_processing_id=audio_sts_record.id
        )
        db_session.commit()

        # Create 1 sync record for STS
        sync_sts = synchronization_factory(
            study="AOA",
            study_id=5005,
            audio_processing_id=audio_sts_record.id,
            biomechanics_import_id=biomech_sts_record.id,
            knee="right",
            maneuver="sts",
            pass_number=None,
            speed=None,
            sync_file_name="AOA5005_sts_sync.pkl",
        )
        repository.save_synchronization(
            sync_sts,
            audio_processing_id=audio_sts_record.id,
            biomechanics_import_id=biomech_sts_record.id,
        )
        db_session.commit()

        # Generate report for WALK maneuver only
        report = ReportGenerator(db_session)
        output_path = report.save_to_excel(
            tmp_path / "test_walk_report.xlsx",
            participant_id=audio_walk_record.participant_id,
            maneuver="walk",
            knee="right",
        )

        # Read Summary sheet
        summary = pd.read_excel(output_path, sheet_name="Summary")
        summary_dict = dict(zip(summary["Metric"], summary["Value"]))

        # Should show 3 sync records for walk, NOT 4 (3 walk + 1 sts)
        assert summary_dict["Synchronization Records"] == 3, (
            f"Walk report should show 3 sync records, but got {summary_dict['Synchronization Records']}. "
            "Report is likely aggregating across maneuvers."
        )

        # Generate report for STS maneuver only
        output_path_sts = report.save_to_excel(
            tmp_path / "test_sts_report.xlsx",
            participant_id=audio_sts_record.participant_id,
            maneuver="sts",
            knee="right",
        )

        # Read Summary sheet
        summary_sts = pd.read_excel(output_path_sts, sheet_name="Summary")
        summary_sts_dict = dict(zip(summary_sts["Metric"], summary_sts["Value"]))

        # Should show 1 sync record for STS, NOT 4
        assert summary_sts_dict["Synchronization Records"] == 1, (
            f"STS report should show 1 sync record, but got {summary_sts_dict['Synchronization Records']}. "
            "Report is likely aggregating across maneuvers."
        )
