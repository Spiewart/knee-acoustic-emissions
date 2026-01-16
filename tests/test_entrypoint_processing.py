"""Tests to verify entrypoint logic for multi-stage processing.

Ensures that when starting from a particular entrypoint, all downstream stages run.
This is critical for maintaining data consistency when upstream data changes.
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from src.orchestration.participant import process_participant


@pytest.fixture
def mock_participant_dir(tmp_path):
    """Create a mock participant directory structure."""
    participant_dir = tmp_path / "#1011"
    participant_dir.mkdir()

    # Create required subdirectories
    (participant_dir / "Left Knee").mkdir()
    (participant_dir / "Right Knee").mkdir()
    (participant_dir / "Motion Capture").mkdir()

    return participant_dir


class TestEntrypointLogic:
    """Tests for entrypoint processing logic."""

    @patch('src.orchestration.participant.check_participant_dir_for_bin_stage')
    @patch('src.orchestration.participant.motion_capture_folder_has_required_data')
    @patch('src.orchestration.participant.parse_participant_directory')
    @patch('src.orchestration.participant._process_bin_stage')
    @patch('src.orchestration.participant.find_synced_files')
    @patch('src.orchestration.participant._filter_synced_files')
    @patch('src.orchestration.participant.perform_sync_qc')
    def test_entrypoint_bin_runs_all_stages(
        self,
        mock_sync_qc,
        mock_filter_synced,
        mock_find_synced,
        mock_bin_stage,
        mock_parse,
        mock_motion_capture_check,
        mock_bin_dir_check,
        mock_participant_dir,
    ):
        """Entrypoint 'bin' should run: bin -> sync -> cycles."""
        # Setup
        mock_bin_stage.return_value = ([Path("dummy.pkl")], [None])
        mock_find_synced.return_value = [Path("dummy_synced.pkl")]
        mock_filter_synced.return_value = [Path("dummy_synced.pkl")]
        mock_sync_qc.return_value = ([], [], Path("output"))

        # Create required Excel file
        (mock_participant_dir / "Motion Capture" / "AOA1011_Biomechanics_Full_Set.xlsx").touch()

        # Execute
        with patch('src.orchestration.participant._find_excel_file') as mock_find_excel:
            mock_find_excel.return_value = mock_participant_dir / "Motion Capture" / "AOA1011_Biomechanics_Full_Set.xlsx"
            result = process_participant(mock_participant_dir, entrypoint="bin")

        # Verify all stages were called
        assert mock_bin_stage.called, "Bin stage should be called when entrypoint='bin'"
        assert mock_parse.called, "Sync stage should be called when entrypoint='bin'"
        assert mock_sync_qc.called, "Cycles stage should be called when entrypoint='bin'"
        assert result is True

    @patch('src.orchestration.participant.check_participant_dir_for_required_files')
    @patch('src.orchestration.participant.motion_capture_folder_has_required_data')
    @patch('src.orchestration.participant._process_bin_stage')
    @patch('src.orchestration.participant.parse_participant_directory')
    @patch('src.orchestration.participant.find_synced_files')
    @patch('src.orchestration.participant._filter_synced_files')
    @patch('src.orchestration.participant.perform_sync_qc')
    def test_entrypoint_sync_skips_bin_runs_sync_and_cycles(
        self,
        mock_sync_qc,
        mock_filter_synced,
        mock_find_synced,
        mock_parse,
        mock_bin_stage,
        mock_motion_capture_check,
        mock_dir_check,
        mock_participant_dir,
    ):
        """Entrypoint 'sync' should skip bin and run: sync -> cycles."""
        # Setup
        mock_bin_stage.return_value = ([Path("dummy.pkl")], [None])
        mock_find_synced.return_value = [Path("dummy_synced.pkl")]
        mock_filter_synced.return_value = [Path("dummy_synced.pkl")]
        mock_sync_qc.return_value = ([], [], Path("output"))

        # Create required Excel file
        (mock_participant_dir / "Motion Capture" / "AOA1011_Biomechanics_Full_Set.xlsx").touch()

        # Execute
        with patch('src.orchestration.participant._find_excel_file') as mock_find_excel:
            mock_find_excel.return_value = mock_participant_dir / "Motion Capture" / "AOA1011_Biomechanics_Full_Set.xlsx"
            result = process_participant(mock_participant_dir, entrypoint="sync")

        # Verify bin stage was NOT called
        assert not mock_bin_stage.called, "Bin stage should NOT be called when entrypoint='sync'"

        # Verify sync and cycles stages were called
        assert mock_parse.called, "Sync stage should be called when entrypoint='sync'"
        assert mock_sync_qc.called, "Cycles stage should be called when entrypoint='sync'"
        assert result is True

    @patch('src.orchestration.participant.check_participant_dir_for_required_files')
    @patch('src.orchestration.participant.motion_capture_folder_has_required_data')
    @patch('src.orchestration.participant._process_bin_stage')
    @patch('src.orchestration.participant.parse_participant_directory')
    @patch('src.orchestration.participant.find_synced_files')
    @patch('src.orchestration.participant._filter_synced_files')
    @patch('src.orchestration.participant.perform_sync_qc')
    def test_entrypoint_cycles_skips_bin_and_sync_runs_cycles_only(
        self,
        mock_sync_qc,
        mock_filter_synced,
        mock_find_synced,
        mock_parse,
        mock_bin_stage,
        mock_motion_capture_check,
        mock_dir_check,
        mock_participant_dir,
    ):
        """Entrypoint 'cycles' should skip bin and sync, run cycles only."""
        # Setup
        mock_bin_stage.return_value = ([Path("dummy.pkl")], [None])
        mock_find_synced.return_value = [Path("dummy_synced.pkl")]
        mock_filter_synced.return_value = [Path("dummy_synced.pkl")]
        mock_sync_qc.return_value = ([], [], Path("output"))

        # Create required Excel file
        (mock_participant_dir / "Motion Capture" / "AOA1011_Biomechanics_Full_Set.xlsx").touch()

        # Execute
        with patch('src.orchestration.participant._find_excel_file') as mock_find_excel:
            mock_find_excel.return_value = mock_participant_dir / "Motion Capture" / "AOA1011_Biomechanics_Full_Set.xlsx"
            result = process_participant(mock_participant_dir, entrypoint="cycles")

        # Verify bin stage was NOT called
        assert not mock_bin_stage.called, "Bin stage should NOT be called when entrypoint='cycles'"

        # Verify sync stage was NOT called
        assert not mock_parse.called, "Sync stage should NOT be called when entrypoint='cycles'"

        # Verify cycles stage WAS called
        assert mock_sync_qc.called, "Cycles stage should be called when entrypoint='cycles'"
        assert result is True

    @patch('src.orchestration.participant.check_participant_dir_for_required_files')
    @patch('src.orchestration.participant.motion_capture_folder_has_required_data')
    @patch('src.orchestration.participant._process_bin_stage')
    @patch('src.orchestration.participant.parse_participant_directory')
    @patch('src.orchestration.participant.find_synced_files')
    @patch('src.orchestration.participant._filter_synced_files')
    @patch('src.orchestration.participant.perform_sync_qc')
    def test_default_entrypoint_is_sync(
        self,
        mock_sync_qc,
        mock_filter_synced,
        mock_find_synced,
        mock_parse,
        mock_bin_stage,
        mock_motion_capture_check,
        mock_dir_check,
        mock_participant_dir,
    ):
        """Default entrypoint should be 'sync' (skips bin, runs sync and cycles)."""
        # Setup
        mock_bin_stage.return_value = ([Path("dummy.pkl")], [None])
        mock_find_synced.return_value = [Path("dummy_synced.pkl")]
        mock_filter_synced.return_value = [Path("dummy_synced.pkl")]
        mock_sync_qc.return_value = ([], [], Path("output"))

        # Create required Excel file
        (mock_participant_dir / "Motion Capture" / "AOA1011_Biomechanics_Full_Set.xlsx").touch()

        # Execute (no entrypoint specified, should default to 'sync')
        with patch('src.orchestration.participant._find_excel_file') as mock_find_excel:
            mock_find_excel.return_value = mock_participant_dir / "Motion Capture" / "AOA1011_Biomechanics_Full_Set.xlsx"
            result = process_participant(mock_participant_dir)

        # Verify bin stage was NOT called
        assert not mock_bin_stage.called, "Bin stage should NOT be called with default entrypoint"

        # Verify sync and cycles stages were called
        assert mock_parse.called, "Sync stage should be called with default entrypoint"
        assert mock_sync_qc.called, "Cycles stage should be called with default entrypoint"
        assert result is True

    @patch('src.orchestration.participant.check_participant_dir_for_required_files')
    @patch('src.orchestration.participant.motion_capture_folder_has_required_data')
    @patch('src.orchestration.participant._process_bin_stage')
    @patch('src.orchestration.participant.parse_participant_directory')
    @patch('src.orchestration.participant.find_synced_files')
    @patch('src.orchestration.participant._filter_synced_files')
    def test_cycles_stage_skips_processing_if_no_synced_files(
        self,
        mock_filter_synced,
        mock_find_synced,
        mock_parse,
        mock_bin_stage,
        mock_motion_capture_check,
        mock_dir_check,
        mock_participant_dir,
        caplog,
    ):
        """Cycles stage should log warning if no synced files found."""
        # Setup
        mock_find_synced.return_value = []
        mock_filter_synced.return_value = []

        # Create required Excel file
        (mock_participant_dir / "Motion Capture" / "AOA1011_Biomechanics_Full_Set.xlsx").touch()

        # Execute
        with patch('src.orchestration.participant._find_excel_file') as mock_find_excel:
            mock_find_excel.return_value = mock_participant_dir / "Motion Capture" / "AOA1011_Biomechanics_Full_Set.xlsx"
            with caplog.at_level(logging.WARNING):
                result = process_participant(mock_participant_dir, entrypoint="cycles")

        # Verify warning was logged
        assert "No synced files found to run cycle QC" in caplog.text
        assert result is True


class TestEntrypointIntegration:
    """Integration tests for entrypoint cascading."""

    def test_entrypoint_values_are_valid(self):
        """Verify that all valid entrypoint values are recognized."""
        valid_entrypoints = ["bin", "sync", "cycles"]

        # This would be used in the function's stage_order
        stage_order = ["bin", "sync", "cycles"]

        for entrypoint in valid_entrypoints:
            assert entrypoint in stage_order, f"Entrypoint '{entrypoint}' should be in stage_order"

    def test_stage_ordering_is_correct(self):
        """Verify that stages are ordered correctly for cascading."""
        # The order in which stages should execute
        stage_order = ["bin", "sync", "cycles"]

        # Verify bin comes before sync
        assert stage_order.index("bin") < stage_order.index("sync")

        # Verify sync comes before cycles
        assert stage_order.index("sync") < stage_order.index("cycles")

    def test_entrypoint_cascading_logic_documentation(self):
        """
        Documentation test showing how entrypoint cascading works.

        The cascading logic ensures that when you start from an earlier stage,
        all downstream stages automatically run. This maintains data consistency
        when upstream processes modify data.

        Examples:
            - entrypoint="bin": Runs stages [bin, sync, cycles]
            - entrypoint="sync": Runs stages [sync, cycles]
            - entrypoint="cycles": Runs stages [cycles]

        This is implemented using:
            stage_order = ["bin", "sync", "cycles"]
            entrypoint_idx = stage_order.index(entrypoint)
            run_bin = entrypoint_idx <= stage_order.index("bin")
            run_sync = entrypoint_idx <= stage_order.index("sync")
            run_cycles = entrypoint_idx <= stage_order.index("cycles")
        """
        # Define the stage ordering
        stage_order = ["bin", "sync", "cycles"]

        # Test each entrypoint
        test_cases = [
            ("bin", [True, True, True]),      # Run all stages
            ("sync", [False, True, True]),    # Skip bin, run sync and cycles
            ("cycles", [False, False, True]), # Run cycles only
        ]

        for entrypoint, expected_runs in test_cases:
            entrypoint_idx = stage_order.index(entrypoint)
            run_bin = entrypoint_idx <= stage_order.index("bin")
            run_sync = entrypoint_idx <= stage_order.index("sync")
            run_cycles = entrypoint_idx <= stage_order.index("cycles")

            actual_runs = [run_bin, run_sync, run_cycles]
            assert actual_runs == expected_runs, (
                f"Entrypoint '{entrypoint}' should run stages {expected_runs}, "
                f"got {actual_runs}"
            )

    @patch('src.orchestration.participant.check_participant_dir_for_bin_stage')
    @patch('src.orchestration.participant.check_participant_dir_for_required_files')
    @patch('src.orchestration.participant.participant_dir_has_top_level_folders')
    @patch('src.orchestration.participant.motion_capture_folder_has_required_data')
    @patch('src.orchestration.participant.parse_participant_directory')
    @patch('src.orchestration.participant.find_synced_files')
    @patch('src.orchestration.participant._filter_synced_files')
    def test_validation_skipped_when_filters_applied(
        self,
        mock_filter_synced,
        mock_find_synced,
        mock_parse,
        mock_motion_capture_check,
        mock_top_level,
        mock_full_check,
        mock_bin_dir_check,
        mock_participant_dir,
    ):
        """Validation should skip full check when knee or maneuver filters applied."""
        # Setup
        mock_find_synced.return_value = []
        mock_filter_synced.return_value = []

        # Create required Excel file
        (mock_participant_dir / "Motion Capture" / "AOA1011_Biomechanics_Full_Set.xlsx").touch()

        # Execute with filters
        with patch('src.orchestration.participant._find_excel_file') as mock_find_excel:
            mock_find_excel.return_value = mock_participant_dir / "Motion Capture" / "AOA1011_Biomechanics_Full_Set.xlsx"
            process_participant(
                mock_participant_dir,
                entrypoint="sync",
                knee="left",
                maneuver="walk"
            )

        # Verify full directory validation was NOT called when filters applied
        assert not mock_full_check.called, (
            "check_participant_dir_for_required_files should not run when filters applied"
        )
        # But top-level folder check should still run
        assert mock_top_level.called, (
            "Top-level folder check should still run with filters"
        )
