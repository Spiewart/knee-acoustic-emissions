"""Tests for the movement cycle QC module."""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sync_qc import (
    MovementCycleQC,
    _infer_maneuver_from_path,
    _infer_speed_from_path,
    find_synced_files,
    perform_sync_qc,
)


def _create_test_cycle(
    duration_s: float = 1.0,
    sample_rate: int = 1000,
    base_amplitude: float = 0.1,
    use_timedelta: bool = True,
) -> pd.DataFrame:
    """Creates a sample cycle DataFrame for testing."""
    n_points = int(duration_s * sample_rate)
    if use_timedelta:
        tt = pd.to_timedelta(np.linspace(0, duration_s, n_points), unit="s")
    else:
        tt = np.linspace(0, duration_s, n_points)

    f_ch_data = np.full(n_points, base_amplitude)
    return pd.DataFrame(
        {
            "tt": tt,
            "f_ch1": f_ch_data,
            "f_ch2": f_ch_data,
            "f_ch3": f_ch_data,
            "f_ch4": f_ch_data,
            "Knee Angle Z": np.sin(np.linspace(0, 2 * np.pi, n_points)) * 30,
        }
    )


@pytest.fixture
def sample_cycles() -> list[pd.DataFrame]:
    """Create a list of sample cycles with varying acoustic energy."""
    # This cycle should be "clean" (high energy). AUC for 1s @ amp 1.0 x 4 ch ≈ 4.0
    clean_cycle = _create_test_cycle(base_amplitude=1.0)

    # This cycle should be an "outlier" (low energy). AUC for 1s @ amp 0.01 x 4 ch ≈ 0.04
    outlier_cycle = _create_test_cycle(base_amplitude=0.01)

    return [clean_cycle, outlier_cycle, clean_cycle.copy()]


class TestMovementCycleQC:
    """Test suite for the MovementCycleQC class."""

    def test_acoustic_thresholding_separates_cycles(self, sample_cycles):
        """Should correctly separate clean cycles from low-signal outliers."""
        qc = MovementCycleQC(maneuver="walk", acoustic_threshold=1.0)
        clean, outliers = qc.analyze_cycles(sample_cycles)

        assert len(clean) == 2
        assert len(outliers) == 1
        # Verify that the correct cycle was identified as the outlier
        assert np.all(outliers[0]["f_ch1"] == 0.01)

    def test_no_cycles_provided_returns_empty_lists(self):
        """Should handle empty input list gracefully."""
        qc = MovementCycleQC(maneuver="walk")
        clean, outliers = qc.analyze_cycles([])
        assert clean == []
        assert outliers == []

    def test_acoustic_channel_selection(self):
        """Should use the correct acoustic channel ('raw' or 'filtered')."""
        # Create a cycle with low filtered energy but high raw energy
        cycle = _create_test_cycle(base_amplitude=0.01)
        for ch in ["ch1", "ch2", "ch3", "ch4"]:
            cycle[ch] = np.full(len(cycle), 1.0)

        # 1. Default 'filtered' channel should result in an outlier
        qc_filtered = MovementCycleQC(maneuver="walk", acoustic_threshold=1.0)
        _, outliers_f = qc_filtered.analyze_cycles([cycle])
        assert len(outliers_f) == 1

        # 2. 'raw' channel should result in a clean cycle
        qc_raw = MovementCycleQC(
            maneuver="walk", acoustic_threshold=1.0, acoustic_channel="raw"
        )
        clean_r, _ = qc_raw.analyze_cycles([cycle])
        assert len(clean_r) == 1


class TestPerformSyncQC:
    """Test suite for the main perform_sync_qc pipeline."""

    def test_perform_sync_qc_pipeline(self, syncd_walk, tmp_path):
        """Should run the full pipeline, creating output files."""
        # Save fixture data to a temporary "synced" file
        synced_dir = tmp_path / "Synced"
        synced_dir.mkdir()
        synced_pkl_path = synced_dir / "Test_L_Walk_Medium_P1_synced.pkl"
        syncd_walk.to_pickle(synced_pkl_path)

        # Run the QC pipeline
        output_dir_base = tmp_path / "QC_Output"
        clean, outliers, output_dir = perform_sync_qc(
            synced_pkl_path,
            output_dir=output_dir_base,
            create_plots=False,  # Disable plotting in tests
            acoustic_threshold=0.05,  # Lower threshold for synthetic data
        )

        # syncd_walk fixture has 2 distinct gait cycles, all should be clean
        assert len(clean) == 2
        assert len(outliers) == 0
        assert output_dir == output_dir_base / "MovementCycles"

        # Check that output directories and files were created
        clean_dir = output_dir / "clean"
        outlier_dir = output_dir / "outliers"
        assert clean_dir.is_dir()
        assert outlier_dir.is_dir()

        clean_files = list(clean_dir.glob("*.pkl"))
        outlier_files = list(outlier_dir.glob("*.pkl"))

        assert len(clean_files) == 2
        assert len(outlier_files) == 0

    def test_pipeline_handles_file_not_found(self):
        """Should raise FileNotFoundError for non-existent input."""
        with pytest.raises(FileNotFoundError):
            perform_sync_qc(Path("non_existent_synced_file.pkl"))


class TestCliHelpers:
    """Test suite for CLI helper functions."""

    @pytest.mark.parametrize(
        "path_str, expected",
        [
            ("data/Left_Walk_Slow_Pass1_synced.pkl", "walk"),
            ("data/Right_Sit_to_Stand_synced.pkl", "sit_to_stand"),
            ("data/Left_Flexion_Extension_synced.pkl", "flexion_extension"),
            ("data/some_flexext_data.pkl", "flexion_extension"),
        ],
    )
    def test_infer_maneuver_from_path(self, path_str, expected):
        assert _infer_maneuver_from_path(Path(path_str)) == expected

    def test_infer_maneuver_raises_on_unknown(self):
        with pytest.raises(ValueError, match="Cannot infer maneuver"):
            _infer_maneuver_from_path(Path("unknown_maneuver.pkl"))

    @pytest.mark.parametrize(
        "path_str, expected",
        [
            ("data/Walk_Slow_synced.pkl", "slow"),
            ("data/Walk_Medium_Pass2_synced.pkl", "medium"),
            ("data/Walk_Fast_synced.pkl", "fast"),
            ("data/Sit_Stand_synced.pkl", None),
        ],
    )
    def test_infer_speed_from_path(self, path_str, expected):
        assert _infer_speed_from_path(Path(path_str)) == expected

    def test_find_synced_files(self, tmp_path):
        """Should correctly find all .pkl files within 'Synced' directories."""
        # Create a realistic directory structure
        p_dir = tmp_path / "Participant1"
        synced_dir = p_dir / "Synced"
        synced_dir.mkdir(parents=True)

        f1 = synced_dir / "file1.pkl"
        f2 = synced_dir / "file2.pkl"
        f_other = synced_dir / "not_a_pickle.txt"
        f1.touch()
        f2.touch()
        f_other.touch()

        # Deeper nested structure
        knee_dir = p_dir / "Left_Knee" / "Synced"
        knee_dir.mkdir(parents=True)
        f3 = knee_dir / "file3.pkl"
        f3.touch()

        # A non-synced pkl file in a different directory
        other_dir = p_dir / "Other"
        other_dir.mkdir()
        f_not_synced = other_dir / "file4.pkl"
        f_not_synced.touch()

        # Case 1: Search from participant root directory
        found = find_synced_files(p_dir)
        assert len(found) == 3
        assert f1 in found
        assert f2 in found
        assert f3 in found

        # Case 2: Search directly from a 'Synced' directory
        found_sub = find_synced_files(synced_dir)
        assert len(found_sub) == 2
        assert f1 in found_sub
        assert f2 in found_sub

        # Case 3: Pass a single synced file path
        found_single = find_synced_files(f1)
        assert found_single == [f1]

        # Case 4: Pass a non-synced file path
        found_none = find_synced_files(f_other)
        assert found_none == []
        found_none_pkl = find_synced_files(f_not_synced)
        assert found_none_pkl == []


class TestSyncQCCLI:
    """Test suite for the sync_qc CLI entry point."""

    @patch("sync_qc.find_synced_files")
    @patch("sync_qc.perform_sync_qc")
    def test_cli_with_valid_file(
        self, mock_perform_qc, mock_find_files, tmp_path
    ):
        """Should succeed with a valid file path and call qc function."""
        synced_file = tmp_path / "synced.pkl"
        synced_file.touch()
        mock_find_files.return_value = [synced_file]
        mock_perform_qc.return_value = ([], [], tmp_path)

        with patch.object(
            sys, "argv", ["sync_qc.py", str(synced_file)]
        ):
            from sync_qc import main

            assert main() == 0
            mock_perform_qc.assert_called_once()
            # Check that default arguments were passed correctly
            call_args, call_kwargs = mock_perform_qc.call_args
            assert call_kwargs["acoustic_threshold"] == 100.0
            assert call_kwargs["create_plots"] is True

    @patch("sync_qc.find_synced_files")
    @patch("sync_qc.perform_sync_qc")
    def test_cli_with_flags(self, mock_perform_qc, mock_find_files, tmp_path):
        """Should correctly parse --no-plots and --threshold flags."""
        synced_file = tmp_path / "synced.pkl"
        synced_file.touch()
        mock_find_files.return_value = [synced_file]
        mock_perform_qc.return_value = ([], [], tmp_path)

        with patch.object(
            sys,
            "argv",
            ["sync_qc.py", str(synced_file), "--no-plots", "--threshold", "50.0"],
        ):
            from sync_qc import main

            assert main() == 0
            mock_perform_qc.assert_called_once()
            # Check that flags were passed correctly
            call_args, call_kwargs = mock_perform_qc.call_args
            assert call_kwargs["acoustic_threshold"] == 50.0
            assert call_kwargs["create_plots"] is False

    def test_cli_path_not_found(self, caplog):
        """Should return exit code 1 if path does not exist."""
        invalid_path = "/path/to/nonexistent/file.pkl"
        with patch.object(sys, "argv", ["sync_qc.py", invalid_path]):
            from sync_qc import main

            assert main() == 1
            assert "Path does not exist" in caplog.text

    @patch("sync_qc.find_synced_files")
    def test_cli_no_synced_files_found(self, mock_find_files, tmp_path, caplog):
        """Should return exit code 1 if no synced files are found."""
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        mock_find_files.return_value = []

        with patch.object(sys, "argv", ["sync_qc.py", str(empty_dir)]):
            from sync_qc import main

            assert main() == 1
            assert "No synced files found" in caplog.text
