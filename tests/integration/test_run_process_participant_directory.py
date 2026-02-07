"""Tests for process_participant_directory execution functions."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.models import BiomechanicsFileMetadata, BiomechanicsRecording
from src.orchestration.participant import sync_single_audio_file


@pytest.fixture
def mock_audio_df():
    """Create mock audio DataFrame."""
    n_samples = 10000
    return pd.DataFrame(
        {
            "tt": np.arange(n_samples) / 1000.0,
            "ch1": np.random.randn(n_samples) * 0.01,
            "ch2": np.random.randn(n_samples) * 0.01,
            "ch3": np.random.randn(n_samples) * 0.01,
            "ch4": np.random.randn(n_samples) * 0.01,
        }
    )


@pytest.fixture
def mock_biomechanics_recordings():
    """Create mock biomechanics recordings for all speeds."""
    recordings = []
    for speed in ["slow", "normal", "fast"]:
        for pass_num in [1, 2, 3, 4]:
            bio_df = pd.DataFrame(
                {
                    "TIME": pd.to_timedelta([0, 1, 2, 3, 4], unit="s"),
                    "Knee Angle Z": [10, 20, 30, 20, 10],
                }
            )
            recording = BiomechanicsRecording(
                study="AOA",
                study_id=1011,
                maneuver="walk",
                speed=speed,
                pass_number=pass_num,
                biomech_file_name=f"bio_{speed}_{pass_num}.c3d",
                data=bio_df,
            )
            recordings.append(recording)
    return recordings


class TestSyncSingleAudioFile:
    """Tests for sync_single_audio_file function."""

    def test_sync_walk_processes_all_speeds(
        self, tmp_path, mock_audio_df, mock_biomechanics_recordings
    ):
        """sync_single_audio_file should process all speeds for walking."""
        # Setup directory structure
        participant_dir = tmp_path / "#1011"
        motion_capture_dir = participant_dir / "Motion Capture"
        motion_capture_dir.mkdir(parents=True)

        knee_dir = participant_dir / "Right Knee"
        walking_dir = knee_dir / "Walking"
        outputs_dir = walking_dir / "HP_W11.2-1-20240126_135704_outputs"
        outputs_dir.mkdir(parents=True)

        audio_file = outputs_dir / "HP_W11.2-1-20240126_135704_with_freq.pkl"
        mock_audio_df.to_pickle(audio_file)

        biomech_file = motion_capture_dir / "AOA1011_Biomechanics_Full_Set.xlsx"
        biomech_file.touch()

        # Mock the imports and sync functions
        with patch(
            "src.orchestration.participant.import_biomechanics_recordings"
        ) as mock_import, patch(
            "src.orchestration.participant.load_audio_data"
        ) as mock_load_audio, patch(
            "src.orchestration.participant._sync_and_save_recording"
        ) as mock_sync, patch(
            "src.orchestration.participant.plot_stomp_detection"
        ) as mock_plot:

            # Configure mocks
            mock_load_audio.return_value = mock_audio_df

            # Mock import_biomechanics_recordings to return different recordings per speed
            # Note: The function is called with "medium" but recordings have speed="normal"
            def import_side_effect(biomechanics_file, maneuver, speed, **kwargs):
                # Map medium -> normal for fixture lookup
                lookup_speed = "normal" if speed == "medium" else speed
                return [
                    rec for rec in mock_biomechanics_recordings
                    if rec.speed == lookup_speed
                ]

            mock_import.side_effect = import_side_effect

            # Mock sync function
            synced_df = mock_audio_df.copy()
            synced_df["Knee Angle Z"] = 15.0
            output_path = tmp_path / "output.pkl"
            mock_sync.return_value = (
                output_path,
                synced_df,
                (10.0, 5.0, 5.0, {}),  # stomp times + detection_results
                pd.DataFrame(),
            )

            # Execute
            result = sync_single_audio_file(audio_file)

            # Verify
            assert result is True

            # Should call import_biomechanics_recordings for each speed
            assert mock_import.call_count == 3
            speeds_called = [call.kwargs["speed"] for call in mock_import.call_args_list]
            assert "slow" in speeds_called
            assert "medium" in speeds_called
            assert "fast" in speeds_called

            # Should sync all 12 recordings (3 speeds × 4 passes)
            assert mock_sync.call_count == 12

    def test_sync_walk_processes_all_passes_per_speed(
        self, tmp_path, mock_audio_df, mock_biomechanics_recordings
    ):
        """Each speed should have all passes synced."""
        # Setup
        participant_dir = tmp_path / "#1011"
        motion_capture_dir = participant_dir / "Motion Capture"
        motion_capture_dir.mkdir(parents=True)

        knee_dir = participant_dir / "Right Knee"
        walking_dir = knee_dir / "Walking"
        outputs_dir = walking_dir / "HP_W11.2-1-20240126_135704_outputs"
        outputs_dir.mkdir(parents=True)

        audio_file = outputs_dir / "HP_W11.2-1-20240126_135704_with_freq.pkl"
        mock_audio_df.to_pickle(audio_file)

        biomech_file = motion_capture_dir / "AOA1011_Biomechanics_Full_Set.xlsx"
        biomech_file.touch()

        with patch(
            "src.orchestration.participant.import_biomechanics_recordings"
        ) as mock_import, patch(
            "src.orchestration.participant.load_audio_data"
        ) as mock_load_audio, patch(
            "src.orchestration.participant._sync_and_save_recording"
        ) as mock_sync, patch(
            "src.orchestration.participant.plot_stomp_detection"
        ) as mock_plot:

            mock_load_audio.return_value = mock_audio_df

            def import_side_effect(biomechanics_file, maneuver, speed, **kwargs):
                # Map medium -> normal for fixture lookup
                lookup_speed = "normal" if speed == "medium" else speed
                return [
                    rec for rec in mock_biomechanics_recordings
                    if rec.speed == lookup_speed
                ]

            mock_import.side_effect = import_side_effect

            synced_df = mock_audio_df.copy()
            synced_df["Knee Angle Z"] = 15.0
            output_path = tmp_path / "output.pkl"
            mock_sync.return_value = (
                output_path,
                synced_df,
                (10.0, 5.0, 5.0, {}),
                pd.DataFrame(),
            )

            result = sync_single_audio_file(audio_file)

            assert result is True

            # Check that all pass numbers were synced
            synced_recordings = [call.kwargs["recording"] for call in mock_sync.call_args_list]
            pass_numbers_synced = [rec.pass_number for rec in synced_recordings]

            # Should have 4 passes per speed (3 speeds)
            assert pass_numbers_synced.count(1) == 3
            assert pass_numbers_synced.count(2) == 3
            assert pass_numbers_synced.count(3) == 3
            assert pass_numbers_synced.count(4) == 3

    def test_sync_continues_on_individual_failures(
        self, tmp_path, mock_audio_df, mock_biomechanics_recordings
    ):
        """Should continue syncing other recordings if one fails."""
        # Setup
        participant_dir = tmp_path / "#1011"
        motion_capture_dir = participant_dir / "Motion Capture"
        motion_capture_dir.mkdir(parents=True)

        knee_dir = participant_dir / "Right Knee"
        walking_dir = knee_dir / "Walking"
        outputs_dir = walking_dir / "HP_W11.2-1-20240126_135704_outputs"
        outputs_dir.mkdir(parents=True)

        audio_file = outputs_dir / "HP_W11.2-1-20240126_135704_with_freq.pkl"
        mock_audio_df.to_pickle(audio_file)

        biomech_file = motion_capture_dir / "AOA1011_Biomechanics_Full_Set.xlsx"
        biomech_file.touch()

        with patch(
            "src.orchestration.participant.import_biomechanics_recordings"
        ) as mock_import, patch(
            "src.orchestration.participant.load_audio_data"
        ) as mock_load_audio, patch(
            "src.orchestration.participant._sync_and_save_recording"
        ) as mock_sync, patch(
            "src.orchestration.participant.plot_stomp_detection"
        ) as mock_plot:

            mock_load_audio.return_value = mock_audio_df

            def import_side_effect(biomechanics_file, maneuver, speed, **kwargs):
                # Map medium -> normal for fixture lookup
                lookup_speed = "normal" if speed == "medium" else speed
                return [
                    rec for rec in mock_biomechanics_recordings
                    if rec.speed == lookup_speed
                ][:2]  # Only 2 passes per speed

            mock_import.side_effect = import_side_effect

            # Make first call fail, rest succeed
            call_count = [0]

            output_path = tmp_path / "output.pkl"

            def sync_side_effect(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise ValueError("Simulated sync failure")
                synced_df = mock_audio_df.copy()
                synced_df["Knee Angle Z"] = 15.0
                return (
                    output_path,
                    synced_df,
                    (10.0, 5.0, 5.0, {}),
                    pd.DataFrame(),
                )

            mock_sync.side_effect = sync_side_effect

            result = sync_single_audio_file(audio_file)

            # Should still succeed overall (5 out of 6 succeeded)
            assert result is True
            assert mock_sync.call_count == 6

    def test_sync_non_walk_maneuver_single_recording(
        self, tmp_path, mock_audio_df
    ):
        """Non-walking maneuvers should sync all recordings without speed."""
        # Setup for sit-to-stand
        participant_dir = tmp_path / "#1011"
        motion_capture_dir = participant_dir / "Motion Capture"
        motion_capture_dir.mkdir(parents=True)

        knee_dir = participant_dir / "Right Knee"
        sts_dir = knee_dir / "Sit-Stand"
        outputs_dir = sts_dir / "HP_W11.2-3-20240126_135706_outputs"
        outputs_dir.mkdir(parents=True)

        audio_file = outputs_dir / "HP_W11.2-3-20240126_135706_with_freq.pkl"
        mock_audio_df.to_pickle(audio_file)

        biomech_file = motion_capture_dir / "AOA1011_Biomechanics_Full_Set.xlsx"
        biomech_file.touch()

        # Create sit-to-stand recording
        bio_df = pd.DataFrame(
            {
                "TIME": pd.to_timedelta([0, 5, 10], unit="s"),
                "Knee Angle Z": [90, 10, 90],
            }
        )
        recording = BiomechanicsRecording(
            study="AOA",
            study_id=1011,
            maneuver="sit_to_stand",
            speed=None,
            pass_number=None,
            biomech_file_name="sit_to_stand.c3d",
            data=bio_df,
        )

        with patch(
            "src.orchestration.participant.import_biomechanics_recordings"
        ) as mock_import, patch(
            "src.orchestration.participant.load_audio_data"
        ) as mock_load_audio, patch(
            "src.orchestration.participant._sync_and_save_recording"
        ) as mock_sync, patch(
            "src.orchestration.participant.plot_stomp_detection"
        ) as mock_plot:

            mock_load_audio.return_value = mock_audio_df
            mock_import.return_value = [recording]

            synced_df = mock_audio_df.copy()
            synced_df["Knee Angle Z"] = 45.0
            output_path = tmp_path / "output.pkl"
            mock_sync.return_value = (
                output_path,
                synced_df,
                (10.0, 5.0, 5.0, {}),
                pd.DataFrame(),
            )

            result = sync_single_audio_file(audio_file)

            assert result is True
            assert mock_sync.call_count == 1

            # Verify pass_number and speed are None
            call_kwargs = mock_sync.call_args_list[0].kwargs
            assert call_kwargs["pass_number"] is None
            assert call_kwargs["speed"] is None

    def test_sync_rejects_already_synced_files(self, tmp_path, mock_audio_df):
        """Should reject files in Synced/ directory."""
        # Setup with file in Synced directory
        participant_dir = tmp_path / "#1011"
        knee_dir = participant_dir / "Right Knee"
        walking_dir = knee_dir / "Walking"
        synced_dir = walking_dir / "Synced"
        synced_dir.mkdir(parents=True)

        # File is already in Synced folder
        audio_file = synced_dir / "Right_walk_Pass0001_slow.pkl"
        mock_audio_df.to_pickle(audio_file)

        result = sync_single_audio_file(audio_file)

        assert result is False

    def test_sync_requires_audio_columns(self, tmp_path):
        """Should validate audio DataFrame has required columns."""
        participant_dir = tmp_path / "#1011"
        motion_capture_dir = participant_dir / "Motion Capture"
        motion_capture_dir.mkdir(parents=True)

        knee_dir = participant_dir / "Right Knee"
        walking_dir = knee_dir / "Walking"
        outputs_dir = walking_dir / "HP_W11.2-1-20240126_135704_outputs"
        outputs_dir.mkdir(parents=True)

        # Create audio file missing required channels
        bad_audio_df = pd.DataFrame(
            {
                "tt": np.arange(100) / 1000.0,
                "ch1": np.random.randn(100),
                # Missing ch2, ch3, ch4
            }
        )

        audio_file = outputs_dir / "HP_W11.2-1-20240126_135704_with_freq.pkl"
        bad_audio_df.to_pickle(audio_file)

        biomech_file = motion_capture_dir / "AOA1011_Biomechanics_Full_Set.xlsx"
        biomech_file.touch()

        with patch("src.orchestration.participant.load_audio_data") as mock_load:
            mock_load.return_value = bad_audio_df

            result = sync_single_audio_file(audio_file)

            assert result is False
