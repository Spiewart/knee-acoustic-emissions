"""Tests for cleanup_outputs utility."""

import pytest

from cli.cleanup_outputs import cleanup_participant_outputs, cleanup_study_directory


@pytest.fixture
def participant_with_outputs(tmp_path):
    """Create a fake participant directory with processing outputs."""
    participant_dir = tmp_path / "#1011"
    participant_dir.mkdir()

    # Create both knee directories
    for knee_side in ["Left", "Right"]:
        knee_dir = participant_dir / f"{knee_side} Knee"
        knee_dir.mkdir()

        # Create knee-level master log file
        knee_log = knee_dir / f"knee_processing_log_1011_{knee_side}.xlsx"
        knee_log.write_text("fake knee-level log data")

        # Create maneuver directories with outputs
        for maneuver in ["Walking", "Sit-Stand", "Flexion-Extension"]:
            maneuver_dir = knee_dir / maneuver
            maneuver_dir.mkdir()

            # Create outputs directory with files
            outputs_dir = maneuver_dir / "audio_outputs"
            outputs_dir.mkdir()
            (outputs_dir / "audio.pkl").write_text("fake pickle data")
            (outputs_dir / "audio_meta.json").write_text('{"fs": 2000}')
            (outputs_dir / "audio_with_freq.pkl").write_text("fake pickle with freq")

            # Create Synced directory with subdirectories
            synced_dir = maneuver_dir / "Synced"
            synced_dir.mkdir()
            (synced_dir / "Left_walk_01.pkl").write_text("fake synced data")

            clean_dir = synced_dir / "clean"
            clean_dir.mkdir()
            (clean_dir / "cycle_001.pkl").write_text("fake clean cycle")
            (clean_dir / "cycle_001.png").write_text("fake plot")

            outliers_dir = synced_dir / "outliers"
            outliers_dir.mkdir()
            (outliers_dir / "outlier_001.pkl").write_text("fake outlier")
            (outliers_dir / "outlier_001.png").write_text("fake plot")

            # Create stomp detection plot
            (maneuver_dir / "stomp_detection.png").write_text("fake stomp plot")

            # Create processing log Excel to be removed by cleanup
            (maneuver_dir / "processing_log_test.xlsx").write_text("fake log file")

            # Create .bin file that should NOT be deleted
            (maneuver_dir / "raw_audio.bin").write_text("raw binary data")

    # Create some files that should NOT be deleted
    (participant_dir / "Motion Capture").mkdir()
    (participant_dir / "Motion Capture" / "AOA1011_Biomechanics_Full_Set.xlsx").write_text("biomech data")

    return participant_dir


def test_cleanup_participant_outputs_removes_outputs_dirs(participant_with_outputs):
    """Test that cleanup removes all *_outputs directories."""
    stats = cleanup_participant_outputs(participant_with_outputs, dry_run=False)

    # Should have removed outputs directories from both knees, all maneuvers
    assert stats["outputs_dirs"] == 6  # 2 knees × 3 maneuvers

    # Verify they're actually gone
    for knee_side in ["Left", "Right"]:
        knee_dir = participant_with_outputs / f"{knee_side} Knee"
        for maneuver in ["Walking", "Sit-Stand", "Flexion-Extension"]:
            outputs_dir = knee_dir / maneuver / "audio_outputs"
            assert not outputs_dir.exists()


def test_cleanup_participant_outputs_removes_synced_dirs(participant_with_outputs):
    """Test that cleanup removes all Synced directories."""
    stats = cleanup_participant_outputs(participant_with_outputs, dry_run=False)

    # Should have removed Synced directories from both knees, all maneuvers
    assert stats["synced_dirs"] == 6  # 2 knees × 3 maneuvers

    # Verify they're actually gone
    for knee_side in ["Left", "Right"]:
        knee_dir = participant_with_outputs / f"{knee_side} Knee"
        for maneuver in ["Walking", "Sit-Stand", "Flexion-Extension"]:
            synced_dir = knee_dir / maneuver / "Synced"
            assert not synced_dir.exists()


def test_cleanup_participant_outputs_removes_png_files(participant_with_outputs):
    """Test that cleanup removes PNG files in maneuver directories."""
    stats = cleanup_participant_outputs(participant_with_outputs, dry_run=False)

    # Should have removed PNG files from both knees, all maneuvers
    assert stats["png_files"] == 6  # 2 knees × 3 maneuvers

    # Verify they're actually gone
    for knee_side in ["Left", "Right"]:
        knee_dir = participant_with_outputs / f"{knee_side} Knee"
        for maneuver in ["Walking", "Sit-Stand", "Flexion-Extension"]:
            png_file = knee_dir / maneuver / "stomp_detection.png"
            assert not png_file.exists()


def test_cleanup_participant_outputs_removes_processing_logs(participant_with_outputs):
    """Test that cleanup removes processing log Excel files."""
    stats = cleanup_participant_outputs(participant_with_outputs, dry_run=False)

    # Should have removed maneuver-level log files (2 knees × 3 maneuvers)
    # plus knee-level master logs (2 knees)
    assert stats["log_files"] == 8  # 6 maneuver logs + 2 knee logs

    # Verify maneuver-level logs are gone
    for knee_side in ["Left", "Right"]:
        knee_dir = participant_with_outputs / f"{knee_side} Knee"
        for maneuver in ["Walking", "Sit-Stand", "Flexion-Extension"]:
            log_file = knee_dir / maneuver / "processing_log_test.xlsx"
            assert not log_file.exists()

    # Verify knee-level master logs are gone
    for knee_side in ["Left", "Right"]:
        knee_dir = participant_with_outputs / f"{knee_side} Knee"
        knee_log = knee_dir / f"knee_processing_log_1011_{knee_side}.xlsx"
        assert not knee_log.exists()


def test_cleanup_removes_knee_level_master_logs(participant_with_outputs):
    """Test that cleanup specifically removes knee-level master log files."""
    # Verify knee logs exist before cleanup
    for knee_side in ["Left", "Right"]:
        knee_dir = participant_with_outputs / f"{knee_side} Knee"
        knee_log = knee_dir / f"knee_processing_log_1011_{knee_side}.xlsx"
        assert knee_log.exists()

    stats = cleanup_participant_outputs(participant_with_outputs, dry_run=False)

    # Verify knee-level master logs are removed
    for knee_side in ["Left", "Right"]:
        knee_dir = participant_with_outputs / f"{knee_side} Knee"
        knee_log = knee_dir / f"knee_processing_log_1011_{knee_side}.xlsx"
        assert not knee_log.exists()

    # Should be counted in log_files stat
    assert stats["log_files"] >= 2


def test_cleanup_dry_run_counts_logs(participant_with_outputs):
    """Test that dry_run mode counts log files but does not delete them."""
    stats = cleanup_participant_outputs(participant_with_outputs, dry_run=True)

    # Stats should include logs to be removed (6 maneuver + 2 knee)
    assert stats["log_files"] == 8

    # But maneuver-level log files should still exist
    for knee_side in ["Left", "Right"]:
        knee_dir = participant_with_outputs / f"{knee_side} Knee"
        for maneuver in ["Walking", "Sit-Stand", "Flexion-Extension"]:
            log_file = knee_dir / maneuver / "processing_log_test.xlsx"
            assert log_file.exists()

    # And knee-level logs should still exist
    for knee_side in ["Left", "Right"]:
        knee_dir = participant_with_outputs / f"{knee_side} Knee"
        knee_log = knee_dir / f"knee_processing_log_1011_{knee_side}.xlsx"
        assert knee_log.exists()


def test_cleanup_preserves_source_files(participant_with_outputs):
    """Test that cleanup preserves .bin files and biomechanics data."""
    cleanup_participant_outputs(participant_with_outputs, dry_run=False)

    # Motion Capture directory should still exist
    assert (participant_with_outputs / "Motion Capture").exists()
    assert (participant_with_outputs / "Motion Capture" / "AOA1011_Biomechanics_Full_Set.xlsx").exists()

    # .bin files should still exist
    for knee_side in ["Left", "Right"]:
        knee_dir = participant_with_outputs / f"{knee_side} Knee"
        for maneuver in ["Walking", "Sit-Stand", "Flexion-Extension"]:
            bin_file = knee_dir / maneuver / "raw_audio.bin"
            assert bin_file.exists()


def test_cleanup_dry_run_doesnt_delete(participant_with_outputs):
    """Test that dry_run mode doesn't actually delete anything."""
    stats = cleanup_participant_outputs(participant_with_outputs, dry_run=True)

    # Stats should show what would be deleted
    assert stats["outputs_dirs"] == 6
    assert stats["synced_dirs"] == 12
    assert stats["png_files"] == 6

    # But files should still exist
    for knee_side in ["Left", "Right"]:
        knee_dir = participant_with_outputs / f"{knee_side} Knee"
        for maneuver in ["Walking", "Sit-Stand", "Flexion-Extension"]:
            assert (knee_dir / maneuver / "audio_outputs").exists()
            assert (knee_dir / maneuver / "Synced").exists()
            assert (knee_dir / maneuver / "stomp_detection.png").exists()


def test_cleanup_nonexistent_participant_raises_error(tmp_path):
    """Test that cleanup raises error for nonexistent directory."""
    with pytest.raises(FileNotFoundError):
        cleanup_participant_outputs(tmp_path / "#9999", dry_run=False)


def test_cleanup_study_directory(tmp_path):
    """Test cleaning multiple participants in a study directory."""
    # Create multiple participant directories
    for i in [1011, 1012, 1013]:
        participant_dir = tmp_path / f"#{i}"
        participant_dir.mkdir()

        for knee_side in ["Left", "Right"]:
            knee_dir = participant_dir / f"{knee_side} Knee"
            knee_dir.mkdir()
            maneuver_dir = knee_dir / "Walking"
            maneuver_dir.mkdir()

            # Create some outputs
            outputs_dir = maneuver_dir / "audio_outputs"
            outputs_dir.mkdir()
            (outputs_dir / "audio.pkl").write_text("fake data")

            synced_dir = maneuver_dir / "Synced"
            synced_dir.mkdir()
            (synced_dir / "synced.pkl").write_text("fake synced")

    # Clean the entire study directory
    cleanup_study_directory(tmp_path, dry_run=False)

    # Verify all outputs are gone
    for i in [1011, 1012, 1013]:
        participant_dir = tmp_path / f"#{i}"
        for knee_side in ["Left", "Right"]:
            knee_dir = participant_dir / f"{knee_side} Knee"
            maneuver_dir = knee_dir / "Walking"
            assert not (maneuver_dir / "audio_outputs").exists()
            assert not (maneuver_dir / "Synced").exists()


def test_cleanup_study_directory_with_limit(tmp_path):
    """Test cleaning study directory with participant limit."""
    # Create 3 participant directories
    for i in [1011, 1012, 1013]:
        participant_dir = tmp_path / f"#{i}"
        participant_dir.mkdir()
        knee_dir = participant_dir / "Left Knee"
        knee_dir.mkdir()
        maneuver_dir = knee_dir / "Walking"
        maneuver_dir.mkdir()

        outputs_dir = maneuver_dir / "audio_outputs"
        outputs_dir.mkdir()
        (outputs_dir / "audio.pkl").write_text("fake data")

    # Clean only first 2 participants
    cleanup_study_directory(tmp_path, dry_run=False, limit=2)

    # First two should be cleaned
    assert not (tmp_path / "#1011" / "Left Knee" / "Walking" / "audio_outputs").exists()
    assert not (tmp_path / "#1012" / "Left Knee" / "Walking" / "audio_outputs").exists()

    # Third should still have outputs
    assert (tmp_path / "#1013" / "Left Knee" / "Walking" / "audio_outputs").exists()


def test_cleanup_handles_missing_knee_directories(tmp_path):
    """Test cleanup handles participants with missing knee directories gracefully."""
    participant_dir = tmp_path / "#1011"
    participant_dir.mkdir()

    # Only create Left Knee, no Right Knee
    left_knee = participant_dir / "Left Knee"
    left_knee.mkdir()
    maneuver_dir = left_knee / "Walking"
    maneuver_dir.mkdir()

    outputs_dir = maneuver_dir / "audio_outputs"
    outputs_dir.mkdir()
    (outputs_dir / "audio.pkl").write_text("fake data")

    # Should not raise an error
    stats = cleanup_participant_outputs(participant_dir, dry_run=False)

    # Should have removed the one outputs directory
    assert stats["outputs_dirs"] == 1
    assert not outputs_dir.exists()


def test_cleanup_calculates_bytes_freed(participant_with_outputs):
    """Test that cleanup calculates total bytes freed."""
    stats = cleanup_participant_outputs(participant_with_outputs, dry_run=False)

    # Should have counted some bytes
    assert stats["total_bytes"] > 0
