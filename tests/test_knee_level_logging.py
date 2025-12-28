"""Tests for knee-level master processing logs."""

import pandas as pd
import pytest
from pathlib import Path
from datetime import datetime
from src.orchestration.processing_log import (
    KneeProcessingLog,
    ManeuverProcessingLog,
    AudioProcessingRecord,
    BiomechanicsImportRecord,
    SynchronizationRecord,
    MovementCyclesRecord,
)


@pytest.fixture
def temp_knee_dir(tmp_path):
    """Create a temporary knee directory."""
    knee_dir = tmp_path / "Left Knee"
    knee_dir.mkdir(parents=True)
    return knee_dir


@pytest.fixture
def sample_maneuver_log(tmp_path):
    """Create a sample maneuver log for testing."""
    maneuver_dir = tmp_path / "Left Knee" / "Walking"
    maneuver_dir.mkdir(parents=True)

    log = ManeuverProcessingLog(
        study_id="1020",
        knee_side="Left",
        maneuver="walk",
        maneuver_directory=maneuver_dir,
        log_created=datetime.now(),
        log_updated=datetime.now(),
    )

    # Add audio record
    log.audio_record = AudioProcessingRecord(
        audio_file_name="test_audio.bin",
        processing_status="success",
        sample_rate=50000.0,
        duration_seconds=120.0,
    )

    # Add biomechanics record
    log.biomechanics_record = BiomechanicsImportRecord(
        biomechanics_file="test_bio.xlsx",
        processing_status="success",
        num_recordings=3,
        duration_seconds=100.0,
    )

    # Add synchronization records with stomp times
    for i in range(3):
        sync_record = SynchronizationRecord(
            sync_file_name=f"sync_pass_{i}",
            pass_number=i,
            speed="slow" if i == 0 else "medium" if i == 1 else "fast",
            processing_status="success",
            audio_stomp_time=5.0 + i,
            bio_left_stomp_time=10.0 + i,
            bio_right_stomp_time=12.0 + i,
            knee_side="left",
            stomp_offset=5.0,
            aligned_audio_stomp_time=10.0 + i,
            aligned_bio_stomp_time=10.0 + i,
        )
        log.synchronization_records.append(sync_record)

    # Add movement cycles records
    for i in range(3):
        cycles_record = MovementCyclesRecord(
            sync_file_name=f"sync_pass_{i}",
            processing_status="success",
            total_cycles_extracted=10 + i,
            clean_cycles=8 + i,
            outlier_cycles=2,
        )
        log.movement_cycles_records.append(cycles_record)

    return log


def test_knee_log_creation(temp_knee_dir):
    """Test creating a new knee processing log."""
    log = KneeProcessingLog.get_or_create(
        study_id="1020",
        knee_side="Left",
        knee_directory=temp_knee_dir,
    )

    assert log.study_id == "1020"
    assert log.knee_side == "Left"
    assert log.knee_directory == temp_knee_dir
    assert log.log_created is not None
    assert log.log_updated is not None
    assert len(log.maneuver_summaries) == 0


def test_knee_log_update_maneuver_summary(temp_knee_dir, sample_maneuver_log):
    """Test updating maneuver summary in knee log."""
    knee_log = KneeProcessingLog.get_or_create(
        study_id="1020",
        knee_side="Left",
        knee_directory=temp_knee_dir,
    )

    # Update with walking maneuver
    knee_log.update_maneuver_summary("walk", sample_maneuver_log)

    assert len(knee_log.maneuver_summaries) == 1
    summary = knee_log.maneuver_summaries[0]

    assert summary["Maneuver"] == "walk"
    assert summary["Audio Processed"] is True
    assert summary["Biomechanics Imported"] is True
    assert summary["Num Synced Files"] == 3
    assert summary["Num Movement Cycles"] == 33  # 10 + 11 + 12


def test_knee_log_stomp_time_aggregation(temp_knee_dir, sample_maneuver_log):
    """Test that knee log correctly aggregates stomp times."""
    knee_log = KneeProcessingLog.get_or_create(
        study_id="1020",
        knee_side="Left",
        knee_directory=temp_knee_dir,
    )

    knee_log.update_maneuver_summary("walk", sample_maneuver_log)
    summary = knee_log.maneuver_summaries[0]

    # Representative audio stomp: (5 + 6 + 7) / 3 = 6
    assert summary["Audio Stomp (s)"] == pytest.approx(6.0)
    # Representative bio stomp (left knee): (10 + 11 + 12) / 3 = 11
    assert summary["Bio Stomp (s)"] == pytest.approx(11.0)
    # Representative offset: 5.0 for all
    assert summary["Stomp Offset (s)"] == pytest.approx(5.0)
    # Representative aligned audio: (10 + 11 + 12) / 3 = 11
    assert summary["Aligned Audio Stomp (s)"] == pytest.approx(11.0)
    # Representative aligned bio: (10 + 11 + 12) / 3 = 11
    assert summary["Aligned Bio Stomp (s)"] == pytest.approx(11.0)


def test_knee_log_multiple_maneuvers(temp_knee_dir, sample_maneuver_log, tmp_path):
    """Test knee log with multiple maneuvers."""
    knee_log = KneeProcessingLog.get_or_create(
        study_id="1020",
        knee_side="Left",
        knee_directory=temp_knee_dir,
    )

    # Add walking
    knee_log.update_maneuver_summary("walk", sample_maneuver_log)

    # Create and add sit-to-stand
    sts_dir = tmp_path / "Left Knee" / "Sit-Stand"
    sts_dir.mkdir(parents=True)
    sts_log = ManeuverProcessingLog(
        study_id="1020",
        knee_side="Left",
        maneuver="sit_to_stand",
        maneuver_directory=sts_dir,
        log_created=datetime.now(),
        log_updated=datetime.now(),
    )
    sts_log.audio_record = AudioProcessingRecord(
        audio_file_name="sts_audio.bin",
        processing_status="success",
    )
    knee_log.update_maneuver_summary("sit_to_stand", sts_log)

    assert len(knee_log.maneuver_summaries) == 2
    maneuvers = [s["Maneuver"] for s in knee_log.maneuver_summaries]
    assert "walk" in maneuvers
    assert "sit_to_stand" in maneuvers


def test_knee_log_replaces_existing_maneuver(temp_knee_dir, sample_maneuver_log):
    """Test that updating same maneuver replaces old summary."""
    knee_log = KneeProcessingLog.get_or_create(
        study_id="1020",
        knee_side="Left",
        knee_directory=temp_knee_dir,
    )

    # Add walking first time
    knee_log.update_maneuver_summary("walk", sample_maneuver_log)
    first_update = knee_log.maneuver_summaries[0]["Last Updated"]

    # Modify the log
    sample_maneuver_log.movement_cycles_records.append(
        MovementCyclesRecord(
            sync_file_name="extra",
            processing_status="success",
            total_cycles_extracted=5,
            clean_cycles=5,
            outlier_cycles=0,
        )
    )

    # Update again
    knee_log.update_maneuver_summary("walk", sample_maneuver_log)

    # Should still have only one walking summary
    assert len(knee_log.maneuver_summaries) == 1
    assert knee_log.maneuver_summaries[0]["Maneuver"] == "walk"
    # Cycles count should be updated: 33 + 5 = 38
    assert knee_log.maneuver_summaries[0]["Num Movement Cycles"] == 38
    # Update time should be later
    assert knee_log.maneuver_summaries[0]["Last Updated"] >= first_update


def test_knee_log_save_and_load(temp_knee_dir, sample_maneuver_log):
    """Test saving and loading knee log from Excel."""
    knee_log = KneeProcessingLog.get_or_create(
        study_id="1020",
        knee_side="Left",
        knee_directory=temp_knee_dir,
    )

    knee_log.update_maneuver_summary("walk", sample_maneuver_log)

    # Save to Excel
    filepath = knee_log.save_to_excel()
    assert filepath.exists()

    # Load from Excel
    loaded_log = KneeProcessingLog.load_from_excel(filepath)

    assert loaded_log is not None
    assert loaded_log.study_id == "1020"
    assert loaded_log.knee_side == "Left"
    assert len(loaded_log.maneuver_summaries) == 1

    summary = loaded_log.maneuver_summaries[0]
    assert summary["Maneuver"] == "walk"
    assert summary["Num Synced Files"] == 3
    assert summary["Num Movement Cycles"] == 33
    assert summary["Audio Stomp (s)"] == pytest.approx(6.0)
    assert summary["Stomp Offset (s)"] == pytest.approx(5.0)


def test_knee_log_no_stomp_times(temp_knee_dir, tmp_path):
    """Test knee log when maneuver has no synchronization records."""
    maneuver_dir = tmp_path / "Left Knee" / "Walking"
    maneuver_dir.mkdir(parents=True)

    log = ManeuverProcessingLog(
        study_id="1020",
        knee_side="Left",
        maneuver="walk",
        maneuver_directory=maneuver_dir,
        log_created=datetime.now(),
        log_updated=datetime.now(),
    )

    # No sync records added

    knee_log = KneeProcessingLog.get_or_create(
        study_id="1020",
        knee_side="Left",
        knee_directory=temp_knee_dir,
    )

    knee_log.update_maneuver_summary("walk", log)
    summary = knee_log.maneuver_summaries[0]

    # Stomp time fields should be None
    assert summary.get("Audio Stomp (s)") is None
    assert summary.get("Bio Stomp (s)") is None
    assert summary.get("Stomp Offset (s)") is None


def test_knee_log_right_knee_stomp_selection(temp_knee_dir, tmp_path):
    """Test that right knee log uses right stomp times."""
    right_knee_dir = tmp_path / "Right Knee"
    right_knee_dir.mkdir(parents=True)

    maneuver_dir = right_knee_dir / "Walking"
    maneuver_dir.mkdir(parents=True)

    log = ManeuverProcessingLog(
        study_id="1020",
        knee_side="Right",
        maneuver="walk",
        maneuver_directory=maneuver_dir,
        log_created=datetime.now(),
        log_updated=datetime.now(),
    )

    # Add sync record with right knee
    sync_record = SynchronizationRecord(
        sync_file_name="sync_pass_1",
        processing_status="success",
        audio_stomp_time=5.0,
        bio_left_stomp_time=10.0,
        bio_right_stomp_time=8.0,  # Different from left
        knee_side="right",
        stomp_offset=3.0,  # 8 - 5
        aligned_audio_stomp_time=8.0,
        aligned_bio_stomp_time=8.0,
    )
    log.synchronization_records.append(sync_record)

    knee_log = KneeProcessingLog.get_or_create(
        study_id="1020",
        knee_side="Right",
        knee_directory=right_knee_dir,
    )

    knee_log.update_maneuver_summary("walk", log)
    summary = knee_log.maneuver_summaries[0]

    # Should use right stomp time (8.0) not left (10.0)
    assert summary["Bio Stomp (s)"] == pytest.approx(8.0)
    assert summary["Stomp Offset (s)"] == pytest.approx(3.0)
