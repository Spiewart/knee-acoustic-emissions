"""Tests for knee-level master processing logs."""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.metadata import AudioProcessing, BiomechanicsImport, Synchronization
from src.orchestration.processing_log import KneeProcessingLog, ManeuverProcessingLog


def _create_test_audio_processing(**kwargs):
    """Helper to create a test AudioProcessing record with required fields."""
    defaults = {
        "study": "AOA",
        "study_id": 1020,
        "linked_biomechanics": False,
        "biomechanics_file": None,
        "biomechanics_type": None,
        "biomechanics_sync_method": None,
        "biomechanics_sample_rate": None,
        "audio_file_name": "test_audio.bin",
        "device_serial": "TEST001",
        "firmware_version": 1,
        "file_time": datetime.now(),
        "file_size_mb": 10.0,
        "recording_date": datetime.now(),
        "recording_time": datetime.now(),
        "knee": "left",
        "maneuver": "walk",
        "sample_rate": 50000.0,
        "num_channels": 4,
        "mic_1_position": "IPM",
        "mic_2_position": "IPL",
        "mic_3_position": "SPM",
        "mic_4_position": "SPL",
        "processing_date": datetime.now(),
        "processing_status": "success",
        "duration_seconds": 120.0,
        "qc_fail_segments": [],
        "qc_fail_segments_ch1": [],
        "qc_fail_segments_ch2": [],
        "qc_fail_segments_ch3": [],
        "qc_fail_segments_ch4": [],
        "qc_signal_dropout": False,
        "qc_signal_dropout_segments": [],
        "qc_signal_dropout_ch1": False,
        "qc_signal_dropout_segments_ch1": [],
        "qc_signal_dropout_ch2": False,
        "qc_signal_dropout_segments_ch2": [],
        "qc_signal_dropout_ch3": False,
        "qc_signal_dropout_segments_ch3": [],
        "qc_signal_dropout_ch4": False,
        "qc_signal_dropout_segments_ch4": [],
        "qc_artifact": False,
        "qc_artifact_segments": [],
        "qc_artifact_ch1": False,
        "qc_artifact_segments_ch1": [],
        "qc_artifact_ch2": False,
        "qc_artifact_segments_ch2": [],
        "qc_artifact_ch3": False,
        "qc_artifact_segments_ch3": [],
        "qc_artifact_ch4": False,
        "qc_artifact_segments_ch4": [],
    }
    defaults.update(kwargs)
    return AudioProcessing(**defaults)


def _create_test_synchronization(**kwargs):
    """Helper to create a test Synchronization record with required fields."""
    defaults = {
        "study": "AOA",
        "study_id": 1020,
        "linked_biomechanics": True,
        "biomechanics_file": "test_bio.xlsx",
        "biomechanics_type": "Gonio",
        "biomechanics_sync_method": "flick",
        "biomechanics_sample_rate": 100.0,
        "audio_file_name": "test_audio.bin",
        "device_serial": "TEST001",
        "firmware_version": 1,
        "file_time": datetime.now(),
        "file_size_mb": 10.0,
        "recording_date": datetime.now(),
        "recording_time": datetime.now(),
        "knee": "left",
        "maneuver": "walk",
        "sample_rate": 50000.0,
        "num_channels": 4,
        "mic_1_position": "IPM",
        "mic_2_position": "IPL",
        "mic_3_position": "SPM",
        "mic_4_position": "SPL",
        "audio_sync_time": 5.0,
        "bio_left_sync_time": 10.0,
        "bio_right_sync_time": 12.0,
        "sync_offset": 5.0,
        "aligned_audio_sync_time": 10.0,
        "aligned_biomechanics_sync_time": 10.0,
        "sync_method": "consensus",
        "consensus_methods": "consensus",
        "consensus_time": 5.0,
        "rms_time": 5.0,
        "onset_time": 5.0,
        "freq_time": 5.0,
        "sync_file_name": "test_sync.pkl",
        "processing_date": datetime.now(),
        "processing_status": "success",
        "sync_duration": 120.0,
        "total_cycles_extracted": 10,
        "clean_cycles": 8,
        "outlier_cycles": 2,
        "mean_cycle_duration_s": 1.2,
        "median_cycle_duration_s": 1.2,
        "min_cycle_duration_s": 1.0,
        "max_cycle_duration_s": 1.5,
        "mean_acoustic_auc": 0.8,
        "pass_number": 1,
        "speed": "normal",
    }
    defaults.update(kwargs)
    return Synchronization(**defaults)


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
    audio = _create_test_audio_processing(
        audio_file_name="test_audio.bin",
        processing_status="success",
        sample_rate=50000.0,
        duration_seconds=120.0,
    )
    log.audio_record = audio

    # Add biomechanics record
    bio = BiomechanicsImport(
        study="AOA",
        study_id=1020,
        biomechanics_file="test_bio.xlsx",
        sheet_name="test_sheet",
        processing_date=datetime.now(),
        processing_status="success",
        num_sub_recordings=3,
        duration_seconds=100.0,
        sample_rate=100.0,
        num_data_points=10000,
    )
    log.biomechanics_record = bio

    # Add synchronization records with stomp times
    for i in range(3):
        sync = _create_test_synchronization(
            audio_file_name="test_audio.bin",
            sync_file_name=f"sync_pass_{i}",
            pass_number=i+1,
            speed="slow" if i == 0 else "medium" if i == 1 else "fast",
            processing_status="success",
            audio_sync_time=5.0 + i,
            bio_left_sync_time=10.0 + i,
            bio_right_sync_time=12.0 + i,
            sync_offset=5.0,
            aligned_audio_sync_time=10.0 + i,
            aligned_biomechanics_sync_time=10.0 + i,
            sync_duration=120.0,
        )
        log.synchronization_records.append(sync)

    # Add movement cycles records (reusing Synchronization for movement cycle metadata)
    for i in range(3):
        cycles = _create_test_synchronization(
            audio_file_name="test_audio.bin",
            sync_file_name=f"sync_pass_{i}",
            pass_number=i+1,
            speed="slow" if i == 0 else "medium" if i == 1 else "fast",
            processing_status="success",
            sync_duration=120.0,
            total_cycles_extracted=10 + i,
            clean_cycles=8 + i,
            outlier_cycles=2,
        )
        log.movement_cycles_records.append(cycles)

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
    # Use unified metadata classes
    sts_audio = _create_test_audio_processing(
        audio_file_name="sts_audio.bin",
        processing_status="success",
        maneuver="sts",
    )
    sts_log.audio_record = sts_audio
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
    extra_cycles = _create_test_synchronization(
        sync_file_name="extra",
        processing_status="success",
        total_cycles_extracted=5,
        clean_cycles=5,
        outlier_cycles=0,
    )
    sample_maneuver_log.movement_cycles_records.append(extra_cycles)

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
    sync = _create_test_synchronization(
        sync_file_name="sync_pass_1",
        processing_status="success",
        audio_sync_time=5.0,
        bio_left_sync_time=10.0,
        bio_right_sync_time=8.0,  # Different from left
        knee="right",
        sync_offset=3.0,  # 8 - 5
        aligned_audio_sync_time=8.0,
        aligned_biomechanics_sync_time=8.0,
    )
    log.synchronization_records.append(sync)

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
