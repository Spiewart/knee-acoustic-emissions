import types

import pytest

from src.orchestration.processing_log import ManeuverProcessingLog


@pytest.fixture
def log_base(tmp_path):
    return ManeuverProcessingLog(
        study_id="P001",
        knee_side="Left",
        maneuver="walk",
        maneuver_directory=tmp_path,
    )


def test_summary_helpers_walk(monkeypatch, log_base):
    # Arrange
    log = log_base
    log.audio_record = types.SimpleNamespace(processing_status="success")
    log.biomechanics_record = types.SimpleNamespace(processing_status="success")
    log.synchronization_records = [
        types.SimpleNamespace(sync_file_name="sync1.csv"),
        types.SimpleNamespace(sync_file_name="sync2.csv"),
    ]
    log.movement_cycles_records = [
        types.SimpleNamespace(total_cycles_extracted=10),
        types.SimpleNamespace(total_cycles_extracted=5),
    ]
    monkeypatch.setattr(log, "_check_processed_audio_exists", lambda: False)

    # Act
    row = log.build_summary_row()

    # Assert
    assert row["Audio Processed"] is True
    assert row["Biomechanics Imported"] is True
    assert row["Num Synced Files"] == 2
    assert row["Num Movement Cycles"] == 15


def test_audio_processed_uses_disk_fallback(monkeypatch, log_base):
    log = log_base
    log.audio_record = None
    monkeypatch.setattr(log, "_check_processed_audio_exists", lambda: True)

    assert log.is_audio_processed() is True


def test_summary_helpers_non_walk(monkeypatch, tmp_path):
    log = ManeuverProcessingLog(
        study_id="P002",
        knee_side="Right",
        maneuver="sit_to_stand",
        maneuver_directory=tmp_path,
    )
    log.audio_record = types.SimpleNamespace(processing_status="success")
    log.biomechanics_record = types.SimpleNamespace(processing_status="success")
    log.synchronization_records = [types.SimpleNamespace(sync_file_name="sync.csv")]
    log.movement_cycles_records = [types.SimpleNamespace(total_cycles_extracted=3)]
    monkeypatch.setattr(log, "_check_processed_audio_exists", lambda: False)

    row = log.build_summary_row()

    assert row["Audio Processed"] is True
    assert row["Biomechanics Imported"] is True
    assert row["Num Synced Files"] == 1
    assert row["Num Movement Cycles"] == 3
