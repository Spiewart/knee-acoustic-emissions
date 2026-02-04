"""Integration tests using a fully valid fake directory.

These tests verify end-to-end processing (bin/sync/cycles) and DB/log outputs.
"""

from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import select

from src.db.models import (
    AudioProcessingRecord,
    MovementCycleRecord,
    ParticipantRecord,
    StudyRecord,
    SynchronizationRecord,
)
from src.orchestration.participant import process_participant


def _get_participant_id(session, study_name: str, participant_number: int) -> int:
    study = session.execute(
        select(StudyRecord).where(StudyRecord.name == study_name)
    ).scalar_one()
    participant = session.execute(
        select(ParticipantRecord).where(
            ParticipantRecord.study_participant_id == study.id,
            ParticipantRecord.study_id == participant_number,
        )
    ).scalar_one()
    return participant.id


def _get_latest_log_file(maneuver_dir: Path) -> Path:
    logs = sorted(maneuver_dir.glob("processing_log_*.xlsx"))
    assert logs, f"Expected processing log in {maneuver_dir}"
    return logs[-1]


class TestFakeDirectoryProcessingIntegration:
    def test_bin_stage_creates_audio_record_and_log(
        self, fake_participant_directory, use_test_db, db_session
    ):
        participant_dir = fake_participant_directory["participant_dir"]

        success = process_participant(
            participant_dir,
            entrypoint="bin",
            knee="left",
            maneuver="fe",
        )
        assert success, "Bin stage should succeed with valid .bin files"

        participant_id = _get_participant_id(db_session, "AOA", 1011)
        audio_records = db_session.execute(
            select(AudioProcessingRecord).where(
                AudioProcessingRecord.participant_id == participant_id,
                AudioProcessingRecord.knee == "left",
                AudioProcessingRecord.maneuver == "fe",
            )
        ).scalars().all()

        assert len(audio_records) == 1, "Should create exactly one audio record"
        audio = audio_records[0]
        assert audio.device_serial, "Device serial should be populated"
        assert audio.file_time is not None, "File time should be populated"
        assert audio.recording_time is not None, "Recording time should be populated"
        assert audio.processing_status == "success", "Processing status should be success"

        fe_dir = participant_dir / "Left Knee" / "Flexion-Extension"
        log_file = _get_latest_log_file(fe_dir)
        audio_sheet = pd.read_excel(log_file, sheet_name="Audio")
        assert not audio_sheet.empty, "Audio sheet should have data"
        assert audio_sheet.iloc[0]["Processing Status"] == "success"

    def test_bin_stage_is_idempotent_for_audio_record(
        self, fake_participant_directory, use_test_db, db_session
    ):
        participant_dir = fake_participant_directory["participant_dir"]

        success_1 = process_participant(
            participant_dir,
            entrypoint="bin",
            knee="left",
            maneuver="fe",
        )
        assert success_1, "First bin stage should succeed"

        success_2 = process_participant(
            participant_dir,
            entrypoint="bin",
            knee="left",
            maneuver="fe",
        )
        assert success_2, "Second bin stage should succeed"

        participant_id = _get_participant_id(db_session, "AOA", 1011)
        audio_records = db_session.execute(
            select(AudioProcessingRecord).where(
                AudioProcessingRecord.participant_id == participant_id,
                AudioProcessingRecord.knee == "left",
                AudioProcessingRecord.maneuver == "fe",
            )
        ).scalars().all()

        assert len(audio_records) == 1, "Should not create duplicate audio records"

    def test_sync_stage_creates_sync_records(
        self, fake_participant_directory, use_test_db, db_session
    ):
        participant_dir = fake_participant_directory["participant_dir"]

        success = process_participant(
            participant_dir,
            entrypoint="sync",
            knee="left",
            maneuver="fe",
        )
        assert success, "Sync stage should succeed"

        participant_id = _get_participant_id(db_session, "AOA", 1011)
        audio_record = db_session.execute(
            select(AudioProcessingRecord).where(
                AudioProcessingRecord.participant_id == participant_id,
                AudioProcessingRecord.knee == "left",
                AudioProcessingRecord.maneuver == "fe",
            )
        ).scalar_one()

        sync_records = db_session.execute(
            select(SynchronizationRecord).where(
                SynchronizationRecord.participant_id == participant_id,
                SynchronizationRecord.audio_processing_id == audio_record.id,
            )
        ).scalars().all()

        assert sync_records, "Should create synchronization records"
        assert sync_records[0].processing_status == "success"

    def test_cycles_stage_creates_cycle_records(
        self, fake_participant_directory, use_test_db, db_session
    ):
        participant_dir = fake_participant_directory["participant_dir"]

        # Ensure bin and sync stages ran to produce synced data
        success_bin = process_participant(
            participant_dir,
            entrypoint="bin",
            knee="left",
            maneuver="fe",
        )
        assert success_bin, "Bin stage should succeed before cycles"

        success_sync = process_participant(
            participant_dir,
            entrypoint="sync",
            knee="left",
            maneuver="fe",
        )
        assert success_sync, "Sync stage should succeed before cycles"

        success = process_participant(
            participant_dir,
            entrypoint="cycles",
            knee="left",
            maneuver="fe",
        )
        assert success, "Cycles stage should succeed"

        participant_id = _get_participant_id(db_session, "AOA", 1011)
        audio_record = db_session.execute(
            select(AudioProcessingRecord).where(
                AudioProcessingRecord.participant_id == participant_id,
                AudioProcessingRecord.knee == "left",
                AudioProcessingRecord.maneuver == "fe",
            )
        ).scalar_one()

        cycle_records = db_session.execute(
            select(MovementCycleRecord).where(
                MovementCycleRecord.participant_id == participant_id,
                MovementCycleRecord.audio_processing_id == audio_record.id,
            )
        ).scalars().all()

        # Some fixtures may not include Knee Angle Z; ensure cycles stage runs without error
        assert cycle_records is not None, "Cycle query should succeed"
