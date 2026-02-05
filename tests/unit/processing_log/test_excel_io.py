"""DB-backed report generation tests."""

from pathlib import Path

import pandas as pd

from src.orchestration.processing_log import KneeProcessingLog, ManeuverProcessingLog
from src.reports.report_generator import ReportGenerator


def _seed_maneuver_records(
    repository,
    audio_processing_factory,
    biomechanics_import_factory,
    synchronization_factory,
    movement_cycle_factory,
    study_id: int,
    knee: str,
    maneuver: str,
    prefix: str,
):
    audio = audio_processing_factory(
        study="AOA",
        study_id=study_id,
        knee=knee,
        maneuver=maneuver,
        audio_file_name=f"{prefix}_{maneuver}_audio",
    )
    audio_record = repository.save_audio_processing(audio)

    biomech = biomechanics_import_factory(
        study="AOA",
        study_id=study_id,
        knee=knee,
        maneuver=maneuver,
        biomechanics_file=f"{prefix}_{maneuver}_biomech.xlsx",
    )
    biomech_record = repository.save_biomechanics_import(
        biomech,
        audio_processing_id=audio_record.id,
    )

    sync = synchronization_factory(
        study="AOA",
        study_id=study_id,
        audio_processing_id=audio_record.id,
        biomechanics_import_id=biomech_record.id,
        sync_file_name=f"{prefix}_{maneuver}_sync",
    )
    sync_record = repository.save_synchronization(
        sync,
        audio_processing_id=audio_record.id,
        biomechanics_import_id=biomech_record.id,
    )

    cycle = movement_cycle_factory(
        study="AOA",
        study_id=study_id,
        audio_processing_id=audio_record.id,
        biomechanics_import_id=biomech_record.id,
        synchronization_id=sync_record.id,
        cycle_file=f"{prefix}_{maneuver}_cycle_0",
    )
    repository.save_movement_cycle(
        cycle,
        audio_processing_id=audio_record.id,
        biomechanics_import_id=biomech_record.id,
        synchronization_id=sync_record.id,
    )

    return audio_record.id


def test_report_generator_writes_all_sheets(
    db_session,
    repository,
    audio_processing_factory,
    biomechanics_import_factory,
    synchronization_factory,
    movement_cycle_factory,
    tmp_path,
):
    participant_id = _seed_maneuver_records(
        repository,
        audio_processing_factory,
        biomechanics_import_factory,
        synchronization_factory,
        movement_cycle_factory,
        study_id=1011,
        knee="left",
        maneuver="fe",
        prefix="AOA1011",
    )

    report = ReportGenerator(db_session)
    output_path = report.save_to_excel(
        tmp_path / "report.xlsx",
        participant_id=participant_id,
        maneuver="fe",
        knee="left",
    )

    workbook = pd.ExcelFile(output_path)
    assert set(workbook.sheet_names) == {
        "Summary",
        "Audio",
        "Biomechanics",
        "Synchronization",
        "Cycles",
    }


def test_maneuver_processing_log_generates_report(
    db_session,
    repository,
    audio_processing_factory,
    biomechanics_import_factory,
    synchronization_factory,
    movement_cycle_factory,
    tmp_path,
):
    _seed_maneuver_records(
        repository,
        audio_processing_factory,
        biomechanics_import_factory,
        synchronization_factory,
        movement_cycle_factory,
        study_id=1012,
        knee="right",
        maneuver="walk",
        prefix="AOA1012",
    )

    log = ManeuverProcessingLog(
        study_id="AOA1012",
        knee_side="Right",
        maneuver="walk",
        maneuver_directory=tmp_path,
    )
    output_path = log.save_to_excel(session=db_session)
    assert output_path.exists()


def test_knee_processing_log_generates_summary(
    db_session,
    repository,
    audio_processing_factory,
    biomechanics_import_factory,
    synchronization_factory,
    movement_cycle_factory,
    tmp_path,
):
    _seed_maneuver_records(
        repository,
        audio_processing_factory,
        biomechanics_import_factory,
        synchronization_factory,
        movement_cycle_factory,
        study_id=1013,
        knee="left",
        maneuver="walk",
        prefix="AOA1013",
    )
    _seed_maneuver_records(
        repository,
        audio_processing_factory,
        biomechanics_import_factory,
        synchronization_factory,
        movement_cycle_factory,
        study_id=1013,
        knee="left",
        maneuver="fe",
        prefix="AOA1013",
    )

    knee_log = KneeProcessingLog(
        study_id="AOA1013",
        knee_side="Left",
        knee_directory=tmp_path,
    )
    knee_log.update_maneuver_summary("walk", ManeuverProcessingLog(
        study_id="AOA1013",
        knee_side="Left",
        maneuver="walk",
        maneuver_directory=tmp_path,
    ))
    knee_log.update_maneuver_summary("fe", ManeuverProcessingLog(
        study_id="AOA1013",
        knee_side="Left",
        maneuver="fe",
        maneuver_directory=tmp_path,
    ))

    output_path = knee_log.save_to_excel(session=db_session)
    assert output_path.exists()
    summary = pd.read_excel(output_path, sheet_name="Knee Summary")
    assert {"walk", "fe"}.issubset(set(summary["Maneuver"]))
