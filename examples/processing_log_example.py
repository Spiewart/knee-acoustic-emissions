#!/usr/bin/env python3
"""Example demonstrating the processing log system.

This example shows how processing logs are created and updated during
participant processing. The logs are automatically created/updated when
processing is run.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.orchestration.processing_log import (
    AudioProcessingRecord,
    BiomechanicsImportRecord,
    ManeuverProcessingLog,
    MovementCyclesRecord,
    SynchronizationRecord,
    create_audio_record_from_data,
    create_biomechanics_record_from_data,
    create_cycles_record_from_data,
    create_sync_record_from_data,
)


def create_example_log():
    """Create an example processing log with sample data."""

    # Create a processing log for a participant
    log = ManeuverProcessingLog(
        study_id="1011",
        knee_side="Left",
        maneuver="walk",
        maneuver_directory=Path("/path/to/participant/#1011/Left Knee/Walking"),
        log_created=datetime.now(),
        log_updated=datetime.now(),
    )

    # Add audio processing record
    audio_record = AudioProcessingRecord(
        audio_file_name="audio_recording_with_freq",
        audio_bin_file="audio_recording.bin",
        audio_pkl_file="audio_recording_with_freq.pkl",
        processing_date=datetime.now(),
        processing_status="success",
        sample_rate=46875.0,
        num_channels=4,
        duration_seconds=120.5,
        file_size_mb=45.2,
        device_serial="ABC123",
        firmware_version=2,
        channel_1_rms=150.3,
        channel_2_rms=148.7,
        channel_3_rms=152.1,
        channel_4_rms=149.8,
        channel_1_peak=850.2,
        channel_2_peak=820.5,
        channel_3_peak=875.3,
        channel_4_peak=840.1,
        has_instantaneous_freq=True,
    )
    log.update_audio_record(audio_record)

    # Add biomechanics import record
    bio_record = BiomechanicsImportRecord(
        biomechanics_file="AOA1011_Biomechanics_Full_Set.xlsx",
        sheet_name="Walk0001",
        processing_date=datetime.now(),
        processing_status="success",
        num_recordings=3,
        num_passes=9,
        duration_seconds=115.0,
        num_data_points=11500,
        sample_rate=100.0,
        start_time=5.2,
        end_time=120.2,
    )
    log.update_biomechanics_record(bio_record)

    # Add synchronization records (one for each pass/speed combination)
    for pass_num in range(1, 4):
        for speed in ["slow", "medium", "fast"]:
            sync_record = SynchronizationRecord(
                sync_file_name=f"left_walk_{speed}_pass{pass_num}",
                pass_number=pass_num,
                speed=speed,
                processing_date=datetime.now(),
                processing_status="success",
                audio_stomp_time=10.5 + pass_num,
                bio_left_stomp_time=5.2 + pass_num,
                bio_right_stomp_time=5.1 + pass_num,
                knee_side="left",
                num_synced_samples=8000,
                duration_seconds=8.0,
                sync_qc_performed=True,
                sync_qc_passed=True,
            )
            log.add_synchronization_record(sync_record)

    # Add movement cycles records
    for pass_num in range(1, 4):
        for speed in ["slow", "medium", "fast"]:
            cycles_record = MovementCyclesRecord(
                sync_file_name=f"left_walk_{speed}_pass{pass_num}",
                processing_date=datetime.now(),
                processing_status="success",
                total_cycles_extracted=12,
                clean_cycles=10,
                outlier_cycles=2,
                acoustic_threshold=100.0,
                output_directory="MovementCycles",
                plots_created=True,
            )
            log.add_movement_cycles_record(cycles_record)

    return log


def demonstrate_helper_functions():
    """Demonstrate using helper functions to create records from data."""

    print("\n" + "="*60)
    print("Demonstrating helper functions for record creation")
    print("="*60)

    # Create a sample audio DataFrame
    n_samples = 1000
    sample_rate = 46875.0
    audio_df = pd.DataFrame({
        'tt': pd.date_range('2024-01-01', periods=n_samples, freq='21.333us'),
        'ch1': np.random.randn(n_samples) * 150,
        'ch2': np.random.randn(n_samples) * 148,
        'ch3': np.random.randn(n_samples) * 152,
        'ch4': np.random.randn(n_samples) * 149,
        'f_ch1': np.random.randn(n_samples) * 50,
        'f_ch2': np.random.randn(n_samples) * 48,
        'f_ch3': np.random.randn(n_samples) * 52,
        'f_ch4': np.random.randn(n_samples) * 49,
    })

    metadata = {
        'fs': sample_rate,
        'devFirmwareVersion': 2,
        'deviceSerial': '123456',
        'fileTime': datetime.now(),
    }

    # Create audio record from data
    audio_record = create_audio_record_from_data(
        audio_file_name="test_audio_with_freq",
        audio_df=audio_df,
        audio_bin_path=Path("test_audio.bin"),
        audio_pkl_path=Path("test_audio_with_freq.pkl"),
        metadata=metadata,
    )

    print(f"\nAudio Record Created:")
    print(f"  Status: {audio_record.processing_status}")
    print(f"  Sample Rate: {audio_record.sample_rate} Hz")
    print(f"  Duration: {audio_record.duration_seconds:.2f} s")
    print(f"  Ch1 RMS: {audio_record.channel_1_rms:.2f}")
    print(f"  Has Inst. Freq: {audio_record.has_instantaneous_freq}")

    # Create a sample synced DataFrame
    synced_df = pd.DataFrame({
        'tt': pd.date_range('2024-01-01', periods=500, freq='10ms'),
        'ch1': np.random.randn(500) * 150,
        'Knee Angle Z': np.sin(np.linspace(0, 4*np.pi, 500)) * 30 + 40,
    })

    # Create sync record from data
    sync_record = create_sync_record_from_data(
        sync_file_name="left_walk_medium_pass1",
        synced_df=synced_df,
        audio_stomp_time=10.5,
        bio_left_stomp_time=5.2,
        bio_right_stomp_time=5.1,
        knee_side="left",
        pass_number=1,
        speed="medium",
    )

    print(f"\nSync Record Created:")
    print(f"  Status: {sync_record.processing_status}")
    print(f"  Num Samples: {sync_record.num_synced_samples}")
    print(f"  Duration: {sync_record.duration_seconds:.2f} s")
    print(f"  Audio Stomp: {sync_record.audio_stomp_time} s")


def main():
    """Main demonstration."""

    print("="*60)
    print("Processing Log System Demonstration")
    print("="*60)

    # Create example log
    print("\nCreating example processing log...")
    log = create_example_log()

    # Display summary
    print(f"\nLog Summary:")
    print(f"  Study ID: {log.study_id}")
    print(f"  Knee: {log.knee_side}")
    print(f"  Maneuver: {log.maneuver}")
    print(f"  Audio Status: {log.audio_record.processing_status if log.audio_record else 'N/A'}")
    print(f"  Biomechanics Status: {log.biomechanics_record.processing_status if log.biomechanics_record else 'N/A'}")
    print(f"  Sync Records: {len(log.synchronization_records)}")
    print(f"  Cycles Records: {len(log.movement_cycles_records)}")

    # Save to Excel
    output_path = Path("example_processing_log.xlsx")
    print(f"\nSaving log to {output_path}...")
    log.save_to_excel(output_path)
    print(f"✓ Log saved successfully!")

    # Load the log back
    print(f"\nLoading log from {output_path}...")
    loaded_log = ManeuverProcessingLog.load_from_excel(output_path)

    if loaded_log:
        print(f"✓ Log loaded successfully!")
        print(f"  Loaded {len(loaded_log.synchronization_records)} sync records")
        print(f"  Loaded {len(loaded_log.movement_cycles_records)} cycles records")
    else:
        print("✗ Failed to load log")

    # Demonstrate helper functions
    demonstrate_helper_functions()

    print("\n" + "="*60)
    print("Demonstration complete!")
    print("="*60)
    print("\nIn practice, processing logs are created automatically during")
    print("participant processing. The log is saved as an Excel file in each")
    print("maneuver directory (e.g., Left Knee/Walking/) and contains:")
    print("  - Summary sheet with overall statistics")
    print("  - Audio processing details (channels, QC, etc.)")
    print("  - Biomechanics import details")
    print("  - Synchronization details for each file")
    print("  - Movement cycle extraction results")
    print("\nWhen re-processing, the log is updated incrementally to reflect")
    print("only the stages that were re-run.")


if __name__ == "__main__":
    main()
