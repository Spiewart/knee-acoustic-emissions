#!/usr/bin/env python3
"""Quick test of flexion-extension syncing."""

from pathlib import Path

import pandas as pd

from process_biomechanics import import_biomechanics_recordings
from process_participant_directory import _get_foot_from_knee_side, _load_event_data
from sync_audio_with_biomechanics import (
    get_audio_stomp_time,
    get_bio_end_time,
    get_bio_start_time,
    get_stomp_time,
    load_audio_data,
    sync_audio_with_biomechanics,
)


def main():
    """Test flexion-extension syncing."""
    biomechanics_file = Path(
        "/Users/spiewart/kae_signal_processing_ml/sample_project_directory/"
        "#1011/Motion Capture/AOA1011_Biomechanics_Full_Set.xlsx"
    )
    audio_path = Path(
        "/Users/spiewart/kae_signal_processing_ml/sample_project_directory/"
        "#1011/Left Knee/Flexion-Extension/"
        "HP_W11.2-5-20240126_141532_outputs/"
        "HP_W11.2-5-20240126_141532_with_freq.pkl"
    )

    print("Loading data (this may take a moment)...")
    audio_df = load_audio_data(audio_path)
    recordings = import_biomechanics_recordings(
        biomechanics_file, "flexion_extension", None
    )
    bio_df = recordings[0].data

    print(f"  Audio: {len(audio_df)} rows")
    print(f"  Bio: {len(bio_df)} rows")

    print("\nGetting timing information...")
    event_data = _load_event_data(biomechanics_file, "flexion_extension")
    audio_stomp = get_audio_stomp_time(audio_df)
    bio_stomp = get_stomp_time(event_data, _get_foot_from_knee_side("Left"))
    bio_start = get_bio_start_time(event_data, "flexion_extension")
    bio_end = get_bio_end_time(event_data, "flexion_extension")

    print(f"  Audio stomp: {audio_stomp}")
    print(f"  Bio stomp: {bio_stomp}")
    print(f"  Bio start (Movement Start): {bio_start}")
    print(f"  Bio end (Movement End): {bio_end}")

    print("\nSyncing audio with biomechanics (this may take a moment)...")
    result = sync_audio_with_biomechanics(
        audio_stomp_time=audio_stomp,
        bio_stomp_time=bio_stomp,
        audio_df=audio_df.copy(),
        bio_df=bio_df,
        bio_start_time=bio_start,
        bio_end_time=bio_end,
    )

    print(f"\n✓ Sync complete")
    print(f"  Result shape: {result.shape}")

    # Check for NaT values
    valid_rows = result[result['TIME'].notna()]
    invalid_rows = result[result['TIME'].isna()]
    print(f"\nData quality check:")
    print(f"  Total rows: {len(result)}")
    print(f"  Valid TIME rows: {len(valid_rows)} ({100*len(valid_rows)/len(result):.1f}%)")
    print(f"  NaT TIME rows: {len(invalid_rows)}")

    if len(invalid_rows) > 0:
        print(f"\n⚠ WARNING: Found {len(invalid_rows)} rows with NaT!")
        print(f"  This means these audio samples have no matching biomechanics data")
        if len(valid_rows) > 0:
            print(
                f"  Valid TIME range: "
                f"{valid_rows['TIME'].min()} to {valid_rows['TIME'].max()}"
            )
            print(
                f"  Invalid rows TIME ranges from: "
                f"{invalid_rows['tt'].min()} to {invalid_rows['tt'].max()}"
            )
    else:
        print(f"\n✓ Perfect! No NaT values - all audio samples have matching bio data")

    # Check Knee Angle Z
    if len(valid_rows) > 0:
        left_kaz = valid_rows['Left Knee Angle_Z'].astype(float)
        print(f"\nLeft Knee Angle_Z data:")
        print(f"  From {len(left_kaz)} rows with valid biomechanics")
        print(f"  Range: {left_kaz.min():.4f} to {left_kaz.max():.4f}")
        print(f"  First value: {left_kaz.iloc[0]:.4f} at TIME {valid_rows.iloc[0]['TIME']}")
        print(f"  Last value: {left_kaz.iloc[-1]:.4f} at TIME {valid_rows.iloc[-1]['TIME']}")

if __name__ == "__main__":
    main()
