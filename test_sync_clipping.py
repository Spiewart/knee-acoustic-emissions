"""Test to demonstrate the audio clipping issue and fix."""

from datetime import timedelta

import numpy as np
import pandas as pd

from sync_audio_with_biomechanics import sync_audio_with_biomechanics


def test_pass_outside_audio_range():
    """Test case where biomechanics pass is completely outside audio recording."""
    # Audio recording: 0-60 seconds (in audio time)
    audio_df = pd.DataFrame({
        'tt': np.arange(0, 60, 0.01),  # 60 seconds at 100 Hz
        'ch1': np.random.randn(6000) * 0.1,
        'ch2': np.random.randn(6000) * 0.1,
        'ch3': np.random.randn(6000) * 0.1,
        'ch4': np.random.randn(6000) * 0.1,
    })

    # Biomechanics: 0-120 seconds (in bio time)
    bio_df = pd.DataFrame({
        'TIME': pd.to_timedelta(np.arange(0, 120, 0.01), unit='s'),
        'bio_metric': np.sin(np.arange(0, 120, 0.01)),
    })

    # Stomp times: audio stomp at 5s (audio time), bio stomp at 10s (bio time)
    # time_difference = 10 - 5 = 5s
    # After adjustment, audio will span 5-65s in bio time coordinates
    audio_stomp_time = timedelta(seconds=5)
    bio_stomp_time = timedelta(seconds=10)

    # Request a pass that's at 70-75s in bio time (outside adjusted audio range of 5-65s)
    bio_start_time = timedelta(seconds=70)
    bio_end_time = timedelta(seconds=75)

    print("Test 1: Pass completely outside audio range")
    print(f"  Audio: 0-60s (audio time) → 5-65s (bio time after adjustment)")
    print(f"  Requested pass: 70-75s (bio time)")
    print(f"  Expected: ValueError (no overlap)")

    try:
        result = sync_audio_with_biomechanics(
            audio_stomp_time=audio_stomp_time,
            bio_stomp_time=bio_stomp_time,
            audio_df=audio_df.copy(),
            bio_df=bio_df.copy(),
            bio_start_time=bio_start_time,
            bio_end_time=bio_end_time,
        )
        print("  ❌ FAIL: Should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ PASS: Correctly raised ValueError")
        print(f"    Error: {str(e)[:100]}...")


def test_pass_partial_overlap():
    """Test case where biomechanics pass partially overlaps with audio."""
    # Audio recording: 0-60 seconds (in audio time)
    audio_df = pd.DataFrame({
        'tt': np.arange(0, 60, 0.01),
        'ch1': np.random.randn(6000) * 0.1,
        'ch2': np.random.randn(6000) * 0.1,
        'ch3': np.random.randn(6000) * 0.1,
        'ch4': np.random.randn(6000) * 0.1,
    })

    # Biomechanics: 0-120 seconds
    bio_df = pd.DataFrame({
        'TIME': pd.to_timedelta(np.arange(0, 120, 0.01), unit='s'),
        'bio_metric': np.sin(np.arange(0, 120, 0.01)),
    })

    # Stomp times: time_difference = 5s
    # Audio spans 5-65s in bio time
    audio_stomp_time = timedelta(seconds=5)
    bio_stomp_time = timedelta(seconds=10)

    # Request pass at 60-70s (bio time) - partially overlaps (60-65s is available)
    bio_start_time = timedelta(seconds=60)
    bio_end_time = timedelta(seconds=70)

    print("\nTest 2: Pass partially overlaps with audio")
    print(f"  Audio: 0-60s (audio time) → 5-65s (bio time)")
    print(f"  Requested pass: 60-70s (bio time)")
    print(f"  Expected: Warning + synced data for overlap region (60-65s)")

    try:
        result = sync_audio_with_biomechanics(
            audio_stomp_time=audio_stomp_time,
            bio_stomp_time=bio_stomp_time,
            audio_df=audio_df.copy(),
            bio_df=bio_df.copy(),
            bio_start_time=bio_start_time,
            bio_end_time=bio_end_time,
        )
        print(f"  ✓ PASS: Created synced DataFrame")
        print(f"    Shape: {result.shape}")
        print(f"    Audio coverage: {result[['ch1', 'ch2', 'ch3', 'ch4']].notna().any(axis=1).sum() / len(result):.1%}")
        print(f"    Time range: [{result['tt'].min()}, {result['tt'].max()}]")
    except ValueError as e:
        print(f"  ❌ FAIL: Unexpected error: {str(e)}")


def test_pass_fully_within_audio():
    """Test case where biomechanics pass is fully within audio range."""
    # Audio recording: 0-60 seconds (in audio time)
    audio_df = pd.DataFrame({
        'tt': np.arange(0, 60, 0.01),
        'ch1': np.random.randn(6000) * 0.1,
        'ch2': np.random.randn(6000) * 0.1,
        'ch3': np.random.randn(6000) * 0.1,
        'ch4': np.random.randn(6000) * 0.1,
    })

    # Biomechanics: 0-120 seconds
    bio_df = pd.DataFrame({
        'TIME': pd.to_timedelta(np.arange(0, 120, 0.01), unit='s'),
        'bio_metric': np.sin(np.arange(0, 120, 0.01)),
    })

    # Stomp times: time_difference = 5s
    # Audio spans 5-65s in bio time
    audio_stomp_time = timedelta(seconds=5)
    bio_stomp_time = timedelta(seconds=10)

    # Request pass at 20-25s (bio time) - fully within audio range
    bio_start_time = timedelta(seconds=20)
    bio_end_time = timedelta(seconds=25)

    print("\nTest 3: Pass fully within audio range")
    print(f"  Audio: 0-60s (audio time) → 5-65s (bio time)")
    print(f"  Requested pass: 20-25s (bio time)")
    print(f"  Expected: Full sync with 100% audio coverage")

    try:
        result = sync_audio_with_biomechanics(
            audio_stomp_time=audio_stomp_time,
            bio_stomp_time=bio_stomp_time,
            audio_df=audio_df.copy(),
            bio_df=bio_df.copy(),
            bio_start_time=bio_start_time,
            bio_end_time=bio_end_time,
        )
        audio_coverage = result[['ch1', 'ch2', 'ch3', 'ch4']].notna().any(axis=1).sum() / len(result)
        print(f"  ✓ PASS: Created synced DataFrame")
        print(f"    Shape: {result.shape}")
        print(f"    Audio coverage: {audio_coverage:.1%}")
        print(f"    Time range: [{result['tt'].min()}, {result['tt'].max()}]")

        if audio_coverage < 0.95:
            print(f"  ⚠ WARNING: Audio coverage unexpectedly low for fully-contained pass")
    except ValueError as e:
        print(f"  ❌ FAIL: Unexpected error: {str(e)}")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Audio Clipping Logic")
    print("=" * 70)
    test_pass_outside_audio_range()
    test_pass_partial_overlap()
    test_pass_fully_within_audio()
    print("=" * 70)
    print("All tests completed!")
    print("=" * 70)
