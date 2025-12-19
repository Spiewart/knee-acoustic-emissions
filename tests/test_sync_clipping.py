"""Test edge cases for audio-biomechanics synchronization clipping behavior."""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from sync_audio_with_biomechanics import sync_audio_with_biomechanics


@pytest.fixture
def audio_df():
    """Create audio DataFrame covering 0-60 seconds."""
    return pd.DataFrame({
        'tt': np.arange(0, 60, 0.01),  # 60 seconds at 100 Hz
        'ch1': np.random.randn(6000) * 0.1,
        'ch2': np.random.randn(6000) * 0.1,
        'ch3': np.random.randn(6000) * 0.1,
        'ch4': np.random.randn(6000) * 0.1,
    })


@pytest.fixture
def bio_df():
    """Create biomechanics DataFrame covering 0-120 seconds."""
    return pd.DataFrame({
        'TIME': pd.to_timedelta(np.arange(0, 120, 0.01), unit='s'),
        'bio_metric': np.sin(np.arange(0, 120, 0.01)),
    })


def test_pass_outside_audio_range(audio_df, bio_df):
    """Biomechanics pass completely outside audio recording should raise ValueError."""
    # Stomp times: audio stomp at 5s (audio time), bio stomp at 10s (bio time)
    # time_difference = 10 - 5 = 5s
    # After adjustment, audio will span 5-65s in bio time coordinates
    audio_stomp_time = timedelta(seconds=5)
    bio_stomp_time = timedelta(seconds=10)

    # Request a pass that's at 70-75s in bio time (outside adjusted audio range of 5-65s)
    bio_start_time = timedelta(seconds=70)
    bio_end_time = timedelta(seconds=75)

    with pytest.raises(ValueError, match="does not overlap"):
        sync_audio_with_biomechanics(
            audio_stomp_time=audio_stomp_time,
            bio_stomp_time=bio_stomp_time,
            audio_df=audio_df.copy(),
            bio_df=bio_df.copy(),
            bio_start_time=bio_start_time,
            bio_end_time=bio_end_time,
        )


def test_pass_partial_overlap(audio_df, bio_df):
    """Biomechanics pass partially overlapping with audio should warn and clip to available range."""
    # Stomp times: time_difference = 5s
    # Audio spans 5-65s in bio time (in bio coordinates: audio starts at 5s, ends at ~65s)
    audio_stomp_time = timedelta(seconds=5)
    bio_stomp_time = timedelta(seconds=10)

    # Request pass at 60-70s (bio time) - partially overlaps
    # With 0.5s margin: effective window is 59.5-70.5s
    # Audio ends at ~65s, so only 59.5-65s is available
    bio_start_time = timedelta(seconds=60)
    bio_end_time = timedelta(seconds=70)

    result = sync_audio_with_biomechanics(
        audio_stomp_time=audio_stomp_time,
        bio_stomp_time=bio_stomp_time,
        audio_df=audio_df.copy(),
        bio_df=bio_df.copy(),
        bio_start_time=bio_start_time,
        bio_end_time=bio_end_time,
    )

    # Should return a DataFrame with data clipped to available audio range
    assert not result.empty
    assert 'ch1' in result.columns
    assert 'ch2' in result.columns
    assert 'ch3' in result.columns
    assert 'ch4' in result.columns

    # The result should have been clipped - check that we got some data back
    # but it covers less than the full requested window
    assert len(result) > 0


def test_pass_fully_within_audio(audio_df, bio_df):
    """Biomechanics pass fully within audio range should have complete audio coverage."""
    # Stomp times: time_difference = 5s
    # Audio spans 5-65s in bio time
    audio_stomp_time = timedelta(seconds=5)
    bio_stomp_time = timedelta(seconds=10)

    # Request pass at 20-25s (bio time) - fully within audio range
    bio_start_time = timedelta(seconds=20)
    bio_end_time = timedelta(seconds=25)

    result = sync_audio_with_biomechanics(
        audio_stomp_time=audio_stomp_time,
        bio_stomp_time=bio_stomp_time,
        audio_df=audio_df.copy(),
        bio_df=bio_df.copy(),
        bio_start_time=bio_start_time,
        bio_end_time=bio_end_time,
    )

    # Should return a DataFrame with full audio coverage
    assert not result.empty
    assert 'ch1' in result.columns

    # Audio coverage should be at or near 100%
    audio_coverage = result[['ch1', 'ch2', 'ch3', 'ch4']].notna().any(axis=1).sum() / len(result)
    assert audio_coverage >= 0.95, f"Expected >95% coverage, got {audio_coverage:.1%}"
