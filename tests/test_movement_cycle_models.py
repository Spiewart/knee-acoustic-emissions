"""Test suite for FullMovementCycleMetadata and MovementCycle models."""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.models import (
    MicrophonePosition,
    MovementCycle,
    FullMovementCycleMetadata,
    SynchronizedData,
)


@pytest.fixture
def microphones() -> dict[int, MicrophonePosition]:
    """Provide a full microphone map covering required keys 1-4."""

    return {
        1: MicrophonePosition(patellar_position="Infrapatellar", laterality="Lateral"),
        2: MicrophonePosition(patellar_position="Infrapatellar", laterality="Medial"),
        3: MicrophonePosition(patellar_position="Suprapatellar", laterality="Medial"),
        4: MicrophonePosition(patellar_position="Suprapatellar", laterality="Lateral"),
    }


@pytest.fixture
def base_kwargs(microphones: dict[int, MicrophonePosition]) -> dict:
    """Base keyword args for FullMovementCycleMetadata creation."""

    return {
        "scripted_maneuver": "walk",
        "speed": "medium",
        "pass_number": 1,
        "study": "AOA",
        "study_id": 1011,
        "knee": "right",
        "audio_file_name": "audio.pkl",
        "microphones": microphones,
        "biomech_file_name": "AOA1011_Walk0001.c3d",
        "audio_sync_time": timedelta(seconds=0),
        "biomech_sync_left_time": timedelta(seconds=0),
        "biomech_sync_right_time": timedelta(seconds=0),
        "id": 1,
        "cycle_index": 0,
        "cycle_acoustic_energy": 123.4,
        "cycle_qc_pass": True,
    }


@pytest.fixture
def sample_synchronized_df() -> pd.DataFrame:
    """Create a sample synchronized DataFrame with all required columns."""

    n_samples = 100
    return pd.DataFrame(
        {
            "tt": np.arange(n_samples) * 0.001,
            "ch1": np.random.randn(n_samples) * 0.001,
            "ch2": np.random.randn(n_samples) * 0.001,
            "ch3": np.random.randn(n_samples) * 0.001,
            "ch4": np.random.randn(n_samples) * 0.001,
            "f_ch1": np.random.randn(n_samples) * 0.01,
            "f_ch2": np.random.randn(n_samples) * 0.01,
            "f_ch3": np.random.randn(n_samples) * 0.01,
            "f_ch4": np.random.randn(n_samples) * 0.01,
            "TIME": pd.timedelta_range(start="0s", periods=n_samples, freq="1ms"),
        }
    )


class TestFullMovementCycleMetadata:
    """Validation coverage for FullMovementCycleMetadata."""

    def test_walk_requires_speed_and_pass_number(self, base_kwargs: dict) -> None:
        missing_speed = {**base_kwargs, "speed": None}
        with pytest.raises(ValidationError):
            FullMovementCycleMetadata(**missing_speed)

        missing_pass = {**base_kwargs, "pass_number": None}
        with pytest.raises(ValidationError):
            FullMovementCycleMetadata(**missing_pass)

    def test_walk_negative_pass_number_fails(self, base_kwargs: dict) -> None:
        with pytest.raises(ValidationError):
            FullMovementCycleMetadata(**{**base_kwargs, "pass_number": -1})

    def test_non_walk_forces_speed_none(self, base_kwargs: dict) -> None:
        data = {**base_kwargs, "scripted_maneuver": "sit_to_stand", "pass_number": None, "speed": None}
        FullMovementCycleMetadata(**data)

        with pytest.raises(ValidationError):
            FullMovementCycleMetadata(**{**data, "speed": "slow"})

    def test_maneuver_alias_is_accepted(self, base_kwargs: dict) -> None:
        payload = {**base_kwargs, "maneuver": "walk"}
        meta = FullMovementCycleMetadata(**payload)
        assert meta.scripted_maneuver == "walk"
        assert meta.maneuver == "walk"

    def test_cycle_index_and_id_non_negative(self, base_kwargs: dict) -> None:
        FullMovementCycleMetadata(**base_kwargs)
        with pytest.raises(ValidationError):
            FullMovementCycleMetadata(**{**base_kwargs, "cycle_index": -1})
        with pytest.raises(ValidationError):
            FullMovementCycleMetadata(**{**base_kwargs, "id": -5})

    def test_cycle_acoustic_energy_and_qc_required(self, base_kwargs: dict) -> None:
        missing_energy = base_kwargs.copy()
        missing_energy.pop("cycle_acoustic_energy")
        with pytest.raises(ValidationError):
            FullMovementCycleMetadata(**missing_energy)

        missing_qc = base_kwargs.copy()
        missing_qc.pop("cycle_qc_pass")
        with pytest.raises(ValidationError):
            FullMovementCycleMetadata(**missing_qc)


class TestMovementCycle:
    """Validation coverage for MovementCycle objects."""

    def test_movement_cycle_creation_with_valid_data(
        self, base_kwargs: dict, sample_synchronized_df: pd.DataFrame
    ) -> None:
        cycle = MovementCycle(
            **base_kwargs,
            data=sample_synchronized_df,
        )

        assert cycle.scripted_maneuver == "walk"
        assert cycle.speed == "medium"
        assert len(cycle.data) == 100

    def test_movement_cycle_requires_data(self, base_kwargs: dict) -> None:
        with pytest.raises(ValidationError):
            MovementCycle(**base_kwargs)

    def test_movement_cycle_data_validation(self, base_kwargs: dict, sample_synchronized_df: pd.DataFrame) -> None:
        incomplete_df = sample_synchronized_df[["tt", "ch1"]].copy()
        with pytest.raises(ValidationError):
            MovementCycle(**{**base_kwargs, "data": incomplete_df})

    def test_non_walk_cycle(self, base_kwargs: dict, sample_synchronized_df: pd.DataFrame) -> None:
        cycle = MovementCycle(
            **{
                **base_kwargs,
                "scripted_maneuver": "flexion_extension",
                "speed": None,
                "pass_number": None,
            },
            data=sample_synchronized_df,
        )

        assert cycle.scripted_maneuver == "flexion_extension"
        assert cycle.speed is None
        assert cycle.pass_number is None
        assert isinstance(cycle.data, SynchronizedData)
