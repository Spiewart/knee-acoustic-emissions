"""Test suite for MovementCycleMetadata and MovementCycle models."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.models import MovementCycle, MovementCycleMetadata, SynchronizedRecording


class TestMovementCycleMetadata:
    """Test suite for MovementCycleMetadata validation."""

    def test_walk_with_speed_and_pass_number_succeeds(self):
        """Walk maneuver with both speed and pass_number should succeed."""
        metadata = MovementCycleMetadata(
            maneuver="walk",
            speed="medium",
            pass_number=1,
            cycle_index=0,
            knee="right",
            acoustic_energy=150.0,
        )
        assert metadata.maneuver == "walk"
        assert metadata.speed == "medium"
        assert metadata.pass_number == 1

    def test_walk_with_all_speeds(self):
        """Test walk maneuver accepts slow, medium, and fast speeds."""
        for speed in ["slow", "medium", "fast"]:
            metadata = MovementCycleMetadata(
                maneuver="walk",
                speed=speed,
                pass_number=0,
                cycle_index=0,
                knee="left",
                acoustic_energy=100.0,
            )
            assert metadata.speed == speed

    def test_walk_without_speed_fails(self):
        """Walk maneuver without speed should fail."""
        with pytest.raises(ValidationError) as exc_info:
            MovementCycleMetadata(
                maneuver="walk",
                speed=None,
                pass_number=1,
                cycle_index=0,
                knee="right",
                acoustic_energy=150.0,
            )
        assert "speed" in str(exc_info.value).lower()
        assert "missing" in str(exc_info.value).lower()

    def test_walk_without_pass_number_fails(self):
        """Walk maneuver without pass_number should fail."""
        with pytest.raises(ValidationError) as exc_info:
            MovementCycleMetadata(
                maneuver="walk",
                speed="medium",
                pass_number=None,
                cycle_index=0,
                knee="right",
                acoustic_energy=150.0,
            )
        assert "pass_number" in str(exc_info.value).lower()
        assert "missing" in str(exc_info.value).lower()

    def test_walk_without_speed_and_pass_number_fails(self):
        """Walk maneuver without both speed and pass_number should fail."""
        with pytest.raises(ValidationError) as exc_info:
            MovementCycleMetadata(
                maneuver="walk",
                speed=None,
                pass_number=None,
                cycle_index=0,
                knee="right",
                acoustic_energy=150.0,
            )
        error_msg = str(exc_info.value).lower()
        assert "speed" in error_msg or "pass_number" in error_msg

    def test_walk_with_negative_pass_number_fails(self):
        """Walk maneuver with negative pass_number should fail."""
        with pytest.raises(ValidationError) as exc_info:
            MovementCycleMetadata(
                maneuver="walk",
                speed="medium",
                pass_number=-1,
                cycle_index=0,
                knee="right",
                acoustic_energy=150.0,
            )
        assert "non-negative" in str(exc_info.value).lower()

    def test_sit_to_stand_with_none_speed_and_pass_number_succeeds(self):
        """Sit-to-stand maneuver with both None should succeed."""
        metadata = MovementCycleMetadata(
            maneuver="sit_to_stand",
            speed=None,
            pass_number=None,
            cycle_index=2,
            knee="left",
            acoustic_energy=200.0,
        )
        assert metadata.maneuver == "sit_to_stand"
        assert metadata.speed is None
        assert metadata.pass_number is None

    def test_sit_to_stand_with_speed_fails(self):
        """Sit-to-stand maneuver with speed should fail."""
        with pytest.raises(ValidationError) as exc_info:
            MovementCycleMetadata(
                maneuver="sit_to_stand",
                speed="medium",
                pass_number=None,
                cycle_index=2,
                knee="left",
                acoustic_energy=200.0,
            )
        error_msg = str(exc_info.value).lower()
        assert "speed" in error_msg or "do not support" in error_msg

    def test_sit_to_stand_with_pass_number_fails(self):
        """Sit-to-stand maneuver with pass_number should fail."""
        with pytest.raises(ValidationError) as exc_info:
            MovementCycleMetadata(
                maneuver="sit_to_stand",
                speed=None,
                pass_number=1,
                cycle_index=2,
                knee="left",
                acoustic_energy=200.0,
            )
        error_msg = str(exc_info.value).lower()
        assert "pass_number" in error_msg or "do not support" in error_msg

    def test_sit_to_stand_with_both_fails(self):
        """Sit-to-stand maneuver with both speed and pass_number should fail."""
        with pytest.raises(ValidationError) as exc_info:
            MovementCycleMetadata(
                maneuver="sit_to_stand",
                speed="fast",
                pass_number=1,
                cycle_index=2,
                knee="left",
                acoustic_energy=200.0,
            )
        error_msg = str(exc_info.value).lower()
        assert "do not support" in error_msg

    def test_flexion_extension_with_none_speed_and_pass_number_succeeds(self):
        """Flexion-extension maneuver with both None should succeed."""
        metadata = MovementCycleMetadata(
            maneuver="flexion_extension",
            speed=None,
            pass_number=None,
            cycle_index=5,
            knee="right",
            acoustic_energy=175.0,
        )
        assert metadata.maneuver == "flexion_extension"
        assert metadata.speed is None
        assert metadata.pass_number is None

    def test_flexion_extension_with_speed_fails(self):
        """Flexion-extension maneuver with speed should fail."""
        with pytest.raises(ValidationError) as exc_info:
            MovementCycleMetadata(
                maneuver="flexion_extension",
                speed="slow",
                pass_number=None,
                cycle_index=5,
                knee="right",
                acoustic_energy=175.0,
            )
        error_msg = str(exc_info.value).lower()
        assert "speed" in error_msg or "do not support" in error_msg

    def test_cycle_index_required(self):
        """cycle_index is required field."""
        with pytest.raises(ValidationError) as exc_info:
            MovementCycleMetadata(
                maneuver="walk",
                speed="medium",
                pass_number=1,
                # cycle_index missing
                knee="right",
                acoustic_energy=150.0,
            )
        assert "cycle_index" in str(exc_info.value).lower()

    def test_knee_laterality(self):
        """Test left and right knee values."""
        for knee in ["left", "right"]:
            metadata = MovementCycleMetadata(
                maneuver="walk",
                speed="medium",
                pass_number=0,
                cycle_index=0,
                knee=knee,
                acoustic_energy=100.0,
            )
            assert metadata.knee == knee

    def test_optional_fields(self):
        """Test optional fields (participant_id, notes, is_outlier)."""
        metadata = MovementCycleMetadata(
            maneuver="walk",
            speed="medium",
            pass_number=1,
            cycle_index=0,
            knee="right",
            acoustic_energy=150.0,
            participant_id="1011",
            notes="Test cycle",
            is_outlier=True,
        )
        assert metadata.participant_id == "1011"
        assert metadata.notes == "Test cycle"
        assert metadata.is_outlier is True

    def test_acoustic_energy_required(self):
        """acoustic_energy is required field."""
        with pytest.raises(ValidationError) as exc_info:
            MovementCycleMetadata(
                maneuver="walk",
                speed="medium",
                pass_number=1,
                cycle_index=0,
                knee="right",
                # acoustic_energy missing
            )
        assert "acoustic_energy" in str(exc_info.value).lower()


class TestMovementCycle:
    """Test suite for MovementCycle model."""

    @pytest.fixture
    def sample_synchronized_df(self):
        """Create a sample synchronized DataFrame with all required columns."""
        n_samples = 100
        return pd.DataFrame(
            {
                "tt": np.arange(n_samples) * 0.001,  # Time in seconds
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

    def test_movement_cycle_creation_with_valid_data(self, sample_synchronized_df):
        """Create a MovementCycle with valid metadata and data."""
        metadata = MovementCycleMetadata(
            maneuver="walk",
            speed="medium",
            pass_number=1,
            cycle_index=0,
            knee="right",
            acoustic_energy=150.0,
        )

        cycle = MovementCycle(
            metadata=metadata,
            data=sample_synchronized_df,
        )

        assert cycle.metadata.maneuver == "walk"
        assert cycle.metadata.speed == "medium"
        assert len(cycle.data) == 100

    def test_movement_cycle_requires_metadata(self, sample_synchronized_df):
        """MovementCycle requires metadata field."""
        with pytest.raises(ValidationError) as exc_info:
            MovementCycle(
                # metadata missing
                data=sample_synchronized_df,
            )
        assert "metadata" in str(exc_info.value).lower()

    def test_movement_cycle_requires_data(self):
        """MovementCycle requires data field."""
        metadata = MovementCycleMetadata(
            maneuver="walk",
            speed="medium",
            pass_number=1,
            cycle_index=0,
            knee="right",
            acoustic_energy=150.0,
        )

        with pytest.raises(ValidationError) as exc_info:
            MovementCycle(
                metadata=metadata,
                # data missing
            )
        assert "data" in str(exc_info.value).lower()

    def test_movement_cycle_sit_to_stand_data(self, sample_synchronized_df):
        """Create a sit-to-stand MovementCycle."""
        metadata = MovementCycleMetadata(
            maneuver="sit_to_stand",
            speed=None,
            pass_number=None,
            cycle_index=0,
            knee="left",
            acoustic_energy=200.0,
        )

        cycle = MovementCycle(
            metadata=metadata,
            data=sample_synchronized_df,
        )

        assert cycle.metadata.maneuver == "sit_to_stand"
        assert cycle.metadata.speed is None
        assert cycle.metadata.pass_number is None

    def test_movement_cycle_flexion_extension_data(self, sample_synchronized_df):
        """Create a flexion-extension MovementCycle."""
        metadata = MovementCycleMetadata(
            maneuver="flexion_extension",
            speed=None,
            pass_number=None,
            cycle_index=3,
            knee="right",
            acoustic_energy=180.0,
        )

        cycle = MovementCycle(
            metadata=metadata,
            data=sample_synchronized_df,
        )

        assert cycle.metadata.maneuver == "flexion_extension"
        assert cycle.metadata.cycle_index == 3

    def test_movement_cycle_data_validation(self, sample_synchronized_df):
        """Test that data is validated as SynchronizedRecording."""
        metadata = MovementCycleMetadata(
            maneuver="walk",
            speed="medium",
            pass_number=1,
            cycle_index=0,
            knee="right",
            acoustic_energy=150.0,
        )

        # Create incomplete DataFrame (missing required columns)
        incomplete_df = pd.DataFrame(
            {
                "tt": np.arange(10) * 0.001,
                "ch1": np.random.randn(10),
            }
        )

        with pytest.raises(ValidationError) as exc_info:
            MovementCycle(
                metadata=metadata,
                data=incomplete_df,
            )
        assert (
            "missing" in str(exc_info.value).lower()
            or "column" in str(exc_info.value).lower()
        )


class TestMovementCycleIntegration:
    """Integration tests for movement cycle models."""

    @pytest.fixture
    def sample_synchronized_df(self):
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

    def test_create_multiple_cycles_different_maneuvers(self, sample_synchronized_df):
        """Create multiple cycles with different maneuvers."""
        walk_cycle = MovementCycle(
            metadata=MovementCycleMetadata(
                maneuver="walk",
                speed="medium",
                pass_number=1,
                cycle_index=0,
                knee="right",
                acoustic_energy=150.0,
            ),
            data=sample_synchronized_df,
        )

        sts_cycle = MovementCycle(
            metadata=MovementCycleMetadata(
                maneuver="sit_to_stand",
                speed=None,
                pass_number=None,
                cycle_index=0,
                knee="left",
                acoustic_energy=200.0,
            ),
            data=sample_synchronized_df,
        )

        flex_cycle = MovementCycle(
            metadata=MovementCycleMetadata(
                maneuver="flexion_extension",
                speed=None,
                pass_number=None,
                cycle_index=2,
                knee="right",
                acoustic_energy=180.0,
            ),
            data=sample_synchronized_df,
        )

        # Verify cycles created correctly
        assert walk_cycle.metadata.maneuver == "walk"
        assert sts_cycle.metadata.maneuver == "sit_to_stand"
        assert flex_cycle.metadata.maneuver == "flexion_extension"

        # Verify all have data
        assert len(walk_cycle.data) == 100
        assert len(sts_cycle.data) == 100
        assert len(flex_cycle.data) == 100

    def test_cycle_list_with_different_speeds(self, sample_synchronized_df):
        """Create a list of walk cycles with different speeds."""
        cycles = []
        for speed in ["slow", "medium", "fast"]:
            for pass_num in range(2):
                cycle = MovementCycle(
                    metadata=MovementCycleMetadata(
                        maneuver="walk",
                        speed=speed,
                        pass_number=pass_num,
                        cycle_index=len(cycles),
                        knee="right",
                        acoustic_energy=100.0 + len(cycles),
                    ),
                    data=sample_synchronized_df,
                )
                cycles.append(cycle)

        assert len(cycles) == 6  # 3 speeds × 2 passes
        assert all(c.metadata.maneuver == "walk" for c in cycles)
        assert cycles[0].metadata.speed == "slow"
        assert cycles[2].metadata.speed == "medium"
        assert cycles[4].metadata.speed == "fast"
