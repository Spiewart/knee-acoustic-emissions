"""Integration tests for OOP participant processor."""

from pathlib import Path

import pytest

from src.orchestration.participant_processor import (
    AudioData,
    BiomechanicsData,
    CycleData,
    KneeProcessor,
    ManeuverProcessor,
    ParticipantProcessor,
    SyncData,
)
from tests.conftest import (
    _create_acoustic_legend,
    _create_biomechanics_excel,
    _generate_audio_dataframe,
    _write_audio_files,
)


@pytest.fixture
def test_participant_dir(tmp_path_factory):
    """Create a complete test participant directory."""
    project_dir = tmp_path_factory.mktemp("project")
    participant_dir = project_dir / "#1011"
    participant_dir.mkdir()

    # Create acoustic legend
    _create_acoustic_legend(participant_dir)

    # Create knee directories
    left_knee_dir = participant_dir / "Left Knee"
    right_knee_dir = participant_dir / "Right Knee"
    left_knee_dir.mkdir()
    right_knee_dir.mkdir()

    # Create audio files
    audio_df = _generate_audio_dataframe()
    _write_audio_files(left_knee_dir, audio_df)
    _write_audio_files(right_knee_dir, audio_df)

    # Create biomechanics Excel
    biomechanics_info = _create_biomechanics_excel(
        motion_capture_dir=participant_dir / "Motion Capture",
        study_id="1011",
    )

    return {
        "participant_dir": participant_dir,
        "project_dir": project_dir,
        "biomechanics_file": biomechanics_info["excel_path"],
    }


class TestAudioData:
    """Test AudioData dataclass."""

    def test_init(self):
        """Test AudioData initialization."""
        pkl_path = Path("/tmp/test.pkl")

        audio_data = AudioData(
            pkl_path=pkl_path,
        )

        assert audio_data.pkl_path == pkl_path
        assert audio_data.df is None
        assert audio_data.record is None


class TestBiomechanicsData:
    """Test BiomechanicsData dataclass."""

    def test_init(self):
        """Test BiomechanicsData initialization."""
        file_path = Path("/tmp/bio.xlsx")

        bio_data = BiomechanicsData(
            file_path=file_path,
        )

        assert bio_data.file_path == file_path
        assert bio_data.recordings == []
        assert bio_data.record is None


class TestSyncData:
    """Test SyncData dataclass."""

    def test_init(self):
        """Test SyncData initialization."""
        import pandas as pd

        output_path = Path("/tmp/synced.pkl")
        df = pd.DataFrame({"col": [1, 2, 3]})
        stomp_times = (1.0, 2.0, 3.0)

        sync_data = SyncData(
            output_path=output_path,
            df=df,
            stomp_times=stomp_times,
        )

        assert sync_data.output_path == output_path
        assert sync_data.df is not None
        assert sync_data.stomp_times == stomp_times
        assert sync_data.record is None


class TestCycleData:
    """Test CycleData dataclass."""

    def test_init(self):
        """Test CycleData initialization."""
        synced_file = Path("/tmp/synced.pkl")

        cycle_data = CycleData(
            synced_file_path=synced_file,
        )

        assert cycle_data.synced_file_path == synced_file
        assert cycle_data.output_dir is None
        assert cycle_data.record is None


class TestManeuverProcessorInit:
    """Test ManeuverProcessor initialization and basic methods."""

    def test_init(self, test_participant_dir):
        """Test ManeuverProcessor initialization."""
        maneuver_dir = test_participant_dir["participant_dir"] / "Left Knee" / "Walking"

        processor = ManeuverProcessor(
            maneuver_dir=maneuver_dir,
            maneuver_key="walk",
            knee_side="Left",
            study_id="1011",
            biomechanics_file=test_participant_dir["biomechanics_file"],
            biomechanics_type="IMU",
        )

        assert processor.maneuver_dir == maneuver_dir
        assert processor.maneuver_key == "walk"
        assert processor.knee_side == "Left"
        assert processor.study_id == "1011"
        assert processor.audio is None
        assert processor.biomechanics is None
        assert processor.synced_data == []
        assert processor.cycle_data == []

    def test_find_audio_pickle(self, test_participant_dir):
        """Test finding audio pickle file."""
        maneuver_dir = test_participant_dir["participant_dir"] / "Left Knee" / "Walking"

        processor = ManeuverProcessor(
            maneuver_dir=maneuver_dir,
            maneuver_key="walk",
            knee_side="Left",
            study_id="1011",
            biomechanics_file=test_participant_dir["biomechanics_file"],
        )

        pkl = processor._find_audio_pickle()
        assert pkl is not None
        assert pkl.exists()
        assert pkl.suffix == ".pkl"

    def test_find_bin_file(self, test_participant_dir):
        """Test finding .bin file."""
        maneuver_dir = test_participant_dir["participant_dir"] / "Left Knee" / "Walking"

        processor = ManeuverProcessor(
            maneuver_dir=maneuver_dir,
            maneuver_key="walk",
            knee_side="Left",
            study_id="1011",
            biomechanics_file=test_participant_dir["biomechanics_file"],
        )

        bin_file = processor._find_bin_file()
        assert bin_file is not None
        assert bin_file.exists()
        assert bin_file.suffix == ".bin"


class TestKneeProcessorInit:
    """Test KneeProcessor initialization."""

    def test_init(self, test_participant_dir):
        """Test KneeProcessor initialization."""
        knee_dir = test_participant_dir["participant_dir"] / "Left Knee"

        processor = KneeProcessor(
            knee_dir=knee_dir,
            knee_side="Left",
            study_id="1011",
            biomechanics_file=test_participant_dir["biomechanics_file"],
            biomechanics_type="IMU",
        )

        assert processor.knee_dir == knee_dir
        assert processor.knee_side == "Left"
        assert processor.study_id == "1011"
        assert processor.maneuver_processors == {}

    def test_find_maneuver_dir(self, test_participant_dir):
        """Test finding maneuver directories."""
        knee_dir = test_participant_dir["participant_dir"] / "Left Knee"

        processor = KneeProcessor(
            knee_dir=knee_dir,
            knee_side="Left",
            study_id="1011",
            biomechanics_file=test_participant_dir["biomechanics_file"],
        )

        walk_dir = processor._find_maneuver_dir("walk")
        assert walk_dir is not None
        assert walk_dir.exists()
        assert walk_dir.name == "Walking"

        sit_stand_dir = processor._find_maneuver_dir("sit_to_stand")
        assert sit_stand_dir is not None
        assert sit_stand_dir.name == "Sit-Stand"

    def test_find_maneuver_dir_with_naming_variants(self, tmp_path):
        """Test _find_maneuver_dir handles naming variants like 'Sit_Stand', 'sit-to-stand', etc."""
        from src.orchestration.participant_processor import KneeProcessor

        # Create test directories with different naming conventions
        knee_dir = tmp_path / "Left Knee"
        knee_dir.mkdir()

        # Create directories with different naming variants
        (knee_dir / "Walking").mkdir()
        (knee_dir / "Sit_Stand").mkdir()  # Underscore variant
        (knee_dir / "Flexion-Extension").mkdir()

        # Create biomechanics file (dummy)
        biomech_file = tmp_path / "Motion Capture" / "AOA1011_Biomechanics_Full_Set.xlsx"
        biomech_file.parent.mkdir(parents=True)
        biomech_file.touch()

        processor = KneeProcessor(
            knee_dir=knee_dir,
            knee_side="Left",
            study_id="1011",
            biomechanics_file=biomech_file,
        )

        # Should find "Walking" for walk
        walk_dir = processor._find_maneuver_dir("walk")
        assert walk_dir is not None
        assert walk_dir.name == "Walking"

        # Should find "Sit_Stand" even though it's not the standard "Sit-Stand"
        sit_stand_dir = processor._find_maneuver_dir("sit_to_stand")
        assert sit_stand_dir is not None
        assert sit_stand_dir.name == "Sit_Stand"

        # Should find "Flexion-Extension"
        flex_ext_dir = processor._find_maneuver_dir("flexion_extension")
        assert flex_ext_dir is not None
        assert flex_ext_dir.name == "Flexion-Extension"

    def test_find_maneuver_dir_with_multiple_variants(self, tmp_path):
        """Test that _find_maneuver_dir matches against all known naming variants."""
        from src.orchestration.participant_processor import KneeProcessor

        knee_dir = tmp_path / "Left Knee"
        knee_dir.mkdir()

        # Test sit-to-stand variants
        variants_to_test = [
            "sit-stand",
            "sit-to-stand",
            "sit_to_stand",
            "sitstand",
            "sit to stand",
            "sittostand",
        ]

        biomech_file = tmp_path / "Motion Capture" / "AOA1011_Biomechanics_Full_Set.xlsx"
        biomech_file.parent.mkdir(parents=True)
        biomech_file.touch()

        for variant in variants_to_test:
            test_dir = knee_dir / variant
            test_dir.mkdir(exist_ok=True)

            processor = KneeProcessor(
                knee_dir=knee_dir,
                knee_side="Left",
                study_id="1011",
                biomechanics_file=biomech_file,
            )

            # Clear the directory for next iteration
            sit_stand_dir = processor._find_maneuver_dir("sit_to_stand")
            assert sit_stand_dir is not None, f"Failed to find sit_to_stand variant: {variant}"
            assert sit_stand_dir.name == variant

            # Clean up for next test
            test_dir.rmdir()

    def test_find_maneuver_dir_returns_none_when_not_found(self, tmp_path):
        """Test that _find_maneuver_dir returns None when directory doesn't exist."""
        from src.orchestration.participant_processor import KneeProcessor

        knee_dir = tmp_path / "Left Knee"
        knee_dir.mkdir()

        biomech_file = tmp_path / "Motion Capture" / "AOA1011_Biomechanics_Full_Set.xlsx"
        biomech_file.parent.mkdir(parents=True)
        biomech_file.touch()

        processor = KneeProcessor(
            knee_dir=knee_dir,
            knee_side="Left",
            study_id="1011",
            biomechanics_file=biomech_file,
        )

        # Should return None when directory doesn't exist
        result = processor._find_maneuver_dir("walk")
        assert result is None


class TestParticipantProcessorInit:
    """Test ParticipantProcessor initialization."""

    def test_init(self, test_participant_dir):
        """Test ParticipantProcessor initialization."""
        processor = ParticipantProcessor(
            participant_dir=test_participant_dir["participant_dir"],
            biomechanics_type="IMU",
        )

        assert processor.participant_dir == test_participant_dir["participant_dir"]
        assert processor.study_id == "1011"
        assert processor.biomechanics_type == "IMU"
        assert processor.biomechanics_file is not None

    def test_find_biomechanics_file(self, test_participant_dir):
        """Test finding biomechanics file."""
        processor = ParticipantProcessor(
            participant_dir=test_participant_dir["participant_dir"],
        )

        assert processor.biomechanics_file.exists()
        assert "AOA1011" in processor.biomechanics_file.name
