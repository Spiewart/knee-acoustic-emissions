"""Integration tests for QC versioning across the pipeline."""

import pandas as pd
import pytest

from src.models import (
    AcousticsFileMetadata,
    BiomechanicsFileMetadata,
    MicrophonePosition,
    FullMovementCycleMetadata,
)
from src.qc_versions import (
    AUDIO_QC_VERSION,
    BIOMECH_QC_VERSION,
    CYCLE_QC_VERSION,
)


class TestQCVersionIntegration:
    """Test that QC versions are properly tracked throughout the pipeline."""

    def test_audio_metadata_includes_version(self):
        """AcousticsFileMetadata should include current audio QC version."""
        microphones = {
            1: MicrophonePosition(patellar_position="Infrapatellar", laterality="Lateral"),
            2: MicrophonePosition(patellar_position="Infrapatellar", laterality="Medial"),
            3: MicrophonePosition(patellar_position="Suprapatellar", laterality="Medial"),
            4: MicrophonePosition(patellar_position="Suprapatellar", laterality="Lateral"),
        }

        metadata = AcousticsFileMetadata(
            study="AOA",
            study_id=123,
            knee="left",
            scripted_maneuver="walk",
            speed="medium",
            pass_number=1,
            file_name="test_audio.bin",
            microphones=microphones,
        )

        assert metadata.audio_qc_version == AUDIO_QC_VERSION
        assert metadata.audio_qc_version == 1  # Current version

    def test_biomech_metadata_includes_version(self):
        """BiomechanicsFileMetadata should include current biomech QC version."""
        metadata = BiomechanicsFileMetadata(
            study="AOA",
            study_id=123,
            scripted_maneuver="walk",
            speed="medium",
            pass_number=1,
            file_name="test_biomech.xlsx",
        )

        assert metadata.biomech_qc_version == BIOMECH_QC_VERSION
        assert metadata.biomech_qc_version == 1  # Current version

    def test_cycle_metadata_includes_all_versions(self):
        """FullMovementCycleMetadata should include all three QC versions."""
        from datetime import datetime, timedelta

        microphones = {
            1: MicrophonePosition(patellar_position="Infrapatellar", laterality="Lateral"),
            2: MicrophonePosition(patellar_position="Infrapatellar", laterality="Medial"),
            3: MicrophonePosition(patellar_position="Suprapatellar", laterality="Medial"),
            4: MicrophonePosition(patellar_position="Suprapatellar", laterality="Lateral"),
        }

        metadata = FullMovementCycleMetadata(
            id=0,
            cycle_index=0,
            cycle_acoustic_energy=100.0,
            cycle_qc_pass=True,
            cycle_qc_version=CYCLE_QC_VERSION,
            study="AOA",
            study_id=123,
            knee="left",
            scripted_maneuver="walk",
            speed="medium",
            pass_number=1,
            audio_file_name="test_audio.bin",
            audio_serial_number="12345",
            audio_firmware_version=1,
            date_of_recording=datetime.now(),
            microphones=microphones,
            audio_sync_time=timedelta(seconds=10),
            audio_qc_version=AUDIO_QC_VERSION,
            biomech_file_name="test_biomech.xlsx",
            biomech_sync_left_time=timedelta(seconds=15),
            biomech_sync_right_time=timedelta(seconds=15),
            biomech_qc_version=BIOMECH_QC_VERSION,
        )

        # Verify all three QC versions are present and correct
        assert metadata.audio_qc_version == AUDIO_QC_VERSION
        assert metadata.biomech_qc_version == BIOMECH_QC_VERSION
        assert metadata.cycle_qc_version == CYCLE_QC_VERSION

        # Verify current values (all should be 1 initially)
        assert metadata.audio_qc_version == 1
        assert metadata.biomech_qc_version == 1
        assert metadata.cycle_qc_version == 1

    def test_metadata_serialization_includes_versions(self):
        """QC versions should be included when metadata is serialized."""
        microphones = {
            1: MicrophonePosition(patellar_position="Infrapatellar", laterality="Lateral"),
            2: MicrophonePosition(patellar_position="Infrapatellar", laterality="Medial"),
            3: MicrophonePosition(patellar_position="Suprapatellar", laterality="Medial"),
            4: MicrophonePosition(patellar_position="Suprapatellar", laterality="Lateral"),
        }

        metadata = AcousticsFileMetadata(
            study="AOA",
            study_id=123,
            knee="left",
            scripted_maneuver="walk",
            speed="medium",
            pass_number=1,
            file_name="test_audio.bin",
            microphones=microphones,
        )

        # Serialize to dict
        metadata_dict = metadata.model_dump()

        # Check that version is in the serialized output
        assert "audio_qc_version" in metadata_dict
        assert metadata_dict["audio_qc_version"] == AUDIO_QC_VERSION

    def test_version_fields_are_integers(self):
        """All QC version fields should be integers."""
        from datetime import datetime, timedelta

        microphones = {
            1: MicrophonePosition(patellar_position="Infrapatellar", laterality="Lateral"),
            2: MicrophonePosition(patellar_position="Infrapatellar", laterality="Medial"),
            3: MicrophonePosition(patellar_position="Suprapatellar", laterality="Medial"),
            4: MicrophonePosition(patellar_position="Suprapatellar", laterality="Lateral"),
        }

        metadata = FullMovementCycleMetadata(
            id=0,
            cycle_index=0,
            cycle_acoustic_energy=100.0,
            cycle_qc_pass=True,
            cycle_qc_version=CYCLE_QC_VERSION,
            study="AOA",
            study_id=123,
            knee="left",
            scripted_maneuver="walk",
            speed="medium",
            pass_number=1,
            audio_file_name="test_audio.bin",
            audio_serial_number="12345",
            audio_firmware_version=1,
            date_of_recording=datetime.now(),
            microphones=microphones,
            audio_sync_time=timedelta(seconds=10),
            audio_qc_version=AUDIO_QC_VERSION,
            biomech_file_name="test_biomech.xlsx",
            biomech_sync_left_time=timedelta(seconds=15),
            biomech_sync_right_time=timedelta(seconds=15),
            biomech_qc_version=BIOMECH_QC_VERSION,
        )

        assert isinstance(metadata.audio_qc_version, int)
        assert isinstance(metadata.biomech_qc_version, int)
        assert isinstance(metadata.cycle_qc_version, int)

    def test_default_versions_used_when_not_specified(self):
        """Models should use default QC versions from qc_versions module."""
        microphones = {
            1: MicrophonePosition(patellar_position="Infrapatellar", laterality="Lateral"),
            2: MicrophonePosition(patellar_position="Infrapatellar", laterality="Medial"),
            3: MicrophonePosition(patellar_position="Suprapatellar", laterality="Medial"),
            4: MicrophonePosition(patellar_position="Suprapatellar", laterality="Lateral"),
        }

        # Create metadata without explicitly specifying QC versions
        audio_metadata = AcousticsFileMetadata(
            study="AOA",
            study_id=123,
            knee="left",
            scripted_maneuver="walk",
            speed="medium",
            pass_number=1,
            file_name="test.bin",
            microphones=microphones,
        )

        biomech_metadata = BiomechanicsFileMetadata(
            study="AOA",
            study_id=123,
            scripted_maneuver="walk",
            speed="medium",
            pass_number=1,
            file_name="test.xlsx",
        )

        # Versions should still be set to defaults
        assert audio_metadata.audio_qc_version == AUDIO_QC_VERSION
        assert biomech_metadata.biomech_qc_version == BIOMECH_QC_VERSION
