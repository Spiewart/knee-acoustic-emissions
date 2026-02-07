"""Tests for PersistentParticipantProcessor with optional database persistence."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.orchestration.persistent_processor import (
    PersistentParticipantProcessor,
    create_persistent_processor,
)


class TestPersistentParticipantProcessor:
    """Test PersistentParticipantProcessor wrapper."""

    @patch("src.orchestration.persistent_processor.ParticipantProcessor")
    def test_initialization_without_database(self, mock_processor_class):
        """Test processor can be initialized without database."""
        participant_dir = Path("/tmp/test_participant")
        mock_processor_class.return_value = MagicMock()

        processor = PersistentParticipantProcessor(
            participant_dir=participant_dir,
            biomechanics_type="Motion Analysis",
            db_session=None,
        )

        assert processor.participant_dir == participant_dir
        assert processor.biomechanics_type == "Motion Analysis"
        assert processor.processor is not None
        assert processor.persistence is not None
        assert not processor.persistence.enabled
        assert processor.tracker is None

    @patch("src.orchestration.persistent_processor.ParticipantProcessor")
    def test_initialization_with_database(self, mock_processor_class):
        """Test processor can be initialized with database session."""
        participant_dir = Path("/tmp/test_participant")
        mock_session = MagicMock()
        mock_processor_class.return_value = MagicMock()

        processor = PersistentParticipantProcessor(
            participant_dir=participant_dir,
            biomechanics_type="Motion Analysis",
            db_session=mock_session,
        )

        assert processor.participant_dir == participant_dir
        assert processor.persistence.enabled
        assert processor.tracker is not None

    @patch("src.orchestration.persistent_processor.ParticipantProcessor")
    def test_process_delegates_to_core_processor(self, mock_processor_class):
        """Test that process() delegates to core processor."""
        participant_dir = Path("/tmp/test_participant")
        mock_core = MagicMock()
        mock_core.process.return_value = True
        mock_processor_class.return_value = mock_core

        processor = PersistentParticipantProcessor(
            participant_dir=participant_dir,
            db_session=None,
        )

        # Call process
        result = processor.process(entrypoint="sync", knee="left", maneuver="walk")

        # Verify delegation
        mock_core.process.assert_called_once_with(
            entrypoint="sync",
            knee="left",
            maneuver="walk",
        )
        assert result is True

    @patch("src.orchestration.persistent_processor.ParticipantProcessor")
    def test_process_handles_core_processor_failure(self, mock_processor_class):
        """Test that failures in core processor are propagated."""
        participant_dir = Path("/tmp/test_participant")
        mock_core = MagicMock()
        mock_core.process.return_value = False
        mock_processor_class.return_value = mock_core

        processor = PersistentParticipantProcessor(
            participant_dir=participant_dir,
            db_session=None,
        )

        result = processor.process()

        assert result is False

    @patch("src.orchestration.persistent_processor.ParticipantProcessor")
    def test_persistence_results_called_when_database_enabled(self, mock_processor_class):
        """Test that persistence is triggered when database is enabled."""
        participant_dir = Path("/tmp/test_participant")
        mock_session = MagicMock()
        mock_core = MagicMock()
        mock_core.process.return_value = True
        mock_processor_class.return_value = mock_core

        processor = PersistentParticipantProcessor(
            participant_dir=participant_dir,
            db_session=mock_session,
        )

        # Mock persistence method
        processor._persist_processor_results = MagicMock()

        result = processor.process()

        assert result is True
        processor._persist_processor_results.assert_called_once()

    @patch("src.orchestration.persistent_processor.ParticipantProcessor")
    def test_persistence_results_not_called_without_database(self, mock_processor_class):
        """Test that persistence is not triggered when database disabled."""
        participant_dir = Path("/tmp/test_participant")
        mock_core = MagicMock()
        mock_core.process.return_value = True
        mock_processor_class.return_value = mock_core

        processor = PersistentParticipantProcessor(
            participant_dir=participant_dir,
            db_session=None,
        )

        # Mock persistence method
        processor._persist_processor_results = MagicMock()

        result = processor.process()

        assert result is True
        processor._persist_processor_results.assert_not_called()

    @patch("src.orchestration.persistent_processor.ParticipantProcessor")
    def test_process_handles_exceptions(self, mock_processor_class):
        """Test that exceptions during processing are handled gracefully."""
        participant_dir = Path("/tmp/test_participant")
        mock_core = MagicMock()
        mock_core.process.side_effect = ValueError("Test error")
        mock_processor_class.return_value = mock_core

        processor = PersistentParticipantProcessor(
            participant_dir=participant_dir,
            db_session=None,
        )

        result = processor.process()

        assert result is False


class TestCreatePersistentProcessor:
    """Test factory function for creating persistent processors."""

    @patch("src.orchestration.persistent_processor.ParticipantProcessor")
    def test_factory_without_database_url(self, mock_processor_class):
        """Test factory creates processor without database when url not provided."""
        participant_dir = Path("/tmp/test_participant")
        mock_processor_class.return_value = MagicMock()

        processor = create_persistent_processor(participant_dir)

        assert processor is not None
        assert not processor.persistence.enabled

    @patch("src.orchestration.persistent_processor.ParticipantProcessor")
    @patch("src.orchestration.cli_db_helpers.create_db_session")
    def test_factory_with_valid_database_url(self, mock_create_session, mock_processor_class):
        """Test factory creates processor with database when url is provided."""
        participant_dir = Path("/tmp/test_participant")
        mock_session = MagicMock()
        mock_create_session.return_value = mock_session
        mock_processor_class.return_value = MagicMock()

        processor = create_persistent_processor(
            participant_dir,
            db_url="postgresql://localhost/test"
        )

        assert processor is not None
        assert processor.persistence.enabled
        mock_create_session.assert_called_once_with("postgresql://localhost/test")

    @patch("src.orchestration.persistent_processor.ParticipantProcessor")
    @patch("src.orchestration.cli_db_helpers.create_db_session")
    def test_factory_graceful_degradation_on_db_error(self, mock_create_session, mock_processor_class):
        """Test factory falls back to no-database when db connection fails."""
        participant_dir = Path("/tmp/test_participant")
        mock_create_session.side_effect = Exception("Connection failed")
        mock_processor_class.return_value = MagicMock()

        processor = create_persistent_processor(
            participant_dir,
            db_url="postgresql://localhost/test"
        )

        assert processor is not None
        assert not processor.persistence.enabled

    @patch("src.orchestration.persistent_processor.ParticipantProcessor")
    def test_factory_accepts_biomechanics_type(self, mock_processor_class):
        """Test factory accepts biomechanics_type parameter."""
        participant_dir = Path("/tmp/test_participant")
        mock_processor_class.return_value = MagicMock()

        processor = create_persistent_processor(
            participant_dir,
            biomechanics_type="Motion Analysis"
        )

        assert processor.biomechanics_type == "Motion Analysis"


class TestIntegrationWithRealParticipantDirectory:
    """Integration tests with sample participant directory."""

    @pytest.fixture
    def sample_participant_dir(self):
        """Get sample participant directory if available."""
        import os
        root = os.getenv("AE_DATA_ROOT")
        if not root:
            pytest.skip("AE_DATA_ROOT not set")

        participant_dir = Path(root) / "#1011"
        if not participant_dir.exists():
            pytest.skip(f"Sample participant directory not found: {participant_dir}")

        return participant_dir

    def test_processor_initialization_with_real_directory(self, sample_participant_dir):
        """Test processor can be initialized with real participant directory."""
        processor = PersistentParticipantProcessor(
            participant_dir=sample_participant_dir,
            db_session=None,
        )

        assert processor.participant_dir == sample_participant_dir
        assert processor.processor is not None

    def test_factory_initialization_with_real_directory(self, sample_participant_dir):
        """Test factory can create processor with real participant directory."""
        processor = create_persistent_processor(sample_participant_dir)

        assert processor is not None
        assert processor.participant_dir == sample_participant_dir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
