"""Database persistence layer for orchestration pipeline.

This module provides integration between the orchestration pipeline and the
PostgreSQL database via the Repository layer. It handles saving metadata records
created during processing while maintaining FK relationships.

Design:
- Optional persistence: can be disabled if DB not available
- Transactional: saves records with proper FK linking
- Non-blocking: doesn't interrupt processing if DB unavailable
- Recording-time relationships: captures which audio/biomechanics recorded together
"""

import logging
from typing import Optional

from sqlalchemy.orm import Session

from src.db.repository import Repository
from src.metadata import AudioProcessing, BiomechanicsImport, MovementCycle, Synchronization

logger = logging.getLogger(__name__)


class OrchestrationDatabasePersistence:
    """Handles persistence of orchestration results to PostgreSQL database."""

    def __init__(self, session: Optional[Session] = None):
        """Initialize persistence layer.

        Args:
            session: Optional SQLAlchemy session. If None, persistence is disabled.
        """
        self.session = session
        self.repository: Optional[Repository] = None
        self.enabled = False

        if session is not None:
            try:
                self.repository = Repository(session)
                self.enabled = True
                logger.info("Database persistence enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize database persistence: {e}")

    def save_audio_processing(
        self,
        audio: AudioProcessing,
        pkl_file_path: Optional[str] = None,
        biomechanics_import_id: Optional[int] = None,
    ) -> Optional[int]:
        """Save audio processing record to database.

        Args:
            audio: AudioProcessing metadata
            pkl_file_path: Optional path to pkl file on disk
            biomechanics_import_id: Optional FK to related biomechanics import

        Returns:
            Audio record ID if saved, None if DB not available or error
        """
        if not self.enabled or self.repository is None:
            return None

        try:
            record = self.repository.save_audio_processing(
                audio,
                pkl_file_path=pkl_file_path,
                biomechanics_import_id=biomechanics_import_id,
            )
            logger.debug(f"Saved audio processing record: {record.id}")
            return record.id
        except Exception as e:
            logger.error(f"Failed to save audio processing: {e}")
            return None

    def save_biomechanics_import(
        self,
        biomech: BiomechanicsImport,
        audio_processing_id: Optional[int] = None,
    ) -> Optional[int]:
        """Save biomechanics import record to database.

        Args:
            biomech: BiomechanicsImport metadata
            audio_processing_id: Optional FK to related audio processing

        Returns:
            Biomechanics record ID if saved, None if DB not available or error
        """
        if not self.enabled or self.repository is None:
            return None

        try:
            record = self.repository.save_biomechanics_import(
                biomech,
                audio_processing_id=audio_processing_id,
            )
            logger.debug(f"Saved biomechanics import record: {record.id}")
            return record.id
        except Exception as e:
            logger.error(f"Failed to save biomechanics import: {e}")
            return None

    def save_synchronization(
        self,
        sync: Synchronization,
        audio_processing_id: int,
        biomechanics_import_id: int,
        sync_file_path: Optional[str] = None,
    ) -> Optional[int]:
        """Save synchronization record to database.

        Args:
            sync: Synchronization metadata
            audio_processing_id: FK to audio processing record
            biomechanics_import_id: FK to biomechanics import record
            sync_file_path: Optional path to sync pkl file on disk

        Returns:
            Synchronization record ID if saved, None if DB not available or error
        """
        if not self.enabled or self.repository is None:
            return None

        try:
            record = self.repository.save_synchronization(
                sync,
                audio_processing_id=audio_processing_id,
                biomechanics_import_id=biomechanics_import_id,
                sync_file_path=sync_file_path,
            )
            logger.debug(f"Saved synchronization record: {record.id}")
            return record.id
        except Exception as e:
            logger.error(f"Failed to save synchronization: {e}")
            return None

    def save_movement_cycle(
        self,
        cycle: MovementCycle,
        audio_processing_id: int,
        biomechanics_import_id: Optional[int] = None,
        synchronization_id: Optional[int] = None,
        cycles_file_path: Optional[str] = None,
    ) -> Optional[int]:
        """Save movement cycle record to database.

        Args:
            cycle: MovementCycle metadata
            audio_processing_id: FK to audio processing record (required)
            biomechanics_import_id: Optional FK to biomechanics import
            synchronization_id: Optional FK to synchronization record
            cycles_file_path: Optional path to cycles pkl file on disk

        Returns:
            Movement cycle record ID if saved, None if DB not available or error
        """
        if not self.enabled or self.repository is None:
            return None

        try:
            record = self.repository.save_movement_cycle(
                cycle,
                audio_processing_id=audio_processing_id,
                biomechanics_import_id=biomechanics_import_id,
                synchronization_id=synchronization_id,
                cycles_file_path=cycles_file_path,
            )
            logger.debug(f"Saved movement cycle record: {record.id}")
            return record.id
        except Exception as e:
            logger.error(f"Failed to save movement cycle: {e}")
            return None


class RecordTracker:
    """Tracks database record IDs created during a processing session.

    Used to maintain FK relationships between audio, biomechanics, sync, and cycle records.
    """

    def __init__(self):
        """Initialize the tracker."""
        self.audio_processing_id: Optional[int] = None
        self.biomechanics_import_id: Optional[int] = None
        self.synchronization_ids: dict = {}  # maps pass_number -> sync record ID
        self.movement_cycle_ids: list = []

    def set_audio_processing(self, record_id: int) -> None:
        """Record audio processing record ID."""
        self.audio_processing_id = record_id
        logger.debug(f"Tracked audio processing record: {record_id}")

    def set_biomechanics_import(self, record_id: int) -> None:
        """Record biomechanics import record ID."""
        self.biomechanics_import_id = record_id
        logger.debug(f"Tracked biomechanics import record: {record_id}")

    def set_synchronization(self, pass_number: Optional[int], record_id: int) -> None:
        """Record synchronization record ID (indexed by pass_number for walk maneuvers)."""
        key = pass_number if pass_number is not None else "default"
        self.synchronization_ids[key] = record_id
        logger.debug(f"Tracked synchronization record (pass {key}): {record_id}")

    def add_movement_cycle(self, record_id: int) -> None:
        """Record a movement cycle record ID."""
        self.movement_cycle_ids.append(record_id)
        logger.debug(f"Tracked movement cycle record: {record_id}")

    def get_audio_processing(self) -> Optional[int]:
        """Get audio processing record ID."""
        return self.audio_processing_id

    def get_biomechanics_import(self) -> Optional[int]:
        """Get biomechanics import record ID."""
        return self.biomechanics_import_id

    def get_synchronization(self, pass_number: Optional[int] = None) -> Optional[int]:
        """Get synchronization record ID."""
        key = pass_number if pass_number is not None else "default"
        return self.synchronization_ids.get(key)

    def get_movement_cycles(self) -> list:
        """Get all movement cycle record IDs."""
        return self.movement_cycle_ids.copy()

    def summary(self) -> dict:
        """Get a summary of all tracked records."""
        return {
            "audio_processing_id": self.audio_processing_id,
            "biomechanics_import_id": self.biomechanics_import_id,
            "synchronization_ids": self.synchronization_ids.copy(),
            "movement_cycle_count": len(self.movement_cycle_ids),
        }
