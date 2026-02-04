"""Optional database persistence integration for participant processing.

This module provides a thin wrapper around ParticipantProcessor that optionally
saves processed records to PostgreSQL without modifying the core orchestration logic.

Key Design Principles:
- Optional: Persistence is disabled if no SQLAlchemy Session is provided
- Non-invasive: No changes to ParticipantProcessor class
- Backward compatible: Works with or without database
- Graceful degradation: Processing continues if database unavailable
"""

import logging
from pathlib import Path
from typing import Literal, Optional

from sqlalchemy.orm import Session

from src.orchestration.database_persistence import (
    OrchestrationDatabasePersistence,
    RecordTracker,
)
from src.orchestration.dual_write_persistence import DualWritePersistence
from src.orchestration.participant_processor import ParticipantProcessor

logger = logging.getLogger(__name__)


class PersistentParticipantProcessor:
    """Wrapper around ParticipantProcessor that optionally saves records to database.

    Usage:
        # Without database persistence (standard operation)
        processor = PersistentParticipantProcessor(
            participant_dir=Path("#1011"),
            db_session=None,  # Database disabled
        )
        success = processor.process()

        # With database persistence
        from src.orchestration.cli_db_helpers import create_db_session
        session = create_db_session("postgresql://...")
        try:
            processor = PersistentParticipantProcessor(
                participant_dir=Path("#1011"),
                db_session=session,
            )
            success = processor.process()
        finally:
            session.close()
    """

    def __init__(
        self,
        participant_dir: Path,
        biomechanics_type: Optional[str] = None,
        db_session: Optional[Session] = None,
    ):
        """Initialize persistent participant processor.

        Args:
            participant_dir: Path to participant directory (e.g., /path/to/#1011)
            biomechanics_type: Optional biomechanics type hint
            db_session: Optional SQLAlchemy Session for database persistence
                       If None, processing runs without database saves
        """
        self.participant_dir = participant_dir
        self.biomechanics_type = biomechanics_type

        # Core processor (non-database-aware)
        self.processor = ParticipantProcessor(
            participant_dir=participant_dir,
            biomechanics_type=biomechanics_type,
        )

        # Optional persistence layer with dual-write (DB + local storage)
        self.persistence = DualWritePersistence(
            db_session=db_session,
            local_storage_root=Path.home() / ".ae_processing_index",
        )
        self.tracker = self.persistence.tracker if db_session else None

        if self.persistence.enabled:
            logger.info(
                "Processing participant %s with database persistence enabled (dual-write)",
                participant_dir.name,
            )
        else:
            logger.info(
                "Processing participant %s without database persistence (local-only)",
                participant_dir.name,
            )

    def process(
        self,
        entrypoint: Literal["bin", "sync", "cycles"] = "sync",
        knee: Optional[str] = None,
        maneuver: Optional[str] = None,
    ) -> bool:
        """Process participant with optional database persistence.

        Delegates to core ParticipantProcessor.process(), then optionally
        saves records to database if persistence is enabled.

        Args:
            entrypoint: Stage to start from: 'bin' | 'sync' | 'cycles'
            knee: Specify which knee to process ('left' or 'right')
            maneuver: Specify which maneuver to process ('walk', 'fe', or 'sts')

        Returns:
            True if processing succeeded, False otherwise
        """
        try:
            # Run core processing (this is where all the work happens)
            success = self.processor.process(
                entrypoint=entrypoint,
                knee=knee,
                maneuver=maneuver,
            )

            if not success:
                logger.warning("Core participant processing failed")
                return False

            # Optionally persist records if database is available
            if self.persistence.enabled and self.tracker:
                self._persist_processor_results()

            return True

        except Exception as e:
            logger.error(f"Error in persistent participant processor: {e}")
            return False

    def _persist_processor_results(self) -> None:
        """Save processed results to database with proper FK cascading.

        Phase 2A Implementation: Full record extraction and cascading saves.

        Saves records in this order to maintain FK relationships:
        1. AudioProcessing → get audio_id
        2. BiomechanicsImport (with audio_id FK) → get biomech_id
        3. Synchronization (with both FKs) → get sync_id per pass
        4. MovementCycle (with all FKs) for each cycle

        This creates a complete audit trail linking every cycle back to
        the original audio file and biomechanics import.
        """
        if not self.persistence.enabled or not self.tracker:
            return

        logger.debug("Persisting processor results to database (Phase 2A)")

        try:
            # Iterate through all processed knees
            for knee_side, knee_processor in self.processor.knee_processors.items():
                logger.debug(f"Persisting {knee_side} knee results")

                # Iterate through all processed maneuvers for this knee
                for maneuver_key, maneuver_processor in knee_processor.maneuver_processors.items():
                    logger.debug(f"Persisting {maneuver_key} maneuver results")

                    # Phase 2A Step 1: Save AudioProcessing record
                    audio_id = self._persist_audio_record(maneuver_processor)
                    if audio_id:
                        self.tracker.set_audio_processing(audio_id)

                    # Phase 2A Step 2: Save BiomechanicsImport record with audio_id FK
                    biomech_id = self._persist_biomechanics_record(
                        maneuver_processor, audio_id
                    )
                    if biomech_id:
                        self.tracker.set_biomechanics_import(biomech_id)

                    # Phase 2A Step 3 & 4: Save Synchronization records with cycles
                    self._persist_synchronization_and_cycles(
                        maneuver_processor, audio_id, biomech_id
                    )

            summary = self.tracker.summary()
            logger.info(
                f"Database persistence complete: {summary['audio_processing_id']} "
                f"audio, {summary['biomechanics_import_id']} biomech, "
                f"{len(summary['synchronization_ids'])} sync, "
                f"{summary['movement_cycle_count']} cycles"
            )

        except Exception as e:
            logger.error(f"Error persisting processor results: {e}")

    def _persist_audio_record(self, maneuver_processor) -> int | None:
        """Persist AudioProcessing record from maneuver processor.

        Returns record ID if successful, None otherwise.
        """
        if not hasattr(maneuver_processor, 'audio') or not maneuver_processor.audio:
            logger.debug("No audio record available to persist")
            return None

        if not maneuver_processor.audio.record:
            logger.debug("Audio record not processed")
            return None

        try:
            audio_id = self.persistence.save_audio_processing(
                audio=maneuver_processor.audio.record,
                pkl_file_path=str(maneuver_processor.audio.pkl_path) if maneuver_processor.audio.pkl_path else None,
                biomechanics_import_id=None,  # Will be updated after biomechanics save
            )

            if audio_id:
                logger.debug(f"Persisted AudioProcessing record: {audio_id}")
            return audio_id

        except Exception as e:
            logger.warning(f"Failed to persist audio record: {e}")
            return None

    def _persist_biomechanics_record(
        self, maneuver_processor, audio_id: int | None
    ) -> int | None:
        """Persist BiomechanicsImport record with FK to audio.

        Returns record ID if successful, None otherwise.
        """
        if not hasattr(maneuver_processor, 'biomechanics') or not maneuver_processor.biomechanics:
            logger.debug("No biomechanics record available to persist")
            return None

        if not maneuver_processor.biomechanics.record:
            logger.debug("Biomechanics record not processed")
            return None

        try:
            biomech_id = self.persistence.save_biomechanics_import(
                biomech=maneuver_processor.biomechanics.record,
                audio_processing_id=audio_id,  # FK to audio
            )

            if biomech_id:
                logger.debug(f"Persisted BiomechanicsImport record: {biomech_id}")
            return biomech_id

        except Exception as e:
            logger.warning(f"Failed to persist biomechanics record: {e}")
            return None

    def _persist_synchronization_and_cycles(
        self,
        maneuver_processor,
        audio_id: int | None,
        biomech_id: int | None,
    ) -> None:
        """Persist Synchronization records with their movement cycles.

        Handles the FK cascade: audio_id + biomech_id → sync → cycles
        """
        if not hasattr(maneuver_processor, 'synced_files') or not maneuver_processor.synced_files:
            logger.debug("No synchronization records available to persist")
            return

        try:
            # Iterate through each synchronization file (pass)
            for pass_number, sync_data in enumerate(maneuver_processor.synced_files, start=1):
                if not sync_data or not sync_data.record:
                    logger.debug(f"No sync record for pass {pass_number}")
                    continue

                # Phase 2A Step 3: Save Synchronization with both FKs
                sync_id = self.persistence.save_synchronization(
                    sync=sync_data.record,
                    audio_processing_id=audio_id,
                    biomechanics_import_id=biomech_id,
                    sync_file_path=str(sync_data.output_path) if sync_data.output_path else None,
                )

                if sync_id:
                    self.tracker.set_synchronization(pass_number=pass_number, record_id=sync_id)
                    logger.debug(f"Persisted Synchronization record: {sync_id} (pass {pass_number})")

                # Phase 2A Step 4: Save MovementCycle records with all FKs
                self._persist_movement_cycles(
                    maneuver_processor, pass_number, audio_id, biomech_id, sync_id
                )

        except Exception as e:
            logger.warning(f"Failed to persist synchronization/cycles: {e}")

    def _persist_movement_cycles(
        self,
        maneuver_processor,
        pass_number: int,
        audio_id: int | None,
        biomech_id: int | None,
        sync_id: int | None,
    ) -> None:
        """Persist MovementCycle records for a sync pass.

        Each cycle gets FKs to audio, biomechanics, and synchronization.
        """
        if not hasattr(maneuver_processor, 'processed_cycles') or not maneuver_processor.processed_cycles:
            logger.debug(f"No cycle records for pass {pass_number}")
            return

        # Filter cycles for this pass (if pass_number tracking is available)
        cycles_for_pass = maneuver_processor.processed_cycles

        try:
            for cycle_record in cycles_for_pass:
                if not cycle_record:
                    continue

                cycle_id = self.persistence.save_movement_cycle(
                    cycle=cycle_record,
                    audio_processing_id=audio_id,
                    biomechanics_import_id=biomech_id,
                    synchronization_id=sync_id,
                    cycles_file_path=None,
                )

                if cycle_id:
                    self.tracker.add_movement_cycle(cycle_id)
                    logger.debug(
                        f"Persisted MovementCycle: {cycle_id} "
                        f"(audio_id={audio_id}, biomech_id={biomech_id}, sync_id={sync_id})"
                    )

        except Exception as e:
            logger.warning(f"Failed to persist cycles for pass {pass_number}: {e}")


def create_persistent_processor(
    participant_dir: Path,
    biomechanics_type: Optional[str] = None,
    db_url: Optional[str] = None,
) -> PersistentParticipantProcessor:
    """Factory function to create a processor with optional database persistence.

    Args:
        participant_dir: Path to participant directory
        biomechanics_type: Optional biomechanics type hint
        db_url: Optional database URL. If provided, creates session and enables persistence.
               If None, processing runs without database.

    Returns:
        PersistentParticipantProcessor instance

    Example:
        # Without database
        processor = create_persistent_processor(Path("#1011"))

        # With database
        processor = create_persistent_processor(
            Path("#1011"),
            db_url="postgresql://user@localhost/acoustic_emissions"
        )
    """
    if db_url:
        from src.orchestration.cli_db_helpers import create_db_session
        try:
            session = create_db_session(db_url)
            return PersistentParticipantProcessor(
                participant_dir=participant_dir,
                biomechanics_type=biomechanics_type,
                db_session=session,
            )
        except Exception as e:
            logger.warning(f"Could not create database session: {e}. Running without persistence.")
            return PersistentParticipantProcessor(
                participant_dir=participant_dir,
                biomechanics_type=biomechanics_type,
                db_session=None,
            )

    return PersistentParticipantProcessor(
        participant_dir=participant_dir,
        biomechanics_type=biomechanics_type,
        db_session=None,
    )
