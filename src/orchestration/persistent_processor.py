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
        
        # Optional persistence layer
        self.persistence = OrchestrationDatabasePersistence(session=db_session)
        self.tracker = RecordTracker() if db_session else None
        
        if self.persistence.enabled:
            logger.info(
                "Processing participant %s with database persistence enabled",
                participant_dir.name,
            )
        else:
            logger.info(
                "Processing participant %s without database persistence",
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
        """Save processed results to database (if persistence enabled).
        
        Iterates through processed knee/maneuver combinations and saves
        AudioProcessing, BiomechanicsImport, Synchronization, and MovementCycle
        records with proper FK relationships.
        
        Note: This is a placeholder for full implementation. In production,
        would need to extract records from processor and save via persistence layer.
        """
        if not self.persistence.enabled or not self.tracker:
            return
        
        logger.debug("Persisting processor results to database")
        
        # TODO: Extract records from self.processor.knee_processors
        # and save via self.persistence layer with proper FK tracking
        # This would look something like:
        #
        # for knee_side, knee_processor in self.processor.knee_processors.items():
        #     for maneuver_key, maneuver_processor in knee_processor.maneuver_processors.items():
        #         if maneuver_processor.audio.record:
        #             audio_id = self.persistence.save_audio_processing(
        #                 audio=maneuver_processor.audio.record,
        #                 pkl_file_path=str(maneuver_processor.audio.pkl_path),
        #             )
        #             if audio_id:
        #                 self.tracker.set_audio_processing(audio_id)
        #
        # etc. for biomechanics, synchronization, cycles...
        
        summary = self.tracker.summary()
        logger.debug(f"Persistence summary: {summary}")


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
