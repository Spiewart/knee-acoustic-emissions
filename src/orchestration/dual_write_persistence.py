"""Phase 2B: Dual-Write Pattern - Database + Local Storage.

Saves processing results to both PostgreSQL database and local storage simultaneously
for resilience, debuggability, and audit trails.

Design: If database write fails, processing continues and results are saved locally.
This creates a searchable index of all processing attempts.
"""

from datetime import datetime
import json
import logging
from pathlib import Path

from sqlalchemy.orm import Session

from src.orchestration.database_persistence import (
    OrchestrationDatabasePersistence,
    RecordTracker,
)

logger = logging.getLogger(__name__)


class LocalStorageIndex:
    """Manages local storage of processing results for debugging and recovery.

    Creates a JSON index file that tracks:
    - What was processed (audio file, biomechanics, etc.)
    - When it was processed
    - Whether DB save succeeded
    - Local file paths for recovery

    Directory structure:
        ~/.ae_processing_index/
            participants/
                1011/
                    index.json        # Master index for participant
                    2024-01-29.json   # Daily log
                1012/
                    index.json
    """

    def __init__(self, index_root: Path | None = None):
        """Initialize local storage index.

        Args:
            index_root: Root directory for index files.
                       Defaults to ~/.ae_processing_index/
        """
        if index_root is None:
            index_root = Path.home() / ".ae_processing_index"

        self.index_root = index_root
        self.index_root.mkdir(parents=True, exist_ok=True)

    def record_audio_processing(
        self,
        study_id: str,
        audio_file: str,
        pkl_path: str,
        db_saved: bool,
        db_record_id: int | None = None,
    ) -> None:
        """Record audio processing attempt.

        Args:
            study_id: Participant ID (e.g., "1011")
            audio_file: Audio file name
            pkl_path: Path to pickle file
            db_saved: Whether database save succeeded
            db_record_id: Record ID if saved to DB
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "audio_processing",
            "audio_file": audio_file,
            "pkl_path": pkl_path,
            "db_saved": db_saved,
            "db_record_id": db_record_id,
        }
        self._append_to_index(study_id, entry)

    def record_biomechanics_import(
        self,
        study_id: str,
        biomech_file: str,
        db_saved: bool,
        db_record_id: int | None = None,
        audio_id: int | None = None,
    ) -> None:
        """Record biomechanics import attempt.

        Args:
            study_id: Participant ID
            biomech_file: Biomechanics file name
            db_saved: Whether database save succeeded
            db_record_id: Record ID if saved to DB
            audio_id: FK to audio processing record
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "biomechanics_import",
            "biomech_file": biomech_file,
            "db_saved": db_saved,
            "db_record_id": db_record_id,
            "audio_id": audio_id,
        }
        self._append_to_index(study_id, entry)

    def record_synchronization(
        self,
        study_id: str,
        maneuver: str,
        pass_number: int,
        db_saved: bool,
        db_record_id: int | None = None,
        audio_id: int | None = None,
        biomech_id: int | None = None,
    ) -> None:
        """Record synchronization attempt.

        Args:
            study_id: Participant ID
            maneuver: Maneuver type (walk, etc.)
            pass_number: Pass number for walk
            db_saved: Whether database save succeeded
            db_record_id: Record ID if saved to DB
            audio_id: FK to audio
            biomech_id: FK to biomechanics
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "synchronization",
            "maneuver": maneuver,
            "pass_number": pass_number,
            "db_saved": db_saved,
            "db_record_id": db_record_id,
            "audio_id": audio_id,
            "biomech_id": biomech_id,
        }
        self._append_to_index(study_id, entry)

    def record_movement_cycle(
        self,
        study_id: str,
        cycle_number: int,
        maneuver: str,
        db_saved: bool,
        db_record_id: int | None = None,
        sync_id: int | None = None,
    ) -> None:
        """Record movement cycle attempt.

        Args:
            study_id: Participant ID
            cycle_number: Cycle number
            maneuver: Maneuver type
            db_saved: Whether database save succeeded
            db_record_id: Record ID if saved to DB
            sync_id: FK to synchronization
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "movement_cycle",
            "cycle_number": cycle_number,
            "maneuver": maneuver,
            "db_saved": db_saved,
            "db_record_id": db_record_id,
            "sync_id": sync_id,
        }
        self._append_to_index(study_id, entry)

    def _append_to_index(self, study_id: str, entry: dict) -> None:
        """Append entry to participant index.

        Creates participant directory if needed and appends JSON entry.
        """
        try:
            participant_dir = self.index_root / "participants" / str(study_id)
            participant_dir.mkdir(parents=True, exist_ok=True)

            # Master index file
            index_file = participant_dir / "index.json"

            # Load existing or create new
            if index_file.exists():
                with open(index_file) as f:
                    data = json.load(f)
                    if not isinstance(data, dict):
                        data = {"entries": []}
            else:
                data = {"entries": []}

            # Append new entry
            if "entries" not in data:
                data["entries"] = []
            data["entries"].append(entry)

            # Write back
            with open(index_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to write to local index: {e}")

    def get_unsynced_entries(self, study_id: str) -> list[dict]:
        """Get entries that failed to sync to database.

        Useful for retry logic or manual recovery.

        Returns:
            List of entries where db_saved=False
        """
        try:
            index_file = self.index_root / "participants" / str(study_id) / "index.json"
            if not index_file.exists():
                return []

            with open(index_file) as f:
                data = json.load(f)

            return [e for e in data.get("entries", []) if not e.get("db_saved", False)]

        except Exception as e:
            logger.warning(f"Failed to read local index: {e}")
            return []


class DualWritePersistence:
    """Wraps OrchestrationDatabasePersistence with dual-write (DB + local storage).

    Pattern: Try to save to database, always save to local storage.
    If DB fails, local storage provides recovery mechanism.

    Usage:
        persistence = DualWritePersistence(
            db_session=session,
            local_storage_root=Path.home() / ".ae_processing"
        )

        # Returns DB record ID if successful, None if DB failed but local saved
        audio_id = persistence.save_audio_processing(audio, pkl_path)
    """

    def __init__(
        self,
        db_session: Session | None = None,
        local_storage_root: Path | None = None,
    ):
        """Initialize dual-write persistence.

        Args:
            db_session: Optional SQLAlchemy Session for database writes
            local_storage_root: Root directory for local index
        """
        self.db_persistence = OrchestrationDatabasePersistence(session=db_session)
        self.local_index = LocalStorageIndex(local_storage_root)
        self.tracker = RecordTracker() if db_session else None

    @property
    def enabled(self) -> bool:
        """Check if database persistence is enabled."""
        return self.db_persistence.enabled

    def save_audio_processing(
        self,
        audio,
        pkl_file_path: str,
        biomechanics_import_id: int | None = None,
    ) -> int | None:
        """Save audio processing with dual-write.

        Returns record ID if database save succeeded, None if local-only save.
        """
        db_record_id = None

        # Try database first
        try:
            if self.db_persistence.enabled:
                db_record_id = self.db_persistence.save_audio_processing(audio, pkl_file_path, biomechanics_import_id)
        except Exception as e:
            logger.warning(f"Database save failed: {e}, continuing with local save")

        # Always save to local storage
        try:
            self.local_index.record_audio_processing(
                study_id=audio.study_id,
                audio_file=audio.audio_file_name,
                pkl_path=pkl_file_path,
                db_saved=db_record_id is not None,
                db_record_id=db_record_id,
            )
        except Exception as e:
            logger.warning(f"Local save failed: {e}")

        return db_record_id

    def save_biomechanics_import(
        self,
        biomech,
        audio_processing_id: int | None = None,
    ) -> int | None:
        """Save biomechanics import with dual-write."""
        db_record_id = None

        try:
            if self.db_persistence.enabled:
                db_record_id = self.db_persistence.save_biomechanics_import(biomech, audio_processing_id)
        except Exception as e:
            logger.warning(f"Database save failed: {e}, continuing with local save")

        try:
            self.local_index.record_biomechanics_import(
                study_id=biomech.study_id,
                biomech_file=biomech.biomechanics_file,
                db_saved=db_record_id is not None,
                db_record_id=db_record_id,
                audio_id=audio_processing_id,
            )
        except Exception as e:
            logger.warning(f"Local save failed: {e}")

        return db_record_id

    def save_synchronization(
        self,
        sync,
        audio_processing_id: int | None = None,
        biomechanics_import_id: int | None = None,
        sync_file_path: str | None = None,
    ) -> int | None:
        """Save synchronization with dual-write."""
        db_record_id = None

        try:
            if self.db_persistence.enabled and audio_processing_id is not None and biomechanics_import_id is not None:
                db_record_id = self.db_persistence.save_synchronization(
                    sync, audio_processing_id, biomechanics_import_id, sync_file_path
                )
        except Exception as e:
            logger.warning(f"Database save failed: {e}, continuing with local save")

        try:
            pass_num = getattr(sync, "pass_number", None) or 1
            self.local_index.record_synchronization(
                study_id=sync.study_id,
                maneuver=sync.maneuver,
                pass_number=pass_num,
                db_saved=db_record_id is not None,
                db_record_id=db_record_id,
                audio_id=audio_processing_id,
                biomech_id=biomechanics_import_id,
            )
        except Exception as e:
            logger.warning(f"Local save failed: {e}")

        return db_record_id

    def save_movement_cycle(
        self,
        cycle,
        audio_processing_id: int | None = None,
        biomechanics_import_id: int | None = None,
        synchronization_id: int | None = None,
        cycles_file_path: str | None = None,
    ) -> int | None:
        """Save movement cycle with dual-write."""
        db_record_id = None

        try:
            if self.db_persistence.enabled and audio_processing_id is not None:
                db_record_id = self.db_persistence.save_movement_cycle(
                    cycle,
                    audio_processing_id,
                    biomechanics_import_id,
                    synchronization_id,
                    cycles_file_path,
                )
        except Exception as e:
            logger.warning(f"Database save failed: {e}, continuing with local save")

        try:
            cycle_num = getattr(cycle, "cycle_number", None) or 1
            self.local_index.record_movement_cycle(
                study_id=cycle.study_id,
                cycle_number=cycle_num,
                maneuver=cycle.maneuver,
                db_saved=db_record_id is not None,
                db_record_id=db_record_id,
                sync_id=synchronization_id,
            )
        except Exception as e:
            logger.warning(f"Local save failed: {e}")

        return db_record_id
