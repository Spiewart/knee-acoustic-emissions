"""Database upsert tests replacing local reference IDs."""

from datetime import datetime
import hashlib
import json
from typing import Any, Dict, Optional


class LocalReferenceIDGenerator:
    """Generate deterministic IDs for audio/biomechanics/sync/cycle records."""

    @staticmethod
    def audio_processing_id(
        participant_id: int,
        knee: str,
        maneuver: str,
        recording_date: datetime,
        audio_file_name: str,
    ) -> int:
        """Generate deterministic ID for audio processing record."""
        content = f"{participant_id}_{knee}_{maneuver}_{recording_date.isoformat()}_{audio_file_name}"
        hash_value = hashlib.md5(content.encode()).hexdigest()
        return int(hash_value[:8], 16)

    @staticmethod
    def biomechanics_import_id(
        participant_id: int,
        knee: str,
        maneuver: str,
        biomechanics_file: str,
        recording_date: datetime,
    ) -> int:
        """Generate deterministic ID for biomechanics import record."""
        content = f"{participant_id}_{knee}_{maneuver}_{biomechanics_file}_{recording_date.isoformat()}"
        hash_value = hashlib.md5(content.encode()).hexdigest()
        return int(hash_value[:8], 16)

    @staticmethod
    def synchronization_id(
        audio_id: int,
        biomechanics_id: int,
        pass_number: int,
    ) -> int:
        """Generate deterministic ID for synchronization record."""
        content = f"{audio_id}_{biomechanics_id}_{pass_number}"
        hash_value = hashlib.md5(content.encode()).hexdigest()
        return int(hash_value[:8], 16)

    @staticmethod
    def movement_cycle_id(
        audio_id: int,
        cycle_index: int,
        cycle_file: str,
    ) -> int:
        """Generate deterministic ID for movement cycle record."""
        content = f"{audio_id}_{cycle_index}_{cycle_file}"
        hash_value = hashlib.md5(content.encode()).hexdigest()
        return int(hash_value[:8], 16)


class LocalReferenceRecord:
    """Data class for a local reference record."""

    def __init__(
        self,
        local_id: int,
        participant_id: int,
        study: str,
        record_type: str,
        file_name: str,
        file_path: str,
        related_ids: Optional[Dict[str, int]] = None,
        **kwargs: Any,
    ):
        self.local_id = local_id
        self.participant_id = participant_id
        self.study = study
        self.record_type = record_type  # 'audio', 'biomechanics', 'sync', 'cycle'
        self.file_name = file_name
        self.file_path = file_path
        self.related_ids = related_ids or {}
        self.metadata = kwargs
        # Expose commonly referenced fields
        self.pass_number = kwargs.get("pass_number")
        self.cycle_index = kwargs.get("cycle_index")


class LocalReferenceRegistry:
    """Registry for mapping local IDs to file paths and metadata."""

    _global_registry: Dict[int, LocalReferenceRecord] = {}
    _by_path: Dict[str, LocalReferenceRecord] = {}

    def register_audio(
        self,
        local_id: int,
        participant_id: int,
        study: str,
        knee: str,
        maneuver: str,
        file_path: str,
        recording_date: datetime,
    ) -> None:
        """Register audio file in registry."""
        file_name = file_path.split("/")[-1]
        record = LocalReferenceRecord(
            local_id=local_id,
            participant_id=participant_id,
            study=study,
            record_type="audio",
            file_name=file_name,
            file_path=file_path,
            knee=knee,
            maneuver=maneuver,
            recording_date=recording_date,
        )
        self._global_registry[local_id] = record
        self._by_path[file_path] = record

    def register_biomechanics(
        self,
        local_id: int,
        participant_id: int,
        study: str,
        knee: str,
        maneuver: str,
        file_path: str,
        recording_date: datetime,
    ) -> None:
        """Register biomechanics file in registry."""
        file_name = file_path.split("/")[-1]
        record = LocalReferenceRecord(
            local_id=local_id,
            participant_id=participant_id,
            study=study,
            record_type="biomechanics",
            file_name=file_name,
            file_path=file_path,
            knee=knee,
            maneuver=maneuver,
            recording_date=recording_date,
        )
        self._global_registry[local_id] = record
        self._by_path[file_path] = record

    def register_synchronization(
        self,
        local_id: int,
        participant_id: int,
        study: str,
        audio_id: int,
        biomechanics_id: int,
        knee: str,
        maneuver: str,
        pass_number: Optional[int] = None,
    ) -> None:
        """Register synchronization record in registry."""
        record = LocalReferenceRecord(
            local_id=local_id,
            participant_id=participant_id,
            study=study,
            record_type="sync",
            file_name=f"sync_{local_id}",
            file_path="",
            related_ids={"audio_id": audio_id, "biomechanics_id": biomechanics_id},
            knee=knee,
            maneuver=maneuver,
            pass_number=pass_number,
        )
        self._global_registry[local_id] = record

    def register_movement_cycle(
        self,
        local_id: int,
        participant_id: int,
        study: str,
        audio_id: int,
        cycle_index: int,
        knee: str,
        maneuver: str,
        pass_number: Optional[int] = None,
    ) -> None:
        """Register movement cycle record in registry."""
        record = LocalReferenceRecord(
            local_id=local_id,
            participant_id=participant_id,
            study=study,
            record_type="cycle",
            file_name=f"cycle_{local_id}",
            file_path="",
            related_ids={"audio_id": audio_id},
            knee=knee,
            maneuver=maneuver,
            cycle_index=cycle_index,
            pass_number=pass_number,
        )
        self._global_registry[local_id] = record

    def get_by_id(self, local_id: int) -> Optional[LocalReferenceRecord]:
        """Get record by local ID."""
        return self._global_registry.get(local_id)

    def get_by_file_path(self, file_path: str) -> Optional[LocalReferenceRecord]:
        """Get record by file path."""
        return self._by_path.get(file_path)

    def find_audio_files_for_participant(self, participant_id: int) -> list:
        """Find all audio files for a participant."""
        return [
            rec
            for rec in self._global_registry.values()
            if rec.participant_id == participant_id and rec.record_type == "audio"
        ]

    def find_by_study(self, study: str) -> list:
        """Find all records for a study."""
        return [rec for rec in self._global_registry.values() if rec.study == study]

    def find_sync_for_audio(self, audio_id: int) -> Optional[LocalReferenceRecord]:
        """Find synchronization record associated with an audio ID."""
        for rec in self._global_registry.values():
            if rec.record_type == "sync" and rec.related_ids.get("audio_id") == audio_id:
                return rec
        return None

    def find_cycles_for_audio(self, audio_id: int) -> list:
        """Find all movement cycles associated with an audio ID."""
        cycles = [
            rec
            for rec in self._global_registry.values()
            if rec.record_type == "cycle" and rec.related_ids.get("audio_id") == audio_id
        ]
        return sorted(cycles, key=lambda rec: rec.pass_number or 0)

    def to_json(self) -> str:
        """Serialize registry to JSON string."""
        payload = []
        for rec in self._global_registry.values():
            data = {
                "local_id": rec.local_id,
                "participant_id": rec.participant_id,
                "study": rec.study,
                "record_type": rec.record_type,
                "file_name": rec.file_name,
                "file_path": rec.file_path,
                "related_ids": rec.related_ids,
                "metadata": rec.metadata,
            }
            payload.append(data)
        return json.dumps(payload, default=str)

    def from_json(self, json_str: str) -> None:
        """Load registry from JSON string."""
        records = json.loads(json_str)
        for rec in records:
            record = LocalReferenceRecord(
                local_id=rec["local_id"],
                participant_id=rec["participant_id"],
                study=rec["study"],
                record_type=rec["record_type"],
                file_name=rec.get("file_name", ""),
                file_path=rec.get("file_path", ""),
                related_ids=rec.get("related_ids", {}),
                **rec.get("metadata", {}),
            )
            self._global_registry[record.local_id] = record
            if record.file_path:
                self._by_path[record.file_path] = record

    def save_to_file(self, file_path) -> None:
        """Persist registry to a JSON file."""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    def load_from_file(self, file_path) -> None:
        """Load registry from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            self.from_json(f.read())


def reset_global_registry() -> None:
    """Reset global registry for test isolation."""
    LocalReferenceRegistry._global_registry.clear()
    LocalReferenceRegistry._by_path.clear()


_GLOBAL_REGISTRY: Optional[LocalReferenceRegistry] = None


def get_global_registry() -> LocalReferenceRegistry:
    """Return a process-wide registry singleton."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = LocalReferenceRegistry()
    return _GLOBAL_REGISTRY


def test_audio_upsert_preserves_primary_key(repository, audio_processing_factory):
    audio = audio_processing_factory(study="AOA", study_id=6001, audio_file_name="AOA6001_fe_sync")
    first = repository.save_audio_processing(audio)

    updated = audio_processing_factory(
        study="AOA",
        study_id=6001,
        audio_file_name="AOA6001_fe_sync",
        firmware_version=2,
    )
    second = repository.save_audio_processing(updated)

    assert first.id == second.id
    assert second.firmware_version == 2


def test_sync_upsert_preserves_primary_key(
    repository, synchronization_factory, audio_processing_factory, biomechanics_import_factory
):
    audio = audio_processing_factory(study="AOA", study_id=6002, audio_file_name="AOA6002_walk_audio")
    audio_record = repository.save_audio_processing(audio)

    biomech = biomechanics_import_factory(study="AOA", study_id=6002, biomechanics_file="AOA6002_walk_biomech.xlsx")
    biomech_record = repository.save_biomechanics_import(biomech, audio_processing_id=audio_record.id)

    sync = synchronization_factory(
        study="AOA",
        study_id=6002,
        audio_processing_id=audio_record.id,
        biomechanics_import_id=biomech_record.id,
        sync_file_name="AOA6002_walk_sync_slow_1",
    )
    first = repository.save_synchronization(
        sync, audio_processing_id=audio_record.id, biomechanics_import_id=biomech_record.id
    )

    updated = synchronization_factory(
        study="AOA",
        study_id=6002,
        audio_processing_id=audio_record.id,
        biomechanics_import_id=biomech_record.id,
        sync_file_name="AOA6002_walk_sync_slow_1",
        sync_duration=200.0,
    )
    second = repository.save_synchronization(
        updated, audio_processing_id=audio_record.id, biomechanics_import_id=biomech_record.id
    )

    assert first.id == second.id
    assert second.sync_duration == 200.0


class TestLocalReferenceIDGenerator:
    """Test deterministic ID generation."""

    def test_audio_id_is_deterministic(self):
        """Same audio file always produces same ID."""
        id1 = LocalReferenceIDGenerator.audio_processing_id(
            participant_id=1016,
            knee="right",
            maneuver="walk",
            recording_date=datetime(2024, 1, 15),
            audio_file_name="1016_right_walk_20240115.bin",
        )
        id2 = LocalReferenceIDGenerator.audio_processing_id(
            participant_id=1016,
            knee="right",
            maneuver="walk",
            recording_date=datetime(2024, 1, 15),
            audio_file_name="1016_right_walk_20240115.bin",
        )
        assert id1 == id2, "Same audio file should produce same ID"
        assert id1 > 0, "ID should be positive"

    def test_different_files_have_different_ids(self):
        """Different audio files produce different IDs."""
        id_walk = LocalReferenceIDGenerator.audio_processing_id(
            participant_id=1016,
            knee="right",
            maneuver="walk",
            recording_date=datetime(2024, 1, 15),
            audio_file_name="1016_right_walk_20240115.bin",
        )
        id_sts = LocalReferenceIDGenerator.audio_processing_id(
            participant_id=1016,
            knee="right",
            maneuver="sts",
            recording_date=datetime(2024, 1, 15),
            audio_file_name="1016_right_sts_20240115.bin",
        )
        assert id_walk != id_sts, "Different maneuvers should have different IDs"

    def test_biomechanics_id_is_deterministic(self):
        """Same biomechanics file always produces same ID."""
        id1 = LocalReferenceIDGenerator.biomechanics_import_id(
            participant_id=1016,
            knee="right",
            maneuver="walk",
            biomechanics_file="1016_right_walk_20240115_biomech.xlsx",
            recording_date=datetime(2024, 1, 15),
        )
        id2 = LocalReferenceIDGenerator.biomechanics_import_id(
            participant_id=1016,
            knee="right",
            maneuver="walk",
            biomechanics_file="1016_right_walk_20240115_biomech.xlsx",
            recording_date=datetime(2024, 1, 15),
        )
        assert id1 == id2, "Same biomechanics file should produce same ID"
        assert id1 > 0, "ID should be positive"

    def test_sync_id_combines_audio_and_biomechanics(self):
        """Sync ID depends on both audio and biomechanics IDs."""
        audio_id = 12345
        biomech_id = 67890

        sync_id_1 = LocalReferenceIDGenerator.synchronization_id(
            audio_id=audio_id,
            biomechanics_id=biomech_id,
            pass_number=1,
        )

        # Different pass number should produce different ID
        sync_id_2 = LocalReferenceIDGenerator.synchronization_id(
            audio_id=audio_id,
            biomechanics_id=biomech_id,
            pass_number=2,
        )
        assert sync_id_1 != sync_id_2, "Different pass numbers should produce different sync IDs"

    def test_movement_cycle_id_includes_cycle_index(self):
        """Cycle IDs differ by index."""
        audio_id = 12345

        cycle_id_0 = LocalReferenceIDGenerator.movement_cycle_id(
            audio_id=audio_id,
            cycle_index=0,
            cycle_file="cycles_0.pkl",
        )

        cycle_id_1 = LocalReferenceIDGenerator.movement_cycle_id(
            audio_id=audio_id,
            cycle_index=1,
            cycle_file="cycles_1.pkl",
        )

        assert cycle_id_0 != cycle_id_1, "Different cycle indices should produce different IDs"


class TestLocalReferenceRegistry:
    """Test backwards lookup from ID to files/metadata."""

    def setup_method(self):
        """Reset global registry before each test."""
        reset_global_registry()

    def test_register_and_lookup_audio(self):
        """Register audio file and retrieve by ID."""
        registry = LocalReferenceRegistry()

        local_id = 12345
        registry.register_audio(
            local_id=local_id,
            participant_id=1016,
            study="AOA",
            knee="right",
            maneuver="walk",
            file_path="/path/to/1016_right_walk.bin",
            recording_date=datetime(2024, 1, 15),
        )

        record = registry.get_by_id(local_id)
        assert record is not None
        assert record.participant_id == 1016
        assert record.file_name == "1016_right_walk.bin"
        assert record.record_type == "audio"

    def test_register_and_lookup_by_file_path(self):
        """Register and retrieve by file path."""
        registry = LocalReferenceRegistry()
        file_path = "/data/participant_1016/audio.bin"

        local_id = 12345
        registry.register_audio(
            local_id=local_id,
            participant_id=1016,
            study="AOA",
            knee="right",
            maneuver="walk",
            file_path=file_path,
            recording_date=datetime(2024, 1, 15),
        )

        record = registry.get_by_file_path(file_path)
        assert record is not None
        assert record.local_id == local_id

    def test_find_audio_files_for_participant(self):
        """Find all audio files for a participant."""
        registry = LocalReferenceRegistry()

        # Register multiple audio files for same participant
        registry.register_audio(
            local_id=100,
            participant_id=1016,
            study="AOA",
            knee="right",
            maneuver="walk",
            file_path="/path/to/walk_right.bin",
            recording_date=datetime(2024, 1, 15),
        )
        registry.register_audio(
            local_id=101,
            participant_id=1016,
            study="AOA",
            knee="left",
            maneuver="sts",
            file_path="/path/to/sts_left.bin",
            recording_date=datetime(2024, 1, 16),
        )

        audio_files = registry.find_audio_files_for_participant(1016)
        assert len(audio_files) == 2
        assert all(r.record_type == "audio" for r in audio_files)

    def test_find_sync_for_audio(self):
        """Find sync record associated with audio."""
        registry = LocalReferenceRegistry()
        audio_id = 100
        biomech_id = 200
        sync_id = 300

        registry.register_audio(
            local_id=audio_id,
            participant_id=1016,
            study="AOA",
            knee="right",
            maneuver="walk",
            file_path="/path/to/audio.bin",
            recording_date=datetime(2024, 1, 15),
        )

        registry.register_synchronization(
            local_id=sync_id,
            participant_id=1016,
            study="AOA",
            audio_id=audio_id,
            biomechanics_id=biomech_id,
            knee="right",
            maneuver="walk",
        )

        sync = registry.find_sync_for_audio(audio_id)
        assert sync is not None
        assert sync.local_id == sync_id
        assert sync.related_ids["audio_id"] == audio_id

    def test_find_cycles_for_audio(self):
        """Find all movement cycles for an audio file."""
        registry = LocalReferenceRegistry()
        audio_id = 100

        for i in range(3):
            registry.register_movement_cycle(
                local_id=300 + i,
                participant_id=1016,
                study="AOA",
                audio_id=audio_id,
                cycle_index=i,
                knee="right",
                maneuver="walk",
                pass_number=i + 1,
            )

        cycles = registry.find_cycles_for_audio(audio_id)
        assert len(cycles) == 3
        assert [c.pass_number for c in cycles] == [1, 2, 3]

    def test_registry_persistence_to_json(self):
        """Save and load registry as JSON."""
        registry1 = LocalReferenceRegistry()

        registry1.register_audio(
            local_id=100,
            participant_id=1016,
            study="AOA",
            knee="right",
            maneuver="walk",
            file_path="/path/to/audio.bin",
            recording_date=datetime(2024, 1, 15),
        )

        # Export to JSON
        json_str = registry1.to_json()
        assert "1016" in json_str
        assert "audio" in json_str

        # Import to new registry
        registry2 = LocalReferenceRegistry()
        registry2.from_json(json_str)

        record = registry2.get_by_id(100)
        assert record is not None
        assert record.participant_id == 1016

    def test_registry_file_persistence(self, tmp_path):
        """Save and load registry from file."""
        registry_file = tmp_path / "registry.json"

        registry1 = LocalReferenceRegistry()
        registry1.register_audio(
            local_id=100,
            participant_id=1016,
            study="AOA",
            knee="right",
            maneuver="walk",
            file_path="/path/to/audio.bin",
            recording_date=datetime(2024, 1, 15),
        )
        registry1.save_to_file(registry_file)

        assert registry_file.exists()

        registry2 = LocalReferenceRegistry()
        registry2.load_from_file(registry_file)

        record = registry2.get_by_id(100)
        assert record is not None
        assert record.participant_id == 1016


class TestOfflineWorkflow:
    """Test complete offline workflow without database."""

    def test_workflow_same_ids_across_excel_roundtrips(self):
        """Verify IDs remain consistent across multiple Excel save/load cycles."""
        # Simulate: Generate IDs for files
        audio_id = LocalReferenceIDGenerator.audio_processing_id(
            participant_id=1016,
            knee="right",
            maneuver="walk",
            recording_date=datetime(2024, 1, 15),
            audio_file_name="1016_right_walk.bin",
        )

        biomech_id = LocalReferenceIDGenerator.biomechanics_import_id(
            participant_id=1016,
            knee="right",
            maneuver="walk",
            biomechanics_file="1016_right_walk_biomech.xlsx",
            recording_date=datetime(2024, 1, 15),
        )

        sync_id = LocalReferenceIDGenerator.synchronization_id(
            audio_id=audio_id,
            biomechanics_id=biomech_id,
            pass_number=None,
        )

        # Simulate: Excel export (IDs written to cells)
        excel_data = {
            "Audio Processing ID": audio_id,
            "Biomechanics Import ID": biomech_id,
            "Synchronization ID": sync_id,
        }

        # Simulate: Excel load (IDs read back)
        # In real scenario, these come from Excel cells
        loaded_audio_id = excel_data["Audio Processing ID"]
        loaded_biomech_id = excel_data["Biomechanics Import ID"]
        loaded_sync_id = excel_data["Synchronization ID"]

        # Verify: IDs are identical
        assert loaded_audio_id == audio_id
        assert loaded_biomech_id == biomech_id
        assert loaded_sync_id == sync_id

        # Simulate: Re-export (same IDs should be generated again)
        audio_id_2 = LocalReferenceIDGenerator.audio_processing_id(
            participant_id=1016,
            knee="right",
            maneuver="walk",
            recording_date=datetime(2024, 1, 15),
            audio_file_name="1016_right_walk.bin",
        )

        assert audio_id_2 == audio_id, "Re-generation should produce same ID"

    def test_workflow_reconstruct_relationships_from_registry(self):
        """Demonstrate relationship reconstruction from registry."""
        registry = LocalReferenceRegistry()

        # Register participant's files
        audio_id = 100
        biomech_id = 200
        sync_id = 300
        cycle_ids = [400, 401, 402]

        registry.register_audio(
            local_id=audio_id,
            participant_id=1016,
            study="AOA",
            knee="right",
            maneuver="walk",
            file_path="/recordings/1016_right_walk_audio.bin",
            recording_date=datetime(2024, 1, 15),
        )

        registry.register_biomechanics(
            local_id=biomech_id,
            participant_id=1016,
            study="AOA",
            knee="right",
            maneuver="walk",
            file_path="/recordings/1016_right_walk_biomech.xlsx",
            recording_date=datetime(2024, 1, 15),
        )

        registry.register_synchronization(
            local_id=sync_id,
            participant_id=1016,
            study="AOA",
            audio_id=audio_id,
            biomechanics_id=biomech_id,
            knee="right",
            maneuver="walk",
        )

        for i, cid in enumerate(cycle_ids):
            registry.register_movement_cycle(
                local_id=cid,
                participant_id=1016,
                study="AOA",
                audio_id=audio_id,
                cycle_index=i,
                knee="right",
                maneuver="walk",
            )

        # Simulate: Load from Excel
        # In real scenario, user loads Excel with these IDs
        loaded_audio_id = audio_id

        # Reconstruct relationships
        audio_record = registry.get_by_id(loaded_audio_id)
        assert audio_record.file_path == "/recordings/1016_right_walk_audio.bin"

        sync_record = registry.find_sync_for_audio(loaded_audio_id)
        assert sync_record is not None
        assert sync_record.related_ids["biomechanics_id"] == biomech_id

        biomech_record = registry.get_by_id(sync_record.related_ids["biomechanics_id"])
        assert biomech_record.file_path == "/recordings/1016_right_walk_biomech.xlsx"

        cycles = registry.find_cycles_for_audio(loaded_audio_id)
        assert len(cycles) == 3


class TestGlobalRegistry:
    """Test global registry singleton."""

    def test_global_registry_singleton(self):
        """Global registry maintains state across calls."""
        reset_global_registry()

        registry1 = get_global_registry()
        registry1.register_audio(
            local_id=100,
            participant_id=1016,
            study="AOA",
            knee="right",
            maneuver="walk",
            file_path="/path/to/audio.bin",
            recording_date=datetime(2024, 1, 15),
        )

        registry2 = get_global_registry()
        record = registry2.get_by_id(100)
        assert record is not None, "Global registry should persist between calls"


class TestRelatedIDsTracking:
    """Test tracking of relationships between records."""

    def test_synchronization_tracks_audio_and_biomechanics(self):
        """Sync record maintains references to audio and biomechanics IDs."""
        registry = LocalReferenceRegistry()

        audio_id = 100
        biomech_id = 200
        sync_id = 300

        registry.register_synchronization(
            local_id=sync_id,
            participant_id=1016,
            study="AOA",
            audio_id=audio_id,
            biomechanics_id=biomech_id,
            knee="right",
            maneuver="walk",
        )

        sync_record = registry.get_by_id(sync_id)
        assert sync_record.related_ids["audio_id"] == audio_id
        assert sync_record.related_ids["biomechanics_id"] == biomech_id

    def test_movement_cycle_tracks_audio_id(self):
        """Cycle record maintains reference to audio ID."""
        registry = LocalReferenceRegistry()

        audio_id = 100
        cycle_id = 400

        registry.register_movement_cycle(
            local_id=cycle_id,
            participant_id=1016,
            study="AOA",
            audio_id=audio_id,
            cycle_index=0,
            knee="right",
            maneuver="walk",
        )

        cycle_record = registry.get_by_id(cycle_id)
        assert cycle_record.related_ids["audio_id"] == audio_id
