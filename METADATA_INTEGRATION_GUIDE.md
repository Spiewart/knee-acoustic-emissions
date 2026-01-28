# Metadata Integration Implementation Guide

## Detailed Implementation Steps

This document provides code examples and specific implementation details for each phase.

---

## Phase 1: Implement New Metadata Classes

### 1.1 WalkMetadata Mixin

```python
from dataclasses import dataclass, field
from typing import Optional
from pydantic.dataclasses import dataclass as pydantic_dataclass

@pydantic_dataclass(config=dict(validate_assignment=True))
class WalkMetadata:
    """Mixin providing optional walk-specific metadata fields.
    
    These fields are populated when the maneuver is "walk" and required 
    for proper identification of walk passes.
    """
    
    pass_number: Optional[int] = field(
        default=None,
        metadata={"description": "Pass number for walk maneuver (1-indexed)"}
    )
    
    speed: Optional[str] = field(
        default=None,
        metadata={"description": "Walk speed for walk maneuver (e.g., 'comfortable', 'fast')"}
    )
    
    def __post_init__(self):
        """Validate that walk maneuver has pass_number and speed."""
        # Note: This will be called after dataclass initialization
        # Validation happens in parent class that includes maneuver field
        pass
```

**Key Points**:
- Optional fields (None allowed for non-walk maneuvers)
- Will be combined with `StudyMetadata` (which has `maneuver` field)
- Parent class will handle the conditional validation in `__post_init__`

---

### 1.2 SynchronizationMetadata Class

```python
from dataclasses import dataclass, field
from typing import Optional, List
from src.metadata import AcousticsFile, StudyMetadata

@pydantic_dataclass(config=dict(validate_assignment=True))
class SynchronizationMetadata(AcousticsFile, WalkMetadata):
    """Metadata for synchronization process.
    
    Inherits from:
    - AcousticsFile: audio file details (file path, channels, format)
    - WalkMetadata: optional walk-specific fields (pass_number, speed)
    
    Contains:
    - Stomp time data (what was detected/aligned)
    - Sync method details (how it was detected)
    - Detection method times (from each detection algorithm)
    """
    
    # ===== Stomp Time Data =====
    # Primary stomp times (result of sync process)
    audio_sync_time: Optional[float] = field(
        default=None,
        metadata={"description": "Audio stomp detection time (seconds)"}
    )
    sync_offset: Optional[float] = field(
        default=None,
        metadata={"description": "Offset between audio and biomechanics stomp (seconds)"}
    )
    
    # Aligned stomp times (after biomechanics guidance if applicable)
    aligned_audio_sync_time: Optional[float] = field(
        default=None,
        metadata={"description": "Audio stomp after biomechanics alignment (seconds)"}
    )
    aligned_biomechanics_sync_time: Optional[float] = field(
        default=None,
        metadata={"description": "Biomechanics stomp (after alignment if applicable) (seconds)"}
    )
    
    # ===== Sync Method =====
    sync_method: Optional[str] = field(
        default=None,
        metadata={"description": "How sync was determined: 'consensus', 'biomechanics', or 'manual'"}
    )
    
    consensus_methods: Optional[str] = field(
        default=None,
        metadata={"description": "Methods included in consensus (comma-separated)"}
    )
    
    # ===== Detection Method Times =====
    # Individual detection algorithm results
    consensus_time: Optional[float] = field(
        default=None,
        metadata={"description": "Consensus method result (seconds)"}
    )
    rms_time: Optional[float] = field(
        default=None,
        metadata={"description": "RMS detection time (seconds)"}
    )
    onset_time: Optional[float] = field(
        default=None,
        metadata={"description": "Onset detection time (seconds)"}
    )
    freq_time: Optional[float] = field(
        default=None,
        metadata={"description": "Frequency detection time (seconds)"}
    )
    
    # ===== Biomechanics-Guided Detection (Optional) =====
    # Used when sync is determined via biomechanics guidance
    selected_audio_sync_time: Optional[float] = field(
        default=None,
        metadata={"description": "Guided selection of audio stomp from biomechanics"}
    )
    contra_selected_audio_sync_time: Optional[float] = field(
        default=None,
        metadata={"description": "Contralateral guided selection"}
    )
    detected_sync_energy_ratio: Optional[float] = field(
        default=None,
        metadata={"description": "Energy ratio used for biomechanics guidance"}
    )
    
    # ===== Audio-Visual Sync (Optional) =====
    # Used when visual/audio sync applied
    audio_visual_sync_time: Optional[float] = field(
        default=None,
        metadata={"description": "Time of audio-visual sync event (seconds)"}
    )
    bio_visual_sync_time: Optional[float] = field(
        default=None,
        metadata={"description": "Time of biomechanics-visual sync event (seconds)"}
    )
```

**Key Points**:
- Inherits from `AcousticsFile` (gets file path, channels, format, etc.)
- Inherits from `WalkMetadata` (gets optional pass_number and speed)
- Contains ALL stomp time and sync method data
- Used as base class for both Synchronization and MovementCycle

---

### 1.3 Refactored Synchronization Class

```python
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime

@pydantic_dataclass(config=dict(validate_assignment=True))
class Synchronization(SynchronizationMetadata):
    """Synchronization process record.
    
    Inherits from:
    - SynchronizationMetadata: stomp times and sync method data
    
    Contains:
    - Synchronization process metadata (file name, date)
    - Aggregate statistics (total cycles, clean cycles, cycle durations)
    - Processing details
    
    Excludes:
    - Per-cycle details (stored in MovementCycle records)
    - Cycle-level QC flags (biomechanics_qc_fail, sync_qc_fail - stored in MovementCycle)
    """
    
    # ===== Sync Process Metadata =====
    sync_file_name: str = field(
        metadata={"description": "Name of synchronization file"}
    )
    processing_date: datetime = field(
        metadata={"description": "Date synchronization was processed"}
    )
    
    # ===== Processing Details =====
    sync_duration: Optional[float] = field(
        default=None,
        metadata={"description": "Duration of synchronization process (seconds)"}
    )
    
    # ===== Aggregate Statistics (from all cycles) =====
    total_cycles_extracted: int = field(
        default=0,
        metadata={"description": "Total number of cycles extracted"}
    )
    clean_cycles: int = field(
        default=0,
        metadata={"description": "Number of clean cycles (no QC issues)"}
    )
    outlier_cycles: int = field(
        default=0,
        metadata={"description": "Number of cycles marked as outliers"}
    )
    
    # ===== Cycle Duration Statistics =====
    mean_cycle_duration_s: Optional[float] = field(
        default=None,
        metadata={"description": "Mean cycle duration across all cycles (seconds)"}
    )
    median_cycle_duration_s: Optional[float] = field(
        default=None,
        metadata={"description": "Median cycle duration across all cycles (seconds)"}
    )
    min_cycle_duration_s: Optional[float] = field(
        default=None,
        metadata={"description": "Minimum cycle duration (seconds)"}
    )
    max_cycle_duration_s: Optional[float] = field(
        default=None,
        metadata={"description": "Maximum cycle duration (seconds)"}
    )
    
    # ===== Acoustic Statistics =====
    mean_acoustic_auc: Optional[float] = field(
        default=None,
        metadata={"description": "Mean acoustic AUC across all cycles"}
    )
    
    # ===== Constraints =====
    linked_biomechanics: bool = field(
        default=True,
        metadata={"description": "Whether biomechanics data was required (always True for sync)"}
    )
    
    @property
    def linked_biomechanics(self) -> bool:
        """Synchronization always requires linked biomechanics."""
        return True
```

**Key Points**:
- Inherits from SynchronizationMetadata (gets all stomp time + sync method data)
- Contains sync process summary (file name, date, duration)
- Contains aggregate statistics (from all cycles)
- Does NOT contain per-cycle details
- Does NOT contain cycle-level QC flags (those go in MovementCycle)
- `linked_biomechanics` is always True for Synchronization

---

### 1.4 Refactored MovementCycle Class

```python
from dataclasses import dataclass, field
from typing import Optional
from src.metadata import AudioProcessing

@pydantic_dataclass(config=dict(validate_assignment=True))
class MovementCycle(SynchronizationMetadata, AudioProcessing):
    """Movement cycle record.
    
    Inherits from:
    - SynchronizationMetadata: stomp times, sync method data, walk metadata
    - AudioProcessing: all audio QC fields (qc_artifact, qc_signal_dropout, etc.)
    
    Contains ONLY:
    - Cycle identification (cycle_file, cycle_index, is_outlier)
    - Cycle timing (start_time_s, end_time_s, duration_s, audio/bio timestamps)
    - Cycle-level QC flags (biomechanics_qc_fail, sync_qc_fail)
    
    All other fields inherited from parents.
    """
    
    # ===== Cycle Identification =====
    cycle_file: str = field(
        metadata={"description": "Path/name of cycle file"}
    )
    cycle_index: int = field(
        metadata={"description": "0-indexed cycle number within maneuver"}
    )
    is_outlier: bool = field(
        default=False,
        metadata={"description": "Whether cycle was marked as outlier"}
    )
    
    # ===== Cycle Timing =====
    start_time_s: float = field(
        metadata={"description": "Cycle start time in seconds"}
    )
    end_time_s: float = field(
        metadata={"description": "Cycle end time in seconds"}
    )
    duration_s: float = field(
        metadata={"description": "Cycle duration in seconds"}
    )
    
    # ===== Audio/Biomechanics Timestamps =====
    audio_start_time: Optional[float] = field(
        default=None,
        metadata={"description": "Audio timestamp at cycle start"}
    )
    audio_end_time: Optional[float] = field(
        default=None,
        metadata={"description": "Audio timestamp at cycle end"}
    )
    bio_start_time: Optional[float] = field(
        default=None,
        metadata={"description": "Biomechanics timestamp at cycle start"}
    )
    bio_end_time: Optional[float] = field(
        default=None,
        metadata={"description": "Biomechanics timestamp at cycle end"}
    )
    
    # ===== Cycle-Level QC Flags (CYCLE-LEVEL ONLY - NOT inherited) =====
    biomechanics_qc_fail: bool = field(
        default=False,
        metadata={"description": "Whether biomechanics data failed QC for this cycle"}
    )
    sync_qc_fail: bool = field(
        default=False,
        metadata={"description": "Whether synchronization failed QC for this cycle"}
    )
    
    # ===== Inherited from AudioProcessing =====
    # All audio QC fields automatically inherited:
    # - qc_fail_segments: List[tuple[float, float]]
    # - qc_artifact: bool
    # - qc_artifact_type: Optional[str]
    # - qc_artifact_type_ch1/2/3/4: Optional[str]
    # - qc_signal_dropout: bool
    # - qc_signal_dropout_segments: List[tuple[float, float]]
    # - qc_signal_dropout_ch1/2/3/4: bool
    # - qc_signal_dropout_segments_ch1/2/3/4: List[tuple[float, float]]
    
    # ===== Inherited from SynchronizationMetadata =====
    # All stomp time and sync method data automatically inherited:
    # - audio_sync_time, sync_offset, aligned_audio_sync_time, etc.
    # - sync_method, consensus_methods
    # - consensus_time, rms_time, onset_time, freq_time
    # - pass_number, speed (from WalkMetadata)
    
    # ===== Inherited from AcousticsFile =====
    # Audio file details automatically inherited:
    # - file_path, num_channels, sample_rate, etc.
```

**Key Points**:
- Inherits from BOTH SynchronizationMetadata AND AudioProcessing
- No field duplication (all QC fields from AudioProcessing, all sync data from SynchronizationMetadata)
- Includes cycle-specific fields only (cycle_file, cycle_index, timing)
- Includes ONLY cycle-level QC flags (biomechanics_qc_fail, sync_qc_fail)
- Future-proof: If AudioProcessing adds new QC fields, MovementCycle automatically gets them

---

## Phase 2: Update Record Creation in processing_log.py

### 2.1 Helper Function to Create SynchronizationMetadata

```python
def _create_synchronization_metadata(
    row: Dict[str, Any],
    acoustics_file: AcousticsFile,
    maneuver: str,
) -> SynchronizationMetadata:
    """Create SynchronizationMetadata from row data.
    
    Extracts stomp times and sync method info, combines with acoustics file.
    """
    sync_method, consensus_methods = _get_sync_method_defaults(row)
    
    return SynchronizationMetadata(
        # From AcousticsFile
        file_path=acoustics_file.file_path,
        num_channels=acoustics_file.num_channels,
        sample_rate=acoustics_file.sample_rate,
        recording_date=acoustics_file.recording_date,
        # From WalkMetadata (conditional)
        pass_number=row.get("Pass Number") if maneuver == "walk" else None,
        speed=row.get("Speed") if maneuver == "walk" else None,
        # Stomp times
        audio_sync_time=pd.to_numeric(row.get("Audio Stomp (s)"), errors="coerce"),
        sync_offset=pd.to_numeric(row.get("Stomp Offset (s)"), errors="coerce"),
        aligned_audio_sync_time=pd.to_numeric(row.get("Aligned Audio Stomp (s)"), errors="coerce"),
        aligned_biomechanics_sync_time=pd.to_numeric(row.get("Aligned Bio Stomp (s)"), errors="coerce"),
        # Sync method
        sync_method=sync_method,
        consensus_methods=consensus_methods,
        # Detection method times
        consensus_time=pd.to_numeric(row.get("Consensus (s)"), errors="coerce"),
        rms_time=pd.to_numeric(row.get("RMS Detect (s)"), errors="coerce"),
        onset_time=pd.to_numeric(row.get("Onset Detect (s)"), errors="coerce"),
        freq_time=pd.to_numeric(row.get("Freq Detect (s)"), errors="coerce"),
        # Optional fields
        selected_audio_sync_time=pd.to_numeric(row.get("Selected Audio Sync (s)"), errors="coerce"),
        contra_selected_audio_sync_time=pd.to_numeric(row.get("Contra Selected (s)"), errors="coerce"),
        detected_sync_energy_ratio=pd.to_numeric(row.get("Energy Ratio"), errors="coerce"),
    )
```

### 2.2 Dual-Track Record Creation

Add this to `ManeuverProcessingLog` class:

```python
def add_synchronization_records_dual_track(
    self,
    synced_files_df: pd.DataFrame,
    biomechanics_data: Optional[BiomechanicsRecording] = None,
) -> None:
    """Create Synchronization records with dual-track validation.
    
    Creates both old-format and new-format records side-by-side,
    validates they match, then stores old-format for now.
    
    This dual-track mode allows us to verify the new metadata structure
    works correctly before switching production code.
    """
    if synced_files_df.empty:
        return
    
    old_format_records = []
    new_format_records = []
    validation_errors = []
    
    for idx, row in synced_files_df.iterrows():
        try:
            # Create old format (current production)
            old_record = Synchronization(
                # ... existing code to create old format ...
            )
            old_format_records.append(old_record)
            
            # Create new format (parallel validation)
            sync_metadata = _create_synchronization_metadata(
                row, 
                AcousticsFile(...),  # Extract from row
                self.maneuver,
            )
            
            new_record = Synchronization(
                # Inherit from sync_metadata
                **sync_metadata.to_dict(),
                # Add sync-specific fields
                sync_file_name=old_record.sync_file_name,
                processing_date=datetime.now(),
                # ... other sync fields ...
            )
            new_format_records.append(new_record)
            
            # Validate key fields match
            key_fields = [
                'audio_sync_time', 'sync_offset', 'aligned_audio_sync_time',
                'sync_method', 'consensus_methods'
            ]
            for field_name in key_fields:
                old_val = getattr(old_record, field_name, None)
                new_val = getattr(new_record, field_name, None)
                if old_val != new_val:
                    validation_errors.append(
                        f"Field {field_name} mismatch for {row.get('Sync File')}: "
                        f"old={old_val}, new={new_val}"
                    )
        
        except Exception as e:
            logger.error(f"Error creating records for row {idx}: {e}")
            validation_errors.append(str(e))
    
    # Log validation results
    if validation_errors:
        logger.warning(f"Dual-track validation errors: {validation_errors}")
        for error in validation_errors:
            logger.warning(error)
    else:
        logger.info(f"Dual-track validation passed: {len(old_format_records)} records")
    
    # Store old format (production), keep new format in memory for validation
    self.synchronization_records = old_format_records
    self._new_format_synchronization_records = new_format_records  # For testing
```

### 2.3 Validation Method

Add to `ManeuverProcessingLog` class:

```python
def validate_metadata_consistency(self) -> List[str]:
    """Validate that metadata is consistent across records.
    
    Returns list of validation errors (empty list if all OK).
    """
    errors = []
    
    # Check Synchronization records
    if self.synchronization_records:
        for rec in self.synchronization_records:
            # Verify required fields present
            if not rec.sync_file_name:
                errors.append("Synchronization record missing sync_file_name")
            if rec.linked_biomechanics != True:
                errors.append("Synchronization record linked_biomechanics not True")
            # Verify no cycle-level fields
            if hasattr(rec, 'biomechanics_qc_fail'):
                errors.append("Synchronization record has cycle-level field biomechanics_qc_fail")
    
    # Check MovementCycle records
    if self.movement_cycles_records:
        for rec in self.movement_cycles_records:
            # Verify required fields present
            if not rec.cycle_file:
                errors.append("MovementCycle record missing cycle_file")
            if not hasattr(rec, 'cycle_index'):
                errors.append("MovementCycle record missing cycle_index")
            # Verify has inherited fields
            if not hasattr(rec, 'audio_sync_time'):
                errors.append("MovementCycle record missing inherited audio_sync_time")
            if not hasattr(rec, 'qc_artifact'):
                errors.append("MovementCycle record missing inherited qc_artifact")
            # Verify has cycle-level fields
            if not hasattr(rec, 'biomechanics_qc_fail'):
                errors.append("MovementCycle record missing biomechanics_qc_fail")
    
    return errors
```

---

## Phase 4: Integration Testing

### 4.1 Test File Structure

Create `tests/test_metadata_integration.py`:

```python
"""Integration tests for metadata classes and processing pipeline.

Tests the complete flow of metadata through the processing pipeline:
1. Creation and field population
2. Inheritance correctness
3. Data persistence (Excel save/load)
4. Resume scenarios
5. Cross-field consistency
"""

import pytest
from pathlib import Path
from datetime import datetime
from src.metadata import (
    WalkMetadata, SynchronizationMetadata, Synchronization, MovementCycle,
    AcousticsFile, AudioProcessing, StudyMetadata
)
from src.orchestration.processing_log import ManeuverProcessingLog
```

### 4.2 Field Inheritance Tests

```python
class TestFieldInheritance:
    """Test that fields are properly inherited, not duplicated."""
    
    def test_synchronization_metadata_has_acoustics_fields(self):
        """Verify SynchronizationMetadata inherits AcousticsFile fields."""
        metadata = SynchronizationMetadata(
            # AcousticsFile fields
            file_path="/path/to/file.wav",
            num_channels=4,
            sample_rate=10000.0,
            # WalkMetadata fields
            pass_number=1,
            speed="comfortable",
            # SynchronizationMetadata fields
            audio_sync_time=1.5,
            sync_method="consensus",
        )
        
        assert metadata.file_path == "/path/to/file.wav"
        assert metadata.num_channels == 4
        assert metadata.audio_sync_time == 1.5
    
    def test_movement_cycle_inherits_synchronization_metadata_fields(self):
        """Verify MovementCycle inherits SynchronizationMetadata fields."""
        cycle = MovementCycle(
            # SynchronizationMetadata fields
            file_path="/path/to/file.wav",
            audio_sync_time=1.5,
            pass_number=1,
            # AudioProcessing fields
            qc_artifact=False,
            qc_artifact_type=None,
            # MovementCycle fields
            cycle_file="cycle_0.pkl",
            cycle_index=0,
            start_time_s=0.0,
            end_time_s=1.0,
            duration_s=1.0,
        )
        
        assert cycle.file_path == "/path/to/file.wav"  # Inherited from AcousticsFile
        assert cycle.audio_sync_time == 1.5  # Inherited from SynchronizationMetadata
        assert cycle.qc_artifact == False  # Inherited from AudioProcessing
        assert cycle.cycle_index == 0  # Own field
    
    def test_no_field_duplication_in_movement_cycle(self):
        """Verify MovementCycle doesn't define fields from parents."""
        # Should inherit qc_artifact from AudioProcessing, not define it
        cycle = MovementCycle(
            file_path="/path",
            audio_sync_time=1.5,
            cycle_file="c.pkl",
            cycle_index=0,
            start_time_s=0.0,
            end_time_s=1.0,
            duration_s=1.0,
            qc_artifact=True,  # Should come from AudioProcessing parent
        )
        
        # Verify all AudioProcessing QC fields present
        qc_fields = [
            'qc_artifact', 'qc_signal_dropout', 'qc_fail_segments',
            'qc_artifact_type', 'qc_artifact_type_ch1', 'qc_artifact_type_ch2',
        ]
        for field_name in qc_fields:
            assert hasattr(cycle, field_name), f"Missing QC field: {field_name}"
```

### 4.3 Data Flow Tests

```python
class TestFullPipelineDataFlow:
    """Test data flowing through complete processing pipeline."""
    
    @pytest.fixture
    def sample_maneuver_processor(self, tmp_path):
        """Create a processor with sample data through sync stage."""
        # Create temp maneuver directory with sample audio files
        # Run ManeuverProcessor through sync stage
        # Return processor with data loaded
        pass
    
    def test_sync_record_creation_populates_all_fields(
        self, sample_maneuver_processor
    ):
        """Verify Synchronization record created with all fields."""
        log = sample_maneuver_processor.log
        
        assert len(log.synchronization_records) > 0
        rec = log.synchronization_records[0]
        
        # Required fields
        assert rec.sync_file_name is not None
        assert rec.processing_date is not None
        assert rec.linked_biomechanics == True
        
        # Inherited from SynchronizationMetadata
        assert hasattr(rec, 'audio_sync_time')
        assert hasattr(rec, 'sync_method')
        
        # Should NOT have cycle-level fields
        assert not hasattr(rec, 'cycle_file')
        assert not hasattr(rec, 'is_outlier')
    
    def test_movement_cycle_record_has_all_inherited_fields(
        self, sample_maneuver_processor
    ):
        """Verify MovementCycle record has all inherited fields."""
        log = sample_maneuver_processor.log
        
        assert len(log.movement_cycles_records) > 0
        cycle = log.movement_cycles_records[0]
        
        # Own fields
        assert cycle.cycle_file is not None
        assert cycle.cycle_index >= 0
        assert cycle.duration_s > 0
        
        # Inherited from AudioProcessing
        assert hasattr(cycle, 'qc_artifact')
        assert hasattr(cycle, 'qc_signal_dropout')
        assert hasattr(cycle, 'qc_fail_segments')
        
        # Inherited from SynchronizationMetadata
        assert hasattr(cycle, 'audio_sync_time')
        assert hasattr(cycle, 'sync_method')
        assert hasattr(cycle, 'pass_number')  # From WalkMetadata
        
        # Cycle-level QC flags
        assert hasattr(cycle, 'biomechanics_qc_fail')
        assert hasattr(cycle, 'sync_qc_fail')
```

### 4.4 Persistence Tests

```python
class TestPersistence:
    """Test saving and loading from Excel."""
    
    def test_synchronization_excel_roundtrip(
        self, sample_maneuver_processor, tmp_path
    ):
        """Verify Synchronization can be saved to Excel and read back."""
        log = sample_maneuver_processor.log
        excel_path = tmp_path / "test.xlsx"
        
        # Save
        log.save_to_excel(excel_path)
        assert excel_path.exists()
        
        # Verify sheet exists
        sheets = pd.ExcelFile(excel_path).sheet_names
        assert "Synchronization" in sheets
        
        # Load back
        df = pd.read_excel(excel_path, sheet_name="Synchronization")
        assert len(df) > 0
        
        # Verify key columns present
        key_columns = [
            'sync_file_name', 'audio_sync_time', 'sync_method',
            'clean_cycles', 'mean_cycle_duration_s'
        ]
        for col in key_columns:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_movement_cycle_excel_roundtrip(
        self, sample_maneuver_processor, tmp_path
    ):
        """Verify MovementCycle can be saved to Excel and read back."""
        log = sample_maneuver_processor.log
        excel_path = tmp_path / "test.xlsx"
        
        # Save
        log.save_to_excel(excel_path)
        
        # Verify sheet exists
        sheets = pd.ExcelFile(excel_path).sheet_names
        assert "Movement Cycles" in sheets
        
        # Load back
        df = pd.read_excel(excel_path, sheet_name="Movement Cycles")
        assert len(df) > 0
        
        # Verify key columns present
        key_columns = [
            'cycle_file', 'cycle_index', 'start_time_s', 'end_time_s',
            'biomechanics_qc_fail', 'sync_qc_fail',
            'qc_artifact', 'qc_signal_dropout',  # From AudioProcessing
            'audio_sync_time', 'sync_method',  # From SynchronizationMetadata
        ]
        for col in key_columns:
            assert col in df.columns, f"Missing column in Movement Cycles: {col}"
```

### 4.5 Validation Tests

```python
class TestMetadataValidation:
    """Test validation of metadata."""
    
    def test_walk_metadata_validates_for_walk_maneuver(self):
        """Verify pass_number and speed required for walk maneuver."""
        # With walk maneuver, pass_number and speed should be required
        with pytest.raises(ValueError):
            # Missing pass_number/speed for walk
            SynchronizationMetadata(
                file_path="/path",
                maneuver="walk",
                audio_sync_time=1.5,
                sync_method="consensus",
                # Missing pass_number and speed
            )
    
    def test_synchronization_aggregate_fields_optional(self):
        """Verify aggregate fields optional and default correctly."""
        rec = Synchronization(
            sync_file_name="file.h5",
            processing_date=datetime.now(),
            file_path="/path",
            audio_sync_time=1.5,
            sync_method="consensus",
            # Omit aggregate fields
        )
        
        assert rec.total_cycles_extracted == 0
        assert rec.clean_cycles == 0
        assert rec.linked_biomechanics == True
```

---

## Success Criteria Checklist

- [ ] All new metadata classes implemented
- [ ] WalkMetadata mixin works correctly
- [ ] SynchronizationMetadata properly inherits from AcousticsFile + WalkMetadata
- [ ] Synchronization inherits from SynchronizationMetadata
- [ ] MovementCycle inherits from SynchronizationMetadata + AudioProcessing
- [ ] No field duplication across inheritance hierarchy
- [ ] Dual-track record creation working in processing_log.py
- [ ] Validation catches field mismatches
- [ ] All 600+ existing tests still pass
- [ ] All new integration tests pass
- [ ] Excel save/load roundtrip verified
- [ ] Resume scenarios tested
- [ ] Documentation updated

