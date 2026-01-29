"""SQLAlchemy ORM models for acoustic emissions metadata.

NORMALIZED SCHEMA - Uses foreign key relationships instead of inheritance.

Design principles:
- Audio files, biomechanics imports, and synchronizations are independent entities
- Movement cycles reference these entities via foreign keys
- No data duplication across tables
- Proper relational database design

Large binary files (.pkl) remain on disk; only paths, checksums, and
metadata are stored in the database.

Note: These models are designed for PostgreSQL and use PostgreSQL-specific
features like ARRAY types. Run tests against PostgreSQL, not SQLite.
"""

from datetime import datetime
from typing import List

from sqlalchemy import (
    ARRAY,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class StudyRecord(Base):
    """Study-level metadata.

    Top-level entity representing a research study (AOA, preOA, SMoCK).
    """
    __tablename__ = "studies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)  # AOA, preOA, SMoCK
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    participants: Mapped[List["ParticipantRecord"]] = relationship(
        "ParticipantRecord", back_populates="study", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("name IN ('AOA', 'preOA', 'SMoCK')", name="valid_study_name"),
    )


class ParticipantRecord(Base):
    """Participant-level metadata.

    Represents a single participant within a study.
    """
    __tablename__ = "participants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    study_id: Mapped[int] = mapped_column(Integer, ForeignKey("studies.id"), nullable=False)
    participant_number: Mapped[int] = mapped_column(Integer, nullable=False)  # e.g., 1011 from #AOA1011
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    study: Mapped["StudyRecord"] = relationship("StudyRecord", back_populates="participants")
    audio_processing: Mapped[List["AudioProcessingRecord"]] = relationship(
        "AudioProcessingRecord", back_populates="participant", cascade="all, delete-orphan"
    )
    biomechanics_imports: Mapped[List["BiomechanicsImportRecord"]] = relationship(
        "BiomechanicsImportRecord", back_populates="participant", cascade="all, delete-orphan"
    )
    synchronizations: Mapped[List["SynchronizationRecord"]] = relationship(
        "SynchronizationRecord", back_populates="participant", cascade="all, delete-orphan"
    )
    movement_cycles: Mapped[List["MovementCycleRecord"]] = relationship(
        "MovementCycleRecord", back_populates="participant", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("study_id", "participant_number", name="uq_study_participant"),
    )


class AudioProcessingRecord(Base):
    """Audio processing metadata.

    Standalone table for audio file metadata and QC results.
    Does NOT contain biomechanics information - that's in BiomechanicsImportRecord.
    """
    __tablename__ = "audio_processing"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    participant_id: Mapped[int] = mapped_column(Integer, ForeignKey("participants.id"), nullable=False)

    # Recording-time FK: which biomechanics (if any) was recorded with this audio
    biomechanics_import_id = mapped_column(Integer, ForeignKey("biomechanics_imports.id"), nullable=True)

    # File identification
    audio_file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    device_serial: Mapped[str] = mapped_column(String(50), nullable=False)
    firmware_version: Mapped[int] = mapped_column(Integer, nullable=False)
    file_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    file_size_mb: Mapped[float] = mapped_column(Float, nullable=False)

    # Recording metadata
    recording_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    recording_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # Maneuver metadata
    knee: Mapped[str] = mapped_column(String(10), nullable=False)  # left, right
    maneuver: Mapped[str] = mapped_column(String(20), nullable=False)  # fe, sts, walk

    # Audio characteristics
    num_channels: Mapped[int] = mapped_column(Integer, nullable=False)
    sample_rate: Mapped[float] = mapped_column(Float, default=46875.0, nullable=False)

    # Microphone positions
    mic_1_position: Mapped[str] = mapped_column(String(10), nullable=False)  # IPM, IPL, SPM, SPL
    mic_2_position: Mapped[str] = mapped_column(String(10), nullable=False)
    mic_3_position: Mapped[str] = mapped_column(String(10), nullable=False)
    mic_4_position: Mapped[str] = mapped_column(String(10), nullable=False)

    # Optional notes
    mic_1_notes = mapped_column(Text, nullable=True)
    mic_2_notes = mapped_column(Text, nullable=True)
    mic_3_notes = mapped_column(Text, nullable=True)
    mic_4_notes = mapped_column(Text, nullable=True)
    notes = mapped_column(Text, nullable=True)

    # Pickle file storage (path only, file on disk)
    pkl_file_path = mapped_column(String(500), nullable=True)
    pkl_file_checksum = mapped_column(String(64), nullable=True)  # SHA-256
    pkl_file_size_mb = mapped_column(Float, nullable=True)
    pkl_file_modified = mapped_column(DateTime, nullable=True)

    # Audio QC metadata
    audio_qc_version = mapped_column(String(20), nullable=True)
    audio_qc_fail: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    audio_qc_fail_ch1: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    audio_qc_fail_ch2: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    audio_qc_fail_ch3: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    audio_qc_fail_ch4: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # QC artifact types (can be arrays of strings)
    qc_artifact_type = mapped_column(ARRAY(String), nullable=True)
    qc_artifact_type_ch1 = mapped_column(ARRAY(String), nullable=True)
    qc_artifact_type_ch2 = mapped_column(ARRAY(String), nullable=True)
    qc_artifact_type_ch3 = mapped_column(ARRAY(String), nullable=True)
    qc_artifact_type_ch4 = mapped_column(ARRAY(String), nullable=True)

    # QC fail segments (stored as arrays of floats - flattened [start1, end1, start2, end2, ...])
    qc_fail_segments = mapped_column(ARRAY(Float), nullable=True)
    qc_fail_segments_ch1 = mapped_column(ARRAY(Float), nullable=True)
    qc_fail_segments_ch2 = mapped_column(ARRAY(Float), nullable=True)
    qc_fail_segments_ch3 = mapped_column(ARRAY(Float), nullable=True)
    qc_fail_segments_ch4 = mapped_column(ARRAY(Float), nullable=True)

    # Signal dropout QC
    qc_signal_dropout: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    qc_signal_dropout_segments = mapped_column(ARRAY(Float), nullable=True)
    qc_signal_dropout_ch1: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    qc_signal_dropout_segments_ch1 = mapped_column(ARRAY(Float), nullable=True)
    qc_signal_dropout_ch2: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    qc_signal_dropout_segments_ch2 = mapped_column(ARRAY(Float), nullable=True)
    qc_signal_dropout_ch3: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    qc_signal_dropout_segments_ch3 = mapped_column(ARRAY(Float), nullable=True)
    qc_signal_dropout_ch4: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    qc_signal_dropout_segments_ch4 = mapped_column(ARRAY(Float), nullable=True)

    # Artifact QC
    qc_artifact: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    qc_artifact_segments = mapped_column(ARRAY(Float), nullable=True)
    qc_artifact_ch1: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    qc_artifact_segments_ch1 = mapped_column(ARRAY(Float), nullable=True)
    qc_artifact_ch2: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    qc_artifact_segments_ch2 = mapped_column(ARRAY(Float), nullable=True)
    qc_artifact_ch3: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    qc_artifact_segments_ch3 = mapped_column(ARRAY(Float), nullable=True)
    qc_artifact_ch4: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    qc_artifact_segments_ch4 = mapped_column(ARRAY(Float), nullable=True)

    # Processing metadata
    processing_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    processing_status: Mapped[str] = mapped_column(String(50), default="not_processed", nullable=False)
    error_message = mapped_column(Text, nullable=True)
    duration_seconds = mapped_column(Float, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    participant: Mapped["ParticipantRecord"] = relationship("ParticipantRecord", back_populates="audio_processing")
    biomechanics_import: Mapped["BiomechanicsImportRecord"] = relationship(
        "BiomechanicsImportRecord", back_populates="audio_processing", foreign_keys=[biomechanics_import_id]
    )
    synchronizations: Mapped[List["SynchronizationRecord"]] = relationship(
        "SynchronizationRecord", back_populates="audio_processing", cascade="all, delete-orphan"
    )
    movement_cycles: Mapped[List["MovementCycleRecord"]] = relationship(
        "MovementCycleRecord", back_populates="audio_processing", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("knee IN ('left', 'right')", name="audio_valid_knee"),
        CheckConstraint("maneuver IN ('fe', 'sts', 'walk')", name="audio_valid_maneuver"),
        CheckConstraint("processing_status IN ('not_processed', 'success', 'error')", name="audio_valid_status"),
        UniqueConstraint("participant_id", "audio_file_name", "knee", "maneuver", name="uq_audio_processing"),
    )


class BiomechanicsImportRecord(Base):
    """Biomechanics import metadata.

    Standalone table for biomechanics file import and QC results.
    Does NOT contain audio information - that's in AudioProcessingRecord.
    """
    __tablename__ = "biomechanics_imports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    participant_id: Mapped[int] = mapped_column(Integer, ForeignKey("participants.id"), nullable=False)

    # Recording-time FK: which audio (if any) was recorded with this biomechanics
    audio_processing_id = mapped_column(Integer, ForeignKey("audio_processing.id"), nullable=True)

    # File identification
    biomechanics_file: Mapped[str] = mapped_column(String(255), nullable=False)
    sheet_name = mapped_column(String(100), nullable=True)
    biomechanics_type: Mapped[str] = mapped_column(String(50), nullable=False)  # Gonio, IMU, Motion Analysis

    # Maneuver metadata
    knee: Mapped[str] = mapped_column(String(10), nullable=False)
    maneuver: Mapped[str] = mapped_column(String(20), nullable=False)

    # Walk-specific metadata (optional)
    pass_number = mapped_column(Integer, nullable=True)
    speed = mapped_column(String(20), nullable=True)

    # Biomechanics characteristics
    biomechanics_sync_method: Mapped[str] = mapped_column(String(20), nullable=False)  # flick, stomp
    biomechanics_sample_rate: Mapped[float] = mapped_column(Float, nullable=False)
    biomechanics_notes = mapped_column(Text, nullable=True)

    # Import statistics
    num_sub_recordings: Mapped[int] = mapped_column(Integer, nullable=False)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    num_data_points: Mapped[int] = mapped_column(Integer, nullable=False)
    num_passes = mapped_column(Integer, default=0, nullable=False)

    # QC metadata
    biomech_qc_version = mapped_column(String(20), nullable=True)
    biomechanics_qc_fail: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    biomechanics_qc_notes = mapped_column(Text, nullable=True)

    # Processing metadata
    processing_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    processing_status: Mapped[str] = mapped_column(String(50), default="not_processed", nullable=False)
    error_message = mapped_column(Text, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    participant: Mapped["ParticipantRecord"] = relationship("ParticipantRecord", back_populates="biomechanics_imports")
    audio_processing: Mapped["AudioProcessingRecord"] = relationship(
        "AudioProcessingRecord", back_populates="biomechanics_import", foreign_keys=[audio_processing_id]
    )
    synchronizations: Mapped[List["SynchronizationRecord"]] = relationship(
        "SynchronizationRecord", back_populates="biomechanics_import", cascade="all, delete-orphan"
    )
    movement_cycles: Mapped[List["MovementCycleRecord"]] = relationship(
        "MovementCycleRecord", back_populates="biomechanics_import"
    )

    __table_args__ = (
        CheckConstraint("knee IN ('left', 'right')", name="biomech_valid_knee"),
        CheckConstraint("maneuver IN ('fe', 'sts', 'walk')", name="biomech_valid_maneuver"),
        CheckConstraint("processing_status IN ('not_processed', 'success', 'error')", name="biomech_valid_status"),
        UniqueConstraint("participant_id", "biomechanics_file", "knee", "maneuver", name="uq_biomechanics_import"),
    )


class SynchronizationRecord(Base):
    """Synchronization metadata.

    Tracks audio-biomechanics synchronization results.
    References AudioProcessingRecord and BiomechanicsImportRecord via foreign keys.
    """
    __tablename__ = "synchronizations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    participant_id: Mapped[int] = mapped_column(Integer, ForeignKey("participants.id"), nullable=False)
    
    # Foreign keys to related records
    audio_processing_id: Mapped[int] = mapped_column(Integer, ForeignKey("audio_processing.id"), nullable=False)
    biomechanics_import_id: Mapped[int] = mapped_column(Integer, ForeignKey("biomechanics_imports.id"), nullable=False)

    # Walk-specific metadata (optional, inherited from parent records)
    pass_number = mapped_column(Integer, nullable=True)
    speed = mapped_column(String(20), nullable=True)

    # Synchronization times
    audio_sync_time = mapped_column(Float, nullable=True)
    bio_left_sync_time = mapped_column(Float, nullable=True)
    bio_right_sync_time = mapped_column(Float, nullable=True)
    sync_offset = mapped_column(Float, nullable=True)
    aligned_audio_sync_time = mapped_column(Float, nullable=True)
    aligned_biomechanics_sync_time = mapped_column(Float, nullable=True)

    # Sync method details
    sync_method = mapped_column(String(50), nullable=True)  # consensus, rms, onset, freq, biomechanics
    consensus_methods = mapped_column(String(100), nullable=True)
    consensus_time = mapped_column(Float, nullable=True)
    rms_time = mapped_column(Float, nullable=True)
    onset_time = mapped_column(Float, nullable=True)
    freq_time = mapped_column(Float, nullable=True)
    
    # Biomechanics-guided sync fields
    selected_audio_sync_time = mapped_column(Float, nullable=True)
    contra_selected_audio_sync_time = mapped_column(Float, nullable=True)
    audio_visual_sync_time = mapped_column(Float, nullable=True)
    audio_visual_sync_time_contralateral = mapped_column(Float, nullable=True)
    audio_stomp_method = mapped_column(String(50), nullable=True)

    # Pickle file storage (sync results)
    sync_file_name = mapped_column(String(255), nullable=True)
    sync_file_path = mapped_column(String(500), nullable=True)
    sync_file_checksum = mapped_column(String(64), nullable=True)
    sync_file_size_mb = mapped_column(Float, nullable=True)
    sync_file_modified = mapped_column(DateTime, nullable=True)

    # Duration & aggregate stats
    sync_duration = mapped_column(Float, nullable=True)
    total_cycles_extracted = mapped_column(Integer, default=0, nullable=False)
    clean_cycles = mapped_column(Integer, default=0, nullable=False)
    outlier_cycles = mapped_column(Integer, default=0, nullable=False)
    
    # Cycle duration statistics
    mean_cycle_duration_s = mapped_column(Float, nullable=True)
    median_cycle_duration_s = mapped_column(Float, nullable=True)
    min_cycle_duration_s = mapped_column(Float, nullable=True)
    max_cycle_duration_s = mapped_column(Float, nullable=True)
    method_agreement_span = mapped_column(Float, nullable=True)

    # QC metadata
    audio_qc_version = mapped_column(String(20), nullable=True)
    biomech_qc_version = mapped_column(String(20), nullable=True)
    cycle_qc_version = mapped_column(String(20), nullable=True)
    sync_qc_fail: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    sync_qc_notes = mapped_column(Text, nullable=True)

    # Processing metadata
    processing_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    processing_status: Mapped[str] = mapped_column(String(50), default="not_processed", nullable=False)
    error_message = mapped_column(Text, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    participant: Mapped["ParticipantRecord"] = relationship("ParticipantRecord", back_populates="synchronizations")
    audio_processing: Mapped["AudioProcessingRecord"] = relationship("AudioProcessingRecord", back_populates="synchronizations")
    biomechanics_import: Mapped["BiomechanicsImportRecord"] = relationship("BiomechanicsImportRecord", back_populates="synchronizations")
    movement_cycles: Mapped[List["MovementCycleRecord"]] = relationship(
        "MovementCycleRecord", back_populates="synchronization", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("processing_status IN ('not_processed', 'success', 'error')", name="sync_valid_status"),
        # Unique constraint: one sync per audio+biomech combination
        UniqueConstraint("audio_processing_id", "biomechanics_import_id", name="uq_synchronization"),
    )


class MovementCycleRecord(Base):
    """Individual movement cycle metadata.

    Represents a single extracted movement cycle with references to:
    - Audio processing (required)
    - Biomechanics import (optional)
    - Synchronization (optional, only if biomechanics present)
    """
    __tablename__ = "movement_cycles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    participant_id: Mapped[int] = mapped_column(Integer, ForeignKey("participants.id"), nullable=False)
    
    # Foreign keys to related records
    audio_processing_id: Mapped[int] = mapped_column(Integer, ForeignKey("audio_processing.id"), nullable=False)
    biomechanics_import_id = mapped_column(Integer, ForeignKey("biomechanics_imports.id"), nullable=True)
    synchronization_id = mapped_column(Integer, ForeignKey("synchronizations.id"), nullable=True)

    # Cycle identification
    cycle_file: Mapped[str] = mapped_column(String(255), nullable=False)
    cycle_index: Mapped[int] = mapped_column(Integer, nullable=False)
    is_outlier: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Cycle temporal characteristics
    start_time_s: Mapped[float] = mapped_column(Float, nullable=False)
    end_time_s: Mapped[float] = mapped_column(Float, nullable=False)
    duration_s: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Audio timestamps (always present)
    audio_start_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    audio_end_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    
    # Biomechanics timestamps (optional - only if biomechanics_import_id is not null)
    bio_start_time = mapped_column(DateTime, nullable=True)
    bio_end_time = mapped_column(DateTime, nullable=True)

    # QC flags (cycle-specific)
    biomechanics_qc_fail: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    sync_qc_fail: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # QC versions
    biomechanics_qc_version = mapped_column(String(20), nullable=True)
    sync_qc_version = mapped_column(String(20), nullable=True)

    # Pickle file storage (cycle data)
    cycle_file_path = mapped_column(String(500), nullable=True)
    cycle_file_checksum = mapped_column(String(64), nullable=True)
    cycle_file_size_mb = mapped_column(Float, nullable=True)
    cycle_file_modified = mapped_column(DateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    participant: Mapped["ParticipantRecord"] = relationship("ParticipantRecord", back_populates="movement_cycles")
    audio_processing: Mapped["AudioProcessingRecord"] = relationship("AudioProcessingRecord", back_populates="movement_cycles")
    biomechanics_import: Mapped["BiomechanicsImportRecord"] = relationship("BiomechanicsImportRecord", back_populates="movement_cycles")
    synchronization: Mapped["SynchronizationRecord"] = relationship("SynchronizationRecord", back_populates="movement_cycles")

    __table_args__ = (
        # Unique constraint: one cycle per audio+index combination
        UniqueConstraint("audio_processing_id", "cycle_index", name="uq_movement_cycle"),
    )
