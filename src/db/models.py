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

from __future__ import annotations

from datetime import UTC, datetime

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


def utcnow() -> datetime:
    """Return timezone-aware UTC timestamp."""
    return datetime.now(UTC)


class ParticipantRecord(Base):
    """Stable participant identity.

    Permanent anchor row — never deleted.  System-generated PK that
    survives cleanup / re-processing cycles.  All mutable data lives in
    StudyRecord and downstream tables.
    """

    __tablename__ = "participants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    # Relationships
    studies: Mapped[list[StudyRecord]] = relationship(
        "StudyRecord", back_populates="participant", cascade="all, delete-orphan"
    )


class StudyRecord(Base):
    """Study enrollment — one row per (participant, study).

    Links a permanent participant to a named study with a
    study-specific participant number (e.g. 1016 in AOA).
    The ``id`` of this table is the FK used by all downstream
    processing tables.
    """

    __tablename__ = "studies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    participant_id: Mapped[int] = mapped_column(Integer, ForeignKey("participants.id"), nullable=False)
    study_name: Mapped[str] = mapped_column(String(50), nullable=False)  # AOA, preOA, SMoCK
    study_participant_id: Mapped[int] = mapped_column(Integer, nullable=False)  # e.g. 1016
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    # Relationships
    participant: Mapped[ParticipantRecord] = relationship("ParticipantRecord", back_populates="studies")
    audio_processing: Mapped[list[AudioProcessingRecord]] = relationship(
        "AudioProcessingRecord", back_populates="study", cascade="all, delete-orphan"
    )
    biomechanics_imports: Mapped[list[BiomechanicsImportRecord]] = relationship(
        "BiomechanicsImportRecord", back_populates="study", cascade="all, delete-orphan"
    )
    synchronizations: Mapped[list[SynchronizationRecord]] = relationship(
        "SynchronizationRecord", back_populates="study", cascade="all, delete-orphan"
    )
    movement_cycles: Mapped[list[MovementCycleRecord]] = relationship(
        "MovementCycleRecord", back_populates="study", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("study_name IN ('AOA', 'preOA', 'SMoCK')", name="valid_study_name"),
        UniqueConstraint("study_name", "study_participant_id", name="uq_study_participant"),
    )


class AudioProcessingRecord(Base):
    """Audio processing metadata.

    Standalone table for audio file metadata and QC results.
    Does NOT contain biomechanics information - that's in BiomechanicsImportRecord.
    """

    __tablename__ = "audio_processing"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    study_id: Mapped[int] = mapped_column(Integer, ForeignKey("studies.id"), nullable=False)

    # Recording-time FK: which biomechanics (if any) was recorded with this audio
    biomechanics_import_id = mapped_column(
        Integer,
        ForeignKey(
            "biomechanics_imports.id",
            use_alter=True,
            name="fk_audio_processing_biomechanics_import",
        ),
        nullable=True,
    )

    # File identification
    audio_file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    device_serial: Mapped[str] = mapped_column(String(50), nullable=False)
    firmware_version: Mapped[int] = mapped_column(Integer, nullable=False)
    file_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    file_size_mb: Mapped[float] = mapped_column(Float, nullable=False)

    # Recording metadata
    recording_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    recording_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    recording_timezone: Mapped[str] = mapped_column(String(10), nullable=True)

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

    # Continuous Artifact QC (detected at audio processing stage)
    qc_continuous_artifact: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    qc_continuous_artifact_segments = mapped_column(ARRAY(Float), nullable=True)
    qc_continuous_artifact_ch1: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    qc_continuous_artifact_segments_ch1 = mapped_column(ARRAY(Float), nullable=True)
    qc_continuous_artifact_ch2: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    qc_continuous_artifact_segments_ch2 = mapped_column(ARRAY(Float), nullable=True)
    qc_continuous_artifact_ch3: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    qc_continuous_artifact_segments_ch3 = mapped_column(ARRAY(Float), nullable=True)
    qc_continuous_artifact_ch4: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    qc_continuous_artifact_segments_ch4 = mapped_column(ARRAY(Float), nullable=True)

    # Soft-delete: False when record is not present in latest processing run
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Processing metadata
    processing_date: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    processing_status: Mapped[str] = mapped_column(String(50), default="not_processed", nullable=False)
    error_message = mapped_column(Text, nullable=True)
    duration_seconds = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)

    # Relationships
    study: Mapped[StudyRecord] = relationship("StudyRecord", back_populates="audio_processing")
    biomechanics_import: Mapped[BiomechanicsImportRecord | None] = relationship(
        "BiomechanicsImportRecord", foreign_keys=[biomechanics_import_id], uselist=False
    )
    synchronizations: Mapped[list[SynchronizationRecord]] = relationship(
        "SynchronizationRecord", back_populates="audio_processing", cascade="all, delete-orphan"
    )
    movement_cycles: Mapped[list[MovementCycleRecord]] = relationship(
        "MovementCycleRecord", back_populates="audio_processing", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("knee IN ('left', 'right')", name="audio_valid_knee"),
        CheckConstraint("maneuver IN ('fe', 'sts', 'walk')", name="audio_valid_maneuver"),
        CheckConstraint("processing_status IN ('not_processed', 'success', 'error')", name="audio_valid_status"),
        UniqueConstraint("study_id", "audio_file_name", "knee", "maneuver", name="uq_audio_processing"),
    )


class BiomechanicsImportRecord(Base):
    """Biomechanics import metadata.

    Standalone table for biomechanics file import and QC results.
    Does NOT contain audio information - that's in AudioProcessingRecord.
    """

    __tablename__ = "biomechanics_imports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    study_id: Mapped[int] = mapped_column(Integer, ForeignKey("studies.id"), nullable=False)

    # Recording-time FK: which audio (if any) was recorded with this biomechanics
    audio_processing_id = mapped_column(
        Integer,
        ForeignKey(
            "audio_processing.id",
            use_alter=True,
            name="fk_biomechanics_import_audio_processing",
        ),
        nullable=True,
    )

    # File identification
    biomechanics_file: Mapped[str] = mapped_column(String(255), nullable=False)
    sheet_name = mapped_column(String(100), nullable=True)
    biomechanics_type: Mapped[str] = mapped_column(String(50), nullable=False)  # Gonio, IMU, Motion Analysis

    # Maneuver metadata
    knee: Mapped[str] = mapped_column(String(10), nullable=False)
    maneuver: Mapped[str] = mapped_column(String(20), nullable=False)

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

    # Soft-delete: False when record is not present in latest processing run
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Processing metadata
    processing_date: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    processing_status: Mapped[str] = mapped_column(String(50), default="not_processed", nullable=False)
    error_message = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)

    # Relationships
    study: Mapped[StudyRecord] = relationship("StudyRecord", back_populates="biomechanics_imports")
    audio_processing: Mapped[AudioProcessingRecord | None] = relationship(
        "AudioProcessingRecord", foreign_keys=[audio_processing_id], uselist=False
    )
    synchronizations: Mapped[list[SynchronizationRecord]] = relationship(
        "SynchronizationRecord", back_populates="biomechanics_import", cascade="all, delete-orphan"
    )
    movement_cycles: Mapped[list[MovementCycleRecord]] = relationship(
        "MovementCycleRecord", back_populates="biomechanics_import"
    )

    __table_args__ = (
        CheckConstraint("knee IN ('left', 'right')", name="biomech_valid_knee"),
        CheckConstraint("maneuver IN ('fe', 'sts', 'walk')", name="biomech_valid_maneuver"),
        CheckConstraint("processing_status IN ('not_processed', 'success', 'error')", name="biomech_valid_status"),
        UniqueConstraint("study_id", "biomechanics_file", "knee", "maneuver", name="uq_biomechanics_import"),
    )


class SynchronizationRecord(Base):
    """Synchronization metadata.

    Tracks audio-biomechanics synchronization results.
    References AudioProcessingRecord and BiomechanicsImportRecord via foreign keys.
    """

    __tablename__ = "synchronizations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    study_id: Mapped[int] = mapped_column(Integer, ForeignKey("studies.id"), nullable=False)

    # Foreign keys to related records
    audio_processing_id: Mapped[int] = mapped_column(Integer, ForeignKey("audio_processing.id"), nullable=False)
    biomechanics_import_id: Mapped[int] = mapped_column(Integer, ForeignKey("biomechanics_imports.id"), nullable=False)

    # Maneuver metadata (denormalized from audio_processing for semantic uniqueness)
    knee: Mapped[str] = mapped_column(String(10), nullable=False)  # left, right
    maneuver: Mapped[str] = mapped_column(String(20), nullable=False)  # fe, sts, walk

    # Walk-specific metadata (optional, inherited from parent records)
    pass_number = mapped_column(Integer, nullable=True)
    speed = mapped_column(String(20), nullable=True)

    # Synchronization times
    # Note: Biomechanics is synchronized to audio data (audio time 0 = sync time 0)
    bio_left_sync_time = mapped_column(Float, nullable=True)
    bio_right_sync_time = mapped_column(Float, nullable=True)
    bio_sync_offset = mapped_column(Float, nullable=True)  # Biomechanics sync offset between legs
    aligned_sync_time = mapped_column(Float, nullable=True)  # Unified aligned sync time on merged dataframes

    # Sync method details
    sync_method = mapped_column(String(50), nullable=True)  # consensus, rms, onset, freq, biomechanics
    consensus_methods = mapped_column(String(100), nullable=True)
    consensus_time = mapped_column(Float, nullable=True)
    rms_time = mapped_column(Float, nullable=True)
    onset_time = mapped_column(Float, nullable=True)
    freq_time = mapped_column(Float, nullable=True)

    # Biomechanics-guided sync fields
    bio_selected_sync_time = mapped_column(Float, nullable=True)  # Renamed from selected_audio_sync_time
    contra_bio_selected_sync_time = mapped_column(Float, nullable=True)  # Renamed from contra_selected_audio_sync_time
    audio_visual_sync_time = mapped_column(Float, nullable=True)
    audio_visual_sync_time_contralateral = mapped_column(Float, nullable=True)

    # Audio sync times (optional - time between mic on and participant stopping for each leg)
    audio_sync_time_left = mapped_column(Float, nullable=True)
    audio_sync_time_right = mapped_column(Float, nullable=True)
    audio_sync_offset = mapped_column(Float, nullable=True)  # Required if both left and right are present

    # Audio-based sync fields (new - different from bio-based)
    audio_selected_sync_time = mapped_column(Float, nullable=True)  # Required if 'audio' in stomp_detection_methods
    contra_audio_selected_sync_time = mapped_column(
        Float, nullable=True
    )  # Required if 'audio' in stomp_detection_methods

    # Detection method fields
    stomp_detection_methods = mapped_column(
        ARRAY(String), nullable=True
    )  # List of methods used: ['audio', 'consensus', 'biomechanics']
    selected_stomp_method = mapped_column(
        String(50), nullable=True
    )  # Single selected method (replaces audio_stomp_method)

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

    # Periodic artifact detection (detected on full exercise portion of synced recording)
    periodic_artifact_detected: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    periodic_artifact_detected_ch1: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    periodic_artifact_detected_ch2: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    periodic_artifact_detected_ch3: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    periodic_artifact_detected_ch4: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    periodic_artifact_segments = mapped_column(ARRAY(Float), nullable=True)
    periodic_artifact_segments_ch1 = mapped_column(ARRAY(Float), nullable=True)
    periodic_artifact_segments_ch2 = mapped_column(ARRAY(Float), nullable=True)
    periodic_artifact_segments_ch3 = mapped_column(ARRAY(Float), nullable=True)
    periodic_artifact_segments_ch4 = mapped_column(ARRAY(Float), nullable=True)

    # Soft-delete: False when record is not present in latest processing run
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Processing metadata
    processing_date: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    processing_status: Mapped[str] = mapped_column(String(50), default="not_processed", nullable=False)
    error_message = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)

    # Relationships
    study: Mapped[StudyRecord] = relationship("StudyRecord", back_populates="synchronizations")
    audio_processing: Mapped[AudioProcessingRecord] = relationship(
        "AudioProcessingRecord", back_populates="synchronizations"
    )
    biomechanics_import: Mapped[BiomechanicsImportRecord] = relationship(
        "BiomechanicsImportRecord", back_populates="synchronizations"
    )
    movement_cycles: Mapped[list[MovementCycleRecord]] = relationship(
        "MovementCycleRecord", back_populates="synchronization", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("knee IN ('left', 'right')", name="sync_valid_knee"),
        CheckConstraint("maneuver IN ('fe', 'sts', 'walk')", name="sync_valid_maneuver"),
        CheckConstraint("processing_status IN ('not_processed', 'success', 'error')", name="sync_valid_status"),
        # Semantic uniqueness: one sync record per (study, knee, maneuver, pass, speed)
        UniqueConstraint("study_id", "knee", "maneuver", "pass_number", "speed", name="uq_synchronization"),
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
    study_id: Mapped[int] = mapped_column(Integer, ForeignKey("studies.id"), nullable=False)

    # Foreign keys to related records
    audio_processing_id: Mapped[int] = mapped_column(Integer, ForeignKey("audio_processing.id"), nullable=False)
    biomechanics_import_id = mapped_column(Integer, ForeignKey("biomechanics_imports.id"), nullable=True)
    synchronization_id = mapped_column(Integer, ForeignKey("synchronizations.id"), nullable=True)

    # Maneuver metadata (denormalized for semantic uniqueness)
    knee: Mapped[str] = mapped_column(String(10), nullable=False)  # left, right
    maneuver: Mapped[str] = mapped_column(String(20), nullable=False)  # fe, sts, walk
    pass_number = mapped_column(Integer, nullable=True)  # Walk-specific
    speed = mapped_column(String(20), nullable=True)  # Walk-specific

    # Cycle identification
    cycle_file: Mapped[str] = mapped_column(String(255), nullable=False)
    cycle_index: Mapped[int] = mapped_column(Integer, nullable=False)

    # Cycle temporal characteristics (in seconds)
    start_time_s: Mapped[float] = mapped_column(Float, nullable=False)
    end_time_s: Mapped[float] = mapped_column(Float, nullable=False)
    duration_s: Mapped[float] = mapped_column(Float, nullable=False)

    # Timestamps (datetime objects)
    start_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    end_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # QC flags (cycle-specific)
    is_outlier: Mapped[bool] = mapped_column(Boolean, nullable=False)
    biomechanics_qc_fail: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    sync_qc_fail: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    audio_qc_fail: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Audio QC failure types (list of failure modes)
    audio_qc_failures = mapped_column(ARRAY(String), nullable=True)

    # Audio artifact QC - intermittent artifacts detected at cycle stage
    audio_artifact_intermittent_fail: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    audio_artifact_intermittent_fail_ch1: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    audio_artifact_intermittent_fail_ch2: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    audio_artifact_intermittent_fail_ch3: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    audio_artifact_intermittent_fail_ch4: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Audio artifact timestamps (all artifacts within cycle: intermittent + continuous + periodic)
    audio_artifact_timestamps = mapped_column(ARRAY(Float), nullable=True)
    audio_artifact_timestamps_ch1 = mapped_column(ARRAY(Float), nullable=True)
    audio_artifact_timestamps_ch2 = mapped_column(ARRAY(Float), nullable=True)
    audio_artifact_timestamps_ch3 = mapped_column(ARRAY(Float), nullable=True)
    audio_artifact_timestamps_ch4 = mapped_column(ARRAY(Float), nullable=True)

    # Audio-stage dropout artifacts trimmed to cycle boundaries
    audio_artifact_dropout_fail: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    audio_artifact_dropout_fail_ch1: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    audio_artifact_dropout_fail_ch2: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    audio_artifact_dropout_fail_ch3: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    audio_artifact_dropout_fail_ch4: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    audio_artifact_dropout_timestamps = mapped_column(ARRAY(Float), nullable=True)
    audio_artifact_dropout_timestamps_ch1 = mapped_column(ARRAY(Float), nullable=True)
    audio_artifact_dropout_timestamps_ch2 = mapped_column(ARRAY(Float), nullable=True)
    audio_artifact_dropout_timestamps_ch3 = mapped_column(ARRAY(Float), nullable=True)
    audio_artifact_dropout_timestamps_ch4 = mapped_column(ARRAY(Float), nullable=True)

    # Audio-stage continuous artifacts trimmed to cycle boundaries
    audio_artifact_continuous_fail: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    audio_artifact_continuous_fail_ch1: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    audio_artifact_continuous_fail_ch2: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    audio_artifact_continuous_fail_ch3: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    audio_artifact_continuous_fail_ch4: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    audio_artifact_continuous_timestamps = mapped_column(ARRAY(Float), nullable=True)
    audio_artifact_continuous_timestamps_ch1 = mapped_column(ARRAY(Float), nullable=True)
    audio_artifact_continuous_timestamps_ch2 = mapped_column(ARRAY(Float), nullable=True)
    audio_artifact_continuous_timestamps_ch3 = mapped_column(ARRAY(Float), nullable=True)
    audio_artifact_continuous_timestamps_ch4 = mapped_column(ARRAY(Float), nullable=True)

    # Periodic artifact QC (propagated from sync-level detection, trimmed to cycle bounds)
    audio_artifact_periodic_fail: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    audio_artifact_periodic_fail_ch1: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    audio_artifact_periodic_fail_ch2: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    audio_artifact_periodic_fail_ch3: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    audio_artifact_periodic_fail_ch4: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    audio_artifact_periodic_timestamps = mapped_column(ARRAY(Float), nullable=True)
    audio_artifact_periodic_timestamps_ch1 = mapped_column(ARRAY(Float), nullable=True)
    audio_artifact_periodic_timestamps_ch2 = mapped_column(ARRAY(Float), nullable=True)
    audio_artifact_periodic_timestamps_ch3 = mapped_column(ARRAY(Float), nullable=True)
    audio_artifact_periodic_timestamps_ch4 = mapped_column(ARRAY(Float), nullable=True)

    # QC versions
    biomechanics_qc_version = mapped_column(String(20), nullable=True)
    sync_qc_version = mapped_column(String(20), nullable=True)

    # Soft-delete: False when record is not present in latest processing run
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Pickle file storage (cycle data)
    cycle_file_path = mapped_column(String(500), nullable=True)
    cycle_file_checksum = mapped_column(String(64), nullable=True)
    cycle_file_size_mb = mapped_column(Float, nullable=True)
    cycle_file_modified = mapped_column(DateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)

    # Relationships
    study: Mapped[StudyRecord] = relationship("StudyRecord", back_populates="movement_cycles")
    audio_processing: Mapped[AudioProcessingRecord] = relationship(
        "AudioProcessingRecord", back_populates="movement_cycles"
    )
    biomechanics_import: Mapped[BiomechanicsImportRecord] = relationship(
        "BiomechanicsImportRecord", back_populates="movement_cycles"
    )
    synchronization: Mapped[SynchronizationRecord] = relationship(
        "SynchronizationRecord", back_populates="movement_cycles"
    )

    __table_args__ = (
        CheckConstraint("knee IN ('left', 'right')", name="cycle_valid_knee"),
        CheckConstraint("maneuver IN ('fe', 'sts', 'walk')", name="cycle_valid_maneuver"),
        # Semantic uniqueness: one cycle per (study, knee, maneuver, pass, speed, cycle_index)
        UniqueConstraint(
            "study_id", "knee", "maneuver", "pass_number", "speed", "cycle_index", name="uq_movement_cycle"
        ),
    )
