"""Object-oriented participant processing with clear state management."""

from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
import re
from typing import Literal, cast

import pandas as pd

from src.biomechanics.importers import import_biomechanics_recordings
from src.metadata import AudioProcessing, BiomechanicsImport, CycleQCResult, Synchronization
from src.orchestration.participant import (
    _normalize_maneuver as _expand_maneuver_shorthand,
)
from src.orchestration.processing_log import (
    KneeProcessingLog,
    ManeuverProcessingLog,
    create_audio_record_from_data,
    create_biomechanics_record_from_data,
    create_sync_record_from_data,
)
from src.orchestration.processing_log import (
    _normalize_maneuver as _maneuver_to_db_code,
)
from src.synchronization.sync import load_audio_data

# Maneuver folder naming variants
_MANEUVER_ALIAS_MAP: dict[str, set[str]] = {
    "walk": {"walking", "walk"},
    "sit_to_stand": {
        "sit-stand",
        "sit-to-stand",
        "sit_to_stand",
        "sitstand",
        "sit to stand",
        "sittostand",
    },
    "flexion_extension": {
        "flexion-extension",
        "flexion_extension",
        "flexion extension",
        "flexionextension",
    },
}


def _normalize_folder_name(name: str) -> str:
    """Normalize maneuver folder names to compare variants."""
    name = name.lower()
    name = name.replace("_", "-").replace(" ", "-")
    name = re.sub(r"-+", "-", name)
    return name


@dataclass
class AudioData:
    """Encapsulates audio processing state for a maneuver."""

    pkl_path: Path
    df: pd.DataFrame | None = None
    metadata: dict | None = None
    record: AudioProcessing | None = None


@dataclass
class BiomechanicsData:
    """Encapsulates biomechanics processing state for a maneuver."""

    file_path: Path
    recordings: list = field(default_factory=list)
    record: BiomechanicsImport | None = None


@dataclass
class SyncData:
    """Encapsulates a single synchronized pass."""

    output_path: Path
    df: pd.DataFrame
    stomp_times: tuple  # (audio_stomp, bio_left, bio_right, detection_results)
    record: Synchronization | None = None
    pass_number: int | None = None  # For walk maneuvers
    speed: str | None = None  # For walk maneuvers


@dataclass
class CycleData:
    """Encapsulates cycle QC results for a sync file."""

    synced_file_path: Path
    output_dir: Path | None = None
    record: Synchronization | None = None  # Synchronization record with cycle details
    sync_file_stem: str | None = None
    cycle_qc_results: list[CycleQCResult] = field(default_factory=list)


class ManeuverProcessor:
    """Processes a single maneuver (walk/sit-to-stand/flexion-extension) for a knee."""

    def __init__(
        self,
        maneuver_dir: Path,
        maneuver_key: Literal["walk", "sit_to_stand", "flexion_extension"],
        knee_side: Literal["Left", "Right"],
        study_id: str,
        biomechanics_file: Path,
        biomechanics_type: str | None = None,
        study_name: str = "AOA",
    ):
        self.maneuver_dir = maneuver_dir
        self.maneuver_key = maneuver_key
        self.knee_side = knee_side
        self.study_id = study_id
        self.biomechanics_file = biomechanics_file
        self.biomechanics_type = biomechanics_type
        self.study_name = study_name

        # Processing state
        self.audio: AudioData | None = None
        self.biomechanics: BiomechanicsData | None = None
        self.synced_data: list[SyncData] = []
        self.cycle_data: list[CycleData] = []
        self.log: ManeuverProcessingLog | None = None

    def process_bin_stage(self) -> bool:
        """Read .bin file, run QC, add frequency, and produce *_with_freq.pkl."""
        try:
            logging.info(f"Processing {self.knee_side} {self.maneuver_key} bin stage")

            # Find .bin file
            bin_path = self._find_bin_file()
            if not bin_path or not bin_path.exists():
                logging.error(f".bin file not found in {self.maneuver_dir}")
                return False

            # Create outputs directory
            audio_base_name = bin_path.stem
            outputs_dir = self.maneuver_dir / f"{audio_base_name}_outputs"
            outputs_dir.mkdir(exist_ok=True)

            # Always reprocess when bin stage is invoked
            audio_pkl_path = outputs_dir / f"{audio_base_name}_with_freq.pkl"

            # Read .bin file (creates base pkl)
            from src.audio.readers import read_audio_board_file

            audio_df = read_audio_board_file(str(bin_path), str(outputs_dir))

            # Check that base pkl was created
            base_pkl = outputs_dir / f"{audio_base_name}.pkl"
            if not base_pkl.exists():
                logging.error(f"Base pickle not created after reading {bin_path}")
                return False

            # Run raw audio QC
            from src.audio.raw_qc import (
                detect_artifactual_noise_per_mic,
                detect_continuous_background_noise,
                detect_continuous_background_noise_per_mic,
                detect_signal_dropout_per_mic,
                merge_bad_intervals,
                run_raw_audio_qc,
            )

            dropout_intervals, artifact_intervals = run_raw_audio_qc(audio_df)
            merge_bad_intervals(dropout_intervals, artifact_intervals)
            dropout_per_mic = detect_signal_dropout_per_mic(audio_df)
            artifact_per_mic, artifact_types_per_mic = detect_artifactual_noise_per_mic(audio_df)

            # Continuous narrowband background noise detection (spectral)
            continuous_intervals = detect_continuous_background_noise(audio_df)
            continuous_per_mic = detect_continuous_background_noise_per_mic(audio_df)

            fs = self._infer_sample_rate(audio_df)

            # Add instantaneous frequency
            from src.audio.instantaneous_frequency import add_instantaneous_frequency

            audio_df = add_instantaneous_frequency(audio_df, fs)

            # Save frequency-augmented pickle
            audio_pkl_path = outputs_dir / f"{audio_base_name}_with_freq.pkl"
            audio_df.to_pickle(audio_pkl_path)

            # Load metadata if available
            audio_metadata = self._load_audio_metadata(audio_pkl_path) or {}
            audio_metadata.update(self._load_mic_positions_from_legend())
            audio_metadata["recording_timezone"] = "UTC"
            audio_metadata["fs"] = fs

            # Format QC data for audio record
            # Convert intervals to lists for Pydantic validation
            # Compute overall artifact type by combining all channels
            overall_artifact_types = []
            for ch_num in range(1, 5):
                ch_name = f"ch{ch_num}"
                if ch_name in artifact_types_per_mic:
                    overall_artifact_types.extend(artifact_types_per_mic[ch_name])
            # Remove duplicates while preserving order
            seen = set()
            overall_artifact_types = [x for x in overall_artifact_types if not (x in seen or seen.add(x))]  # type: ignore[func-returns-value]

            # Merge ALL fail types into overall qc_fail_segments:
            # signal dropout + intermittent artifacts + continuous artifacts
            all_fail_intervals = (
                list(dropout_intervals or []) + list(artifact_intervals or []) + list(continuous_intervals or [])
            )
            overall_fail_segments = merge_bad_intervals(all_fail_intervals, []) if all_fail_intervals else []

            qc_data = {
                "qc_fail_segments": overall_fail_segments,
                "qc_signal_dropout": bool(dropout_intervals),
                "qc_signal_dropout_segments": list(dropout_intervals) if dropout_intervals else [],
                "qc_continuous_artifact": bool(continuous_intervals),
                "qc_continuous_artifact_segments": list(continuous_intervals) if continuous_intervals else [],
            }

            # Add per-channel QC results and merge into per-channel qc_fail_segments
            for ch_num in range(1, 5):
                ch_name = f"ch{ch_num}"
                ch_fail_sources = []

                # Per-channel continuous artifact (spectral detection)
                if continuous_per_mic.get(ch_name):
                    ch_continuous = list(continuous_per_mic[ch_name])
                    qc_data[f"qc_continuous_artifact_ch{ch_num}"] = True
                    qc_data[f"qc_continuous_artifact_segments_ch{ch_num}"] = ch_continuous
                    ch_fail_sources.extend(ch_continuous)

                # Per-channel signal dropout
                if dropout_per_mic.get(ch_name):
                    ch_dropout = list(dropout_per_mic[ch_name])
                    qc_data[f"qc_signal_dropout_ch{ch_num}"] = True
                    qc_data[f"qc_signal_dropout_segments_ch{ch_num}"] = ch_dropout
                    ch_fail_sources.extend(ch_dropout)

                # Per-channel intermittent artifact
                if artifact_per_mic.get(ch_name):
                    ch_fail_sources.extend(artifact_per_mic[ch_name])

                # Merge all per-channel fail sources into qc_fail_segments_chX
                qc_data[f"qc_fail_segments_ch{ch_num}"] = (
                    merge_bad_intervals(ch_fail_sources, []) if ch_fail_sources else []
                )

            self.audio = AudioData(
                pkl_path=audio_pkl_path,
                df=audio_df,
                metadata=audio_metadata,
            )

            # Ensure maneuver is in metadata
            if audio_metadata is None:
                audio_metadata = {}
            audio_metadata["maneuver"] = self.maneuver_key
            audio_metadata["study_id"] = int(self.study_id)
            audio_metadata["processing_date"] = datetime.now()
            audio_metadata["processing_status"] = "success"

            # Create audio record with QC data
            self.audio.record = create_audio_record_from_data(
                audio_file_name=audio_base_name,
                audio_df=audio_df,
                audio_bin_path=bin_path,
                audio_pkl_path=audio_pkl_path,
                metadata=audio_metadata,
                biomechanics_type=None,  # Not yet linked
                qc_data=qc_data,
                knee=self.knee_side.lower(),
                maneuver=self.maneuver_key,
            )

            return True
        except Exception as e:
            logging.error(f"Bin stage failed for {self.knee_side} {self.maneuver_key}: {e}", exc_info=True)
            return False

    def process_sync_stage(self) -> bool:
        """Synchronize audio with biomechanics."""
        # Load audio state if not already loaded (handles resuming from sync)
        if (not self.audio or self.audio.df is None) and not self._load_existing_audio_state():
            logging.error("Audio state must be available to run sync stage")
            return False

        try:
            logging.info(f"Processing {self.knee_side} {self.maneuver_key} sync stage")

            # Load biomechanics
            self.biomechanics = self._load_biomechanics()
            if not self.biomechanics.recordings:
                logging.warning(f"No biomechanics recordings found for {self.knee_side} {self.maneuver_key}")
                return True  # Not a failure, just no biomechanics

            # Sync each recording
            for recording in self.biomechanics.recordings:
                sync_result = self._sync_recording(recording)
                if sync_result:
                    self.synced_data.append(sync_result)

            # Update audio record with biomechanics info
            if self.audio.record and self.biomechanics.recordings:  # type: ignore[union-attr]
                self.audio.record.linked_biomechanics = True  # type: ignore[union-attr]
                self.audio.record.biomechanics_file = str(self.biomechanics_file)  # type: ignore[union-attr]
                self.audio.record.biomechanics_type = self.biomechanics_type  # type: ignore[union-attr]
                self.audio.record.log_updated = datetime.now()  # type: ignore[union-attr]
                if self.biomechanics.recordings:
                    first_rec = self.biomechanics.recordings[0]
                    # Try to infer sample rate from the biomechanics data
                    try:
                        # Check if data is a BiomechanicsData object with a TIME column
                        if hasattr(first_rec, "data") and first_rec.data is not None:
                            data_obj = first_rec.data
                            # BiomechanicsData is a dataclass with a 'data' DataFrame attribute
                            if hasattr(data_obj, "data"):
                                time_col = data_obj.data.get("TIME", None)
                            else:
                                # data_obj is a DataFrame
                                time_col = data_obj.get("TIME", None) if hasattr(data_obj, "get") else None

                            if time_col is not None and len(time_col) > 1:
                                time_diffs = time_col.diff().dropna()
                                if len(time_diffs) > 0:
                                    avg_diff_sec = time_diffs.mean().total_seconds()
                                    if avg_diff_sec > 0:
                                        self.audio.record.biomechanics_sample_rate = 1.0 / avg_diff_sec  # type: ignore[union-attr]
                        elif hasattr(first_rec, "sample_rate"):
                            self.audio.record.biomechanics_sample_rate = float(first_rec.sample_rate)  # type: ignore[union-attr]
                    except Exception as e:
                        logging.debug(f"Could not determine biomechanics sample rate: {e}")
                # Set sync method based on biomechanics type
                if self.biomechanics_type == "Gonio":
                    self.audio.record.biomechanics_sync_method = "flick"  # type: ignore[union-attr]
                else:
                    self.audio.record.biomechanics_sync_method = "stomp"  # type: ignore[union-attr]

            # Create biomechanics record
            self.biomechanics.record = create_biomechanics_record_from_data(
                biomechanics_file=self.biomechanics_file,
                recordings=self.biomechanics.recordings,
                sheet_name=f"{self.maneuver_key}_data",
                maneuver=self.maneuver_key,
                biomechanics_type=self.biomechanics_type,
                knee=self.knee_side.lower(),
                biomechanics_sync_method=("flick" if self.biomechanics_type == "Gonio" else "stomp"),
                biomechanics_sample_rate=getattr(self.audio.record, "biomechanics_sample_rate", None)
                if self.audio
                else None,
                study_id=int(self.study_id),
            )

            # Create Synchronization records from SyncData objects
            for sync_data in self.synced_data:
                # Always create/update sync record during sync stage to ensure
                # detection_results (including method_agreement_span) are current
                # Unpack stomp_times tuple: (audio_stomp, bio_left, bio_right, detection_results)
                audio_stomp, bio_left, bio_right, detection_results = sync_data.stomp_times

                # Prepare kwargs for create_sync_record_from_data
                sync_kwargs = {
                    "sync_file_name": sync_data.output_path.stem,
                    "synced_df": sync_data.df,
                    "audio_stomp_time": audio_stomp,
                    "bio_left_stomp_time": bio_left,
                    "bio_right_stomp_time": bio_right,
                    "knee_side": self.knee_side.lower(),
                    "maneuver": self.maneuver_key,
                    "pass_number": sync_data.pass_number,
                    "speed": sync_data.speed,
                    "detection_results": detection_results,
                    "audio_record": self.audio.record if self.audio else None,
                    "metadata": {},
                }

                sync_data.record = create_sync_record_from_data(**sync_kwargs)

            return True
        except Exception as e:
            logging.error(f"Sync stage failed for {self.knee_side} {self.maneuver_key}: {e}")
            return False

    def process_cycles_stage(self) -> bool:
        """Run movement cycle extraction and QC on all synced files.

        Extracts movement cycles from synchronized data, applies quality control
        checks (biomechanics ROM, acoustic energy, audio QC integration), and
        generates cycle-level records with per-cycle details.

        If resuming from cycles stage (synced_data is empty), loads synced files
        from disk and processes them.

        Returns:
            True if successful, False otherwise
        """
        # Load synced files from disk if needed (handles resuming from cycles)
        if not self.synced_data and not self._load_existing_synced_data():
            logging.warning(f"No synced data to run cycles on for {self.knee_side} {self.maneuver_key}")
            return True  # Not a failure

        try:
            from src.synchronization.quality_control import perform_sync_qc

            logging.info(f"Processing {self.knee_side} {self.maneuver_key} cycles stage")

            # Process each synced file
            for sync_data in self.synced_data:
                try:
                    # Determine maneuver and speed from sync_data
                    maneuver = self.maneuver_key

                    # Extract speed from output path if available (for walking)
                    speed = None
                    if maneuver == "walk":
                        # Infer speed from filename: left_walk_p1_slow.pkl
                        filename = sync_data.output_path.stem
                        for s in ["slow", "medium", "normal", "fast"]:
                            if s in filename.lower():
                                # Normalize "normal" → "medium" for DB consistency
                                speed = "medium" if s == "normal" else s
                                break

                    # Determine output directory for cycle files
                    output_dir = sync_data.output_path.parent

                    # Run complete QC pipeline (extraction + QC)
                    qc_output = perform_sync_qc(
                        synced_pkl_path=sync_data.output_path,
                        output_dir=output_dir,
                        maneuver=maneuver,  # type: ignore[arg-type]
                        speed=speed,  # type: ignore[arg-type]
                        acoustic_threshold=100.0,  # Default threshold
                        create_plots=True,
                        bad_audio_segments=None,  # Will load from processing log
                    )

                    clean_cycles = qc_output.clean_cycles  # type: ignore[attr-defined]
                    outlier_cycles = qc_output.outlier_cycles  # type: ignore[attr-defined]

                    logging.info(
                        f"Extracted {len(clean_cycles)} clean and {len(outlier_cycles)} outlier cycles "
                        f"from {sync_data.output_path.name}"
                    )

                    # Update sync record with cycle stats and periodic artifacts
                    # (record is None when entering from cycles entrypoint)
                    if sync_data.record is not None:
                        # Store sync-level periodic artifact results (Phase E.1)
                        sync_periodic = qc_output.sync_periodic_results  # type: ignore[attr-defined]
                        if sync_periodic:
                            sync_data.record.periodic_artifact_detected = sync_periodic.get(
                                "periodic_artifact_detected", False
                            )
                            sync_data.record.periodic_artifact_segments = sync_periodic.get(
                                "periodic_artifact_segments", None
                            )
                            for ch in ["ch1", "ch2", "ch3", "ch4"]:
                                setattr(
                                    sync_data.record,
                                    f"periodic_artifact_detected_{ch}",
                                    sync_periodic.get(f"periodic_artifact_detected_{ch}", False),
                                )
                                setattr(
                                    sync_data.record,
                                    f"periodic_artifact_segments_{ch}",
                                    sync_periodic.get(f"periodic_artifact_segments_{ch}", None),
                                )

                        # Update sync record with cycle statistics
                        sync_data.record.total_cycles_extracted = len(clean_cycles) + len(outlier_cycles)
                        sync_data.record.clean_cycles = len(clean_cycles)
                        sync_data.record.outlier_cycles = len(outlier_cycles)

                        # Calculate cycle duration statistics from ALL cycles (clean + outliers)
                        all_cycles = clean_cycles + outlier_cycles
                        if all_cycles:
                            cycle_durations = []
                            for cycle in all_cycles:
                                duration_value = None
                                if hasattr(cycle, "duration_s"):
                                    duration_value = cycle.duration_s
                                elif isinstance(cycle, dict):
                                    duration_value = cycle.get("duration_s") or cycle.get("duration_seconds")
                                elif isinstance(cycle, pd.DataFrame):
                                    if "tt" in cycle.columns and len(cycle) > 1:
                                        delta = cycle["tt"].iloc[-1] - cycle["tt"].iloc[0]
                                        if hasattr(delta, "total_seconds"):
                                            duration_value = delta.total_seconds()
                                        else:
                                            try:
                                                duration_value = float(delta)
                                            except (TypeError, ValueError):
                                                duration_value = None

                                if duration_value is not None:
                                    cycle_durations.append(float(duration_value))

                            if cycle_durations:
                                import statistics

                                sync_data.record.mean_cycle_duration_s = statistics.mean(cycle_durations)
                                sync_data.record.median_cycle_duration_s = statistics.median(cycle_durations)
                                sync_data.record.min_cycle_duration_s = min(cycle_durations)
                                sync_data.record.max_cycle_duration_s = max(cycle_durations)

                    # Store cycle record in CycleData (using the updated sync record)
                    cycle_data = CycleData(
                        synced_file_path=sync_data.output_path,
                        output_dir=qc_output.output_dir,  # type: ignore[attr-defined]
                        record=sync_data.record,  # Use the updated sync record directly
                        sync_file_stem=sync_data.output_path.stem,
                        cycle_qc_results=qc_output.cycle_qc_results,  # type: ignore[attr-defined]
                    )
                    self.cycle_data.append(cycle_data)

                except Exception as e:
                    logging.error(f"Failed to process cycles for {sync_data.output_path}: {e}", exc_info=True)
                    # Continue processing other files
                    continue

            return True
        except Exception as e:
            logging.error(f"Cycles stage failed for {self.knee_side} {self.maneuver_key}: {e}", exc_info=True)
            return False

    def _load_existing_synced_data(self) -> bool:
        """Load existing synced files when resuming from cycles stage.

        Returns:
            True if synced files were found and loaded, False otherwise
        """
        try:
            synced_dir = self.maneuver_dir / "Synced"
            if not synced_dir.exists():
                synced_dir = self.maneuver_dir / "synced"
            if not synced_dir.exists():
                return False

            # Find all synced pickle files
            synced_files = list(synced_dir.glob("*.pkl"))
            if not synced_files:
                return False

            # Load each synced file
            for synced_file in synced_files:
                try:
                    df = pd.read_pickle(synced_file)
                    sync_data = SyncData(
                        output_path=synced_file,
                        df=df,
                        stomp_times=(0.0, 0.0, 0.0, {}),  # Placeholder
                    )
                    self.synced_data.append(sync_data)
                except Exception as e:
                    logging.warning(f"Failed to load synced file {synced_file}: {e}")

            if self.synced_data:
                logging.info(f"Loaded {len(self.synced_data)} synced file(s) from {synced_dir}")
                return True

            return False
        except Exception as e:
            logging.error(f"Failed to load existing synced data: {e}")
            return False

    def save_logs(self) -> bool:
        """Update and save processing logs."""
        try:
            from src.db.repository import Repository
            from src.orchestration.processing_log import (
                close_db_session,
                create_db_session,
            )

            # Get or create maneuver log
            self.log = ManeuverProcessingLog.get_or_create(
                study_id=self.study_id,
                knee_side=self.knee_side,
                maneuver=self.maneuver_key,
                maneuver_directory=self.maneuver_dir,
            )

            # Update audio record
            if self.audio and self.audio.record:
                self.log.update_audio_record(self.audio.record)

            # Update biomechanics record
            if self.biomechanics and self.biomechanics.record:
                self.log.update_biomechanics_record(self.biomechanics.record)

            # Update synchronization records
            for sync in self.synced_data:
                if sync.record:
                    self.log.add_synchronization_record(sync.record)

            # Update movement cycles records
            for cycle in self.cycle_data:
                if cycle.record:
                    self.log.add_movement_cycles_record(cycle.record)

            # Persist records to database before generating report
            session = create_db_session()
            if session is not None:
                try:
                    repo = Repository(session)
                    sync_records_by_name: dict[str, object] = {}

                    # Track all persisted record IDs for deactivation
                    seen_audio_ids: set[int] = set()
                    seen_biomech_ids: set[int] = set()
                    seen_sync_ids: set[int] = set()
                    seen_cycle_ids: set[int] = set()

                    audio_db_record = None
                    biomech_db_record = None

                    # Save audio record (available when running from bin/sync)
                    if self.log.audio_record:
                        audio_db_record = repo.save_audio_processing(self.log.audio_record)
                        seen_audio_ids.add(audio_db_record.id)

                    # Save biomechanics record linked to audio
                    if self.log.biomechanics_record and audio_db_record:
                        biomech_db_record = repo.save_biomechanics_import(
                            self.log.biomechanics_record, audio_processing_id=audio_db_record.id
                        )
                        seen_biomech_ids.add(biomech_db_record.id)

                        # Link audio record to biomechanics record
                        audio_db_record.biomechanics_import_id = biomech_db_record.id
                        session.flush()

                    # Save synchronization records linked to both
                    if audio_db_record and biomech_db_record:
                        for sync_record in self.log.synchronization_records:
                            db_sync = repo.save_synchronization(
                                sync_record,
                                audio_processing_id=audio_db_record.id,
                                biomechanics_import_id=biomech_db_record.id,
                            )
                            seen_sync_ids.add(db_sync.id)
                            if sync_record.sync_file_name:
                                sync_records_by_name[sync_record.sync_file_name] = db_sync
                                sync_records_by_name[Path(sync_record.sync_file_name).stem] = db_sync

                    # When resuming from cycles-only, look up existing parent
                    # records from DB so we can persist cycles and run deactivation
                    if audio_db_record is None:
                        audio_db_record, biomech_db_record = self._lookup_existing_parent_records(
                            repo, sync_records_by_name
                        )
                        if audio_db_record:
                            seen_audio_ids.add(audio_db_record.id)
                        if biomech_db_record:
                            seen_biomech_ids.add(biomech_db_record.id)
                        # Mark sync records as "seen" if their synced file was processed
                        processed_stems = {sd.output_path.stem for sd in self.synced_data if sd.output_path}
                        for name, sr in sync_records_by_name.items():
                            if hasattr(sr, "id") and (name in processed_stems or Path(name).stem in processed_stems):
                                seen_sync_ids.add(sr.id)

                    # Save movement cycle records
                    if audio_db_record:
                        seen_cycle_ids = self._persist_cycle_records(
                            repo=repo,
                            audio_db_record=audio_db_record,
                            biomech_db_record=biomech_db_record,
                            sync_records_by_name=sync_records_by_name,
                        )

                    # Deactivate records not seen in this processing run
                    if audio_db_record:
                        study_db_id = audio_db_record.study_id
                        deactivated = repo.deactivate_unseen_records(
                            study_id=study_db_id,
                            knee=self.knee_side.lower(),
                            maneuver=_maneuver_to_db_code(self.maneuver_key),
                            seen_audio_ids=seen_audio_ids,
                            seen_biomech_ids=seen_biomech_ids,
                            seen_sync_ids=seen_sync_ids,
                            seen_cycle_ids=seen_cycle_ids,
                        )
                        total_deactivated = sum(deactivated.values())
                        if total_deactivated > 0:
                            logging.info(f"Deactivated {total_deactivated} stale record(s): {deactivated}")

                    session.commit()
                finally:
                    close_db_session(session)

            # Generate Excel report from database
            legend_mismatches = getattr(self, "_legend_mismatches", None)
            self.log.save_to_excel(legend_mismatches=legend_mismatches)
            return True
        except Exception as e:
            logging.error(f"Failed to save logs: {e}", exc_info=True)
            return False

    @staticmethod
    def _unflatten_intervals(
        flat: list[float] | list[list[float]] | list[tuple[float, float]] | None,
    ) -> list[tuple[float, float]]:
        """Convert DB segment data to [(start, end), ...] tuple pairs.

        PostgreSQL ARRAY(Float) columns return nested lists when the data
        was originally stored as list[tuple].  This method handles both
        formats:
          - Nested: [[s1, e1], [s2, e2], ...]  (from PostgreSQL round-trip)
          - Flat:   [s1, e1, s2, e2, ...]       (from explicit flattening)
        """
        if not flat:
            return []
        # Detect nested format: first element is a list or tuple of length 2
        first = flat[0]
        if isinstance(first, (list, tuple)):
            return [(float(pair[0]), float(pair[1])) for pair in flat]  # type: ignore[index]
        # Flat format: pair up consecutive elements
        if len(flat) % 2 != 0:
            return []
        return [(float(flat[i]), float(flat[i + 1])) for i in range(0, len(flat), 2)]  # type: ignore[arg-type]

    @staticmethod
    def _flatten_intervals(intervals: list[tuple[float, float]]) -> list[float]:
        """Convert [(s1,e1), (s2,e2), ...] to flattened [s1, e1, s2, e2, ...]."""
        result: list[float] = []
        for start, end in intervals:
            result.extend([float(start), float(end)])
        return result

    def _persist_cycle_records(
        self,
        *,
        repo,
        audio_db_record,
        biomech_db_record,
        sync_records_by_name: dict[str, object],
    ) -> set[int]:
        """Persist per-cycle records to the database.

        Returns:
            Set of MovementCycleRecord.id values for all persisted cycles.
        """
        from datetime import datetime, timedelta

        from src.metadata import MovementCycle

        def _to_seconds(value) -> float:
            if value is None:
                return 0.0
            if hasattr(value, "total_seconds"):
                return float(value.total_seconds())
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        def _combine_recording_datetime() -> datetime:
            date_val = getattr(audio_db_record, "recording_date", None)
            time_val = getattr(audio_db_record, "recording_time", None)
            if isinstance(date_val, datetime) and isinstance(time_val, datetime):
                return datetime.combine(date_val.date(), time_val.time())
            if isinstance(time_val, datetime):
                return time_val
            if isinstance(date_val, datetime):
                return date_val
            file_time = getattr(audio_db_record, "file_time", None)
            if isinstance(file_time, datetime):
                return file_time
            return datetime.now()

        base_dt = _combine_recording_datetime()
        seen_cycle_ids: set[int] = set()

        for cycle_data in self.cycle_data:
            if not cycle_data.output_dir or not cycle_data.output_dir.exists():
                continue

            sync_stem = cycle_data.sync_file_stem or cycle_data.synced_file_path.stem
            sync_db_record = sync_records_by_name.get(sync_stem)
            if sync_db_record is None:
                sync_db_record = sync_records_by_name.get(f"{sync_stem}.pkl")
            if sync_db_record is None:
                from src.db.models import SynchronizationRecord

                base_sync_query = repo.session.query(SynchronizationRecord).filter(
                    SynchronizationRecord.audio_processing_id == audio_db_record.id
                )
                if biomech_db_record is not None:
                    base_sync_query = base_sync_query.filter(
                        SynchronizationRecord.biomechanics_import_id == biomech_db_record.id
                    )
                sync_query = base_sync_query
                if sync_stem:
                    sync_query = sync_query.filter(
                        SynchronizationRecord.sync_file_name.in_([sync_stem, f"{sync_stem}.pkl"])
                    )
                sync_db_record = sync_query.first()
                if sync_db_record is None and self.maneuver_key != "walk":
                    sync_db_record = base_sync_query.order_by(SynchronizationRecord.id.desc()).first()

            # Build lookup from in-memory CycleQCResult objects
            qc_by_file: dict[str, CycleQCResult] = {r.cycle_file: r for r in cycle_data.cycle_qc_results}

            for pkl_path in sorted(cycle_data.output_dir.rglob("*.pkl")):
                if not pkl_path.name.startswith(sync_stem):
                    continue

                qc = qc_by_file.get(pkl_path.name)
                if qc is None:
                    logging.warning(f"No CycleQCResult for {pkl_path.name}, skipping")
                    continue

                try:
                    df = pd.read_pickle(pkl_path)
                except Exception:
                    continue

                if "tt" not in df.columns or len(df) == 0:
                    continue

                start_time_s = _to_seconds(df["tt"].iloc[0])
                end_time_s = _to_seconds(df["tt"].iloc[-1])
                duration_s = max(0.0, end_time_s - start_time_s)

                start_time = base_dt + timedelta(seconds=start_time_s)
                end_time = base_dt + timedelta(seconds=end_time_s)

                pass_number = qc.pass_number
                speed = qc.speed

                if sync_db_record is not None:
                    if pass_number is None:
                        pass_number = getattr(sync_db_record, "pass_number", None)
                    if speed is None:
                        speed = getattr(sync_db_record, "speed", None)

                synchronization_id = None
                if sync_db_record is not None:
                    if self.maneuver_key == "walk":
                        if pass_number is not None and speed is not None:
                            synchronization_id = getattr(sync_db_record, "id", None)
                    else:
                        synchronization_id = getattr(sync_db_record, "id", None)

                # --- Audio-stage QC: trim to cycle boundaries ---
                from src.audio.raw_qc import trim_intervals_to_cycle

                audio_qc_failures: list[str] = []

                # Check signal dropout overlap with this cycle — keep trimmed intervals
                dropout_per_ch: dict[int, list[tuple]] = {}
                has_dropout = False
                for ch_num in range(1, 5):
                    flat = getattr(audio_db_record, f"qc_signal_dropout_segments_ch{ch_num}", None)
                    trimmed = trim_intervals_to_cycle(self._unflatten_intervals(flat), start_time_s, end_time_s)
                    dropout_per_ch[ch_num] = trimmed
                    if trimmed:
                        has_dropout = True
                if has_dropout:
                    audio_qc_failures.append("dropout")

                # Check continuous artifact overlap with this cycle — keep trimmed intervals
                continuous_per_ch: dict[int, list[tuple]] = {}
                has_continuous = False
                for ch_num in range(1, 5):
                    flat = getattr(audio_db_record, f"qc_continuous_artifact_segments_ch{ch_num}", None)
                    trimmed = trim_intervals_to_cycle(self._unflatten_intervals(flat), start_time_s, end_time_s)
                    continuous_per_ch[ch_num] = trimmed
                    if trimmed:
                        has_continuous = True
                if has_continuous:
                    audio_qc_failures.append("continuous")

                # --- Cycle-stage QC: intermittent artifacts ---
                has_intermittent = any(
                    [
                        qc.intermittent_intervals_ch1,
                        qc.intermittent_intervals_ch2,
                        qc.intermittent_intervals_ch3,
                        qc.intermittent_intervals_ch4,
                    ]
                )
                if has_intermittent:
                    audio_qc_failures.append("intermittent")

                # --- Cycle-stage QC: periodic artifacts ---
                if qc.periodic_noise_detected:
                    audio_qc_failures.append("periodic")

                # Aggregate audio QC fail
                audio_qc_fail = len(audio_qc_failures) > 0

                cycle = MovementCycle(
                    study=self.log.audio_record.study if self.log and self.log.audio_record else self.study_name,  # type: ignore[arg-type]
                    study_id=int(self.study_id),
                    audio_processing_id=audio_db_record.id,
                    biomechanics_import_id=biomech_db_record.id if biomech_db_record else None,
                    synchronization_id=synchronization_id,
                    knee=self.knee_side.lower(),  # type: ignore[arg-type]
                    maneuver=_maneuver_to_db_code(self.maneuver_key),  # type: ignore[arg-type]
                    pass_number=pass_number,
                    speed=speed,
                    cycle_file=pkl_path.name,
                    cycle_index=int(qc.cycle_index),
                    is_outlier=qc.is_outlier,
                    start_time_s=float(start_time_s),
                    end_time_s=float(end_time_s),
                    duration_s=float(duration_s),
                    start_time=start_time,
                    end_time=end_time,
                    biomechanics_qc_fail=qc.is_outlier,
                    sync_qc_fail=not qc.sync_qc_pass,
                    # Aggregate audio QC
                    audio_qc_fail=audio_qc_fail,
                    audio_qc_failures=audio_qc_failures if audio_qc_failures else None,  # type: ignore[arg-type]
                    # Dropout artifacts (audio-stage, trimmed to cycle)
                    audio_artifact_dropout_fail=has_dropout,
                    audio_artifact_dropout_fail_ch1=bool(dropout_per_ch.get(1)),
                    audio_artifact_dropout_fail_ch2=bool(dropout_per_ch.get(2)),
                    audio_artifact_dropout_fail_ch3=bool(dropout_per_ch.get(3)),
                    audio_artifact_dropout_fail_ch4=bool(dropout_per_ch.get(4)),
                    audio_artifact_dropout_timestamps=self._flatten_intervals(
                        [iv for ch in dropout_per_ch.values() for iv in ch]
                    )
                    or None,
                    audio_artifact_dropout_timestamps_ch1=self._flatten_intervals(dropout_per_ch.get(1, [])) or None,
                    audio_artifact_dropout_timestamps_ch2=self._flatten_intervals(dropout_per_ch.get(2, [])) or None,
                    audio_artifact_dropout_timestamps_ch3=self._flatten_intervals(dropout_per_ch.get(3, [])) or None,
                    audio_artifact_dropout_timestamps_ch4=self._flatten_intervals(dropout_per_ch.get(4, [])) or None,
                    # Continuous artifacts (audio-stage, trimmed to cycle)
                    audio_artifact_continuous_fail=has_continuous,
                    audio_artifact_continuous_fail_ch1=bool(continuous_per_ch.get(1)),
                    audio_artifact_continuous_fail_ch2=bool(continuous_per_ch.get(2)),
                    audio_artifact_continuous_fail_ch3=bool(continuous_per_ch.get(3)),
                    audio_artifact_continuous_fail_ch4=bool(continuous_per_ch.get(4)),
                    audio_artifact_continuous_timestamps=self._flatten_intervals(
                        [iv for ch in continuous_per_ch.values() for iv in ch]
                    )
                    or None,
                    audio_artifact_continuous_timestamps_ch1=self._flatten_intervals(continuous_per_ch.get(1, []))
                    or None,
                    audio_artifact_continuous_timestamps_ch2=self._flatten_intervals(continuous_per_ch.get(2, []))
                    or None,
                    audio_artifact_continuous_timestamps_ch3=self._flatten_intervals(continuous_per_ch.get(3, []))
                    or None,
                    audio_artifact_continuous_timestamps_ch4=self._flatten_intervals(continuous_per_ch.get(4, []))
                    or None,
                    # Intermittent artifacts (cycle-stage)
                    audio_artifact_intermittent_fail=has_intermittent,
                    audio_artifact_intermittent_fail_ch1=bool(qc.intermittent_intervals_ch1),
                    audio_artifact_intermittent_fail_ch2=bool(qc.intermittent_intervals_ch2),
                    audio_artifact_intermittent_fail_ch3=bool(qc.intermittent_intervals_ch3),
                    audio_artifact_intermittent_fail_ch4=bool(qc.intermittent_intervals_ch4),
                    audio_artifact_timestamps=self._flatten_intervals(
                        qc.intermittent_intervals_ch1
                        + qc.intermittent_intervals_ch2
                        + qc.intermittent_intervals_ch3
                        + qc.intermittent_intervals_ch4
                    )
                    or None,
                    audio_artifact_timestamps_ch1=self._flatten_intervals(qc.intermittent_intervals_ch1) or None,
                    audio_artifact_timestamps_ch2=self._flatten_intervals(qc.intermittent_intervals_ch2) or None,
                    audio_artifact_timestamps_ch3=self._flatten_intervals(qc.intermittent_intervals_ch3) or None,
                    audio_artifact_timestamps_ch4=self._flatten_intervals(qc.intermittent_intervals_ch4) or None,
                    # Periodic artifacts (sync-level, trimmed to cycle)
                    audio_artifact_periodic_fail=qc.periodic_noise_detected,
                    audio_artifact_periodic_fail_ch1=qc.periodic_noise_ch1,
                    audio_artifact_periodic_fail_ch2=qc.periodic_noise_ch2,
                    audio_artifact_periodic_fail_ch3=qc.periodic_noise_ch3,
                    audio_artifact_periodic_fail_ch4=qc.periodic_noise_ch4,
                    audio_artifact_periodic_timestamps=self._flatten_intervals(
                        qc.periodic_intervals_ch1
                        + qc.periodic_intervals_ch2
                        + qc.periodic_intervals_ch3
                        + qc.periodic_intervals_ch4
                    )
                    or None,
                    audio_artifact_periodic_timestamps_ch1=self._flatten_intervals(qc.periodic_intervals_ch1) or None,
                    audio_artifact_periodic_timestamps_ch2=self._flatten_intervals(qc.periodic_intervals_ch2) or None,
                    audio_artifact_periodic_timestamps_ch3=self._flatten_intervals(qc.periodic_intervals_ch3) or None,
                    audio_artifact_periodic_timestamps_ch4=self._flatten_intervals(qc.periodic_intervals_ch4) or None,
                )

                cycle_record = repo.save_movement_cycle(
                    cycle,
                    audio_processing_id=audio_db_record.id,
                    biomechanics_import_id=biomech_db_record.id if biomech_db_record else None,
                    synchronization_id=synchronization_id,
                    cycles_file_path=str(pkl_path),
                )
                seen_cycle_ids.add(cycle_record.id)

        return seen_cycle_ids

    def _lookup_existing_parent_records(
        self,
        repo,
        sync_records_by_name: dict[str, object],
    ) -> tuple:
        """Look up existing audio/biomech/sync records from DB.

        Used when resuming from cycles-only entrypoint — audio and biomechanics
        records were not created in this run but exist from a previous run.

        Returns:
            (audio_db_record, biomech_db_record) tuple. Either may be None.
        """
        from sqlalchemy import and_, select

        from src.db.models import (
            AudioProcessingRecord,
            BiomechanicsImportRecord,
            StudyRecord,
            SynchronizationRecord,
        )

        db_maneuver = _maneuver_to_db_code(self.maneuver_key)
        knee = self.knee_side.lower()

        # Look up the study record for this participant
        study = repo.session.execute(
            select(StudyRecord).where(
                and_(
                    StudyRecord.study_name == self.study_name,
                    StudyRecord.study_participant_id == int(self.study_id),
                )
            )
        ).scalar_one_or_none()
        if study is None:
            logging.warning(f"No study record found for participant {self.study_id}")
            return None, None

        # Look up existing audio record
        audio_db_record = repo.session.execute(
            select(AudioProcessingRecord).where(
                and_(
                    AudioProcessingRecord.study_id == study.id,
                    AudioProcessingRecord.knee == knee,
                    AudioProcessingRecord.maneuver == db_maneuver,
                    AudioProcessingRecord.is_active == True,
                )
            )
        ).scalar_one_or_none()

        if audio_db_record is None:
            logging.warning(f"No active audio record found for {knee} {db_maneuver} (participant {self.study_id})")
            return None, None

        # Look up existing biomechanics record
        biomech_db_record = repo.session.execute(
            select(BiomechanicsImportRecord).where(
                and_(
                    BiomechanicsImportRecord.study_id == study.id,
                    BiomechanicsImportRecord.knee == knee,
                    BiomechanicsImportRecord.maneuver == db_maneuver,
                    BiomechanicsImportRecord.is_active == True,
                )
            )
        ).scalar_one_or_none()

        # Populate sync_records_by_name from existing sync records
        sync_records = (
            repo.session.execute(
                select(SynchronizationRecord).where(
                    and_(
                        SynchronizationRecord.study_id == study.id,
                        SynchronizationRecord.knee == knee,
                        SynchronizationRecord.maneuver == db_maneuver,
                        SynchronizationRecord.is_active == True,
                    )
                )
            )
            .scalars()
            .all()
        )
        for sr in sync_records:
            if sr.sync_file_name:
                sync_records_by_name[sr.sync_file_name] = sr
                sync_records_by_name[Path(sr.sync_file_name).stem] = sr

        logging.info(
            f"Looked up existing parent records: audio={audio_db_record.id}, "
            f"biomech={biomech_db_record.id if biomech_db_record else None}, "
            f"syncs={len(sync_records)}"
        )
        return audio_db_record, biomech_db_record

    def _load_existing_audio_state(self) -> bool:
        """Load existing audio state when resuming from a later stage.

        This handles the case where bin stage has already been completed
        and we're resuming from sync or cycles. Loads the audio pickle
        and metadata from disk.

        Returns:
            True if audio state was successfully loaded, False otherwise
        """
        try:
            audio_pkl_path = self._find_audio_pickle()
            if not audio_pkl_path or not audio_pkl_path.exists():
                logging.warning(f"No existing audio pickle found in {self.maneuver_dir}")
                return False

            # Load audio data
            audio_df = load_audio_data(audio_pkl_path)

            # Load metadata if available
            audio_metadata = self._load_audio_metadata(audio_pkl_path) or {}
            audio_metadata.update(self._load_mic_positions_from_legend())
            audio_metadata["recording_timezone"] = "UTC"
            audio_metadata["fs"] = self._infer_sample_rate(audio_df)

            # Create AudioData with loaded state
            self.audio = AudioData(
                pkl_path=audio_pkl_path,
                df=audio_df,
                metadata=audio_metadata,
            )

            # Ensure maneuver is in metadata
            audio_metadata["maneuver"] = self.maneuver_key
            audio_metadata["study_id"] = int(self.study_id)
            audio_metadata["processing_date"] = datetime.now()
            audio_metadata["processing_status"] = "success"

            # Create audio record
            audio_bin_path = self._find_bin_file()
            audio_file_base = (
                Path(audio_bin_path).stem if audio_bin_path else audio_pkl_path.stem.replace("_with_freq", "")
            )
            self.audio.record = create_audio_record_from_data(
                audio_file_name=audio_file_base,
                audio_df=audio_df,
                audio_bin_path=audio_bin_path,
                audio_pkl_path=audio_pkl_path,
                metadata=audio_metadata,
                biomechanics_type=None,  # Will be linked during sync
                knee=self.knee_side.lower(),
                maneuver=self.maneuver_key,
            )

            logging.info(f"Loaded existing audio state for {self.knee_side} {self.maneuver_key}")
            return True
        except Exception as e:
            logging.error(f"Failed to load existing audio state: {e}")
            return False

    def _find_audio_pickle(self) -> Path | None:
        """Find the audio pickle file."""
        # Look in outputs subdirectories first
        for pkl in self.maneuver_dir.glob("*_outputs/*_with_freq.pkl"):
            return pkl
        # Fall back to direct search in maneuver dir
        for pkl in self.maneuver_dir.glob("*_with_freq.pkl"):
            return pkl
        return None

    def _find_bin_file(self) -> Path | None:
        """Find the .bin audio file."""
        for bin_file in self.maneuver_dir.glob("*.bin"):
            return bin_file
        return None

    def _infer_sample_rate(self, df: pd.DataFrame) -> float:
        """Infer sample rate from the time column when possible."""
        if "tt" not in df.columns:
            return 46875.0
        try:
            tt = pd.to_numeric(df["tt"], errors="coerce").dropna().to_numpy()
            if len(tt) < 2:
                return 46875.0
            dt = float(pd.Series(tt).diff().median())
            if dt <= 0:
                return 46875.0
            return 1.0 / dt
        except Exception:
            return 46875.0

    def _load_mic_positions_from_legend(self) -> dict:
        """Load microphone positions from the acoustics file legend."""
        from src.studies import get_study_config

        study_config = get_study_config(self.study_name)
        participant_dir = self.maneuver_dir.parents[1]
        legend_pattern = study_config.get_legend_file_pattern()
        legend_files = [
            f
            for f in participant_dir.glob(f"{legend_pattern}.xls*")
            if not f.name.startswith("~$")  # Exclude Excel temp/lock files
        ]
        if not legend_files:
            raise FileNotFoundError(f"No acoustic file legend found in {participant_dir}")
        legend_path = legend_files[0]

        from src.audio.parsers import get_acoustics_metadata

        meta, legend_mismatches = get_acoustics_metadata(
            metadata_file_path=str(legend_path),
            scripted_maneuver=self.maneuver_key,
            knee=self.knee_side.lower(),  # type: ignore[arg-type]
            study_name=self.study_name,
        )
        self._legend_mismatches = legend_mismatches

        def _to_code(pos) -> str:
            patellar = "I" if pos.patellar_position == "Infrapatellar" else "S"
            lateral = "M" if pos.laterality == "Medial" else "L"
            return f"{patellar}P{lateral}"

        mic_positions = {}
        for mic_num, pos in meta.microphones.items():
            mic_positions[f"mic_{mic_num}_position"] = _to_code(pos)

        if len(mic_positions) != 4:
            raise ValueError(f"Incomplete microphone positions in legend: {legend_path}")

        return mic_positions

    def _load_audio_metadata(self, pkl_path: Path) -> dict | None:
        """Load audio metadata from _meta.json file."""
        meta_json_path = pkl_path.parent / f"{pkl_path.stem.replace('_with_freq', '')}_meta.json"
        if meta_json_path.exists():
            import json

            try:
                with open(meta_json_path) as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load metadata from {meta_json_path}: {e}")
                return {}
        return {}

    def _load_biomechanics(self) -> BiomechanicsData:
        """Load biomechanics recordings."""
        recordings = []
        try:
            if self.maneuver_key == "walk":
                for speed in ["slow", "medium", "fast"]:
                    speed_recordings = import_biomechanics_recordings(
                        biomechanics_file=self.biomechanics_file,
                        maneuver=self.maneuver_key,
                        speed=speed,  # type: ignore[arg-type]
                    )
                    recordings.extend(speed_recordings)
            else:
                recordings = import_biomechanics_recordings(
                    biomechanics_file=self.biomechanics_file,
                    maneuver=self.maneuver_key,
                )
        except Exception as e:
            logging.warning(f"Failed to load biomechanics recordings: {e}")

        return BiomechanicsData(file_path=self.biomechanics_file, recordings=recordings)

    def _load_event_data(self) -> pd.DataFrame | None:
        """Load event metadata from biomechanics file.

        Returns:
            DataFrame with event metadata, or None if loading fails
        """
        try:
            from src.orchestration.participant import _load_event_data

            return _load_event_data(
                biomechanics_file=self.biomechanics_file,
                maneuver_key=self.maneuver_key,
            )
        except Exception as e:
            logging.warning(f"Failed to load event metadata: {e}")
            return None

    def _sync_recording(self, recording) -> SyncData | None:
        """Synchronize a single recording with audio using stomp detection.

        Performs multi-method stomp detection (RMS, onset, frequency) on audio,
        retrieves biomechanics stomp times from event metadata, and synchronizes
        the data streams by aligning their stomp times.

        Args:
            recording: BiomechanicsRecording object with biomechanics data and metadata

        Returns:
            SyncData object containing synchronized DataFrame and stomp times, or None on failure
        """
        try:
            from src.orchestration.participant import (
                _get_foot_from_knee_side,
                _trim_and_rename_biomechanics_columns,
            )
            from src.synchronization.sync import (
                get_audio_stomp_time,
                get_bio_end_time,
                get_bio_start_time,
                get_left_stomp_time,
                get_right_stomp_time,
                get_stomp_time,
                plot_stomp_detection,
                sync_audio_with_biomechanics,
            )

            if not self.audio or self.audio.df is None:
                logging.warning("No audio data available for syncing")
                return None

            audio_df = self.audio.df
            bio_df = recording.data

            # Load event metadata from biomechanics file
            event_meta_data = self._load_event_data()
            if event_meta_data is None:
                logging.warning(f"No event metadata available for {self.maneuver_key}")
                return None

            # Get both stomp times from biomechanics (for dual-knee disambiguation)
            right_stomp_time = get_right_stomp_time(event_meta_data)
            left_stomp_time = get_left_stomp_time(event_meta_data)

            # Get recorded knee
            recorded_knee = "left" if self.knee_side == "Left" else "right"

            # Get audio stomp time with multi-method detection and consensus
            try:
                study_name = self.biomechanics_file.stem.split("_")[0]
            except Exception:
                study_name = None

            audio_stomp_time, detection_results = get_audio_stomp_time(  # type: ignore[misc]
                audio_df,
                recorded_knee=recorded_knee,  # type: ignore[arg-type]
                right_stomp_time=right_stomp_time,
                left_stomp_time=left_stomp_time,
                return_details=True,
                biomechanics_type=self.biomechanics_type,
                study_name=study_name,
            )

            # Get biomechanics stomp time for the recorded knee
            bio_stomp_time = get_stomp_time(
                bio_meta=event_meta_data,
                foot=_get_foot_from_knee_side(self.knee_side),
            )

            # Get pass number and speed from recording
            pass_number = getattr(recording, "pass_number", None)
            speed = getattr(recording, "speed", None)

            # Normalize speed (medium -> normal for event metadata lookups)
            normalized_speed = None
            if speed is not None:
                event_speed_map = {
                    "slow": "slow",
                    "medium": "normal",
                    "normal": "normal",
                    "fast": "fast",
                }
                normalized_speed = event_speed_map.get(speed, speed)

            # Get maneuver timing for clipping
            if self.maneuver_key == "walk":
                bio_start_time = get_bio_start_time(
                    event_metadata=event_meta_data,
                    maneuver=self.maneuver_key,
                    speed=normalized_speed,
                    pass_number=pass_number,
                )
                bio_end_time = get_bio_end_time(
                    event_metadata=event_meta_data,
                    maneuver=self.maneuver_key,
                    speed=normalized_speed,
                    pass_number=pass_number,
                )
            else:
                bio_start_time = get_bio_start_time(
                    event_metadata=event_meta_data,
                    maneuver=self.maneuver_key,
                    speed=None,
                    pass_number=None,
                )
                bio_end_time = get_bio_end_time(
                    event_metadata=event_meta_data,
                    maneuver=self.maneuver_key,
                    speed=None,
                    pass_number=None,
                )

            # Synchronize audio with biomechanics
            synced_df = sync_audio_with_biomechanics(
                audio_stomp_time=audio_stomp_time,
                bio_stomp_time=bio_stomp_time,
                audio_df=audio_df,
                bio_df=bio_df,
                bio_start_time=bio_start_time,
                bio_end_time=bio_end_time,
                maneuver_key=self.maneuver_key,
                knee_side=self.knee_side,
                pass_number=pass_number,
                speed=speed,
            )

            # Trim and rename biomechanics columns for knee laterality
            trimmed_df = _trim_and_rename_biomechanics_columns(synced_df, self.knee_side)

            # Generate output path
            output_path = self._generate_sync_output_path(recording)

            # Save synchronized data to disk
            output_path.parent.mkdir(parents=True, exist_ok=True)
            trimmed_df.to_pickle(output_path)
            logging.debug(f"Saved synchronized data to {output_path}")

            # Generate stomp visualization
            try:
                plot_stomp_detection(
                    audio_df,
                    bio_df,
                    trimmed_df,
                    audio_stomp_time,
                    left_stomp_time,
                    right_stomp_time,
                    output_path,
                    detection_results,
                )
                logging.debug(f"Saved stomp detection visualization for {output_path.stem}")
            except Exception as e:
                logging.debug(f"Could not generate stomp visualization: {e}")

            # Return SyncData with actual stomp times and detection results
            stomp_times = (audio_stomp_time, left_stomp_time, right_stomp_time, detection_results)

            return SyncData(
                output_path=output_path,
                df=trimmed_df,
                stomp_times=stomp_times,
                pass_number=pass_number,
                speed=speed,
            )

        except Exception as e:
            logging.error(f"Failed to sync recording: {e}", exc_info=True)
            return None

    def _generate_sync_output_path(self, recording) -> Path:
        """Generate output path for synchronized data."""
        from src.studies.file_naming import generate_sync_filename

        synced_dir = self.maneuver_dir / "synced"
        synced_dir.mkdir(exist_ok=True)

        speed = getattr(recording, "speed", None)
        pass_num = getattr(recording, "pass_number", None)

        filename = generate_sync_filename(
            knee=self.knee_side.lower(),
            maneuver=_maneuver_to_db_code(self.maneuver_key),
            pass_number=pass_num,
            speed=speed,
        )

        return synced_dir / filename


class KneeProcessor:
    """Processes all maneuvers for a single knee."""

    def __init__(
        self,
        knee_dir: Path,
        knee_side: Literal["Left", "Right"],
        study_id: str,
        biomechanics_file: Path,
        biomechanics_type: str | None = None,
        study_name: str = "AOA",
    ):
        from src.studies import get_study_config

        self.knee_dir = knee_dir
        self.knee_side = knee_side
        self.study_id = study_id
        self.biomechanics_file = biomechanics_file
        self.biomechanics_type = biomechanics_type
        self.study_config = get_study_config(study_name)

        self.maneuver_processors: dict[str, ManeuverProcessor] = {}
        self.knee_log: KneeProcessingLog | None = None

    def process(
        self,
        entrypoint: Literal["bin", "sync", "cycles"] = "sync",
        maneuver: str | None = None,
    ) -> bool:
        """Process all maneuvers for this knee."""
        try:
            # Normalize maneuver if provided (convert CLI shorthand like "fe" to internal format)
            if maneuver:
                maneuver = _expand_maneuver_shorthand(maneuver)
            maneuvers_to_process = [maneuver] if maneuver else ["walk", "sit_to_stand", "flexion_extension"]

            for maneuver_key in maneuvers_to_process:
                maneuver_dir = self._find_maneuver_dir(maneuver_key)
                if not maneuver_dir:
                    logging.warning(f"Maneuver {maneuver_key} not found for {self.knee_side}")
                    continue

                processor = ManeuverProcessor(
                    maneuver_dir=maneuver_dir,
                    maneuver_key=cast(Literal["walk", "sit_to_stand", "flexion_extension"], maneuver_key),
                    knee_side=self.knee_side,
                    study_id=self.study_id,
                    biomechanics_file=self.biomechanics_file,
                    biomechanics_type=self.biomechanics_type,
                    study_name=self.study_config.study_name,
                )

                if not self._run_processor(processor, entrypoint):
                    return False

                self.maneuver_processors[maneuver_key] = processor

            # Knee log saving is non-critical — don't abort the participant
            self._save_knee_log()
            return True
        except Exception as e:
            logging.error(f"Knee processing failed: {e}")
            return False

    def _run_processor(self, proc: ManeuverProcessor, entrypoint: str) -> bool:
        """Run stages from entrypoint onwards."""
        stages_map = {
            "bin": [
                proc.process_bin_stage,
                proc.process_sync_stage,
                proc.process_cycles_stage,
            ],
            "sync": [proc.process_sync_stage, proc.process_cycles_stage],
            "cycles": [proc.process_cycles_stage],
        }

        for stage_func in stages_map[entrypoint]:
            if not stage_func():
                return False

        # Log saving is non-critical — don't abort remaining maneuvers
        # if the DB is unavailable or the report generator fails.
        proc.save_logs()
        return True

    def _find_maneuver_dir(self, maneuver_key: str) -> Path | None:
        """Find the directory for a maneuver using alias matching.

        Handles naming variations like "Sit-Stand", "Sit_Stand", "sit-to-stand", etc.
        """
        aliases = _MANEUVER_ALIAS_MAP.get(maneuver_key, set())
        try:
            for child in self.knee_dir.iterdir():
                if not child.is_dir():
                    continue
                norm = _normalize_folder_name(child.name)
                if norm in aliases:
                    return child
        except (OSError, PermissionError):
            pass
        return None

    def _save_knee_log(self) -> bool:
        """Save knee-level aggregated log."""
        try:
            self.knee_log = KneeProcessingLog.get_or_create(
                study_id=self.study_id,
                knee_side=self.knee_side,
                knee_directory=self.knee_dir,
            )

            for maneuver_key, proc in self.maneuver_processors.items():
                if proc.log:
                    self.knee_log.update_maneuver_summary(
                        cast(Literal["walk", "sit_to_stand", "flexion_extension"], maneuver_key), proc.log
                    )

            self.knee_log.save_to_excel()
            return True
        except Exception as e:
            logging.error(f"Failed to save knee log: {e}")
            return False


class ParticipantProcessor:
    """Orchestrates processing of a complete participant directory."""

    def __init__(
        self,
        participant_dir: Path,
        biomechanics_type: str | None = None,
        study_name: str = "AOA",
    ):
        from src.studies import get_study_config

        self.participant_dir = participant_dir
        self.study_config = get_study_config(study_name)
        self.study_id = participant_dir.name.lstrip("#")
        self.biomechanics_type = biomechanics_type
        self.biomechanics_file = self._find_biomechanics_file()

        self.knee_processors: dict[str, KneeProcessor] = {}

    def process(
        self,
        entrypoint: Literal["bin", "sync", "cycles"] = "sync",
        knee: str | None = None,
        maneuver: str | None = None,
    ) -> bool:
        """Process participant."""
        try:
            logging.info(f"Processing participant {self.study_id}")

            # Validate directory structure based on entrypoint
            self._validate_directory_structure(entrypoint, knee, maneuver)

            knees_to_process = [knee] if knee else ["Left", "Right"]

            for knee_side in knees_to_process:
                knee_dir_name = self.study_config.get_knee_directory_name(knee_side.lower())  # type: ignore[arg-type]
                knee_dir = self.participant_dir / knee_dir_name
                if not knee_dir.exists():
                    logging.warning(f"{knee_dir_name} directory not found")
                    continue

                processor = KneeProcessor(
                    knee_dir=knee_dir,
                    knee_side=cast(Literal["Left", "Right"], knee_side),
                    study_id=self.study_id,
                    biomechanics_file=self.biomechanics_file,
                    biomechanics_type=self.biomechanics_type,
                    study_name=self.study_config.study_name,
                )

                if not processor.process(entrypoint, maneuver):
                    return False

                self.knee_processors[knee_side] = processor

            return True
        except Exception as e:
            logging.error(f"Participant processing failed: {e}")
            return False

    def _validate_directory_structure(
        self,
        entrypoint: Literal["bin", "sync", "cycles"],
        knee: str | None = None,
        maneuver: str | None = None,
    ) -> None:
        """Validate directory structure based on entrypoint.

        Raises FileNotFoundError if required directories or files are missing.
        """
        # Always require Motion Capture folder with biomechanics file
        if not self.biomechanics_file.exists():
            raise FileNotFoundError(f"Biomechanics Excel file not found: {self.biomechanics_file}")

        # Validate knee directories
        knees_to_check = [knee] if knee else ["Left", "Right"]
        for knee_side in knees_to_check:
            knee_dir = self.participant_dir / self.study_config.get_knee_directory_name(knee_side.lower())  # type: ignore[arg-type]
            if not knee_dir.exists():
                raise FileNotFoundError(f"Required knee directory not found: {knee_dir}")

            # Validate maneuver directories based on entrypoint
            if entrypoint == "bin":
                # For bin stage, just check that maneuver directories exist
                self._validate_maneuver_dirs(knee_dir, knee_side, entrypoint, maneuver)
            elif entrypoint in ("sync", "cycles"):
                # For sync/cycles, maneuver directories should exist but processed files
                # may or may not exist (they get loaded if available)
                pass

    def _find_biomechanics_file(self) -> Path:
        """Find the biomechanics Excel file using study-specific naming pattern."""
        mc_dir_name = self.study_config.get_motion_capture_directory_name()
        motion_capture_dir = self.participant_dir / mc_dir_name
        if not motion_capture_dir.exists():
            raise FileNotFoundError(f"{mc_dir_name} directory not found in {self.participant_dir}")

        biomech_pattern = self.study_config.get_biomechanics_file_pattern(self.study_id)
        found = self.study_config.find_excel_file(motion_capture_dir, biomech_pattern)
        if found:
            return found

        raise FileNotFoundError(
            f"Biomechanics Excel file matching '{biomech_pattern}' not found in {motion_capture_dir}"
        )

    def _validate_maneuver_dirs(
        self,
        knee_dir: Path,
        knee_side: str,
        entrypoint: str,
        maneuver: str | None = None,
    ) -> None:
        """Validate that required maneuver directories exist.

        Raises FileNotFoundError if required maneuver directories are missing.
        """
        maneuvers_to_check = [maneuver] if maneuver else ["walk", "sit_to_stand", "flexion_extension"]

        for maneuver_key in maneuvers_to_check:
            # Use KneeProcessor's find method for flexible folder name matching
            processor = KneeProcessor(
                knee_dir=knee_dir,
                knee_side=cast(Literal["Left", "Right"], knee_side),
                study_id=self.study_id,
                biomechanics_file=self.biomechanics_file,
                study_name=self.study_config.study_name,
            )
            maneuver_dir = processor._find_maneuver_dir(maneuver_key)

            if maneuver_dir is None:
                raise FileNotFoundError(f"Required maneuver directory for '{maneuver_key}' not found in {knee_dir}")
