"""Object-oriented participant processing with clear state management."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, cast

import pandas as pd

from src.biomechanics.importers import import_biomechanics_recordings
from src.metadata import AudioProcessing, BiomechanicsImport, Synchronization
from src.orchestration.processing_log import (
    KneeProcessingLog,
    ManeuverProcessingLog,
    create_audio_record_from_data,
    create_biomechanics_record_from_data,
    create_cycles_record_from_data,
    create_sync_record_from_data,
)
from src.synchronization.sync import load_audio_data


@dataclass
class AudioData:
    """Encapsulates audio processing state for a maneuver."""
    pkl_path: Path
    df: Optional[pd.DataFrame] = None
    metadata: Optional[dict] = None
    qc_not_passed: Optional[str] = None
    qc_not_passed_mic_1: Optional[str] = None
    qc_not_passed_mic_2: Optional[str] = None
    qc_not_passed_mic_3: Optional[str] = None
    qc_not_passed_mic_4: Optional[str] = None
    record: Optional[AudioProcessing] = None


@dataclass
class BiomechanicsData:
    """Encapsulates biomechanics processing state for a maneuver."""
    file_path: Path
    recordings: list = field(default_factory=list)
    record: Optional[BiomechanicsImport] = None


@dataclass
class SyncData:
    """Encapsulates a single synchronized pass."""
    output_path: Path
    df: pd.DataFrame
    stomp_times: tuple  # (audio_stomp, bio_left, bio_right, detection_results)
    record: Optional[Synchronization] = None


@dataclass
class CycleData:
    """Encapsulates cycle QC results for a sync file."""
    synced_file_path: Path
    output_dir: Optional[Path] = None
    record: Optional[dict] = None


class ManeuverProcessor:
    """Processes a single maneuver (walk/sit-to-stand/flexion-extension) for a knee."""

    def __init__(
        self,
        maneuver_dir: Path,
        maneuver_key: Literal["walk", "sit_to_stand", "flexion_extension"],
        knee_side: Literal["Left", "Right"],
        study_id: str,
        biomechanics_file: Path,
        biomechanics_type: Optional[str] = None,
    ):
        self.maneuver_dir = maneuver_dir
        self.maneuver_key = maneuver_key
        self.knee_side = knee_side
        self.study_id = study_id
        self.biomechanics_file = biomechanics_file
        self.biomechanics_type = biomechanics_type

        # Processing state
        self.audio: Optional[AudioData] = None
        self.biomechanics: Optional[BiomechanicsData] = None
        self.synced_data: list[SyncData] = []
        self.cycle_data: list[CycleData] = []
        self.log: Optional[ManeuverProcessingLog] = None

    def process_bin_stage(self) -> bool:
        """Load and process raw audio to frequency-augmented pickle."""
        try:
            logging.info(f"Processing {self.knee_side} {self.maneuver_key} bin stage")

            # Find audio pickle file
            audio_pkl_path = self._find_audio_pickle()
            if not audio_pkl_path or not audio_pkl_path.exists():
                logging.error(f"Audio pickle not found in {self.maneuver_dir}")
                return False

            # Load audio data
            audio_df = load_audio_data(audio_pkl_path)

            # Load metadata if available
            audio_metadata = self._load_audio_metadata(audio_pkl_path)

            # TODO: Add QC processing here (extract qc_not_passed values)
            qc_not_passed = None
            qc_not_passed_mic_1 = None
            qc_not_passed_mic_2 = None
            qc_not_passed_mic_3 = None
            qc_not_passed_mic_4 = None

            self.audio = AudioData(
                pkl_path=audio_pkl_path,
                df=audio_df,
                metadata=audio_metadata,
                qc_not_passed=qc_not_passed,
                qc_not_passed_mic_1=qc_not_passed_mic_1,
                qc_not_passed_mic_2=qc_not_passed_mic_2,
                qc_not_passed_mic_3=qc_not_passed_mic_3,
                qc_not_passed_mic_4=qc_not_passed_mic_4,
            )

            # Create audio record
            self.audio.record = create_audio_record_from_data(
                audio_file_name=audio_pkl_path.stem,
                audio_df=audio_df,
                audio_bin_path=self._find_bin_file(),
                audio_pkl_path=audio_pkl_path,
                metadata=audio_metadata,
                biomechanics_type=None,  # Not yet linked
            )

            return True
        except Exception as e:
            logging.error(f"Bin stage failed for {self.knee_side} {self.maneuver_key}: {e}")
            return False

    def process_sync_stage(self) -> bool:
        """Synchronize audio with biomechanics."""
        if not self.audio or self.audio.df is None:
            logging.error("Audio must be processed before sync")
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
            if self.audio.record and self.biomechanics.recordings:
                self.audio.record.linked_biomechanics = True
                self.audio.record.biomechanics_file = str(self.biomechanics_file)
                self.audio.record.biomechanics_type = self.biomechanics_type
                self.audio.record.log_updated = datetime.now()
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
                                        self.audio.record.biomechanics_sample_rate = 1.0 / avg_diff_sec
                        elif hasattr(first_rec, "sample_rate"):
                            self.audio.record.biomechanics_sample_rate = float(first_rec.sample_rate)
                    except Exception as e:
                        logging.debug(f"Could not determine biomechanics sample rate: {e}")
                # Set sync method based on biomechanics type
                if self.biomechanics_type == "Gonio":
                    self.audio.record.biomechanics_sync_method = "flick"
                else:
                    self.audio.record.biomechanics_sync_method = "stomp"

            # Create biomechanics record
            self.biomechanics.record = create_biomechanics_record_from_data(
                biomechanics_file=self.biomechanics_file,
                recordings=self.biomechanics.recordings,
                sheet_name=f"{self.maneuver_key}_data",
                maneuver=self.maneuver_key,
                biomechanics_type=self.biomechanics_type,
            )

            return True
        except Exception as e:
            logging.error(f"Sync stage failed for {self.knee_side} {self.maneuver_key}: {e}")
            return False

    def process_cycles_stage(self) -> bool:
        """Run movement cycle QC on all synced files."""
        if not self.synced_data:
            logging.warning(f"No synced data to run cycles on for {self.knee_side} {self.maneuver_key}")
            return True  # Not a failure

        try:
            logging.info(f"Processing {self.knee_side} {self.maneuver_key} cycles stage")

            # TODO: Implement cycle QC when import is available
            # For now, this is a placeholder that just creates empty records

            return True
        except Exception as e:
            logging.error(f"Cycles stage failed for {self.knee_side} {self.maneuver_key}: {e}")
            return False

    def save_logs(self) -> bool:
        """Update and save processing logs."""
        try:
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

            # Save to Excel
            self.log.save_to_excel()
            return True
        except Exception as e:
            logging.error(f"Failed to save logs: {e}")
            return False

    def _find_audio_pickle(self) -> Optional[Path]:
        """Find the audio pickle file."""
        # Look in outputs subdirectories first
        for pkl in self.maneuver_dir.glob("*_outputs/*_with_freq.pkl"):
            return pkl
        # Fall back to direct search in maneuver dir
        for pkl in self.maneuver_dir.glob("*_with_freq.pkl"):
            return pkl
        return None

    def _find_bin_file(self) -> Optional[Path]:
        """Find the .bin audio file."""
        for bin_file in self.maneuver_dir.glob("*.bin"):
            return bin_file
        return None

    def _load_audio_metadata(self, pkl_path: Path) -> Optional[dict]:
        """Load audio metadata from JSON if available."""
        meta_path = pkl_path.parent / f"{pkl_path.stem.replace('_with_freq', '')}_meta.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load audio metadata: {e}")
        return None

    def _load_biomechanics(self) -> BiomechanicsData:
        """Load biomechanics recordings."""
        recordings = []
        try:
            if self.maneuver_key == "walk":
                for speed in ["slow", "medium", "fast"]:
                    speed_recordings = import_biomechanics_recordings(
                        biomechanics_file=self.biomechanics_file,
                        maneuver=self.maneuver_key,
                        speed=speed,
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

    def _sync_recording(self, recording) -> Optional[SyncData]:
        """Synchronize a single recording (placeholder)."""
        try:
            # TODO: Implement actual sync logic when needed
            # For now return a placeholder SyncData
            output_path = self._generate_sync_output_path(recording)

            return SyncData(
                output_path=output_path,
                df=self.audio.df.copy() if self.audio.df is not None else pd.DataFrame(),
                stomp_times=(0.0, 0.0, 0.0, {}),
            )
        except Exception as e:
            logging.warning(f"Failed to sync recording: {e}")
            return None

    def _generate_sync_output_path(self, recording) -> Path:
        """Generate output path for synchronized data."""
        synced_dir = self.maneuver_dir / "synced"
        synced_dir.mkdir(exist_ok=True)

        # Generate filename based on recording metadata
        speed = getattr(recording, "speed", None)
        pass_num = getattr(recording, "pass_number", None)

        if speed and pass_num:
            filename = f"{self.knee_side}_{self.maneuver_key}_Pass{pass_num:04d}_{speed}.pkl"
        else:
            filename = f"{self.knee_side}_{self.maneuver_key}.pkl"

        return synced_dir / filename


class KneeProcessor:
    """Processes all maneuvers for a single knee."""

    def __init__(
        self,
        knee_dir: Path,
        knee_side: Literal["Left", "Right"],
        study_id: str,
        biomechanics_file: Path,
        biomechanics_type: Optional[str] = None,
    ):
        self.knee_dir = knee_dir
        self.knee_side = knee_side
        self.study_id = study_id
        self.biomechanics_file = biomechanics_file
        self.biomechanics_type = biomechanics_type

        self.maneuver_processors: dict[str, ManeuverProcessor] = {}
        self.knee_log: Optional[KneeProcessingLog] = None

    def process(
        self,
        entrypoint: Literal["bin", "sync", "cycles"] = "sync",
        maneuver: Optional[str] = None,
    ) -> bool:
        """Process all maneuvers for this knee."""
        try:
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
                )

                if not self._run_processor(processor, entrypoint):
                    return False

                self.maneuver_processors[maneuver_key] = processor

            return self._save_knee_log()
        except Exception as e:
            logging.error(f"Knee processing failed: {e}")
            return False

    def _run_processor(self, proc: ManeuverProcessor, entrypoint: str) -> bool:
        """Run stages from entrypoint onwards."""
        stages_map = {
            "bin": [proc.process_bin_stage, proc.process_sync_stage, proc.process_cycles_stage],
            "sync": [proc.process_sync_stage, proc.process_cycles_stage],
            "cycles": [proc.process_cycles_stage],
        }

        for stage_func in stages_map[entrypoint]:
            if not stage_func():
                return False

        return proc.save_logs()

    def _find_maneuver_dir(self, maneuver_key: str) -> Optional[Path]:
        """Find the directory for a maneuver."""
        maneuver_map = {
            "walk": "Walking",
            "sit_to_stand": "Sit-Stand",
            "flexion_extension": "Flexion-Extension",
        }
        maneuver_dir = self.knee_dir / maneuver_map.get(maneuver_key, maneuver_key)
        return maneuver_dir if maneuver_dir.exists() else None

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
                    self.knee_log.add_maneuver_log(proc.log)

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
        biomechanics_type: Optional[str] = None,
    ):
        self.participant_dir = participant_dir
        self.study_id = participant_dir.name.lstrip("#")
        self.biomechanics_type = biomechanics_type
        self.biomechanics_file = self._find_biomechanics_file()

        self.knee_processors: dict[str, KneeProcessor] = {}

    def process(
        self,
        entrypoint: Literal["bin", "sync", "cycles"] = "sync",
        knee: Optional[str] = None,
        maneuver: Optional[str] = None,
    ) -> bool:
        """Process participant."""
        try:
            logging.info(f"Processing participant {self.study_id}")

            knees_to_process = [knee] if knee else ["Left", "Right"]

            for knee_side in knees_to_process:
                knee_dir = self.participant_dir / f"{knee_side} Knee"
                if not knee_dir.exists():
                    logging.warning(f"{knee_side} Knee directory not found")
                    continue

                processor = KneeProcessor(
                    knee_dir=knee_dir,
                    knee_side=cast(Literal["Left", "Right"], knee_side),
                    study_id=self.study_id,
                    biomechanics_file=self.biomechanics_file,
                    biomechanics_type=self.biomechanics_type,
                )

                if not processor.process(entrypoint, maneuver):
                    return False

                self.knee_processors[knee_side] = processor

            return True
        except Exception as e:
            logging.error(f"Participant processing failed: {e}")
            return False

    def _find_biomechanics_file(self) -> Path:
        """Find the biomechanics Excel file."""
        motion_capture_dir = self.participant_dir / "Motion Capture"
        if not motion_capture_dir.exists():
            raise FileNotFoundError(f"Motion Capture directory not found in {self.participant_dir}")

        excel_files = list(motion_capture_dir.glob(f"AOA{self.study_id}_Biomechanics_Full_Set.xlsx"))
        if excel_files:
            return excel_files[0]

        raise FileNotFoundError(f"Biomechanics Excel file not found in {motion_capture_dir}")
