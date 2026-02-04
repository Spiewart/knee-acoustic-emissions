"""QC module for synchronized audio and biomechanics data.

Performs two-stage QC on synchronized recordings:
1. Identifies movement cycles with insufficient acoustic signal
2. Compares clean cycles to expected acoustic-biomechanics relationships
"""

import ast
import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, NamedTuple, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from src.biomechanics.cycle_parsing import extract_movement_cycles
from src.biomechanics.quality_control import (
    MIN_VALID_DATA_FRACTION,
    get_default_min_rom,
    validate_knee_angle_waveform,
)
from src.models import FullMovementCycleMetadata, MicrophonePosition

logger = logging.getLogger(__name__)


class _CycleResult(NamedTuple):
    cycle: pd.DataFrame
    index: int
    energy: float
    qc_pass: bool
    audio_qc_pass: bool = True  # Overall audio QC pass/fail (any mic)
    audio_qc_mic_1_pass: bool = True  # Per-mic audio QC results
    audio_qc_mic_2_pass: bool = True
    audio_qc_mic_3_pass: bool = True
    audio_qc_mic_4_pass: bool = True
    periodic_noise_detected: bool = False  # Overall periodic noise detection
    periodic_noise_ch1: bool = False  # Per-channel periodic noise
    periodic_noise_ch2: bool = False
    periodic_noise_ch3: bool = False
    periodic_noise_ch4: bool = False
    sync_quality_score: float = 0.0  # Cross-modal sync quality score
    sync_qc_pass: bool = True  # Cross-modal sync QC pass/fail


def _find_participant_dir(path: Path) -> Optional[Path]:
    for candidate in (path, *path.parents):
        if candidate.name.startswith("#"):
            return candidate
    return None


def _extract_study_id_from_path(path: Path) -> int:
    participant_dir = _find_participant_dir(path)
    if participant_dir is None:
        return 0
    try:
        return int(participant_dir.name.lstrip("#"))
    except ValueError:
        return 0


def _infer_knee_from_path(path: Path) -> Optional[str]:
    for part in path.parts:
        normalized = part.lower().replace(" ", "")
        if normalized == "leftknee":
            return "left"
        if normalized == "rightknee":
            return "right"
    return None


def _parse_pass_number(file_stem: str) -> Optional[int]:
    match = re.search(r"pass(\d+)", file_stem, re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def _find_acoustics_bin(maneuver_dir: Path) -> Optional[Path]:
    bin_files = sorted(maneuver_dir.glob("*.bin"))
    if bin_files:
        return bin_files[0]
    return None


def _parse_acoustics_filename(file_name: str) -> tuple[str, int, datetime]:
    stem = Path(file_name).stem
    serial = "unknown"
    firmware_version = 0
    recording_date = datetime.min

    serial_match = re.search(r"(?:sn|serial)[-_]?([0-9]+)", stem, re.IGNORECASE)
    if serial_match:
        serial = serial_match.group(1)

    firmware_match = re.search(r"(?:fw|firmware)[-_]?([0-9]+)", stem, re.IGNORECASE)
    if firmware_match:
        try:
            firmware_version = int(firmware_match.group(1))
        except ValueError:
            firmware_version = 0

    date_match = re.search(r"(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)", stem)
    if date_match:
        try:
            recording_date = datetime(
                int(date_match.group(1)),
                int(date_match.group(2)),
                int(date_match.group(3)),
            )
        except ValueError:
            recording_date = datetime.min

    return serial, firmware_version, recording_date


def _default_microphones() -> dict[int, MicrophonePosition]:
    return {
        1: MicrophonePosition(patellar_position="Infrapatellar", laterality="Lateral"),
        2: MicrophonePosition(patellar_position="Infrapatellar", laterality="Medial"),
        3: MicrophonePosition(patellar_position="Suprapatellar", laterality="Medial"),
        4: MicrophonePosition(patellar_position="Suprapatellar", laterality="Lateral"),
    }


def _load_microphone_metadata(
    participant_dir: Optional[Path],
    maneuver: str,
    knee: str,
) -> tuple[dict[int, MicrophonePosition], Optional[dict[int, str]]]:
    legend_candidates: list[Path] = []
    if participant_dir:
        legend_candidates.extend(sorted(participant_dir.glob("*acoustic_file_legend*.xlsx")))
        legend_candidates.extend(sorted(participant_dir.glob("*acoustic_file_legend*.xlsm")))

    if legend_candidates:
        legend_path = legend_candidates[0]
        try:
            # Local import to avoid circular dependency
            from src.audio.parsers import get_acoustics_metadata

            meta = get_acoustics_metadata(
                metadata_file_path=str(legend_path),
                scripted_maneuver=maneuver,
                knee=knee,
            )
            return meta.microphones, meta.microphone_notes
        except Exception as exc:  # pragma: no cover - best-effort fallback
            logger.debug("Failed to read acoustic legend %s: %s", legend_path, exc)

    return _default_microphones(), None


def _find_biomech_file_name(
    participant_dir: Optional[Path],
    study_id: int,
) -> str:
    default_name = f"AOA{study_id}_Biomechanics_Full_Set.xlsx"
    if participant_dir:
        biomechanics_dir = participant_dir / "Biomechanics"
        for ext in (".xlsx", ".xlsm"):
            candidate = biomechanics_dir / f"AOA{study_id}_Biomechanics_Full_Set{ext}"
            if candidate.exists():
                return candidate.name
    return default_name


def _build_cycle_metadata_context(
    synced_pkl_path: Path,
    maneuver: str,
    speed: Optional[str],
) -> dict[str, Any]:
    participant_dir = _find_participant_dir(synced_pkl_path)
    study_id = _extract_study_id_from_path(synced_pkl_path)
    knee = _infer_knee_from_path(synced_pkl_path) or "left"
    maneuver_dir = synced_pkl_path.parent.parent
    bin_file = _find_acoustics_bin(maneuver_dir)
    audio_file_name = bin_file.name if bin_file else f"{synced_pkl_path.stem}.bin"
    serial, firmware_version, recording_date = _parse_acoustics_filename(audio_file_name)
    microphones, microphone_notes = _load_microphone_metadata(participant_dir, maneuver, knee)
    biomech_file_name = _find_biomech_file_name(participant_dir, study_id)
    pass_number = _parse_pass_number(synced_pkl_path.stem) if maneuver == "walk" else None
    effective_speed = speed or ("medium" if maneuver == "walk" else None)
    if maneuver == "walk" and pass_number is None:
        pass_number = 0

    # Load sync times and audio QC results from processing log
    audio_sync_time = timedelta(0)
    biomech_sync_left_time = timedelta(0)
    biomech_sync_right_time = timedelta(0)
    audio_qc_bad_intervals = []  # Bad intervals from raw audio QC (any mic)
    audio_qc_bad_intervals_per_mic = {}  # Per-mic bad intervals

    if participant_dir:
        try:
            from src.orchestration.processing_log import ManeuverProcessingLog

            knee_label = "Left" if knee == "left" else "Right"
            maneuver_map = {
                "walk": "walk",
                "sit_to_stand": "sit_to_stand",
                "flexion_extension": "flexion_extension",
            }
            maneuver_key = maneuver_map.get(maneuver, maneuver)

            # TODO: Query database directly for audio QC intervals
            # For now, skip loading Excel logs
            # log_path = maneuver_dir / f"processing_log_{study_id}_{knee_label}_{maneuver_key}.xlsx"
            log = None

            if log:
                # Load audio QC bad intervals (all mics)
                if log.audio_record and log.audio_record.QC_not_passed:
                    try:
                        # Parse the string representation of the list of tuples
                        audio_qc_bad_intervals = ast.literal_eval(log.audio_record.QC_not_passed)
                    except Exception as exc:
                        logger.debug("Failed to parse QC_not_passed: %s", exc)

                # Load per-mic audio QC bad intervals
                audio_qc_bad_intervals_per_mic = {}
                if log.audio_record:
                    try:
                        for mic_num in [1, 2, 3, 4]:
                            field_name = f"QC_not_passed_mic_{mic_num}"
                            field_value = getattr(log.audio_record, field_name, None)
                            if field_value:
                                try:
                                    intervals = ast.literal_eval(field_value)
                                    audio_qc_bad_intervals_per_mic[mic_num] = intervals
                                except Exception as exc:
                                    # Malformed data in log, skip this mic
                                    logger.debug(
                                        "Failed to parse %s for mic %d: %s",
                                        field_name,
                                        mic_num,
                                        exc,
                                    )
                    except Exception as exc:
                        logger.debug("Failed to parse per-mic QC_not_passed: %s", exc)

                # Load sync times
                sync_file_stem = synced_pkl_path.stem
                for rec in log.synchronization_records:
                    if rec.sync_file_name == sync_file_stem:
                        # audio_stomp_time is time from start of audio to sync event
                        if rec.audio_stomp_time is not None:
                            audio_sync_time = timedelta(seconds=rec.audio_stomp_time)
                        # bio stomp times are times from start of biomechanics to sync event
                        if rec.bio_left_stomp_time is not None:
                            biomech_sync_left_time = timedelta(seconds=rec.bio_left_stomp_time)
                        if rec.bio_right_stomp_time is not None:
                            biomech_sync_right_time = timedelta(seconds=rec.bio_right_stomp_time)
                        break
        except Exception as exc:  # pragma: no cover - best-effort fallback
            logger.debug("Failed to load sync times from processing log: %s", exc)

    return {
        "study": "AOA",
        "study_id": study_id,
        "knee": knee,
        "maneuver": maneuver,
        "speed": effective_speed,
        "pass_number": pass_number,
        "audio_file_name": audio_file_name,
        "audio_serial_number": serial,
        "audio_firmware_version": firmware_version,
        "date_of_recording": recording_date,
        "microphones": microphones,
        "microphone_notes": microphone_notes,
        "biomech_file_name": biomech_file_name,
        "audio_sync_time": audio_sync_time,
        "biomech_sync_left_time": biomech_sync_left_time,
        "biomech_sync_right_time": biomech_sync_right_time,
        "audio_qc_bad_intervals": audio_qc_bad_intervals,
        "audio_qc_bad_intervals_per_mic": audio_qc_bad_intervals_per_mic,
    }


def _build_cycle_metadata_for_cycle(
    result: _CycleResult,
    context: dict[str, Any],
) -> FullMovementCycleMetadata:
    return FullMovementCycleMetadata(
        id=result.index,
        cycle_index=result.index,
        cycle_acoustic_energy=float(result.energy),
        cycle_qc_pass=result.qc_pass,
        audio_qc_pass=result.audio_qc_pass,
        audio_qc_mic_1_pass=result.audio_qc_mic_1_pass,
        audio_qc_mic_2_pass=result.audio_qc_mic_2_pass,
        audio_qc_mic_3_pass=result.audio_qc_mic_3_pass,
        audio_qc_mic_4_pass=result.audio_qc_mic_4_pass,
        periodic_noise_detected=result.periodic_noise_detected,
        periodic_noise_ch1=result.periodic_noise_ch1,
        periodic_noise_ch2=result.periodic_noise_ch2,
        periodic_noise_ch3=result.periodic_noise_ch3,
        periodic_noise_ch4=result.periodic_noise_ch4,
        sync_quality_score=result.sync_quality_score,
        sync_qc_pass=result.sync_qc_pass,
        scripted_maneuver=context["maneuver"],
        speed=context["speed"],
        pass_number=context["pass_number"],
        study=context["study"],
        study_id=context["study_id"],
        knee=context["knee"],
        audio_file_name=context["audio_file_name"],
        audio_serial_number=context["audio_serial_number"],
        audio_firmware_version=context["audio_firmware_version"],
        date_of_recording=context["date_of_recording"],
        microphones=context["microphones"],
        microphone_notes=context["microphone_notes"],
        audio_sync_time=context["audio_sync_time"],
        biomech_file_name=context["biomech_file_name"],
        biomech_sync_left_time=context["biomech_sync_left_time"],
        biomech_sync_right_time=context["biomech_sync_right_time"],
    )


def _format_timedelta_readable(td: timedelta) -> str:
    """Format a timedelta as a human-readable string.

    Args:
        td: timedelta to format

    Returns:
        String like "16.64 s" or "0.00 s"
    """
    total_seconds = td.total_seconds()
    return f"{total_seconds:.3f} s"


def _write_cycle_metadata_json(path: Path, metadata: FullMovementCycleMetadata) -> None:
    """Write cycle metadata to JSON file with readable timedelta formatting.

    Args:
        path: Path to write JSON to
        metadata: FullMovementCycleMetadata to serialize
    """
    try:
        payload = metadata.model_dump(mode="json")

        # Format timedeltas as readable strings
        for key in ("audio_sync_time", "biomech_sync_left_time", "biomech_sync_right_time"):
            if key in payload and payload[key] is not None:
                # payload[key] is ISO 8601 duration string from model_dump
                # Parse it back to timedelta to get the value
                iso_str = payload[key]
                if isinstance(iso_str, str):
                    # ISO 8601 duration format like "PT16.636824S"
                    # Simple parsing for seconds format
                    if iso_str.startswith("PT") and iso_str.endswith("S"):
                        seconds_str = iso_str[2:-1]
                        try:
                            seconds = float(seconds_str)
                            payload[key] = _format_timedelta_readable(timedelta(seconds=seconds))
                        except (ValueError, AttributeError):
                            pass

        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - metadata best effort
        logger.warning("Failed to write metadata JSON %s: %s", path, exc)


class MovementCycleQC:
    """Perform QC analysis on extracted movement cycles."""

    def __init__(
        self,
        maneuver: Literal["walk", "sit_to_stand", "flexion_extension"],
        speed: Optional[Literal["slow", "medium", "fast"]] = None,
        acoustic_threshold: float = 100.0,
        acoustic_channel: Literal["raw", "filtered"] = "filtered",
        biomech_min_rom: Optional[float] = None,
    ):
        """Initialize QC analyzer.

        Args:
            maneuver: Type of movement (walk, sit_to_stand, flexion_extension)
            speed: Speed level for walking (slow, medium, fast)
            acoustic_threshold: Minimum RMS acoustic energy threshold per cycle
            acoustic_channel: Whether to use raw (ch) or filtered (f_ch) channels
            biomech_min_rom: Minimum knee angle range of motion (degrees) for valid biomechanics.
                           If None, uses maneuver-specific defaults:
                           - walk: 20 degrees
                           - sit_to_stand: 40 degrees
                           - flexion_extension: 40 degrees
        """
        self.maneuver = maneuver
        self.speed = speed
        self.acoustic_threshold = acoustic_threshold
        self.acoustic_channel = acoustic_channel

        # Set default biomechanics ROM thresholds based on maneuver
        if biomech_min_rom is None:
            self.biomech_min_rom = get_default_min_rom(maneuver)
        else:
            self.biomech_min_rom = biomech_min_rom

    def analyze_cycles(
        self,
        cycles: list[pd.DataFrame],
    ) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
        """Perform two-stage QC analysis on movement cycles.

        **Stage 1**: Filters cycles by acoustic signal strength using area-under-curve (AUC).
        Cycles with acoustic energy below the threshold are flagged as outliers.

        **Stage 2**: Validates biomechanics data to ensure appropriate knee angle fluctuations.
        Cycles with insufficient range of motion are flagged as outliers.

        Args:
            cycles: List of movement cycle DataFrames from cycle parser.

        Returns:
            Tuple of (clean_cycles, outlier_cycles):
            - clean_cycles: Cycles passing all QC checks.
            - outlier_cycles: Cycles flagged as problematic (low signal, insufficient ROM, etc.).
        """
        if not cycles:
            logger.warning("No cycles provided for QC analysis")
            return [], []

        # Stage 1: Identify cycles with insufficient acoustic signal
        clean_cycles, outliers_low_signal = self._stage1_acoustic_threshold(cycles)

        # Stage 2: Validate biomechanics data (knee angle range of motion)
        clean_cycles_final, outliers_biomech = self._stage2_biomechanics_validation(clean_cycles)

        # Combine all outliers
        outlier_cycles = outliers_low_signal + outliers_biomech

        logger.info(
            f"QC complete: {len(clean_cycles_final)} clean cycles, "
            f"{len(outlier_cycles)} outliers "
            f"(acoustic: {len(outliers_low_signal)}, biomechanics: {len(outliers_biomech)})"
        )

        return clean_cycles_final, outlier_cycles

    def _stage1_acoustic_threshold(
        self,
        cycles: list[pd.DataFrame],
    ) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
        """Stage 1: Filter cycles by acoustic signal strength.

        Uses area-under-curve (AUC) of acoustic energy to identify cycles with
        insufficient signal, typically at the start/end of recording or due to
        microphone artifacts.

        Args:
            cycles: List of cycle DataFrames.

        Returns:
            Tuple of (clean_cycles, outlier_cycles)
        """
        clean_cycles = []
        outlier_cycles = []

        for cycle_idx, cycle_df in enumerate(cycles):
            acoustic_energy = self._compute_cycle_acoustic_energy(cycle_df)

            if acoustic_energy >= self.acoustic_threshold:
                clean_cycles.append(cycle_df)
                logger.debug(f"Cycle {cycle_idx}: CLEAN (energy={acoustic_energy:.1f})")
            else:
                outlier_cycles.append(cycle_df)
                logger.debug(f"Cycle {cycle_idx}: OUTLIER (energy={acoustic_energy:.1f})")

        return clean_cycles, outlier_cycles

    def _compute_cycle_acoustic_energy(self, cycle_df: pd.DataFrame) -> float:
        """Compute total acoustic energy for a cycle using area-under-curve (AUC).

        Integrates the absolute value of acoustic signals across all available
        channels over the cycle duration. Selects filtered channels (f_ch*) if
        available, falling back to raw channels (ch*) if not.

        Args:
            cycle_df: Single movement cycle DataFrame with audio data and time column 'tt'.

        Returns:
            Total acoustic energy (AUC) as a float. Sums AUC across all channels.
        """
        # Select channels based on acoustic_channel preference
        if self.acoustic_channel == "filtered":
            channel_names = ["f_ch1", "f_ch2", "f_ch3", "f_ch4"]
        else:
            channel_names = ["ch1", "ch2", "ch3", "ch4"]

        # Fall back to raw if filtered not available
        if not all(ch in cycle_df.columns for ch in channel_names):
            channel_names = ["ch1", "ch2", "ch3", "ch4"]

        # Compute AUC for each channel
        total_auc = 0.0
        if "tt" in cycle_df.columns:
            # Convert tt to seconds if timedelta
            if isinstance(cycle_df["tt"].iloc[0], pd.Timedelta):
                tt_seconds = cycle_df["tt"].dt.total_seconds().values
            else:
                tt_seconds = cycle_df["tt"].values
        else:
            # Use uniform spacing if tt not available
            tt_seconds = np.arange(len(cycle_df))

        for ch in channel_names:
            if ch in cycle_df.columns:
                ch_data = np.abs(cycle_df[ch].values)
                try:
                    auc = np.trapezoid(ch_data, tt_seconds)
                except AttributeError:
                    auc = np.trapz(ch_data, tt_seconds)
                total_auc += auc

        return total_auc

    def _stage2_biomechanics_validation(
        self,
        cycles: list[pd.DataFrame],
    ) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
        """Stage 2: Validate biomechanics data for appropriate knee angle waveform patterns.

        Validates that each cycle exhibits the stereotypic knee angle fluctuation pattern
        expected for the maneuver type. This includes:
        - Sufficient range of motion (ROM)
        - Presence of expected peaks/troughs at appropriate locations
        - Proper waveform shape characteristics

        Cycles with improper patterns may indicate:
        - Incorrect cycle boundaries
        - Incomplete movements
        - Data quality issues in biomechanics recordings

        Args:
            cycles: List of cycle DataFrames that passed acoustic threshold.

        Returns:
            Tuple of (clean_cycles, outlier_cycles)
        """
        clean_cycles = []
        outlier_cycles = []

        for cycle_idx, cycle_df in enumerate(cycles):
            is_valid, reason = self._validate_knee_angle_waveform(cycle_df)

            if is_valid:
                clean_cycles.append(cycle_df)
                logger.debug(
                    f"Cycle {cycle_idx}: CLEAN biomechanics ({reason})"
                )
            else:
                outlier_cycles.append(cycle_df)
                logger.debug(
                    f"Cycle {cycle_idx}: OUTLIER biomechanics ({reason})"
                )

        return clean_cycles, outlier_cycles

    def _compute_knee_angle_rom(self, cycle_df: pd.DataFrame) -> float:
        """Compute range of motion (ROM) for knee angle in a cycle.

        ROM is calculated as the difference between maximum and minimum knee angle
        values within the cycle, representing the extent of joint movement.

        Args:
            cycle_df: Single movement cycle DataFrame with 'Knee Angle Z' column.

        Returns:
            Range of motion in degrees. Returns 0.0 if 'Knee Angle Z' column is missing
            or if the cycle has insufficient data.
        """
        if "Knee Angle Z" not in cycle_df.columns:
            logger.warning("Cycle missing 'Knee Angle Z' column for ROM computation")
            return 0.0

        if len(cycle_df) < 2:
            logger.warning("Cycle too short for ROM computation")
            return 0.0

        knee_angle = pd.to_numeric(cycle_df["Knee Angle Z"], errors="coerce").values

        # Remove any NaN values before computing ROM
        knee_angle_clean = knee_angle[~np.isnan(knee_angle)]

        if len(knee_angle_clean) == 0:
            logger.warning("Cycle has only NaN values in 'Knee Angle Z'")
            return 0.0

        rom = float(np.max(knee_angle_clean) - np.min(knee_angle_clean))
        return rom

    def _validate_knee_angle_waveform(self, cycle_df: pd.DataFrame) -> tuple[bool, str]:
        """Validate that knee angle waveform matches expected pattern for the maneuver.

        Delegates to biomechanics.quality_control for waveform-level validation.

        Args:
            cycle_df: Single movement cycle DataFrame with 'Knee Angle Z' column.

        Returns:
            Tuple of (is_valid, reason) where is_valid indicates if the cycle passes
            validation and reason provides details about the validation result.
        """
        if "Knee Angle Z" not in cycle_df.columns:
            return False, "missing 'Knee Angle Z' column"

        if len(cycle_df) < 10:
            return False, "insufficient data points"

        knee_angle = pd.to_numeric(cycle_df["Knee Angle Z"], errors="coerce").values

        # Remove any NaN values
        valid_mask = ~np.isnan(knee_angle)
        if not valid_mask.any():
            return False, "all values are NaN"

        total_points = len(knee_angle)
        valid_points = int(valid_mask.sum())

        # If too many NaNs are present, the temporal structure of the cycle
        # is no longer reliable for waveform analysis.
        valid_fraction = valid_points / float(total_points)
        if valid_fraction < MIN_VALID_DATA_FRACTION:
            return False, f"too many NaN values in cycle (valid={valid_fraction:.0%})"

        knee_angle_clean = knee_angle[valid_mask]

        if len(knee_angle_clean) < 10:
            return False, "insufficient valid data points after NaN removal"

        # Delegate to biomechanics quality control module
        return validate_knee_angle_waveform(
            knee_angle_clean,
            self.maneuver,
            min_rom=self.biomech_min_rom,
        )


def perform_sync_qc(
    synced_pkl_path: Path,
    output_dir: Optional[Path] = None,
    maneuver: Optional[Literal["walk", "sit_to_stand", "flexion_extension"]] = None,
    speed: Optional[Literal["slow", "medium", "fast"]] = None,
    acoustic_threshold: float = 100.0,
    biomech_min_rom: Optional[float] = None,
    create_plots: bool = True,
    bad_audio_segments: Optional[list[tuple[float, float]]] = None,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], Path]:
    """Perform complete QC pipeline on a synchronized recording.

    Args:
        synced_pkl_path: Path to synchronized pickle file.
        output_dir: Directory to save QC results. Defaults to parent directory.
        maneuver: Type of movement. If None, inferred from file path.
        speed: Speed level for walking. If None, inferred from file path.
        acoustic_threshold: Minimum acoustic energy per cycle.
        biomech_min_rom: Minimum knee angle range of motion (degrees) for valid biomechanics.
                        If None, uses maneuver-specific defaults.
        create_plots: Whether to create visualization plots.
        bad_audio_segments: Optional list of (start_time, end_time) tuples in audio coordinates
                           indicating bad audio segments from raw audio QC. If None, attempts
                           to load from processing log. Useful for providing segments from
                           alternative sources (e.g., database) in the future.

    Returns:
        Tuple of (clean_cycles, outlier_cycles, output_directory)

    Raises:
        FileNotFoundError: If synced_pkl_path does not exist.
        ValueError: If maneuver cannot be determined.
    """
    synced_pkl_path = Path(synced_pkl_path)
    if not synced_pkl_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {synced_pkl_path}")

    # Load synchronized data
    synced_df = pd.read_pickle(synced_pkl_path)
    logger.info(f"Loaded synchronized data: {synced_df.shape}")

    # Infer maneuver and speed if not provided
    if maneuver is None:
        maneuver = _infer_maneuver_from_path(synced_pkl_path)
    if speed is None and maneuver == "walk":
        speed = _infer_speed_from_path(synced_pkl_path)

    logger.info(f"Maneuver: {maneuver}, Speed: {speed}")

    # Extract movement cycles
    try:
        cycles = extract_movement_cycles(synced_df, maneuver=maneuver, speed=speed)
    except ValueError as exc:
        logger.warning("Skipping cycle extraction: %s", exc)
        return [], [], output_dir or synced_pkl_path.parent
    logger.info(f"Extracted {len(cycles)} movement cycles")

    if not cycles:
        logger.warning("No movement cycles extracted")
        return [], [], output_dir or synced_pkl_path.parent

    # Perform QC analysis
    qc = MovementCycleQC(
        maneuver=maneuver,
        speed=speed,
        acoustic_threshold=acoustic_threshold,
        biomech_min_rom=biomech_min_rom,
    )
    clean_cycles, outlier_cycles = qc.analyze_cycles(cycles)

    # Compute per-cycle metadata (retain original ordering)
    clean_ids = {id(cycle) for cycle in clean_cycles}
    cycle_results: list[_CycleResult] = []

    # Import cycle QC functions locally to avoid circular imports
    # TODO: Consider refactoring module structure to avoid this pattern
    # (e.g., moving shared types to a separate module)
    from src.audio.cycle_qc import (
        check_cycle_periodic_noise,
        validate_acoustic_waveform,
    )

    for idx, cycle_df in enumerate(cycles):
        energy = qc._compute_cycle_acoustic_energy(cycle_df)

        # Run cycle-level audio QC (periodic noise detection)
        periodic_noise_results = check_cycle_periodic_noise(cycle_df)
        periodic_noise_detected = any(periodic_noise_results.values())

        # Run waveform-based sync quality validation (replaces phase-based)
        waveform_valid, validation_reason = validate_acoustic_waveform(
            cycle_df,
            maneuver=maneuver
        )
        # Convert waveform validation to sync quality score (1.0 if valid, 0.0 if not)
        sync_quality_score = 1.0 if waveform_valid else 0.0
        sync_qc_pass = waveform_valid

        cycle_results.append(
            _CycleResult(
                cycle=cycle_df,
                index=idx,
                energy=energy,
                qc_pass=id(cycle_df) in clean_ids,
                audio_qc_pass=True,  # Will be updated if raw audio QC finds issues
                audio_qc_mic_1_pass=True,  # Per-mic QC, will be updated
                audio_qc_mic_2_pass=True,
                audio_qc_mic_3_pass=True,
                audio_qc_mic_4_pass=True,
                periodic_noise_detected=periodic_noise_detected,
                periodic_noise_ch1=periodic_noise_results.get('ch1', False),
                periodic_noise_ch2=periodic_noise_results.get('ch2', False),
                periodic_noise_ch3=periodic_noise_results.get('ch3', False),
                periodic_noise_ch4=periodic_noise_results.get('ch4', False),
                sync_quality_score=sync_quality_score,
                sync_qc_pass=sync_qc_pass,
            )
        )

    metadata_context = _build_cycle_metadata_context(
        synced_pkl_path=synced_pkl_path,
        maneuver=maneuver,
        speed=speed,
    )

    # Check cycles against raw audio QC bad intervals
    # Use provided bad_audio_segments or load from processing log
    if bad_audio_segments is not None:
        # Use provided bad segments
        audio_qc_bad_intervals = bad_audio_segments
        logger.debug("Using provided bad_audio_segments for QC checks")
    else:
        # Fall back to loading from processing log (default behavior)
        audio_qc_bad_intervals = metadata_context.get("audio_qc_bad_intervals", [])
        if audio_qc_bad_intervals:
            logger.debug("Loaded bad audio segments from processing log")

    # Get per-mic bad intervals from metadata context
    audio_qc_bad_intervals_per_mic = metadata_context.get("audio_qc_bad_intervals_per_mic", {})

    # Adjust bad intervals from audio coordinates to synchronized coordinates
    if audio_qc_bad_intervals or audio_qc_bad_intervals_per_mic:
        from src.audio.raw_qc import (
            adjust_bad_intervals_for_sync,
            check_cycle_in_bad_interval,
        )

        audio_sync_time = metadata_context["audio_sync_time"].total_seconds()
        # Use the appropriate biomech stomp time (left or right based on knee)
        knee = metadata_context.get("knee", "left")
        if knee == "left":
            bio_sync_time = metadata_context["biomech_sync_left_time"].total_seconds()
        else:
            bio_sync_time = metadata_context["biomech_sync_right_time"].total_seconds()

        # Adjust overall bad intervals to synchronized coordinates
        synced_bad_intervals = []
        if audio_qc_bad_intervals:
            synced_bad_intervals = adjust_bad_intervals_for_sync(
                audio_qc_bad_intervals,
                audio_sync_time,
                bio_sync_time,
            )

        # Adjust per-mic bad intervals to synchronized coordinates
        synced_bad_intervals_per_mic = {}
        for mic_num, intervals in audio_qc_bad_intervals_per_mic.items():
            if intervals:
                synced_intervals = adjust_bad_intervals_for_sync(
                    intervals,
                    audio_sync_time,
                    bio_sync_time,
                )
                synced_bad_intervals_per_mic[mic_num] = synced_intervals

        if synced_bad_intervals:
            logger.info(
                f"Checking {len(cycles)} cycles against {len(synced_bad_intervals)} "
                f"raw audio QC bad intervals (adjusted for sync)"
            )

        if synced_bad_intervals_per_mic:
            logger.info(
                f"Checking {len(cycles)} cycles against per-microphone bad intervals: "
                f"{', '.join(f'mic {k}: {len(v)} intervals' for k, v in synced_bad_intervals_per_mic.items())}"
            )

        # Check each cycle and update results
        updated_results = []
        for result in cycle_results:
            cycle_df = result.cycle
            audio_qc_pass = True  # Default to pass (overall)
            audio_qc_mic_1_pass = True  # Per-mic defaults
            audio_qc_mic_2_pass = True
            audio_qc_mic_3_pass = True
            audio_qc_mic_4_pass = True

            if "tt" in cycle_df.columns:
                # Get cycle time bounds
                if isinstance(cycle_df["tt"].iloc[0], pd.Timedelta):
                    cycle_start = cycle_df["tt"].iloc[0].total_seconds()
                    cycle_end = cycle_df["tt"].iloc[-1].total_seconds()
                else:
                    cycle_start = float(cycle_df["tt"].iloc[0])
                    cycle_end = float(cycle_df["tt"].iloc[-1])

                # Check if cycle overlaps with overall bad intervals
                if synced_bad_intervals:
                    fails_audio_qc = check_cycle_in_bad_interval(
                        cycle_start,
                        cycle_end,
                        synced_bad_intervals,
                        overlap_threshold=0.1,  # 10% overlap threshold
                    )

                    if fails_audio_qc:
                        logger.debug(
                            f"Cycle {result.index} failed overall audio QC "
                            f"(time range: {cycle_start:.2f}-{cycle_end:.2f}s)"
                        )
                        audio_qc_pass = False

                # Check if cycle overlaps with per-mic bad intervals
                for mic_num, mic_bad_intervals in synced_bad_intervals_per_mic.items():
                    if mic_bad_intervals:
                        fails_mic_qc = check_cycle_in_bad_interval(
                            cycle_start,
                            cycle_end,
                            mic_bad_intervals,
                            overlap_threshold=0.1,  # 10% overlap threshold
                        )

                        if fails_mic_qc:
                            logger.debug(
                                f"Cycle {result.index} failed mic {mic_num} audio QC "
                                f"(time range: {cycle_start:.2f}-{cycle_end:.2f}s)"
                            )
                            # Update the corresponding per-mic pass/fail
                            if mic_num == 1:
                                audio_qc_mic_1_pass = False
                            elif mic_num == 2:
                                audio_qc_mic_2_pass = False
                            elif mic_num == 3:
                                audio_qc_mic_3_pass = False
                            elif mic_num == 4:
                                audio_qc_mic_4_pass = False

            # Create updated result with audio_qc_pass and per-mic results set
            updated_results.append(
                _CycleResult(
                    cycle=result.cycle,
                    index=result.index,
                    energy=result.energy,
                    qc_pass=result.qc_pass and audio_qc_pass,  # Fail overall QC if audio QC fails
                    audio_qc_pass=audio_qc_pass,
                    audio_qc_mic_1_pass=audio_qc_mic_1_pass,
                    audio_qc_mic_2_pass=audio_qc_mic_2_pass,
                    audio_qc_mic_3_pass=audio_qc_mic_3_pass,
                    audio_qc_mic_4_pass=audio_qc_mic_4_pass,
                )
            )

        cycle_results = updated_results

    # Update clean/outlier lists based on final QC results
    final_clean_cycles = [r.cycle for r in cycle_results if r.qc_pass]
    final_outlier_cycles = [r.cycle for r in cycle_results if not r.qc_pass]

    logger.info(
        f"Final QC: {len(final_clean_cycles)} clean cycles, "
        f"{len(final_outlier_cycles)} outliers (after raw audio QC check)"
    )

    # Set output directory (flat structure: MovementCycles/clean|outliers)
    base_dir = Path(output_dir) if output_dir is not None else synced_pkl_path.parent
    output_dir = base_dir / "MovementCycles"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results and metadata
    _save_qc_results(
        final_clean_cycles,
        final_outlier_cycles,
        output_dir,
        synced_pkl_path.stem,
        create_plots=create_plots,
        cycle_results=cycle_results,
        metadata_context=metadata_context,
    )

    logger.info(f"QC results saved to {output_dir}")

    return final_clean_cycles, final_outlier_cycles, output_dir


def _infer_maneuver_from_path(path: Path) -> str:
    """Infer maneuver type from file path.

    Args:
        path: Path to synchronized pickle file.

    Returns:
        Maneuver type (walk, sit_to_stand, flexion_extension)

    Raises:
        ValueError: If maneuver cannot be inferred.
    """
    path_str = str(path).lower()

    if "walk" in path_str:
        return "walk"
    elif "sit" in path_str and "stand" in path_str:
        return "sit_to_stand"
    elif "flexion" in path_str or "extension" in path_str or "flexext" in path_str:
        return "flexion_extension"
    else:
        raise ValueError(f"Cannot infer maneuver from path: {path}")


def _infer_speed_from_path(path: Path) -> Optional[str]:
    """Infer walking speed from file path.

    Args:
        path: Path to synchronized pickle file.

    Returns:
        Speed level (slow, medium, fast) or None.
    """
    path_str = str(path).lower()

    if "slow" in path_str:
        return "slow"
    elif "medium" in path_str:
        return "medium"
    elif "fast" in path_str:
        return "fast"

    return None


def _infer_pass_number_from_path(path: Path) -> Optional[int]:
    """Infer pass number from file path.

    For walking maneuvers, synced files are named like:
    Left_walk_Pass0001_slow.pkl

    Args:
        path: Path to synchronized pickle file.

    Returns:
        Pass number (1-9999) for walk maneuvers, None otherwise.
    """
    import re

    # Look for pattern like "Pass0001", "Pass0002", etc.
    path_str = str(path)
    match = re.search(r'Pass(\d{4})', path_str, re.IGNORECASE)
    if match:
        pass_num_str = match.group(1)
        try:
            return int(pass_num_str)
        except ValueError:
            return None

    return None


def _save_qc_results(
    clean_cycles: list[pd.DataFrame],
    outlier_cycles: list[pd.DataFrame],
    output_dir: Path,
    file_stem: str,
    create_plots: bool = True,
    *,
    cycle_results: Optional[list[_CycleResult]] = None,
    metadata_context: Optional[dict[str, Any]] = None,
) -> None:
    """Save QC results to disk.

    Args:
        clean_cycles: List of clean cycle DataFrames.
        outlier_cycles: List of outlier cycle DataFrames.
        output_dir: Directory to save results.
        file_stem: Stem of original file (for naming).
        create_plots: Whether to create visualization plots.
        cycle_results: Optional detailed cycle results including metadata.
        metadata_context: Optional context for building metadata JSON files.
    """
    # Create subdirectories
    clean_dir = output_dir / "clean"
    outlier_dir = output_dir / "outliers"
    clean_dir.mkdir(parents=True, exist_ok=True)
    outlier_dir.mkdir(parents=True, exist_ok=True)

    # Fallback to legacy behavior when no metadata provided
    if cycle_results is None:
        for i, cycle_df in enumerate(clean_cycles):
            filename = clean_dir / f"{file_stem}_cycle_{i:03d}.pkl"
            cycle_df.to_pickle(filename)

            if create_plots and MATPLOTLIB_AVAILABLE:
                _create_cycle_plot(cycle_df, clean_dir / f"{file_stem}_cycle_{i:03d}.png")

        for i, cycle_df in enumerate(outlier_cycles):
            filename = outlier_dir / f"{file_stem}_outlier_{i:03d}.pkl"
            cycle_df.to_pickle(filename)

            if create_plots and MATPLOTLIB_AVAILABLE:
                _create_cycle_plot(
                    cycle_df,
                    outlier_dir / f"{file_stem}_outlier_{i:03d}.png",
                    title_suffix="(OUTLIER)",
                )

        logger.info(f"Saved {len(clean_cycles)} clean cycles and {len(outlier_cycles)} outliers")
        return

    clean_counter = 0
    outlier_counter = 0

    for result in cycle_results:
        is_clean = result.qc_pass
        suffix = "cycle" if is_clean else "outlier"
        group_index = clean_counter if is_clean else outlier_counter

        if is_clean:
            clean_counter += 1
            target_dir = clean_dir
        else:
            outlier_counter += 1
            target_dir = outlier_dir

        filename = target_dir / f"{file_stem}_{suffix}_{group_index:03d}.pkl"
        result.cycle.to_pickle(filename)

        if create_plots and MATPLOTLIB_AVAILABLE:
            _create_cycle_plot(
                result.cycle,
                target_dir / f"{file_stem}_{suffix}_{group_index:03d}.png",
                title_suffix="(OUTLIER)" if not is_clean else "",
            )

        if metadata_context is not None:
            try:
                metadata = _build_cycle_metadata_for_cycle(result, metadata_context)
                _write_cycle_metadata_json(filename.with_suffix(".json"), metadata)
            except Exception as exc:  # pragma: no cover - metadata best effort
                logger.warning("Failed to create metadata for %s: %s", filename, exc)

    logger.info(f"Saved {clean_counter} clean cycles and {outlier_counter} outliers")


def _create_cycle_plot(
    cycle_df: pd.DataFrame,
    output_path: Path,
    title_suffix: str = "",
) -> None:
    """Create visualization plot for a single cycle.

    Args:
        cycle_df: Movement cycle DataFrame.
        output_path: Path to save PNG file.
        title_suffix: Additional suffix for title.
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Get time axis
        if "tt" in cycle_df.columns:
            if isinstance(cycle_df["tt"].iloc[0], pd.Timedelta):
                tt_seconds = cycle_df["tt"].dt.total_seconds().values
            else:
                tt_seconds = cycle_df["tt"].values
        else:
            tt_seconds = np.arange(len(cycle_df))

        # Plot 1: Knee angle
        if "Knee Angle Z" in cycle_df.columns:
            ax1.plot(tt_seconds, cycle_df["Knee Angle Z"], "k-", linewidth=2)
            ax1.set_ylabel("Knee Angle Z (degrees)", fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.set_title(f"Movement Cycle - Biomechanics {title_suffix}", fontsize=12)

        # Plot 2: Acoustic energy
        ax2.twinx()

        # Plot acoustic channels
        acoustic_channels = ["f_ch1", "f_ch2", "f_ch3", "f_ch4"]
        colors = ["b", "g", "r", "m"]
        for ch, color in zip(acoustic_channels, colors):
            if ch in cycle_df.columns:
                ax2.plot(tt_seconds, cycle_df[ch], color=color, alpha=0.6, label=ch, linewidth=0.8)

        ax2.set_xlabel("Time (seconds)", fontsize=12)
        ax2.set_ylabel("Acoustic Amplitude", fontsize=12, color="blue")
        ax2.tick_params(axis="y", labelcolor="blue")
        ax2.grid(True, alpha=0.3)
        ax2.set_title(f"Acoustic Channels {title_suffix}", fontsize=12)
        handles, labels = ax2.get_legend_handles_labels()
        if handles:
            ax2.legend(loc="upper left", fontsize=8)

        plt.tight_layout()
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        logger.debug(f"Created plot: {output_path}")

    except Exception as e:
        logger.warning(f"Failed to create plot {output_path}: {e}")


# CLI Support Functions


def setup_logging(verbose: bool = False) -> None:
    """Configure logging output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def find_synced_files(path: Path) -> list[Path]:
    """Find all pickle files located in any directory named 'Synced'.

    Args:
        path: Root path to search.

    Returns:
        List of paths to synced pickle files.
    """
    synced_files = []

    if path.is_file():
        # If a single file is provided, check if its parent is 'Synced'
        if path.suffix == ".pkl" and path.parent.name == "Synced":
            synced_files.append(path)
    elif path.is_dir():
        # Recursively find all .pkl files and check if their parent is 'Synced'
        for pkl_file in path.rglob("*.pkl"):
            if pkl_file.parent.name == "Synced":
                synced_files.append(pkl_file)

    return sorted(synced_files)


def main() -> int:
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Perform QC analysis on synchronized audio and biomechanics data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "path",
        type=Path,
        help="Path to synced pickle file, Synced directory, or participant directory",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=100.0,
        help="Minimum acoustic energy threshold per cycle (default: 100.0)",
    )

    parser.add_argument(
        "--biomech-min-rom",
        type=float,
        default=None,
        help="Minimum knee angle range of motion (degrees) for valid biomechanics. "
             "If not specified, uses maneuver-specific defaults: "
             "walk=20°, sit_to_stand=40°, flexion_extension=40°",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip creation of visualization plots",
    )

    parser.add_argument(
        "--maneuver",
        type=str,
        choices=["walk", "sit_to_stand", "flexion_extension"],
        help="Override maneuver type inference",
    )

    parser.add_argument(
        "--speed",
        type=str,
        choices=["slow", "medium", "fast"],
        help="Override speed inference (for walking)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)

    # Validate path
    if not args.path.exists():
        logger.error(f"Path does not exist: {args.path}")
        return 1

    # Find files to process
    files_to_process = find_synced_files(args.path)

    if not files_to_process:
        logger.warning(f"No synced files found in {args.path}")
        return 1

    logger.info(f"Found {len(files_to_process)} synced file(s) to process")

    # Process each file
    total_clean = 0
    total_outliers = 0

    for synced_file in files_to_process:
        logger.info(f"\nProcessing: {synced_file}")

        try:
            clean_cycles, outlier_cycles, output_dir = perform_sync_qc(
                synced_file,
                maneuver=args.maneuver,
                speed=args.speed,
                acoustic_threshold=args.threshold,
                biomech_min_rom=args.biomech_min_rom,
                create_plots=not args.no_plots,
            )

            total_clean += len(clean_cycles)
            total_outliers += len(outlier_cycles)

            logger.info(
                f"✓ Complete: {len(clean_cycles)} clean, {len(outlier_cycles)} outliers "
                f"→ {output_dir}"
            )

        except Exception as e:
            logger.error(f"✗ Failed to process {synced_file}: {e}", exc_info=args.verbose)
            continue

    logger.info(f"\n{'='*60}")
    logger.info(f"Total: {total_clean} clean cycles, {total_outliers} outliers")
    logger.info(f"{'='*60}")

    return 0


if __name__ == "__main__":
    exit(main())
