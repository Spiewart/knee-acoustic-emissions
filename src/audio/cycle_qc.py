"""Movement cycle-specific audio quality control.

This module will provide specialized QC checks for audio data during synchronized
movement cycles. These checks are performed during sync_qc after cycles have been
identified and synchronized with biomechanics data.

IMPLEMENTATION STATUS: TO BE IMPLEMENTED
This module is currently a placeholder. Implementation will occur during future
sync_qc improvements.

Planned Features
----------------
1. **Periodic Noise Detection During Cycles**: Detect consistent periodic background
   noise (e.g., noisy fan) using spectral analysis (Welch's method) within specific
   movement cycles. This is computationally expensive but more meaningful when
   applied to identified cycles rather than raw continuous audio.

2. **Cycle-Specific Artifact Characterization**: Advanced artifact detection tuned
   to movement patterns, distinguishing between movement-related sounds and true
   artifacts.

3. **Cross-Modal QC**: Check consistency between audio patterns and biomechanics
   data during synchronized cycles.

Usage (Future)
--------------
```python
from src.audio.cycle_qc import check_cycle_periodic_noise

# During sync_qc, check if a specific cycle has periodic background noise
has_periodic_noise = check_cycle_periodic_noise(
    cycle_audio_data,
    fs=sampling_rate,
    threshold=0.3
)

if has_periodic_noise:
    cycle_metadata['periodic_noise_detected'] = True
```

Integration Point
-----------------
This module will be called from `perform_sync_qc()` in
`src/synchronization/quality_control.py` after movement cycles have been extracted
and synchronized with biomechanics data.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


# Reference range constants for acoustic RMS energy by phase
# Units: RMS amplitude (same units as raw audio signal data)
# These are conservative ranges based on expected acoustic patterns
# NOTE: In production, these should be calibrated from a reference dataset
DEFAULT_MIN_RMS_ENERGY = 0.001  # Minimum detectable RMS energy (baseline/quiet)
DEFAULT_MAX_RMS_ENERGY = 10.0   # Maximum expected RMS energy (loud events)
DEFAULT_LOW_ENERGY_MAX = 5.0    # Maximum for low-energy phases (e.g., swing phase in gait)
DEFAULT_MID_ENERGY_MIN = 0.01   # Minimum for moderate-energy phases (e.g., stance phase)
# HIGH_ENERGY_MIN is intentionally lower than MID_ENERGY_MIN to be permissive
# for phases where energy may vary (e.g., flexion/extension transitions)
DEFAULT_VARIABLE_ENERGY_MIN = 0.005  # Minimum for phases with variable acoustic activity


def _detect_periodic_noise_in_cycle(
    data: np.ndarray,
    fs: float,
    threshold: float = 0.3,
) -> bool:
    """Detect periodic background noise in a movement cycle using spectral analysis.
    
    IMPLEMENTATION STATUS: CODE PRESERVED, READY FOR INTEGRATION
    
    This function was moved from raw_qc.py to avoid performance issues during
    bin processing. The code is complete and functional, but not yet called
    from any production workflow. It will be integrated into sync_qc for
    per-cycle analysis in a future enhancement.
    
    Identifies consistent periodic noise (e.g., fan running) by analyzing
    the power spectral density. This is computationally expensive (uses Welch's
    method with 2-second windows) but more meaningful when applied to specific
    movement cycles rather than entire raw recordings.
    
    Args:
        data: Audio signal data for a single cycle
        fs: Sampling frequency (Hz)
        threshold: Threshold for periodic noise detection (0-1)
                  Higher values = less sensitive
    
    Returns:
        True if periodic noise detected in cycle, False otherwise
        
    Notes:
        - Requires scipy.signal.welch
        - Ignores DC and very low frequencies (< 5 Hz)
        - Uses relative power (peak / median) to identify strong periodic components
        - Typical threshold range: 0.1 (sensitive) to 0.5 (conservative)
    """
    from scipy.signal import welch
    
    if len(data) < 256:
        return False
    
    # Compute power spectral density
    try:
        nperseg = min(len(data), int(fs * 2))  # 2 second windows
        freqs, psd = welch(data, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    except Exception:
        return False
    
    if len(psd) == 0:
        return False
    
    # Identify prominent spectral peaks (potential periodic noise)
    # Ignore DC and very low frequencies (< 5 Hz)
    freq_mask = freqs > 5.0
    if not np.any(freq_mask):
        return False
    
    psd_nondc = psd[freq_mask]
    if len(psd_nondc) == 0:
        return False
    
    # Calculate relative power: peak power / median power
    median_power = np.median(psd_nondc)
    if median_power <= 0:
        return False
    
    max_power = np.max(psd_nondc)
    relative_power = max_power / median_power
    
    # If relative power is high, we have a strong periodic component
    # This indicates consistent background noise at a specific frequency
    return bool(relative_power > (1.0 / threshold))


def check_cycle_periodic_noise(
    cycle_df: pd.DataFrame,
    time_col: str = "tt",
    audio_channels: list[str] | None = None,
    threshold: float = 0.3,
) -> dict[str, bool]:
    """Check for periodic background noise in a movement cycle (per channel).
    
    Analyzes audio data from a specific movement cycle to detect periodic
    background noise on a per-channel basis using spectral analysis.
    
    Args:
        cycle_df: DataFrame containing audio data for a single movement cycle
        time_col: Name of time column
        audio_channels: List of channel names to check (default: ch1-ch4)
        threshold: Detection threshold (0-1), higher = less sensitive
        
    Returns:
        Dictionary mapping channel name to boolean (True = periodic noise detected)
        
    Example:
        ```python
        # In perform_sync_qc(), for each cycle:
        periodic_noise = check_cycle_periodic_noise(cycle_data)
        cycle_metadata['periodic_noise_ch1'] = periodic_noise.get('ch1', False)
        cycle_metadata['periodic_noise_ch2'] = periodic_noise.get('ch2', False)
        # ... etc
        ```
    """
    if audio_channels is None:
        audio_channels = ["ch1", "ch2", "ch3", "ch4"]
    
    # Filter to available channels
    available_channels = [ch for ch in audio_channels if ch in cycle_df.columns]
    
    if not available_channels or time_col not in cycle_df.columns:
        return {ch: False for ch in available_channels}
    
    # Compute sampling rate from time column
    try:
        if isinstance(cycle_df[time_col].iloc[0], pd.Timedelta):
            time_s = cycle_df[time_col].dt.total_seconds().values
        else:
            time_s = cycle_df[time_col].values
        
        if len(time_s) < 2:
            return {ch: False for ch in available_channels}
        
        # Calculate sampling rate
        dt = np.median(np.diff(time_s))
        if dt <= 0:
            return {ch: False for ch in available_channels}
        
        fs = 1.0 / dt
    except Exception:
        # If sampling rate calculation fails, return False for all channels
        return {ch: False for ch in available_channels}
    
    # Check each channel for periodic noise
    results = {}
    for ch in available_channels:
        try:
            ch_data = cycle_df[ch].values
            has_noise = _detect_periodic_noise_in_cycle(ch_data, fs, threshold)
            results[ch] = has_noise
        except Exception:
            # If detection fails for a channel, assume no periodic noise
            results[ch] = False
    
    return results


def validate_acoustic_waveform(
    cycle_df: pd.DataFrame,
    maneuver: str,
    time_col: str = "tt",
    audio_channels: list[str] | None = None,
    min_rms_threshold: float = 0.001,
    reference_waveform: np.ndarray | None = None,
    correlation_threshold: float = 0.5,
    dtw_threshold: float | None = None,
) -> tuple[bool, str]:
    """Validate acoustic waveform shape matches expected pattern for the maneuver.
    
    This function performs waveform-level validation of acoustic data, checking that
    the acoustic signal exhibits characteristics consistent with the biomechanical
    movement pattern. Supports two validation modes:
    
    1. **Rule-based validation** (default): Uses predefined characteristics for each
       maneuver type (peaks, timing, variation patterns)
    
    2. **Model-based validation** (optional): Compares against a reference waveform
       using correlation and/or DTW distance, enabling ML-based quality control
    
    Args:
        cycle_df: DataFrame containing synchronized audio and biomechanics data
        maneuver: Type of movement (walk, sit_to_stand, flexion_extension)
        time_col: Name of time column
        audio_channels: List of channel names to check (default: f_ch1-f_ch4)
        min_rms_threshold: Minimum RMS threshold to consider signal present
        reference_waveform: Optional reference/model waveform for comparison.
                           If provided, uses correlation-based validation instead of
                           rule-based validation. Should be normalized RMS envelope.
        correlation_threshold: Minimum correlation coefficient for model-based validation
                              (default: 0.5). Only used if reference_waveform is provided.
        dtw_threshold: Optional DTW (Dynamic Time Warping) distance threshold for
                      time-series similarity. If None, only correlation is used.
                      Lower values indicate more similar waveforms.
        
    Returns:
        Tuple of (is_valid, reason) where is_valid indicates if the waveform passes
        validation and reason provides details about the validation result.
        
    Examples:
        # Rule-based validation (default)
        is_valid, reason = validate_acoustic_waveform(cycle_df, "walk")
        
        # Model-based validation with reference waveform
        reference = get_reference_waveform("walk")  # From ML model or reference dataset
        is_valid, reason = validate_acoustic_waveform(
            cycle_df, 
            "walk",
            reference_waveform=reference,
            correlation_threshold=0.7
        )
    """
    if audio_channels is None:
        audio_channels = ["f_ch1", "f_ch2", "f_ch3", "f_ch4"]
    
    # Filter to available channels
    available_channels = [ch for ch in audio_channels if ch in cycle_df.columns]
    
    if not available_channels:
        return False, "no audio channels available"
    
    if time_col not in cycle_df.columns:
        return False, "missing time column"
    
    # Compute RMS envelope for acoustic signal (average across channels)
    rms_envelope = np.zeros(len(cycle_df))
    for ch in available_channels:
        ch_data = cycle_df[ch].values
        # Use a simple squared signal as RMS-like measure
        rms_envelope += ch_data ** 2
    
    rms_envelope = np.sqrt(rms_envelope / len(available_channels))
    
    # Check for sufficient signal
    if np.max(rms_envelope) < min_rms_threshold:
        return False, f"insufficient acoustic signal (max RMS: {np.max(rms_envelope):.4f})"
    
    # Choose validation method based on whether reference waveform is provided
    if reference_waveform is not None:
        # Model-based validation using reference waveform comparison
        return _validate_against_reference_waveform(
            rms_envelope, 
            reference_waveform,
            correlation_threshold,
            dtw_threshold
        )
    else:
        # Rule-based validation using maneuver-specific patterns
        if maneuver == "walk":
            return _validate_walking_acoustic_waveform(rms_envelope)
        elif maneuver == "sit_to_stand":
            return _validate_sit_to_stand_acoustic_waveform(rms_envelope)
        elif maneuver == "flexion_extension":
            return _validate_flexion_extension_acoustic_waveform(rms_envelope)
        else:
            # Unknown maneuver - fall back to basic signal check
            return True, f"sufficient signal present (max RMS: {np.max(rms_envelope):.4f})"


def _validate_against_reference_waveform(
    observed_waveform: np.ndarray,
    reference_waveform: np.ndarray,
    correlation_threshold: float = 0.5,
    dtw_threshold: float | None = None,
) -> tuple[bool, str]:
    """Validate observed waveform against a reference/model waveform.
    
    Uses correlation and optionally DTW distance to assess similarity between
    the observed acoustic waveform and a reference waveform (e.g., from ML model
    or reference dataset).
    
    Args:
        observed_waveform: RMS envelope of observed acoustic signal
        reference_waveform: Reference/model waveform to compare against
        correlation_threshold: Minimum correlation coefficient required (0-1)
        dtw_threshold: Optional maximum DTW distance allowed. If None, DTW not used.
        
    Returns:
        Tuple of (is_valid, reason)
    """
    if len(observed_waveform) < 10:
        return False, "insufficient data points for waveform comparison"
    
    if len(reference_waveform) < 10:
        return False, "invalid reference waveform (too short)"
    
    # Normalize both waveforms for comparison
    observed_norm = _normalize_waveform(observed_waveform)
    reference_norm = _normalize_waveform(reference_waveform)
    
    # Resample observed to match reference length if needed
    if len(observed_norm) != len(reference_norm):
        observed_norm = _resample_waveform(observed_norm, len(reference_norm))
    
    # Compute correlation coefficient
    try:
        correlation = np.corrcoef(observed_norm, reference_norm)[0, 1]
    except Exception:
        return False, "failed to compute correlation"
    
    # Check correlation threshold
    if np.isnan(correlation):
        return False, "correlation calculation failed (NaN result)"
    
    if correlation < correlation_threshold:
        return False, f"low correlation with reference (r={correlation:.3f} < {correlation_threshold:.3f})"
    
    # Optionally compute DTW distance if threshold provided
    if dtw_threshold is not None:
        try:
            dtw_distance = _compute_dtw_distance(observed_norm, reference_norm)
            
            if dtw_distance > dtw_threshold:
                return False, f"high DTW distance (d={dtw_distance:.3f} > {dtw_threshold:.3f}), correlation={correlation:.3f}"
            
            return True, f"valid waveform (r={correlation:.3f}, DTW={dtw_distance:.3f})"
        except Exception as e:
            # DTW failed but correlation passed - still pass
            return True, f"valid waveform (r={correlation:.3f}, DTW unavailable)"
    
    return True, f"valid waveform (r={correlation:.3f})"


def _normalize_waveform(waveform: np.ndarray) -> np.ndarray:
    """Normalize waveform to zero mean and unit variance.
    
    Args:
        waveform: Input waveform array
        
    Returns:
        Normalized waveform
    """
    mean = np.mean(waveform)
    std = np.std(waveform)
    
    if std == 0:
        # Constant signal - return zeros
        return np.zeros_like(waveform)
    
    return (waveform - mean) / std


def _resample_waveform(waveform: np.ndarray, target_length: int) -> np.ndarray:
    """Resample waveform to target length using linear interpolation.
    
    Args:
        waveform: Input waveform array
        target_length: Desired output length
        
    Returns:
        Resampled waveform
    """
    if len(waveform) == target_length:
        return waveform
    
    # Use linear interpolation to resample
    old_indices = np.linspace(0, len(waveform) - 1, len(waveform))
    new_indices = np.linspace(0, len(waveform) - 1, target_length)
    
    resampled = np.interp(new_indices, old_indices, waveform)
    return resampled


def _compute_dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """Compute Dynamic Time Warping distance between two sequences.
    
    Simplified DTW implementation for time-series similarity measurement.
    Lower distance indicates more similar temporal patterns.
    
    Args:
        seq1: First sequence
        seq2: Second sequence
        
    Returns:
        DTW distance (normalized by sequence length)
    """
    n, m = len(seq1), len(seq2)
    
    # Initialize DTW matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    # Fill DTW matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(seq1[i - 1] - seq2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # insertion
                dtw_matrix[i, j - 1],      # deletion
                dtw_matrix[i - 1, j - 1]   # match
            )
    
    # Return normalized DTW distance
    return dtw_matrix[n, m] / max(n, m)


def _validate_walking_acoustic_waveform(rms_envelope: np.ndarray) -> tuple[bool, str]:
    """Validate acoustic waveform pattern for walking gait cycles.
    
    Walking acoustic patterns should exhibit:
    1. Peak(s) during stance/heel strike phase (typically early-to-mid cycle)
    2. Lower energy during swing phase
    3. Not completely flat (should have variation)
    
    Args:
        rms_envelope: RMS envelope of acoustic signal
        
    Returns:
        Tuple of (is_valid, reason)
    """
    if len(rms_envelope) < 10:
        return False, "insufficient data points for waveform validation"
    
    # Calculate signal variation
    signal_std = np.std(rms_envelope)
    signal_mean = np.mean(rms_envelope)
    
    if signal_mean == 0:
        return False, "zero mean signal"
    
    # Check for variation (coefficient of variation)
    cv = signal_std / signal_mean
    if cv < 0.1:
        return False, f"insufficient waveform variation (CV: {cv:.2f})"
    
    # Find peaks in acoustic signal
    # Use prominence relative to signal range
    signal_range = np.max(rms_envelope) - np.min(rms_envelope)
    prominence = max(signal_range * 0.2, signal_mean * 0.5)
    
    peaks, peak_properties = find_peaks(rms_envelope, prominence=prominence)
    
    if len(peaks) == 0:
        return False, "no acoustic peaks detected"
    
    # Check peak timing - should be in first 70% of cycle (stance phase)
    cycle_length = len(rms_envelope)
    dominant_peak_idx = peaks[np.argmax(rms_envelope[peaks])]
    peak_position = dominant_peak_idx / cycle_length
    
    if peak_position > 0.8:
        return False, f"acoustic peak too late in cycle ({peak_position*100:.0f}%)"
    
    return True, f"valid walking acoustic pattern ({len(peaks)} peak(s), dominant at {peak_position*100:.0f}%)"


def _validate_sit_to_stand_acoustic_waveform(rms_envelope: np.ndarray) -> tuple[bool, str]:
    """Validate acoustic waveform pattern for sit-to-stand maneuvers.
    
    Sit-to-stand acoustic patterns should exhibit:
    1. Increased energy during transition phase (middle of cycle)
    2. Lower energy at beginning (sitting) and end (standing)
    3. General increase then decrease pattern
    
    Args:
        rms_envelope: RMS envelope of acoustic signal
        
    Returns:
        Tuple of (is_valid, reason)
    """
    if len(rms_envelope) < 10:
        return False, "insufficient data points for waveform validation"
    
    # Check signal is present
    signal_mean = np.mean(rms_envelope)
    if signal_mean == 0:
        return False, "zero mean signal"
    
    # Divide into thirds: sitting, transition, standing
    third = len(rms_envelope) // 3
    if third < 2:
        return False, "insufficient data points for phase analysis"
    
    sitting_mean = np.mean(rms_envelope[:third])
    transition_mean = np.mean(rms_envelope[third:2*third])
    standing_mean = np.mean(rms_envelope[2*third:])
    
    # Transition should have higher energy than sitting and standing
    if transition_mean < sitting_mean or transition_mean < standing_mean:
        return False, "transition phase does not show increased acoustic energy"
    
    # Check for energy increase ratio
    energy_ratio = transition_mean / max(sitting_mean, standing_mean)
    if energy_ratio < 1.2:
        return False, f"insufficient energy increase during transition (ratio: {energy_ratio:.2f})"
    
    return True, f"valid sit-to-stand acoustic pattern (transition energy {energy_ratio:.2f}x higher)"


def _validate_flexion_extension_acoustic_waveform(rms_envelope: np.ndarray) -> tuple[bool, str]:
    """Validate acoustic waveform pattern for flexion-extension maneuvers.
    
    Flexion-extension acoustic patterns should exhibit:
    1. Peaks during movement transitions
    2. Cyclic pattern with variation
    3. Energy throughout the cycle (not just at endpoints)
    
    Args:
        rms_envelope: RMS envelope of acoustic signal
        
    Returns:
        Tuple of (is_valid, reason)
    """
    if len(rms_envelope) < 10:
        return False, "insufficient data points for waveform validation"
    
    # Check for signal variation
    signal_std = np.std(rms_envelope)
    signal_mean = np.mean(rms_envelope)
    
    if signal_mean == 0:
        return False, "zero mean signal"
    
    cv = signal_std / signal_mean
    if cv < 0.15:
        return False, f"insufficient waveform variation for flexion-extension (CV: {cv:.2f})"
    
    # Find peaks
    signal_range = np.max(rms_envelope) - np.min(rms_envelope)
    prominence = max(signal_range * 0.2, signal_mean * 0.5)
    
    peaks, _ = find_peaks(rms_envelope, prominence=prominence)
    
    if len(peaks) == 0:
        return False, "no acoustic peaks detected"
    
    # For flexion-extension, expect at least one peak in the middle portion
    cycle_length = len(rms_envelope)
    peak_positions = peaks / cycle_length
    
    # Check if any peak is in the middle 30-70% of cycle
    middle_peaks = [p for p in peak_positions if 0.3 <= p <= 0.7]
    
    if not middle_peaks:
        return False, "no acoustic peaks detected in middle portion of cycle"
    
    return True, f"valid flexion-extension acoustic pattern ({len(peaks)} peak(s))"


def run_cycle_audio_qc(
    cycle_df: pd.DataFrame,
    time_col: str = "tt",
    audio_channels: list[str] | None = None,
    check_periodic_noise: bool = True,
    fail_on_periodic_noise: bool = False,
) -> dict[str, any]:
    """Run comprehensive audio QC checks on a movement cycle.
    
    This is the main entry point for cycle-specific audio QC, coordinating
    all cycle-level audio quality checks.
    
    Args:
        cycle_df: DataFrame containing audio data for a single movement cycle
        time_col: Name of time column
        audio_channels: List of channel names to check (default: ch1-ch4)
        check_periodic_noise: Whether to run periodic noise detection
        fail_on_periodic_noise: Whether to mark cycle as failed if periodic noise detected
        
    Returns:
        Dictionary containing QC results:
        - 'periodic_noise': Per-channel periodic noise detection results (dict)
        - 'has_periodic_noise': True if any channel has periodic noise (bool)
        - 'qc_pass': Overall QC pass/fail for the cycle (bool)
    """
    results: dict[str, any] = {
        'periodic_noise': {},
        'has_periodic_noise': False,
        'qc_pass': True,
    }
    
    if check_periodic_noise:
        results['periodic_noise'] = check_cycle_periodic_noise(
            cycle_df, time_col, audio_channels
        )
        
        # Check if any channel has periodic noise
        results['has_periodic_noise'] = any(results['periodic_noise'].values())
        
        # Update QC pass/fail based on configuration
        if fail_on_periodic_noise and results['has_periodic_noise']:
            results['qc_pass'] = False
    
    return results


def check_sync_quality_by_phase(
    cycle_df: pd.DataFrame,
    maneuver: str,
    time_col: str = "tt",
    knee_angle_col: str = "Knee Angle Z",
    audio_channels: list[str] | None = None,
    reference_ranges: dict[str, tuple[float, float]] | None = None,
) -> dict[str, any]:
    """Check synchronization quality by comparing acoustic features across movement phases.
    
    This function validates the quality of audio-biomechanics synchronization by analyzing
    whether acoustic features appear at expected joint angle phases throughout the movement
    cycle. For example, in walking, peak acoustic emissions should occur during specific
    gait phases (e.g., heel strike, mid-stance).
    
    The validation divides each cycle into phases based on knee angle, computes acoustic
    features (RMS energy) for each phase, and compares them to expected reference ranges.
    
    Args:
        cycle_df: DataFrame containing synchronized audio and biomechanics data
        maneuver: Type of movement (walk, sit_to_stand, flexion_extension)
        time_col: Name of time column
        knee_angle_col: Name of knee angle column
        audio_channels: List of channel names to check (default: f_ch1-f_ch4)
        reference_ranges: Optional dict mapping phase names to (min, max) RMS energy ranges.
                         If None, uses default ranges based on maneuver type.
        
    Returns:
        Dictionary containing sync quality results:
        - 'phase_acoustic_features': Dict mapping phase to acoustic feature values
        - 'phase_in_range': Dict mapping phase to whether features are in expected range
        - 'sync_quality_score': Overall score (0-1) representing fraction of phases in range
        - 'sync_qc_pass': Boolean indicating if sync quality is acceptable
        
    Notes:
        - For walking: Phases are extension (low angle), mid-swing, flexion (high angle)
        - For sit-to-stand: Phases are sitting (high angle), transition, standing (low angle)
        - For flexion-extension: Phases are extension (low angle), mid-phase, flexion (high angle)
    """
    if audio_channels is None:
        audio_channels = ["f_ch1", "f_ch2", "f_ch3", "f_ch4"]
    
    # Filter to available channels
    available_channels = [ch for ch in audio_channels if ch in cycle_df.columns]
    
    if not available_channels or knee_angle_col not in cycle_df.columns:
        return {
            'phase_acoustic_features': {},
            'phase_in_range': {},
            'sync_quality_score': 0.0,
            'sync_qc_pass': False,
            'error': 'Missing required data columns',
        }
    
    # Extract knee angle data
    knee_angle = cycle_df[knee_angle_col].values
    
    # Remove NaN values and check for sufficient data
    valid_mask = ~np.isnan(knee_angle)
    if valid_mask.sum() < 10:
        return {
            'phase_acoustic_features': {},
            'phase_in_range': {},
            'sync_quality_score': 0.0,
            'sync_qc_pass': False,
            'error': 'Insufficient valid knee angle data',
        }
    
    # Filter knee angle to valid values only
    knee_angle_valid = knee_angle[valid_mask]
    
    # Define phases based on knee angle percentiles
    phases = _define_movement_phases(knee_angle_valid, maneuver)
    
    # Get reference ranges for this maneuver
    if reference_ranges is None:
        reference_ranges = _get_default_reference_ranges(maneuver)
    
    # Compute acoustic features for each phase
    phase_features = {}
    phase_in_range = {}
    
    for phase_name, (angle_min, angle_max) in phases.items():
        # Find indices within this phase (working with valid data only)
        phase_mask_valid = (knee_angle_valid >= angle_min) & (knee_angle_valid <= angle_max)
        
        if phase_mask_valid.sum() < 5:
            # Not enough data in this phase
            phase_features[phase_name] = None
            phase_in_range[phase_name] = False
            continue
        
        # Map back to original indices for data extraction
        valid_indices = np.where(valid_mask)[0]
        phase_indices = valid_indices[phase_mask_valid]
        
        # Compute RMS acoustic energy for this phase across all channels
        phase_rms = 0.0
        for ch in available_channels:
            ch_data = cycle_df[ch].values[phase_indices]
            ch_rms = np.sqrt(np.mean(ch_data ** 2))
            phase_rms += ch_rms
        
        phase_rms /= len(available_channels)  # Average across channels
        phase_features[phase_name] = float(phase_rms)
        
        # Check if feature is in expected range
        if phase_name in reference_ranges:
            ref_min, ref_max = reference_ranges[phase_name]
            phase_in_range[phase_name] = ref_min <= phase_rms <= ref_max
        else:
            # No reference range defined for this phase - default to pass
            phase_in_range[phase_name] = True
    
    # Calculate sync quality score (fraction of phases in range)
    if len(phase_in_range) > 0:
        sync_quality_score = sum(phase_in_range.values()) / len(phase_in_range)
    else:
        sync_quality_score = 0.0
    
    # Pass if at least 75% of phases are in range
    sync_qc_pass = sync_quality_score >= 0.75
    
    return {
        'phase_acoustic_features': phase_features,
        'phase_in_range': phase_in_range,
        'sync_quality_score': float(sync_quality_score),
        'sync_qc_pass': sync_qc_pass,
    }


def _define_movement_phases(
    knee_angle: np.ndarray,
    maneuver: str,
) -> dict[str, tuple[float, float]]:
    """Define movement phases based on knee angle ranges.
    
    Divides the movement cycle into phases (extension, mid-phase, flexion) based
    on knee angle quartiles or terciles.
    
    Args:
        knee_angle: Clean knee angle array (no NaNs)
        maneuver: Type of movement
        
    Returns:
        Dictionary mapping phase names to (min_angle, max_angle) tuples
    """
    min_angle = float(np.min(knee_angle))
    max_angle = float(np.max(knee_angle))
    
    if maneuver in ["walk", "flexion_extension"]:
        # For walking and flexion-extension: extension -> mid-swing -> flexion
        # Use terciles to divide into 3 phases
        tercile_33 = float(np.percentile(knee_angle, 33))
        tercile_67 = float(np.percentile(knee_angle, 67))
        
        return {
            'extension': (min_angle, tercile_33),
            'mid_phase': (tercile_33, tercile_67),
            'flexion': (tercile_67, max_angle),
        }
    elif maneuver == "sit_to_stand":
        # For sit-to-stand: sitting (high angle) -> transition -> standing (low angle)
        # Use terciles but reverse the naming since angle decreases
        tercile_33 = float(np.percentile(knee_angle, 33))
        tercile_67 = float(np.percentile(knee_angle, 67))
        
        return {
            'standing': (min_angle, tercile_33),  # Low angles = standing
            'transition': (tercile_33, tercile_67),
            'sitting': (tercile_67, max_angle),  # High angles = sitting
        }
    else:
        # Default: simple 3-phase division
        tercile_33 = float(np.percentile(knee_angle, 33))
        tercile_67 = float(np.percentile(knee_angle, 67))
        
        return {
            'low_angle': (min_angle, tercile_33),
            'mid_angle': (tercile_33, tercile_67),
            'high_angle': (tercile_67, max_angle),
        }


def _get_default_reference_ranges(
    maneuver: str,
) -> dict[str, tuple[float, float]]:
    """Get default reference ranges for acoustic features by phase.
    
    These are conservative reference ranges based on expected acoustic patterns.
    In practice, these should be calibrated from a reference dataset.
    
    Args:
        maneuver: Type of movement
        
    Returns:
        Dictionary mapping phase names to (min_rms, max_rms) tuples
    """
    if maneuver == "walk":
        # Walking: Higher acoustic energy during heel strike/stance (extension/mid)
        # Lower energy during swing (flexion)
        return {
            'extension': (DEFAULT_MID_ENERGY_MIN, DEFAULT_MAX_RMS_ENERGY),
            'mid_phase': (DEFAULT_MID_ENERGY_MIN, DEFAULT_MAX_RMS_ENERGY),
            'flexion': (DEFAULT_MIN_RMS_ENERGY, DEFAULT_LOW_ENERGY_MAX),
        }
    elif maneuver == "sit_to_stand":
        # Sit-to-stand: Energy increases during transition as muscles engage
        return {
            'sitting': (DEFAULT_MIN_RMS_ENERGY, DEFAULT_LOW_ENERGY_MAX),
            'transition': (DEFAULT_MID_ENERGY_MIN, DEFAULT_MAX_RMS_ENERGY),
            'standing': (DEFAULT_MIN_RMS_ENERGY, DEFAULT_LOW_ENERGY_MAX),
        }
    elif maneuver == "flexion_extension":
        # Flexion-extension: Energy throughout movement, peaks during transitions
        return {
            'extension': (DEFAULT_VARIABLE_ENERGY_MIN, DEFAULT_MAX_RMS_ENERGY),
            'mid_phase': (DEFAULT_MID_ENERGY_MIN, DEFAULT_MAX_RMS_ENERGY),
            'flexion': (DEFAULT_VARIABLE_ENERGY_MIN, DEFAULT_MAX_RMS_ENERGY),
        }
    else:
        # Default: permissive ranges
        return {
            'low_angle': (DEFAULT_MIN_RMS_ENERGY, DEFAULT_MAX_RMS_ENERGY),
            'mid_angle': (DEFAULT_MIN_RMS_ENERGY, DEFAULT_MAX_RMS_ENERGY),
            'high_angle': (DEFAULT_MIN_RMS_ENERGY, DEFAULT_MAX_RMS_ENERGY),
        }


# ---------------------------------------------------------------------------
# Phase E: Sync-level periodic artifact detection + per-cycle intermittent
# ---------------------------------------------------------------------------

def detect_periodic_artifacts_sync_level(
    synced_df: pd.DataFrame,
    cycles: list[pd.DataFrame],
    time_col: str = "tt",
    audio_channels: list[str] | None = None,
    knee_angle_col: str = "Knee Angle Z",
    *,
    window_size_s: float = 2.0,
    peak_snr_threshold: float = 10.0,
    movement_freq_tolerance_hz: float = 0.1,
) -> dict[str, list[tuple[float, float]]]:
    """Detect periodic artifacts on the full exercise portion of a sync'd recording.

    This function is designed to run on the **entire synchronized DataFrame**
    (exercise portion only, from first to last extracted cycle boundary).
    Periodic artifacts may only appear during specific movement phases but
    repeat across cycles (e.g., wire rubbing, mic bumping thigh). Detecting
    on the full exercise portion captures these patterns with better
    frequency resolution than per-cycle analysis.

    The algorithm:
    1. Trims ``synced_df`` to the exercise portion (first cycle start →
       last cycle end).
    2. Estimates the movement-cycle rate from the knee angle signal.
    3. Runs per-channel sliding-window Welch PSD on the audio channels.
    4. Flags windows where a narrowband peak (peak/median > SNR threshold)
       exists at a frequency NOT correlated with the movement cycle rate.
    5. Returns per-channel time intervals of flagged windows.

    Args:
        synced_df: Full synchronized DataFrame.
        cycles: List of cycle DataFrames (used to determine exercise bounds).
        time_col: Name of the time column.
        audio_channels: Audio channel names (default ch1-ch4).
        knee_angle_col: Name of knee angle column for movement rate estimation.
        window_size_s: Sliding PSD window length in seconds.
        peak_snr_threshold: Peak/median power ratio for flagging.
        movement_freq_tolerance_hz: Tolerance around the movement fundamental
            and harmonics when excluding movement-related peaks.

    Returns:
        ``dict[str, list[tuple[float, float]]]`` mapping channel name to
        list of (start_time, end_time) tuples of periodic artifact intervals.
    """
    if audio_channels is None:
        audio_channels = ["ch1", "ch2", "ch3", "ch4"]

    if not cycles:
        return {ch: [] for ch in audio_channels}

    # Determine exercise-portion time bounds from first/last cycles
    def _cycle_time_bounds(cycle_df: pd.DataFrame) -> tuple[float, float]:
        if time_col not in cycle_df.columns:
            return (0.0, 0.0)
        tt = cycle_df[time_col]
        if isinstance(tt.iloc[0], pd.Timedelta):
            return (tt.iloc[0].total_seconds(), tt.iloc[-1].total_seconds())
        return (float(tt.iloc[0]), float(tt.iloc[-1]))

    first_start, _ = _cycle_time_bounds(cycles[0])
    _, last_end = _cycle_time_bounds(cycles[-1])

    # Trim synced_df to exercise portion
    if time_col not in synced_df.columns:
        return {ch: [] for ch in audio_channels}

    if isinstance(synced_df[time_col].iloc[0], pd.Timedelta):
        time_s = synced_df[time_col].dt.total_seconds().values
    else:
        time_s = synced_df[time_col].values.astype(float)

    exercise_mask = (time_s >= first_start) & (time_s <= last_end)
    exercise_df = synced_df.loc[exercise_mask]
    exercise_time = time_s[exercise_mask]

    if len(exercise_df) < 256:
        return {ch: [] for ch in audio_channels}

    # Estimate sampling rate
    dt = float(np.median(np.diff(exercise_time)))
    if dt <= 0:
        return {ch: [] for ch in audio_channels}
    fs = 1.0 / dt

    # Estimate movement-cycle frequency from knee angle (fundamental)
    movement_freq = _estimate_movement_frequency(
        exercise_df, knee_angle_col, fs
    )

    available_channels = [ch for ch in audio_channels if ch in exercise_df.columns]
    per_channel: dict[str, list[tuple[float, float]]] = {}

    for ch in available_channels:
        ch_data = pd.to_numeric(exercise_df[ch], errors="coerce").to_numpy()
        ch_intervals = _sliding_psd_periodic_detection(
            ch_data,
            exercise_time,
            fs,
            window_size_s=window_size_s,
            peak_snr_threshold=peak_snr_threshold,
            movement_freq=movement_freq,
            movement_freq_tolerance_hz=movement_freq_tolerance_hz,
        )
        per_channel[ch] = ch_intervals

    # Fill missing channels with empty lists
    for ch in audio_channels:
        if ch not in per_channel:
            per_channel[ch] = []

    return per_channel


def _estimate_movement_frequency(
    df: pd.DataFrame,
    knee_angle_col: str,
    fs: float,
) -> float | None:
    """Estimate the fundamental movement-cycle frequency from knee angle.

    Uses the autocorrelation of the knee angle signal to find the dominant
    period, then converts to Hz. Returns ``None`` if estimation fails.
    """
    if knee_angle_col not in df.columns:
        return None

    angle = pd.to_numeric(df[knee_angle_col], errors="coerce").to_numpy()
    angle = np.nan_to_num(angle, nan=0.0)

    if len(angle) < 100:
        return None

    # Detrend
    angle = angle - np.mean(angle)

    # Autocorrelation
    corr = np.correlate(angle, angle, mode="full")
    corr = corr[len(corr) // 2:]
    if np.max(corr) <= 0:
        return None
    corr = corr / corr[0]

    # Find first peak after at least 0.5s (to avoid DC/short-range correlation)
    min_lag = max(int(fs * 0.5), 1)
    if min_lag >= len(corr):
        return None

    peaks, _ = find_peaks(corr[min_lag:], prominence=0.1)
    if len(peaks) == 0:
        return None

    dominant_lag = peaks[0] + min_lag
    period_s = dominant_lag / fs
    if period_s <= 0:
        return None

    return 1.0 / period_s


def _sliding_psd_periodic_detection(
    data: np.ndarray,
    time_s: np.ndarray,
    fs: float,
    *,
    window_size_s: float,
    peak_snr_threshold: float,
    movement_freq: float | None,
    movement_freq_tolerance_hz: float,
) -> list[tuple[float, float]]:
    """Sliding-window PSD to find periodic artifact intervals.

    Marks windows where a narrowband spectral peak exists that is NOT
    at the movement fundamental or its harmonics.
    """
    from scipy.signal import welch as scipy_welch

    n = len(data)
    window_samples = max(int(fs * window_size_s), 256)
    step = window_samples // 2

    flagged_intervals: list[tuple[float, float]] = []

    for start_idx in range(0, n - window_samples + 1, step):
        end_idx = start_idx + window_samples
        segment = data[start_idx:end_idx]

        try:
            nperseg = min(window_samples, max(int(fs * 0.5), 64))
            freqs, psd = scipy_welch(
                segment, fs=fs, nperseg=nperseg, noverlap=nperseg // 2
            )
        except Exception:
            continue

        if len(psd) == 0:
            continue

        # Ignore DC / very low frequencies
        freq_mask = freqs > 5.0
        if not np.any(freq_mask):
            continue

        psd_nondc = psd[freq_mask]
        freqs_nondc = freqs[freq_mask]

        median_power = np.median(psd_nondc)
        if median_power <= 0:
            continue

        # Find peaks in PSD that exceed SNR threshold
        peak_idx = np.argmax(psd_nondc)
        if psd_nondc[peak_idx] / median_power < peak_snr_threshold:
            continue  # No dominant peak

        peak_freq = freqs_nondc[peak_idx]

        # Exclude movement-related frequencies (fundamental + harmonics)
        if movement_freq is not None and movement_freq > 0:
            is_movement_related = False
            for harmonic in range(1, 6):  # up to 5th harmonic
                if abs(peak_freq - harmonic * movement_freq) < movement_freq_tolerance_hz:
                    is_movement_related = True
                    break
            if is_movement_related:
                continue

        # This window has a non-movement periodic peak — flag it
        t_start = time_s[start_idx] if start_idx < len(time_s) else 0.0
        t_end = time_s[min(end_idx - 1, len(time_s) - 1)]
        flagged_intervals.append((float(t_start), float(t_end)))

    # Merge overlapping flagged intervals
    if not flagged_intervals:
        return []

    from src.audio.raw_qc import merge_artifact_intervals
    return merge_artifact_intervals(flagged_intervals)


def classify_cycle_artifacts_ml(
    cycle_df: pd.DataFrame,
    maneuver: str,
    time_col: str = "tt",
    audio_channels: list[str] | None = None,
) -> dict[str, list[tuple[float, float]]]:
    """Classify per-cycle artifacts using a trained ML model.

    .. todo::
       Implement when ML model for maneuver-specific acoustic patterns
       is available.  The model should learn to distinguish normal
       acoustic emissions (knee crepitus, cartilage friction) from
       external artifacts (talking, bumps, environmental noise).

    Expected interface:
        - Input: cycle DataFrame with audio + biomechanics columns, maneuver type
        - Output: per-channel dict mapping channel name to list of
          (start_time, end_time) artifact intervals within the cycle

    Raises:
        NotImplementedError: Always — this is a placeholder for future ML work.
    """
    raise NotImplementedError(
        "ML-based cycle artifact classification is not yet implemented. "
        "Use the statistical fallback detect_cycle_intermittent_artifacts() instead."
    )


def detect_cycle_intermittent_artifacts(
    cycle_df: pd.DataFrame,
    time_col: str = "tt",
    audio_channels: list[str] | None = None,
    *,
    rms_window_s: float = 0.005,
    baseline_window_s: float = 0.05,
    threshold_sigma: float = 3.5,
    min_artifact_duration_s: float = 0.05,
    max_artifact_duration_s: float = 4.0,
) -> dict[str, list[tuple[float, float]]]:
    """Detect short transient (intermittent) artifacts within a single cycle.

    Uses short-time RMS envelope analysis: computes a fast RMS (5 ms windows)
    and compares against a slower baseline RMS (50 ms windows).  Transient
    events that exceed the baseline by ``threshold_sigma`` standard deviations
    are flagged, subject to duration filters.

    Duration filters:
    - ``min_artifact_duration_s`` (default 50 ms): shorter events are likely
      legitimate acoustic emissions from noisy knees, not artifacts.
    - ``max_artifact_duration_s`` (default 4 s): longer events are classified
      as continuous artifacts and handled by Phase C/D.

    Args:
        cycle_df: Single cycle DataFrame.
        time_col: Time column name.
        audio_channels: Channel names (default ch1-ch4).
        rms_window_s: Fast RMS window in seconds.
        baseline_window_s: Slow baseline RMS window in seconds.
        threshold_sigma: Number of baseline std-devs for detection.
        min_artifact_duration_s: Minimum event duration (seconds).
        max_artifact_duration_s: Maximum event duration (seconds).

    Returns:
        ``dict[str, list[tuple[float, float]]]`` mapping channel name to
        list of (start, end) artifact intervals.
    """
    if audio_channels is None:
        audio_channels = ["ch1", "ch2", "ch3", "ch4"]

    available_channels = [ch for ch in audio_channels if ch in cycle_df.columns]
    if not available_channels or time_col not in cycle_df.columns:
        return {ch: [] for ch in audio_channels}

    # Get time array
    if isinstance(cycle_df[time_col].iloc[0], pd.Timedelta):
        time_s = cycle_df[time_col].dt.total_seconds().values
    else:
        time_s = cycle_df[time_col].values.astype(float)

    if len(time_s) < 10:
        return {ch: [] for ch in audio_channels}

    dt = float(np.median(np.diff(time_s)))
    if dt <= 0:
        return {ch: [] for ch in audio_channels}
    fs = 1.0 / dt

    fast_window = max(int(fs * rms_window_s), 3)
    slow_window = max(int(fs * baseline_window_s), fast_window * 2)

    from src.audio.raw_qc import _sliding_window_rms, _mask_to_intervals

    per_channel: dict[str, list[tuple[float, float]]] = {}

    for ch in audio_channels:
        if ch not in available_channels:
            per_channel[ch] = []
            continue

        ch_data = pd.to_numeric(cycle_df[ch], errors="coerce").to_numpy()
        ch_data = np.nan_to_num(ch_data, nan=0.0)

        # Fast and slow RMS envelopes
        fast_rms = _sliding_window_rms(ch_data, fast_window)
        slow_rms = _sliding_window_rms(ch_data, slow_window)

        # Baseline std-dev (using slow window)
        from src.audio.raw_qc import _sliding_window_std
        slow_std = _sliding_window_std(ch_data, slow_window)

        # Flag where fast RMS exceeds baseline + threshold * std
        artifact_mask = fast_rms > (slow_rms + threshold_sigma * slow_std)

        # Convert to intervals
        raw_intervals = _mask_to_intervals(artifact_mask, time_s, min_duration_s=0.0)

        # Duration filters
        filtered: list[tuple[float, float]] = []
        for start, end in raw_intervals:
            duration = end - start
            if min_artifact_duration_s <= duration <= max_artifact_duration_s:
                filtered.append((start, end))

        per_channel[ch] = filtered

    return per_channel


def detect_cycle_artifacts(
    cycle_df: pd.DataFrame,
    maneuver: str,
    time_col: str = "tt",
    audio_channels: list[str] | None = None,
    *,
    continuous_intervals_per_ch: dict[str, list[tuple[float, float]]] | None = None,
    periodic_intervals_per_ch: dict[str, list[tuple[float, float]]] | None = None,
    use_ml: bool = False,
) -> dict[str, dict[str, list[tuple[float, float]]]]:
    """Orchestrate all per-cycle artifact detection and combine results.

    Combines three sources of artifact intervals for each cycle:
    1. **Intermittent**: Detected per-cycle via statistical or ML methods.
    2. **Continuous**: Carried forward from audio-stage detection (Phase C/D),
       already trimmed to cycle bounds.
    3. **Periodic**: Carried forward from sync-level detection (Phase E.1),
       already trimmed to cycle bounds.

    Args:
        cycle_df: Single cycle DataFrame.
        maneuver: Maneuver type.
        time_col: Time column name.
        audio_channels: Channel names.
        continuous_intervals_per_ch: Pre-trimmed continuous artifact intervals
            per channel for this cycle (from Phase D).
        periodic_intervals_per_ch: Pre-trimmed periodic artifact intervals
            per channel for this cycle (from Phase E.1).
        use_ml: If True, try ML classification first.

    Returns:
        Nested dict: ``{channel: {"intermittent": [...], "continuous": [...],
        "periodic": [...], "all": [...]}}`` where each value is a list of
        (start, end) tuples.
    """
    if audio_channels is None:
        audio_channels = ["ch1", "ch2", "ch3", "ch4"]
    if continuous_intervals_per_ch is None:
        continuous_intervals_per_ch = {}
    if periodic_intervals_per_ch is None:
        periodic_intervals_per_ch = {}

    # Step 1: Intermittent detection (ML or statistical)
    intermittent_per_ch: dict[str, list[tuple[float, float]]] = {}
    if use_ml:
        try:
            intermittent_per_ch = classify_cycle_artifacts_ml(
                cycle_df, maneuver, time_col, audio_channels
            )
        except NotImplementedError:
            intermittent_per_ch = detect_cycle_intermittent_artifacts(
                cycle_df, time_col, audio_channels
            )
    else:
        intermittent_per_ch = detect_cycle_intermittent_artifacts(
            cycle_df, time_col, audio_channels
        )

    from src.audio.raw_qc import merge_artifact_intervals

    results: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for ch in audio_channels:
        intermittent = intermittent_per_ch.get(ch, [])
        continuous = continuous_intervals_per_ch.get(ch, [])
        periodic = periodic_intervals_per_ch.get(ch, [])

        all_merged = merge_artifact_intervals(intermittent, continuous, periodic)

        results[ch] = {
            "intermittent": intermittent,
            "continuous": continuous,
            "periodic": periodic,
            "all": all_merged,
        }

    return results


# ---------------------------------------------------------------------------
# Phase F: Sync QC ghost + rule-based improvement
# ---------------------------------------------------------------------------

def validate_sync_quality_model_based(
    cycle_df: pd.DataFrame,
    maneuver: str,
    time_col: str = "tt",
    audio_channels: list[str] | None = None,
    knee_angle_col: str = "Knee Angle Z",
) -> tuple[bool, float, str]:
    """Validate sync quality using a trained model.

    .. todo::
       Implement when a model that predicts expected acoustic patterns
       from a joint angle trajectory is available.

    Expected interface:
        - Input: cycle DataFrame + maneuver type
        - Output: (is_valid, quality_score 0-1, reason string)

    Raises:
        NotImplementedError: Always — placeholder for future ML work.
    """
    raise NotImplementedError(
        "Model-based sync quality validation is not yet implemented. "
        "Use the rule-based fallback validate_sync_quality_rule_based() instead."
    )


def validate_sync_quality_rule_based(
    cycle_df: pd.DataFrame,
    maneuver: str,
    time_col: str = "tt",
    audio_channels: list[str] | None = None,
    knee_angle_col: str = "Knee Angle Z",
) -> tuple[bool, float, str]:
    """Validate sync quality using rule-based acoustic-biomechanics cross-reference.

    Wraps the existing ``validate_acoustic_waveform()`` but additionally
    checks that acoustic peaks coincide with expected biomechanical events:

    - **Walking**: acoustic peaks near heel strikes (angle minima).
    - **Sit-to-stand**: energy peaks during BOTH transitions (consistent
      with bidirectional cycles from Phase B).
    - **Flexion-extension**: peaks align with mid-range angles.

    Args:
        cycle_df: Single cycle DataFrame.
        maneuver: Maneuver type.
        time_col: Time column name.
        audio_channels: Audio channel names.
        knee_angle_col: Knee angle column name.

    Returns:
        ``(is_valid, quality_score, reason)``
    """
    # First run the existing waveform validation
    waveform_valid, waveform_reason = validate_acoustic_waveform(
        cycle_df, maneuver, time_col, audio_channels
    )

    if not waveform_valid:
        return False, 0.0, f"waveform: {waveform_reason}"

    # Cross-reference acoustic peaks with biomechanics
    if knee_angle_col not in cycle_df.columns or time_col not in cycle_df.columns:
        return waveform_valid, 0.5, f"waveform ok, no biomechanics cross-ref available"

    knee_angle = pd.to_numeric(cycle_df[knee_angle_col], errors="coerce").values
    valid_mask = ~np.isnan(knee_angle)
    if valid_mask.sum() < 10:
        return waveform_valid, 0.5, f"waveform ok, insufficient biomechanics data"

    # Compute RMS envelope across audio channels
    if audio_channels is None:
        audio_channels = ["f_ch1", "f_ch2", "f_ch3", "f_ch4"]
    available = [ch for ch in audio_channels if ch in cycle_df.columns]
    if not available:
        return waveform_valid, 0.5, "waveform ok, no audio channels for cross-ref"

    rms_envelope = np.zeros(len(cycle_df))
    for ch in available:
        rms_envelope += cycle_df[ch].values ** 2
    rms_envelope = np.sqrt(rms_envelope / len(available))

    # Find acoustic peaks
    signal_range = np.max(rms_envelope) - np.min(rms_envelope)
    if signal_range <= 0:
        return waveform_valid, 0.5, "waveform ok, flat acoustic signal"

    prominence = signal_range * 0.2
    acoustic_peaks, _ = find_peaks(rms_envelope, prominence=prominence)

    if len(acoustic_peaks) == 0:
        return waveform_valid, 0.5, "waveform ok, no prominent acoustic peaks for cross-ref"

    # Maneuver-specific cross-referencing
    cycle_length = len(cycle_df)

    if maneuver == "walk":
        # Acoustic peaks should be near knee angle minima (heel strikes)
        angle_minima, _ = find_peaks(-knee_angle, prominence=5.0)
        if len(angle_minima) == 0:
            return waveform_valid, 0.5, "waveform ok, no angle minima found"

        # Check proximity of acoustic peaks to angle minima
        proximity_score = _peak_proximity_score(acoustic_peaks, angle_minima, cycle_length)
        is_valid = proximity_score >= 0.3
        return is_valid, proximity_score, (
            f"walk cross-ref: proximity={proximity_score:.2f}"
        )

    elif maneuver == "sit_to_stand":
        # Energy should increase during BOTH transitions (bidirectional)
        third = cycle_length // 3
        if third < 2:
            return waveform_valid, 0.5, "waveform ok, cycle too short for phase analysis"

        energy_thirds = [
            np.mean(rms_envelope[:third]),
            np.mean(rms_envelope[third:2*third]),
            np.mean(rms_envelope[2*third:]),
        ]
        # Middle third (transition) should have higher energy
        transition_ratio = energy_thirds[1] / max(max(energy_thirds[0], energy_thirds[2]), 1e-10)
        score = min(transition_ratio / 2.0, 1.0)  # Normalize
        is_valid = transition_ratio >= 1.1
        return is_valid, score, (
            f"STS cross-ref: transition_ratio={transition_ratio:.2f}"
        )

    elif maneuver == "flexion_extension":
        # Peaks should align with mid-range angles
        angle_range = np.max(knee_angle[valid_mask]) - np.min(knee_angle[valid_mask])
        mid_angle = np.min(knee_angle[valid_mask]) + angle_range * 0.5
        mid_tolerance = angle_range * 0.3

        mid_range_peaks = [
            p for p in acoustic_peaks
            if valid_mask[p] and abs(knee_angle[p] - mid_angle) < mid_tolerance
        ]
        fraction_mid = len(mid_range_peaks) / max(len(acoustic_peaks), 1)
        is_valid = fraction_mid >= 0.3
        return is_valid, fraction_mid, (
            f"FE cross-ref: {len(mid_range_peaks)}/{len(acoustic_peaks)} peaks at mid-range"
        )

    return waveform_valid, 0.5, f"waveform ok, no cross-ref for maneuver={maneuver}"


def _peak_proximity_score(
    peaks_a: np.ndarray,
    peaks_b: np.ndarray,
    signal_length: int,
    max_distance_frac: float = 0.15,
) -> float:
    """Score how well two sets of peaks align (0 = no alignment, 1 = perfect).

    For each peak in ``peaks_a``, finds the nearest peak in ``peaks_b`` and
    scores based on distance (as fraction of signal length).
    """
    if len(peaks_a) == 0 or len(peaks_b) == 0:
        return 0.0

    max_dist = max_distance_frac * signal_length
    scores = []
    for pa in peaks_a:
        min_dist = np.min(np.abs(peaks_b.astype(float) - float(pa)))
        if min_dist <= max_dist:
            scores.append(1.0 - min_dist / max_dist)
        else:
            scores.append(0.0)

    return float(np.mean(scores))


def validate_sync_quality(
    cycle_df: pd.DataFrame,
    maneuver: str,
    time_col: str = "tt",
    audio_channels: list[str] | None = None,
    knee_angle_col: str = "Knee Angle Z",
    use_model: bool = False,
) -> tuple[bool, float, str]:
    """Orchestrate sync quality validation: model-based first, then rule-based.

    Args:
        cycle_df: Single cycle DataFrame.
        maneuver: Maneuver type.
        time_col: Time column name.
        audio_channels: Audio channel names.
        knee_angle_col: Knee angle column name.
        use_model: If True, try model-based first.

    Returns:
        ``(is_valid, quality_score, reason)``
    """
    if use_model:
        try:
            return validate_sync_quality_model_based(
                cycle_df, maneuver, time_col, audio_channels, knee_angle_col
            )
        except NotImplementedError:
            pass

    return validate_sync_quality_rule_based(
        cycle_df, maneuver, time_col, audio_channels, knee_angle_col
    )


def run_comprehensive_cycle_qc(
    cycle_df: pd.DataFrame,
    maneuver: str,
    time_col: str = "tt",
    knee_angle_col: str = "Knee Angle Z",
    audio_channels: list[str] | None = None,
    check_periodic_noise: bool = True,
    check_waveform_shape: bool = True,
    fail_on_periodic_noise: bool = False,
    reference_waveform: np.ndarray | None = None,
    correlation_threshold: float = 0.5,
) -> dict[str, any]:
    """Run all cycle-level QC checks: audio QC, waveform validation, and sync quality.
    
    This is the comprehensive entry point that coordinates all cycle-level quality checks.
    
    Args:
        cycle_df: DataFrame containing synchronized audio and biomechanics data
        maneuver: Type of movement (walk, sit_to_stand, flexion_extension)
        time_col: Name of time column
        knee_angle_col: Name of knee angle column (used if reference_waveform provided)
        audio_channels: List of channel names to check
        check_periodic_noise: Whether to check for periodic background noise
        check_waveform_shape: Whether to check acoustic waveform shape
        fail_on_periodic_noise: Whether to fail QC if periodic noise detected
        reference_waveform: Optional reference waveform for model-based validation.
                           If provided, uses ML-based validation instead of rule-based.
        correlation_threshold: Minimum correlation for model-based validation (default: 0.5)
        
    Returns:
        Dictionary containing all QC results:
        - 'audio_qc': Results from audio-specific checks (periodic noise)
        - 'waveform_qc': Results from waveform shape validation
        - 'overall_qc_pass': Boolean indicating if all checks passed
    """
    results = {
        'audio_qc': {},
        'waveform_qc': {},
        'overall_qc_pass': True,
    }
    
    # Run audio QC (periodic noise detection)
    if check_periodic_noise:
        audio_qc_results = run_cycle_audio_qc(
            cycle_df,
            time_col=time_col,
            audio_channels=audio_channels,
            check_periodic_noise=True,
            fail_on_periodic_noise=fail_on_periodic_noise,
        )
        results['audio_qc'] = audio_qc_results
        if not audio_qc_results['qc_pass']:
            results['overall_qc_pass'] = False
    
    # Run waveform shape validation (replaces phase-based validation)
    if check_waveform_shape:
        is_valid, reason = validate_acoustic_waveform(
            cycle_df,
            maneuver=maneuver,
            time_col=time_col,
            audio_channels=audio_channels,
            reference_waveform=reference_waveform,
            correlation_threshold=correlation_threshold,
        )
        results['waveform_qc'] = {
            'waveform_valid': is_valid,
            'validation_reason': reason,
            'validation_method': 'model-based' if reference_waveform is not None else 'rule-based',
        }
        if not is_valid:
            results['overall_qc_pass'] = False
    
    return results
