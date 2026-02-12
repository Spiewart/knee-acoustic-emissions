"""Quality control for biomechanics data.

This module provides validation functions for biomechanics waveform patterns
and project-level constants for knee angle thresholds across different maneuvers.
"""

from typing import Literal, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks

# Project-level constants for knee angle validation
# These are the expected minimum ROM values for each maneuver type
DEFAULT_MIN_ROM_WALK = 20.0  # degrees - Walking has smaller ROM per gait cycle
DEFAULT_MIN_ROM_SIT_TO_STAND = 40.0  # degrees - Large ROM from sitting to standing
DEFAULT_MIN_ROM_FLEXION_EXTENSION = 40.0  # degrees - Significant ROM during controlled cycles

# Tolerance constants for waveform pattern validation
DEFAULT_WALK_ANGLE_TOLERANCE = 10.0  # degrees - Start/end angle match tolerance for walking
DEFAULT_FLEXION_EXTENSION_ANGLE_TOLERANCE = 15.0  # degrees - More lenient for flexion-extension

# Peak detection parameters
DEFAULT_PEAK_PROMINENCE_RATIO = 0.3  # Prominence as fraction of ROM
MIN_ABSOLUTE_PROMINENCE = 5.0  # degrees - Minimum prominence to avoid noise detection

# Sit-to-stand validation parameters
DEFAULT_STS_ANGLE_CHANGE_RATIO = 0.5  # Minimum angle change as fraction of ROM
DEFAULT_STS_SEATED_THRESHOLD = 45.0  # degrees - Above this, participant is seated at start
DEFAULT_STS_ENDPOINT_TOLERANCE = 20.0  # degrees - Start/end angle match tolerance for STS cycles

# NaN filtering threshold
MIN_VALID_DATA_FRACTION = 0.8  # Minimum fraction of non-NaN data points required


def get_default_min_rom(maneuver: Literal["walk", "sit_to_stand", "flexion_extension"]) -> float:
    """Get the default minimum ROM threshold for a maneuver.
    
    Args:
        maneuver: Type of movement
        
    Returns:
        Default minimum ROM in degrees
    """
    if maneuver == "walk":
        return DEFAULT_MIN_ROM_WALK
    elif maneuver == "sit_to_stand":
        return DEFAULT_MIN_ROM_SIT_TO_STAND
    elif maneuver == "flexion_extension":
        return DEFAULT_MIN_ROM_FLEXION_EXTENSION
    else:
        return DEFAULT_MIN_ROM_WALK  # Default fallback


def compute_knee_angle_rom(knee_angle: np.ndarray) -> float:
    """Compute range of motion (ROM) for knee angle data.
    
    ROM is calculated as the difference between maximum and minimum knee angle
    values, representing the extent of joint movement.
    
    Args:
        knee_angle: Array of knee angle values (should not contain NaNs)
        
    Returns:
        Range of motion in degrees
    """
    if len(knee_angle) == 0:
        return 0.0
    
    return float(np.max(knee_angle) - np.min(knee_angle))


def validate_knee_angle_waveform(
    knee_angle: np.ndarray,
    maneuver: Literal["walk", "sit_to_stand", "flexion_extension"],
    min_rom: Optional[float] = None,
    angle_tolerance: Optional[float] = None,
    peak_prominence_ratio: float = DEFAULT_PEAK_PROMINENCE_RATIO,
    min_absolute_prominence: float = MIN_ABSOLUTE_PROMINENCE,
) -> Tuple[bool, str]:
    """Validate that knee angle waveform matches expected pattern for the maneuver.
    
    Performs waveform-level validation beyond simple ROM checks, verifying that
    the knee angle exhibits the stereotypic fluctuation pattern characteristic
    of the maneuver type.
    
    Args:
        knee_angle: Clean knee angle array (no NaNs, sufficient data points)
        maneuver: Type of movement
        min_rom: Minimum ROM threshold (uses default if None)
        angle_tolerance: Start/end angle matching tolerance (uses default if None)
        peak_prominence_ratio: Peak prominence as fraction of ROM
        min_absolute_prominence: Minimum absolute prominence for peak detection
        
    Returns:
        Tuple of (is_valid, reason) where is_valid indicates if the cycle passes
        validation and reason provides details about the validation result.
    """
    if len(knee_angle) < 10:
        return False, "insufficient data points"
    
    # Get default ROM threshold if not provided
    if min_rom is None:
        min_rom = get_default_min_rom(maneuver)
    
    # Compute ROM
    rom = compute_knee_angle_rom(knee_angle)
    if rom < min_rom:
        return False, f"ROM={rom:.1f}° below threshold {min_rom:.1f}°"
    
    # Perform maneuver-specific waveform validation
    if maneuver == "walk":
        return validate_walking_waveform(
            knee_angle, rom, angle_tolerance, peak_prominence_ratio, min_absolute_prominence
        )
    elif maneuver == "sit_to_stand":
        return validate_sit_to_stand_waveform(knee_angle, rom)
    elif maneuver == "flexion_extension":
        return validate_flexion_extension_waveform(
            knee_angle, rom, angle_tolerance, peak_prominence_ratio, min_absolute_prominence
        )
    else:
        # Unknown maneuver - fall back to ROM-only check
        return True, f"ROM={rom:.1f}°"


def validate_walking_waveform(
    knee_angle: np.ndarray,
    rom: float,
    angle_tolerance: Optional[float] = None,
    peak_prominence_ratio: float = DEFAULT_PEAK_PROMINENCE_RATIO,
    min_absolute_prominence: float = MIN_ABSOLUTE_PROMINENCE,
) -> Tuple[bool, str]:
    """Validate waveform pattern for walking gait cycles.
    
    Walking gait cycles should exhibit:
    1. Start and end at similar low angles (heel strike)
    2. A flexion peak during swing phase (typically in middle portion, ~20-80% of cycle)
    3. Single dominant peak (not multiple peaks of similar magnitude)
    
    Args:
        knee_angle: Clean knee angle array (no NaNs)
        rom: Pre-computed range of motion
        angle_tolerance: Start/end angle match tolerance (uses default if None)
        peak_prominence_ratio: Peak prominence as fraction of ROM
        min_absolute_prominence: Minimum absolute prominence for peak detection
        
    Returns:
        Tuple of (is_valid, reason)
    """
    if angle_tolerance is None:
        angle_tolerance = DEFAULT_WALK_ANGLE_TOLERANCE
    
    # Check for proper start/end angles (should be at extension, i.e., minima)
    start_angle = knee_angle[0]
    end_angle = knee_angle[-1]
    
    if abs(end_angle - start_angle) > angle_tolerance:
        return False, f"start/end angle mismatch: {abs(end_angle - start_angle):.1f}° > {angle_tolerance}°"
    
    # Find flexion peaks (maxima) with minimum absolute prominence
    prominence = max(rom * peak_prominence_ratio, min_absolute_prominence)
    peaks, _ = find_peaks(knee_angle, prominence=prominence)
    
    if len(peaks) == 0:
        return False, "no flexion peak detected"
    
    # Should have one dominant peak during swing phase
    if len(peaks) > 2:
        return False, f"too many peaks detected ({len(peaks)})"
    
    # Peak should be in middle portion of cycle (20-80%)
    cycle_length = len(knee_angle)
    peak_idx = peaks[0]  # Use first/dominant peak
    peak_position = peak_idx / cycle_length
    
    if peak_position < 0.2 or peak_position > 0.8:
        return False, f"peak at {peak_position*100:.0f}% of cycle (expected 20-80%)"
    
    return True, f"ROM={rom:.1f}°, valid gait pattern"


def validate_sit_to_stand_waveform(
    knee_angle: np.ndarray,
    rom: float,
    angle_change_ratio: float = DEFAULT_STS_ANGLE_CHANGE_RATIO,
    seated_threshold: float = DEFAULT_STS_SEATED_THRESHOLD,
    endpoint_tolerance: float = DEFAULT_STS_ENDPOINT_TOLERANCE,
) -> Tuple[bool, str]:
    """Validate waveform pattern for full sit-to-stand-sit cycles.

    Handles both cycle orientations by inferring the starting position from
    the initial knee angle:

    - **Standing start** (initial angle < seated_threshold): expects a
      low→high→low pattern (stand→sit→stand) with a sitting peak in the middle.
    - **Seated start** (initial angle >= seated_threshold): expects a
      high→low→high pattern (sit→stand→sit) with a standing trough in the middle.

    Validation checks:
    1. Start and end angles are roughly similar (same phase of cycle)
    2. A prominent peak or trough exists in the middle 25-75% of the cycle
    3. The peak/trough amplitude is substantial relative to ROM

    Args:
        knee_angle: Clean knee angle array (no NaNs)
        rom: Pre-computed range of motion
        angle_change_ratio: Minimum peak/trough amplitude as fraction of ROM
        seated_threshold: Angle threshold (degrees) above which participant
                         is considered to start in seated position
        endpoint_tolerance: Maximum allowed difference between start and end
                          angles (degrees)

    Returns:
        Tuple of (is_valid, reason)
    """
    if len(knee_angle) < 10:
        return False, "insufficient data points for sit-to-stand waveform validation"

    start_angle = knee_angle[0]
    end_angle = knee_angle[-1]

    # Check that start and end angles are roughly similar (same phase)
    endpoint_diff = abs(end_angle - start_angle)
    if endpoint_diff > endpoint_tolerance:
        return False, (
            f"start/end angle mismatch: {endpoint_diff:.1f}° > {endpoint_tolerance:.1f}° "
            f"(start={start_angle:.1f}°, end={end_angle:.1f}°)"
        )

    # Infer starting position from initial angle
    starts_seated = start_angle >= seated_threshold

    min_prominence = max(rom * angle_change_ratio, MIN_ABSOLUTE_PROMINENCE)
    cycle_length = len(knee_angle)

    if starts_seated:
        # Sit→Stand→Sit: expect a trough (standing) in the middle
        # Find minima (troughs) by searching for peaks in inverted signal
        troughs, properties = find_peaks(-knee_angle, prominence=min_prominence)

        if len(troughs) == 0:
            return False, "no standing trough detected (sit→stand→sit pattern)"

        # Find the most prominent trough
        trough_idx = troughs[np.argmin(knee_angle[troughs])]
        trough_position = trough_idx / cycle_length

        if trough_position < 0.15 or trough_position > 0.85:
            return False, (
                f"standing trough at {trough_position*100:.0f}% of cycle "
                f"(expected 15-85%)"
            )

        # Verify the trough is substantially lower than endpoints
        trough_depth = start_angle - knee_angle[trough_idx]
        min_depth = rom * angle_change_ratio
        if trough_depth < min_depth:
            return False, (
                f"insufficient trough depth: {trough_depth:.1f}° < {min_depth:.1f}°"
            )

        return True, f"ROM={rom:.1f}°, valid sit→stand→sit pattern (seated start)"

    else:
        # Stand→Sit→Stand: expect a peak (sitting) in the middle
        peaks, properties = find_peaks(knee_angle, prominence=min_prominence)

        if len(peaks) == 0:
            return False, "no sitting peak detected (stand→sit→stand pattern)"

        # Find the most prominent peak
        peak_idx = peaks[np.argmax(knee_angle[peaks])]
        peak_position = peak_idx / cycle_length

        if peak_position < 0.15 or peak_position > 0.85:
            return False, (
                f"sitting peak at {peak_position*100:.0f}% of cycle "
                f"(expected 15-85%)"
            )

        # Verify the peak is substantially higher than endpoints
        peak_height = knee_angle[peak_idx] - start_angle
        min_height = rom * angle_change_ratio
        if peak_height < min_height:
            return False, (
                f"insufficient peak height: {peak_height:.1f}° < {min_height:.1f}°"
            )

        return True, f"ROM={rom:.1f}°, valid stand→sit→stand pattern (standing start)"


def validate_flexion_extension_waveform(
    knee_angle: np.ndarray,
    rom: float,
    angle_tolerance: Optional[float] = None,
    peak_prominence_ratio: float = DEFAULT_PEAK_PROMINENCE_RATIO,
    min_absolute_prominence: float = MIN_ABSOLUTE_PROMINENCE,
) -> Tuple[bool, str]:
    """Validate waveform pattern for flexion-extension maneuvers.
    
    Flexion-extension cycles should exhibit:
    1. Start and end at similar angles (extension position)
    2. Clear flexion peak in the middle
    3. Relatively smooth, cyclic pattern
    
    Args:
        knee_angle: Clean knee angle array (no NaNs)
        rom: Pre-computed range of motion
        angle_tolerance: Start/end angle match tolerance (uses default if None)
        peak_prominence_ratio: Peak prominence as fraction of ROM
        min_absolute_prominence: Minimum absolute prominence for peak detection
        
    Returns:
        Tuple of (is_valid, reason)
    """
    if angle_tolerance is None:
        angle_tolerance = DEFAULT_FLEXION_EXTENSION_ANGLE_TOLERANCE
    
    # Check for proper start/end angles (should be similar - at extension)
    start_angle = knee_angle[0]
    end_angle = knee_angle[-1]
    
    if abs(end_angle - start_angle) > angle_tolerance:
        return False, f"start/end angle mismatch: {abs(end_angle - start_angle):.1f}° > {angle_tolerance}°"
    
    # Find flexion peaks (maxima) with minimum absolute prominence
    prominence = max(rom * peak_prominence_ratio, min_absolute_prominence)
    peaks, _ = find_peaks(knee_angle, prominence=prominence)
    
    if len(peaks) == 0:
        return False, "no flexion peak detected"
    
    # Should have at least one clear peak
    if len(peaks) > 3:
        return False, f"too many peaks detected ({len(peaks)})"
    
    # Main peak should be in middle portion of cycle
    cycle_length = len(knee_angle)
    peak_idx = peaks[np.argmax(knee_angle[peaks])]  # Get index of highest peak
    peak_position = peak_idx / cycle_length
    
    if peak_position < 0.25 or peak_position > 0.75:
        return False, f"peak at {peak_position*100:.0f}% of cycle (expected 25-75%)"
    
    return True, f"ROM={rom:.1f}°, valid flexion-extension pattern"
