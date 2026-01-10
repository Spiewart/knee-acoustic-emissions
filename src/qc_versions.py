"""QC Version Management

This module defines and tracks versions for different QA/QC methods used in
acoustic emissions processing. As QC methods evolve over time, version tracking
ensures that metadata correctly indicates which version of QC each file passed.

Version History:
    Audio QC v1: Initial implementation with periodic audio detection
    Biomechanics QC v1: Basic biomechanics validation
    Cycle QC v1: Initial cycle-level acoustic energy thresholding
    Cycle QC v2: Added biomechanics validation with knee angle range of motion checks
"""

from typing import Literal

# Current QC version constants
# Increment these when QC methods are modified
AUDIO_QC_VERSION = 1
BIOMECH_QC_VERSION = 1
CYCLE_QC_VERSION = 2


def get_audio_qc_version() -> int:
    """Get the current audio QC version.
    
    This version applies to QC performed on audio after processing .bin files.
    
    Returns:
        Current audio QC version number
    """
    return AUDIO_QC_VERSION


def get_biomech_qc_version() -> int:
    """Get the current biomechanics QC version.
    
    This version applies to QC performed on biomechanics data pre-synchronization.
    
    Returns:
        Current biomechanics QC version number
    """
    return BIOMECH_QC_VERSION


def get_cycle_qc_version() -> int:
    """Get the current cycle QC version.
    
    This version applies to QC performed on synchronized audio/biomechanics
    after parsing into individual movement cycles.
    
    Returns:
        Current cycle QC version number
    """
    return CYCLE_QC_VERSION


def get_qc_version(
    qc_type: Literal["audio", "biomech", "cycle"]
) -> int:
    """Get the current QC version for a specific QC type.
    
    Args:
        qc_type: Type of QC ("audio", "biomech", or "cycle")
        
    Returns:
        Current QC version number for the specified type
        
    Raises:
        ValueError: If qc_type is not recognized
    """
    if qc_type == "audio":
        return get_audio_qc_version()
    elif qc_type == "biomech":
        return get_biomech_qc_version()
    elif qc_type == "cycle":
        return get_cycle_qc_version()
    else:
        raise ValueError(
            f"Unknown QC type: {qc_type}. "
            f"Must be 'audio', 'biomech', or 'cycle'"
        )
