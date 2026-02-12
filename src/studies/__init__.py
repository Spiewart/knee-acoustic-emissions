"""Multi-study architecture for the acoustic emissions pipeline.

Each study (AOA, preOA, SMoCK) has its own directory structure, file naming
conventions, biomechanics sheet names, and metadata formats. This module
provides a StudyConfig protocol that encapsulates study-specific logic,
keeping the core pipeline generic.

Usage:
    from src.studies import get_study_config

    config = get_study_config("AOA")
    knee_dir = config.get_knee_directory_name("left")  # "Left Knee"
    biomech_pattern = config.get_biomechanics_file_pattern("1011")  # "AOA1011_Biomechanics_Full_Set"
"""

from src.studies.registry import get_study_config, register_study

__all__ = ["get_study_config", "register_study"]
