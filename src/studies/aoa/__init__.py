"""AOA (Assessment of Osteoarthritis) study configuration.

Auto-registers AOAConfig with the study registry on import.
"""

from src.studies.aoa.config import AOAConfig
from src.studies.aoa.legend import parse_aoa_mic_setup_sheet
from src.studies.registry import register_study

# Auto-register on import
_config = AOAConfig()
register_study(_config)

__all__ = ["AOAConfig", "parse_aoa_mic_setup_sheet"]
