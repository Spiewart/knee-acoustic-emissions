"""AOA (Assessment of Osteoarthritis) study configuration.

Auto-registers AOAConfig with the study registry on import.
"""

from src.studies.aoa.config import AOAConfig
from src.studies.registry import register_study

# Auto-register on import
_config = AOAConfig()
register_study(_config)

__all__ = ["AOAConfig"]
