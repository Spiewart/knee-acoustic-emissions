"""Study configuration registry.

Provides a simple registry for looking up study configurations by name.
Studies auto-register themselves when their module is imported.
"""

from typing import Optional

from src.studies.base import StudyConfig

_registry: dict[str, StudyConfig] = {}


def register_study(config: StudyConfig) -> None:
    """Register a study configuration.

    Called automatically when a study module is imported.

    Args:
        config: Study configuration instance implementing StudyConfig protocol
    """
    _registry[config.study_name] = config


def get_study_config(study_name: str) -> StudyConfig:
    """Look up a study configuration by name.

    Args:
        study_name: Short study identifier (e.g., "AOA", "preOA")

    Returns:
        Study configuration instance

    Raises:
        ValueError: If the study name is not registered
    """
    # Auto-import known studies on first lookup
    if not _registry:
        _import_known_studies()

    if study_name not in _registry:
        available = ", ".join(sorted(_registry.keys())) or "(none)"
        raise ValueError(
            f"Unknown study '{study_name}'. Available: {available}"
        )

    return _registry[study_name]


def list_studies() -> list[str]:
    """List all registered study names."""
    if not _registry:
        _import_known_studies()
    return sorted(_registry.keys())


def _import_known_studies() -> None:
    """Import known study modules to trigger auto-registration."""
    try:
        import src.studies.aoa  # noqa: F401 — triggers AOAConfig auto-registration
    except ImportError:
        pass
