"""Tests for the study configuration registry.

Verifies that study configs can be registered, looked up, and that
unknown studies raise appropriate errors.
"""

import pytest

from src.studies.registry import get_study_config, list_studies


class TestStudyRegistry:
    """Verify study config registration and lookup."""

    def test_get_aoa_config(self):
        config = get_study_config("AOA")
        assert config.study_name == "AOA"

    def test_unknown_study_raises(self):
        with pytest.raises(ValueError, match="Unknown study"):
            get_study_config("NONEXISTENT")

    def test_list_studies_includes_aoa(self):
        studies = list_studies()
        assert "AOA" in studies

    def test_aoa_config_satisfies_protocol(self):
        """AOAConfig should satisfy the StudyConfig protocol."""
        config = get_study_config("AOA")
        # Verify all protocol methods exist
        assert hasattr(config, "study_name")
        assert hasattr(config, "get_knee_directory_name")
        assert hasattr(config, "get_biomechanics_file_pattern")
        assert hasattr(config, "get_acoustics_sheet_name")
        assert hasattr(config, "construct_biomechanics_sheet_names")
        assert hasattr(config, "get_legend_file_pattern")
        assert hasattr(config, "parse_participant_id")
        assert hasattr(config, "format_study_prefix")
        assert hasattr(config, "find_excel_file")
