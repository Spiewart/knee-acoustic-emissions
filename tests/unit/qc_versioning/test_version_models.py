"""Tests for QC version management."""

import pytest

from src.qc_versions import (
    AUDIO_QC_VERSION,
    BIOMECH_QC_VERSION,
    CYCLE_QC_VERSION,
    get_audio_qc_version,
    get_biomech_qc_version,
    get_cycle_qc_version,
    get_qc_version,
)


class TestQCVersionConstants:
    """Test QC version constants are properly defined."""

    def test_audio_qc_version_is_positive_int(self):
        """Audio QC version should be a positive integer."""
        assert isinstance(AUDIO_QC_VERSION, int)
        assert AUDIO_QC_VERSION > 0

    def test_biomech_qc_version_is_positive_int(self):
        """Biomechanics QC version should be a positive integer."""
        assert isinstance(BIOMECH_QC_VERSION, int)
        assert BIOMECH_QC_VERSION > 0

    def test_cycle_qc_version_is_positive_int(self):
        """Cycle QC version should be a positive integer."""
        assert isinstance(CYCLE_QC_VERSION, int)
        assert CYCLE_QC_VERSION > 0


class TestQCVersionGetters:
    """Test QC version getter functions."""

    def test_get_audio_qc_version_returns_constant(self):
        """get_audio_qc_version should return AUDIO_QC_VERSION."""
        assert get_audio_qc_version() == AUDIO_QC_VERSION

    def test_get_biomech_qc_version_returns_constant(self):
        """get_biomech_qc_version should return BIOMECH_QC_VERSION."""
        assert get_biomech_qc_version() == BIOMECH_QC_VERSION

    def test_get_cycle_qc_version_returns_constant(self):
        """get_cycle_qc_version should return CYCLE_QC_VERSION."""
        assert get_cycle_qc_version() == CYCLE_QC_VERSION

    def test_get_qc_version_audio(self):
        """get_qc_version('audio') should return audio QC version."""
        assert get_qc_version("audio") == AUDIO_QC_VERSION

    def test_get_qc_version_biomech(self):
        """get_qc_version('biomech') should return biomech QC version."""
        assert get_qc_version("biomech") == BIOMECH_QC_VERSION

    def test_get_qc_version_cycle(self):
        """get_qc_version('cycle') should return cycle QC version."""
        assert get_qc_version("cycle") == CYCLE_QC_VERSION

    def test_get_qc_version_invalid_type_raises_error(self):
        """get_qc_version with invalid type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown QC type"):
            get_qc_version("invalid_type")  # type: ignore


class TestVersionConsistency:
    """Test that version values are consistent across all getters."""

    def test_audio_version_consistent(self):
        """Audio QC version should be consistent across calls."""
        version1 = get_audio_qc_version()
        version2 = get_audio_qc_version()
        assert version1 == version2

    def test_biomech_version_consistent(self):
        """Biomech QC version should be consistent across calls."""
        version1 = get_biomech_qc_version()
        version2 = get_biomech_qc_version()
        assert version1 == version2

    def test_cycle_version_consistent(self):
        """Cycle QC version should be consistent across calls."""
        version1 = get_cycle_qc_version()
        version2 = get_cycle_qc_version()
        assert version1 == version2

    def test_get_qc_version_consistent_with_direct_getters(self):
        """get_qc_version should return same values as direct getters."""
        assert get_qc_version("audio") == get_audio_qc_version()
        assert get_qc_version("biomech") == get_biomech_qc_version()
        assert get_qc_version("cycle") == get_cycle_qc_version()
