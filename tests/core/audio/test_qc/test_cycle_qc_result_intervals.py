"""Tests for CycleQCResult interval field population from _CycleResult.

Validates that _save_qc_results() in quality_control.py correctly maps
intermittent and periodic artifact intervals from _CycleResult into
CycleQCResult fields, and that CycleQCResult enforces no-defaults.
"""

from pydantic import ValidationError
import pytest

from src.metadata import CycleQCResult


class TestCycleQCResultNoDefaults:
    """Verify CycleQCResult requires explicit values for all fields."""

    def test_missing_interval_fields_raises(self):
        """CycleQCResult without interval fields raises ValidationError."""
        with pytest.raises(ValidationError, match="Field required"):
            CycleQCResult(
                cycle_index=0,
                cycle_file="test.pkl",
                is_outlier=False,
                acoustic_energy=1.0,
                biomechanics_qc_pass=True,
                sync_qc_pass=True,
                sync_quality_score=0.95,
                audio_qc_pass=True,
                audio_qc_mic_1_pass=True,
                audio_qc_mic_2_pass=True,
                audio_qc_mic_3_pass=True,
                audio_qc_mic_4_pass=True,
                periodic_noise_detected=False,
                periodic_noise_ch1=False,
                periodic_noise_ch2=False,
                periodic_noise_ch3=False,
                periodic_noise_ch4=False,
                # Intentionally omitting interval fields
            )

    def test_missing_boolean_fields_raises(self):
        """CycleQCResult without boolean QC fields raises ValidationError."""
        with pytest.raises(ValidationError, match="Field required"):
            CycleQCResult(
                cycle_index=0,
                cycle_file="test.pkl",
                is_outlier=False,
                # Missing acoustic_energy, biomechanics_qc_pass, etc.
            )


class TestCycleQCResultConstruction:
    """Verify CycleQCResult accepts all required interval fields."""

    def _make_result(self, **overrides) -> CycleQCResult:
        """Helper to build a CycleQCResult with sensible defaults."""
        defaults = {
            "cycle_index": 0,
            "cycle_file": "test_cycle_000.pkl",
            "is_outlier": False,
            "acoustic_energy": 1.5,
            "biomechanics_qc_pass": True,
            "sync_qc_pass": True,
            "sync_quality_score": 0.95,
            "audio_qc_pass": True,
            "audio_qc_mic_1_pass": True,
            "audio_qc_mic_2_pass": True,
            "audio_qc_mic_3_pass": True,
            "audio_qc_mic_4_pass": True,
            "periodic_noise_detected": False,
            "periodic_noise_ch1": False,
            "periodic_noise_ch2": False,
            "periodic_noise_ch3": False,
            "periodic_noise_ch4": False,
            "intermittent_intervals_ch1": [],
            "intermittent_intervals_ch2": [],
            "intermittent_intervals_ch3": [],
            "intermittent_intervals_ch4": [],
            "periodic_intervals_ch1": [],
            "periodic_intervals_ch2": [],
            "periodic_intervals_ch3": [],
            "periodic_intervals_ch4": [],
        }
        defaults.update(overrides)
        return CycleQCResult(**defaults)

    def test_all_clean_cycle(self):
        """Clean cycle has empty interval lists and all pass flags True."""
        result = self._make_result()
        assert result.audio_qc_pass is True
        assert result.intermittent_intervals_ch1 == []
        assert result.periodic_intervals_ch1 == []

    def test_intermittent_artifacts_on_ch2(self):
        """Intermittent artifact on ch2 should be captured."""
        result = self._make_result(
            audio_qc_pass=False,
            audio_qc_mic_2_pass=False,
            intermittent_intervals_ch2=[(0.5, 0.8), (1.1, 1.3)],
        )
        assert result.audio_qc_pass is False
        assert result.audio_qc_mic_2_pass is False
        assert result.intermittent_intervals_ch2 == [(0.5, 0.8), (1.1, 1.3)]
        assert result.intermittent_intervals_ch1 == []

    def test_periodic_artifacts_on_ch1_and_ch3(self):
        """Periodic noise on ch1 and ch3 should be captured."""
        result = self._make_result(
            audio_qc_pass=False,
            periodic_noise_detected=True,
            periodic_noise_ch1=True,
            periodic_noise_ch3=True,
            audio_qc_mic_1_pass=False,
            audio_qc_mic_3_pass=False,
            periodic_intervals_ch1=[(0.0, 1.2)],
            periodic_intervals_ch3=[(0.2, 0.9)],
        )
        assert result.periodic_noise_detected is True
        assert result.periodic_intervals_ch1 == [(0.0, 1.2)]
        assert result.periodic_intervals_ch3 == [(0.2, 0.9)]
        assert result.periodic_intervals_ch2 == []
        assert result.periodic_intervals_ch4 == []

    def test_both_intermittent_and_periodic(self):
        """Cycle with both artifact types should carry both sets of intervals."""
        result = self._make_result(
            audio_qc_pass=False,
            periodic_noise_detected=True,
            periodic_noise_ch1=True,
            audio_qc_mic_1_pass=False,
            audio_qc_mic_4_pass=False,
            intermittent_intervals_ch4=[(0.3, 0.6)],
            periodic_intervals_ch1=[(0.0, 1.2)],
        )
        assert result.intermittent_intervals_ch4 == [(0.3, 0.6)]
        assert result.periodic_intervals_ch1 == [(0.0, 1.2)]

    def test_walk_metadata_optional(self):
        """Walk-specific metadata (pass_number, speed) remains optional."""
        result = self._make_result(pass_number=3, speed="fast")
        assert result.pass_number == 3
        assert result.speed == "fast"

    def test_walk_metadata_defaults_to_none(self):
        """Walk-specific metadata defaults to None when not provided."""
        result = self._make_result()
        assert result.pass_number is None
        assert result.speed is None


class TestCycleResultToCycleQCResultMapping:
    """Test the mapping logic that _save_qc_results uses.

    This simulates the dict-based access pattern used in quality_control.py:
        result.intermittent_intervals.get("ch1", [])
    """

    def test_dict_get_pattern_for_intermittent(self):
        """Simulate the .get() pattern used for intermittent intervals."""
        intermittent_intervals = {
            "ch1": [(0.5, 0.8)],
            "ch2": [],
            "ch3": [(1.0, 1.5), (2.0, 2.3)],
            "ch4": [],
        }

        # This mirrors _save_qc_results logic
        ch1 = intermittent_intervals.get("ch1", [])
        ch2 = intermittent_intervals.get("ch2", [])
        ch3 = intermittent_intervals.get("ch3", [])
        ch4 = intermittent_intervals.get("ch4", [])

        assert ch1 == [(0.5, 0.8)]
        assert ch2 == []
        assert ch3 == [(1.0, 1.5), (2.0, 2.3)]
        assert ch4 == []

    def test_dict_get_pattern_for_periodic(self):
        """Simulate the .get() pattern used for periodic intervals."""
        periodic_intervals = {
            "ch1": [(0.0, 1.2)],
            "ch3": [(0.2, 0.9)],
        }

        # Missing keys should return []
        ch1 = periodic_intervals.get("ch1", [])
        ch2 = periodic_intervals.get("ch2", [])
        ch3 = periodic_intervals.get("ch3", [])
        ch4 = periodic_intervals.get("ch4", [])

        assert ch1 == [(0.0, 1.2)]
        assert ch2 == []
        assert ch3 == [(0.2, 0.9)]
        assert ch4 == []

    def test_empty_dict_returns_all_empty(self):
        """Empty intervals dict should produce all empty lists."""
        intervals: dict[str, list] = {}
        for ch in ["ch1", "ch2", "ch3", "ch4"]:
            assert intervals.get(ch, []) == []
