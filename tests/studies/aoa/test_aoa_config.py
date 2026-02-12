"""Tests for AOA study configuration.

Verifies that AOAConfig correctly implements all study-specific conventions:
directory names, file patterns, sheet names, and participant ID parsing.
"""

from pathlib import Path

import pytest

from src.studies.aoa.config import AOAConfig


@pytest.fixture
def config():
    return AOAConfig()


class TestAOAKneeDirectory:
    """Verify knee directory naming convention."""

    def test_left_knee(self, config):
        assert config.get_knee_directory_name("left") == "Left Knee"

    def test_right_knee(self, config):
        assert config.get_knee_directory_name("right") == "Right Knee"


class TestAOABiomechanicsFilePattern:
    """Verify biomechanics file pattern generation."""

    def test_numeric_id(self, config):
        assert config.get_biomechanics_file_pattern(1011) == "AOA1011_Biomechanics_Full_Set"

    def test_string_id(self, config):
        assert config.get_biomechanics_file_pattern("1011") == "AOA1011_Biomechanics_Full_Set"


class TestAOAAcousticsSheet:
    """Verify acoustics sheet name."""

    def test_sheet_name(self, config):
        assert config.get_acoustics_sheet_name() == "Acoustic Notes"


class TestAOABiomechanicsSheetNames:
    """Verify biomechanics Excel sheet name construction."""

    def test_walk_slow(self, config):
        sheets = config.construct_biomechanics_sheet_names("AOA1011", "walk", speed="slow")
        assert sheets["data_sheet"] == "AOA1011_Slow_Walking"
        assert sheets["event_sheet"] == "AOA1011_Walk0001"

    def test_walk_medium(self, config):
        sheets = config.construct_biomechanics_sheet_names("AOA1011", "walk", speed="medium")
        assert sheets["data_sheet"] == "AOA1011_Medium_Walking"

    def test_walk_fast(self, config):
        sheets = config.construct_biomechanics_sheet_names("AOA1011", "walk", speed="fast")
        assert sheets["data_sheet"] == "AOA1011_Fast_Walking"

    def test_sit_to_stand(self, config):
        sheets = config.construct_biomechanics_sheet_names("AOA1011", "sit_to_stand")
        assert sheets["data_sheet"] == "AOA1011_SitToStand"
        assert sheets["event_sheet"] == "AOA1011_StoS_Events"

    def test_flexion_extension(self, config):
        sheets = config.construct_biomechanics_sheet_names("AOA1011", "flexion_extension")
        assert sheets["data_sheet"] == "AOA1011_FlexExt"
        assert sheets["event_sheet"] == "AOA1011_FE_Events"

    def test_invalid_speed_raises(self, config):
        with pytest.raises(ValueError, match="Invalid speed"):
            config.construct_biomechanics_sheet_names("AOA1011", "walk", speed="turbo")

    def test_unknown_maneuver_raises(self, config):
        with pytest.raises(ValueError, match="Unknown maneuver"):
            config.construct_biomechanics_sheet_names("AOA1011", "jumping")


class TestAOAParticipantIdParsing:
    """Verify participant ID parsing from directory names."""

    def test_hash_prefix(self, config):
        study, num = config.parse_participant_id("#AOA1011")
        assert study == "AOA"
        assert num == 1011

    def test_no_hash(self, config):
        study, num = config.parse_participant_id("AOA1011")
        assert study == "AOA"
        assert num == 1011

    def test_numeric_only(self, config):
        study, num = config.parse_participant_id("1011")
        assert study == "AOA"
        assert num == 1011

    def test_invalid_raises(self, config):
        with pytest.raises(ValueError, match="Cannot parse"):
            config.parse_participant_id("INVALID")


class TestAOAStudyPrefix:
    """Verify study prefix formatting."""

    def test_format(self, config):
        assert config.format_study_prefix(1011) == "AOA1011"


class TestAOALegendFilePattern:
    """Verify legend file glob pattern."""

    def test_pattern(self, config):
        assert config.get_legend_file_pattern() == "*acoustic_file_legend*"


class TestAOAFindExcelFile:
    """Verify Excel file finding utility."""

    def test_finds_xlsx(self, config, tmp_path):
        (tmp_path / "test_file.xlsx").touch()
        found = config.find_excel_file(tmp_path, "test_file")
        assert found is not None
        assert found.name == "test_file.xlsx"

    def test_finds_xlsm(self, config, tmp_path):
        (tmp_path / "test_file.xlsm").touch()
        found = config.find_excel_file(tmp_path, "test_file")
        assert found is not None
        assert found.name == "test_file.xlsm"

    def test_prefers_xlsx_over_xlsm(self, config, tmp_path):
        (tmp_path / "test_file.xlsx").touch()
        (tmp_path / "test_file.xlsm").touch()
        found = config.find_excel_file(tmp_path, "test_file")
        assert found.name == "test_file.xlsx"

    def test_returns_none_when_not_found(self, config, tmp_path):
        found = config.find_excel_file(tmp_path, "nonexistent")
        assert found is None

    def test_glob_pattern(self, config, tmp_path):
        (tmp_path / "AOA1011_Biomechanics_Full_Set.xlsx").touch()
        found = config.find_excel_file(tmp_path, "AOA1011_Biomechanics_Full_Set")
        assert found is not None
