"""Tests for AOA study configuration.

Verifies that AOAConfig correctly implements all study-specific conventions:
directory names, file patterns, sheet names, participant ID parsing,
UID parsing, and speed code mappings.
"""

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


class TestAOAWalkEventSheetBaseName:
    """Verify walk event sheet base name."""

    def test_base_name(self, config):
        assert config.get_walk_event_sheet_base_name() == "Walk0001"


class TestAOASpeedCodeMap:
    """Verify speed code mapping."""

    def test_all_speeds_present(self, config):
        speed_map = config.get_speed_code_map()
        assert speed_map == {
            "slow": "SS",
            "normal": "NS",
            "fast": "FS",
        }


class TestAOASpeedEventKeywords:
    """Verify speed event section header keywords."""

    def test_all_keywords_present(self, config):
        keywords = config.get_speed_event_keywords()
        assert keywords == {
            "Slow Speed": "SS",
            "Normal Speed": "NS",
            "Medium Speed": "NS",
            "Fast Speed": "FS",
        }

    def test_medium_maps_to_normal(self, config):
        keywords = config.get_speed_event_keywords()
        assert keywords["Medium Speed"] == keywords["Normal Speed"]


class TestAOAParseBiomechanicsUid:
    """Verify UID parsing into BiomechanicsFileMetadata."""

    def test_walk_uid(self, config):
        meta = config.parse_biomechanics_uid(
            "AOA1011_Walk0001_NSP1_Filt",
        )
        assert meta.scripted_maneuver == "walk"
        assert meta.speed == "normal"
        assert meta.pass_number == 1
        assert meta.study == "AOA"
        assert meta.study_id == 1011

    def test_walk_slow_pass2(self, config):
        meta = config.parse_biomechanics_uid(
            "AOA1011_Walk0001_SSP2_Filt",
        )
        assert meta.scripted_maneuver == "walk"
        assert meta.speed == "slow"
        assert meta.pass_number == 2

    def test_walk_fast_pass3(self, config):
        meta = config.parse_biomechanics_uid(
            "AOA1011_Walk0001_FSP3_Filt",
        )
        assert meta.scripted_maneuver == "walk"
        assert meta.speed == "fast"
        assert meta.pass_number == 3

    def test_sit_to_stand_uid(self, config):
        meta = config.parse_biomechanics_uid(
            "AOA1011_SitToStand0001_Filt",
        )
        assert meta.scripted_maneuver == "sit_to_stand"
        assert meta.speed is None
        assert meta.pass_number is None

    def test_flexion_extension_uid(self, config):
        meta = config.parse_biomechanics_uid(
            "AOA1011_FlexExt0001_Filt",
        )
        assert meta.scripted_maneuver == "flexion_extension"
        assert meta.speed is None
        assert meta.pass_number is None

    def test_invalid_maneuver_raises(self, config):
        with pytest.raises(ValueError, match="Unknown maneuver"):
            config.parse_biomechanics_uid(
                "AOA1011_Jumping0001_Filt",
            )

    def test_invalid_speed_code_raises(self, config):
        with pytest.raises(ValueError, match="Unknown speed code"):
            config.parse_biomechanics_uid(
                "AOA1011_Walk0001_XSP1_Filt",
            )
