"""Tests for Excel file finding functionality (.xlsx and .xlsm)."""

from pathlib import Path

import pytest

from src.orchestration.participant import _find_excel_file


@pytest.fixture
def temp_excel_dir(tmp_path):
    """Create a temporary directory for testing Excel file finding."""
    return tmp_path


def test_find_excel_file_xlsx(temp_excel_dir):
    """Test finding .xlsx file."""
    # Create a .xlsx file
    xlsx_file = temp_excel_dir / "test_file.xlsx"
    xlsx_file.write_text("test")

    result = _find_excel_file(temp_excel_dir, "test_file")
    assert result is not None
    assert result.name == "test_file.xlsx"


def test_find_excel_file_xlsm(temp_excel_dir):
    """Test finding .xlsm file."""
    # Create a .xlsm file
    xlsm_file = temp_excel_dir / "test_file.xlsm"
    xlsm_file.write_text("test")

    result = _find_excel_file(temp_excel_dir, "test_file")
    assert result is not None
    assert result.name == "test_file.xlsm"


def test_find_excel_file_prefers_xlsx_over_xlsm(temp_excel_dir):
    """Test that .xlsx is preferred when both exist."""
    # Create both .xlsx and .xlsm files
    xlsx_file = temp_excel_dir / "test_file.xlsx"
    xlsm_file = temp_excel_dir / "test_file.xlsm"
    xlsx_file.write_text("test")
    xlsm_file.write_text("test")

    result = _find_excel_file(temp_excel_dir, "test_file")
    assert result is not None
    assert result.name == "test_file.xlsx"


def test_find_excel_file_with_glob_pattern(temp_excel_dir):
    """Test finding Excel file with glob pattern."""
    # Create a file with pattern
    xlsx_file = temp_excel_dir / "acoustic_file_legend.xlsx"
    xlsx_file.write_text("test")

    result = _find_excel_file(temp_excel_dir, "*acoustic_file_legend*")
    assert result is not None
    assert result.name == "acoustic_file_legend.xlsx"


def test_find_excel_file_not_found(temp_excel_dir):
    """Test that None is returned when file is not found."""
    result = _find_excel_file(temp_excel_dir, "nonexistent*")
    assert result is None


def test_find_excel_file_with_study_id_pattern(temp_excel_dir):
    """Test finding biomechanics file with study ID pattern."""
    # Create files matching the biomechanics pattern
    study_id = "1011"
    xlsx_file = temp_excel_dir / f"AOA{study_id}_Biomechanics_Full_Set.xlsx"
    xlsx_file.write_text("test")

    result = _find_excel_file(
        temp_excel_dir,
        f"AOA{study_id}_Biomechanics_Full_Set",
    )
    assert result is not None
    assert result.name == f"AOA{study_id}_Biomechanics_Full_Set.xlsx"


def test_find_excel_file_with_study_id_xlsm(temp_excel_dir):
    """Test finding biomechanics file with .xlsm extension."""
    # Create files matching the biomechanics pattern
    study_id = "1011"
    xlsm_file = temp_excel_dir / f"AOA{study_id}_Biomechanics_Full_Set.xlsm"
    xlsm_file.write_text("test")

    result = _find_excel_file(
        temp_excel_dir,
        f"AOA{study_id}_Biomechanics_Full_Set",
    )
    assert result is not None
    assert result.name == f"AOA{study_id}_Biomechanics_Full_Set.xlsm"
