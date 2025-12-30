import pandas as pd
import pytest

from src.characteristics import (
    import_demographics,
    import_kl,
    import_koos,
    import_oai,
    import_varus_thurst,
    load_characteristics,
)


@pytest.fixture
def sample_characteristics_workbook(tmp_path):
    path = tmp_path / "cohort_chars.xlsx"

    demo_df = pd.DataFrame(
        {
            "Study ID": [1001, "AOA1002"],
            "Age (years)": [39, 55],
            "BMI": [27.3, 31.1],
            "Gender": ["M", "female"],
            "Knee Pain": ["Yes", "No"],
        }
    )

    koos_cols = (
        [f"s{i}" for i in range(1, 8)]
        + [f"p{i}" for i in range(1, 10)]
        + [f"a{i}" for i in range(1, 18)]
        + [f"sp{i}" for i in range(1, 6)]
        + [f"q{i}" for i in range(1, 5)]
        + ["Symptoms", "Pain", "ADL", "Sports/Rec", "QOL"]
    )

    def make_koos_row(base_value: int):
        data = {col: base_value + idx for idx, col in enumerate(koos_cols)}
        data["Study ID"] = 1001
        return data

    def make_koos_row_two(base_value: int):
        data = {col: base_value + idx for idx, col in enumerate(koos_cols)}
        data["Study ID"] = "AOA1002"
        return data

    koos_r = pd.DataFrame([make_koos_row(1), make_koos_row_two(2)])
    koos_l = pd.DataFrame([make_koos_row(10), make_koos_row_two(20)])

    varus_df = pd.DataFrame(
        {
            "Study ID": [1001, "AOA1002"],
            "Right Knee": ["y", "n"],
            "Left Knee": ["n", "y"],
        }
    )

    tfm_df = pd.DataFrame(
        {
            "Study ID": [1001, 1001, "AOA1002"],
            "Knee": ["Right", "Left", "Right"],
            "OST MFC (0-3+)": [1, 2, 0],
            "OST MTP (0-3+)": [0, 1, 0],
            "OST LFC (0-3+)": [1, 2, 0],
            "OST LTP (0-3+)": [0, 1, 0],
            "JSN M (0-3+)": [1, 1, 0],
            "JSN L (0-3+)": [0, 1, 0],
            "MT Attrition (0=absent, 1=present)": [0, 1, 0],
            "MT Sclerosis (0=absent, 1=present)": [1, 1, 0],
            "LF Sclerosis (0=absent, 1=present)": [0, 1, 0],
        }
    )

    pfm_df = pd.DataFrame(
        {
            "Study ID": [1001, 1001, "AOA1002"],
            "Knee": ["Right", "Left", "Right"],
            "OST MP (0-3+)": [1, 1, 0],
            "OST MT (0-3+)": [0, 1, 0],
            "OST LP (0-3+)": [1, 2, 0],
            "OST LT (0-3+)": [0, 1, 0],
            "JSN M (0-3+)": [1, 1, 0],
            "JSN L (0-3+)": [0, 1, 0],
            "MP Attrition (0=absent, 1=present)": [0, 1, 0],
            "MT Attrition (0=absent, 1=present)": [1, 1, 0],
            "MP Sclerosis (0=absent, 1=present)": [0, 1, 0],
            "MT Sclerosis (0=absent, 1=present)": [1, 1, 0],
            "LP Attrition (0=absent, 1=present)": [0, 1, 0],
            "LT Attrition (0=absent, 1=present)": [1, 1, 0],
            "LP Sclerosis (0=absent, 1=present)": [0, 1, 0],
            "LT Sclerosis (0=absent, 1=present)": [1, 1, 0],
        }
    )

    tfm_kl_df = pd.DataFrame(
        {
            "Study ID": [1001, 1001, "AOA1002"],
            "Knee": ["Left", "Right", "Right"],
            "Grade": [2, 3, 1],
        }
    )

    pfm_kl_df = pd.DataFrame(
        {
            "Study ID": [1001, 1001, "AOA1002"],
            "Knee": ["Left", "Right", "Right"],
            "Grade": [1, 2, 0],
        }
    )

    with pd.ExcelWriter(path) as writer:
        demo_df.to_excel(writer, sheet_name="Demographics", index=False)
        koos_r.to_excel(writer, sheet_name="KOOS R Knee", index=False)
        koos_l.to_excel(writer, sheet_name="KOOS L Knee", index=False)
        varus_df.to_excel(writer, sheet_name="Varus Thrust", index=False)
        tfm_df.to_excel(writer, sheet_name="TFM OAI", index=False)
        pfm_df.to_excel(writer, sheet_name="PFM OAI", index=False)
        tfm_kl_df.to_excel(writer, sheet_name="TFM KL", index=False)
        pfm_kl_df.to_excel(writer, sheet_name="PFM KL", index=False)

    return path


def test_demographics_handles_aoa_prefix(sample_characteristics_workbook):
    data = import_demographics(sample_characteristics_workbook, ids=[1001, "AOA1002"])
    assert set(data.keys()) == {1001, 1002}
    assert data[1001]["Gender"] == "M"
    assert data[1002]["Gender"] == "F"
    assert data[1001]["Knee Pain"] is True
    assert data[1002]["Knee Pain"] is False


def test_koos_applies_knee_suffixes(sample_characteristics_workbook):
    data = import_koos(sample_characteristics_workbook, ids=[1001])
    assert "s1_r_knee" in data[1001]
    assert "q4_l_knee" in data[1001]


def test_varus_thurst_returns_booleans(sample_characteristics_workbook):
    data = import_varus_thurst(sample_characteristics_workbook, ids=[1001, 1002])
    assert data[1001]["Varus Thrust Right"] is True
    assert data[1001]["Varus Thrust Left"] is False


def test_oai_requires_both_knees_when_requested(sample_characteristics_workbook):
    with pytest.raises(ValueError):
        import_oai(sample_characteristics_workbook, ids=[1002], knees="both")

    right_only = import_oai(sample_characteristics_workbook, ids=[1002], knees="right")
    assert "OST MFC (0-3+)_right" in right_only[1002]


def test_kl_imports_per_knee(sample_characteristics_workbook):
    data = import_kl(sample_characteristics_workbook, ids=[1001])
    assert data[1001]["kl_tfm_grade_left"] == 2
    assert data[1001]["kl_tfm_grade_right"] == 3
    assert data[1001]["kl_pfm_grade_left"] == 1
    assert data[1001]["kl_pfm_grade_right"] == 2


def test_master_loader_combines_all(sample_characteristics_workbook):
    combined = load_characteristics(sample_characteristics_workbook, [1001], oai_knees="both")
    entry = combined[1001]
    assert entry["Age (years)"] == 39.0
    assert "s1_r_knee" in entry
    assert "Varus Thrust Right" in entry
    assert "OST MFC (0-3+)_right" in entry
    assert "kl_tfm_grade_right" in entry
    assert "kl_pfm_grade_right" in entry
