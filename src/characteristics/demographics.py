from __future__ import annotations

from pathlib import Path
from typing import Union

from ._utils import (
    attach_normalized_ids,
    coerce_choice_column,
    coerce_numeric_column,
    coerce_yes_no_column,
    filter_by_ids,
    normalize_id_sequence,
    read_sheet,
)

_REQUIRED_COLUMNS = ["Age (years)", "BMI", "Gender", "Knee Pain"]


def import_demographics(
    excel_path: Union[str, Path],
    ids: Union[int, str, list[Union[int, str]], None] = None,
) -> dict[int, dict[str, object]]:
    """Load basic participant demographics from the "Demographics" sheet.

    Extracts age, BMI, gender, and a Yes/No flag for knee pain for the
    requested Study IDs. Gender values are normalized to "M"/"F" and knee pain
    is returned as a boolean for easy downstream use.
    """
    path = Path(excel_path)
    id_list = normalize_id_sequence(ids) if ids is not None else []

    df = read_sheet(path, "Demographics")
    missing = [col for col in _REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in Demographics sheet: {missing}")

    df = attach_normalized_ids(df)
    df = filter_by_ids(df, id_list)

    # Coerce numeric columns
    df["Age (years)"] = coerce_numeric_column(df["Age (years)"], "Age (years)")
    df["BMI"] = coerce_numeric_column(df["BMI"], "BMI")
    df["Knee Pain"] = coerce_yes_no_column(df["Knee Pain"], "Knee Pain")

    # Normalize gender
    gender_series = coerce_choice_column(
        df["Gender"],
        "Gender",
        {"m", "f", "male", "female"},
    )
    df["Gender"] = gender_series.map(lambda g: "M" if g.startswith("m") else "F")

    results: dict[int, dict[str, object]] = {}
    for _, row in df.iterrows():
        study_id = int(row["study_id"])
        results[study_id] = {
            "Age (years)": float(row["Age (years)"]),
            "BMI": float(row["BMI"]),
            "Gender": row["Gender"],
            "Knee Pain": bool(row["Knee Pain"]),
        }

    return results
