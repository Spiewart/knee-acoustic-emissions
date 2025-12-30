from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

from ._utils import (
    attach_normalized_ids,
    coerce_choice_column,
    coerce_numeric_column,
    filter_by_ids,
    normalize_id_sequence,
    read_sheet,
)


def _load_kl_sheet(
    excel_path: Path,
    sheet_name: str,
    prefix: str,
    ids: List[int],
) -> Dict[int, Dict[str, int]]:
    df = read_sheet(excel_path, sheet_name)
    missing_cols = [c for c in ["Knee", "Grade"] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in {sheet_name} sheet: {missing_cols}")

    df = attach_normalized_ids(df)
    df = filter_by_ids(df, ids)

    df["Knee"] = coerce_choice_column(df["Knee"], "Knee", {"left", "right"})
    df["Grade"] = coerce_numeric_column(df["Grade"], "Grade", integer=True)

    results: Dict[int, Dict[str, int]] = {}
    for _, row in df.iterrows():
        study_id = int(row["study_id"])
        knee = row["Knee"]
        grade = int(row["Grade"])
        key = f"{prefix}_{'left' if knee == 'left' else 'right'}"
        results.setdefault(study_id, {})[key] = grade
    return results


def import_kl(
    excel_path: Union[str, Path],
    ids: Union[int, str, List[Union[int, str]], None] = None,
) -> Dict[int, Dict[str, int]]:
    """Load Kellgren–Lawrence grades for tibiofemoral and patellofemoral joints.

    Reads grades from both the tibiofemoral ("TFM KL") and patellofemoral
    ("PFM KL") sheets and returns knee-specific keys like
    ``kl_tfm_grade_right`` and ``kl_pfm_grade_left``.
    """

    path = Path(excel_path)
    id_list = normalize_id_sequence(ids) if ids is not None else []

    combined: Dict[int, Dict[str, int]] = {}
    for sheet_name, prefix in [("TFM KL", "kl_tfm_grade"), ("PFM KL", "kl_pfm_grade")]:
        sheet_data = _load_kl_sheet(path, sheet_name, prefix, id_list)
        for study_id, payload in sheet_data.items():
            combined.setdefault(study_id, {}).update(payload)

    return combined
