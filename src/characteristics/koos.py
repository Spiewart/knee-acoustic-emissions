from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

from ._utils import (
    attach_normalized_ids,
    coerce_numeric_column,
    filter_by_ids,
    normalize_id_sequence,
    read_sheet,
)

_KOOS_BASE_COLUMNS: List[str] = (
    [f"s{i}" for i in range(1, 8)]
    + [f"p{i}" for i in range(1, 10)]
    + [f"a{i}" for i in range(1, 18)]
    + [f"sp{i}" for i in range(1, 6)]
    + [f"q{i}" for i in range(1, 5)]
    + ["Symptoms", "Pain", "ADL", "Sports/Rec", "QOL"]
)


def _load_knee(
    excel_path: Path,
    sheet_name: str,
    suffix: str,
    ids: List[int],
) -> Dict[int, Dict[str, float]]:
    df = read_sheet(excel_path, sheet_name)

    missing = [c for c in _KOOS_BASE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in KOOS sheet '{sheet_name}': {missing}")

    df = attach_normalized_ids(df)
    df = filter_by_ids(df, ids)

    # Coerce all KOOS items to numeric
    for col in _KOOS_BASE_COLUMNS:
        df[col] = coerce_numeric_column(df[col], col)

    result: Dict[int, Dict[str, float]] = {}
    for _, row in df.iterrows():
        study_id = int(row["study_id"])
        per_id = result.setdefault(study_id, {})
        for col in _KOOS_BASE_COLUMNS:
            per_id[f"{col}_{suffix}"] = float(row[col])
    return result


def import_koos(
    excel_path: Union[str, Path],
    ids: Union[int, str, List[Union[int, str]], None] = None,
) -> Dict[int, Dict[str, float]]:
    """Load KOOS questionnaire scores for each knee.

    Reads the "KOOS R Knee" and "KOOS L Knee" sheets and tags every question
    with "_r_knee" or "_l_knee" so right/left results stay distinct.
    """
    path = Path(excel_path)
    id_list = normalize_id_sequence(ids) if ids is not None else []

    combined: Dict[int, Dict[str, float]] = {}

    for sheet_name, suffix in [("KOOS R Knee", "r_knee"), ("KOOS L Knee", "l_knee")]:
        knee_data = _load_knee(path, sheet_name, suffix, id_list)
        for study_id, values in knee_data.items():
            combined.setdefault(study_id, {}).update(values)

    return combined
