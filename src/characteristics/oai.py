from __future__ import annotations

from pathlib import Path
from typing import Literal, Union

from ._utils import (
    attach_normalized_ids,
    coerce_choice_column,
    coerce_numeric_column,
    filter_by_ids,
    normalize_id_sequence,
    read_sheet,
)

_TFM_COLUMNS = [
    "OST MFC (0-3+)",
    "OST MTP (0-3+)",
    "OST LFC (0-3+)",
    "OST LTP (0-3+)",
    "JSN M (0-3+)",
    "JSN L (0-3+)",
    "MT Attrition (0=absent, 1=present)",
    "MT Sclerosis (0=absent, 1=present)",
    "LF Sclerosis (0=absent, 1=present)",
]

_PFM_COLUMNS = [
    "OST MP (0-3+)",
    "OST MT (0-3+)",
    "OST LP (0-3+)",
    "OST LT (0-3+)",
    "JSN M (0-3+)",
    "JSN L (0-3+)",
    "MP Attrition (0=absent, 1=present)",
    "MT Attrition (0=absent, 1=present)",
    "MP Sclerosis (0=absent, 1=present)",
    "MT Sclerosis (0=absent, 1=present)",
    "LP Attrition (0=absent, 1=present)",
    "LT Attrition (0=absent, 1=present)",
    "LP Sclerosis (0=absent, 1=present)",
    "LT Sclerosis (0=absent, 1=present)",
]


def _load_oai_sheet(
    excel_path: Path,
    sheet_name: str,
    columns: list[str],
    ids: list[int],
    knees: Literal["left", "right", "both"],
) -> tuple[dict[int, dict[str, int]], dict[int, set[str]]]:
    df = read_sheet(excel_path, sheet_name)
    missing = [c for c in ["Knee", *columns] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in OAI sheet '{sheet_name}': {missing}")

    df = attach_normalized_ids(df)
    df = filter_by_ids(df, ids)

    df["Knee"] = coerce_choice_column(df["Knee"], "Knee", {"left", "right"})

    if knees != "both":
        df = df[df["Knee"] == knees]

    for col in columns:
        df[col] = coerce_numeric_column(df[col], col, integer=True)

    data: dict[int, dict[str, int]] = {}
    seen_knees: dict[int, set[str]] = {}

    for _, row in df.iterrows():
        study_id = int(row["study_id"])
        knee = row["Knee"]
        seen_knees.setdefault(study_id, set()).add(knee)
        per_id = data.setdefault(study_id, {})
        for col in columns:
            per_id[f"{col}_{knee}"] = int(row[col])

    return data, seen_knees


def import_oai(
    excel_path: Union[str, Path],
    ids: Union[int, str, list[Union[int, str]], None] = None,
    *,
    knees: Literal["left", "right", "both"] = "both",
) -> dict[int, dict[str, int]]:
    """Load Osteoarthritis Initiative imaging scores for both compartments.

    Pulls tibiofemoral (TFM) and patellofemoral (PFM) scores from the "TFM OAI"
    and "PFM OAI" sheets. Column names are suffixed with the knee side so left
    and right data stay separate. When `knees="both"`, the importer confirms
    both knees are present for each requested Study ID.
    """
    path = Path(excel_path)
    id_list = normalize_id_sequence(ids) if ids is not None else []

    tfm_data, tfm_seen = _load_oai_sheet(path, "TFM OAI", _TFM_COLUMNS, id_list, knees)
    pfm_data, pfm_seen = _load_oai_sheet(path, "PFM OAI", _PFM_COLUMNS, id_list, knees)

    combined: dict[int, dict[str, int]] = {}
    for source in (tfm_data, pfm_data):
        for study_id, payload in source.items():
            combined.setdefault(study_id, {}).update(payload)

    if knees == "both":
        expected_ids = id_list if id_list else sorted(set(tfm_seen) | set(pfm_seen))
        for study_id in expected_ids:
            tfm_knees = tfm_seen.get(study_id, set())
            pfm_knees = pfm_seen.get(study_id, set())
            for knee in ("left", "right"):
                if knee not in tfm_knees:
                    raise ValueError(f"Missing TFM OAI data for {knee} knee (Study ID {study_id})")
                if knee not in pfm_knees:
                    raise ValueError(f"Missing PFM OAI data for {knee} knee (Study ID {study_id})")

    return combined
