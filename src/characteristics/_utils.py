from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, List

import pandas as pd


def normalize_study_id(value: Any) -> int:
    """Normalize a Study ID to an integer.

    Accepts values like 1001 or "AOA1001" (case-insensitive) and strips the
    optional "AOA" prefix. Raises ValueError for invalid formats.
    """
    if pd.isna(value):
        raise ValueError("Study ID is missing")

    text = str(value).strip().upper()
    if text.startswith("AOA"):
        text = text[3:]

    if not re.fullmatch(r"\d{4}", text):
        raise ValueError(f"Invalid Study ID format: {value}")

    return int(text)


def normalize_id_sequence(ids: Any) -> List[int]:
    """Normalize a single id or an iterable of ids to a list of ints."""
    if isinstance(ids, (str, int)):
        ids_iterable: Iterable[Any] = [ids]
    else:
        ids_iterable = ids

    normalized: List[int] = []
    for item in ids_iterable:
        normalized.append(normalize_study_id(item))
    return normalized


def read_sheet(excel_path: Path, sheet_name: str) -> pd.DataFrame:
    """Load a sheet from Excel and ensure Study ID column exists."""
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    if "Study ID" not in df.columns:
        raise ValueError(f"Study ID column missing in sheet '{sheet_name}'")
    return df


def attach_normalized_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with a normalized 'study_id' column."""
    df = df.copy()
    df["study_id"] = df["Study ID"].apply(normalize_study_id)
    return df


def filter_by_ids(df: pd.DataFrame, ids: list[int]) -> pd.DataFrame:
    """Filter rows to the requested Study IDs."""
    if not ids:
        return df
    allowed = set(ids)
    return df[df["study_id"].isin(allowed)].copy()


def coerce_numeric_column(series: pd.Series, name: str, *, integer: bool = False) -> pd.Series:
    """Coerce a series to numeric, raising on errors."""
    numeric = pd.to_numeric(series, errors="raise")
    return numeric.astype(int if integer else float)


def coerce_choice_column(series: pd.Series, name: str, choices: set[str]) -> pd.Series:
    """Normalize string choices to lowercase and validate membership."""
    normalized = series.astype(str).str.strip().str.lower()
    invalid = normalized[~normalized.isin(choices)]
    if not invalid.empty:
        raise ValueError(f"Invalid values for {name}: {sorted(set(invalid))}")
    return normalized


def coerce_yes_no_column(series: pd.Series, name: str) -> pd.Series:
    """Normalize yes/no style values to booleans.

    Accepts common variants (yes/no, y/n, true/false, 1/0). Returns a boolean
    Series. Raises ValueError on unexpected values.
    """
    normalized = series.astype(str).str.strip().str.lower()
    truthy = {"yes", "y", "true", "1"}
    falsy = {"no", "n", "false", "0"}
    allowed = truthy | falsy
    invalid = normalized[~normalized.isin(allowed)]
    if not invalid.empty:
        raise ValueError(f"Invalid values for {name}: {sorted(set(invalid))}")
    return normalized.isin(truthy)
