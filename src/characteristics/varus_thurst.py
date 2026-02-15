from __future__ import annotations

from pathlib import Path
from typing import Union

from ._utils import (
    attach_normalized_ids,
    coerce_choice_column,
    filter_by_ids,
    normalize_id_sequence,
    read_sheet,
)


def import_varus_thurst(
    excel_path: Union[str, Path],
    ids: Union[int, str, list[Union[int, str]], None] = None,
) -> dict[int, dict[str, bool]]:
    """Load varus thrust findings for each knee.

    Reads the "Varus Thrust" sheet (columns: Study ID, Right Knee, Left Knee)
    where values are Yes/No style ("y"/"n") and returns booleans for easier
    use downstream.
    """
    path = Path(excel_path)
    id_list = normalize_id_sequence(ids) if ids is not None else []

    df = read_sheet(path, "Varus Thrust")
    missing_cols = [c for c in ["Right Knee", "Left Knee"] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in Varus Thrust sheet: {missing_cols}")

    df = attach_normalized_ids(df)
    df = filter_by_ids(df, id_list)

    df["Right Knee"] = coerce_choice_column(df["Right Knee"], "Right Knee", {"y", "n"})
    df["Left Knee"] = coerce_choice_column(df["Left Knee"], "Left Knee", {"y", "n"})

    results: dict[int, dict[str, bool]] = {}
    for _, row in df.iterrows():
        study_id = int(row["study_id"])
        results[study_id] = {
            "Varus Thrust Right": row["Right Knee"] == "y",
            "Varus Thrust Left": row["Left Knee"] == "y",
        }

    return results
