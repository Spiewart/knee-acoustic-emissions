from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal, Union, cast

from ._utils import normalize_id_sequence
from .demographics import import_demographics
from .kl import import_kl
from .koos import import_koos
from .oai import import_oai
from .varus_thurst import import_varus_thurst

__all__ = [
    "import_demographics",
    "import_kl",
    "import_koos",
    "import_oai",
    "import_varus_thurst",
    "load_characteristics",
]


def load_characteristics(
    excel_path: Union[str, Path],
    study_ids: Union[int, str, Iterable[Union[int, str]]],
    *,
    oai_knees: str = "both",
) -> dict[int, dict[str, object]]:
    """Load all participant characteristics for the requested Study IDs.

    Aggregates demographics, KOOS, varus thrust, OAI (tibiofemoral/patellofemoral),
    and KL grades (tibiofemoral/patellofemoral) into a single dictionary keyed by
    Study ID. Use `oai_knees` to restrict to left, right, or both knees.
    """
    path = Path(excel_path)
    ids = normalize_id_sequence(study_ids)
    combined: dict[int, dict[str, object]] = {study_id: {} for study_id in ids}
    knees = cast(Literal["left", "right", "both"], oai_knees)

    importers: list[Callable[[], dict[int, dict[str, object]]]] = [
        lambda: dict(import_demographics(path, ids)),  # type: ignore[arg-type]
        lambda: dict(import_koos(path, ids)),  # type: ignore[arg-type]
        lambda: dict(import_varus_thurst(path, ids)),  # type: ignore[arg-type]
        lambda: dict(import_oai(path, ids, knees=knees)),  # type: ignore[arg-type]
        lambda: dict(import_kl(path, ids)),  # type: ignore[arg-type]
    ]

    for importer in importers:
        data = importer()
        for study_id, values in data.items():
            combined.setdefault(study_id, {}).update(values)

    return combined
