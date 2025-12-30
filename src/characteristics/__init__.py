from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Union

from ._utils import normalize_id_sequence
from .demographics import import_demographics
from .kl import import_kl
from .koos import import_koos
from .oai import import_oai
from .varus_thurst import import_varus_thurst

__all__ = [
    "import_demographics",
    "import_koos",
    "import_varus_thurst",
    "import_oai",
    "import_kl",
    "load_characteristics",
]


def load_characteristics(
    excel_path: Union[str, Path],
    study_ids: Union[int, str, Iterable[Union[int, str]]],
    *,
    oai_knees: str = "both",
) -> Dict[int, Dict[str, object]]:
    """Load all participant characteristics for the requested Study IDs.

    Aggregates demographics, KOOS, varus thrust, OAI (tibiofemoral/patellofemoral),
    and KL grades (tibiofemoral/patellofemoral) into a single dictionary keyed by
    Study ID. Use `oai_knees` to restrict to left, right, or both knees.
    """
    path = Path(excel_path)
    ids = normalize_id_sequence(study_ids)
    combined: Dict[int, Dict[str, object]] = {study_id: {} for study_id in ids}

    importers = [
        lambda: import_demographics(path, ids),
        lambda: import_koos(path, ids),
        lambda: import_varus_thurst(path, ids),
        lambda: import_oai(path, ids, knees=oai_knees),
        lambda: import_kl(path, ids),
    ]

    for importer in importers:
        data = importer()
        for study_id, values in data.items():
            combined.setdefault(study_id, {}).update(values)

    return combined
