"""Project configuration helpers for environment-driven defaults."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _default_env_path() -> Path:
    return Path(__file__).resolve().parents[1] / ".env.local"


def load_env_file(env_path: Optional[Path] = None) -> None:
    """Load environment variables from a .env file if present.

    This is intentionally minimal and does not override already-set variables.
    """
    path = env_path or _default_env_path()
    if not path.exists():
        return

    for line in path.read_text().splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"").strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_data_root(default: Optional[Path] = None) -> Optional[Path]:
    """Return the configured participant data root, if available."""
    raw = os.environ.get("AE_DATA_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    return default


def get_database_url() -> Optional[str]:
    """Return the configured Postgres connection string, if available."""
    return os.environ.get("AE_DATABASE_URL") or os.environ.get("DATABASE_URL")
