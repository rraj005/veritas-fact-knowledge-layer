"""On-disk prompt→response cache keyed by sha256."""

from __future__ import annotations

import hashlib
from pathlib import Path


def cache_key(model: str, system: str, user: str) -> str:
    """Return a sha256 hex digest uniquely identifying this (model, system, user) triple."""
    payload = f"{model}\x00{system}\x00{user}".encode()
    return hashlib.sha256(payload).hexdigest()


class DiskCache:
    """Simple file-per-entry cache stored under *dir*."""

    def __init__(self, dir: str | Path) -> None:
        self._dir = Path(dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.txt"

    def get(self, key: str) -> str | None:
        p = self._path(key)
        if p.exists():
            return p.read_text(encoding="utf-8")
        return None

    def set(self, key: str, val: str) -> None:
        self._path(key).write_text(val, encoding="utf-8")
