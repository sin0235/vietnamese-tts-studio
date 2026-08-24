from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


@contextmanager
def _storage_file_lock(path: Path, *, exclusive: bool) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / ".storage.lock"
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _read_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def load_tts_history(path: Path) -> list[dict[str, Any]]:
    with _storage_file_lock(path, exclusive=False):
        return _read_history(path)


def append_tts_history(path: Path, entry: dict[str, Any], max_items: int = 50) -> None:
    with _storage_file_lock(path, exclusive=True):
        history = _read_history(path)
        entry_id = entry.get("id")
        if entry_id is not None:
            history = [item for item in history if item.get("id") != entry_id]
        history.insert(0, entry)

        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                delete=False,
            ) as handle:
                json.dump(history[:max_items], handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
                temp_path = Path(handle.name)
            os.replace(temp_path, path)
            temp_path = None
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
