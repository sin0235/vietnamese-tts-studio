from __future__ import annotations

import fcntl
import multiprocessing
import tempfile
import time
import unittest
from pathlib import Path

from webapp.storage import append_tts_history, load_tts_history


def _append_in_child(path: str) -> None:
    append_tts_history(Path(path), {"id": "child", "text": "Xin chào"})


class HistoryStorageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name) / "tts_history.json"

    def test_append_waits_for_interprocess_file_lock(self) -> None:
        lock_path = self.path.parent / ".storage.lock"
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            process = multiprocessing.get_context("spawn").Process(
                target=_append_in_child,
                args=(str(self.path),),
            )
            process.start()
            time.sleep(0.2)
            self.assertTrue(process.is_alive())
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

        process.join(timeout=5)
        self.assertEqual(process.exitcode, 0)
        self.assertEqual(load_tts_history(self.path), [{"id": "child", "text": "Xin chào"}])

    def test_append_deduplicates_id_and_enforces_limit(self) -> None:
        append_tts_history(self.path, {"id": "same", "value": 1}, max_items=2)
        append_tts_history(self.path, {"id": "other", "value": 2}, max_items=2)
        append_tts_history(self.path, {"id": "same", "value": 3}, max_items=2)

        self.assertEqual(
            load_tts_history(self.path),
            [{"id": "same", "value": 3}, {"id": "other", "value": 2}],
        )


if __name__ == "__main__":
    unittest.main()
