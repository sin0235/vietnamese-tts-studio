from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from webapp.job_queue import SQLiteJobQueue, SynthesisSubmission
from webapp.tts_service import SynthesisResult
from webapp.tts_worker import SynthesisOutcome, execute_submission, run_worker


class FakeStudio:
    def __init__(self, root: Path, result: SynthesisResult) -> None:
        self.root = root
        self.result = result
        self.calls: list[dict] = []

    def synthesize(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class RecordingExecutor:
    def __init__(self, stop_event: threading.Event, *, fail_first: bool = False) -> None:
        self.stop_event = stop_event
        self.fail_first = fail_first
        self.calls = 0
        self.active = 0
        self.maximum_concurrency = 0

    def __call__(self, submission: SynthesisSubmission) -> SynthesisOutcome:
        self.calls += 1
        self.active += 1
        self.maximum_concurrency = max(self.maximum_concurrency, self.active)
        try:
            if self.fail_first and self.calls == 1:
                raise RuntimeError("chi tiết nội bộ")
            return SynthesisOutcome(payload={"ok": True, "text": submission.text}, history_entry={})
        finally:
            self.active -= 1
            if self.calls == 2:
                self.stop_event.set()


class TTSWorkerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.queue = SQLiteJobQueue(self.root / "queue.sqlite3", lease_seconds=3.0)
        self.submission = SynthesisSubmission(
            engine_id="gwen",
            text="Xin chào",
            reference_path=self.root / "reference.wav",
            reference_text="Xin chào",
            speed=1.0,
            remove_silence=False,
            seed=7,
            model_key="default",
            custom_model="",
            gwen_generation_config=None,
            pronunciation_overrides=[],
        )

    def test_execute_submission_uses_injected_studio(self) -> None:
        output_path = self.root / "runtime" / "outputs" / "done.wav"
        result = SynthesisResult(
            engine_id="gwen",
            engine_label="Gwen-TTS",
            model_key="default",
            model_label="Default",
            output_path=output_path,
            sample_rate=24000,
            duration_seconds=2.345,
            inference_seconds=1.234,
            chunk_count=1,
            reference_text_used=True,
            seed=7,
            notes=[],
        )
        studio = FakeStudio(self.root, result)

        outcome = execute_submission(self.submission, studio=studio)

        self.assertEqual(outcome.payload["download_name"], "done.wav")
        self.assertEqual(outcome.payload["storage_path"], "runtime/outputs/done.wav")
        self.assertEqual(studio.calls[0]["text"], "Xin chào")

    def test_worker_completes_claimed_jobs_sequentially(self) -> None:
        first = self.queue.enqueue(self.submission)
        second = self.queue.enqueue(self.submission)
        stop_event = threading.Event()
        execute = RecordingExecutor(stop_event)

        run_worker(self.queue, execute, "worker-0", 0, stop_event, poll_seconds=0.0)

        self.assertEqual(execute.maximum_concurrency, 1)
        self.assertEqual(self.queue.get_snapshot(first).status, "completed")
        self.assertEqual(self.queue.get_snapshot(second).status, "completed")

    def test_worker_fails_one_job_and_continues_with_next(self) -> None:
        first = self.queue.enqueue(self.submission)
        second = self.queue.enqueue(self.submission)
        stop_event = threading.Event()
        execute = RecordingExecutor(stop_event, fail_first=True)

        run_worker(self.queue, execute, "worker-0", 0, stop_event, poll_seconds=0.0)

        first_snapshot = self.queue.get_snapshot(first)
        self.assertEqual(first_snapshot.status, "failed")
        self.assertEqual(first_snapshot.error, "Lỗi không mong muốn khi synthesize.")
        self.assertEqual(self.queue.get_snapshot(second).status, "completed")


if __name__ == "__main__":
    unittest.main()
