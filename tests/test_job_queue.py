from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from webapp.job_queue import QueueFullError, SQLiteJobQueue, SynthesisSubmission
from webapp.tts_service import PresetVoice


class JobQueueTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.db_path = Path(self.temp_dir.name) / "tts_jobs.sqlite3"
        self.preset = PresetVoice(
            id="voice-1",
            name="Giọng mẫu",
            avatar="G",
            style="Rõ ràng",
            audio_filename="voice.wav",
            reference_text="Xin chào",
        )
        self.submission = SynthesisSubmission(
            engine_id="gwen",
            text="Xin chào",
            reference_path=Path("/tmp/reference.wav"),
            reference_text="Xin chào",
            speed=1.0,
            remove_silence=False,
            seed=7,
            model_key="default",
            custom_model="",
            gwen_generation_config={"max_new_tokens": 1024},
            pronunciation_overrides=[("AI", "ây ai")],
            preset_voice=self.preset,
        )

    def test_submission_json_round_trip_preserves_path_and_preset(self) -> None:
        restored = SynthesisSubmission.from_json(self.submission.to_json())

        self.assertEqual(restored, self.submission)
        self.assertIsInstance(restored.reference_path, Path)
        self.assertIsInstance(restored.preset_voice, PresetVoice)

    def test_enqueue_persists_job_across_queue_instances(self) -> None:
        queue = SQLiteJobQueue(self.db_path, max_pending=2)
        job_id = queue.enqueue(self.submission)

        reopened = SQLiteJobQueue(self.db_path, max_pending=2)
        snapshot = reopened.get_snapshot(job_id)

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.status, "queued")
        self.assertEqual(snapshot.submission.text, "Xin chào")

    def test_enqueue_rejects_when_pending_capacity_is_full(self) -> None:
        queue = SQLiteJobQueue(self.db_path, max_pending=1)
        queue.enqueue(self.submission)

        with self.assertRaises(QueueFullError):
            queue.enqueue(self.submission)

    def test_two_claimers_cannot_claim_same_job(self) -> None:
        queue = SQLiteJobQueue(self.db_path)
        job_id = queue.enqueue(self.submission)

        with ThreadPoolExecutor(max_workers=2) as executor:
            claims = list(
                executor.map(
                    lambda worker: queue.claim_next(worker, int(worker[-1])),
                    ["worker-0", "worker-1"],
                )
            )

        self.assertEqual([claim.id for claim in claims if claim is not None], [job_id])

    def test_queue_position_is_one_based_and_only_available_while_queued(self) -> None:
        queue = SQLiteJobQueue(self.db_path)
        first = queue.enqueue(self.submission)
        second = queue.enqueue(self.submission)

        self.assertEqual(queue.queue_position(first), 1)
        self.assertEqual(queue.queue_position(second), 2)
        queue.claim_next("worker-0", 0)
        self.assertIsNone(queue.queue_position(first))
        self.assertEqual(queue.queue_position(second), 1)

    def test_only_claim_owner_can_complete_job(self) -> None:
        queue = SQLiteJobQueue(self.db_path)
        job_id = queue.enqueue(self.submission)
        queue.claim_next("worker-0", 0, now=100.0)

        self.assertFalse(queue.complete(job_id, "worker-1", {"ok": True}))
        self.assertTrue(queue.complete(job_id, "worker-0", {"ok": True}))
        snapshot = queue.get_snapshot(job_id)
        assert snapshot is not None
        self.assertEqual(snapshot.status, "completed")
        self.assertEqual(snapshot.result, {"ok": True})

    def test_expired_job_is_requeued_once_then_failed(self) -> None:
        queue = SQLiteJobQueue(self.db_path, lease_seconds=30.0)
        job_id = queue.enqueue(self.submission)
        queue.claim_next("worker-0", 0, now=100.0)

        self.assertEqual(queue.recover_expired(now=131.0, max_attempts=2), (1, 0))
        queue.claim_next("worker-1", 1, now=140.0)
        self.assertEqual(queue.recover_expired(now=171.0, max_attempts=2), (0, 1))

        snapshot = queue.get_snapshot(job_id)
        assert snapshot is not None
        self.assertEqual(snapshot.status, "failed")
        self.assertEqual(snapshot.error, "Worker mất lease sau số lần thử tối đa.")

    def test_renew_lease_requires_current_owner(self) -> None:
        queue = SQLiteJobQueue(self.db_path, lease_seconds=30.0)
        job_id = queue.enqueue(self.submission)
        queue.claim_next("worker-0", 0, now=100.0)

        self.assertFalse(queue.renew_lease(job_id, "worker-1", now=110.0))
        self.assertTrue(queue.renew_lease(job_id, "worker-0", now=110.0))
        snapshot = queue.get_snapshot(job_id)
        assert snapshot is not None
        self.assertEqual(snapshot.lease_expires_at, 140.0)

    def test_purge_terminal_does_not_remove_pending_jobs(self) -> None:
        queue = SQLiteJobQueue(self.db_path)
        completed = queue.enqueue(self.submission)
        pending = queue.enqueue(self.submission)
        queue.claim_next("worker-0", 0)
        queue.complete(completed, "worker-0", {"ok": True})

        self.assertEqual(queue.purge_terminal(retention_seconds=0, now=10**12), 1)
        self.assertIsNone(queue.get_snapshot(completed))
        self.assertIsNotNone(queue.get_snapshot(pending))


if __name__ == "__main__":
    unittest.main()
