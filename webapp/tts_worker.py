from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Any, Callable
from urllib.parse import quote

from webapp.job_queue import SQLiteJobQueue, SynthesisSubmission
from webapp.storage import append_tts_history
from webapp.tts_service import TTSError


@dataclass(frozen=True, slots=True)
class SynthesisOutcome:
    payload: dict[str, Any]
    history_entry: dict[str, Any]


def execute_submission(submission: SynthesisSubmission, *, studio: Any) -> SynthesisOutcome:
    result = studio.synthesize(
        engine_id=submission.engine_id,
        text=submission.text,
        reference_audio=submission.reference_path,
        reference_text=submission.reference_text,
        speed=submission.speed,
        remove_silence=submission.remove_silence,
        seed=submission.seed,
        model_key=submission.model_key,
        custom_model=submission.custom_model,
        gwen_generation_config=submission.gwen_generation_config,
        pronunciation_overrides=submission.pronunciation_overrides,
    )
    notes = list(result.notes)
    if submission.preset_voice is not None:
        notes.insert(0, f"Đã dùng preset voice Gwen: {submission.preset_voice.name}.")
    payload = {
        "ok": True,
        "engine": result.engine_id,
        "engine_label": result.engine_label,
        "model_key": result.model_key,
        "model_label": result.model_label,
        "preset_voice_id": submission.preset_voice.id if submission.preset_voice is not None else "",
        "preset_voice_label": submission.preset_voice.name if submission.preset_voice is not None else "",
        "audio_url": f"/outputs/{quote(result.output_path.name)}",
        "download_name": result.output_path.name,
        "sample_rate": result.sample_rate,
        "duration_seconds": round(result.duration_seconds, 2),
        "inference_seconds": round(result.inference_seconds, 2),
        "chunk_count": result.chunk_count,
        "reference_text_used": result.reference_text_used,
        "seed": result.seed,
        "input_text": submission.text,
        "notes": notes,
        "storage_path": str(result.output_path.relative_to(studio.root)),
    }
    history_entry = {
        "id": uuid.uuid4().hex,
        "input_text": submission.text,
        "engine": payload["engine"],
        "voice_id": payload["preset_voice_id"],
        "voice_label": payload["preset_voice_label"] or payload["engine_label"],
        "model_label": payload["model_label"],
        "audio_url": payload["audio_url"],
        "download_name": payload["download_name"],
        "duration_seconds": payload["duration_seconds"],
        "inference_seconds": payload["inference_seconds"],
        "sample_rate": payload["sample_rate"],
        "chunk_count": payload["chunk_count"],
        "created_at": datetime.now().astimezone().isoformat(),
    }
    return SynthesisOutcome(payload=payload, history_entry=history_entry)


def run_worker(
    queue: SQLiteJobQueue,
    execute: Callable[[SynthesisSubmission], SynthesisOutcome],
    worker_id: str,
    gpu_index: int,
    stop_event: threading.Event,
    poll_seconds: float,
    max_attempts: int = 2,
    history_path: Path | None = None,
) -> None:
    while not stop_event.is_set():
        queue.recover_expired(max_attempts=max_attempts)
        job = queue.claim_next(worker_id, gpu_index)
        if job is None:
            if poll_seconds:
                stop_event.wait(poll_seconds)
            else:
                sleep(0)
            continue

        heartbeat_stop = threading.Event()
        heartbeat = threading.Thread(
            target=_renew_lease,
            args=(queue, job.id, worker_id, heartbeat_stop),
            daemon=True,
        )
        heartbeat.start()
        try:
            outcome = execute(job.submission)
            if history_path is not None:
                history_entry = dict(outcome.history_entry)
                history_entry["id"] = job.id
                append_tts_history(history_path, history_entry)
            queue.complete(job.id, worker_id, outcome.payload)
        except TTSError as exc:
            queue.fail(job.id, worker_id, str(exc))
        except Exception as exc:
            logging.error("TTS worker failed job %s (%s)", job.id, type(exc).__name__)
            queue.fail(job.id, worker_id, "Lỗi không mong muốn khi synthesize.")
        finally:
            heartbeat_stop.set()
            heartbeat.join()


def _renew_lease(
    queue: SQLiteJobQueue,
    job_id: str,
    worker_id: str,
    stop_event: threading.Event,
) -> None:
    interval = max(0.1, queue.lease_seconds / 3)
    while not stop_event.wait(interval):
        if not queue.renew_lease(job_id, worker_id):
            return


def _bounded_env_number(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential TTS worker for one physical GPU")
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--worker-id", required=True)
    args = parser.parse_args()
    if args.gpu_id < 0:
        parser.error("--gpu-id phải không âm")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    from webapp.tts_service import TTSStudioService

    root = Path(__file__).resolve().parent.parent
    queue_path = Path(os.environ.get("TTS_QUEUE_DB", root / "webapp" / "runtime" / "tts_jobs.sqlite3"))
    lease_seconds = _bounded_env_number("TTS_JOB_LEASE_SECONDS", 30.0, 3.0, 3600.0)
    poll_seconds = _bounded_env_number("TTS_QUEUE_POLL_MS", 200.0, 10.0, 5000.0) / 1000.0
    max_attempts = int(_bounded_env_number("TTS_JOB_MAX_ATTEMPTS", 2, 1, 10))
    queue = SQLiteJobQueue(queue_path, lease_seconds=lease_seconds)
    studio = TTSStudioService(root)
    stop_event = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    run_worker(
        queue,
        lambda submission: execute_submission(submission, studio=studio),
        args.worker_id,
        args.gpu_id,
        stop_event,
        poll_seconds,
        max_attempts,
        root / "webapp" / "data" / "tts_history.json",
    )


if __name__ == "__main__":
    main()
