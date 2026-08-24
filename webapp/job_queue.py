from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from webapp.tts_service import PresetVoice


class QueueFullError(RuntimeError):
    pass


@dataclass(slots=True)
class SynthesisSubmission:
    engine_id: str
    text: str
    reference_path: Path
    reference_text: str
    speed: float
    remove_silence: bool
    seed: int | None
    model_key: str
    custom_model: str
    gwen_generation_config: dict[str, Any] | None
    pronunciation_overrides: list[tuple[str, str]]
    preset_voice: PresetVoice | None = None

    def to_json(self) -> str:
        payload = asdict(self)
        payload["reference_path"] = str(self.reference_path)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str) -> SynthesisSubmission:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("Submission JSON phải là một object.")
        preset_payload = payload.get("preset_voice")
        if preset_payload is not None and not isinstance(preset_payload, dict):
            raise ValueError("preset_voice phải là object hoặc null.")
        overrides = payload.get("pronunciation_overrides")
        if not isinstance(overrides, list):
            raise ValueError("pronunciation_overrides phải là một list.")
        payload["reference_path"] = Path(payload["reference_path"])
        payload["pronunciation_overrides"] = [tuple(item) for item in overrides]
        payload["preset_voice"] = PresetVoice(**preset_payload) if preset_payload is not None else None
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class JobSnapshot:
    id: str
    status: str
    submission: SynthesisSubmission
    result: dict[str, Any] | None
    error: str | None
    worker_id: str | None
    gpu_index: int | None
    attempts: int
    created_at: float
    updated_at: float
    started_at: float | None
    finished_at: float | None
    lease_expires_at: float | None


class SQLiteJobQueue:
    def __init__(self, path: Path, max_pending: int = 32, lease_seconds: float = 30.0) -> None:
        self.path = Path(path)
        self.max_pending = max_pending
        self.lease_seconds = lease_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'completed', 'failed')),
                    submission_json TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    worker_id TEXT,
                    gpu_index INTEGER,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL,
                    lease_expires_at REAL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS jobs_status_created_idx ON jobs(status, created_at, id)"
            )

    def enqueue(self, submission: SynthesisSubmission) -> str:
        now = time.time()
        job_id = uuid.uuid4().hex
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                pending = connection.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'running')"
                ).fetchone()[0]
                if pending >= self.max_pending:
                    raise QueueFullError("Hàng đợi TTS đã đầy.")
                connection.execute(
                    """
                    INSERT INTO jobs (id, status, submission_json, created_at, updated_at)
                    VALUES (?, 'queued', ?, ?, ?)
                    """,
                    (job_id, submission.to_json(), now, now),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return job_id

    def get_snapshot(self, job_id: str) -> JobSnapshot | None:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._to_snapshot(row) if row is not None else None

    def claim_next(
        self,
        worker_id: str,
        gpu_index: int,
        now: float | None = None,
    ) -> JobSnapshot | None:
        claimed_at = time.time() if now is None else now
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT id FROM jobs WHERE status = 'queued' ORDER BY created_at, id LIMIT 1"
                ).fetchone()
                if row is None:
                    connection.commit()
                    return None
                job_id = row["id"]
                connection.execute(
                    """
                    UPDATE jobs
                    SET status = 'running', worker_id = ?, gpu_index = ?, attempts = attempts + 1,
                        started_at = ?, updated_at = ?, lease_expires_at = ?
                    WHERE id = ? AND status = 'queued'
                    """,
                    (
                        worker_id,
                        gpu_index,
                        claimed_at,
                        claimed_at,
                        claimed_at + self.lease_seconds,
                        job_id,
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return self.get_snapshot(job_id)

    def queue_position(self, job_id: str) -> int | None:
        with closing(self._connect()) as connection:
            target = connection.execute(
                "SELECT created_at, id FROM jobs WHERE id = ? AND status = 'queued'",
                (job_id,),
            ).fetchone()
            if target is None:
                return None
            row = connection.execute(
                """
                SELECT COUNT(*) FROM jobs
                WHERE status = 'queued'
                  AND (created_at < ? OR (created_at = ? AND id <= ?))
                """,
                (target["created_at"], target["created_at"], target["id"]),
            ).fetchone()
        return int(row[0])

    def renew_lease(self, job_id: str, worker_id: str, now: float | None = None) -> bool:
        renewed_at = time.time() if now is None else now
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                """
                UPDATE jobs SET lease_expires_at = ?, updated_at = ?
                WHERE id = ? AND status = 'running' AND worker_id = ?
                """,
                (renewed_at + self.lease_seconds, renewed_at, job_id, worker_id),
            )
        return cursor.rowcount == 1

    def complete(self, job_id: str, worker_id: str, result: dict[str, Any]) -> bool:
        finished_at = time.time()
        result_json = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET status = 'completed', result_json = ?, error = NULL,
                    updated_at = ?, finished_at = ?, lease_expires_at = NULL
                WHERE id = ? AND status = 'running' AND worker_id = ?
                """,
                (result_json, finished_at, finished_at, job_id, worker_id),
            )
        return cursor.rowcount == 1

    def fail(self, job_id: str, worker_id: str, error: str) -> bool:
        finished_at = time.time()
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET status = 'failed', error = ?, result_json = NULL,
                    updated_at = ?, finished_at = ?, lease_expires_at = NULL
                WHERE id = ? AND status = 'running' AND worker_id = ?
                """,
                (error, finished_at, finished_at, job_id, worker_id),
            )
        return cursor.rowcount == 1

    def recover_expired(
        self,
        now: float | None = None,
        max_attempts: int = 2,
    ) -> tuple[int, int]:
        recovered_at = time.time() if now is None else now
        requeued = 0
        failed = 0
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                expired = connection.execute(
                    """
                    SELECT id, attempts FROM jobs
                    WHERE status = 'running' AND lease_expires_at < ?
                    """,
                    (recovered_at,),
                ).fetchall()
                for row in expired:
                    if row["attempts"] < max_attempts:
                        connection.execute(
                            """
                            UPDATE jobs
                            SET status = 'queued', worker_id = NULL, gpu_index = NULL,
                                updated_at = ?, started_at = NULL, lease_expires_at = NULL
                            WHERE id = ? AND status = 'running'
                            """,
                            (recovered_at, row["id"]),
                        )
                        requeued += 1
                    else:
                        connection.execute(
                            """
                            UPDATE jobs
                            SET status = 'failed', error = ?, updated_at = ?,
                                finished_at = ?, lease_expires_at = NULL
                            WHERE id = ? AND status = 'running'
                            """,
                            (
                                "Worker mất lease sau số lần thử tối đa.",
                                recovered_at,
                                recovered_at,
                                row["id"],
                            ),
                        )
                        failed += 1
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return requeued, failed

    def purge_terminal(self, retention_seconds: float, now: float | None = None) -> int:
        purge_at = time.time() if now is None else now
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                """
                DELETE FROM jobs
                WHERE status IN ('completed', 'failed') AND updated_at < ?
                """,
                (purge_at - retention_seconds,),
            )
        return cursor.rowcount

    @staticmethod
    def _to_snapshot(row: sqlite3.Row) -> JobSnapshot:
        return JobSnapshot(
            id=row["id"],
            status=row["status"],
            submission=SynthesisSubmission.from_json(row["submission_json"]),
            result=json.loads(row["result_json"]) if row["result_json"] is not None else None,
            error=row["error"],
            worker_id=row["worker_id"],
            gpu_index=row["gpu_index"],
            attempts=row["attempts"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            lease_expires_at=row["lease_expires_at"],
        )
