# GPU Job Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Thay hàng đợi thread trong RAM bằng SQLite queue có backpressure và chạy đúng một Gwen worker process trên mỗi GPU cho Colab/Kaggle.

**Architecture:** Flask chỉ validate, enqueue và đọc trạng thái. Mỗi GPU có một subprocess với `CUDA_VISIBLE_DEVICES` riêng, một `TTSStudioService` riêng và xử lý tuần tự theo FIFO. SQLite chịu trách nhiệm claim nguyên tử, lease, retry hữu hạn và trạng thái bền trên cùng filesystem.

**Tech Stack:** Python stdlib (`sqlite3`, `subprocess`, `threading`, `fcntl` trên Linux), Flask, NumPy/SoundFile, `unittest`.

**Spec:** `docs/superpowers/specs/2026-08-24-gpu-job-queue-design.md`

## Global Constraints

- Colab dùng `TTS_GPU_IDS=0`; Kaggle dùng `0,1` chỉ khi runtime xác nhận có hai GPU.
- Một GPU chỉ có một worker và một inference đang chạy.
- Không thêm Redis, Celery, RQ hoặc dependency mới.
- Không chia chunk của cùng một request qua nhiều GPU.
- Queue tối đa 32 job chưa terminal; retry tối đa 2 lần claim.
- Mọi CUDA import/load trong worker xảy ra sau khi set `CUDA_VISIBLE_DEVICES`.
- `async=0` phải đi qua queue, không được gọi `studio.synthesize()` trực tiếp trong Flask process.
- Mỗi behavior mới phải có test đã quan sát fail trước khi sửa production code.

---

### Task 1: Submission contract và SQLite schema

**Files:**

- Create: `webapp/job_queue.py`
- Create: `tests/test_job_queue.py`
- Modify: `webapp/app.py:89-103`

**Interfaces:**

- Produces: `SynthesisSubmission.to_json() -> str` và `SynthesisSubmission.from_json(raw: str) -> SynthesisSubmission`.
- Produces: `SQLiteJobQueue(path: Path, max_pending: int = 32)`.
- Produces: `QueueFullError` và immutable `JobSnapshot`.

- [ ] **Step 1: Viết test fail cho round-trip submission**

```python
def test_submission_json_round_trip_preserves_path_and_preset() -> None:
    submission = SynthesisSubmission(
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
        preset_voice=None,
    )
    restored = SynthesisSubmission.from_json(submission.to_json())
    self.assertEqual(restored, submission)
```

- [ ] **Step 2: Chạy test và xác nhận fail vì contract chưa tồn tại**

Run: `uv run --isolated --with-requirements requirements.txt python -m unittest tests.test_job_queue.JobQueueTest.test_submission_json_round_trip_preserves_path_and_preset -v`

Expected: FAIL vì chưa import được `SynthesisSubmission` từ `webapp.job_queue`.

- [ ] **Step 3: Chuyển `SynthesisSubmission` khỏi `app.py` và thêm serializer literal**

`to_json()` phải chuyển `Path` thành chuỗi, pronunciation tuple thành list JSON, và preset dataclass thành dict. `from_json()` phải validate object shape trước khi dựng lại type; không dùng pickle.

- [ ] **Step 4: Viết test fail cho schema và enqueue/get snapshot**

```python
def test_enqueue_persists_job_across_queue_instances(self) -> None:
    queue = SQLiteJobQueue(self.db_path, max_pending=2)
    job_id = queue.enqueue(self.submission)
    reopened = SQLiteJobQueue(self.db_path, max_pending=2)
    snapshot = reopened.get_snapshot(job_id)
    self.assertEqual(snapshot.status, "queued")
    self.assertEqual(snapshot.submission.text, "Xin chào")
```

- [ ] **Step 5: Chạy test và xác nhận fail vì chưa có schema/DB write**

Run: `uv run --isolated --with-requirements requirements.txt python -m unittest tests.test_job_queue.JobQueueTest.test_enqueue_persists_job_across_queue_instances -v`

Expected: FAIL vì `SQLiteJobQueue.enqueue()` chưa tồn tại.

- [ ] **Step 6: Implement schema tối thiểu và connection lifecycle**

```python
class SQLiteJobQueue:
    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection
```

Tạo bảng/cột đúng spec, index `(status, created_at, id)`, commit rõ ràng và đóng connection bằng `contextlib.closing`.

- [ ] **Step 7: Chạy toàn bộ test Task 1**

Run: `uv run --isolated --with-requirements requirements.txt python -m unittest tests/test_job_queue.py -v`

Expected: PASS.

- [ ] **Step 8: Commit riêng Task 1**

```bash
git add webapp/job_queue.py webapp/app.py tests/test_job_queue.py
git commit -m "feat: add persistent TTS job queue"
```

---

### Task 2: Backpressure và atomic FIFO claim

**Files:**

- Modify: `webapp/job_queue.py`
- Modify: `tests/test_job_queue.py`

**Interfaces:**

- Produces: `enqueue(submission) -> str`, raise `QueueFullError` khi số job `queued|running` đạt `max_pending`.
- Produces: `claim_next(worker_id: str, gpu_index: int, now: float | None = None) -> JobSnapshot | None`.
- Produces: `queue_position(job_id: str) -> int | None` với vị trí bắt đầu từ 1.

- [ ] **Step 1: Viết test fail cho queue capacity**

```python
def test_enqueue_rejects_when_pending_capacity_is_full(self) -> None:
    queue = SQLiteJobQueue(self.db_path, max_pending=1)
    queue.enqueue(self.submission)
    with self.assertRaises(QueueFullError):
        queue.enqueue(self.submission)
```

- [ ] **Step 2: Chạy test và xác nhận job thứ hai vẫn được nhận**

Run: `uv run --isolated --with-requirements requirements.txt python -m unittest tests.test_job_queue.JobQueueTest.test_enqueue_rejects_when_pending_capacity_is_full -v`

Expected: FAIL vì chưa có backpressure.

- [ ] **Step 3: Implement capacity check trong cùng transaction enqueue**

Dùng `BEGIN IMMEDIATE`, đếm `status IN ('queued', 'running')`, raise trước `INSERT`, rollback khi đầy.

- [ ] **Step 4: Viết concurrency test fail cho duplicate claim**

```python
def test_two_claimers_cannot_claim_same_job(self) -> None:
    queue = SQLiteJobQueue(self.db_path)
    job_id = queue.enqueue(self.submission)
    with ThreadPoolExecutor(max_workers=2) as executor:
        claims = list(executor.map(lambda worker: queue.claim_next(worker, int(worker[-1])), ["w0", "w1"]))
    claimed_ids = [claim.id for claim in claims if claim is not None]
    self.assertEqual(claimed_ids, [job_id])
```

- [ ] **Step 5: Chạy test và xác nhận fail vì claim chưa nguyên tử/chưa tồn tại**

Run: `uv run --isolated --with-requirements requirements.txt python -m unittest tests.test_job_queue.JobQueueTest.test_two_claimers_cannot_claim_same_job -v`

Expected: FAIL trước khi có atomic claim.

- [ ] **Step 6: Implement FIFO claim bằng `BEGIN IMMEDIATE`**

Trong transaction: select `queued ORDER BY created_at, id LIMIT 1`, update đúng `id` và `status='queued'`, tăng `attempts`, set worker/GPU/lease; commit rồi đọc snapshot.

- [ ] **Step 7: Thêm test queue position và chạy file test**

Run: `uv run --isolated --with-requirements requirements.txt python -m unittest tests/test_job_queue.py -v`

Expected: PASS, kể cả claim đồng thời lặp 20 vòng.

- [ ] **Step 8: Commit riêng Task 2**

```bash
git add webapp/job_queue.py tests/test_job_queue.py
git commit -m "feat: bound and atomically claim TTS jobs"
```

---

### Task 3: Lease, terminal transitions và recovery hữu hạn

**Files:**

- Modify: `webapp/job_queue.py`
- Modify: `tests/test_job_queue.py`

**Interfaces:**

- Produces: `renew_lease(job_id, worker_id, now=None) -> bool`.
- Produces: `complete(job_id, worker_id, result) -> bool`.
- Produces: `fail(job_id, worker_id, error) -> bool`.
- Produces: `recover_expired(now=None, max_attempts=2) -> tuple[int, int]`, trả `(requeued, failed)`.
- Produces: `purge_terminal(retention_seconds, now=None) -> int`.

- [ ] **Step 1: Viết test fail cho owner-checked completion**

```python
def test_only_claim_owner_can_complete_job(self) -> None:
    job_id = self.queue.enqueue(self.submission)
    self.queue.claim_next("worker-0", 0, now=100.0)
    self.assertFalse(self.queue.complete(job_id, "worker-1", {"ok": True}))
    self.assertTrue(self.queue.complete(job_id, "worker-0", {"ok": True}))
```

- [ ] **Step 2: Viết test fail cho expired lease retry đúng một lần**

```python
def test_expired_job_is_requeued_once_then_failed(self) -> None:
    job_id = self.queue.enqueue(self.submission)
    self.queue.claim_next("worker-0", 0, now=100.0)
    self.assertEqual(self.queue.recover_expired(now=131.0, max_attempts=2), (1, 0))
    self.queue.claim_next("worker-1", 1, now=140.0)
    self.assertEqual(self.queue.recover_expired(now=171.0, max_attempts=2), (0, 1))
    self.assertEqual(self.queue.get_snapshot(job_id).status, "failed")
```

- [ ] **Step 3: Chạy hai test và xác nhận fail do thiếu transition/lease**

Run: `uv run --isolated --with-requirements requirements.txt python -m unittest tests.test_job_queue.JobQueueTest.test_only_claim_owner_can_complete_job tests.test_job_queue.JobQueueTest.test_expired_job_is_requeued_once_then_failed -v`

Expected: FAIL.

- [ ] **Step 4: Implement transition có điều kiện trong transaction**

Mọi update terminal phải có `WHERE id=? AND status='running' AND worker_id=?`. `recover_expired()` chỉ xét `lease_expires_at < now`; khi fail cuối ghi thông báo cố định, không đưa traceback/secret vào DB.

- [ ] **Step 5: Thêm purge test và chạy file test**

Run: `uv run --isolated --with-requirements requirements.txt python -m unittest tests/test_job_queue.py -v`

Expected: PASS.

- [ ] **Step 6: Commit riêng Task 3**

```bash
git add webapp/job_queue.py tests/test_job_queue.py
git commit -m "feat: recover leased TTS jobs"
```

---

### Task 4: Một sequential worker cho mỗi GPU

**Files:**

- Create: `webapp/tts_worker.py`
- Create: `tests/test_tts_worker.py`
- Modify: `webapp/app.py:523-555`

**Interfaces:**

- Produces: `run_worker(queue, execute, worker_id, gpu_index, stop_event, poll_seconds) -> None`.
- Produces: `SynthesisOutcome(payload: dict[str, Any], history_entry: dict[str, Any])`.
- Produces: `execute_submission(submission: SynthesisSubmission, *, studio: TTSStudioService) -> SynthesisOutcome`.
- Consumes: lease/transition API từ Task 3.

- [ ] **Step 1: Tách execution khỏi Flask request/global job manager bằng test fail**

```python
def test_execute_submission_uses_injected_studio(self) -> None:
    fake_studio = FakeStudio(result=self.result)
    outcome = execute_submission(self.submission, studio=fake_studio)
    self.assertEqual(outcome.payload["download_name"], self.result.output_path.name)
    self.assertEqual(fake_studio.calls, [self.submission])
```

- [ ] **Step 2: Chạy test và xác nhận fail vì execution còn đóng trong `app.py`**

Run: `uv run --isolated --with-requirements requirements.txt python -m unittest tests.test_tts_worker.TTSWorkerTest.test_execute_submission_uses_injected_studio -v`

Expected: FAIL.

- [ ] **Step 3: Di chuyển phần gọi synth/build payload sang hàm nhận dependency rõ ràng**

Worker không import Flask request context. Chuyển logic build payload/history entry thành helper thuần trong `tts_worker.py`; URL output phải giữ `/outputs/<filename>`. Worker ghi `outcome.payload` vào queue; `outcome.history_entry` được persist ở Task 6.

- [ ] **Step 4: Viết test fail chứng minh worker chạy tuần tự và cập nhật terminal state**

```python
def test_worker_completes_claimed_jobs_sequentially(self) -> None:
    first = self.queue.enqueue(self.submission)
    second = self.queue.enqueue(self.submission)
    execute = RecordingExecutor(stop_after=2)
    run_worker(self.queue, execute, "worker-0", 0, execute.stop_event, poll_seconds=0.0)
    self.assertEqual(execute.maximum_concurrency, 1)
    self.assertEqual(self.queue.get_snapshot(first).status, "completed")
    self.assertEqual(self.queue.get_snapshot(second).status, "completed")
```

- [ ] **Step 5: Chạy test và xác nhận fail trước khi có loop**

Run: `uv run --isolated --with-requirements requirements.txt python -m unittest tests.test_tts_worker.TTSWorkerTest.test_worker_completes_claimed_jobs_sequentially -v`

Expected: FAIL.

- [ ] **Step 6: Implement worker loop và heartbeat thread không chạm CUDA**

CLI phải parse GPU ID, set `os.environ['CUDA_VISIBLE_DEVICES']` rồi mới import/khởi tạo `TTSStudioService`. Đọc và clamp `TTS_QUEUE_POLL_MS`, `TTS_JOB_LEASE_SECONDS`, `TTS_JOB_MAX_ATTEMPTS`; worker gọi `recover_expired()` trước mỗi claim. Heartbeat thread chỉ gọi `renew_lease`; luôn stop/join thread trong `finally`.

- [ ] **Step 7: Thêm test error path và chạy worker tests**

Run: `uv run --isolated --with-requirements requirements.txt python -m unittest tests/test_tts_worker.py -v`

Expected: PASS; exception đã sanitize thành `failed` và worker tiếp tục claim job sau.

- [ ] **Step 8: Commit riêng Task 4**

```bash
git add webapp/tts_worker.py webapp/app.py tests/test_tts_worker.py
git commit -m "feat: run one TTS worker per GPU process"
```

---

### Task 5: Chuyển Flask API hoàn toàn sang persistent queue

**Files:**

- Modify: `webapp/app.py:105-206, 738-835`
- Modify: `tests/test_api_errors.py:227-300`
- Modify: `tests/test_job_queue.py`

**Interfaces:**

- Replaces: in-memory `TTSJobManager` bằng adapter dùng `SQLiteJobQueue`.
- Produces: `wait_for_terminal(job_id, timeout_seconds) -> JobSnapshot`.
- Preserves: async `202 + job_id`, status names và completed result shape.
- Adds: HTTP `429` với `Retry-After` khi queue đầy.

- [ ] **Step 1: Viết API test fail cho queue full**

```python
def test_generate_async_returns_429_when_queue_is_full(self) -> None:
    with patch("webapp.app.tts_job_manager.enqueue", side_effect=QueueFullError("Hàng đợi đã đầy.")):
        response = self.client.post("/api/tts/generate", data=self.valid_preset_form, content_type="multipart/form-data")
    self.assertEqual(response.status_code, 429)
    self.assertEqual(response.headers["Retry-After"], "5")
    self.assertFalse(response.get_json()["ok"])
```

- [ ] **Step 2: Viết test fail chứng minh `async=0` không gọi synth trực tiếp**

```python
def test_sync_generation_waits_on_queue_instead_of_running_studio(self) -> None:
    with patch("webapp.app.tts_job_manager.enqueue", return_value="job-1") as enqueue:
        with patch("webapp.app.tts_job_manager.wait_for_terminal", return_value=self.completed_snapshot):
            with patch("webapp.app._execute_synthesis_submission") as direct_execute:
                response = self.client.post("/api/tts/generate", data={**self.valid_preset_form, "async": "0"})
    enqueue.assert_called_once()
    direct_execute.assert_not_called()
    self.assertEqual(response.status_code, 200)
```

- [ ] **Step 3: Chạy test và xác nhận route hiện tại bypass queue**

Run: `uv run --isolated --with-requirements requirements.txt python -m unittest tests.test_api_errors.ApiErrorHandlingTest.test_generate_async_returns_429_when_queue_is_full tests.test_api_errors.ApiErrorHandlingTest.test_sync_generation_waits_on_queue_instead_of_running_studio -v`

Expected: FAIL.

- [ ] **Step 4: Implement route behavior và timeout fallback**

Khởi tạo queue từ `TTS_QUEUE_DB` và `TTS_QUEUE_MAX_PENDING`. `async=0` enqueue rồi poll DB tối đa `TTS_SYNC_WAIT_SECONDS`. Nếu hết thời gian nhưng job chưa terminal, trả cùng response `202` như async để client có thể tiếp tục poll. Không cancel job khi HTTP client ngắt kết nối.

- [ ] **Step 5: Bổ sung status fields không phá compatibility**

Queued response có `queue_position`; running response có `worker_id`, `gpu_index`, `attempts`. Không xóa field hiện có.

- [ ] **Step 6: Chạy API tests và toàn suite**

Run: `uv run --isolated --with-requirements requirements.txt python -m unittest tests/test_api_errors.py tests/test_job_queue.py -v`

Expected: PASS.

- [ ] **Step 7: Commit riêng Task 5**

```bash
git add webapp/app.py tests/test_api_errors.py tests/test_job_queue.py
git commit -m "feat: route TTS requests through bounded queue"
```

---

### Task 6: Tách storage và khóa history liên process

**Files:**

- Create: `webapp/storage.py`
- Modify: `webapp/app.py:330-407`
- Modify: `webapp/tts_worker.py`
- Create: `tests/test_history_storage.py`

**Interfaces:**

- Produces: `append_tts_history(path, entry, max_items=50) -> None` và `load_tts_history(path) -> list[dict[str, Any]]` trong `webapp/storage.py`.
- Produces: `_storage_file_lock(lock_path)` dùng `fcntl.flock` trên `webapp/data/.storage.lock`.
- Preserves: JSON history shape, tối đa 50 entry và API `/api/history`.

- [ ] **Step 1: Viết multiprocessing test fail chứng minh append phải tôn trọng OS lock**

Parent giữ `fcntl.LOCK_EX` trên lock file tạm, sau đó spawn child gọi `append_tts_history()`. Sau 200 ms, assert child vẫn còn sống vì đang chờ lock; release lock, join child và assert entry được ghi. Implementation hiện tại chỉ có `threading.Lock` nên child sẽ kết thúc trước assertion này.

- [ ] **Step 2: Chạy test và xác nhận thiếu entry hoặc parse race với implementation hiện tại**

Run: `uv run --isolated --with-requirements requirements.txt python -m unittest tests/test_history_storage.py -v`

Expected: FAIL vì child không bị OS file lock chặn.

- [ ] **Step 3: Implement lock Linux bao quanh toàn bộ read-modify-write**

Mở lock file, `fcntl.flock(fd, LOCK_EX)`, giữ cả read-modify-write, sau đó unlock trong `finally`. `storage.py` giữ thêm `threading.Lock` để tránh thread trong cùng process chen vào. `app.py` dùng helper mới cho API history; worker append `SynthesisOutcome.history_entry` trước khi chuyển job sang `completed`.

- [ ] **Step 4: Chạy race test lặp 10 lần và toàn API history tests**

Run: `for run in $(seq 1 10); do uv run --isolated --with-requirements requirements.txt python -m unittest tests/test_history_storage.py -q || exit 1; done`

Expected: 10/10 PASS.

- [ ] **Step 5: Commit riêng Task 6**

```bash
git add webapp/storage.py webapp/app.py webapp/tts_worker.py tests/test_history_storage.py
git commit -m "fix: serialize TTS history across workers"
```

---

### Task 7: GPU process launcher và notebook profiles

**Files:**

- Create: `webapp/serve.py`
- Create: `tests/test_serve.py`
- Modify: `colab_tts_cloudflare.ipynb:42-70, 623-639`
- Modify: `kaggle_tts_cloudflare.ipynb:42-75, 570-586`

**Interfaces:**

- Produces: `parse_gpu_ids(raw: str) -> list[int]`.
- Produces: `build_worker_env(base_env, gpu_id) -> dict[str, str]`.
- Produces CLI: `python -m webapp.serve --host 0.0.0.0 --port 8386`.

- [ ] **Step 1: Viết test fail cho GPU ID validation**

```python
def test_parse_gpu_ids_deduplicates_and_rejects_negative_values(self) -> None:
    self.assertEqual(parse_gpu_ids("0,1,1"), [0, 1])
    with self.assertRaises(ValueError):
        parse_gpu_ids("0,-1")
```

- [ ] **Step 2: Viết test fail cho process isolation environment**

```python
def test_build_worker_env_exposes_only_assigned_gpu(self) -> None:
    env = build_worker_env({"PATH": "/usr/bin"}, 1)
    self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")
    self.assertEqual(env["TTS_WORKER_GPU_INDEX"], "1")
```

- [ ] **Step 3: Chạy tests và xác nhận launcher chưa tồn tại**

Run: `uv run --isolated --with-requirements requirements.txt python -m unittest tests/test_serve.py -v`

Expected: FAIL.

- [ ] **Step 4: Implement launcher không khởi tạo CUDA ở parent**

Launcher dùng `subprocess.Popen([sys.executable, '-m', 'webapp.tts_worker', ...], env=worker_env)` cho từng GPU rồi chạy Flask. Trong `finally`, gửi `SIGTERM`, chờ timeout hữu hạn và kill đúng PID còn sống. Không dùng `fork` sau khi import torch.

- [ ] **Step 5: Sửa profile notebook**

Colab set `TTS_GPU_IDS=0`. Kaggle chạy `torch.cuda.device_count()` trong process notebook, set `0,1` nếu count >= 2, ngược lại `0`. Cả hai đổi launch command sang `python -m webapp.serve`; PID file tiếp tục trỏ tới process supervisor.

- [ ] **Step 6: Kiểm tra notebook JSON và source cell**

Run: `uv run --isolated --with-requirements requirements.txt python -m json.tool colab_tts_cloudflare.ipynb >/dev/null && uv run --isolated --with-requirements requirements.txt python -m json.tool kaggle_tts_cloudflare.ipynb >/dev/null`

Expected: exit 0.

- [ ] **Step 7: Chạy launcher unit tests**

Run: `uv run --isolated --with-requirements requirements.txt python -m unittest tests/test_serve.py -v`

Expected: PASS; subprocess test dùng fake worker command, không cần CUDA.

- [ ] **Step 8: Commit riêng Task 7**

```bash
git add webapp/serve.py tests/test_serve.py colab_tts_cloudflare.ipynb kaggle_tts_cloudflare.ipynb
git commit -m "feat: launch one TTS worker per GPU"
```

---

### Task 8: UI observability, docs và benchmark harness

**Files:**

- Modify: `webapp/templates/studio_engine.html:1234-1257`
- Modify: `README.md:44-64, 346-365, 674-692`
- Modify: `README.en.md:44-64, 346-365, 674-692`
- Create: `scripts/benchmark_tts_queue.py`
- Create: `tests/test_queue_benchmark_helpers.py`

**Interfaces:**

- UI consumes optional `queue_position`, `gpu_index`, `attempts`.
- Benchmark CLI consumes base URL, concurrency và request count; outputs JSON summary gồm completed/failed, latency p50/p95 và throughput.

- [ ] **Step 1: Cập nhật UI bằng optional fields**

Queued message hiển thị `Đang chờ, vị trí N` khi có `queue_position`; running message hiển thị `Đang sinh trên GPU N` khi có `gpu_index`. Nếu server cũ không trả field mới, giữ wording hiện tại.

- [ ] **Step 2: Viết test fail cho percentile/summary helper của benchmark**

```python
def test_summarize_results_reports_latency_and_throughput(self) -> None:
    summary = summarize_results([1.0, 2.0, 3.0, 4.0], completed=4, failed=0, wall_seconds=4.0)
    self.assertEqual(summary["latency_p50_seconds"], 2.5)
    self.assertEqual(summary["throughput_jobs_per_second"], 1.0)
```

- [ ] **Step 3: Chạy test và xác nhận helper chưa tồn tại**

Run: `uv run --isolated --with-requirements requirements.txt python -m unittest tests/test_queue_benchmark_helpers.py -v`

Expected: FAIL.

- [ ] **Step 4: Implement benchmark bằng stdlib**

Dùng `urllib.request`, `ThreadPoolExecutor` ở client để submit đồng thời và poll mỗi job. Không đọc credential, không tự chạy benchmark tới public tunnel; người vận hành truyền `--base-url` rõ ràng.

- [ ] **Step 5: Đồng bộ tài liệu**

README phải mô tả SQLite queue, giới hạn 32, một process/GPU, profile Colab/Kaggle, retry hữu hạn, `429`, queue DB không backup và giới hạn chưa có auth/cancel running.

- [ ] **Step 6: Chạy toàn bộ verification local**

Run: `uv run --isolated --with-requirements requirements.txt python -m unittest discover -s tests -v`

Expected: tất cả test PASS, không cần GPU.

Run: `python -m py_compile webapp/app.py webapp/job_queue.py webapp/tts_worker.py webapp/serve.py scripts/benchmark_tts_queue.py`

Expected: exit 0.

Run: `git diff --check`

Expected: exit 0.

- [ ] **Step 7: Chạy smoke/benchmark Colab**

Set `TTS_GPU_IDS=0`, submit lần lượt 1, 4 và 8 job. Ghi queue wait, inference time, total wall time, peak VRAM từ `nvidia-smi`, số completed/failed. Xác nhận không bao giờ có hai job running trên GPU 0.

- [ ] **Step 8: Chạy A/B benchmark Kaggle**

Chạy cùng corpus với `TTS_GPU_IDS=0`, sau đó restart sạch với `TTS_GPU_IDS=0,1`. So sánh throughput, p50/p95 và peak VRAM từng GPU. Chỉ kết luận mức tăng sau khi có số đo.

- [ ] **Step 9: Commit riêng Task 8**

```bash
git add webapp/templates/studio_engine.html README.md README.en.md scripts/benchmark_tts_queue.py tests/test_queue_benchmark_helpers.py
git commit -m "docs: document and benchmark GPU job queue"
```
