# Thiết kế hàng đợi TTS đa GPU cho Colab và Kaggle

## Mục tiêu

Nâng luồng job TTS từ một `ThreadPoolExecutor` trong process thành hàng đợi có giới hạn và có trạng thái bền trên đĩa, đồng thời chạy đúng một worker process trên mỗi GPU khả dụng:

- Colab: một GPU khoảng 16 GB, một worker TTS;
- Kaggle: tối đa hai GPU khoảng 16 GB/GPU, hai worker TTS độc lập;
- mỗi worker xử lý tuần tự một job trên GPU của nó;
- nhiều job khác nhau có thể chạy song song trên hai GPU;
- không chia các chunk của cùng một job sang nhiều GPU.

## Hiện trạng đã xác minh

- `TTSJobManager` trong `webapp/app.py` dùng `ThreadPoolExecutor(max_workers=1)`.
- Queue nội bộ của executor không có giới hạn nhận job.
- Job và trạng thái chỉ nằm trong `_jobs`, nên mất khi Flask restart.
- `TTSStudioService` là singleton toàn process; `_loaded_models` và model Gwen được dùng chung.
- Lock `gwen` chỉ bao quanh `_load_gwen()`, không bao quanh inference.
- Gwen luôn load vào `device_map="cuda:0"`.
- Route `async=0` chạy inference ngay trên request thread, có thể vượt qua hàng đợi.
- `tts_history.json` chỉ được bảo vệ bằng `threading.Lock`, không an toàn khi hai process cùng ghi.
- Colab/Kaggle hiện khởi chạy trực tiếp `from webapp.app import app; app.run(...)` và không có worker process riêng.

## Các phương án

### 1. Tăng `ThreadPoolExecutor(max_workers=2)`

Không chọn. Hai thread dùng cùng model Gwen và cùng `cuda:0`; source hiện tại không chứng minh `generate_voice_clone()` thread-safe. Trên Colab, hai inference còn cạnh tranh cùng 16 GB VRAM.

### 2. Chia một model hoặc một job qua hai GPU

Không chọn. Đây là model/tensor parallelism hoặc chunk fan-out, làm phức tạp prompt reuse, thứ tự chunk, lỗi từng phần và ghép audio. Không có bằng chứng local hoặc upstream cho thấy Gwen runtime hiện tại hỗ trợ cách này.

### 3. Một worker process trên mỗi GPU, chia song song theo job

Chọn. Mỗi worker có CUDA context, model cache và `TTSStudioService` riêng. Launcher đặt `CUDA_VISIBLE_DEVICES` cho từng process; do đó worker vẫn dùng `cuda:0` bên trong nhưng ánh xạ tới GPU vật lý được giao.

PyTorch yêu cầu CUDA subprocess dùng start method sạch như `spawn` hoặc `forkserver`, không khởi tạo CUDA rồi `fork`. Thiết kế dùng subprocess mới với environment riêng, không truyền CUDA tensor hay model giữa các process.

## Kiến trúc đích

```text
Browser/API
    |
    v
Flask coordinator (không load Gwen)
    |
    v
SQLite bounded queue
    |                       |
    v                       v
worker-gpu-0             worker-gpu-1
CUDA_VISIBLE_DEVICES=0   CUDA_VISIBLE_DEVICES=1
Gwen cuda:0              Gwen cuda:0
```

### Coordinator

- Validate request và lưu reference như hiện tại.
- Serialize `SynthesisSubmission` thành JSON.
- Ghi job `queued` vào SQLite nếu queue chưa đầy.
- `async=1` trả `202` và `job_id` như hiện tại.
- `async=0` cũng enqueue, sau đó chờ trạng thái terminal trong giới hạn cấu hình; không chạy CUDA trong Flask process.
- Poll status từ SQLite thay vì dict trong RAM.

### SQLite queue

Database mặc định: `webapp/runtime/tts_jobs.sqlite3`. File này không được sync sang Drive/Kaggle backup vì queued/running job không còn hợp lệ khi GPU session bị hủy; lịch sử hoàn tất vẫn dùng cơ chế backup hiện tại.

Schema `tts_jobs`:

| Cột | Kiểu | Ý nghĩa |
| --- | --- | --- |
| `id` | `TEXT PRIMARY KEY` | UUID job |
| `status` | `TEXT` | `queued`, `running`, `completed`, `failed` |
| `submission_json` | `TEXT NOT NULL` | Input đã serialize |
| `result_json` | `TEXT` | Payload thành công |
| `error` | `TEXT` | Lỗi đã sanitize |
| `worker_id` | `TEXT` | Worker đang giữ lease |
| `gpu_index` | `INTEGER` | GPU vật lý do launcher giao |
| `attempts` | `INTEGER` | Số lần claim, tối đa 2 |
| `created_at` | `REAL` | Unix timestamp |
| `updated_at` | `REAL` | Unix timestamp |
| `started_at` | `REAL` | Thời điểm bắt đầu |
| `finished_at` | `REAL` | Thời điểm kết thúc |
| `lease_expires_at` | `REAL` | Hạn lease của worker |

Mỗi thao tác mở connection riêng, đặt `busy_timeout`, bật WAL và đóng connection sau commit. Claim job chạy trong `BEGIN IMMEDIATE`: chọn FIFO một job `queued`, rồi update có điều kiện sang `running` trước khi commit. Hai worker không thể claim cùng một job.

Queue mặc định nhận tối đa 32 job chưa terminal. Khi đầy, API trả HTTP `429`, giữ JSON error shape và thêm `Retry-After`.

### Worker

- Một worker process tương ứng một GPU vật lý.
- Set `CUDA_VISIBLE_DEVICES` trước khi import module có thể load PyTorch/Qwen.
- Tạo một `TTSStudioService` và reuse model trong suốt vòng đời worker.
- Poll/claim FIFO, chạy một job tại một thời điểm.
- Gia hạn lease mỗi 5 giây bằng một thread chỉ làm SQLite I/O; thread này không chạm CUDA.
- Ghi `completed` hoặc `failed` trong transaction.
- Job có lease hết hạn được requeue nếu `attempts < 2`; sau đó chuyển `failed` để tránh retry vô hạn.

### GPU profile

- `TTS_GPU_IDS=0` là mặc định an toàn và profile Colab.
- Notebook Kaggle xác minh `torch.cuda.device_count()` trong process notebook; dùng `TTS_GPU_IDS=0,1` khi có hai GPU, ngược lại fallback `0`.
- Không tự tăng số worker vượt quá danh sách GPU.
- `TTS_WORKERS_PER_GPU` cố định là 1 trong phạm vi này.

### Đồng bộ file và history

- Output filename hiện có UUID nên hai process không đụng nhau.
- `tts_history.json` phải dùng file lock liên process trên Linux bao quanh toàn bộ read-modify-write; `threading.Lock` tiếp tục bảo vệ thread trong cùng process.
- User settings vẫn do coordinator ghi.
- Queue DB nằm ngoài các thư mục notebook đang backup.

## API và UI

Giữ các status hiện tại: `queued`, `running`, `completed`, `failed`.

Status response bổ sung các field không phá compatibility:

- `queue_position` khi đang queued;
- `worker_id` và `gpu_index` khi running;
- `attempts` cho chẩn đoán retry.

Studio tiếp tục poll 1,5 giây và hiển thị vị trí queue nếu có. Không thêm cancel running trong đợt này.

## Cấu hình

| Biến | Mặc định | Mô tả |
| --- | --- | --- |
| `TTS_GPU_IDS` | `0` | Danh sách GPU vật lý, ví dụ `0,1` |
| `TTS_QUEUE_DB` | `webapp/runtime/tts_jobs.sqlite3` | Queue SQLite |
| `TTS_QUEUE_MAX_PENDING` | `32` | Số job chưa terminal tối đa |
| `TTS_QUEUE_POLL_MS` | `250` | Chu kỳ worker poll |
| `TTS_JOB_LEASE_SECONDS` | `30` | Hạn lease trước khi recover |
| `TTS_JOB_MAX_ATTEMPTS` | `2` | Tổng lần claim tối đa |
| `TTS_SYNC_WAIT_SECONDS` | `120` | Thời gian `async=0` chờ trước khi trả `202` |

## Giới hạn phạm vi

- Không thêm Redis, Celery, RQ hoặc service ngoài.
- Không chạy hai inference đồng thời trên cùng GPU.
- Không batch nhiều request trong một model call.
- Không chia chunk của một request qua nhiều GPU.
- Không thêm authentication/rate limiting trong thay đổi này; queue capacity chỉ là backpressure toàn cục.
- Không thay đổi ASR scheduling.
- Không hủy inference CUDA đang chạy; chỉ retry hữu hạn khi lease mất.

## Tiêu chí chấp nhận

1. Một GPU chỉ có tối đa một job `running` tại mọi thời điểm.
2. Hai GPU có thể claim và chạy hai job khác nhau đồng thời.
3. Không job nào được hai worker claim cùng lúc.
4. Restart Flask không làm mất job terminal hoặc queued trên cùng filesystem.
5. Worker chết khiến job được requeue đúng một lần, không retry vô hạn.
6. Queue đầy trả `429` và không đưa thêm Future/job vào RAM.
7. `async=0` không bypass GPU scheduler.
8. Hai worker hoàn thành gần đồng thời không làm hỏng hoặc mất entry history.
9. Colab dùng một model replica; Kaggle dùng tối đa hai replica, mỗi GPU một replica.
10. Test không yêu cầu CUDA/model weights; smoke test GPU được chạy riêng trong notebook.

## Xác minh thực tế bắt buộc

Sau unit/integration test bằng fake worker, cần benchmark notebook riêng:

- Colab: 1, 4 và 8 job; đo queue wait, inference time, peak VRAM và lỗi OOM.
- Kaggle: cùng corpus với `TTS_GPU_IDS=0` và `0,1`; so sánh throughput tổng, latency p50/p95 và peak VRAM từng GPU.
- Xác nhận bằng `nvidia-smi` rằng mỗi worker chỉ dùng GPU được giao.
- Không tuyên bố tăng gấp đôi throughput trước khi có kết quả benchmark.
