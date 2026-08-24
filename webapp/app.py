from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict
import mimetypes
from pathlib import Path
from threading import Lock
from typing import Any

from flask import Flask, Response, abort, jsonify, redirect, render_template, request, send_from_directory, url_for
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from webapp.job_queue import QueueFullError, SQLiteJobQueue, SynthesisSubmission
from webapp.storage import load_tts_history

try:
    from webapp.tts_service import (
        GWEN_GENERATION_DEFAULTS,
        GWEN_GENERATION_LIMITS,
        TTSError,
        TEXT_INPUT_LIMIT,
        TTSStudioService,
        _normalize_gwen_generation_config,
        _parse_pronunciation_overrides,
    )
except ImportError:  # pragma: no cover
    from tts_service import (
        GWEN_GENERATION_DEFAULTS,
        GWEN_GENERATION_LIMITS,
        TTSError,
        TEXT_INPUT_LIMIT,
        TTSStudioService,
        _normalize_gwen_generation_config,
        _parse_pronunciation_overrides,
    )

ALLOWED_ASR_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
MAX_ASR_DURATION_MINUTES = 60


ROOT = Path(__file__).resolve().parent.parent
CSS_BUNDLE_VERSION = int((ROOT / "webapp" / "static" / "css" / "style.css").stat().st_mtime)
app = Flask(__name__, static_folder="static", template_folder="templates")
app.logger.setLevel(logging.INFO)
logging.getLogger("werkzeug").setLevel(logging.WARNING)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024
MAX_UPLOAD_MB = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)

studio = TTSStudioService(ROOT)

TEXT_EXAMPLES = [
    "Xin chào, đây là bản demo chuyển văn bản thành giọng nói tiếng Việt bằng Gwen-TTS.",
    "Thông báo: lớp học xử lý ngôn ngữ tự nhiên sẽ bắt đầu lúc tám giờ sáng tại phòng A3.",
    "Hôm nay chúng ta sẽ thu voice-over cho landing page với tông giọng rõ, sáng và chuyên nghiệp.",
    "Chúc bạn có một ngày làm việc hiệu quả, nhẹ nhàng và nhiều năng lượng tích cực.",
]

REFERENCE_TIPS = [
    "Audio mẫu nên dài 3 đến 10 giây, một người nói, ít nhiễu.",
    "Gwen-TTS cần transcript đúng với câu trong audio mẫu.",
    "Cần GPU CUDA để chạy Gwen-TTS local.",
]

SETUP_STEPS = [
    {
        "title": "Cài webapp",
        "body": "Cài dependencies tối thiểu để chạy giao diện và API.",
    },
    {
        "title": "Bật Gwen-TTS",
        "body": "Cài `qwen-tts` và `sea-g2p`; model sẽ load khi bạn generate.",
    },
]

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
JOB_RETENTION_SECONDS = 60 * 60
DATA_DIR = ROOT / "webapp" / "data"
USER_SETTINGS_PATH = DATA_DIR / "user_settings.json"
TTS_HISTORY_PATH = DATA_DIR / "tts_history.json"
MAX_HISTORY_ITEMS = 50
_storage_lock = Lock()


class TTSJobManager:
    def __init__(self, queue: SQLiteJobQueue | None = None) -> None:
        if queue is None:
            queue_path = Path(
                os.environ.get("TTS_QUEUE_DB", ROOT / "webapp" / "runtime" / "tts_jobs.sqlite3")
            )
            try:
                max_pending = int(os.environ.get("TTS_QUEUE_MAX_PENDING", "32"))
                lease_seconds = float(os.environ.get("TTS_JOB_LEASE_SECONDS", "30"))
            except ValueError:
                max_pending, lease_seconds = 32, 30.0
            queue = SQLiteJobQueue(
                queue_path,
                max_pending=max(1, min(1024, max_pending)),
                lease_seconds=max(3.0, min(3600.0, lease_seconds)),
            )
        self._queue = queue

    def enqueue(self, submission: SynthesisSubmission) -> str:
        return self._queue.enqueue(submission)

    def get_snapshot(self, job_id: str) -> dict[str, Any] | None:
        self._queue.purge_terminal(JOB_RETENTION_SECONDS)
        snapshot = self._queue.get_snapshot(job_id)
        if snapshot is None:
            return None
        message = {
            "queued": "Đã nhận yêu cầu. Job đang chờ worker synthesize.",
            "running": "Model đang synthesize trong worker GPU.",
            "completed": "Đã synthesize xong audio.",
            "failed": snapshot.error or "Job synthesize thất bại.",
        }[snapshot.status]
        return {
            "id": snapshot.id,
            "status": snapshot.status,
            "created_at": snapshot.created_at,
            "updated_at": snapshot.updated_at,
            "message": message,
            "result": snapshot.result,
            "error": snapshot.error,
            "worker_id": snapshot.worker_id,
            "gpu_index": snapshot.gpu_index,
            "attempts": snapshot.attempts,
            "queue_position": self._queue.queue_position(snapshot.id),
        }

    def wait_for_terminal(self, job_id: str, timeout_seconds: float) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout_seconds
        while True:
            snapshot = self.get_snapshot(job_id)
            if snapshot is None or snapshot["status"] in {"completed", "failed"}:
                return snapshot
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return snapshot
            time.sleep(min(0.1, remaining))


tts_job_manager = TTSJobManager()


def _parse_gwen_generation_form() -> dict[str, Any]:
    subtalker_do_sample_values = request.form.getlist("gwen_subtalker_do_sample")
    raw_config: dict[str, Any] = {
        "speed": request.form.get("speed"),
        "temperature": request.form.get("gwen_temperature"),
        "top_p": request.form.get("gwen_top_p"),
        "top_k": request.form.get("gwen_top_k"),
        "repetition_penalty": request.form.get("gwen_repetition_penalty"),
        "max_new_tokens": request.form.get("gwen_max_new_tokens"),
        "subtalker_temperature": request.form.get("gwen_subtalker_temperature"),
        "subtalker_top_k": request.form.get("gwen_subtalker_top_k"),
        "subtalker_top_p": request.form.get("gwen_subtalker_top_p"),
        "subtalker_sampling_method": request.form.get("gwen_subtalker_sampling_method"),
    }
    if subtalker_do_sample_values:
        raw_config["subtalker_do_sample"] = any(
            str(value).strip().lower() in {"1", "true", "yes", "on"}
            for value in subtalker_do_sample_values
        )
    normalized = _normalize_gwen_generation_config(raw_config)
    return normalized


def _parse_pronunciation_form() -> list[tuple[str, str]]:
    sources = request.form.getlist("pronunciation_from")
    targets = request.form.getlist("pronunciation_to")
    row_count = max(len(sources), len(targets))
    for index in range(row_count):
        source = (sources[index] if index < len(sources) else "").strip()
        target = (targets[index] if index < len(targets) else "").strip()
        if bool(source) != bool(target):
            raise ValueError(f"Dòng phát âm {index + 1} phải có đủ chữ gốc và cách đọc.")
    return _parse_pronunciation_overrides(sources, targets)


def _gwen_settings_payload() -> dict[str, Any]:
    return {
        "defaults": {
            "speed": GWEN_GENERATION_DEFAULTS["speed"],
            "temperature": GWEN_GENERATION_DEFAULTS["temperature"],
            "topP": GWEN_GENERATION_DEFAULTS["top_p"],
            "topK": GWEN_GENERATION_DEFAULTS["top_k"],
            "repetitionPenalty": GWEN_GENERATION_DEFAULTS["repetition_penalty"],
            "maxNewTokens": GWEN_GENERATION_DEFAULTS["max_new_tokens"],
            "codePredictorDoSample": GWEN_GENERATION_DEFAULTS["subtalker_do_sample"],
            "codePredictorTemperature": GWEN_GENERATION_DEFAULTS["subtalker_temperature"],
            "codePredictorTopK": GWEN_GENERATION_DEFAULTS["subtalker_top_k"],
            "codePredictorTopP": GWEN_GENERATION_DEFAULTS["subtalker_top_p"],
            "codePredictorSamplingMethod": GWEN_GENERATION_DEFAULTS["subtalker_sampling_method"],
        },
        "ranges": {
            "speed": GWEN_GENERATION_LIMITS["speed"],
            "temperature": GWEN_GENERATION_LIMITS["temperature"],
            "topP": GWEN_GENERATION_LIMITS["top_p"],
            "topK": GWEN_GENERATION_LIMITS["top_k"],
            "repetitionPenalty": GWEN_GENERATION_LIMITS["repetition_penalty"],
            "maxNewTokens": GWEN_GENERATION_LIMITS["max_new_tokens"],
            "codePredictorTemperature": GWEN_GENERATION_LIMITS["subtalker_temperature"],
            "codePredictorTopK": GWEN_GENERATION_LIMITS["subtalker_top_k"],
            "codePredictorTopP": GWEN_GENERATION_LIMITS["subtalker_top_p"],
        },
    }


def _base_context(active: str) -> dict:
    engine_cards = studio.get_engine_cards()
    return {
        "active": active,
        "asr_enabled": studio.is_asr_enabled(),
        "engine_cards": engine_cards,
        "status_summary": studio.summary(),
        "static_version": int((ROOT / "webapp" / "static" / "css" / "style.css").stat().st_mtime),
    }


def _pick_default_engine(engine_cards: list) -> str:
    return "gwen"


def _engine_page_context(engine_id: str) -> dict:
    context = _base_context(engine_id)
    try:
        engine_card = studio.get_engine_card(engine_id)
    except TTSError as exc:
        abort(404, description=str(exc))

    preset_voices = studio.get_preset_voices(engine_card.id)
    context.update(
        examples=TEXT_EXAMPLES,
        reference_tips=REFERENCE_TIPS,
        max_upload_mb=MAX_UPLOAD_MB,
        text_input_limit=TEXT_INPUT_LIMIT,
        engine_card=engine_card,
        engine_card_payload=asdict(engine_card),
        model_selection=engine_card.metadata.get("model_selection", studio.get_model_selection(engine_card.id)),
        gwen_settings_payload=_gwen_settings_payload() if engine_card.id == "gwen" else None,
        preset_voices_payload=[
            {
                "id": voice.id,
                "name": voice.name,
                "avatar": voice.avatar,
                "style": voice.style,
                "reference_text": voice.reference_text,
                "audio_url": url_for("static", filename=f"voice_presets/{voice.audio_filename}"),
            }
            for voice in preset_voices
        ],
    )
    return context


def _is_api_request() -> bool:
    return request.path.startswith("/api/")


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _read_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        app.logger.warning("Failed to read JSON file: %s", path)
        return default


def _write_json_file(path: Path, data: Any) -> None:
    _ensure_data_dir()
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def _sanitize_pronunciation_rules(raw_rules: Any) -> list[dict[str, str]]:
    rules: list[dict[str, str]] = []
    if not isinstance(raw_rules, list):
        return rules
    for item in raw_rules:
        if not isinstance(item, dict):
            continue
        source = str(item.get("from", "")).strip()
        target = str(item.get("to", "")).strip()
        if source and target:
            rules.append({"from": source, "to": target})
    return rules


def _load_user_settings() -> dict[str, Any]:
    with _storage_lock:
        payload = _read_json_file(
            USER_SETTINGS_PATH,
            {
                "pronunciation_rules": [],
                "last_text": "",
            },
        )
    if not isinstance(payload, dict):
        payload = {}
    return {
        "pronunciation_rules": _sanitize_pronunciation_rules(payload.get("pronunciation_rules")),
        "last_text": str(payload.get("last_text", "") or ""),
    }


def _save_user_settings(raw_payload: Any) -> dict[str, Any]:
    if not isinstance(raw_payload, dict):
        raise ValueError("Payload settings không hợp lệ.")
    sanitized = {
        "pronunciation_rules": _sanitize_pronunciation_rules(raw_payload.get("pronunciation_rules")),
        "last_text": str(raw_payload.get("last_text", "") or ""),
    }
    with _storage_lock:
        _write_json_file(USER_SETTINGS_PATH, sanitized)
    return sanitized


def _load_tts_history() -> list[dict[str, Any]]:
    return load_tts_history(TTS_HISTORY_PATH)


def _build_transcription_payload(result) -> dict[str, Any]:
    return {
        "ok": True,
        "text": result.text,
        "language": result.language,
        "model_id": result.model_id,
        "duration_seconds": round(result.duration_seconds, 2),
        "inference_seconds": round(result.inference_seconds, 2),
        "notes": list(result.notes),
    }


def _asr_disabled_response():
    return jsonify({"ok": False, "error": studio.asr_disabled_message()}), 503


def _build_synthesis_submission() -> SynthesisSubmission:
    upload = request.files.get("reference_audio")
    text = (request.form.get("text") or "").strip()
    engine_id = "gwen"
    preset_voice_id = (request.form.get("preset_voice_id") or "").strip().lower()
    model_key = (request.form.get("model_key") or "default").strip() or "default"
    custom_model = (request.form.get("custom_model") or "").strip()
    reference_text = (request.form.get("reference_text") or "").strip()
    remove_silence = (request.form.get("remove_silence") or "").strip().lower() in {"1", "true", "yes", "on"}

    speed_raw = (request.form.get("speed") or "1.0").strip()
    seed_raw = (request.form.get("seed") or "").strip()
    gwen_generation_config = None
    pronunciation_overrides: list[tuple[str, str]] = []

    preset_voice = None
    if preset_voice_id:
        preset_voice, reference_path = studio.get_preset_voice_reference(engine_id, preset_voice_id)
        reference_text = preset_voice.reference_text
        app.logger.info(
            "Resolved preset voice for synth: engine=%s preset_voice_id=%s preset_name=%s audio=%s reference_chars=%d",
            engine_id,
            preset_voice.id,
            preset_voice.name,
            reference_path.name,
            len(reference_text),
        )
    else:
        if not upload or not upload.filename:
            raise ValueError("Cần upload audio tham chiếu trước khi sinh giọng.")

        extension = Path(upload.filename).suffix.lower()
        if extension not in ALLOWED_AUDIO_EXTENSIONS:
            raise ValueError("Định dạng audio chưa hỗ trợ. Chỉ nhận .wav, .mp3, .m4a, .flac, .ogg.")

        reference_path = studio.save_reference_file(upload.filename, upload.read())
        app.logger.info(
            "Resolved custom reference for synth: engine=%s filename=%s reference_chars=%d",
            engine_id,
            reference_path.name,
            len(reference_text),
        )

    gwen_generation_config = _parse_gwen_generation_form()
    pronunciation_overrides = _parse_pronunciation_form()
    speed = float(gwen_generation_config["speed"])

    seed = None
    if seed_raw:
        try:
            seed = int(seed_raw)
        except ValueError as exc:
            raise ValueError("Seed phải là số nguyên nếu được nhập.") from exc

    return SynthesisSubmission(
        engine_id=engine_id,
        text=text,
        reference_path=reference_path,
        reference_text=reference_text,
        speed=speed,
        remove_silence=remove_silence,
        seed=seed,
        model_key=model_key,
        custom_model=custom_model,
        gwen_generation_config=gwen_generation_config,
        pronunciation_overrides=pronunciation_overrides,
        preset_voice=preset_voice,
    )


def _should_use_async_generation() -> bool:
    async_raw = (request.form.get("async") or "").strip().lower()
    if async_raw in {"1", "true", "yes", "on"}:
        return True
    if async_raw in {"0", "false", "no", "off"}:
        return False
    return bool(request.headers.get("CF-Connecting-IP") or request.headers.get("CF-Ray"))


@app.errorhandler(RequestEntityTooLarge)
def handle_request_entity_too_large(exc: RequestEntityTooLarge):
    if not _is_api_request():
        return exc
    return (
        jsonify(
            {
                "ok": False,
                "error": f"File upload quá lớn. Giới hạn hiện tại là {MAX_UPLOAD_MB} MB.",
            }
        ),
        413,
    )


@app.errorhandler(HTTPException)
def handle_api_http_exception(exc: HTTPException):
    if not _is_api_request():
        return exc
    return jsonify({"ok": False, "error": exc.description or exc.name}), exc.code or 500


@app.route("/")
def home():
    return redirect(url_for("engine_page", engine_id=_pick_default_engine(studio.get_engine_cards())))


@app.route("/studio/<engine_id>")
def engine_page(engine_id: str):
    if engine_id != "gwen":
        abort(404)
    return render_template("studio_engine.html", **_engine_page_context(engine_id))


@app.route("/voices")
def voices_page():
    context = _base_context("voices")
    preset_voices = studio.get_preset_voices("gwen")
    context.update(
        reference_tips=REFERENCE_TIPS,
        max_upload_mb=MAX_UPLOAD_MB,
        preset_voices_payload=[
            {
                "id": voice.id,
                "name": voice.name,
                "avatar": voice.avatar,
                "style": voice.style,
                "reference_text": voice.reference_text,
                "audio_url": url_for("static", filename=f"voice_presets/{voice.audio_filename}"),
            }
            for voice in preset_voices
        ],
    )
    return render_template("voices.html", **context)


@app.route("/asr")
def asr_page():
    if not studio.is_asr_enabled():
        return redirect(url_for("engine_page", engine_id=_pick_default_engine(studio.get_engine_cards())))
    context = _base_context("asr")
    context.update(max_upload_mb=MAX_UPLOAD_MB)
    return render_template("asr.html", **context)


@app.route("/api/tts/status")
def api_tts_status():
    cards = studio.get_engine_cards()
    return jsonify(
        {
            "ok": True,
            "asr_enabled": studio.is_asr_enabled(),
            "summary": studio.summary(),
            "engines": [
                {
                    "id": card.id,
                    "label": card.label,
                    "ready": card.ready,
                    "summary": card.summary,
                    "warning": card.warning,
                    "metadata": card.metadata,
                }
                for card in cards
            ],
        }
    )


@app.route("/api/tts/transcribe-reference", methods=["POST"])
def api_tts_transcribe_reference():
    if not studio.is_asr_enabled():
        return _asr_disabled_response()

    upload = request.files.get("reference_audio")
    if not upload or not upload.filename:
        return jsonify({"ok": False, "error": "Cần upload audio tham chiếu để nhận diện transcript."}), 400

    extension = Path(upload.filename).suffix.lower()
    if extension not in ALLOWED_AUDIO_EXTENSIONS:
        return jsonify({"ok": False, "error": "Định dạng audio chưa hỗ trợ. Chỉ nhận .wav, .mp3, .m4a, .flac, .ogg."}), 400

    reference_path = studio.save_reference_file(upload.filename, upload.read())
    try:
        result = studio.transcribe_reference_audio(reference_path)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except TTSError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    finally:
        reference_path.unlink(missing_ok=True)

    return jsonify(_build_transcription_payload(result))


@app.route("/api/asr/transcribe", methods=["POST"])
def api_asr_transcribe():
    if not studio.is_asr_enabled():
        return _asr_disabled_response()

    upload = request.files.get("audio")
    if not upload or not upload.filename:
        return jsonify({"ok": False, "error": "Cần upload file audio để nhận diện."}), 400

    extension = Path(upload.filename).suffix.lower()
    if extension not in ALLOWED_ASR_EXTENSIONS:
        return jsonify({"ok": False, "error": f"Định dạng audio chưa hỗ trợ. Chỉ nhận: {', '.join(sorted(ALLOWED_ASR_EXTENSIONS))}"}), 400

    hotwords = (request.form.get("hotwords") or "").strip()
    max_new_tokens_raw = (request.form.get("max_new_tokens") or "8192").strip()
    chunk_seconds_raw = (request.form.get("tokenizer_chunk_seconds") or "60").strip()
    return_parsed = (request.form.get("return_parsed") or "1").strip().lower() in {"1", "true", "yes", "on"}
    dtype = (request.form.get("dtype") or "").strip() or None
    system_prompt = (request.form.get("system_prompt") or "").strip() or None

    try:
        max_new_tokens = max(2048, min(16384, int(max_new_tokens_raw)))
    except ValueError:
        max_new_tokens = 8192
    try:
        chunk_seconds = max(5, min(60, int(chunk_seconds_raw)))
    except ValueError:
        chunk_seconds = 60

    reference_path = studio.save_reference_file(upload.filename, upload.read())
    try:
        result = studio.transcribe_audio_vibevoice(
            reference_path,
            hotwords=hotwords,
            max_new_tokens=max_new_tokens,
            tokenizer_chunk_seconds=chunk_seconds,
            return_parsed=return_parsed,
            dtype_override=dtype,
            system_prompt=system_prompt,
        )
    except TTSError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Lỗi không mong muốn: {exc}"}), 500

    return jsonify({
        "ok": True,
        "text": result.raw_text,
        "segments": result.parsed_segments,
        "model_id": result.model_id,
        "duration_seconds": round(result.duration_seconds, 2),
        "inference_seconds": round(result.inference_seconds, 2),
        "notes": result.notes,
    })


@app.route("/api/tts/generate", methods=["POST"])
def api_tts_generate():
    try:
        submission = _build_synthesis_submission()
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except TTSError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        return jsonify({"ok": False, "error": f"Lỗi không mong muốn: {exc}"}), 500

    try:
        job_id = tts_job_manager.enqueue(submission)
    except QueueFullError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 429, {"Retry-After": "2"}

    if _should_use_async_generation():
        return (
            jsonify(
                {
                    "ok": True,
                    "queued": True,
                    "job_id": job_id,
                    "status": "queued",
                    "message": "Job đã được tách nền để tránh timeout Cloudflare. Hãy poll trạng thái cho tới khi hoàn tất.",
                }
            ),
            202,
        )

    try:
        timeout_seconds = float(os.environ.get("TTS_SYNC_WAIT_SECONDS", "120"))
    except ValueError:
        timeout_seconds = 120.0
    snapshot = tts_job_manager.wait_for_terminal(
        job_id,
        max(1.0, min(3600.0, timeout_seconds)),
    )
    if snapshot is None:
        return jsonify({"ok": False, "error": "Không tìm thấy job vừa tạo."}), 500
    if snapshot["status"] == "completed":
        return jsonify(snapshot["result"])
    if snapshot["status"] == "failed":
        return jsonify({"ok": False, "error": snapshot["error"] or snapshot["message"]}), 422
    return (
        jsonify(
            {
                "ok": True,
                "queued": True,
                "job_id": job_id,
                "status": snapshot["status"],
                "message": "Job vẫn đang xử lý; hãy tiếp tục poll trạng thái.",
            }
        ),
        202,
    )

@app.route("/api/tts/jobs/<job_id>")
def api_tts_job_status(job_id: str):
    snapshot = tts_job_manager.get_snapshot(job_id.strip())
    if snapshot is None:
        return jsonify({"ok": False, "error": "Không tìm thấy job generate hoặc job đã hết hạn."}), 404

    status = snapshot["status"]
    if status in {"queued", "running"}:
        return jsonify(
            {
                "ok": True,
                "done": False,
                "job_id": snapshot["id"],
                "status": status,
                "message": snapshot["message"],
                "queue_position": snapshot.get("queue_position"),
                "worker_id": snapshot.get("worker_id"),
                "gpu_index": snapshot.get("gpu_index"),
                "attempts": snapshot.get("attempts", 0),
            }
        )

    if status == "completed":
        return jsonify(
            {
                "ok": True,
                "done": True,
                "job_id": snapshot["id"],
                "status": status,
                "result": snapshot["result"],
            }
        )

    return jsonify(
        {
            "ok": False,
            "done": True,
            "job_id": snapshot["id"],
            "status": status,
            "error": snapshot["error"] or snapshot["message"],
        }
    )


ALLOWED_PRONUNCIATION_EXTENSIONS = {".json", ".txt"}
MAX_PRONUNCIATION_FILE_BYTES = 512 * 1024


def _parse_pronunciation_file_content(raw_bytes: bytes, filename: str) -> list[dict[str, str]]:
    """Parse uploaded pronunciation file into list of {from, to} dicts."""
    extension = Path(filename).suffix.lower()
    text = raw_bytes.decode("utf-8", errors="replace").strip()
    if not text:
        raise ValueError("File rỗng, không chứa cặp phát âm nào.")

    if extension == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"File JSON không hợp lệ: {exc}") from exc

        if isinstance(data, dict):
            return [{"from": str(k).strip(), "to": str(v).strip()} for k, v in data.items() if str(k).strip() and str(v).strip()]
        if isinstance(data, list):
            rules: list[dict[str, str]] = []
            for index, item in enumerate(data):
                if not isinstance(item, dict):
                    raise ValueError(f"Phần tử {index + 1} trong JSON array phải là object.")
                source = str(item.get("from", "")).strip()
                target = str(item.get("to", "")).strip()
                if not source and not target:
                    continue
                if not source or not target:
                    raise ValueError(f"Dòng {index + 1}: thiếu 'from' hoặc 'to'.")
                rules.append({"from": source, "to": target})
            return rules
        raise ValueError("JSON phải là array [{from, to}, ...] hoặc object {key: value, ...}.")

    # TXT format: each line is "source -> target" or "source|target" or "source=target" or "source\ttarget"
    rules = []
    for line_num, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pair = None
        for separator in (" -> ", "->" , "\t", "|", "="):
            if separator in line:
                parts = line.split(separator, 1)
                if len(parts) == 2:
                    pair = (parts[0].strip(), parts[1].strip())
                    break
        if pair is None:
            raise ValueError(f"Dòng {line_num}: không tìm thấy dấu phân cách (->  |  =  tab). Định dạng: 'chữ gốc -> cách đọc'.")
        source, target = pair
        if not source or not target:
            raise ValueError(f"Dòng {line_num}: thiếu chữ gốc hoặc cách đọc.")
        rules.append({"from": source, "to": target})
    return rules


@app.route("/api/pronunciation/upload", methods=["POST"])
def api_pronunciation_upload():
    """Upload a .json or .txt file containing pronunciation overrides."""
    upload = request.files.get("pronunciation_file")
    if not upload or not upload.filename:
        return jsonify({"ok": False, "error": "Cần upload file cài đặt phát âm (.json hoặc .txt)."}), 400

    extension = Path(upload.filename).suffix.lower()
    if extension not in ALLOWED_PRONUNCIATION_EXTENSIONS:
        return jsonify({"ok": False, "error": "Chỉ hỗ trợ file .json hoặc .txt."}), 400

    raw_bytes = upload.read()
    if len(raw_bytes) > MAX_PRONUNCIATION_FILE_BYTES:
        return jsonify({"ok": False, "error": f"File quá lớn. Giới hạn {MAX_PRONUNCIATION_FILE_BYTES // 1024} KB."}), 400

    try:
        rules = _parse_pronunciation_file_content(raw_bytes, upload.filename)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    if not rules:
        return jsonify({"ok": False, "error": "File không chứa cặp phát âm hợp lệ nào."}), 400

    app.logger.info("Parsed %d pronunciation rules from uploaded file: %s", len(rules), upload.filename)
    return jsonify({"ok": True, "rules": rules, "count": len(rules), "filename": upload.filename})


@app.route("/api/pronunciation/export", methods=["POST"])
def api_pronunciation_export():
    """Export current pronunciation rules as a downloadable JSON file."""
    try:
        data = request.get_json(silent=True)
    except Exception:
        data = None

    if not data or not isinstance(data.get("rules"), list):
        return jsonify({"ok": False, "error": "Cần gửi JSON body chứa key 'rules'."}), 400

    rules = []
    for item in data["rules"]:
        if not isinstance(item, dict):
            continue
        source = str(item.get("from", "")).strip()
        target = str(item.get("to", "")).strip()
        if source and target:
            rules.append({"from": source, "to": target})

    content = json.dumps(rules, ensure_ascii=False, indent=2)
    return Response(
        content,
        mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=pronunciation_rules.json"},
    )


@app.route("/api/settings/load")
def api_settings_load():
    return jsonify({"ok": True, **_load_user_settings()})


@app.route("/api/settings/save", methods=["POST"])
def api_settings_save():
    data = request.get_json(silent=True)
    try:
        saved = _save_user_settings(data)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except OSError as exc:
        return jsonify({"ok": False, "error": f"Không thể lưu settings: {exc}"}), 500
    return jsonify({"ok": True, **saved})


@app.route("/api/history")
def api_history():
    history = _load_tts_history()
    return jsonify({"ok": True, "items": history[:MAX_HISTORY_ITEMS]})


@app.route("/outputs/<path:filename>")
def serve_generated_audio(filename: str):
    guessed = mimetypes.guess_type(filename)[0] or "audio/wav"
    return send_from_directory(studio.output_dir, filename, mimetype=guessed, as_attachment=False)


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    host = (os.getenv("TTS_WEBAPP_HOST") or "0.0.0.0").strip() or "0.0.0.0"
    port_raw = (os.getenv("TTS_WEBAPP_PORT") or "8386").strip()
    try:
        port = int(port_raw)
    except ValueError:
        app.logger.warning("Invalid TTS_WEBAPP_PORT=%r, fallback to 8386.", port_raw)
        port = 8386

    debug = _env_flag("TTS_WEBAPP_DEBUG", default=False)
    use_reloader = _env_flag("TTS_WEBAPP_RELOADER", default=False)
    app.run(debug=debug, host=host, port=port, use_reloader=use_reloader)
