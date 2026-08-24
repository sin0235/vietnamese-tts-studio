from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_results(
    latencies: list[float],
    *,
    completed: int,
    failed: int,
    wall_seconds: float,
) -> dict[str, Any]:
    return {
        "completed": completed,
        "failed": failed,
        "latency_p50_seconds": round(_percentile(latencies, 0.50), 3),
        "latency_p95_seconds": round(_percentile(latencies, 0.95), 3),
        "throughput_jobs_per_second": round(completed / wall_seconds, 3) if wall_seconds > 0 else 0.0,
        "wall_seconds": round(wall_seconds, 3),
    }


def _request_json(request: urllib.request.Request) -> dict[str, Any]:
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise RuntimeError("API không trả JSON object.")
    return payload


def _run_job(base_url: str, text: str, preset_voice_id: str) -> tuple[bool, float]:
    started_at = time.monotonic()
    body = urllib.parse.urlencode(
        {
            "engine": "gwen",
            "text": text,
            "preset_voice_id": preset_voice_id,
            "async": "1",
        }
    ).encode()
    payload = _request_json(
        urllib.request.Request(
            f"{base_url}/api/tts/generate",
            data=body,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    )
    job_id = payload.get("job_id")
    if not isinstance(job_id, str):
        return False, time.monotonic() - started_at
    while True:
        status = _request_json(
            urllib.request.Request(f"{base_url}/api/tts/jobs/{urllib.parse.quote(job_id)}")
        )
        if status.get("done"):
            return bool(status.get("ok")), time.monotonic() - started_at
        time.sleep(0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark hàng đợi TTS qua HTTP local")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--preset-voice-id", default="yen_nhi")
    parser.add_argument("--text", default="Xin chào, đây là bài đo hàng đợi TTS.")
    args = parser.parse_args()
    if args.concurrency < 1 or args.requests < 1:
        parser.error("--concurrency và --requests phải lớn hơn 0")

    started_at = time.monotonic()
    completed = 0
    failed = 0
    latencies: list[float] = []
    with ThreadPoolExecutor(max_workers=min(args.concurrency, args.requests)) as executor:
        futures = [
            executor.submit(
                _run_job,
                args.base_url.rstrip("/"),
                args.text,
                args.preset_voice_id,
            )
            for _ in range(args.requests)
        ]
        for future in futures:
            try:
                ok, latency = future.result()
            except Exception:
                failed += 1
                continue
            latencies.append(latency)
            completed += int(ok)
            failed += int(not ok)

    print(
        json.dumps(
            summarize_results(
                latencies,
                completed=completed,
                failed=failed,
                wall_seconds=time.monotonic() - started_at,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
