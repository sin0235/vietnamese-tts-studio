from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Mapping


def parse_gpu_ids(raw: str) -> list[int]:
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            raise ValueError("TTS_GPU_IDS không được rỗng.")
        try:
            gpu_id = int(item)
        except ValueError as exc:
            raise ValueError(f"GPU ID không hợp lệ: {item}") from exc
        if gpu_id < 0:
            raise ValueError("GPU ID phải không âm.")
        if gpu_id not in values:
            values.append(gpu_id)
    return values


def build_worker_env(base_env: Mapping[str, str], gpu_id: int) -> dict[str, str]:
    env = dict(base_env)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["TTS_WORKER_GPU_INDEX"] = str(gpu_id)
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Flask and one sequential TTS worker per GPU")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8386)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port phải nằm trong 1..65535")
    try:
        gpu_ids = parse_gpu_ids(os.environ.get("TTS_GPU_IDS", "0"))
    except ValueError as exc:
        parser.error(str(exc))

    workers: list[subprocess.Popen] = []
    try:
        for gpu_id in gpu_ids:
            workers.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "webapp.tts_worker",
                        "--gpu-id",
                        str(gpu_id),
                        "--worker-id",
                        f"gpu-{gpu_id}-{os.getpid()}",
                    ],
                    env=build_worker_env(os.environ, gpu_id),
                )
            )

        from webapp.app import app

        app.run(host=args.host, port=args.port, debug=False, threaded=True, use_reloader=False)
    finally:
        for process in workers:
            if process.poll() is None:
                process.terminate()
        for process in workers:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    main()
