from __future__ import annotations

import unittest

from webapp.serve import build_worker_env, parse_gpu_ids


class ServeTest(unittest.TestCase):
    def test_parse_gpu_ids_deduplicates_and_rejects_negative_values(self) -> None:
        self.assertEqual(parse_gpu_ids("0,1,1"), [0, 1])
        with self.assertRaises(ValueError):
            parse_gpu_ids("0,-1")

    def test_parse_gpu_ids_rejects_empty_or_non_numeric_values(self) -> None:
        for raw in ("", "0,x"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_gpu_ids(raw)

    def test_build_worker_env_exposes_only_assigned_gpu(self) -> None:
        env = build_worker_env({"PATH": "/usr/bin", "CUDA_VISIBLE_DEVICES": "0,1"}, 1)

        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(env["TTS_WORKER_GPU_INDEX"], "1")
        self.assertEqual(env["PATH"], "/usr/bin")


if __name__ == "__main__":
    unittest.main()
