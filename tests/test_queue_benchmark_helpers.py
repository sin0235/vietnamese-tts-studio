from __future__ import annotations

import unittest

from scripts.benchmark_tts_queue import summarize_results


class QueueBenchmarkHelpersTest(unittest.TestCase):
    def test_summarize_results_reports_latency_and_throughput(self) -> None:
        summary = summarize_results(
            [1.0, 2.0, 3.0, 4.0],
            completed=4,
            failed=0,
            wall_seconds=4.0,
        )

        self.assertEqual(summary["latency_p50_seconds"], 2.5)
        self.assertEqual(summary["latency_p95_seconds"], 3.85)
        self.assertEqual(summary["throughput_jobs_per_second"], 1.0)


if __name__ == "__main__":
    unittest.main()
