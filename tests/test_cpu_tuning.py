import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cpu_tuning
from cpu_tuning import GIB, CpuConfig


class CpuTuningTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache_dir = Path(self.directory.name) / "profiles"
        self.env = patch.dict(os.environ, {
            "WHISPER_AUTOTUNE": "1", "WHISPER_WORKERS": "auto",
            "WHISPER_CPU_THREADS": "auto", "SERVER_MAX_CONCURRENT_JOBS": "1",
            "WHISPER_AUTOTUNE_TIMEOUT_SECONDS": "180",
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        platform_patch = patch.object(cpu_tuning.sys, "platform", "linux")
        platform_patch.start()
        self.addCleanup(platform_patch.stop)

    def select(self, **kwargs):
        defaults = {
            "model_name": "large-v3", "chunk_files": [f"chunk{i}.wav" for i in range(40)],
            "cpu_count": 16, "available_memory": 32 * GIB, "fallback": CpuConfig(4, 4),
            "cache_dir": self.cache_dir,
        }
        return cpu_tuning.select_cpu_config(**(defaults | kwargs))

    @staticmethod
    def result(seconds=10, loaded=2, peak=4):
        return {"seconds": seconds, "cpu_seconds": seconds * 8,
                "loaded_memory_bytes": loaded * GIB,
                "peak_memory_bytes": peak * GIB, "segments": 16}

    def test_candidates_respect_cpu_budget_and_available_tasks(self):
        self.assertEqual(cpu_tuning.candidate_configs(16, 40),
                         [CpuConfig(4, 4), CpuConfig(8, 2), CpuConfig(16, 1)])
        for cpus in (1, 3, 6, 16):
            for tasks in (1, 2, 7, 40):
                for config in cpu_tuning.candidate_configs(cpus, tasks):
                    self.assertLessEqual(config.workers, tasks)
                    self.assertEqual(config.workers * config.threads, cpus)

    def test_balancing_uses_remaining_cpu_budget_within_worker_limit(self):
        self.assertEqual(cpu_tuning.balanced_cpu_config(16, 7), CpuConfig(4, 4))
        self.assertEqual(cpu_tuning.balanced_cpu_config(16, 3), CpuConfig(2, 8))
        self.assertEqual(cpu_tuning.balanced_cpu_config(16, 8), CpuConfig(8, 2))
        for cpus in (3, 6, 10, 14, 16, 17):
            for limit in range(1, cpus + 1):
                config = cpu_tuning.balanced_cpu_config(cpus, limit)
                self.assertLessEqual(config.workers, limit)
                self.assertEqual(config.workers * config.threads, cpus)

    def test_selects_fastest_and_reuses_persistent_cache(self):
        with (
            patch.object(cpu_tuning, "_extract_samples", return_value=["sample.wav"]),
            patch.object(cpu_tuning, "_run_probe", side_effect=[
                self.result(30), self.result(10), self.result(20),
            ]) as probe,
        ):
            self.assertEqual(self.select(), CpuConfig(8, 2))
            # Equal audio and number of requests, regardless of parallelism.
            self.assertEqual([call.args[2:4] for call in probe.call_args_list],
                             [(["sample.wav"], 16)] * 3)
        with (
            patch.object(cpu_tuning, "_run_probe") as probe,
            patch.object(cpu_tuning, "_extract_samples") as extraction,
        ):
            self.assertEqual(self.select(), CpuConfig(8, 2))
            probe.assert_not_called()
            extraction.assert_not_called()

    def test_cache_key_changes_with_cpu_model_and_library_configuration(self):
        original = cpu_tuning._cache_key("large-v3", 16, 40)
        self.assertNotEqual(original, cpu_tuning._cache_key("medium", 16, 40))
        self.assertNotEqual(original, cpu_tuning._cache_key("large-v3", 8, 40))
        self.assertNotEqual(original, cpu_tuning._cache_key("large-v3", 16, 2))
        with patch.dict(os.environ, {"OMP_WAIT_POLICY": "PASSIVE"}):
            self.assertNotEqual(original, cpu_tuning._cache_key("large-v3", 16, 40))

    def test_corrupted_cache_is_replaced(self):
        self.cache_dir.mkdir()
        key = cpu_tuning._cache_key("large-v3", 16, 40)
        path = self.cache_dir / f"{key}.json"
        path.write_text("{broken json")
        with (
            patch.object(cpu_tuning, "_extract_samples", return_value=["sample.wav"]),
            patch.object(cpu_tuning, "_run_probe", return_value=self.result(10)),
        ):
            self.assertEqual(self.select(), CpuConfig(4, 4))
        self.assertEqual(json.loads(path.read_text())["config"], {"workers": 4, "threads": 4})

    def test_cache_is_rejected_when_available_memory_falls(self):
        path = Path(self.directory.name) / "cache.json"
        path.write_text(json.dumps({"config": {"workers": 8, "threads": 2},
                                    "peak_memory_bytes": 8 * GIB}))
        candidates = cpu_tuning.candidate_configs(16, 40)
        self.assertEqual(cpu_tuning._read_cache(path, candidates, 20 * GIB), CpuConfig(8, 2))
        self.assertIsNone(cpu_tuning._read_cache(path, candidates, 6 * GIB))

    def test_does_not_try_larger_candidate_when_measured_memory_is_too_high(self):
        with (
            patch.object(cpu_tuning, "_extract_samples", return_value=["sample.wav"]),
            patch.object(cpu_tuning, "_run_probe", side_effect=[
                self.result(30, loaded=2, peak=5), self.result(20, loaded=2, peak=9),
            ]) as probe,
        ):
            self.assertEqual(self.select(available_memory=16 * GIB), CpuConfig(8, 2))
            self.assertEqual(probe.call_count, 2)

    def test_probe_failures_fall_back_without_caching(self):
        with (
            patch.object(cpu_tuning, "_extract_samples", return_value=["sample.wav"]),
            patch.object(cpu_tuning, "_run_probe",
                         side_effect=subprocess.TimeoutExpired("probe", 180)),
        ):
            self.assertEqual(self.select(), CpuConfig(4, 4))
        self.assertFalse(self.cache_dir.exists())

    def test_manual_settings_disabled_tuning_and_concurrent_jobs_skip_probes(self):
        for override in ({"WHISPER_WORKERS": "4"}, {"WHISPER_CPU_THREADS": "1"},
                         {"WHISPER_AUTOTUNE": "0"}, {"SERVER_MAX_CONCURRENT_JOBS": "2"}):
            with (self.subTest(override=override), patch.dict(os.environ, override),
                  patch.object(cpu_tuning, "_run_probe") as probe):
                self.assertEqual(self.select(), CpuConfig(4, 4))
                probe.assert_not_called()
        self.assertEqual(self.select(available_memory=None), CpuConfig(4, 4))
        with patch.object(cpu_tuning, "_run_probe") as probe:
            self.assertEqual(self.select(chunk_files=["a.wav", "b.wav"]), CpuConfig(4, 4))
            probe.assert_not_called()

    def test_probe_rejects_silence_and_invalid_timings(self):
        for override in ({"segments": 0}, {"seconds": float("nan")}, {"seconds": -1},
                         {"cpu_seconds": float("nan")}, {"cpu_seconds": -1}):
            result = self.result() | override
            completed = subprocess.CompletedProcess([], 0, json.dumps(result), "")
            with (patch.object(cpu_tuning.subprocess, "run", return_value=completed),
                  self.assertRaises(ValueError)):
                cpu_tuning._run_probe("large-v3", CpuConfig(4, 4), ["sample.wav"],
                                      16, 30 * GIB, 180)

    @unittest.skipUnless(sys.platform == "linux", "Linux probe isolation")
    def test_real_probe_subprocess_exhausts_lazy_segments(self):
        # Exercise subprocess transport, memory limit, parent binding and lazy
        # generation without downloading model weights or using real audio.
        module = Path(self.directory.name) / "faster_whisper.py"
        module.write_text(
            "from types import SimpleNamespace\n"
            "class WhisperModel:\n"
            "    def __init__(self, *args, **kwargs): pass\n"
            "    def transcribe(self, path, **kwargs):\n"
            "        def segments():\n"
            "            yield SimpleNamespace(text='речь')\n"
            "            yield SimpleNamespace(text='ещё речь')\n"
            "        return segments(), None\n",
            encoding="utf-8",
        )
        with patch.dict(os.environ, {"PYTHONPATH": self.directory.name}):
            result = cpu_tuning._run_probe("fake", CpuConfig(4, 1), ["sample.wav"],
                                           8, GIB, 20)
        self.assertEqual(result["segments"], 16)
        self.assertGreater(result["seconds"], 0)
        self.assertGreaterEqual(result["cpu_seconds"], 0)
        self.assertGreater(result["peak_memory_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
