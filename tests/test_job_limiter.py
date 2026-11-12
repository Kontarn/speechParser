import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import bot
import pipeline


class JobLimiterTests(unittest.TestCase):
    def setUp(self):
        bot.ACTIVE_JOBS = 0
        bot.JOB_STATUS = {}
        bot.JOB_QUEUE = []

    def test_acquire_slot_when_idle(self):
        self.assertTrue(bot.acquire_job_slot())
        self.assertEqual(bot.ACTIVE_JOBS, 1)

    def test_rejects_when_limit_reached(self):
        bot.MAX_CONCURRENT_JOBS = 1
        self.assertTrue(bot.acquire_job_slot())
        self.assertFalse(bot.acquire_job_slot())
        self.assertEqual(bot.ACTIVE_JOBS, 1)

    def test_release_slot_releases_capacity(self):
        bot.MAX_CONCURRENT_JOBS = 1
        self.assertTrue(bot.acquire_job_slot())
        bot.release_job_slot()
        self.assertTrue(bot.acquire_job_slot())
        self.assertEqual(bot.ACTIVE_JOBS, 1)

    def test_registers_job_and_tracks_position(self):
        job_id = bot.register_job(job_type="test", user_id=1, chat_id=2, message_id=3)
        self.assertIn(job_id, bot.JOB_STATUS)
        self.assertEqual(bot.get_queue_position(job_id), 1)

    def test_waits_for_slot_with_notifications(self):
        bot.MAX_CONCURRENT_JOBS = 1
        first = bot.register_job(job_type="first", user_id=1, chat_id=2, message_id=3)
        self.assertTrue(bot.acquire_job_slot(first))
        second = bot.register_job(job_type="second", user_id=1, chat_id=2, message_id=4)
        self.assertEqual(bot.get_queue_position(second), 1)
        bot.release_job_slot(first)
        self.assertTrue(bot.acquire_job_slot(second))

    def test_memory_safe_model_defaults_to_medium(self):
        self.assertEqual(pipeline.get_effective_model_name(None), "medium")
        self.assertEqual(pipeline.get_effective_model_name("large-v3"), "large-v3")
        self.assertEqual(pipeline.get_effective_model_name("medium"), "medium")

    def test_clear_model_resets_model_cache(self):
        pipeline._model = object()
        pipeline._model_name = "large-v3"
        pipeline.clear_model()
        self.assertIsNone(pipeline._model)
        self.assertIsNone(pipeline._model_name)

    def test_single_model_worker_uses_all_cpu_threads(self):
        whisper_model = Mock(return_value=object())
        fake_module = Mock(WhisperModel=whisper_model)
        pipeline.clear_model()

        with (
            patch.dict("sys.modules", {"faster_whisper": fake_module}),
            patch.object(pipeline, "_get_cpu_count", return_value=8),
            patch.object(pipeline, "get_worker_count", return_value=8),
        ):
            pipeline.get_model("medium", workers=1)

        whisper_model.assert_called_once_with(
            "medium",
            device="cpu",
            compute_type="int8",
            num_workers=1,
            cpu_threads=8,
        )
        pipeline.clear_model()

    def test_automatic_threads_use_all_cpu_within_worker_limit(self):
        whisper_model = Mock(return_value=object())
        fake_module = Mock(WhisperModel=whisper_model)
        pipeline.clear_model()

        with (
            patch.dict("sys.modules", {"faster_whisper": fake_module}),
            patch.object(pipeline, "_get_cpu_count", return_value=16),
        ):
            pipeline.get_model("large-v3", workers=7)

        whisper_model.assert_called_once_with(
            "large-v3",
            device="cpu",
            compute_type="int8",
            num_workers=4,
            cpu_threads=4,
        )
        pipeline.clear_model()

    def test_default_cpu_config_respects_short_jobs_and_manual_threads(self):
        with patch.dict("os.environ", {"WHISPER_CPU_THREADS": "auto"}):
            self.assertEqual(pipeline.get_default_cpu_config(16, 3), pipeline.CpuConfig(2, 8))
        with patch.dict("os.environ", {"WHISPER_CPU_THREADS": "1"}):
            self.assertEqual(pipeline.get_default_cpu_config(16, 7), pipeline.CpuConfig(7, 1))
        with patch.dict("os.environ", {"WHISPER_CPU_THREADS": "invalid"}):
            self.assertEqual(pipeline.get_default_cpu_config(16, 7), pipeline.CpuConfig(4, 4))

    def test_model_cache_includes_cpu_threads(self):
        whisper_model = Mock(side_effect=lambda *args, **kwargs: object())
        pipeline.clear_model()
        self.addCleanup(pipeline.clear_model)
        with (
            patch.dict("sys.modules", {"faster_whisper": SimpleNamespace(WhisperModel=whisper_model)}),
            patch.object(pipeline, "_get_cpu_count", return_value=16),
        ):
            first = pipeline.get_model("medium", workers=4, cpu_threads=4)
            self.assertIs(pipeline.get_model("medium", workers=4, cpu_threads=4), first)
            self.assertIsNot(pipeline.get_model("medium", workers=4, cpu_threads=2), first)
        self.assertEqual(whisper_model.call_count, 2)

    def test_unlimited_cgroup_v1_does_not_reduce_cpu_count_to_one(self):
        def read_cgroup(path, *args, **kwargs):
            if str(path).endswith("cpu.cfs_quota_us"):
                return "-1"
            if str(path).endswith("cpu.cfs_period_us"):
                return "100000"
            raise OSError("cgroup v2 is not mounted")

        with (
            patch.object(pipeline.os, "cpu_count", return_value=16),
            patch.object(pipeline.os, "sched_getaffinity", return_value=set(range(16)), create=True),
            patch.object(Path, "read_text", read_cgroup),
        ):
            self.assertEqual(pipeline._get_cpu_count(), 16)

    def test_tuned_parallelism_is_used_for_all_chunks(self):
        model = Mock()
        model.transcribe.side_effect = lambda *args, **kwargs: (
            iter([SimpleNamespace(text="текст")]),
            SimpleNamespace(duration=120, language="ru", language_probability=1),
        )
        constructor = Mock(return_value=model)
        pipeline.clear_model()
        self.addCleanup(pipeline.clear_model)
        with tempfile.TemporaryDirectory() as directory:
            def extract(command, **kwargs):
                Path(command[-1]).write_bytes(b"wav")

            with (
                patch.dict("sys.modules", {"faster_whisper": SimpleNamespace(WhisperModel=constructor)}),
                patch.object(pipeline, "BASE_DIR", Path(directory)),
                patch.object(pipeline, "_get_cpu_count", return_value=16),
                patch.object(pipeline, "get_worker_count", return_value=2),
                patch.object(pipeline, "get_media_duration", return_value=1201),
                patch.object(pipeline.subprocess, "run", side_effect=extract),
                patch.object(pipeline, "select_cpu_config", return_value=pipeline.CpuConfig(8, 1)),
            ):
                output = Path(directory) / "result.txt"
                pipeline.transcribe_file("input.wav", "large-v3", str(output))
                self.assertEqual(output.read_text(), "текст\n" * 11)
        constructor.assert_called_once_with(
            "large-v3", device="cpu", compute_type="int8", num_workers=8, cpu_threads=1,
        )

    def test_single_worker_reuses_model_across_chunks(self):
        model = Mock()
        model.transcribe.side_effect = lambda *args, **kwargs: (
            iter([SimpleNamespace(text="текст")]),
            SimpleNamespace(duration=120, language="ru", language_probability=1),
        )
        constructor = Mock(return_value=model)
        pipeline.clear_model()
        self.addCleanup(pipeline.clear_model)
        with tempfile.TemporaryDirectory() as directory:
            def extract(command, **kwargs):
                Path(command[-1]).write_bytes(b"wav")

            with (
                patch.dict("os.environ", {"WHISPER_AUTOTUNE": "0"}),
                patch.dict("sys.modules", {"faster_whisper": SimpleNamespace(WhisperModel=constructor)}),
                patch.object(pipeline, "BASE_DIR", Path(directory)),
                patch.object(pipeline, "_get_cpu_count", return_value=4),
                patch.object(pipeline, "get_worker_count", return_value=1),
                patch.object(pipeline, "get_media_duration", return_value=361),
                patch.object(pipeline.subprocess, "run", side_effect=extract),
            ):
                output = Path(directory) / "result.txt"
                pipeline.transcribe_file("input.wav", "large-v3", str(output))
                self.assertEqual(output.read_text(), "текст\n" * 4)
        constructor.assert_called_once()
        self.assertIsNone(pipeline._model)

    def test_audio_chunks_are_extracted_in_parallel_and_ordered(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "source.m4a"
            output_path = Path(temp_dir) / "transcript.txt"
            input_path.touch()
            max_workers_used = []
            status_updates = []
            original_executor = pipeline.ThreadPoolExecutor

            def create_executor(max_workers):
                max_workers_used.append(max_workers)
                return original_executor(max_workers=max_workers)

            def fake_ffmpeg(command, **kwargs):
                Path(command[-1]).write_bytes(b"wav")
                return subprocess.CompletedProcess(command, 0, "", "")

            def fake_get_model(model_name, workers=None, cpu_threads=None):
                pipeline._model_workers = workers
                return object()

            def fake_transcribe_one_file(input_path, model_name, output_path, **kwargs):
                Path(output_path).write_text(f"part{kwargs['chunk_number']}\n", encoding="utf-8")

            with (
                patch.dict("os.environ", {"WHISPER_AUTOTUNE": "0"}),
                patch.object(pipeline, "BASE_DIR", Path(temp_dir)),
                patch.object(pipeline, "get_media_duration", return_value=1201),
                patch.object(pipeline, "_get_cpu_count", return_value=16),
                patch.object(pipeline, "get_worker_count", return_value=7),
                patch.object(pipeline, "get_model", side_effect=fake_get_model),
                patch.object(pipeline, "_transcribe_one_file", side_effect=fake_transcribe_one_file),
                patch.object(pipeline.subprocess, "run", side_effect=fake_ffmpeg),
                patch.object(pipeline, "ThreadPoolExecutor", side_effect=create_executor),
            ):
                pipeline.transcribe_file(
                    str(input_path), "medium", str(output_path), status_updates.append
                )

            self.assertEqual(max_workers_used, [11, 4])
            expected_output = "".join(f"part{part}\n" for part in range(1, 12))
            self.assertEqual(output_path.read_text(encoding="utf-8"), expected_output)
            self.assertEqual(status_updates[0], "🎙 Готово частей: 0/11 · 0,0/20,0 мин")
            self.assertEqual(status_updates[-1], "🎙 Готово частей: 11/11 · 20,0/20,0 мин")

    def test_build_chunk_ranges(self):
        ranges = pipeline.build_chunk_ranges(3700, chunk_seconds=600)
        self.assertEqual(len(ranges), 7)
        self.assertEqual(ranges[0], (0.0, 600.0))
        self.assertEqual(ranges[-1][1], 3700.0)

    def test_worker_count_is_limited_by_available_memory(self):
        workers = pipeline.calculate_worker_count(
            "medium",
            cpu_count=12,
            available_memory_bytes=int(1.6 * pipeline._GIB),
        )
        self.assertEqual(workers, 1)

    def test_worker_count_uses_cpu_and_memory_capacity(self):
        workers = pipeline.calculate_worker_count(
            "small",
            cpu_count=4,
            available_memory_bytes=8 * pipeline._GIB,
        )
        self.assertEqual(workers, 4)


if __name__ == "__main__":
    unittest.main()
