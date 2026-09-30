import subprocess
import tempfile
import unittest
from pathlib import Path
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

    def test_audio_chunks_are_extracted_in_parallel_and_ordered(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "source.m4a"
            output_path = Path(temp_dir) / "transcript.txt"
            input_path.touch()
            max_workers_used = []
            original_executor = pipeline.ThreadPoolExecutor

            def create_executor(max_workers):
                max_workers_used.append(max_workers)
                return original_executor(max_workers=max_workers)

            def fake_ffmpeg(command, **kwargs):
                Path(command[-1]).write_bytes(b"wav")
                return subprocess.CompletedProcess(command, 0, "", "")

            def fake_get_model(model_name, workers=None):
                pipeline._model_workers = workers
                return object()

            def fake_transcribe_one_file(input_path, model_name, output_path, **kwargs):
                Path(output_path).write_text(f"part{kwargs['chunk_number']}\n", encoding="utf-8")

            with (
                patch.object(pipeline, "BASE_DIR", Path(temp_dir)),
                patch.object(pipeline, "get_media_duration", return_value=1201),
                patch.object(pipeline, "_get_cpu_count", return_value=4),
                patch.object(pipeline, "get_worker_count", return_value=4),
                patch.object(pipeline, "get_model", side_effect=fake_get_model),
                patch.object(pipeline, "_transcribe_one_file", side_effect=fake_transcribe_one_file),
                patch.object(pipeline.subprocess, "run", side_effect=fake_ffmpeg),
                patch.object(pipeline, "ThreadPoolExecutor", side_effect=create_executor),
            ):
                pipeline.transcribe_file(str(input_path), "medium", str(output_path))

            self.assertEqual(max_workers_used, [4, 4])
            expected_output = "".join(f"part{part}\n" for part in range(1, 12))
            self.assertEqual(output_path.read_text(encoding="utf-8"), expected_output)

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
