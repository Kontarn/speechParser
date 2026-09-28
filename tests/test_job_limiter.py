import unittest

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
