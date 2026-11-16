import asyncio
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx

import bot
import pipeline
from common import parse_mts_url
from media import extract_audio_chunk
from processor_client import ProcessorClient
from transcript_parts import assemble_partial, commit_part


class QueueTests(unittest.TestCase):
    def setUp(self):
        for name, value in (("JOB_STATUS", {}), ("JOB_QUEUE", []), ("ACTIVE_JOBS", 0),
                            ("MAX_CONCURRENT_JOBS", 1)):
            setting = patch.object(bot, name, value)
            setting.start()
            self.addCleanup(setting.stop)

    def test_fifo_waiters_start_after_a_slot_is_released(self):
        first, second, third = [bot.register_job(1, 1) for _ in range(3)]
        self.assertTrue(bot.acquire_job_slot(first))
        self.assertFalse(bot.acquire_job_slot(second))
        self.assertEqual(bot.JOB_QUEUE, [second, third])
        order = []

        def run(job_id):
            if bot.wait_for_job_slot(job_id):
                order.append(job_id)
                bot.release_job_slot(job_id)

        threads = [threading.Thread(target=run, args=(job_id,), daemon=True) for job_id in (third, second)]
        for thread in threads:
            thread.start()
        bot.release_job_slot(first)
        for thread in threads:
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
        self.assertEqual(order, [second, third])
        self.assertEqual(bot.ACTIVE_JOBS, 0)

    def test_cancelling_a_queued_job_does_not_release_another_jobs_slot(self):
        active = bot.register_job(1, 1)
        queued = bot.register_job(1, 1)
        self.assertTrue(bot.acquire_job_slot(active))
        bot.JOB_STATUS[queued]["cancel_requested"] = True
        self.assertFalse(bot.wait_for_job_slot(queued))
        bot.release_job_slot(queued, "cancelled")
        self.assertEqual(bot.ACTIVE_JOBS, 1)
        self.assertEqual(bot.JOB_STATUS[active]["status"], "running")

    def test_two_submitted_links_both_run_and_use_distinct_result_paths(self):
        first_started = threading.Event()
        allow_first_finish = threading.Event()
        calls = []

        def run(kind, args, update_status, job_id):
            calls.append(job_id)
            if len(calls) == 1:
                first_started.set()
                if not allow_first_finish.wait(3):
                    raise AssertionError("first job was never released")
            return bot.JOB_STATUS[job_id]["output_path"]

        def update(number):
            message = Mock(text="https://my.mts-link.ru/1/2/record-new/3", message_id=number)
            message.reply_text = AsyncMock(return_value=SimpleNamespace(message_id=number + 10))
            return SimpleNamespace(message=message, effective_user=SimpleNamespace(id=1),
                                   effective_chat=SimpleNamespace(id=1))

        with (
            patch.object(bot, "check_access", return_value=True),
            patch.object(bot, "_run_pipeline_with_timeout", side_effect=run),
            patch.object(bot, "_send_result_sync"), patch.object(bot, "_cleanup_remote_job"),
        ):
            context = SimpleNamespace(bot=Mock())
            asyncio.run(bot.handle_message(update(1), context))
            self.assertTrue(first_started.wait(2))
            asyncio.run(bot.handle_message(update(2), context))
            self.assertEqual(len(calls), 1)
            allow_first_finish.set()
            jobs = list(bot.JOB_STATUS.values())
            for job in jobs:
                job["thread"].join(timeout=3)
                self.assertFalse(job["thread"].is_alive())
        self.assertEqual(len(calls), 2)
        self.assertEqual(len({job["output_path"] for job in jobs}), 2)
        self.assertTrue(all(job["status"] == "completed" for job in jobs))
        self.assertEqual(bot.ACTIVE_JOBS, 0)

    def test_telegram_delivery_does_not_hold_the_transcription_slot(self):
        delivery_started = threading.Event()
        allow_delivery = threading.Event()
        next_started = threading.Event()
        first = bot.register_job(1, 1)
        second = bot.register_job(1, 1)

        def run(kind, args, update_status, job_id):
            if job_id == second:
                next_started.set()
            return bot.JOB_STATUS[job_id]["output_path"]

        def deliver(telegram_bot, chat_id, message_id, path, loop):
            if message_id == 1:
                delivery_started.set()
                self.assertTrue(allow_delivery.wait(3))

        with (
            patch.object(bot, "_run_pipeline_with_timeout", side_effect=run),
            patch.object(bot, "_send_result_sync", side_effect=deliver),
            patch.object(bot, "_cleanup_remote_job"),
        ):
            bot._start_job(first, "url", ("https://my.mts-link.ru/1/2/record-new/3", "medium"), Mock(), 1, 1, None)
            try:
                self.assertTrue(delivery_started.wait(2))
                bot._start_job(second, "url", ("https://my.mts-link.ru/1/2/record-new/4", "medium"), Mock(), 1, 2, None)
                self.assertTrue(next_started.wait(2))
            finally:
                allow_delivery.set()
                for job in bot.JOB_STATUS.values():
                    if job.get("thread"):
                        job["thread"].join(timeout=3)


class ResourceTests(unittest.TestCase):
    def test_concurrent_jobs_share_cpu_and_respect_memory_high(self):
        def read(path, *args, **kwargs):
            values = {"/proc/meminfo": "MemAvailable: 33554432 kB\n",
                      "/sys/fs/cgroup/cpu.max": "max 100000",
                      "/sys/fs/cgroup/memory.max": str(32 * 1024**3),
                      "/sys/fs/cgroup/memory.high": str(16 * 1024**3),
                      "/sys/fs/cgroup/memory.current": str(8 * 1024**3)}
            if str(path) not in values:
                raise OSError("not mounted")
            return values[str(path)]

        with (
            patch.dict("os.environ", {"SERVER_MAX_CONCURRENT_JOBS": "2"}),
            patch.object(pipeline.os, "cpu_count", return_value=16),
            patch.object(pipeline.os, "sched_getaffinity", return_value=set(range(16)), create=True),
            patch.object(Path, "read_text", read),
        ):
            self.assertEqual(pipeline._get_cpu_count(), 8)
            self.assertEqual(pipeline._get_available_memory_bytes(), 4 * 1024**3)


class TranscriptTests(unittest.TestCase):
    def test_delivery_reads_the_file_and_sends_bytes_without_blocking_the_event_loop(self):
        async def scenario(path):
            telegram_bot = SimpleNamespace(edit_message_text=AsyncMock(), send_document=AsyncMock())
            loop = asyncio.get_running_loop()
            await asyncio.to_thread(bot._send_result_sync, telegram_bot, 1, 2, str(path), loop)
            self.assertEqual(telegram_bot.send_document.await_args.kwargs["document"], "текст".encode())

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.txt"
            path.write_text("текст", encoding="utf-8")
            asyncio.run(scenario(path))

    def test_closed_status_updater_cannot_overwrite_final_status(self):
        async def scenario():
            telegram_bot = SimpleNamespace(edit_message_text=AsyncMock())
            updater = bot.make_status_updater(telegram_bot, 1, 2, asyncio.get_running_loop())
            updater("old progress")
            updater.close()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            telegram_bot.edit_message_text.assert_not_awaited()
        asyncio.run(scenario())

    def test_cancel_rebuilds_ordered_completed_parts_without_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / "transcript.txt")
            commit_part(output, 3, lambda path: Path(path).write_text("third\n"))
            commit_part(output, 1, lambda path: Path(path).write_text("first\n"))

            def interrupted(path):
                Path(path).write_text("incomplete\n")
                raise RuntimeError("interrupted")

            with self.assertRaises(RuntimeError):
                commit_part(output, 2, interrupted)
            Path(output).write_text("first\npar")
            assemble_partial(output)
            self.assertEqual(Path(output).read_text(), "first\nthird\n")
            assemble_partial(output)
            self.assertEqual(Path(output).read_text(), "first\nthird\n")

    def test_pcm_slice_preserves_samples_without_ffmpeg(self):
        with tempfile.TemporaryDirectory() as directory:
            source = str(Path(directory) / "source.wav")
            target = str(Path(directory) / "part.wav")
            data = bytes(range(256)) * 500
            with wave.open(source, "wb") as stream:
                stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                stream.writeframes(data)
            with patch("media.subprocess.run") as ffmpeg:
                extract_audio_chunk(source, target, 1, 2)
                ffmpeg.assert_not_called()
            with wave.open(target, "rb") as stream:
                self.assertEqual(stream.getnframes(), 16000)
                self.assertEqual(stream.readframes(16000), data[32000:64000])

    def test_ready_prefix_is_published_before_all_parts_finish(self):
        first_can_finish = threading.Event()
        last_can_finish = threading.Event()
        published = []
        with tempfile.TemporaryDirectory() as directory:
            source = str(Path(directory) / "source.wav")
            output = str(Path(directory) / "transcript.txt")
            with wave.open(source, "wb") as stream:
                stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                stream.writeframes(bytes(6 * 16000 * 2))

            def transcribe(input_path, output_path, model):
                chunk_number = int(Path(input_path).stem.rsplit("_", 1)[1])
                if chunk_number == 1:
                    self.assertTrue(first_can_finish.wait(3))
                elif chunk_number == 3:
                    self.assertTrue(last_can_finish.wait(3))
                Path(output_path).write_text(f"part{chunk_number}\n")

            def progress(message):
                if "Готово частей: 1/3" in message:
                    first_can_finish.set()
                if "Готово частей: 2/3" in message:
                    published.append(Path(output).read_text())
                    last_can_finish.set()

            with (
                patch.dict("os.environ", {"WHISPER_AUTOTUNE": "0", "WHISPER_CPU_THREADS": "auto"}),
                patch.object(pipeline, "TRANSCRIPTION_CHUNK_SECONDS", 2),
                patch.object(pipeline, "_get_cpu_count", return_value=4),
                patch.object(pipeline, "get_worker_count", return_value=2),
                patch.object(pipeline, "get_model", return_value=object()),
                patch.object(pipeline, "_transcribe_one_file", side_effect=transcribe),
            ):
                pipeline.transcribe_file(source, "medium", output, progress)
            self.assertEqual(published, ["part1\npart2\n"])
            self.assertEqual(Path(output).read_text(), "part1\npart2\npart3\n")


class ClientTests(unittest.TestCase):
    def test_fallback_uses_final_render_and_keeps_parent_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "mtslinker"
            package.mkdir()
            (package / "__init__.py").write_text("")
            (package / "webinar.py").write_text(
                "from pathlib import Path\n"
                "def fetch_webinar_data(**kwargs):\n"
                "    folder = Path('recording'); folder.mkdir()\n"
                "    (folder / 'recording.mp4').write_bytes(b'final')\n"
                "    (folder / 'raw-participant.mp4').write_bytes(b'raw' * 1000)\n"
                "    return 1\n"
            )
            working = root / "job"
            working.mkdir()
            previous = Path.cwd()
            with patch.dict("os.environ", {"PYTHONPATH": directory}):
                result = pipeline.download_recording_full("1", None, None, str(working))
            self.assertEqual(Path(result).read_bytes(), b"final")
            self.assertEqual(Path.cwd(), previous)

    def test_deadline_is_not_replaced_by_failed_remote_cancellation(self):
        clock = Mock(return_value=0)
        requests = []

        def request(req):
            requests.append(req.method)
            if req.method == "POST":
                self.assertLessEqual(req.extensions["timeout"]["read"], 1)
                return httpx.Response(202, json={"job_id": "id"})
            if req.method == "GET":
                return httpx.Response(200, json={"status": "running", "message": "working"})
            raise httpx.ConnectError("worker disconnected")

        client = ProcessorClient("https://worker.test", "key", 1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.txt"
            with (
                patch.object(client, "_client", side_effect=lambda: httpx.Client(transport=httpx.MockTransport(request))),
                patch("processor_client.time.monotonic", clock),
                patch("processor_client.time.sleep", side_effect=lambda _: setattr(clock, "return_value", 2)),
                self.assertRaises(TimeoutError),
            ):
                client.run("url", ("url", "medium"), output, lambda _: None)
        self.assertEqual(requests, ["POST", "GET", "DELETE"])

    def test_failed_stream_does_not_publish_a_partial_or_stale_result(self):
        class BrokenStream(httpx.SyncByteStream):
            def __iter__(self):
                yield bytes(65536)
                raise httpx.ReadError("disconnected")

        def request(req):
            if req.method == "POST":
                return httpx.Response(202, json={"job_id": "id"})
            if req.url.path.endswith("result"):
                return httpx.Response(200, stream=BrokenStream())
            return httpx.Response(200, json={"status": "completed"})

        client = ProcessorClient("https://worker.test", "key", 60)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.txt"
            output.write_text("previous job")
            with (patch.object(client, "_client", side_effect=lambda: httpx.Client(transport=httpx.MockTransport(request))),
                  self.assertRaises(httpx.ReadError)):
                client.run("url", ("url", "medium"), output, lambda _: None)
            self.assertFalse(output.exists())
            self.assertFalse(list(Path(directory).glob("*.download")))

    def test_url_parser_rejects_a_valid_prefix_followed_by_garbage(self):
        with self.assertRaises(ValueError):
            parse_mts_url("https://my.mts-link.ru/1/2/record-new/3/invalid")
        self.assertEqual(parse_mts_url("https://my.mts-link.ru/1/2/record-new/3?view=1"), ("3", None))


if __name__ == "__main__":
    unittest.main()
