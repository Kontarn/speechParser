import asyncio
import threading
import tempfile
import unittest
import uuid
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx

import bot
import worker_server


class RemoteProcessingTests(unittest.TestCase):
    def _start_server(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), worker_server.ProcessorHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        return server, server_thread

    def _complete_job(self, job_id):
        job = worker_server.JOBS[job_id]
        Path(job["output_path"]).write_text("Удалённая расшифровка\n", encoding="utf-8")
        worker_server._set_job(job_id, status="completed", message="Расшифровка готова.")

    def test_ping_reports_worker_available(self):
        server, server_thread = self._start_server()
        message = Mock()
        message.reply_text = AsyncMock()
        update = Mock(message=message, effective_message=message)

        try:
            with (
                patch.object(bot, "PROCESSOR_URL", f"http://127.0.0.1:{server.server_port}"),
                patch.object(bot, "check_access", return_value=True),
            ):
                asyncio.run(bot.cmd_ping(update, Mock()))

            self.assertRegex(
                message.reply_text.await_args.args[0],
                r"^✅ Обработчик доступен \(\d+ мс\)\.$",
            )
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=2)

    def test_ping_logs_worker_unavailable_without_chat_error(self):
        server, server_thread = self._start_server()
        server_port = server.server_port
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)

        message = Mock()
        message.reply_text = AsyncMock()
        update = Mock(message=message, effective_message=message)
        with (
            patch.object(bot, "PROCESSOR_URL", f"http://127.0.0.1:{server_port}"),
            patch.object(bot, "check_access", return_value=True),
            self.assertLogs(bot.log, level="WARNING") as logs,
        ):
            asyncio.run(bot.cmd_ping(update, Mock()))

        message.reply_text.assert_not_awaited()
        self.assertIn("Воркер недоступен при /ping", logs.output[0])
        self.assertIsNone(logs.records[0].exc_info)

    def test_worker_unavailability_does_not_hide_auth_or_processing_errors(self):
        for status in (401, 403, 422, 500, 502, 503, 504):
            with self.subTest(status=status):
                response = httpx.Response(status, request=httpx.Request("POST", "https://worker.test/jobs"))
                with self.assertRaises(httpx.HTTPStatusError) as caught:
                    response.raise_for_status()
                self.assertEqual(bot._is_worker_unavailable(caught.exception), status in {502, 503, 504})
        for method, path, expected in (
            ("GET", "/jobs/id", True), ("DELETE", "/jobs/id", True),
            ("GET", "/jobs/id/result", True), ("POST", "/jobs", False),
        ):
            response = httpx.Response(404, request=httpx.Request(method, "https://worker.test" + path))
            with self.assertRaises(httpx.HTTPStatusError) as caught:
                response.raise_for_status()
            self.assertEqual(bot._is_worker_unavailable(caught.exception), expected)
        self.assertFalse(bot._is_worker_unavailable(RuntimeError("Whisper failed")))
        self.assertFalse(bot._is_worker_unavailable(httpx.UnsupportedProtocol("invalid URL")))

    def test_disconnected_jobs_are_logged_and_release_slot_for_urls_and_files(self):
        for kind in ("url", "file"):
            for error in (httpx.ConnectError("connection refused"), httpx.ReadTimeout("read timed out"),
                          httpx.RemoteProtocolError("peer closed connection")):
                with self.subTest(kind=kind, error=type(error).__name__):
                    message = Mock()
                    message.message_id = 10
                    message.text = "https://my.mts-link.ru/1/2/record-new/3"
                    message.from_user.id = 1
                    message.audio = SimpleNamespace(file_id="audio", file_name="audio.wav", file_size=3)
                    message.reply_text = AsyncMock(return_value=SimpleNamespace(message_id=11))
                    update = Mock(message=message, effective_user=SimpleNamespace(id=1),
                                  effective_chat=SimpleNamespace(id=1))
                    tg_file = SimpleNamespace(download_to_drive=AsyncMock(
                        side_effect=lambda path: Path(path).write_bytes(b"wav")))
                    telegram_bot = Mock(get_file=AsyncMock(return_value=tg_file))
                    context = SimpleNamespace(bot=telegram_bot)

                    # Run the handler's background target immediately so its
                    # exception path and finally cleanup can be asserted.
                    def start_thread(*, target, daemon):
                        return SimpleNamespace(start=target)

                    with (
                        patch.object(bot, "ACTIVE_JOBS", 0),
                        patch.object(bot, "MAX_CONCURRENT_JOBS", 1),
                        patch.object(bot, "JOB_STATUS", {}),
                        patch.object(bot, "JOB_QUEUE", []),
                        patch.object(bot, "check_access", return_value=True),
                        patch.object(bot.threading, "Thread", side_effect=start_thread),
                        patch.object(bot, "_run_pipeline_with_timeout", side_effect=error) as run,
                        patch.object(bot, "_send_processing_failure") as failure,
                        patch.object(bot, "_send_result_sync") as result,
                        self.assertLogs(bot.log, level="WARNING") as logs,
                    ):
                        handler = bot.handle_message if kind == "url" else bot.handle_file
                        asyncio.run(handler(update, context))
                        failure.assert_not_called()
                        result.assert_not_called()
                        self.assertEqual(bot.ACTIVE_JOBS, 0)
                        self.assertTrue(all(job["status"] == "done" for job in bot.JOB_STATUS.values()))
                        self.assertEqual(len(logs.records), 1)
                        self.assertIn("Воркер недоступен", logs.output[0])
                        self.assertIsNone(logs.records[0].exc_info)
                        telegram_bot.send_message.assert_not_called()
                        telegram_bot.edit_message_text.assert_not_called()
                        if kind == "file":
                            self.assertFalse(Path(run.call_args.args[1][0]).exists())

    def test_bot_submits_job_and_downloads_result(self):
        server, server_thread = self._start_server()

        url = f"https://my.mts-link.ru/1/2/record-new/{uuid.uuid4().int}"
        local_job_id = uuid.uuid4().hex
        output_path = None
        bot.JOB_STATUS[local_job_id] = {"remote_job_id": None}

        try:
            with (
                patch.object(worker_server, "API_KEY", "test-api-key"),
                patch.object(worker_server, "_run_job", side_effect=self._complete_job),
                patch.object(bot, "PROCESSOR_URL", f"http://127.0.0.1:{server.server_port}"),
                patch.object(bot, "PROCESSOR_API_KEY", "test-api-key"),
                patch.object(bot, "REMOTE_POLL_INTERVAL_SECONDS", 0.01),
            ):
                output_path = bot.get_transcript_path("url", (url, "medium"))
                result = bot._run_pipeline_with_timeout(
                    "url", (url, "medium"), lambda _: None, local_job_id
                )
                self.assertEqual(Path(result).read_text(encoding="utf-8"), "Удалённая расшифровка\n")
                remote_job_id = bot.JOB_STATUS[local_job_id]["remote_job_id"]
                remote_job_dir = worker_server.JOBS_DIR / remote_job_id
                self.assertTrue(remote_job_dir.is_dir())

                bot._cleanup_remote_job(local_job_id)

                self.assertFalse(remote_job_dir.exists())
                self.assertNotIn(remote_job_id, worker_server.JOBS)
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=2)
            bot.JOB_STATUS.pop(local_job_id, None)
            if output_path and output_path.exists():
                output_path.unlink()

    def test_bot_uploads_file_and_downloads_result(self):
        server, server_thread = self._start_server()
        output_path = None

        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3") as input_file:
                input_file.write(b"test audio bytes")
                input_file.flush()
                with (
                    patch.object(worker_server, "API_KEY", "test-api-key"),
                    patch.object(worker_server, "_run_job", side_effect=self._complete_job),
                    patch.object(bot, "PROCESSOR_URL", f"http://127.0.0.1:{server.server_port}"),
                    patch.object(bot, "PROCESSOR_API_KEY", "test-api-key"),
                    patch.object(bot, "REMOTE_POLL_INTERVAL_SECONDS", 0.01),
                ):
                    output_path = bot.get_transcript_path("file", (input_file.name, "medium"))
                    result = bot._run_pipeline_with_timeout(
                        "file",
                        (input_file.name, "medium"),
                        lambda _: None,
                    )
                    self.assertEqual(Path(result).read_text(encoding="utf-8"), "Удалённая расшифровка\n")
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=2)
            if output_path and output_path.exists():
                output_path.unlink()


if __name__ == "__main__":
    unittest.main()
