import asyncio
import threading
import tempfile
import unittest
import uuid
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

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

    def test_ping_reports_worker_unavailable(self):
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
        ):
            asyncio.run(bot.cmd_ping(update, Mock()))

        self.assertTrue(message.reply_text.await_args.args[0].startswith("❌ Обработчик недоступен:"))

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