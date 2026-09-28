import threading
import tempfile
import unittest
import uuid
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

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

    def test_bot_submits_job_and_downloads_result(self):
        server, server_thread = self._start_server()

        url = f"https://my.mts-link.ru/1/2/record-new/{uuid.uuid4().int}"
        output_path = None

        try:
            with (
                patch.object(worker_server, "API_KEY", "test-api-key"),
                patch.object(worker_server, "_run_job", side_effect=self._complete_job),
                patch.object(bot, "PROCESSOR_URL", f"http://127.0.0.1:{server.server_port}"),
                patch.object(bot, "PROCESSOR_API_KEY", "test-api-key"),
                patch.object(bot, "REMOTE_POLL_INTERVAL_SECONDS", 0.01),
            ):
                output_path = bot.get_transcript_path("url", (url, "medium"))
                result = bot._run_pipeline_with_timeout("url", (url, "medium"), lambda _: None)
                self.assertEqual(Path(result).read_text(encoding="utf-8"), "Удалённая расшифровка\n")
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=2)
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