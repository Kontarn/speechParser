import contextlib
import io
import os
import signal
import tempfile
import threading
import unittest
from decimal import Decimal
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import httpx

import cli
import worker_server
from processor_client import ProcessorClient


class CliTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        for name, value in (("JOBS_DIR", self.root / "jobs"), ("JOBS", {}),
                            ("API_KEY", "test-key"), ("MAX_UPLOAD_GB", Decimal(10))):
            setting = patch.object(worker_server, name, value)
            setting.start()
            self.addCleanup(setting.stop)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), worker_server.ProcessorHandler)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()

        def stop():
            self.server.shutdown()
            self.server.server_close()
            thread.join(timeout=2)

        self.addCleanup(stop)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        environment = patch.dict(os.environ, {"PROCESSOR_URL": self.url, "PROCESSOR_API_KEY": "test-key"})
        environment.start()
        self.addCleanup(environment.stop)
        dotenv = patch.object(cli, "load_dotenv")
        dotenv.start()
        self.addCleanup(dotenv.stop)

    def run_cli(self, *arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = cli.main(list(arguments))
        return code, stdout.getvalue(), stderr.getvalue()

    def complete(self, job_id):
        job = worker_server.JOBS[job_id]
        Path(job["output_path"]).write_text("Готовая расшифровка\n", encoding="utf-8")
        worker_server._set_job(job_id, status="completed", message="Готово")

    def media(self, name="видео с пробелами.MP4"):
        path = self.root / name
        path.write_bytes(b"video contents")
        return path

    def test_upload_above_old_200_mib_limit_and_clean_remote_files(self):
        source = self.media()
        # Sparse fixture, real HTTP upload. No model or real video needed to
        # exercise large Content-Length, streaming, result delivery and cleanup.
        size = 201 * 1024**2
        with source.open("r+b") as data:
            data.truncate(size)
        received = []

        def complete(job_id):
            job = worker_server.JOBS[job_id]
            received.append(Path(job["args"][0]).stat().st_size)
            self.complete(job_id)

        original_handler = signal.getsignal(signal.SIGINT)
        with patch.object(worker_server, "_run_job", side_effect=complete):
            code, stdout, stderr = self.run_cli(str(source))
        self.assertEqual(code, 0, stderr)
        self.assertEqual(received, [size])
        output = source.with_suffix(".txt")
        self.assertEqual(output.read_text(encoding="utf-8"), "Готовая расшифровка\n")
        self.assertEqual(stdout.strip(), str(output))
        self.assertIn("100%", stderr)
        self.assertEqual(source.stat().st_size, size)
        self.assertFalse(worker_server.JOBS)
        self.assertFalse(list(worker_server.JOBS_DIR.iterdir()))
        self.assertIs(signal.getsignal(signal.SIGINT), original_handler)
        self.assertFalse(list(self.root.glob(".speechparser-*")))

    def test_url_custom_output_and_model(self):
        output = self.root / "transcripts" / "meeting.txt"
        received = []

        def complete(job_id):
            received.append(worker_server.JOBS[job_id]["args"])
            self.complete(job_id)

        url = "https://my.mts-link.ru/1/2/record-new/3"
        with patch.object(worker_server, "_run_job", side_effect=complete):
            code, stdout, stderr = self.run_cli(url, "-o", str(output), "--model", "tiny")
        self.assertEqual(code, 0, stderr)
        self.assertEqual(received, [(url, "tiny")])
        self.assertEqual(stdout.strip(), str(output))

    def test_reject_size_before_post(self):
        source = self.media()
        with patch.object(worker_server, "MAX_UPLOAD_GB", Decimal("0.000000001")), patch.object(worker_server, "_run_job") as run:
            code, _, stderr = self.run_cli(str(source))
        self.assertEqual(code, 1)
        self.assertIn("MAX_UPLOAD_GB", stderr)
        run.assert_not_called()
        self.assertFalse(worker_server.JOBS)
        self.assertFalse(worker_server.JOBS_DIR.exists())

    def test_authentication_error_is_readable(self):
        with patch.dict(os.environ, {"PROCESSOR_API_KEY": "wrong"}):
            code, _, stderr = self.run_cli(str(self.media()))
        self.assertEqual(code, 1)
        self.assertIn("401", stderr)
        self.assertIn("PROCESSOR_API_KEY", stderr)
        self.assertNotIn("Traceback", stderr)
        self.assertFalse(worker_server.JOBS)

    def test_existing_output_requires_force_and_survives_worker_failure(self):
        source = self.media()
        output = source.with_suffix(".txt")
        output.write_text("previous transcript")
        with patch.object(worker_server, "_run_job") as run:
            code, _, stderr = self.run_cli(str(source))
        self.assertEqual(code, 1)
        self.assertIn("--force", stderr)
        run.assert_not_called()

        def fail(job_id):
            worker_server._set_job(job_id, status="failed", error="Whisper failed")

        with patch.object(worker_server, "_run_job", side_effect=fail):
            code, _, stderr = self.run_cli(str(source), "--force")
        self.assertEqual(code, 1)
        self.assertIn("Whisper failed", stderr)
        self.assertEqual(output.read_text(), "previous transcript")
        self.assertFalse(worker_server.JOBS)

    def test_force_replaces_output_after_success(self):
        source = self.media()
        output = source.with_suffix(".txt")
        output.write_text("previous transcript")
        with patch.object(worker_server, "_run_job", side_effect=self.complete):
            code, _, stderr = self.run_cli(str(source), "--force")
        self.assertEqual(code, 0, stderr)
        self.assertEqual(output.read_text(encoding="utf-8"), "Готовая расшифровка\n")

    def test_interrupt_cancels_remote_job_and_saves_partial_result(self):
        source = self.media()
        original_signal = signal.getsignal(signal.SIGINT)

        def running(job_id):
            job = worker_server.JOBS[job_id]
            Path(job["output_path"]).write_text("Готовая часть\n", encoding="utf-8")
            worker_server._set_job(job_id, status="running")

        def poll(_):
            jobs = list(worker_server.JOBS.items())
            if jobs and jobs[0][1]["cancel_event"].is_set():
                worker_server._set_job(jobs[0][0], status="cancelled")
            else:
                # Exercise the CLI's actual installed handler on the main thread.
                signal.getsignal(signal.SIGINT)(signal.SIGINT, None)

        with patch.object(worker_server, "_run_job", side_effect=running), patch("processor_client.time.sleep", side_effect=poll):
            code, stdout, stderr = self.run_cli(str(source))
        self.assertEqual(code, 130, stderr)
        self.assertIn("Частичная", stderr)
        self.assertEqual(Path(stdout.strip()).read_text(encoding="utf-8"), "Готовая часть\n")
        self.assertFalse(worker_server.JOBS)
        self.assertIs(signal.getsignal(signal.SIGINT), original_signal)

    def test_invalid_inputs_do_not_submit(self):
        source = self.media()
        empty = self.root / "empty.mp4"
        empty.touch()
        unsupported = self.root / "file.pdf"
        unsupported.write_bytes(b"pdf")
        for arguments in (
            [str(self.root / "absent.mp4")], [str(empty)], [str(unsupported)],
            ["https://example.com/video.mp4"], [str(source), "-o", str(source), "--force"],
        ):
            with self.subTest(arguments=arguments), patch.object(ProcessorClient, "run") as run:
                code, _, _ = self.run_cli(*arguments)
                self.assertEqual(code, 1)
                run.assert_not_called()
        self.assertEqual(source.read_bytes(), b"video contents")

    def test_interrupt_when_job_just_completed_does_not_delete_result(self):
        source = self.media()
        status = cli._status

        def interrupt_after_submission(message):
            status(message)
            if message.startswith("Задание принято:"):
                signal.getsignal(signal.SIGINT)(signal.SIGINT, None)

        with (patch.object(worker_server.ProcessorHandler, "_start_job", side_effect=self.complete),
              patch.object(cli, "_status", side_effect=interrupt_after_submission)):
            code, stdout, stderr = self.run_cli(str(source))
        self.assertEqual(code, 0, stderr)
        self.assertEqual(Path(stdout.strip()).read_text(encoding="utf-8"), "Готовая расшифровка\n")
        self.assertFalse(worker_server.JOBS)

    def test_invalid_timeouts(self):
        for value in ("0", "-1", "nan", "inf", "oops"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    cli.main(["video.mp4", "--timeout", value])
                self.assertEqual(error.exception.code, 2)


class UploadProtocolTests(unittest.TestCase):
    def test_old_worker_without_limits_still_receives_stream_and_progress(self):
        body = b"x" * (2 * 64 * 1024 + 123)
        sizes, progress = [], []

        def request(request):
            if request.url.path == "/limits":
                return httpx.Response(404)
            if request.url.path == "/jobs/file":
                self.assertEqual(request.headers["Content-Length"], str(len(body)))
                for chunk in request.stream:
                    sizes.append(len(chunk))
                return httpx.Response(202, json={"job_id": "test"})
            if request.url.path == "/jobs/test":
                return httpx.Response(200, json={"status": "completed"})
            if request.url.path == "/jobs/test/result":
                return httpx.Response(200, content="text")
            self.fail(f"Unexpected request: {request.url}")

        # MockTransport eagerly reads request content before its handler. Wrap
        # its transport to inspect the actual outgoing streaming blocks.
        class Transport(httpx.BaseTransport):
            def handle_request(self, outgoing):
                return request(outgoing)

        client = ProcessorClient("https://worker.test", "key", 60)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "video.mp4"
            source.write_bytes(body)
            result = Path(directory) / "result.txt"
            with patch.object(client, "_client", side_effect=lambda: httpx.Client(transport=Transport())):
                client.run("file", (str(source), "tiny"), result, lambda _: None,
                           on_upload_progress=lambda sent, total: progress.append((sent, total)))
            self.assertEqual(result.read_text(), "text")
        self.assertEqual(sizes, [64 * 1024, 64 * 1024, 123])
        self.assertEqual(progress[0], (0, len(body)))
        self.assertEqual(progress[-1], (len(body), len(body)))


if __name__ == "__main__":
    unittest.main()
