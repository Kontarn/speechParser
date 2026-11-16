import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from decimal import Decimal
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import httpx

import worker_server
from process_control import start_job_session
from transcript_parts import commit_part


class UploadLimitTests(unittest.TestCase):
    def limit(self, environment):
        with patch.dict(os.environ, environment, clear=True):
            return worker_server._upload_limit_gb()

    def test_decimal_gigabytes_preserve_exact_value(self):
        for value in ("0.25", "1", "1.5", "0.000000033", "0.000000001"):
            with self.subTest(value=value):
                self.assertEqual(self.limit({"MAX_UPLOAD_GB": value}), Decimal(value))

    def test_default_is_ten_decimal_gigabytes(self):
        self.assertEqual(self.limit({}), Decimal(10))

    def test_invalid_limits_fail_with_setting_name(self):
        for value in ("", "0", "-1", "NaN", "sNaN", "Infinity", "oops", "0.0000000001"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "MAX_UPLOAD_GB"):
                self.limit({"MAX_UPLOAD_GB": value})


def cancellable_pipeline(kind, args, output_path, status_queue, result_queue):
    """Real spawn target used to verify cancellation across a process group."""
    start_job_session()
    commit_part(output_path, 1, lambda path: Path(path).write_text("ready part\n"))
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    Path(args[0]).write_text(str(child.pid))
    status_queue.put("ready")
    try:
        child.wait()
    finally:
        child.kill()
        child.wait()


class WorkerApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for name, value in (("JOBS_DIR", self.root), ("JOBS", {}), ("API_KEY", "test-key")):
            setting = patch.object(worker_server, name, value)
            setting.start()
            self.addCleanup(setting.stop)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), worker_server.ProcessorHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.client = httpx.Client(base_url=f"http://127.0.0.1:{self.server.server_port}",
                                   headers={"Authorization": "Bearer test-key"})
        self.addCleanup(self.stop_server)

    def stop_server(self):
        self.client.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_capacity_rejects_excess_jobs_without_creating_files(self):
        with patch.object(worker_server, "MAX_PENDING_JOBS", 1), patch.object(worker_server, "_run_job"):
            response = self.client.post("/jobs", json={"url": "https://my.mts-link.ru/1/2/record-new/3"})
            self.assertEqual(response.status_code, 202)
            response = self.client.post("/jobs", json={"url": "https://my.mts-link.ru/1/2/record-new/4"})
            self.assertEqual(response.status_code, 429)
        self.assertEqual(len(worker_server.JOBS), 1)
        self.assertEqual(len(list(self.root.iterdir())), 1)

    def test_truncated_upload_removes_job_and_partial_input(self):
        with socket.create_connection(("127.0.0.1", self.server.server_port), timeout=3) as client:
            client.sendall(b"POST /jobs/file?filename=clip.wav HTTP/1.0\r\n"
                           b"Authorization: Bearer test-key\r\nContent-Length: 20\r\n\r\nabc")
            client.shutdown(socket.SHUT_WR)
            data = bytearray()
            while chunk := client.recv(4096):
                data.extend(chunk)
        self.assertIn(b"400 Bad Request", data)
        self.assertEqual(worker_server.JOBS, {})
        self.assertFalse(list(self.root.iterdir()))

    def test_quarter_gigabyte_limit_rejects_250_mb_plus_one_byte(self):
        with patch.dict(os.environ, {"MAX_UPLOAD_GB": "0.25"}):
            maximum = worker_server._upload_limit_gb()
        with patch.object(worker_server, "MAX_UPLOAD_GB", maximum):
            response = self.client.get("/limits")
            self.assertEqual(response.json(), {"max_upload_gb": "0.25"})
            # Only headers are needed: oversized files must be rejected before
            # any body is transferred or a job directory is created.
            with socket.create_connection(("127.0.0.1", self.server.server_port), timeout=3) as client:
                client.sendall(b"POST /jobs/file?filename=clip.mp4 HTTP/1.1\r\nHost: localhost\r\n"
                               b"Authorization: Bearer test-key\r\nContent-Length: 250000001\r\n\r\n")
                data = bytearray()
                while chunk := client.recv(4096):
                    data.extend(chunk)
        self.assertIn(b"413", data)
        self.assertIn("0.25 ГБ (250 МБ)", data.decode("utf-8"))
        self.assertEqual(worker_server.JOBS, {})
        self.assertFalse(list(self.root.iterdir()))

    def test_retention_counts_from_completion_not_submission(self):
        directory = self.root / "job"
        directory.mkdir()
        worker_server.JOBS["job"] = {
            "status": "completed", "created_at": 1, "finished_at": time.time(),
        }
        worker_server._cleanup_expired_jobs()
        self.assertTrue(directory.exists())
        worker_server.JOBS["job"]["finished_at"] = 1
        worker_server._cleanup_expired_jobs()
        self.assertFalse(directory.exists())
        self.assertNotIn("job", worker_server.JOBS)

    def test_invalid_model_type_is_rejected(self):
        response = self.client.post("/jobs", json={"url": "https://my.mts-link.ru/1/2/record-new/3", "model": ["medium"]})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(worker_server.JOBS, {})


@unittest.skipUnless(sys.platform == "linux", "Process-group cancellation requires Linux")
class ProcessLifecycleTests(unittest.TestCase):
    def test_cancellation_stops_descendant_and_preserves_committed_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = str(root / "result.txt")
            child_pid_path = root / "child.pid"
            cancel = threading.Event()
            job = {"kind": "file", "args": (str(child_pid_path), "fake"),
                   "output_path": output, "cancel_event": cancel, "status": "queued"}
            with (
                patch.object(worker_server, "JOBS", {"job": job}),
                patch.object(worker_server, "JOB_SLOTS", threading.Semaphore(1)),
                patch.object(worker_server, "_pipeline_process_target", cancellable_pipeline),
            ):
                runner = threading.Thread(target=worker_server._run_job, args=("job",), daemon=True)
                runner.start()
                try:
                    deadline = time.monotonic() + 8
                    while not child_pid_path.exists() and time.monotonic() < deadline:
                        time.sleep(0.02)
                    self.assertTrue(child_pid_path.exists(), "spawn target did not start")
                finally:
                    cancel.set()
                    runner.join(timeout=8)
                self.assertFalse(runner.is_alive())
                self.assertEqual(job["status"], "cancelled")
                self.assertEqual(Path(output).read_text(), "ready part\n")
                child_pid = int(child_pid_path.read_text())
                status = Path(f"/proc/{child_pid}/stat")
                # A killed descendant can briefly be a zombie awaiting init's reap.
                if status.exists():
                    self.assertEqual(status.read_text().rsplit(")", 1)[1].split()[0], "Z")


if __name__ == "__main__":
    unittest.main()
