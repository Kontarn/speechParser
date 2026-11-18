import asyncio
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import bot


@unittest.skipUnless(sys.platform == "linux", "Tests POSIX signals and named semaphores")
class StopFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_stop_before_first_word_then_interrupt_worker(self):
        # The spawned process runs the normal worker target. Replace only the
        # external recording/ASR with a cancellable stage that uses tqdm.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "worker_fixture.py"
            script.write_text(textwrap.dedent("""\
                import time
                from pathlib import Path
                import worker_server as worker
                from tqdm import tqdm

                class Server(worker.ThreadingHTTPServer):
                    def server_activate(self):
                        super().server_activate()
                        Path("port").write_text(str(self.server_port))

                def pipeline(*args, **kwargs):
                    with tqdm(total=10, disable=True):
                        Path("ready").touch()
                        time.sleep(60)

                worker.run_pipeline = pipeline
                if __name__ == "__main__":
                    worker.ThreadingHTTPServer = Server
                    worker.JOBS_DIR = Path.cwd()
                    worker.main()
            """))
            env = dict(os.environ, PYTHONPATH=str(Path(bot.__file__).parent),
                       PROCESSOR_API_KEY="test", PROCESSOR_HOST="127.0.0.1", PROCESSOR_PORT="0")
            with subprocess.Popen([sys.executable, str(script)], cwd=root, env=env,
                                  stderr=subprocess.PIPE, text=True) as server:
                try:
                    async def wait_for_file(name):
                        while not (root / name).exists() or (name == "port" and not (root / name).stat().st_size):
                            if server.poll() is not None:
                                self.fail(f"Worker exited with code {server.returncode}")
                            await asyncio.sleep(0.02)

                    await asyncio.wait_for(wait_for_file("port"), 8)
                    update = Mock()
                    update.message.text = "https://my.mts-link.ru/1/2/record-new/3"
                    update.message.reply_text = AsyncMock(return_value=Mock(message_id=10))
                    update.effective_chat.id = update.effective_user.id = 1
                    telegram = Mock(edit_message_text=AsyncMock(), send_message=AsyncMock(),
                                    send_document=AsyncMock())
                    with (
                        patch.object(bot, "ALLOWED_USER_ID", 1),
                        patch.object(bot, "BASE_DIR", root),
                        patch.object(bot, "PROCESSOR_URL", "http://127.0.0.1:" + (root / "port").read_text()),
                        patch.object(bot, "PROCESSOR_API_KEY", "test"),
                        patch.object(bot, "REMOTE_POLL_INTERVAL_SECONDS", 0.05),
                        patch.object(bot, "JOB_STATUS", {}),
                        patch.object(bot, "JOB_QUEUE", []),
                    ):
                        await bot.handle_message(update, Mock(bot=telegram))
                        await asyncio.wait_for(wait_for_file("ready"), 8)
                        await bot.cmd_stop(update, Mock(bot=telegram))
                        job, = bot.JOB_STATUS.values()
                        await asyncio.to_thread(job["thread"].join, 8)
                        self.assertFalse(job["thread"].is_alive())
                        self.assertEqual(job["status"], "cancelled")
                        self.assertIsNone(job["remote_job_id"])
                        telegram.send_message.assert_awaited_once()
                        self.assertIn("готового текста пока нет", telegram.send_message.await_args.kwargs["text"])
                        telegram.send_document.assert_not_awaited()
                    server.send_signal(signal.SIGINT)
                    _, stderr = await asyncio.to_thread(server.communicate, timeout=8)
                    self.assertEqual(server.returncode, 0, stderr)
                    self.assertNotIn("leaked semaphore", stderr)
                finally:
                    if server.poll() is None:
                        server.terminate()
                        server.wait(timeout=10)
