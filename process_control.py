"""Stop a job and its subprocesses, including FFmpeg and CPU probes."""

import os
import signal
import threading


def configure_progress_lock() -> None:
    """Progress bars in a job share threads only. SIGTERM can strand an mp.RLock."""
    from tqdm import tqdm

    tqdm.set_lock(threading.RLock())


def start_job_session() -> None:
    if hasattr(os, "setsid"):
        os.setsid()


def stop_process(process, timeout: float = 5) -> None:
    if process.pid is None:
        return
    if hasattr(os, "killpg"):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            if process.is_alive():
                process.terminate()
        process.join(timeout)
        # The parent may have exited while FFmpeg is still shutting down.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.is_alive():
        process.terminate()
        process.join(timeout)
    if process.is_alive():
        process.kill()
    process.join()
