"""HTTP worker protocol, independent of Telegram and transcription libraries."""

import logging
import os
import time
from decimal import Decimal
from pathlib import Path

import httpx

from common import TranscriptionCancelled

log = logging.getLogger(__name__)


class ProcessorClient:
    def __init__(self, url: str, api_key: str, job_timeout: float, poll_interval: float = 2):
        if not url or not api_key:
            raise RuntimeError("Задай PROCESSOR_URL и PROCESSOR_API_KEY для удалённого обработчика")
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.job_timeout = job_timeout
        self.poll_interval = poll_interval

    def _client(self):
        # Socket timeouts are independent of the hours-long processing deadline.
        return httpx.Client(headers={"Authorization": f"Bearer {self.api_key}"},
                            timeout=httpx.Timeout(30, connect=5))

    def cleanup(self, remote_job_id: str) -> bool:
        try:
            with self._client() as client:
                response = client.delete(f"{self.url}/jobs/{remote_job_id}")
                if response.status_code != 404:
                    response.raise_for_status()
            return True
        except httpx.HTTPError as error:
            log.warning("Не удалось очистить удалённое задание %s (%s)",
                        remote_job_id, type(error).__name__)
            return False

    @staticmethod
    def _download(client, url: str, output_path: Path, allow_missing: bool, timeout):
        temporary = output_path.with_suffix(output_path.suffix + ".download")
        try:
            with client.stream("GET", url, timeout=timeout()) as response:
                if allow_missing and response.status_code in {404, 409}:
                    return
                response.raise_for_status()
                with temporary.open("wb") as output:
                    for chunk in response.iter_bytes(64 * 1024):
                        timeout()
                        output.write(chunk)
            temporary.replace(output_path)
        finally:
            temporary.unlink(missing_ok=True)

    def run(self, kind: str, args: tuple, output_path: Path, update_status,
            cancelled=lambda: False, on_submitted=lambda job_id: None,
            on_upload_progress=lambda sent, total: None) -> str:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # An old transcript must never become the partial result of a new job.
        output_path.unlink(missing_ok=True)
        deadline = time.monotonic() + self.job_timeout

        def timeout():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Превышено время обработки задания")
            return httpx.Timeout(min(30, remaining), connect=min(5, remaining))

        if cancelled():
            raise TranscriptionCancelled
        with self._client() as client:
            if kind == "url":
                response = client.post(f"{self.url}/jobs", json={"url": args[0], "model": args[1]},
                                       timeout=timeout())
            elif kind == "file":
                path, model = args
                with open(path, "rb") as source:
                    size = os.fstat(source.fileno()).st_size
                    # Older workers have no limits endpoint; their POST still
                    # enforces its size limit. Never retry the upload itself.
                    limits = client.get(f"{self.url}/limits", timeout=timeout())
                    if limits.status_code != 404:
                        limits.raise_for_status()
                        maximum_gb = Decimal(str(limits.json()["max_upload_gb"]))
                        if size > maximum_gb * 1_000_000_000:
                            raise ValueError(
                                f"Файл: {size / 1_000_000:g} МБ; "
                                f"лимит воркера: {maximum_gb.normalize():f} ГБ "
                                f"({(maximum_gb * 1000).normalize():f} МБ). "
                                "Увеличь MAX_UPLOAD_GB на сервере и перезапусти воркер."
                            )

                    def upload():
                        sent = 0
                        on_upload_progress(sent, size)
                        while chunk := source.read(64 * 1024):
                            timeout()
                            if cancelled():
                                raise TranscriptionCancelled
                            yield chunk
                            sent += len(chunk)
                            on_upload_progress(sent, size)

                    response = client.post(
                        f"{self.url}/jobs/file", params={"filename": Path(path).name, "model": model},
                        content=upload(), headers={"Content-Type": "application/octet-stream",
                                                  "Content-Length": str(size)},
                        timeout=timeout(),
                    )
            else:
                raise ValueError(f"Неизвестный тип задания: {kind}")
            response.raise_for_status()
            remote_id = response.json()["job_id"]
            on_submitted(remote_id)
            job_url = f"{self.url}/jobs/{remote_id}"
            cancel_sent = False
            last_status = None
            try:
                while True:
                    if cancelled() and not cancel_sent:
                        # Cancelling a job that just completed must not delete
                        # its transcript before the result can be downloaded.
                        client.delete(job_url, params={"cancel_only": "1"}, timeout=timeout()).raise_for_status()
                        cancel_sent = True
                    response = client.get(job_url, timeout=timeout())
                    response.raise_for_status()
                    status = response.json()
                    if status.get("message") and status["message"] != last_status:
                        update_status(status["message"])
                        last_status = status["message"]
                    state = status["status"]
                    if state in {"completed", "cancelled"}:
                        self._download(client, job_url + "/result", output_path,
                                       state == "cancelled", timeout)
                        if state == "cancelled":
                            raise TranscriptionCancelled
                        return str(output_path)
                    if state == "failed":
                        raise RuntimeError(status.get("error") or "Удалённый обработчик завершился с ошибкой")
                    time.sleep(min(self.poll_interval, max(0, deadline - time.monotonic())))
            except TimeoutError:
                # Cancellation must not replace the actual deadline error.
                try:
                    client.delete(job_url, params={"cancel_only": "1"}, timeout=5)
                except httpx.HTTPError:
                    log.warning("Не удалось отменить просроченное задание %s", remote_id)
                raise
