#!/usr/bin/env python3
"""HTTP API for remote speech transcription jobs."""

import hmac
import json
import logging
import multiprocessing
import os
import queue
import shutil
import signal
import threading
import time
import uuid
from decimal import Decimal, InvalidOperation
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from dotenv import load_dotenv

from common import BASE_DIR, SUPPORTED_EXTENSIONS, parse_mts_url
from pipeline import run_pipeline, run_pipeline_file
from process_control import configure_progress_lock, start_job_session, stop_process
from transcript_parts import assemble_partial

load_dotenv(Path(__file__).resolve().with_name(".env.server"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("worker_server")


def _upload_limit_gb() -> Decimal:
    try:
        amount = Decimal(os.getenv("MAX_UPLOAD_GB", "10"))
        if not amount.is_finite() or amount <= 0:
            raise ValueError
        size = amount * 1_000_000_000
        if size != size.to_integral_value():
            raise ValueError
        return amount
    except (InvalidOperation, ValueError):
        raise ValueError(
            "MAX_UPLOAD_GB должен быть положительным числом гигабайт. "
            "Например, 0.25 = 250 МБ (1 ГБ = 1000 МБ)."
        ) from None


API_KEY = os.getenv("PROCESSOR_API_KEY", "")
HOST = os.getenv("PROCESSOR_HOST", "127.0.0.1")
PORT = int(os.getenv("PROCESSOR_PORT", "8080"))
MTS_SESSION_ID = os.getenv("MTS_SESSION_ID", "")
DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "medium")
MAX_UPLOAD_GB = _upload_limit_gb()
MAX_CONCURRENT_JOBS = max(1, int(os.getenv("SERVER_MAX_CONCURRENT_JOBS", "1")))
JOB_TIMEOUT_SECONDS = max(1, int(os.getenv("TRANSCRIPTION_TIMEOUT_SECONDS", str(3 * 60 * 60))))
JOB_RETENTION_SECONDS = max(3600, int(os.getenv("JOB_RETENTION_SECONDS", str(24 * 60 * 60))))
MAX_PENDING_JOBS = max(MAX_CONCURRENT_JOBS, int(os.getenv("SERVER_MAX_PENDING_JOBS", "32")))

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
JOB_SLOTS = threading.Semaphore(MAX_CONCURRENT_JOBS)
JOBS_DIR = BASE_DIR / "remote_jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)


def _cleanup_expired_jobs() -> None:
    now = time.time()
    expired_paths = []
    with JOBS_LOCK:
        for job_id, job in list(JOBS.items()):
            if job["status"] in {"completed", "cancelled", "failed"} and now - job.get("finished_at", job["created_at"]) > JOB_RETENTION_SECONDS:
                JOBS.pop(job_id, None)
                expired_paths.append(JOBS_DIR / job_id)
        known_job_ids = set(JOBS)

    for job_dir in JOBS_DIR.iterdir():
        if job_dir.is_dir() and job_dir.name not in known_job_ids:
            try:
                if now - job_dir.stat().st_mtime > JOB_RETENTION_SECONDS:
                    expired_paths.append(job_dir)
            except OSError:
                continue
    for job_dir in expired_paths:
        shutil.rmtree(job_dir, ignore_errors=True)


def _pipeline_process_target(kind, args, output_path, status_queue, result_queue) -> None:
    start_job_session()
    configure_progress_lock()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    def process_status(message: str) -> None:
        status_queue.put(message)

    try:
        if kind == "file":
            result = run_pipeline_file(args[0], args[1], process_status, output_path=output_path)
        else:
            result = run_pipeline(
                args[0],
                args[1],
                MTS_SESSION_ID or None,
                process_status,
                output_path=output_path,
            )
        result_queue.put(("result", result))
    except Exception as error:
        log.exception("Пайплайн задания завершился с ошибкой")
        result_queue.put(("error", f"{type(error).__name__}: {error}"))


def _set_job(job_id: str, **values) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job:
            job.update(values)
            if values.get("status") in {"completed", "cancelled", "failed"}:
                job["finished_at"] = time.time()


def _run_job(job_id: str) -> None:
    job = JOBS[job_id]
    while not JOB_SLOTS.acquire(timeout=0.5):
        if job["cancel_event"].is_set():
            _set_job(job_id, status="cancelled", message="Задание отменено.")
            return

    process = None
    status_queue = result_queue = None
    try:
        if job["cancel_event"].is_set():
            _set_job(job_id, status="cancelled", message="Задание отменено.")
            return

        _set_job(job_id, status="running", message="Задание принято сервером.")
        context = multiprocessing.get_context("spawn")
        status_queue = context.Queue()
        result_queue = context.Queue()
        process = context.Process(
            target=_pipeline_process_target,
            args=(job["kind"], job["args"], job["output_path"], status_queue, result_queue),
            daemon=True,
        )
        process.start()
        deadline = time.monotonic() + JOB_TIMEOUT_SECONDS

        while process.is_alive():
            if job["cancel_event"].is_set():
                stop_process(process)
                assemble_partial(job["output_path"])
                _set_job(job_id, status="cancelled", message="Расшифровка остановлена.")
                return
            if time.monotonic() >= deadline:
                stop_process(process)
                assemble_partial(job["output_path"])
                _set_job(
                    job_id,
                    status="failed",
                    error=f"Превышено время обработки ({JOB_TIMEOUT_SECONDS // 3600} ч.)",
                )
                return
            try:
                _set_job(job_id, message=status_queue.get(timeout=0.5))
            except queue.Empty:
                pass

        process.join()
        while True:
            try:
                _set_job(job_id, message=status_queue.get_nowait())
            except queue.Empty:
                break

        try:
            result_type, result = result_queue.get(timeout=1)
        except queue.Empty:
            _set_job(job_id, status="failed", error=f"Процесс завершился без результата (код {process.exitcode})")
            return
        if result_type == "error":
            _set_job(job_id, status="failed", error=result)
        else:
            _set_job(job_id, status="completed", message="Расшифровка готова.")
    except Exception as error:
        log.exception("Ошибка обработки задания %s", job_id)
        _set_job(job_id, status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        if process is not None:
            stop_process(process)
            process.close()
        for result_queue_item in (status_queue, result_queue):
            if result_queue_item is not None:
                result_queue_item.close()
        JOB_SLOTS.release()


class ProcessorHandler(BaseHTTPRequestHandler):
    server_version = "SpeechParserProcessor/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, format_string, *args) -> None:
        log.info("%s - %s", self.address_string(), format_string % args)

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        if status >= 400:
            # Rejected uploads can still have unread bytes on the socket.
            self.close_connection = True
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        expected = f"Bearer {API_KEY}"
        return bool(API_KEY) and hmac.compare_digest(self.headers.get("Authorization", "").encode(), expected.encode())

    def _require_auth(self) -> bool:
        if self._authorized():
            return True
        self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "Требуется корректный API-ключ."})
        return False

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def handle(self):
        try:
            super().handle()
        except (TimeoutError, BrokenPipeError, ConnectionResetError):
            log.info("HTTP-клиент отключился или не завершил запрос вовремя")

    def _start_job(self, job_id):
        thread = threading.Thread(target=_run_job, args=(job_id,), daemon=True)
        _set_job(job_id, thread=thread)
        try:
            thread.start()
        except Exception:
            with JOBS_LOCK:
                JOBS.pop(job_id, None)
            shutil.rmtree(JOBS_DIR / job_id, ignore_errors=True)
            raise

    def _create_job(self, kind: str, args: tuple, *, job_id: str | None = None, start: bool = True) -> str:
        job_id = job_id or uuid.uuid4().hex
        job_dir = JOBS_DIR / job_id
        with JOBS_LOCK:
            pending = sum(job["status"] in {"uploading", "queued", "running"} for job in JOBS.values())
            if pending >= MAX_PENDING_JOBS:
                raise OverflowError("Очередь воркера заполнена. Повтори запрос позже.")
            try:
                job_dir.mkdir(parents=True)
                output_path = str(job_dir / "transcript.txt")
                Path(output_path).write_text("", encoding="utf-8")
                JOBS[job_id] = {
                    "kind": kind, "args": args,
                    "output_path": output_path, "status": "queued" if start else "uploading",
                    "message": "Задание в очереди сервера.", "error": None,
                    "created_at": time.time(), "cancel_event": threading.Event(),
                }
            except Exception:
                shutil.rmtree(job_dir, ignore_errors=True)
                raise
        if start:
            self._start_job(job_id)
        return job_id

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        if parsed.path == "/health":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return
        if not self._require_auth():
            return

        if parsed.path == "/limits":
            self._send_json(HTTPStatus.OK, {"max_upload_gb": str(MAX_UPLOAD_GB)})
            return

        parts = parsed.path.strip("/").split("/")
        if len(parts) == 2 and parts[0] == "jobs":
            with JOBS_LOCK:
                job = JOBS.get(parts[1])
                payload = {key: job.get(key) for key in ("status", "message", "error")} if job else None
            if payload is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Задание не найдено."})
            else:
                self._send_json(HTTPStatus.OK, payload)
            return

        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "result":
            with JOBS_LOCK:
                job = JOBS.get(parts[1])
                if job is None:
                    result_file = None
                else:
                    ready = job["status"] in {"completed", "cancelled"}
                    try:
                        result_file = open(job["output_path"], "rb") if ready else None  # noqa: SIM115 -- opened under lock, closed after streaming
                    except OSError:
                        result_file = None
            if job is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Задание не найдено."})
                return
            if result_file is None:
                self._send_json(HTTPStatus.CONFLICT, {"error": "Результат пока не готов."})
                return
            with result_file:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="transcript_{parts[1]}.txt"')
                self.send_header("Content-Length", str(os.fstat(result_file.fileno()).st_size))
                self.end_headers()
                while chunk := result_file.read(64 * 1024):
                    self.wfile.write(chunk)
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Неизвестный endpoint."})

    def do_POST(self) -> None:
        if not self._require_auth():
            return
        parsed = urlsplit(self.path)

        if parsed.path == "/jobs":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 64 * 1024:
                    raise ValueError("Некорректный размер запроса.")
                payload = json.loads(self.rfile.read(length))
                url = payload["url"]
                if not isinstance(url, str):
                    raise TypeError("Поле url должно быть строкой.")
                parse_mts_url(url)
                model = payload.get("model") or DEFAULT_MODEL
                if not isinstance(model, str):
                    raise TypeError("Поле model должно быть строкой.")
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            try:
                job_id = self._create_job("url", (url, model))
            except OverflowError as error:
                self._send_json(HTTPStatus.TOO_MANY_REQUESTS, {"error": str(error)})
                return
            self._send_json(HTTPStatus.ACCEPTED, {"job_id": job_id})
            return

        if parsed.path == "/jobs/file":
            query = parse_qs(parsed.query)
            filename = Path(unquote(query.get("filename", ["upload"])[0])).name
            extension = Path(filename).suffix.lower()
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if extension not in SUPPORTED_EXTENSIONS:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Формат файла не поддерживается."})
                return
            if length <= 0 or length > MAX_UPLOAD_GB * 1_000_000_000:
                self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {
                    "error": f"Файл пуст или превышает лимит {MAX_UPLOAD_GB.normalize():f} ГБ "
                             f"({(MAX_UPLOAD_GB * 1000).normalize():f} МБ). "
                             "Лимит задаётся в MAX_UPLOAD_GB на сервере.",
                })
                return

            job_id = uuid.uuid4().hex
            job_dir = JOBS_DIR / job_id
            input_path = job_dir / f"input{extension}"
            model = query.get("model", [DEFAULT_MODEL])[0]
            try:
                self._create_job("file", (str(input_path), model), job_id=job_id, start=False)
            except OverflowError as error:
                self._send_json(HTTPStatus.TOO_MANY_REQUESTS, {"error": str(error)})
                return
            remaining = length
            try:
                with input_path.open("wb") as uploaded_file:
                    while remaining:
                        if JOBS[job_id]["cancel_event"].is_set():
                            raise OSError("Загрузка отменена")
                        chunk = self.rfile.read(min(64 * 1024, remaining))
                        if not chunk:
                            raise OSError("Загрузка файла оборвалась.")
                        uploaded_file.write(chunk)
                        remaining -= len(chunk)
            except OSError as error:
                with JOBS_LOCK:
                    JOBS.pop(job_id, None)
                shutil.rmtree(job_dir, ignore_errors=True)
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return

            _set_job(job_id, status="queued")
            self._start_job(job_id)
            self._send_json(HTTPStatus.ACCEPTED, {"job_id": job_id})
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Неизвестный endpoint."})

    def do_DELETE(self) -> None:
        if not self._require_auth():
            return
        parsed = urlsplit(self.path)
        parts = parsed.path.strip("/").split("/")
        cancel_only = parse_qs(parsed.query).get("cancel_only") == ["1"]
        if len(parts) != 2 or parts[0] != "jobs":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Неизвестный endpoint."})
            return
        job_dir = None
        with JOBS_LOCK:
            job_id = parts[1]
            job = JOBS.get(job_id)
            if job and job["status"] in {"uploading", "queued", "running"}:
                job["cancel_event"].set()
                status = "cancellation_requested"
            elif job and cancel_only:
                status = job["status"]
            elif job:
                JOBS.pop(job_id)
                job_dir = JOBS_DIR / job_id
                status = "deleted"
        if job is None:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Задание не найдено."})
        elif job_dir is not None:
            shutil.rmtree(job_dir, ignore_errors=True)
            self._send_json(HTTPStatus.OK, {"status": status})
        else:
            self._send_json(HTTPStatus.ACCEPTED if status == "cancellation_requested" else HTTPStatus.OK,
                            {"status": status})


def main() -> None:
    if not API_KEY:
        raise RuntimeError("PROCESSOR_API_KEY не задан")
    def shutdown(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, shutdown)
    missing_tools = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing_tools:
        raise RuntimeError(
            f"Не найдены системные программы: {', '.join(missing_tools)}. "
            "Установи пакет ffmpeg (он включает ffprobe) и перезапусти обработчик."
        )
    _cleanup_expired_jobs()
    maintenance_stop = threading.Event()

    def maintenance():
        while not maintenance_stop.wait(60):
            try:
                _cleanup_expired_jobs()
            except OSError:
                log.warning("Не удалось очистить старые задания", exc_info=True)

    threading.Thread(target=maintenance, daemon=True).start()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    server = ThreadingHTTPServer((HOST, PORT), ProcessorHandler)
    log.info("Удалённый обработчик запущен на %s:%s; одновременно заданий: %s", HOST, PORT, MAX_CONCURRENT_JOBS)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Остановка обработчика")
    finally:
        maintenance_stop.set()
        with JOBS_LOCK:
            active_jobs = list(JOBS.values())
            for job in active_jobs:
                job["cancel_event"].set()
        deadline = time.monotonic() + 12
        for job in active_jobs:
            thread = job.get("thread")
            if thread:
                thread.join(timeout=max(0, deadline - time.monotonic()))
        server.server_close()


if __name__ == "__main__":
    main()
