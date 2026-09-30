#!/usr/bin/env python3
"""HTTP API for remote speech transcription jobs."""

import hmac
import json
import logging
import multiprocessing
import os
import queue
import shutil
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from dotenv import load_dotenv

from pipeline import BASE_DIR, parse_mts_url, run_pipeline, run_pipeline_file

load_dotenv(Path(__file__).resolve().with_name(".env.server"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("worker_server")

API_KEY = os.getenv("PROCESSOR_API_KEY", "")
HOST = os.getenv("PROCESSOR_HOST", "0.0.0.0")
PORT = int(os.getenv("PROCESSOR_PORT", "8080"))
MTS_SESSION_ID = os.getenv("MTS_SESSION_ID", "")
DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "medium")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))
MAX_CONCURRENT_JOBS = max(1, int(os.getenv("SERVER_MAX_CONCURRENT_JOBS", "1")))
JOB_TIMEOUT_SECONDS = max(1, int(os.getenv("TRANSCRIPTION_TIMEOUT_SECONDS", str(3 * 60 * 60))))
JOB_RETENTION_SECONDS = max(3600, int(os.getenv("JOB_RETENTION_SECONDS", str(24 * 60 * 60))))
SUPPORTED_EXTENSIONS = {".mp4", ".mp3", ".wav", ".m4a", ".ogg", ".webm", ".mkv", ".avi", ".mov"}

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
            if job["status"] in {"completed", "cancelled", "failed"} and now - job["created_at"] > JOB_RETENTION_SECONDS:
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
        result_queue.put(("error", f"{type(error).__name__}: {error}"))


def _set_job(job_id: str, **values) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job:
            job.update(values)


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
        _set_job(job_id, process=process)
        process.start()
        deadline = time.monotonic() + JOB_TIMEOUT_SECONDS

        while process.is_alive():
            if job["cancel_event"].is_set():
                process.terminate()
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
                    process.join()
                _set_job(job_id, status="cancelled", message="Расшифровка остановлена.")
                return
            if time.monotonic() >= deadline:
                process.terminate()
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
                    process.join()
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
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
            process.close()
        for result_queue_item in (status_queue, result_queue):
            if result_queue_item is not None:
                result_queue_item.close()
        JOB_SLOTS.release()


class ProcessorHandler(BaseHTTPRequestHandler):
    server_version = "SpeechParserProcessor/1.0"

    def log_message(self, format_string, *args) -> None:
        log.info("%s - %s", self.address_string(), format_string % args)

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        expected = f"Bearer {API_KEY}"
        return bool(API_KEY) and hmac.compare_digest(self.headers.get("Authorization", ""), expected)

    def _require_auth(self) -> bool:
        if self._authorized():
            return True
        self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "Требуется корректный API-ключ."})
        return False

    def _create_job(self, kind: str, args: tuple, input_path: str | None = None) -> str:
        _cleanup_expired_jobs()
        job_id = uuid.uuid4().hex
        job_dir = JOBS_DIR / job_id
        job_dir.mkdir(parents=True)
        output_path = str(job_dir / "transcript.txt")
        Path(output_path).write_text("", encoding="utf-8")
        with JOBS_LOCK:
            JOBS[job_id] = {
                "kind": kind,
                "args": args,
                "input_path": input_path,
                "output_path": output_path,
                "status": "queued",
                "message": "Задание в очереди сервера.",
                "error": None,
                "created_at": time.time(),
                "cancel_event": threading.Event(),
                "process": None,
            }
        threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()
        return job_id

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        if parsed.path == "/health":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return
        if not self._require_auth():
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
                result_path = Path(job["output_path"]) if job else None
                result_status = job["status"] if job else None
            if result_path is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Задание не найдено."})
                return
            if result_status not in {"completed", "cancelled"} or not result_path.is_file():
                self._send_json(HTTPStatus.CONFLICT, {"error": "Результат пока не готов."})
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="transcript_{parts[1]}.txt"')
            self.send_header("Content-Length", str(result_path.stat().st_size))
            self.end_headers()
            with result_path.open("rb") as result_file:
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
                    raise ValueError("Поле url должно быть строкой.")
                parse_mts_url(url)
                model = str(payload.get("model") or DEFAULT_MODEL)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            job_id = self._create_job("url", (url, model))
            self._send_json(HTTPStatus.ACCEPTED, {"job_id": job_id})
            return

        if parsed.path == "/jobs/file":
            _cleanup_expired_jobs()
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
            if length <= 0 or length > MAX_UPLOAD_BYTES:
                self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "Некорректный размер файла."})
                return

            job_id = uuid.uuid4().hex
            job_dir = JOBS_DIR / job_id
            job_dir.mkdir(parents=True)
            input_path = job_dir / f"input{extension}"
            remaining = length
            try:
                with input_path.open("wb") as uploaded_file:
                    while remaining:
                        chunk = self.rfile.read(min(64 * 1024, remaining))
                        if not chunk:
                            raise OSError("Загрузка файла оборвалась.")
                        uploaded_file.write(chunk)
                        remaining -= len(chunk)
            except OSError as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return

            model = query.get("model", [DEFAULT_MODEL])[0]
            output_path = job_dir / "transcript.txt"
            output_path.write_text("", encoding="utf-8")
            with JOBS_LOCK:
                JOBS[job_id] = {
                    "kind": "file",
                    "args": (str(input_path), model),
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "status": "queued",
                    "message": "Задание в очереди сервера.",
                    "error": None,
                    "created_at": time.time(),
                    "cancel_event": threading.Event(),
                    "process": None,
                }
            threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()
            self._send_json(HTTPStatus.ACCEPTED, {"job_id": job_id})
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Неизвестный endpoint."})

    def do_DELETE(self) -> None:
        if not self._require_auth():
            return
        parts = urlsplit(self.path).path.strip("/").split("/")
        if len(parts) != 2 or parts[0] != "jobs":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Неизвестный endpoint."})
            return
        with JOBS_LOCK:
            job = JOBS.get(parts[1])
            if job and job["status"] in {"queued", "running"}:
                job["cancel_event"].set()
        if job is None:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Задание не найдено."})
        else:
            self._send_json(HTTPStatus.ACCEPTED, {"status": "cancellation_requested"})


def main() -> None:
    if not API_KEY:
        raise RuntimeError("PROCESSOR_API_KEY не задан")
    missing_tools = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing_tools:
        raise RuntimeError(
            f"Не найдены системные программы: {', '.join(missing_tools)}. "
            "Установи пакет ffmpeg (он включает ffprobe) и перезапусти обработчик."
        )
    _cleanup_expired_jobs()
    server = ThreadingHTTPServer((HOST, PORT), ProcessorHandler)
    log.info("Удалённый обработчик запущен на %s:%s; одновременно заданий: %s", HOST, PORT, MAX_CONCURRENT_JOBS)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Остановка обработчика")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()