#!/usr/bin/env python3
"""Send local media or an MTS Link recording to a remote transcription worker."""

import argparse
import math
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

from common import SUPPORTED_EXTENSIONS, TranscriptionCancelled, parse_mts_url
from processor_client import ProcessorClient


def _positive_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Нужно положительное число секунд") from None
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("Нужно положительное число секунд")
    return seconds


def _status(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _publish(source: Path, output: Path, force: bool) -> None:
    if force:
        source.replace(output)
    else:
        # Same filesystem: publish atomically and never overwrite an existing
        # transcript, even when another CLI finishes while this job is running.
        os.link(source, output)


def main(argv=None) -> int:
    load_dotenv(Path(__file__).resolve().with_name(".env"))
    parser = argparse.ArgumentParser(description=(
        "Расшифровка аудио/видео на удалённом воркере без Telegram. "
        "Настройки подключения берутся из .env или окружения."
    ))
    parser.add_argument("input", help="Путь к аудио/видео файлу или ссылка на запись МТС Линк")
    parser.add_argument("-o", "--output", type=Path, help="Путь к результату .txt")
    parser.add_argument("--model", default=os.getenv("WHISPER_MODEL", "medium"), help="Модель Whisper")
    parser.add_argument("--url", default=os.getenv("PROCESSOR_URL", ""), help="URL воркера (PROCESSOR_URL)")
    parser.add_argument("--timeout", type=_positive_seconds,
                        default=os.getenv("TRANSCRIPTION_TIMEOUT_SECONDS", "10800"),
                        help="Общий лимит в секундах, включая загрузку и очередь (по умолчанию 10800)")
    parser.add_argument("--force", action="store_true", help="Заменить существующий файл результата")
    options = parser.parse_args(argv)

    try:
        client = ProcessorClient(options.url, os.getenv("PROCESSOR_API_KEY", ""), options.timeout)
        if options.input.startswith("https://") or options.input.startswith("http://"):
            session, record = parse_mts_url(options.input)
            kind, source = "url", options.input
            default_output = Path(f"transcript_{session}{'_' + record if record else ''}.txt")
        else:
            source = Path(options.input).expanduser().resolve()
            if not source.is_file():
                raise ValueError(f"Файл не найден: {source}")
            if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(f"Неподдерживаемый формат. Доступны: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
            if source.stat().st_size == 0:
                raise ValueError("Нельзя отправить пустой файл")
            kind, default_output = "file", source.with_suffix(".txt")
        output = (options.output or default_output).expanduser().resolve()
        if kind == "file" and output == source:
            raise ValueError("Путь результата совпадает с исходным файлом")
        if output.exists() and not options.force:
            raise ValueError(f"Файл уже существует: {output}. Укажи другой --output или --force.")
        if output.is_dir():
            raise ValueError(f"Путь результата является каталогом: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
    except (ValueError, OSError, RuntimeError) as error:
        _status(f"Ошибка: {error}")
        return 1

    stop_requested = False
    remote_id = None
    last_progress = -math.inf

    def interrupt(_signum, _frame):
        nonlocal stop_requested
        if stop_requested:
            raise KeyboardInterrupt
        stop_requested = True

    def submitted(job_id):
        nonlocal remote_id
        remote_id = job_id
        _status(f"Задание принято: {job_id}. Ctrl+C — остановить и сохранить готовую часть.")

    def progress(sent, total):
        nonlocal last_progress
        now = time.monotonic()
        if sent == total or now - last_progress >= 2:
            _status(f"Загрузка: {sent / total:.0%} · {sent / 1_000_000:.1f}/{total / 1_000_000:.1f} МБ")
            last_progress = now

    previous_handler = signal.signal(signal.SIGINT, interrupt)
    try:
        # Keep existing output intact on failures, timeouts and empty cancels.
        with tempfile.TemporaryDirectory(prefix=".speechparser-", dir=output.parent) as directory:
            result = Path(directory) / "transcript.txt"
            try:
                client.run(kind, (str(source), options.model), result, _status,
                           cancelled=lambda: stop_requested, on_submitted=submitted,
                           on_upload_progress=progress)
            except TranscriptionCancelled:
                if result.exists() and result.stat().st_size:
                    _publish(result, output, options.force)
                    _status("Остановлено. Частичная расшифровка сохранена:")
                    print(output)
                else:
                    _status("Остановлено. Готового текста пока нет.")
                return 130
            _publish(result, output, options.force)
            _status("Расшифровка сохранена:")
            print(output)
            return 0
    except httpx.HTTPStatusError as error:
        response = error.response
        detail = {
            401: "Проверь PROCESSOR_API_KEY: воркер отклонил авторизацию.",
            403: "Нет доступа к воркеру.",
            404: "Endpoint или задание не найдено; возможно, воркер перезапускался.",
            413: "Файл превышает лимит воркера или reverse proxy. Проверь MAX_UPLOAD_GB и лимит прокси.",
            429: "Очередь воркера заполнена. Повтори запуск позже.",
        }.get(response.status_code, "Ошибка удалённого воркера.")
        _status(f"Ошибка HTTP {response.status_code}: {detail}")
        return 1
    except (httpx.RequestError, httpx.InvalidURL) as error:
        _status(f"Ошибка соединения с воркером ({type(error).__name__}). Проверь PROCESSOR_URL и SSH-туннель.")
        return 1
    except (ValueError, OSError, RuntimeError) as error:
        _status(f"Ошибка: {error}")
        return 1
    except KeyboardInterrupt:
        _status("Ожидание остановлено.")
        return 130
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        if remote_id:
            client.cleanup(remote_id)


if __name__ == "__main__":
    sys.exit(main())
