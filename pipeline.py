"""
Пайплайн: скачать запись МТС Линк → транскрибировать → вернуть путь к .txt файлу.

По умолчанию используется быстрый режим — скачиваются только аудио чанки.
Если аудио чанки не найдены, автоматически падает на полное скачивание через mtslinker.
"""

import gc
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common import BASE_DIR, TRANSCRIBE_OPTIONS, format_timestamp, parse_mts_url
from cpu_tuning import CpuConfig, balanced_cpu_config, select_cpu_config
from media import extract_audio_chunk
from transcript_parts import commit_part, parts_directory

log = logging.getLogger(__name__)

BASE_DIR.mkdir(exist_ok=True)

# Синглтон модели — загружается один раз при первом запросе и переиспользуется
_model = None
_model_name: str | None = None
_model_workers = 1
_model_cpu_threads = 1
TRANSCRIPTION_CHUNK_SECONDS = 120
_GIB = 1024 ** 3
_WORKER_MEMORY_GIB = {
    "tiny": 0.6,
    "base": 0.9,
    "small": 1.5,
    "medium": 2.5,
    "large": 4.0,
    "turbo": 3.0,
}


def get_effective_model_name(model_name: str | None) -> str:
    """Возвращает модель, безопасную для локального CPU запуска."""
    if model_name and model_name.strip():
        return model_name.strip()
    return "medium"


def _get_cpu_count() -> int:
    cpu_count = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        try:
            cpu_count = min(cpu_count, len(os.sched_getaffinity(0)))
        except OSError:
            pass

    for quota_path, period_path in (
        ("/sys/fs/cgroup/cpu.max", None),
        ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
    ):
        try:
            if period_path is None:
                quota, period = Path(quota_path).read_text().split()
                if quota == "max":
                    continue
            else:
                quota = Path(quota_path).read_text().strip()
                period = Path(period_path).read_text().strip()
                if int(quota) < 0:  # cgroup v1: -1 means unlimited.
                    continue
            quota_cpus = max(1, int(quota) // int(period))
            cpu_count = min(cpu_count, quota_cpus)
        except (OSError, ValueError, ZeroDivisionError):
            continue
    return max(1, cpu_count // _job_capacity())


def _job_capacity() -> int:
    try:
        return max(1, int(os.getenv("SERVER_MAX_CONCURRENT_JOBS", "1")))
    except ValueError:
        return 1


def _get_available_memory_bytes() -> int | None:
    available = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
                break
    except (OSError, ValueError, IndexError):
        pass

    if available is None:
        try:
            available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError):
            return None

    for limit_path, usage_path in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        ("/sys/fs/cgroup/memory.high", "/sys/fs/cgroup/memory.current"),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        try:
            limit_text = Path(limit_path).read_text().strip()
            if limit_text == "max":
                continue
            limit = int(limit_text)
            if limit >= 1 << 60:
                continue
            usage = int(Path(usage_path).read_text().strip())
            available = min(available, max(0, limit - usage))
        except (OSError, ValueError):
            continue
    return available // _job_capacity()


def calculate_worker_count(
    model_name: str,
    cpu_count: int,
    available_memory_bytes: int | None,
) -> int:
    """Оценивает число одновременных транскрибаций по CPU и свободной RAM."""
    if available_memory_bytes is None:
        return 1

    model_key = get_effective_model_name(model_name).lower().split("/")[-1]
    for name in sorted(_WORKER_MEMORY_GIB, key=len, reverse=True):
        if name in model_key:
            worker_memory = _WORKER_MEMORY_GIB[name] * _GIB
            break
    else:
        worker_memory = _WORKER_MEMORY_GIB["medium"] * _GIB

    memory_reserve = _GIB
    memory_workers = max(1, int(max(0, available_memory_bytes - memory_reserve) // worker_memory))
    return max(1, min(max(1, cpu_count), memory_workers))


def get_worker_count(model_name: str) -> int:
    automatic = calculate_worker_count(
        model_name,
        cpu_count=_get_cpu_count(),
        available_memory_bytes=_get_available_memory_bytes(),
    )
    configured = os.getenv("WHISPER_WORKERS", "auto").strip().lower()
    if configured in {"", "auto"}:
        return automatic
    try:
        requested = int(configured)
    except ValueError:
        log.warning("Некорректное значение WHISPER_WORKERS=%r; использую авторасчёт", configured)
        return automatic
    return min(automatic, max(1, requested))


def clear_model() -> None:
    global _model, _model_name, _model_workers, _model_cpu_threads
    had_model = _model is not None
    _model = None
    _model_name = None
    _model_workers = 1
    _model_cpu_threads = 1
    if had_model:
        gc.collect()


def build_chunk_ranges(duration: float, chunk_seconds: int = 600) -> list[tuple[float, float]]:
    """Разбивает длительность на интервалы по chunk_seconds секунд."""
    if not math.isfinite(duration) or chunk_seconds <= 0:
        raise ValueError("Длительность должна быть конечной, размер чанка — положительным")
    if duration <= 0:
        return [(0.0, 0.0)]

    ranges: list[tuple[float, float]] = []
    start = 0.0
    while start < duration:
        end = min(start + chunk_seconds, duration)
        ranges.append((start, end))
        start = end
    return ranges


def get_default_cpu_config(cpu_count: int, worker_limit: int) -> CpuConfig:
    """Use the whole CPU budget unless the thread count is set explicitly."""
    worker_limit = max(1, min(cpu_count, worker_limit))
    configured = os.getenv("WHISPER_CPU_THREADS", "auto").strip().lower()
    if configured in {"", "auto"}:
        return balanced_cpu_config(cpu_count, worker_limit)
    try:
        threads = min(max(1, cpu_count // worker_limit), max(1, int(configured)))
        return CpuConfig(worker_limit, threads)
    except ValueError:
        log.warning("Некорректное WHISPER_CPU_THREADS=%r; использую авторасчёт", configured)
        return balanced_cpu_config(cpu_count, worker_limit)


def get_model(model_name: str, workers: int | None = None,
              cpu_threads: int | None = None):
    global _model, _model_name, _model_workers, _model_cpu_threads
    effective_model = get_effective_model_name(model_name)
    cpu_count = _get_cpu_count()
    workers = min(cpu_count, get_worker_count(effective_model) if workers is None else max(1, workers))
    if cpu_threads is None:
        config = get_default_cpu_config(cpu_count, workers)
        workers, cpu_threads = config.workers, config.threads
    else:
        cpu_threads = min(max(1, cpu_threads), max(1, cpu_count // workers))
    if (_model is None or _model_name != effective_model or _model_workers != workers
            or _model_cpu_threads != cpu_threads):
        clear_model()
        from faster_whisper import WhisperModel
        log.info(
            "Загрузка модели %s (параллельных вызовов: %s, CPU-потоков на вызов: %s)...",
            effective_model,
            workers,
            cpu_threads,
        )
        _model = WhisperModel(
            effective_model,
            device="cpu",
            compute_type="int8",
            num_workers=workers,
            cpu_threads=cpu_threads,
        )
        _model_name = effective_model
        _model_workers = workers
        _model_cpu_threads = cpu_threads
        log.info(
            "Для модели %s выбрано %s ворк. (CPU: %s, доступно RAM: %.1f ГБ)",
            effective_model,
            workers,
            cpu_count,
            (_get_available_memory_bytes() or 0) / _GIB,
        )
        log.info("Модель загружена.")
    return _model


def get_media_duration(input_path: str) -> float:
    """Prefer the duration of the same audio stream that Whisper decodes."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=duration:format=duration", "-of", "json", input_path],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return 0.0
    try:
        metadata = json.loads(result.stdout)
    except (ValueError, TypeError):
        return 0.0
    streams = metadata.get("streams", [])
    candidates = [streams[0].get("duration")] if streams else []
    candidates.append(metadata.get("format", {}).get("duration"))
    for value in candidates:
        try:
            duration = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(duration) and duration > 0:
            return duration
    return 0.0


def download_audio_fast(
    event_session_id: str,
    record_id: str | None,
    session_id: str | None,
    work_dir: str,
) -> str:
    """Скачивает только аудио чанки — быстрый режим."""
    from audio_downloader import download_audio_only
    return download_audio_only(event_session_id, record_id, session_id, work_dir)


def download_recording_full(
    event_session_id: str,
    record_id: str | None,
    session_id: str | None,
    work_dir: str,
) -> str:
    """
    Полное скачивание через mtslinker (видео + аудио).
    Используется как fallback если быстрый режим не нашёл аудио чанки.
    """
    work_dir = str(Path(work_dir).resolve())
    manifest = Path(work_dir) / "mts_result.json"
    request = {"event_session": event_session_id, "record_id": record_id,
               "session_id": session_id, "manifest": str(manifest)}
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("mts_fallback.py"))],
        input=json.dumps(request), text=True, cwd=work_dir, check=False,
    )
    if result.returncode != 0 or not manifest.is_file():
        raise RuntimeError("Резервное скачивание mtslinker завершилось с ошибкой; подробности в логе воркера")
    return json.loads(manifest.read_text(encoding="utf-8"))


def _format_minutes(seconds: float) -> str:
    return f"{seconds / 60:.1f}".replace(".", ",")


def _transcribe_one_file(
    input_path: str,
    model_name: str,
    output_path: str,
    chunk_number: int = 0,
    model_workers: int | None = None,
    model_cpu_threads: int | None = None,
    model=None,
) -> None:
    """Транскрибирует один файл и дописывает результат в output_path."""
    if model is None:
        model = get_model(model_name, workers=model_workers, cpu_threads=model_cpu_threads)
    log.info(f"Начинаю транскрибацию: {input_path}")

    segments, info = model.transcribe(
        input_path,
        **TRANSCRIBE_OPTIONS,
    )

    duration = info.duration
    log.info(f"Язык: {info.language} ({info.language_probability:.0%}), длительность: {format_timestamp(duration)}")

    with open(output_path, "w", encoding="utf-8") as f:
        for segment in segments:
            line = segment.text.strip()
            f.write(line + "\n")
            f.flush()
            log.debug(line)


def transcribe_file(
    input_path: str,
    model_name: str,
    output_path: str,
    update_status: Callable[[str], None] | None = None,
) -> str:
    """
    Транскрибирует файл через faster-whisper.
    Для длинных записей разбивает аудио на короткие задачи для равномерной загрузки workers.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    file_duration = get_media_duration(input_path)
    ranges = build_chunk_ranges(file_duration, chunk_seconds=TRANSCRIPTION_CHUNK_SECONDS)
    total_minutes = _format_minutes(file_duration)

    if len(ranges) > 1:
        log.info(f"Длительность файла {file_duration:.0f}s превышает лимит чанка, разбиваем на {len(ranges)} частей")
        tmp_dir = tempfile.mkdtemp(prefix="chunks_", dir=Path(output_path).parent)
        try:
            def extract_chunk(index: int, start: float, end: float) -> str:
                return extract_audio_chunk(input_path, str(Path(tmp_dir) / f"chunk_{index:06d}.wav"), start, end)

            extract_workers = min(_get_cpu_count(), len(ranges))
            log.info("Нарезаю аудио на %s чанков (параллельных задач: %s)", len(ranges), extract_workers)
            with ThreadPoolExecutor(max_workers=extract_workers) as executor:
                extraction_futures = [
                    executor.submit(extract_chunk, index, start, end)
                    for index, (start, end) in enumerate(ranges, start=1)
                ]
                chunk_files = [future.result() for future in extraction_futures]

            with open(output_path, "w", encoding="utf-8") as f:
                f.write("")

            workers = min(get_worker_count(model_name), len(ranges))
            cpu_count = _get_cpu_count()
            clear_model()
            config = select_cpu_config(
                get_effective_model_name(model_name), chunk_files, cpu_count,
                _get_available_memory_bytes(),
                get_default_cpu_config(cpu_count, workers),
                cache_dir=BASE_DIR / "cpu_tuning", update_status=update_status,
            )
            workers = config.workers
            model = get_model(model_name, workers=workers, cpu_threads=config.threads)
            if update_status:
                update_status(f"🎙 Готово частей: 0/{len(ranges)} · 0,0/{total_minutes} мин записи")
            completed_parts = 0
            completed_seconds = 0.0
            ready = set()
            next_part = 1

            def transcribe_part(number):
                return commit_part(output_path, number, lambda path: _transcribe_one_file(
                    input_path=chunk_files[number - 1], model_name=model_name,
                    output_path=path, chunk_number=number, model_workers=workers,
                    model_cpu_threads=config.threads, model=model,
                ))

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(transcribe_part, i): i for i in range(1, len(ranges) + 1)}
                with open(output_path, "w", encoding="utf-8") as output:
                    for future in as_completed(futures):
                        future.result()
                        number = futures[future]
                        ready.add(number)
                        completed_parts += 1
                        start, end = ranges[number - 1]
                        completed_seconds += end - start
                        # Publish a chronological prefix during processing. Retain
                        # committed parts so cancellation can rebuild without duplicates.
                        while next_part in ready:
                            part = parts_directory(output_path) / f"{next_part:06d}.txt"
                            with part.open(encoding="utf-8") as source:
                                shutil.copyfileobj(source, output)
                            next_part += 1
                        output.flush()
                        if update_status:
                            update_status(
                                f"🎙 Готово частей: {completed_parts}/{len(ranges)} · "
                                f"{_format_minutes(completed_seconds)}/{total_minutes} мин записи"
                            )
            shutil.rmtree(parts_directory(output_path), ignore_errors=True)

            log.info(f"Транскрипт сохранён: {output_path}")
            return output_path
        finally:
            clear_model()
            shutil.rmtree(tmp_dir, ignore_errors=True)

    try:
        model = get_model(model_name, workers=1)
        log.info(f"Начинаю транскрибацию: {input_path}")

        segments, info = model.transcribe(
            input_path,
            **TRANSCRIBE_OPTIONS,
        )

        duration = info.duration
        log.info(f"Язык: {info.language} ({info.language_probability:.0%}), длительность: {format_timestamp(duration)}")

        last_update = time.monotonic()
        update_interval = 15

        if update_status:
            update_status(f"🎙 Часть 1/1 · 0,0/{_format_minutes(duration)} мин записи")

        with open(output_path, "w", encoding="utf-8") as f:
            for segment in segments:
                line = segment.text.strip()
                f.write(line + "\n")
                f.flush()
                log.debug(line)

                now = time.monotonic()
                if update_status and duration and (now - last_update) >= update_interval:
                    update_status(f"🎙 Часть 1/1 · {_format_minutes(segment.end)}/{_format_minutes(duration)} мин записи")
                    last_update = now

        if update_status:
            duration_minutes = _format_minutes(duration)
            update_status(f"🎙 Готово частей: 1/1 · {duration_minutes}/{duration_minutes} мин записи")

        log.info(f"Транскрипт сохранён: {output_path}")
        return output_path
    finally:
        clear_model()


def run_pipeline_file(
    input_path: str,
    model_name: str,
    update_status: Callable[[str], None],
    output_path: str | None = None,
) -> str:
    """
    Пайплайн для уже скачанного файла: транскрибировать → вернуть путь к .txt.
    """
    if output_path is None:
        output_dir = BASE_DIR / "results"
        output_dir.mkdir(exist_ok=True)
        stem = Path(input_path).stem
        output_path = str(output_dir / f"transcript_{stem}.txt")

    update_status("🎙 Транскрибирую...")
    transcribe_file(input_path, model_name, output_path, update_status)

    return output_path


def run_pipeline(
    url: str,
    model_name: str,
    session_id: str | None,
    update_status: Callable[[str], None],
    output_path: str | None = None,
) -> str:
    """
    Полный пайплайн: скачать → транскрибировать → вернуть путь к .txt.

    Сначала пробует быстрый режим (только аудио чанки).
    Если не получилось — падает на полное скачивание через mtslinker.

    update_status — синхронный колбэк (text: str) для обновления статуса в Telegram.
    """
    event_session_id, record_id = parse_mts_url(url)
    log.info(f"event_session={event_session_id}, record_id={record_id}")

    output_dir = BASE_DIR / "results"
    output_dir.mkdir(exist_ok=True)
    final_path = output_path or str(output_dir / f"transcript_{event_session_id}.txt")
    Path(final_path).write_text("", encoding="utf-8")

    work_dir = tempfile.mkdtemp(prefix="job_", dir=Path(final_path).parent)
    try:
        # Шаг 1 — скачать
        update_status("⬇️ Скачиваю аудио дорожку...")
        try:
            media_path = download_audio_fast(event_session_id, record_id, session_id, work_dir)
            log.info(f"Быстрое скачивание успешно: {media_path}")
        except Exception as fast_error:
            log.warning(f"Быстрое скачивание не сработало: {fast_error}", exc_info=True)
            try:
                update_status("⬇️ Быстрый способ не сработал, пробую резервный режим...")
                media_path = download_recording_full(event_session_id, record_id, session_id, work_dir)
                log.info(f"Резервное скачивание успешно: {media_path}")
            except Exception as fallback_error:
                log.exception("Резервное скачивание тоже не сработало")
                raise RuntimeError(
                    f"Не удалось скачать аудио через быстрый и резервный путь: {fallback_error}"
                ) from fallback_error

        # Шаг 2 — транскрибировать
        update_status(f"🎙 Транскрибирую... (модель {model_name}, это займёт время)")
        transcript_path = final_path
        transcribe_file(media_path, model_name, transcript_path, update_status)
    finally:
        # Удаляем рабочую директорию после завершения
        shutil.rmtree(work_dir, ignore_errors=True)
        log.info(f"Рабочая директория удалена: {work_dir}")

    return final_path
