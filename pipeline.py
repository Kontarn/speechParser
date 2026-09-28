"""
Пайплайн: скачать запись МТС Линк → транскрибировать → вернуть путь к .txt файлу.

По умолчанию используется быстрый режим — скачиваются только аудио чанки.
Если аудио чанки не найдены, автоматически падает на полное скачивание через mtslinker.
"""

import gc
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger(__name__)

# Базовая директория для временных файлов и результатов
BASE_DIR = Path(__file__).parent / "tmp"
BASE_DIR.mkdir(exist_ok=True)

# Паттерн для разбора URL МТС Линк
# https://my.mts-link.ru/{org_id}/{room_id}/record-new/{event_session_id}
# https://my.mts-link.ru/{org_id}/{room_id}/record-new/{event_session_id}/record-file/{record_id}
MTS_URL_PATTERN = re.compile(
    r"https://my\.mts-link\.ru/\d+/\d+/record-new/(?P<event_session>\d+)"
    r"(?:/record-file/(?P<record_id>\d+))?"
)

# Синглтон модели — загружается один раз при первом запросе и переиспользуется
_model = None
_model_name: Optional[str] = None
_model_workers = 1
_GIB = 1024 ** 3
_WORKER_MEMORY_GIB = {
    "tiny": 0.6,
    "base": 0.9,
    "small": 1.5,
    "medium": 2.5,
    "large": 4.0,
    "turbo": 3.0,
}


def get_effective_model_name(model_name: Optional[str]) -> str:
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
            quota_cpus = max(1, int(quota) // int(period))
            cpu_count = min(cpu_count, quota_cpus)
        except (OSError, ValueError, ZeroDivisionError):
            continue
    return max(1, cpu_count)


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
    return available


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
    global _model, _model_name, _model_workers
    _model = None
    _model_name = None
    _model_workers = 1
    gc.collect()


def build_chunk_ranges(duration: float, chunk_seconds: int = 600) -> list[tuple[float, float]]:
    """Разбивает длительность на интервалы по chunk_seconds секунд."""
    if duration <= 0:
        return [(0.0, 0.0)]

    ranges: list[tuple[float, float]] = []
    start = 0.0
    while start < duration:
        end = min(start + chunk_seconds, duration)
        ranges.append((start, end))
        start = end
    return ranges


def get_model(model_name: str):
    global _model, _model_name, _model_workers
    effective_model = get_effective_model_name(model_name)
    workers = get_worker_count(effective_model)
    if _model is None or _model_name != effective_model or _model_workers != workers:
        from faster_whisper import WhisperModel
        log.info(f"Загрузка модели {effective_model} (потоки: {workers})...")
        _model = WhisperModel(
            effective_model,
            device="cpu",
            compute_type="int8",
            num_workers=workers,
            cpu_threads=max(1, _get_cpu_count() // workers),
        )
        _model_name = effective_model
        _model_workers = workers
        log.info(
            "Для модели %s выбрано %s ворк. (CPU: %s, доступно RAM: %.1f ГБ)",
            effective_model,
            workers,
            _get_cpu_count(),
            (_get_available_memory_bytes() or 0) / _GIB,
        )
        log.info("Модель загружена.")
    return _model


def get_media_duration(input_path: str) -> float:
    """Возвращает длительность медиафайла в секундах через ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", input_path],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return 0.0
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def parse_mts_url(url: str) -> tuple[str, Optional[str]]:
    """Разбирает URL и возвращает (event_session_id, record_id или None)."""
    m = MTS_URL_PATTERN.match(url.strip())
    if not m:
        raise ValueError(f"Не удалось разобрать ссылку МТС Линк: {url}")
    return m.group("event_session"), m.group("record_id")


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def download_audio_fast(
    event_session_id: str,
    record_id: Optional[str],
    session_id: Optional[str],
    work_dir: str,
) -> str:
    """Скачивает только аудио чанки — быстрый режим."""
    from audio_downloader import download_audio_only
    return download_audio_only(event_session_id, record_id, session_id, work_dir)


def download_recording_full(
    event_session_id: str,
    record_id: Optional[str],
    session_id: Optional[str],
    work_dir: str,
) -> str:
    """
    Полное скачивание через mtslinker (видео + аудио).
    Используется как fallback если быстрый режим не нашёл аудио чанки.
    """
    from mtslinker.webinar import fetch_webinar_data

    prev_dir = os.getcwd()
    os.chdir(work_dir)
    try:
        result = fetch_webinar_data(
            event_sessions=event_session_id,
            record_id=record_id or "",
            session_id=session_id,
        )
        if not result:
            raise RuntimeError("mtslinker вернул пустой результат — проверь session_id или URL")
        # Ищем файл пока ещё находимся в work_dir — mtslinker создаёт папку относительно cwd
        mp4_files = list(Path(".").rglob("*.mp4"))
        log.info(f"mp4 файлы в work_dir: {[str(f.resolve()) for f in mp4_files]}")
        large_files = [f for f in mp4_files if f.stat().st_size > 100_000]
        target_files = large_files if large_files else mp4_files
        if not target_files:
            raise RuntimeError("mp4 файл не найден после скачивания")
        result_path = str(max(target_files, key=lambda f: f.stat().st_size).resolve())
    finally:
        os.chdir(prev_dir)

    return result_path


def _transcribe_one_file(
    input_path: str,
    model_name: str,
    output_path: str,
    update_status: Optional[Callable[[str], None]] = None,
    chunk_offset: float = 0.0,
    chunk_number: int = 0,
    chunk_total: int = 1,
) -> None:
    """Транскрибирует один файл и дописывает результат в output_path."""
    import time

    model = get_model(model_name)
    log.info(f"Начинаю транскрибацию: {input_path}")

    segments, info = model.transcribe(
        input_path,
        language="ru",
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    duration = info.duration
    log.info(f"Язык: {info.language} ({info.language_probability:.0%}), длительность: {format_timestamp(duration)}")

    last_update = time.monotonic()
    update_interval = 15

    with open(output_path, "a", encoding="utf-8") as f:
        for segment in segments:
            start = chunk_offset + segment.start
            end = chunk_offset + segment.end
            line = segment.text.strip()
            f.write(line + "\n")
            f.flush()
            log.debug(line)

            now = time.monotonic()
            if update_status and duration and (now - last_update) >= update_interval:
                pct = min(int((chunk_number + 0.5) / chunk_total * 100), 99)
                update_status(
                    f"🎙 Транскрибирую часть {chunk_number}/{chunk_total}... {pct}%\n"
                    f"[{format_timestamp(start)} / {format_timestamp(chunk_offset + duration)}]"
                )
                last_update = now


def transcribe_file(
    input_path: str,
    model_name: str,
    output_path: str,
    update_status: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Транскрибирует файл через faster-whisper.
    Для длинных записей разбивает на чанки по 10 минут, чтобы уменьшить пик памяти и нагрузку.
    """
    file_duration = get_media_duration(input_path)
    ranges = build_chunk_ranges(file_duration, chunk_seconds=600)

    if len(ranges) > 1:
        log.info(f"Длительность файла {file_duration:.0f}s превышает лимит чанка, разбиваем на {len(ranges)} частей")
        tmp_dir = tempfile.mkdtemp(prefix="chunks_", dir=BASE_DIR)
        try:
            chunk_files: list[str] = []
            for i, (start, end) in enumerate(ranges, start=1):
                chunk_path = os.path.join(tmp_dir, f"chunk_{i:03d}.wav")
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-i", input_path,
                        "-ss", str(start),
                        "-to", str(end),
                        "-vn",
                        "-acodec", "pcm_s16le",
                        "-ar", "16000",
                        "-ac", "1",
                        chunk_path,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if not os.path.exists(chunk_path):
                    raise RuntimeError(f"Не удалось создать чанк {chunk_path}")
                chunk_files.append(chunk_path)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write("")

            get_model(model_name)
            workers = _model_workers
            if workers == 1:
                for i, (start, end) in enumerate(ranges, start=1):
                    if update_status:
                        update_status(f"🎙 Транскрибирую часть {i}/{len(ranges)}...")
                    _transcribe_one_file(
                        input_path=chunk_files[i - 1],
                        model_name=model_name,
                        output_path=output_path,
                        update_status=update_status,
                        chunk_offset=start,
                        chunk_number=i,
                        chunk_total=len(ranges),
                    )
                    clear_model()
            else:
                part_paths = [f"{output_path}.part{i:03d}" for i in range(1, len(ranges) + 1)]
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [
                        executor.submit(
                            _transcribe_one_file,
                            input_path=chunk_files[i - 1],
                            model_name=model_name,
                            output_path=part_paths[i - 1],
                            update_status=update_status,
                            chunk_offset=start,
                            chunk_number=i,
                            chunk_total=len(ranges),
                        )
                        for i, (start, end) in enumerate(ranges, start=1)
                    ]
                    with open(output_path, "a", encoding="utf-8") as output_file:
                        for future, part_path in zip(futures, part_paths):
                            future.result()
                            with open(part_path, encoding="utf-8") as part_file:
                                shutil.copyfileobj(part_file, output_file)
                            output_file.flush()
                            os.unlink(part_path)
                clear_model()

            log.info(f"Транскрипт сохранён: {output_path}")
            return output_path
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    try:
        model = get_model(model_name)
        log.info(f"Начинаю транскрибацию: {input_path}")

        segments, info = model.transcribe(
            input_path,
            language="ru",
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

        duration = info.duration
        log.info(f"Язык: {info.language} ({info.language_probability:.0%}), длительность: {format_timestamp(duration)}")

        last_update = time.monotonic()
        update_interval = 15  # секунд между обновлениями статуса

        with open(output_path, "w", encoding="utf-8") as f:
            for segment in segments:
                line = segment.text.strip()
                f.write(line + "\n")
                f.flush()
                log.debug(line)

                now = time.monotonic()
                if update_status and duration and (now - last_update) >= update_interval:
                    pct = min(int(segment.end / duration * 100), 99)
                    update_status(
                        f"🎙 Транскрибирую... {pct}%\n"
                        f"[{format_timestamp(segment.end)} / {format_timestamp(duration)}]"
                    )
                    last_update = now

        log.info(f"Транскрипт сохранён: {output_path}")
        return output_path
    finally:
        clear_model()


def run_pipeline_file(
    input_path: str,
    model_name: str,
    update_status: Callable[[str], None],
    output_path: Optional[str] = None,
) -> str:
    """
    Пайплайн для уже скачанного файла: транскрибировать → вернуть путь к .txt.
    """
    if output_path is None:
        output_dir = BASE_DIR / "results"
        output_dir.mkdir(exist_ok=True)
        stem = Path(input_path).stem
        output_path = str(output_dir / f"transcript_{stem}.txt")

    update_status(f"🎙 Транскрибирую...")
    try:
        transcribe_file(input_path, model_name, output_path, update_status)
    finally:
        clear_model()

    return output_path


def run_pipeline(
    url: str,
    model_name: str,
    session_id: Optional[str],
    update_status: Callable[[str], None],
    output_path: Optional[str] = None,
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

    work_dir = tempfile.mkdtemp(prefix="job_", dir=BASE_DIR)
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
                log.error(f"Резервное скачивание тоже не сработало: {fallback_error}", exc_info=True)
                raise RuntimeError(
                    f"Не удалось скачать аудио через быстрый и резервный путь: {fallback_error}"
                ) from fallback_error

        # Шаг 2 — транскрибировать
        update_status(f"🎙 Транскрибирую... (модель {model_name}, это займёт время)")
        transcript_path = final_path
        try:
            transcribe_file(media_path, model_name, transcript_path, update_status)
        finally:
            clear_model()
    finally:
        # Удаляем рабочую директорию после завершения
        shutil.rmtree(work_dir, ignore_errors=True)
        log.info(f"Рабочая директория удалена: {work_dir}")

    return final_path
