"""
Пайплайн: скачать запись МТС Линк → транскрибировать → вернуть путь к .txt файлу.

По умолчанию используется быстрый режим — скачиваются только аудио чанки.
Если аудио чанки не найдены, автоматически падает на полное скачивание через mtslinker.
"""

import logging
import os
import re
import shutil
import tempfile
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


def get_model(model_name: str):
    global _model, _model_name
    if _model is None or _model_name != model_name:
        from faster_whisper import WhisperModel
        log.info(f"Загрузка модели {model_name}...")
        _model = WhisperModel(model_name, device="cpu", compute_type="int8")
        _model_name = model_name
        log.info("Модель загружена.")
    return _model


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


def transcribe_file(
    input_path: str,
    model_name: str,
    output_path: str,
    update_status: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Транскрибирует файл через faster-whisper.
    Возвращает путь к .txt файлу с таймстампами.
    Если передан update_status — отправляет прогресс каждые 15 секунд.
    """
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
    update_interval = 15  # секунд между обновлениями статуса

    with open(output_path, "w", encoding="utf-8") as f:
        for segment in segments:
            line = f"[{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}] {segment.text.strip()}"
            f.write(line + "\n")
            log.debug(line)

            # Обновляем прогресс не чаще чем раз в update_interval секунд
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


def run_pipeline_file(
    input_path: str,
    model_name: str,
    update_status: Callable[[str], None],
) -> str:
    """
    Пайплайн для уже скачанного файла: транскрибировать → вернуть путь к .txt.
    """
    output_dir = BASE_DIR / "results"
    output_dir.mkdir(exist_ok=True)

    stem = Path(input_path).stem
    transcript_path = str(output_dir / f"transcript_{stem}.txt")

    update_status(f"🎙 Транскрибирую...")
    transcribe_file(input_path, model_name, transcript_path, update_status)

    return transcript_path


def run_pipeline(
    url: str,
    model_name: str,
    session_id: Optional[str],
    update_status: Callable[[str], None],
) -> str:
    """
    Полный пайплайн: скачать → транскрибировать → вернуть путь к .txt.

    Сначала пробует быстрый режим (только аудио чанки).
    Если не получилось — падает на полное скачивание через mtslinker.

    update_status — синхронный колбэк (text: str) для обновления статуса в Telegram.
    """
    event_session_id, record_id = parse_mts_url(url)
    log.info(f"event_session={event_session_id}, record_id={record_id}")

    work_dir = tempfile.mkdtemp(prefix="job_", dir=BASE_DIR)
    try:
        # Шаг 1 — скачать
        update_status("⬇️ Скачиваю аудио дорожку...")
        try:
            media_path = download_audio_fast(event_session_id, record_id, session_id, work_dir)
            log.info(f"Быстрое скачивание успешно: {media_path}")
        except Exception as e:
            log.error(f"audio_downloader упал: {e}", exc_info=True)
            raise RuntimeError(f"Не удалось скачать аудио: {e}")

        # Шаг 2 — транскрибировать
        update_status(f"🎙 Транскрибирую... (модель {model_name}, это займёт время)")
        transcript_path = os.path.join(work_dir, "transcript.txt")
        transcribe_file(media_path, model_name, transcript_path, update_status)

        # Сохраняем результат
        output_dir = BASE_DIR / "results"
        output_dir.mkdir(exist_ok=True)
        final_path = str(output_dir / f"transcript_{event_session_id}.txt")
        shutil.copy2(transcript_path, final_path)
    finally:
        # Удаляем рабочую директорию после завершения
        shutil.rmtree(work_dir, ignore_errors=True)
        log.info(f"Рабочая директория удалена: {work_dir}")

    return final_path
