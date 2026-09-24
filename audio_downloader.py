"""
Скачивает аудио из записи МТС Линк максимально быстро.

Стратегия:
1. Если есть отдельные аудио чанки (audio/*) — скачиваем только их, пропускаем видео.
2. Если все чанки video/mp4 — скачиваем видео и извлекаем аудио через ffmpeg.

В обоих случаях не нужен moviepy и не нужно перекодировать видео целиком.
"""

import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import httpx

log = logging.getLogger(__name__)

TIMEOUT = httpx.Timeout(None, connect=30.0)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0",
}


def _make_cookies(session_id: Optional[str]) -> dict:
    return {"access": session_id} if session_id else {}


def _fetch_event_json(event_session_id: str, record_id: Optional[str], session_id: Optional[str]) -> dict:
    """Получает JSON с метаданными записи через API МТС Линк."""
    if record_id:
        url = (
            f"https://my.mts-link.ru/api/event-sessions/{event_session_id}"
            f"/record-files/{record_id}/flow?withoutCuts=false"
        )
    else:
        url = (
            f"https://my.mts-link.ru/api/eventsessions/{event_session_id}"
            f"/record?withoutCuts=false"
        )

    log.info(f"Запрос метаданных: {url}")
    with httpx.Client(timeout=TIMEOUT) as client:
        response = client.get(url, headers=HEADERS, cookies=_make_cookies(session_id))

    if response.status_code == 403:
        raise PermissionError("Доступ запрещён (403). Укажи session_id для приватной записи.")

    response.raise_for_status()
    return response.json()


def _get_content_type(url: str, session_id: Optional[str]) -> str:
    """Определяет Content-Type чанка через HEAD запрос."""
    try:
        with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
            r = client.head(url, headers=HEADERS, cookies=_make_cookies(session_id), follow_redirects=True)
        return r.headers.get("content-type", "")
    except Exception as e:
        log.warning(f"HEAD запрос не удался для {url}: {e}")
        return ""


def _download_chunk(url: str, path: str, session_id: Optional[str]) -> None:
    """Скачивает один чанк в файл."""
    with httpx.Client(timeout=TIMEOUT) as client:
        with client.stream("GET", url, headers=HEADERS, cookies=_make_cookies(session_id)) as response:
            response.raise_for_status()
            with open(path, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=65536):
                    f.write(chunk)


def _ffmpeg_concat(chunk_paths: list[str], output_path: str) -> None:
    """Склеивает файлы через ffmpeg concat demuxer без перекодирования."""
    list_path = output_path + ".filelist.txt"
    with open(list_path, "w") as f:
        for p in chunk_paths:
            escaped = p.replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
    try:
        _ffmpeg_run([
            "-fflags", "+genpts",   # пересчитываем PTS чтобы избежать out-of-order
            "-f", "concat",
            "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            output_path,
        ])
    finally:
        os.unlink(list_path)


def _has_audio_stream(video_path: str) -> bool:
    """Проверяет есть ли в файле аудио дорожка."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", video_path],
        capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


def _ffmpeg_extract_audio(video_path: str, output_path: str) -> None:
    """Извлекает аудио дорожку из видео файла без перекодирования."""
    _ffmpeg_run(["-i", video_path, "-vn", "-c:a", "copy", output_path])


def _ffmpeg_run(args: list[str]) -> None:
    result = subprocess.run(
        ["ffmpeg", "-y"] + args,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log.error(f"ffmpeg stderr: {result.stderr[-2000:]}")
        raise RuntimeError(f"ffmpeg завершился с ошибкой (код {result.returncode})")
    elif result.stderr:
        # Предупреждения логируем но не падаем
        log.debug(f"ffmpeg warnings: {result.stderr[-500:]}")


def download_audio_only(
    event_session_id: str,
    record_id: Optional[str],
    session_id: Optional[str],
    work_dir: str,
) -> str:
    """
    Скачивает аудио из записи МТС Линк максимально быстро.
    Возвращает путь к итоговому аудио файлу.
    """
    json_data = _fetch_event_json(event_session_id, record_id, session_id)

    event_logs = json_data.get("eventLogs", [])
    if not event_logs:
        raise RuntimeError("eventLogs пустой — не удалось получить список чанков")

    # Собираем все URL чанков с временными метками
    chunks = []
    for event in event_logs:
        if not isinstance(event, dict):
            continue
        data = event.get("data", {})
        if isinstance(data, dict) and "url" in data:
            chunks.append({
                "url": data["url"],
                "start": event.get("relativeTime", 0),
            })

    if not chunks:
        raise RuntimeError("Не найдено ни одного чанка в eventLogs")

    log.info(f"Всего чанков в записи: {len(chunks)}")

    # Параллельно проверяем Content-Type всех чанков
    content_types: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {
            executor.submit(_get_content_type, c["url"], session_id): c["url"]
            for c in chunks
        }
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            content_types[url] = future.result()

    for c in chunks:
        log.info(f"  {content_types[c['url']]:20s}  {c['url']}")

    audio_chunks = sorted(
        [c for c in chunks if content_types[c["url"]].startswith("audio/")],
        key=lambda c: c["start"],
    )
    video_chunks = sorted(
        [c for c in chunks if content_types[c["url"]].startswith("video/")],
        key=lambda c: c["start"],
    )

    chunks_dir = os.path.join(work_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    output_path = os.path.join(work_dir, f"audio_{event_session_id}.m4a")

    # --- Стратегия 1: есть отдельные аудио чанки ---
    if audio_chunks:
        log.info(f"Стратегия: аудио чанки ({len(audio_chunks)} шт., пропускаем {len(video_chunks)} видео)")
        chunk_paths = []
        for i, chunk in enumerate(audio_chunks, start=1):
            filename = f"{i:04d}_{Path(chunk['url'].split('?')[0]).name}"
            file_path = os.path.join(chunks_dir, filename)
            log.info(f"[{i}/{len(audio_chunks)}] Скачиваю аудио чанк...")
            _download_chunk(chunk["url"], file_path, session_id)
            chunk_paths.append(file_path)

        if len(chunk_paths) == 1:
            # Один файл — просто копируем
            import shutil
            shutil.copy2(chunk_paths[0], output_path)
        else:
            _ffmpeg_concat(chunk_paths, output_path)

        log.info(f"Готово (аудио чанки): {output_path}")
        return output_path

    # --- Стратегия 2: только видео чанки — из каждого извлекаем аудио, потом склеиваем ---
    if video_chunks:
        log.info(f"Стратегия: извлечение аудио из видео ({len(video_chunks)} чанков)")
        chunk_paths = []
        for i, chunk in enumerate(video_chunks, start=1):
            filename = f"{i:04d}_{Path(chunk['url'].split('?')[0]).name}"
            file_path = os.path.join(chunks_dir, filename)
            log.info(f"[{i}/{len(video_chunks)}] Скачиваю видео чанк...")
            _download_chunk(chunk["url"], file_path, session_id)
            chunk_paths.append(file_path)

        if len(chunk_paths) == 1:
            # Один чанк — сразу извлекаем аудио
            if not _has_audio_stream(chunk_paths[0]):
                raise RuntimeError("Единственный видео чанк не содержит аудио дорожки")
            _ffmpeg_extract_audio(chunk_paths[0], output_path)
        else:
            # Несколько чанков — сначала извлекаем аудио из каждого отдельно,
            # потом склеиваем аудио. Прямая склейка видео чанков не работает
            # из-за несовместимых DTS таймстемпов между сегментами.
            audio_dir = os.path.join(work_dir, "audio_parts")
            os.makedirs(audio_dir, exist_ok=True)
            audio_parts = []
            for i, vpath in enumerate(chunk_paths, start=1):
                if not _has_audio_stream(vpath):
                    log.info(f"[{i}/{len(chunk_paths)}] Чанк без аудио — пропускаю")
                    continue
                apath = os.path.join(audio_dir, f"{i:04d}.m4a")
                log.info(f"[{i}/{len(chunk_paths)}] Извлекаю аудио из чанка...")
                _ffmpeg_extract_audio(vpath, apath)
                audio_parts.append(apath)

            if not audio_parts:
                raise RuntimeError("Ни в одном видео чанке не найдена аудио дорожка")

            _ffmpeg_concat(audio_parts, output_path)

        log.info(f"Готово (извлечение из видео): {output_path}")
        return output_path

    raise RuntimeError("Не найдено ни аудио ни видео чанков в записи")
