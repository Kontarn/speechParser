"""
Скачивает аудио из записи МТС Линк максимально быстро.

Стратегия:
1. Если есть отдельные аудио чанки (audio/*) — скачиваем только их, пропускаем видео.
2. Если все чанки video/mp4 — скачиваем видео и извлекаем аудио через ffmpeg.

В обоих случаях не нужен moviepy и не нужно перекодировать видео целиком.
"""

import logging
import math
import os
import subprocess
import tempfile
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


def _ffmpeg_mix_timeline(
    tracks: list[tuple[str, float]], output_path: str, duration: float | None,
) -> None:
    """Mix tracks at their real offsets without keeping the recording in RAM."""
    import numpy as np

    sample_rate = 16000
    sample_bytes = 4  # float32 accumulator prevents clipping while summing tracks.
    block_bytes = 256 * 1024
    total_bytes = round(duration * sample_rate) * sample_bytes if duration is not None else 0
    included = 0
    with tempfile.TemporaryDirectory(prefix="audio_mix_", dir=Path(output_path).parent) as directory:
        mixed_path = Path(directory) / "mixed.f32"
        decoded_path = Path(directory) / "decoded.f32"
        # A sparse disk file preserves gaps with zeroes. Each read/write is bounded;
        # hundreds of late-starting tracks must not allocate huge adelay buffers.
        with mixed_path.open("w+b") as mixed:
            mixed.truncate(total_bytes)
            for path, start in tracks:
                if not math.isfinite(start):
                    raise ValueError("Некорректное время начала аудиофрагмента")
                if duration is not None and start >= duration:
                    continue
                filters = []
                if start < 0:
                    filters.append(f"atrim=start={-start:.9f}")
                # relativeTime supplies the offset, regardless of container PTS.
                filters.append("asetpts=N/SR/TB")
                args = ["-threads", "1", "-i", path, "-map", "0:a:0", "-vn",
                        "-af", ",".join(filters), "-ar", str(sample_rate), "-ac", "1"]
                if duration is not None:
                    args.extend(["-t", f"{duration - max(0, start):.9f}"])
                _ffmpeg_run(args + ["-c:a", "pcm_f32le", "-f", "f32le", str(decoded_path)])
                offset = round(max(0, start) * sample_rate) * sample_bytes
                decoded_bytes = decoded_path.stat().st_size
                if duration is None:
                    total_bytes = max(total_bytes, offset + decoded_bytes)
                    mixed.truncate(total_bytes)
                remaining = min(decoded_bytes, max(0, total_bytes - offset))
                if not remaining:
                    continue
                included += 1
                with decoded_path.open("rb") as decoded:
                    while remaining:
                        data = decoded.read(min(block_bytes, remaining))
                        if not data:
                            raise RuntimeError("Не удалось прочитать декодированный аудиофрагмент")
                        mixed.seek(offset)
                        values = np.frombuffer(mixed.read(len(data)), dtype="<f4").copy()
                        values += np.frombuffer(data, dtype="<f4")
                        mixed.seek(offset)
                        mixed.write(values.tobytes())
                        offset += len(data)
                        remaining -= len(data)
        if not included:
            raise RuntimeError("Нет аудиофрагментов в пределах записи")
        _ffmpeg_run([
            "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", str(mixed_path),
            "-af", "alimiter=limit=0.95:level=0:latency=1",
            "-c:a", "pcm_s16le", "-rf64", "auto", output_path,
        ])


def _has_audio_stream(video_path: str) -> bool:
    """Проверяет есть ли в файле аудио дорожка."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", video_path],
        capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


def _ffmpeg_run(args: list[str]) -> None:
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-y"] + args,
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

    try:
        duration = float(json_data.get("duration", 0))
    except (TypeError, ValueError):
        duration = 0.0
    duration = duration if math.isfinite(duration) and duration > 0 else None

    # An event can be repeated; identical URLs at different times are retained.
    chunks = []
    seen = set()
    for event in event_logs:
        if not isinstance(event, dict):
            continue
        data = event.get("data", {})
        if isinstance(data, dict) and "url" in data:
            start = float(event.get("relativeTime", 0))
            if not math.isfinite(start):
                raise ValueError("Некорректный relativeTime в метаданных МТС Линк")
            key = (data["url"], start)
            if key in seen:
                continue
            seen.add(key)
            chunks.append({
                "url": data["url"],
                "start": start,
            })

    if not chunks:
        raise RuntimeError("Не найдено ни одного чанка в eventLogs")

    log.info(f"Всего чанков в записи: {len(chunks)}")

    # Параллельно проверяем Content-Type всех чанков
    content_types: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {
            executor.submit(_get_content_type, url, session_id): url
            for url in dict.fromkeys(c["url"] for c in chunks)
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
    output_path = os.path.join(work_dir, f"audio_{event_session_id}.wav")

    if audio_chunks:
        log.info(f"Стратегия: аудио чанки ({len(audio_chunks)} шт., пропускаем {len(video_chunks)} видео)")
        selected = audio_chunks
    elif video_chunks:
        log.info("Стратегия: аудиодорожки из %s видеофрагментов", len(video_chunks))
        selected = video_chunks
    else:
        raise RuntimeError("Не найдено ни аудио ни видео чанков в записи")

    tracks = []
    downloaded = {}
    for i, chunk in enumerate(selected, start=1):
        url = chunk["url"]
        if url not in downloaded:
            filename = f"{i:04d}_{Path(chunk['url'].split('?')[0]).name}"
            file_path = os.path.join(chunks_dir, filename)
            log.info("[%s/%s] Скачиваю медиафрагмент...", i, len(selected))
            _download_chunk(url, file_path, session_id)
            downloaded[url] = file_path
        file_path = downloaded[url]
        if not audio_chunks and not _has_audio_stream(file_path):
            log.info("Фрагмент без аудио — пропускаю: %s", file_path)
            continue
        tracks.append((file_path, chunk["start"]))

    log.info("Собираю %s аудиофрагментов по relativeTime; длительность записи: %s с",
             len(tracks), duration)
    _ffmpeg_mix_timeline(tracks, output_path, duration)
    log.info("Аудио по временной шкале записи готово: %s", output_path)
    return output_path
