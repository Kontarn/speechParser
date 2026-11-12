"""Lightweight types and constants shared by the bot, worker and CLI."""

import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent / "tmp"
SUPPORTED_EXTENSIONS = {".mp4", ".mp3", ".wav", ".m4a", ".ogg", ".webm", ".mkv", ".avi", ".mov"}
MTS_LINK_PREFIX = "https://my.mts-link.ru/"
MTS_URL_PATTERN = re.compile(
    r"https://my\.mts-link\.ru/\d+/\d+/record-new/(?P<event_session>\d+)"
    r"(?:/record-file/(?P<record_id>\d+))?/?(?:\?[^\s#]*)?(?:#[^\s]*)?"
)
TRANSCRIBE_OPTIONS = {
    "language": "ru", "beam_size": 5, "vad_filter": True,
    "vad_parameters": {"min_silence_duration_ms": 500},
}


class TranscriptionCancelled(Exception):
    """A cancelled job can still have a downloadable partial transcript."""


def parse_mts_url(url: str) -> tuple[str, str | None]:
    match = MTS_URL_PATTERN.fullmatch(url.strip())
    if match is None:
        raise ValueError("Некорректная ссылка на запись МТС Линк")
    return match.group("event_session"), match.group("record_id")


def format_timestamp(seconds: float) -> str:
    hours, remaining = divmod(max(0, int(seconds)), 3600)
    minutes, seconds = divmod(remaining, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
