#!/usr/bin/env python3
"""
Транскрибация видео/аудио файла через faster-whisper.

Использование:
    python transcribe.py <путь_к_файлу> [--model large-v3] [--language ru]

Примеры:
    python transcribe.py lecture.mp4
    python transcribe.py lecture.mp4 --model medium
    python transcribe.py lecture.mp4 --language auto
"""

import argparse
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def transcribe(input_path: str, model_name: str, language: str | None) -> None:
    if not os.path.exists(input_path):
        log.error(f"Файл не найден: {input_path}")
        sys.exit(1)

    output_path = os.path.splitext(input_path)[0] + ".txt"
    srt_path = os.path.splitext(input_path)[0] + ".srt"

    log.info(f"Загрузка модели {model_name}...")
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        log.error("faster-whisper не установлен. Запусти: pip install faster-whisper")
        sys.exit(1)

    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    log.info("Модель загружена.")

    lang_arg = None if language == "auto" else language
    log.info(f"Начинаю транскрибацию: {input_path}")
    start = time.time()

    segments, info = model.transcribe(
        input_path,
        language=lang_arg,
        beam_size=5,
        vad_filter=True,           # фильтр тишины — ускоряет обработку
        vad_parameters={"min_silence_duration_ms": 500},
    )

    log.info(f"Определён язык: {info.language} (вероятность {info.language_probability:.0%})")
    log.info(f"Длительность: {format_timestamp(info.duration)}")

    with open(output_path, "w", encoding="utf-8") as txt_f, \
         open(srt_path, "w", encoding="utf-8") as srt_f:

        for i, segment in enumerate(segments, start=1):
            # .txt с таймстампами
            line = f"[{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}] {segment.text.strip()}"
            txt_f.write(line + "\n")
            print(line)

            # .srt субтитры
            srt_f.write(f"{i}\n")
            srt_f.write(
                f"{format_timestamp(segment.start)},000 --> {format_timestamp(segment.end)},000\n"
            )
            srt_f.write(segment.text.strip() + "\n\n")

    elapsed = time.time() - start
    log.info(f"Готово за {format_timestamp(elapsed)}")
    log.info(f"Текст:     {output_path}")
    log.info(f"Субтитры:  {srt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Транскрибация видео через faster-whisper")
    parser.add_argument("input", help="Путь к видео или аудио файлу")
    parser.add_argument(
        "--model",
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Модель Whisper (по умолчанию: large-v3)",
    )
    parser.add_argument(
        "--language",
        default="ru",
        help="Язык аудио (ru, en, auto и т.д.). auto — автоопределение (по умолчанию: ru)",
    )
    args = parser.parse_args()

    transcribe(args.input, args.model, args.language)


if __name__ == "__main__":
    main()
