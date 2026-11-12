"""Audio extraction primitives; PCM WAV can be sliced without spawning FFmpeg."""

import subprocess
import wave
from pathlib import Path


def extract_audio_chunk(input_path: str, output_path: str, start: float, end: float) -> str:
    try:
        source = wave.open(input_path, "rb")  # noqa: SIM115 -- choose PCM or FFmpeg before entering the context
    except (wave.Error, EOFError, OSError):
        source = None
    if source is not None:
        with source:
            if (source.getframerate(), source.getnchannels(), source.getsampwidth(), source.getcomptype()) == (16000, 1, 2, "NONE"):
                source.setpos(min(round(start * 16000), source.getnframes()))
                with wave.open(output_path, "wb") as target:
                    target.setparams(source.getparams())
                    remaining = round((end - start) * 16000)
                    while remaining > 0:
                        data = source.readframes(min(remaining, 65536))
                        if not data:
                            break
                        target.writeframesraw(data)
                        remaining -= len(data) // 2
                return output_path
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", str(start),
         "-threads", "1", "-i", input_path, "-t", str(end - start), "-map", "0:a:0",
         "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_path],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0 or not Path(output_path).is_file() or Path(output_path).stat().st_size <= 44:
        raise RuntimeError(f"Не удалось извлечь аудиофрагмент: {result.stderr[-500:]}")
    return output_path
