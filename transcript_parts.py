"""Committed transcript parts survive cancellation and retain audio order."""

import shutil
from pathlib import Path


def parts_directory(output_path: str) -> Path:
    return Path(output_path + ".parts")


def commit_part(output_path: str, number: int, transcribe) -> Path:
    directory = parts_directory(output_path)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{number:06d}.txt"
    temporary = destination.with_suffix(".pending")
    transcribe(str(temporary))
    temporary.replace(destination)
    return destination


def assemble_partial(output_path: str) -> None:
    directory = parts_directory(output_path)
    if not directory.is_dir():
        return  # Short recordings write directly to output_path.
    temporary = Path(output_path + ".assembling")
    try:
        with temporary.open("wb") as destination:
            for part in sorted(directory.glob("*.txt")):
                with part.open("rb") as source:
                    shutil.copyfileobj(source, destination)
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
