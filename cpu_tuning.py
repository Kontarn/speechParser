"""Measure Whisper CPU configurations and persist the fastest usable one."""

import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from common import TRANSCRIBE_OPTIONS
from process_control import configure_progress_lock

log = logging.getLogger(__name__)
GIB = 1024 ** 3
CACHE_VERSION = 2



@dataclass(frozen=True)
class CpuConfig:
    workers: int
    threads: int

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in (self.workers, self.threads)):
            raise ValueError("Число обработчиков и потоков должно быть положительным целым")


def balanced_cpu_config(cpu_count: int, worker_limit: int) -> CpuConfig:
    """Use the entire thread budget, staying within the worker/RAM limit."""
    cpu_count = max(1, cpu_count)
    workers = max(1, min(cpu_count, worker_limit))
    while cpu_count % workers:
        workers -= 1
    return CpuConfig(workers, cpu_count // workers)


def candidate_configs(cpu_count: int, task_count: int) -> list[CpuConfig]:
    capacity = max(1, min(cpu_count, task_count))
    configs = {balanced_cpu_config(cpu_count, limit)
               for limit in (max(1, capacity // 4), max(1, capacity // 2), capacity)}
    return sorted(configs, key=lambda config: config.workers)


def _cache_key(model_name: str, cpu_count: int, task_count: int) -> str:
    try:
        # Includes CPU model and instruction flags, but ignores changing CPU MHz.
        hardware = [
            line.strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith(("model name", "flags", "Features"))
        ][:2]
    except OSError:
        hardware = [platform.processor(), platform.machine()]
    versions = {}
    for package in ("faster-whisper", "ctranslate2", "numpy", "onnxruntime"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    signature = {
        "version": CACHE_VERSION,
        "model": model_name,
        "cpu_count": cpu_count,
        "capacity": min(cpu_count, task_count),
        "hardware": hardware,
        "packages": versions,
        "options": TRANSCRIBE_OPTIONS,
        "environment": {
            name: value for name, value in os.environ.items()
            if name.startswith(("OMP_", "MKL_", "OPENBLAS_", "CT2_", "KMP_"))
        },
    }
    return hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:24]


def _read_cache(path: Path, candidates: list[CpuConfig], budget: int) -> CpuConfig | None:
    try:
        data = json.loads(path.read_text())
        config = CpuConfig(**data["config"])
        peak = int(data["peak_memory_bytes"])
        if config in candidates and 0 < peak * 1.25 <= budget:
            return config
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        pass
    return None


def _write_cache(path: Path, result: dict) -> None:
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(result, stream)
        temporary.replace(path)
    except OSError as error:
        log.warning("Не удалось сохранить подбор CPU: %s", error)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _extract_samples(chunk_files: list[str], directory: str) -> list[str]:
    # The last chunk can be very short; the preceding ones are full-sized.
    sources = chunk_files[:-1] or chunk_files
    indexes = sorted({0, len(sources) // 2, len(sources) - 1})
    samples = []
    for index in indexes:
        output = str(Path(directory) / f"sample_{index}.wav")
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-threads", "1",
             "-i", sources[index], "-t", "30", "-vn", "-ac", "1", "-ar", "16000", output],
            check=True, capture_output=True, timeout=60,
        )
        samples.append(output)
    return samples


def _run_probe(model_name: str, config: CpuConfig, samples: list[str], requests: int,
               budget: int, timeout: float) -> dict:
    request = {
        "model": model_name,
        "config": asdict(config),
        "samples": samples,
        "requests": requests,
        "memory_limit": budget,
        "parent_pid": os.getpid(),
    }
    process = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--probe"],
        input=json.dumps(request), text=True, capture_output=True,
        timeout=timeout, check=True,
    )
    lines = process.stdout.splitlines()
    if not lines:
        raise ValueError("Проба не вернула результат")
    result = json.loads(lines[-1])
    elapsed = float(result["seconds"])
    cpu_seconds = float(result["cpu_seconds"])
    loaded = int(result["loaded_memory_bytes"])
    peak = int(result["peak_memory_bytes"])
    if (not math.isfinite(elapsed) or elapsed <= 0
            or not math.isfinite(cpu_seconds) or cpu_seconds < 0 or loaded <= 0
            or peak < loaded or peak * 1.25 > budget or result["segments"] <= 0):
        raise ValueError("Проба не содержит речи либо превышает бюджет RAM")
    return {**result, "seconds": elapsed, "cpu_seconds": cpu_seconds, "loaded_memory_bytes": loaded,
            "peak_memory_bytes": peak}


def select_cpu_config(model_name: str, chunk_files: list[str], cpu_count: int,
                      available_memory: int | None, fallback: CpuConfig,
                      cache_dir: Path, update_status=None) -> CpuConfig:
    """Tune only automatic settings; all failures fall back to the normal pipeline."""
    if os.getenv("WHISPER_AUTOTUNE", "1").strip().lower() in {"0", "false", "no", "off"}:
        return fallback
    if any(os.getenv(name, "auto").strip().lower() not in {"", "auto"}
           for name in ("WHISPER_WORKERS", "WHISPER_CPU_THREADS")):
        return fallback
    try:
        concurrent_jobs = int(os.getenv("SERVER_MAX_CONCURRENT_JOBS", "1"))
        timeout = float(os.getenv("WHISPER_AUTOTUNE_TIMEOUT_SECONDS", "180"))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("таймаут должен быть положительным числом")
    except ValueError as error:
        log.warning("Автоподбор CPU пропущен: %s", error)
        return fallback
    # Measurements under concurrent jobs are not comparable. Linux provides the
    # per-probe address-space limit and automatic child termination we rely on.
    if concurrent_jobs != 1 or sys.platform != "linux" or not available_memory:
        return fallback
    # Do not spend more time calibrating a short recording than transcribing it.
    if len(chunk_files) < max(4, cpu_count):
        return fallback
    candidates = candidate_configs(cpu_count, len(chunk_files))
    if len(candidates) < 2:
        return fallback
    budget = available_memory - max(GIB, int(available_memory * 0.1))
    if budget <= 0:
        return fallback
    cache_path = cache_dir / f"{_cache_key(model_name, cpu_count, len(chunk_files))}.json"
    cached = _read_cache(cache_path, candidates, budget)
    if cached is not None:
        log.info("CPU из кэша: %s ворк. × %s потоков", cached.workers, cached.threads)
        return cached

    results = []
    try:
        if update_status:
            update_status("⚙️ Подбираю число обработчиков и потоков CPU (первый запуск)...")
        with tempfile.TemporaryDirectory(prefix="cpu_probe_", dir=cache_dir.parent) as directory:
            samples = _extract_samples(chunk_files, directory)
            for config in candidates:
                if results:
                    previous = results[-1]
                    loaded = previous["loaded_memory_bytes"]
                    extra = max(0, previous["peak_memory_bytes"] - loaded)
                    estimated = loaded + extra * config.workers / previous["config"]["workers"]
                    if estimated * 1.25 > budget:
                        log.info("Пропускаю %s × %s: недостаточно RAM по предыдущей пробе",
                                 config.workers, config.threads)
                        continue
                log.info("Проба CPU: %s ворк. × %s потоков", config.workers, config.threads)
                try:
                    result = _run_probe(model_name, config, samples,
                                        min(cpu_count, len(chunk_files)), budget, timeout)
                except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as error:
                    log.warning("Проба %s × %s не завершена (%s)",
                                config.workers, config.threads, type(error).__name__)
                    continue
                results.append({**result, "config": asdict(config)})
                log.info("Проба CPU %s × %s: %.2f с, CPU %.1f%%, пик RAM %.2f GiB",
                         config.workers, config.threads, result["seconds"],
                         100 * result["cpu_seconds"] / result["seconds"] / cpu_count,
                         result["peak_memory_bytes"] / GIB)
    except (OSError, subprocess.SubprocessError) as error:
        log.warning("Автоподбор CPU недоступен: %s", error)
    if not results:
        log.warning("Использую резервную конфигурацию CPU: %s × %s",
                    fallback.workers, fallback.threads)
        return fallback
    best = min(results, key=lambda item: item["seconds"])
    _write_cache(cache_path, best)
    selected = CpuConfig(**best["config"])
    log.info("Выбрана конфигурация CPU: %s ворк. × %s потоков (%.2f с)",
             selected.workers, selected.threads, best["seconds"])
    return selected


def _probe(request: dict) -> dict:
    import ctypes
    import resource
    import signal

    # A worker job is a daemon multiprocessing process, so use a subprocess for
    # isolation. Kill the probe even if /stop or the job deadline kills its parent.
    if ctypes.CDLL(None).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise RuntimeError("Не удалось связать время жизни пробы с заданием")
    if os.getppid() != request["parent_pid"]:
        raise RuntimeError("Родительское задание завершилось")
    limit = int(request["memory_limit"])
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    configure_progress_lock()
    from faster_whisper import WhisperModel

    config = CpuConfig(**request["config"])
    model = WhisperModel(request["model"], device="cpu", compute_type="int8",
                         num_workers=config.workers, cpu_threads=config.threads)
    loaded = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    def transcribe(path):
        segments, _ = model.transcribe(path, **TRANSCRIBE_OPTIONS)
        return sum(bool(segment.text.strip()) for segment in segments)

    transcribe(request["samples"][0])  # Warm up native libraries before timing.
    inputs = [request["samples"][i % len(request["samples"])]
              for i in range(request["requests"])]
    started_cpu = time.process_time()
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        segment_count = sum(executor.map(transcribe, inputs))
    return {
        "seconds": time.perf_counter() - started,
        "cpu_seconds": time.process_time() - started_cpu,
        "loaded_memory_bytes": loaded,
        "peak_memory_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "segments": segment_count,
    }


if __name__ == "__main__":
    if sys.argv[1:] != ["--probe"]:
        raise SystemExit("Этот модуль запускается автоматически из pipeline.py")
    print(json.dumps(_probe(json.load(sys.stdin))), flush=True)
