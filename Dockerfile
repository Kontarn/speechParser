FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-client.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY common.py media.py transcript_parts.py process_control.py mts_fallback.py audio_downloader.py cpu_tuning.py pipeline.py worker_server.py ./
RUN mkdir -p /app/tmp

EXPOSE 8080
CMD ["python", "worker_server.py"]
