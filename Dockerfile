FROM python:3.11-slim

# Системные зависимости: ffmpeg для обработки видео/аудио
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Устанавливаем зависимости только удалённого обработчика
COPY requirements-worker.txt .
RUN pip install --no-cache-dir -r requirements-worker.txt

# Копируем исходники
COPY worker_server.py pipeline.py audio_downloader.py ./

RUN mkdir -p /app/tmp

EXPOSE 8080
CMD ["python", "worker_server.py"]
