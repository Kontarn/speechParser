FROM python:3.11-slim

# Системные зависимости: ffmpeg для обработки видео/аудио
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Устанавливаем Python зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Устанавливаем mtslinker напрямую с GitHub
RUN pip install --no-cache-dir git+https://github.com/motattack/mtslinker.git

# Копируем исходники
COPY bot.py pipeline.py transcribe.py audio_downloader.py ./

# Директория для временных результатов транскрибации
RUN mkdir -p /tmp/speechparser_results

CMD ["python", "bot.py"]
