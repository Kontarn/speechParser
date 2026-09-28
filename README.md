# speechParser

Telegram бот для транскрибации записей [МТС Линк](https://my.mts-link.ru).

Отправь боту ссылку на запись — он скачает видео, расшифрует речь и пришлёт `.txt` файл с таймстампами.

## Структура

```
speechParser/
├── bot.py            # Telegram бот
├── pipeline.py       # Скачивание + транскрибация
├── transcribe.py     # CLI утилита (отдельно от бота)
├── requirements.txt
└── .env.example
```

## Запуск через Docker (рекомендуется)

```bash
# 1. Установить Docker и Docker Compose
apt install docker.io docker-compose-plugin -y

# 2. Настроить .env
cp .env.example .env
nano .env  # вписать BOT_TOKEN и ALLOWED_USER_ID

# 3. Собрать и запустить
docker compose up -d

# Логи
docker compose logs -f

# Остановить
docker compose down
```

Модели Whisper кэшируются в Docker volume `whisper_cache` — при пересборке скачиваться повторно не будут.

## Установка без Docker

```bash
# 1. Системные зависимости
apt install python3 python3-pip python3-venv ffmpeg git -y

# 2. Создать окружение и установить зависимости
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/motattack/mtslinker.git

# 3. Настроить .env
cp .env.example .env
nano .env
```

## Настройка .env

| Переменная       | Описание                                                  |
|-----------------|-----------------------------------------------------------|
| `BOT_TOKEN`     | Токен бота от [@BotFather](https://t.me/BotFather)        |
| `ALLOWED_USER_ID` | Твой Telegram user_id (узнать у [@userinfobot](https://t.me/userinfobot)) |
| `WHISPER_MODEL` | Модель Whisper (по умолчанию `large-v3`)                  |
| `WHISPER_WORKERS` | Число параллельных чанков (по умолчанию `1`; `2` требует больше RAM) |
| `MTS_SESSION_ID`| Cookie `access` с my.mts-link.ru (только для приватных записей) |
| `TRANSCRIPTION_TIMEOUT_SECONDS` | Максимальное время обработки (по умолчанию 10800 секунд / 3 часа) |

## Запуск (без Docker)

```bash
source venv/bin/activate
python bot.py
```

### Через tmux

```bash
tmux new -s speechparser
source venv/bin/activate
python bot.py
# Ctrl+B, D — отсоединиться
# tmux attach -t speechparser — вернуться
```

## Использование

1. Найди запись на my.mts-link.ru
2. Отправь боту ссылку: `https://my.mts-link.ru/...`
3. Бот пришлёт статусные сообщения в процессе работы
4. Когда готово — пришлёт `.txt` файл с таймстампами

Чтобы остановить текущую расшифровку, отправь команду `/stop`. Бот остановит обработку и пришлёт уже готовую часть текста.

## Модели Whisper и скорость на CPU

| Модель   | Размер | Время (2ч аудио) | Качество     |
|----------|--------|-----------------|--------------|
| medium   | 1.5 GB | ~25–40 мин      | хорошее      |
| large-v2 | 3 GB   | ~50–80 мин      | отличное     |
| large-v3 | 3 GB   | ~50–80 мин      | наилучшее    |

Модель скачивается автоматически при первом запуске.
